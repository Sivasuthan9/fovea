"""TypeSafe System One decision server (`POST /v1/systemone`) for Fovea-4B (see engine.py).

Images may arrive in any of the forms the Jev ecosystem uses, and all are accepted:
  * `assets`: {ref: {"data_base64": ..., "format": ...}}, referenced by plain-text matches of `ref` in the state,
    instructions or criteria descriptions (TypeSafe's documented form);
  * `data:image/...;base64,...` URIs embedded in any string of the state, instructions or criteria (Image JevBench form);
  * a top-level `images` list of data URIs or base64 strings.
Every image becomes "[image k]" in the text, in order of first use, and is passed to the model in that order, as in
training. Option names and question ids are never rewritten.

Answers always cover exactly the requested labels and sum to 1: the model's `unknown` candidate is removed and the rest
renormalised (choice, score) or split evenly between yes and no (noul). A question that cannot
be scored for any reason gets a uniform answer instead of an error, since an invalid answer counts as wrong and also
drops out of calibration. Standard library HTTP server; requests are served one at a time on the GPU.
"""
import base64
import binascii
import io
import json
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image, ImageOps

from .engine import answer, candidates, flatten_instructions, prompt_text

Image.MAX_IMAGE_PIXELS = 400_000_000  # never refuse a large benchmark image; it is resized before the model sees it
DATA_URI = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+")
TYPES = ("noul", "choice", "score")


def decode_image(data):
    """bytes, a base64 string or a data URI -> RGB PIL image (EXIF orientation applied, as in training)."""
    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        data = base64.b64decode(re.sub(r"\s+", "", data) + "=" * (-len(data) % 4), validate=False)
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


class Images:
    """Collects the request's images in order of first use and rewrites references to "[image k]"."""

    def __init__(self, assets):
        self.assets = {}
        for ref, a in (assets or {}).items():
            raw = a.get("data_base64") or a.get("data") or a.get("base64") if isinstance(a, dict) else a
            if raw:
                self.assets[str(ref)] = raw
        self.refs = sorted(self.assets, key=len, reverse=True)  # longest first, so no reference eats a longer one
        self.order, self.images = {}, []

    def _marker(self, key, raw):
        if key not in self.order:
            self.order[key] = len(self.images) + 1
            self.images.append(decode_image(raw))
        return f"[image {self.order[key]}]"

    def text(self, s):
        if not isinstance(s, str):
            return s
        s = DATA_URI.sub(lambda m: self._marker(m.group(0)[:200] + str(len(m.group(0))), m.group(0)), s)
        for ref in self.refs:
            if ref in s:
                s = s.replace(ref, self._marker(ref, self.assets[ref]))
        return s

    def walk(self, obj):
        if isinstance(obj, dict):
            return {k: self.walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.walk(v) for v in obj]
        return self.text(obj)

    def extra(self, items):
        """Top-level `images` and assets nobody referenced: still shown to the model, after the referenced ones."""
        for i, raw in enumerate(items or []):
            self._marker(f"__images_{i}", raw)
        for ref in self.refs:
            self._marker(ref, self.assets[ref])


def uniform(qtype, labels):
    p = 1.0 / len(labels)
    if qtype == "noul":
        return {"type": "noul", "noul": 0.5, "confidence": 0.0}
    probs = {lab: p for lab in labels}
    out = {"type": qtype, "probabilities": probs, "confidence": 0.0}
    if qtype == "choice":
        out["choice"] = labels[0]
    else:
        out["score"] = float(sum(i * p for i in range(len(labels))))
    return out


class DecisionServer:
    def __init__(self, engine, model_name, temperature=None, runner=None):
        """runner: an object with the engine's `inputs` and `logits` (the CUDA-graph FastPath); default the engine."""
        self.engine, self.name, self.run = engine, model_name, runner or engine
        self.temperature = engine.temperature if temperature is None else temperature
        self.lock = threading.Lock()

    def _question(self, q, state, imgs):
        qtype = q.get("type")
        if qtype not in TYPES:
            raise ValueError(f"unsupported question type {qtype!r}")
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("choice needs a criteria object")
            criteria = {k: imgs.walk(v) for k, v in criteria.items()}  # option names stay as given
        elif qtype == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError("score needs a list of at least two levels")
            criteria = imgs.walk(criteria)
        else:
            criteria = imgs.walk(criteria) if isinstance(criteria, dict) else None
        question = imgs.text(flatten_instructions(q.get("instructions", "")))
        values, texts = candidates(qtype, criteria)
        return qtype, values, prompt_text(question, texts, state, self.engine.codes), len(texts) + 1

    def answer(self, req):
        if not isinstance(req, dict) or not isinstance(req.get("questions"), dict) or not req["questions"]:
            raise ValueError("request needs a non-empty 'questions' object")
        answers, tokens = {}, 0
        imgs = Images(req.get("assets"))
        state = imgs.walk(req.get("state", {}))
        state = state if isinstance(state, (dict, str)) else {"state": state}
        prepared = {}
        for qid, q in req["questions"].items():
            try:
                prepared[qid] = self._question(q, state, imgs)
            except Exception as e:  # malformed question: answered uniformly below if its labels are known
                prepared[qid] = e
        imgs.extra(req.get("images"))
        with self.lock, torch.inference_mode():
            for qid, q in req["questions"].items():
                item = prepared[qid]
                try:
                    if isinstance(item, Exception):
                        raise item
                    qtype, values, text, n = item
                    inputs = self.run.inputs(imgs.images, text)
                    tokens += int(inputs["input_ids"].shape[1])
                    z = self.run.logits(inputs, n).float().cpu().numpy()
                    t = self.temperature.get(qtype, 1.0) if isinstance(self.temperature, dict) else self.temperature
                    a = answer(qtype, values, z, t)
                    if qtype == "noul":
                        a["confidence"] = abs(2 * a["noul"] - 1)
                    elif qtype == "score":
                        a["legend"] = {str(i): str(d) for i, d in enumerate(q["criteria"])}
                    answers[qid] = a
                except Exception:
                    traceback.print_exc()
                    labels = (list(q.get("criteria") or {}) if q.get("type") == "choice"
                              else [str(i) for i in range(len(q.get("criteria") or [0, 0]))])
                    answers[qid] = uniform(q.get("type") if q.get("type") in TYPES else "noul", labels or ["0", "1"])
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    def warm_up(self, rounds=3):
        """Compile the kernels (flash-linear-attention autotuning) before the first timed request."""
        rng = np.random.default_rng(0)
        for side in (448, 896, 1280):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 255, (side * 3 // 4, side, 3), dtype=np.uint8)).save(buf, "PNG")
            req = {"state": {"photo": "warm"}, "assets": {"warm": {"data_base64": base64.b64encode(buf.getvalue()).decode()}},
                   "questions": {"decision": {"type": "choice", "instructions": "Which colour dominates `photo`?",
                                              "criteria": {"red": None, "green": None, "blue": None}}}}
            for _ in range(rounds):
                self.answer(req)
        self.answer({"state": "warm-up", "questions": {"decision": {"type": "noul", "instructions": "Is this a test?"}}})


def make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/health", ""):
                return self._send(200, {"status": "ok", "model": server.name})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except (ValueError, binascii.Error):
                return self._send(400, {"error": "body is not valid JSON"})
            try:
                t0 = time.perf_counter()
                out = server.answer(req)
                out["usage"]["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
                self._send(200, out)
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # never drop the connection: report the failure
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(server, host="127.0.0.1", port=8000):
    ThreadingHTTPServer((host, port), make_handler(server)).serve_forever()
