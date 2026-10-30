"""Typed decisions about images with Fovea-4B: a client for any TypeSafe System One server (standard library only).

    from fovea_sdk import FoveaClient

    fovea = FoveaClient("http://127.0.0.1:8000")
    a = fovea.choice("What is the total on the receipt?", ["12.40", "14.20", "21.40"], image="receipt.jpg")
    a.value, a.probabilities          # "14.20", {"12.40": 0.02, "14.20": 0.95, "21.40": 0.03}
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

ImageLike = Union[str, Path, bytes, bytearray, Any]  # a file path, raw bytes, a base64 / data-URI string, or a PIL image


class FoveaError(Exception):
    """The server refused the request or returned an answer that is not a valid decision."""


@dataclass
class Answer:
    """One decision. `value` is True/False for `noul`, the option name for `choice` and the level index for `score`;
    `probabilities` covers every allowed answer and sums to 1 ("yes"/"no" for `noul`)."""

    type: str
    value: Any
    probabilities: dict
    confidence: float
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def probability(self) -> float:
        """The probability of the chosen answer."""
        key = ("yes" if self.value else "no") if self.type == "noul" else str(self.value)
        return self.probabilities[key]


_MAGIC = ((b"\xff\xd8\xff", "jpg"), (b"\x89PNG", "png"), (b"RIFF", "webp"), (b"BM", "bmp"), (b"GIF8", "gif"))


def _format(data: bytes) -> str:
    return next((fmt for magic, fmt in _MAGIC if data.startswith(magic)), "png")


def encode_image(image: ImageLike) -> dict:
    """A System One asset for a file path, bytes, a base64 or data-URI string, or a PIL image."""
    if hasattr(image, "save") and not isinstance(image, (str, bytes, bytearray, Path)):  # a PIL image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "PNG")
        data = buf.getvalue()
    elif isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    elif isinstance(image, Path) or (isinstance(image, str) and len(image) < 4096 and Path(image).is_file()):
        data = Path(image).read_bytes()
    elif isinstance(image, str):  # already base64, possibly a data URI
        b64 = image.split(",", 1)[1] if image.startswith("data:") else image
        return {"type": "image", "format": _format(base64.b64decode(b64[:64] + "=" * (-len(b64[:64]) % 4))), "data_base64": b64}
    else:
        raise TypeError(f"unsupported image type: {type(image).__name__}")
    return {"type": "image", "format": _format(data), "data_base64": base64.b64encode(data).decode()}


def build_request(question: str, qtype: str, criteria: Any = None, images: Sequence[ImageLike] = (),
                  state: Optional[Mapping] = None) -> dict:
    """A one-question System One request. Images are sent as assets and referenced from the state as `image`
    (one image) or `images` (several), so the question can mention them, e.g. "the receipt in `image`"."""
    assets, refs = {}, []
    for i, im in enumerate(images):
        ref = f"image-{i + 1}"
        assets[ref] = encode_image(im)
        refs.append(ref)
    st = dict(state or {})
    if len(refs) == 1:
        st.setdefault("image", refs[0])
    elif refs:
        st.setdefault("images", refs)
    q = {"type": qtype, "instructions": question}
    if criteria is not None:
        q["criteria"] = criteria
    req = {"state": st, "questions": {"decision": q}}
    if assets:
        req["assets"] = assets
    return req


def parse_answer(qtype: str, labels: Sequence[str], raw: Mapping) -> Answer:
    """An Answer from the server's answer object, checked like the JevBench harness checks it."""
    if not isinstance(raw, Mapping) or raw.get("type") != qtype:
        raise FoveaError(f"expected a {qtype!r} answer, got {raw!r}")
    if qtype == "noul":
        p = raw.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise FoveaError(f"invalid noul probability: {p!r}")
        return Answer("noul", p >= 0.5, {"yes": float(p), "no": 1.0 - float(p)}, float(raw.get("confidence", abs(2 * p - 1))), dict(raw))
    probs = raw.get("probabilities")
    if not isinstance(probs, Mapping) or set(probs) != set(labels):
        raise FoveaError(f"probabilities must cover exactly {list(labels)}, got {probs!r}")
    if abs(sum(probs.values()) - 1.0) > 0.02:
        raise FoveaError(f"probabilities sum to {sum(probs.values()):.3f}, not 1")
    best = max(labels, key=lambda k: probs[k])
    value = raw.get("choice", best) if qtype == "choice" else int(best)
    return Answer(qtype, value, {k: float(probs[k]) for k in labels}, float(raw.get("confidence", 0.0)), dict(raw))


class _Decisions:
    """The typed question helpers, shared by the HTTP client and the local model."""

    def ask(self, request: dict) -> dict:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _one(self, question, qtype, criteria, labels, image, images, state) -> Answer:
        imgs = ([image] if image is not None else []) + list(images)
        out = self.ask(build_request(question, qtype, criteria, imgs, state))
        return parse_answer(qtype, labels, (out.get("answers") or {}).get("decision"))

    def noul(self, statement: str, image: ImageLike = None, *, images: Iterable[ImageLike] = (), yes: str = None,
             no: str = None, state: Mapping = None) -> Answer:
        """A yes/no decision about a statement or question; `yes` / `no` optionally describe each side."""
        crit = {k: v for k, v in (("true", yes), ("false", no)) if v}
        return self._one(statement, "noul", crit or None, ("yes", "no"), image, images, state)

    def choice(self, question: str, options: Union[Sequence[str], Mapping[str, Optional[str]]], image: ImageLike = None,
               *, images: Iterable[ImageLike] = (), state: Mapping = None) -> Answer:
        """One option from a list of names, or from a {name: description} mapping."""
        crit = dict(options) if isinstance(options, Mapping) else {str(o): None for o in options}
        if len(crit) < 2:
            raise ValueError("a choice needs at least two options")
        return self._one(question, "choice", crit, list(crit), image, images, state)

    def score(self, question: str, levels: Sequence[str], image: ImageLike = None, *,
              images: Iterable[ImageLike] = (), state: Mapping = None) -> Answer:
        """One level on an ordered scale; `levels` describes each level, lowest first. `value` is the level index."""
        if len(levels) < 2:
            raise ValueError("a score needs at least two levels")
        return self._one(question, "score", list(levels), [str(i) for i in range(len(levels))], image, images, state)


class FoveaClient(_Decisions):
    """A client for a running Fovea-4B server (`fovea-sdk serve`) or any other System One endpoint."""

    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 120.0, api_key: Optional[str] = None):
        self.url, self.timeout, self.api_key = url.rstrip("/"), timeout, api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health(self) -> dict:
        with urllib.request.urlopen(urllib.request.Request(self.url + "/health"), timeout=self.timeout) as r:
            return json.loads(r.read())

    def ask(self, request: dict) -> dict:
        """Send a full System One request (any number of questions) and return the raw response."""
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(request).encode(), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise FoveaError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
