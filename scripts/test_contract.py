"""Contract test for a running server: every answer must pass the JevBench harness validation rules.

Usage: python scripts/test_contract.py http://127.0.0.1:8000 some_image.jpg
"""
import base64, io, json, math, sys, time, urllib.request
from PIL import Image
URL = sys.argv[1]; IMG = sys.argv[2]
raw = open(IMG, "rb").read()
b64 = base64.b64encode(raw).decode()
fmt = Image.open(io.BytesIO(raw)).format.lower()

def post(req):
    data = json.dumps(req).encode()
    r = urllib.request.Request(URL + "/v1/systemone", data=data, headers={"Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=60) as resp:
        out = json.loads(resp.read()); code = resp.status
    return code, out, time.perf_counter() - t0

def valid(q, a):
    """The harness's rules: right type; probabilities keyed exactly by the labels, finite in [0,1], summing to 1 (+-0.001)."""
    if a.get("type") != q["type"]: return f"type {a.get('type')}"
    if q["type"] == "noul":
        p = a.get("noul"); return "ok" if isinstance(p, float) and 0 <= p <= 1 else f"noul {p!r}"
    labels = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    pr = a.get("probabilities", {})
    if set(pr) != set(labels): return f"labels {sorted(pr)} != {sorted(labels)}"
    if not all(isinstance(v, float) and math.isfinite(v) and 0 <= v <= 1 for v in pr.values()): return "bad values"
    if abs(sum(pr.values()) - 1) > 0.001: return f"sum {sum(pr.values())}"
    if q["type"] == "choice" and a.get("choice") not in labels: return f"choice {a.get('choice')!r}"
    return "ok"

Q = {"choice": {"type": "choice", "instructions": "What is the main subject of `photo`?",
                "criteria": {"a person": None, "an animal": "a dog, cat or other animal", "a vehicle": None, "food": None}},
     "noul": {"type": "noul", "instructions": "The photo in `photo` was taken outdoors.", "criteria": {"true": "outdoors", "false": "indoors"}},
     "score": {"type": "score", "instructions": "How cluttered is `photo`?", "criteria": ["empty", "a few objects", "busy", "very cluttered"]}}
results = {}
for name, q in Q.items():
    forms = {"assets": {"state": {"photo": "ref-1"}, "assets": {"ref-1": {"type": "image", "format": fmt, "data_base64": b64}}, "questions": {"decision": q}},
             "data_uri": {"state": {"photo": f"data:image/{fmt};base64,{b64}"}, "questions": {"decision": q}},
             "images_list": {"state": {"photo": "[image 1]"}, "images": [f"data:image/{fmt};base64,{b64}"], "questions": {"decision": q}}}
    for form, req in forms.items():
        code, out, sec = post(req)
        a = out["answers"]["decision"]
        results[(name, form)] = a
        print(f"{name:6s} {form:11s} HTTP {code} valid={valid(q, a)} {sec*1000:.0f} ms  {json.dumps({k: v for k, v in a.items() if k != 'legend'})[:150]}")
    same = all(json.dumps(results[(name, f)].get("probabilities", results[(name, f)].get("noul"))) == json.dumps(results[(name, "assets")].get("probabilities", results[(name, "assets")].get("noul"))) for f in forms)
    print(f"   identical across image forms: {same}")
# edge cases
big = io.BytesIO(); Image.open(IMG).convert("RGB").resize((5000, 4000)).save(big, "JPEG", quality=85)
bmp = io.BytesIO(); Image.open(IMG).convert("RGB").save(bmp, "BMP")
cases = {"text only": {"state": {"ticket": "My parcel arrived crushed."}, "questions": {"decision": Q["noul"] | {"instructions": "The customer reports damage."}}},
         "unreferenced asset": {"state": {"note": "see attachment"}, "assets": {"zz": {"data_base64": b64}}, "questions": {"decision": Q["choice"]}},
         "BMP image": {"state": {"photo": "b"}, "assets": {"b": {"format": "bmp", "data_base64": base64.b64encode(bmp.getvalue()).decode()}}, "questions": {"decision": Q["score"]}},
         "5000x4000 JPEG": {"state": {"photo": "g"}, "assets": {"g": {"format": "jpg", "data_base64": base64.b64encode(big.getvalue()).decode()}}, "questions": {"decision": Q["choice"]}},
         "bad criteria": {"state": {}, "questions": {"decision": {"type": "score", "instructions": "x", "criteria": "oops"}}},
         "two questions": {"state": {"photo": "ref-1"}, "assets": {"ref-1": {"data_base64": b64}}, "questions": {"q1": Q["choice"], "q2": Q["noul"]}}}
for name, req in cases.items():
    try:
        code, out, sec = post(req)
        print(f"{name:20s} HTTP {code} {sec*1000:.0f} ms tokens {out['usage']['input_tokens']}  " + "; ".join(
            f"{k}: valid={valid(req['questions'][k], a) if isinstance(req['questions'][k].get('criteria'), (dict, list)) or req['questions'][k]['type']=='noul' else a}" for k, a in out["answers"].items()))
    except Exception as e:
        print(f"{name:20s} FAILED {type(e).__name__}: {e}")
