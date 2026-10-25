<h1 align="center">Fovea-4B</h1>

<p align="center"><b>A fast, calibrated decision model for images</b></p>

<p align="center">
  <a href="https://huggingface.co/Sivasuthan/fovea-4b"><img alt="Model on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-fovea--4b-ffcc4d"></a>
  <a href="https://huggingface.co/Sivasuthan/fovea-4b"><img alt="Hugging Face downloads" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FSivasuthan%2Ffovea-4b%3Fexpand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20downloads&color=ffcc4d&cacheSeconds=3600"></a>
  <a href="https://huggingface.co/mradermacher/fovea-4b-GGUF"><img alt="GGUF downloads" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fmradermacher%2Ffovea-4b-GGUF%3Fexpand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20GGUF%20downloads&color=ffcc4d&cacheSeconds=3600"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2a78d6"></a>
  <img alt="Parameters: 4B" src="https://img.shields.io/badge/parameters-4B-1baf7a">
  <img alt="API: TypeSafe System One" src="https://img.shields.io/badge/API-TypeSafe%20System%20One-5c5c58">
</p>

Fovea-4B is a fast decision model for images. Given an image (a photo, document, receipt, chart or screenshot), a JSON
state and a typed question, it returns a probability for every allowed answer in a single forward pass. It is served
over TypeSafe's System One API, so any client of that API can call it unchanged.

- **Weights:** [huggingface.co/Sivasuthan/fovea-4b](https://huggingface.co/Sivasuthan/fovea-4b) (Apache-2.0)
- **Question types:** `noul` (yes/no), `choice` (named options), `score` (ordered levels)
- **Output:** every answer covers exactly the requested labels and sums to 1

<h2 align="center">Results</h2>

<p align="center">
  <img src="assets/comparison.png" alt="Fovea-4B compared with Imajev-4B, H2O-Lightning-4B and Jev-Vision on 2,360 held-out questions" width="100%">
</p>

We compare Fovea-4B with three open decision models: the two strongest open 4B models on the ImageJevBench v0.3
leaderboard by Intelligence, Imajev-4B (#1) and H2O-Lightning-4B image (#2), and the larger Jev-Vision (8B). All four
answer the same 2,360 held-out multiple-choice questions, each served as its authors release it, and are scored with
the leaderboard's Intelligence formula, 100 × (accuracy − chance) / (1 − chance).

Fovea-4B scores 80.6 overall, the highest of the four (H2O-Lightning-4B 79.9, Imajev-4B 79.4, Jev-Vision 75.7). Paired on
the same questions, it leads Imajev-4B by +1.1 points (95% interval +0.3 to +2.0), H2O-Lightning-4B by +0.7 (−0.7 to
+2.0) and the twice-as-large Jev-Vision by +4.9 (+3.0 to +6.7). Its largest leads by category are on multi-image
comparison, charts and counting, while H2O-Lightning-4B is slightly ahead on diagrams and general photos. Its
calibration error is also the lowest of the four (expected calibration error 0.035, against 0.059, 0.061 and 0.179).

**How each model was served.** Fovea-4B: images up to 2 MP, its calibrated temperatures. Imajev-4B: native resolution,
temperature 1.305. H2O-Lightning-4B image: images up to 1 MP, temperature 0.65. Jev-Vision: its own decision function,
images up to 1.29 MP, temperature 0.5.

**Evaluation data.** 2,360 multiple-choice questions from the test or validation splits of public benchmarks, none of
them used for training: DocVQA (400 document questions), ChartQA (300) and SEED-Bench-2-Plus charts (100), AI2D (300
diagrams) and SEED-Bench-2-Plus maps (100), CV-Bench (300: depth, distance, spatial relation and counting), BLINK (560:
spatial reasoning, counting and nine multi-image tasks) and MMStar (300 general photos). The overall score weights the
categories by the ImageJevBench core-track mix. This is our own held-out evaluation, not the ImageJevBench leaderboard:
it contains only multiple-choice questions, and leaderboard scores are lower and can rank models differently.

<h2 align="center">Run the server</h2>

```sh
git clone https://github.com/Sivasuthan9/fovea && cd fovea
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python scripts/serve.py --model Sivasuthan/fovea-4b --revision 55c1dd919bc59dfd3316de6d752e18f3242ad385 --port 8000
```

One GPU with 16 GB or more is enough. Start-up downloads the weights (9 GB), records CUDA graphs and warms up the kernels;
the server prints `ready` when it accepts requests. Then:

```sh
curl -s http://127.0.0.1:8000/health
python scripts/test_contract.py http://127.0.0.1:8000 some_image.jpg   # checks every answer against the harness rules
```

<h2 align="center">Request and response</h2>

```json
POST /v1/systemone
{"state": {"photo": "img-1"},
 "assets": {"img-1": {"type": "image", "format": "jpg", "data_base64": "..."}},
 "questions": {"decision": {"type": "choice",
                            "instructions": "What is the total on the receipt in `photo`?",
                            "criteria": {"12.40": null, "14.20": null, "21.40": null}}}}
```

```json
{"model": "fovea-4b",
 "answers": {"decision": {"type": "choice", "choice": "14.20", "confidence": 0.91,
                          "probabilities": {"12.40": 0.02, "14.20": 0.95, "21.40": 0.03}}},
 "usage": {"input_tokens": 1093, "output_tokens": 0, "total_ms": 41.7}}
```

- Images can be given as `assets` referenced from the state, instructions or criteria (TypeSafe's form), as
  `data:image/...;base64,` URIs inside any string, or as a top-level `images` list. Each becomes `[image k]` in the text.
- `noul` answers carry `"noul": p` (the probability of yes); `choice` and `score` answers carry `probabilities` over the
  option names or level indices, plus `choice` or `score` (the expected level) and `confidence`.
- Several questions per request are allowed. A question that cannot be scored gets a uniform answer, not an error.
- Images larger than the model's pixel budget (calibration.json, `max_pixels`) are downscaled, keeping the aspect ratio.

<h2 align="center">Training</h2>

The model was trained on 57,441 typed image decisions built from public train splits (TextVQA, DocVQA, ChartQA, GQA,
ScienceQA, A-OKVQA), synthetic scenes with answers known by construction (counting, small text, misleading charts,
spatial relations) and Eikos text decisions, with a cross-entropy loss on the decision readout. A rank-32 LoRA on every
language-model projection and the readout were trained in two runs and averaged in weight space, then merged into the
weights. Details are in the model card.

<h2 align="center">Licence</h2>

Apache-2.0 (LICENSE). Third-party work and the changes made are listed in NOTICE.
