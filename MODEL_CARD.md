---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: image-text-to-text
language:
- en
tags:
- decision-model
- typed-decisions
- vision-language
- calibrated-probabilities
- jev
---

# Fovea-4B

Fovea-4B answers typed questions about an image and a record in one forward pass, with a probability for every option.
It reads photos, documents, receipts, charts and screenshots, together with a JSON state, and answers yes/no (`noul`),
multiple-choice (`choice`) and ordinal (`score`) questions over TypeSafe's System One API (`POST /v1/systemone`).

| | |
|---|---|
| Parameters | 4B (Qwen3.5-4B architecture: hybrid full attention and Gated DeltaNet, vision encoder) |
| Weights | bfloat16, all adapters merged; decision readout Linear[256, 2560] in float32 |
| Decision | one forward pass; option codes A, B, ... read at the answer position; an internal `unknown` code absorbs mass when the evidence is insufficient |
| Calibration | temperature per question type: 1.759 for `noul` and `score`, 2.322 for `choice` (calibration.json) |
| Images | any number per request; images larger than 2,097,152 pixels are downscaled, keeping the aspect ratio |
| Licence | Apache-2.0 |

## Serving

```sh
git clone https://github.com/Sivasuthan9/fovea && cd fovea
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python scripts/serve.py --model Sivasuthan/fovea-4b --port 8000
```

The server loads the model, records CUDA graphs and warms up the kernels before it reports `ready`; then
`POST http://127.0.0.1:8000/v1/systemone` with a System One request, for example

```json
{"state": {"photo": "img-1"},
 "assets": {"img-1": {"type": "image", "format": "jpg", "data_base64": "..."}},
 "questions": {"decision": {"type": "choice", "instructions": "What is the total on the receipt in `photo`?",
                            "criteria": {"12.40": null, "14.20": null, "21.40": null}}}}
```

returns `{"model": "fovea-4b", "answers": {"decision": {"type": "choice", "choice": "14.20", "probabilities": {...},
"confidence": ...}}, "usage": {...}}`. Images may also be sent as `data:image/...;base64,` URIs inside the state or as a
top-level `images` list. Every answer covers exactly the requested labels and sums to 1; a question that cannot be scored
gets a uniform answer rather than an error.

## Evaluation

Fovea-4B against the two strongest open 4B models on the ImageJevBench v0.3 leaderboard by Intelligence, Imajev-4B (#1)
and H2O-Lightning-4B image (#2), and the larger Jev-Vision. All four answer the same 2,360 held-out multiple-choice
questions, each served as its authors release it. Scores are Intelligence, 100 × (accuracy − chance) / (1 − chance);
the best value in each column is in bold.

| Model | Size | **Overall** | Documents | Charts | Spatial | Diagrams | Multi-image | Counting | General photos | ECE ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fovea-4B (ours)** | 4B | **80.6** | 99.0 | **86.2** | **57.4** | 76.3 | **52.3** | **71.6** | 45.3 | **0.035** |
| H2O-Lightning-4B image | 4B | 79.9 | **99.3** | 84.8 | 56.4 | **77.0** | 45.9 | 70.5 | **47.1** | 0.061 |
| Imajev-4B | 4B | 79.4 | 98.7 | 84.8 | 55.8 | 74.7 | 49.3 | 70.5 | 43.6 | 0.059 |
| Jev-Vision | 8B | 75.7 | 98.0 | 70.3 | **57.4** | 74.7 | 43.4 | 56.8 | 45.8 | 0.179 |

Paired on the same questions, Fovea-4B leads Imajev-4B by +1.1 points (95% interval +0.3 to +2.0), H2O-Lightning-4B by
+0.7 (−0.7 to +2.0) and Jev-Vision by +4.9 (+3.0 to +6.7) overall. Fovea-4B answers 80.6% of the questions correctly.

**How each model was served.** Fovea-4B: images up to 2 MP, its calibrated temperatures. Imajev-4B: native resolution,
temperature 1.305. H2O-Lightning-4B image: images up to 1 MP, temperature 0.65. Jev-Vision: its own decision function,
images up to 1.29 MP, temperature 0.5.

**Evaluation data.** Test or validation splits of public benchmarks, none used for training: DocVQA (400 document
questions), ChartQA (300) and SEED-Bench-2-Plus charts (100), AI2D (300 diagrams) and SEED-Bench-2-Plus maps (100),
CV-Bench (300: depth, distance, spatial relation, counting), BLINK (560: spatial reasoning, counting and nine
multi-image tasks) and MMStar (300 general photos). Overall weights the categories by the ImageJevBench core-track mix;
ECE is the ten-bin expected calibration error of the top answer. This is our own held-out evaluation, not a leaderboard
result: the sets are easier than ImageJevBench's sealed items and contain only multiple-choice questions, so the
absolute numbers do not predict a leaderboard score.

## Training

Fovea-4B was trained on 57,441 typed image decisions with a cross-entropy loss on the decision readout (teacher
distributions where available), options reshuffled every time a question is seen.

| Skill | Source (train splits only) | Questions |
|---|---|---|
| Counting, small text, misleading charts, spatial relations | Synthetic scenes with answers known by construction | 16,000 |
| Small text in photos | TextVQA | 7,000 |
| Documents | DocVQA | 7,000 |
| Charts | ChartQA | 5,000 |
| Spatial relations and attributes | GQA (balanced) | 4,000 |
| Diagrams | ScienceQA | 3,000 |
| Everyday photos | A-OKVQA | 2,000 |
| Text decisions | Eikos decisions (CC-BY-4.0) | 4,000 |

Each source question becomes a `choice` question, and a share of them also a `noul` claim check (does the image confirm a
claimed answer?) or, for exact counts, a `score` question on an ordinal scale. A LoRA of rank 32 was trained on every
language-model projection (attention, MLP and the Gated DeltaNet projections; the vision encoder frozen), together with
the readout, in two runs (peak learning rate 5e-5 and 2e-5, 8 questions per step, cosine decay to 10%). The released
weights are the exact weight-space average of the best checkpoint of each run, merged into the model. 300 source
questions were held out and checked every 500 steps. No benchmark test items were used for training.

## Limits

- One forward pass, no reasoning: multi-step arithmetic and long chains of inference are weak.
- English only.
- Very small text in large images is read only as well as the image resolution allows (2,097,152 pixels per image).

## Acknowledgements

Training was initialised from Imajev-4B (Apache-2.0); the base model is Qwen3.5-4B (Apache-2.0). See NOTICE.
