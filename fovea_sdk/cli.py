"""Command line: `fovea-sdk serve` starts the System One server; `fovea-sdk ask` asks one question.

  fovea-sdk serve --port 8000
  fovea-sdk ask --image receipt.jpg --question "What is the total?" --options 12.40,14.20,21.40
  fovea-sdk ask --image parcel.jpg --question "The parcel is damaged." --yes-no
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _serve(args):
    from .engine import FastPath, FoveaEngine
    from .server import DecisionServer, serve

    t0 = time.time()
    engine = FoveaEngine.from_pretrained(args.model, max_pixels=args.max_pixels, revision=args.revision)
    server = DecisionServer(engine, "fovea-4b", runner=None if args.eager else FastPath(engine))
    server.warm_up()
    print(f"ready in {time.time() - t0:.0f} s: fovea-4b on http://{args.host}:{args.port}/v1/systemone "
          f"(max_pixels={engine.max_pixels}, temperature={engine.temperature}, {'eager' if args.eager else 'CUDA graphs'})",
          flush=True)
    serve(server, args.host, args.port)


def _ask(args):
    from .client import FoveaClient

    if args.local:
        from .local import FoveaLocal
        fovea = FoveaLocal(args.model, revision=args.revision)
    else:
        fovea = FoveaClient(args.url)
    if args.yes_no:
        a = fovea.noul(args.question, image=args.image)
    elif args.options:
        a = fovea.choice(args.question, [o.strip() for o in args.options.split(",")], image=args.image)
    elif args.levels:
        a = fovea.score(args.question, [lv.strip() for lv in args.levels.split(",")], image=args.image)
    else:
        sys.exit("give --yes-no, --options or --levels")
    print(json.dumps({"answer": a.value, "probability": round(a.probability, 4),
                      "probabilities": {k: round(v, 4) for k, v in a.probabilities.items()}}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="fovea-sdk", description="Fovea-4B: calibrated decisions about images.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="start the System One server (needs fovea-sdk[local] and a GPU)")
    s.add_argument("--model", default="Sivasuthan/fovea-4b", help="Hugging Face repo id or local directory")
    s.add_argument("--revision", default=None, help="pin a Hugging Face revision (commit hash)")
    s.add_argument("--max-pixels", type=int, default=None, help="default: the model's calibration.json")
    s.add_argument("--eager", action="store_true", help="no CUDA graphs (same answers, slower)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = sub.add_parser("ask", help="ask one question about one image")
    a.add_argument("--image", required=True)
    a.add_argument("--question", required=True)
    kind = a.add_mutually_exclusive_group(required=True)
    kind.add_argument("--yes-no", action="store_true", help="a yes/no (noul) question")
    kind.add_argument("--options", help="comma-separated option names (choice)")
    kind.add_argument("--levels", help="comma-separated level descriptions, lowest first (score)")
    a.add_argument("--url", default="http://127.0.0.1:8000", help="a running server")
    a.add_argument("--local", action="store_true", help="run the model in this process instead of calling a server")
    a.add_argument("--model", default="Sivasuthan/fovea-4b")
    a.add_argument("--revision", default=None)
    args = p.parse_args(argv)
    _serve(args) if args.command == "serve" else _ask(args)


if __name__ == "__main__":
    main()
