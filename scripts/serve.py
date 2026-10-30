"""Serve Fovea-4B over TypeSafe's System One API: POST /v1/systemone (and GET /health).

  python scripts/serve.py --model Sivasuthan/fovea-4b --port 8000
"""
import argparse
import time

from fovea_sdk.engine import FastPath, FoveaEngine
from fovea_sdk.server import DecisionServer, serve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Sivasuthan/fovea-4b", help="Hugging Face repo id or local directory")
    p.add_argument("--revision", default=None, help="pin a Hugging Face revision (commit hash)")
    p.add_argument("--max-pixels", type=int, default=None, help="default: the value in the model's calibration.json")
    p.add_argument("--eager", action="store_true", help="no CUDA graphs (same function, slower)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    t0 = time.time()
    engine = FoveaEngine.from_pretrained(args.model, max_pixels=args.max_pixels, revision=args.revision)
    runner = None if args.eager else FastPath(engine)
    server = DecisionServer(engine, "fovea-4b", runner=runner)
    server.warm_up()
    print(f"ready in {time.time() - t0:.0f} s: fovea-4b on http://{args.host}:{args.port}/v1/systemone "
          f"(max_pixels={engine.max_pixels}, temperature={engine.temperature}, "
          f"{'eager' if args.eager else 'CUDA graphs'})", flush=True)
    serve(server, args.host, args.port)


if __name__ == "__main__":
    main()
