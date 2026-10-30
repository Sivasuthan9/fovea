"""Fovea-4B: calibrated single-pass decisions about images, served over TypeSafe's System One API.

`FoveaClient` talks to a running server and needs no extra packages; `FoveaLocal` runs the model on your own GPU and
needs `pip install "fovea-sdk[local]"`.
"""
from .client import Answer, FoveaClient, FoveaError, build_request, encode_image, parse_answer

__version__ = "0.1.0"
__all__ = ["Answer", "FoveaClient", "FoveaError", "FoveaLocal", "build_request", "encode_image", "parse_answer"]


def __getattr__(name):
    if name == "FoveaLocal":  # imported on first use, so the client never needs torch
        from .local import FoveaLocal
        return FoveaLocal
    raise AttributeError(name)
