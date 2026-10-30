"""Fovea-4B on your own GPU, with the same typed helpers as FoveaClient (needs `pip install "fovea-sdk[local]"`).

    from fovea_sdk import FoveaLocal

    fovea = FoveaLocal()                     # downloads Sivasuthan/fovea-4b on first use
    fovea.noul("The parcel in `image` is damaged.", image="parcel.jpg").probability
"""
from __future__ import annotations

from typing import Optional

from .client import _Decisions


class FoveaLocal(_Decisions):
    """Runs the model in this process through the same request path as the server, so answers are identical."""

    def __init__(self, model: str = "Sivasuthan/fovea-4b", revision: Optional[str] = None, device: str = "cuda",
                 fast: bool = True, max_pixels: Optional[int] = None):
        try:
            from .engine import FastPath, FoveaEngine
            from .server import DecisionServer
        except ImportError as e:  # torch / transformers missing
            raise ImportError('FoveaLocal needs the model libraries: pip install "fovea-sdk[local]"') from e
        engine = FoveaEngine.from_pretrained(model, device=device, max_pixels=max_pixels, revision=revision)
        self._server = DecisionServer(engine, "fovea-4b", runner=FastPath(engine) if fast and device.startswith("cuda") else None)

    def ask(self, request: dict) -> dict:
        """Answer a full System One request (any number of questions) and return the response."""
        return self._server.answer(request)
