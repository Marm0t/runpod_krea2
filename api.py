from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from handler import MODEL_ID, REGISTRY, get_worker
from worker_core import InputError, parse_request


_ready = threading.Event()
_initialization_error: Exception | None = None


def _initialize_worker() -> None:
    global _initialization_error
    try:
        get_worker()
        _ready.set()
    except Exception as exc:
        _initialization_error = exc
        print(f"Worker initialization failed: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_initialize_worker, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="Krea 2 LoRA worker", version="1.0.0", lifespan=lifespan)


@app.get("/ping")
async def ping() -> Response:
    if _initialization_error is not None:
        return JSONResponse(
            {
                "status": "model_error",
                "model_ready": False,
                "detail": str(_initialization_error),
            },
            status_code=500,
        )
    if not _ready.is_set():
        # RunPod Load Balancer interprets 204 as "initializing" and does not
        # route generation traffic to this worker until /ping returns 200.
        return Response(status_code=204)
    return JSONResponse({"status": "healthy", "model_ready": True})


@app.get("/loras")
async def list_loras() -> dict[str, Any]:
    return {
        "loras": REGISTRY.public_list(),
        "model": MODEL_ID,
        "directory": str(REGISTRY.directory),
    }


@app.post("/generate")
async def generate(data: dict[str, Any]) -> dict[str, Any]:
    try:
        parse_request(
            data,
            REGISTRY,
            turbo="turbo" in MODEL_ID.lower(),
            max_pixels=int(os.getenv("MAX_PIXELS", str(2048 * 1024))),
        )
    except InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _initialization_error is not None:
        raise HTTPException(status_code=503, detail=str(_initialization_error))
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still loading")
    try:
        return await run_in_threadpool(get_worker().generate, data)
    except InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"HTTP generation failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
