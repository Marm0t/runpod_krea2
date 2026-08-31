from __future__ import annotations

import base64
import io
import os
import secrets
import threading
import time
from typing import Any

import torch
from diffusers import Krea2Pipeline

from worker_core import InputError, LoraRegistry, adapter_configuration, parse_request


MODEL_ID = os.getenv("MODEL_ID", "krea/Krea-2-Turbo")
LORA_DIR = "/runpod-volume/lora"
LOAD_MODE = os.getenv("LOAD_MODE", "cuda").lower()
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(2048 * 1024)))
HF_CACHE_DIR = os.getenv("HF_HOME", "/runpod-volume/huggingface-cache")
REGISTRY = LoraRegistry(LORA_DIR)


class KreaWorker:
    def __init__(self, registry: LoraRegistry) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        if LOAD_MODE not in {"cuda", "cpu_offload"}:
            raise RuntimeError("LOAD_MODE must be 'cuda' or 'cpu_offload'")

        self.registry = registry
        self.turbo = "turbo" in MODEL_ID.lower()
        self.lock = threading.Lock()

        print(f"Loading {MODEL_ID} (mode={LOAD_MODE}, cache={HF_CACHE_DIR})")
        load_started = time.perf_counter()
        self.pipe = Krea2Pipeline.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            cache_dir=HF_CACHE_DIR,
            token=os.getenv("HF_TOKEN") or None,
        )

        for item in self.registry.items:
            print(f"Loading LoRA {item.name} from {item.path.name}")
            self.pipe.load_lora_weights(
                str(item.path),
                adapter_name=item.adapter_name,
                low_cpu_mem_usage=True,
            )
        if self.registry.items:
            self.pipe.disable_lora()

        # Install Accelerate offload hooks only after all adapters are loaded.
        # Model-level offload still places the complete Krea transformer on the
        # GPU and is too large for 24 GB cards during inference.
        if LOAD_MODE == "cpu_offload":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        print(f"Worker ready in {time.perf_counter() - load_started:.1f}s")

    def generate(self, data: Any) -> dict[str, Any]:
        request = parse_request(
            data,
            self.registry,
            turbo=self.turbo,
            max_pixels=MAX_PIXELS,
        )
        seed = request["seed"] if request["seed"] is not None else secrets.randbelow(2**63)
        seeds = [seed + index for index in range(request["num_images"])]
        if seeds[-1] >= 2**63:
            raise InputError("seed is too large for the requested num_images")

        with self.lock:
            selected = request["loras"]
            if selected:
                self.pipe.enable_lora()
                adapter_names, adapter_weights = adapter_configuration(selected)
                self.pipe.set_adapters(
                    adapter_names,
                    adapter_weights=adapter_weights,
                )
            elif self.registry.items:
                self.pipe.disable_lora()

            generators = [torch.Generator(device="cuda").manual_seed(value) for value in seeds]
            pipeline_args: dict[str, Any] = {
                "prompt": request["prompt"],
                "height": request["height"],
                "width": request["width"],
                "num_images_per_prompt": request["num_images"],
                "num_inference_steps": request["num_inference_steps"],
                "guidance_scale": request["guidance_scale"],
                "generator": generators,
            }
            if request["negative_prompt"]:
                pipeline_args["negative_prompt"] = request["negative_prompt"]

            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    images = self.pipe(**pipeline_args).images
            except torch.cuda.OutOfMemoryError:
                if LOAD_MODE == "cpu_offload":
                    try:
                        self.pipe.enable_sequential_cpu_offload()
                    except Exception as reset_error:
                        print(f"Failed to reset CPU offload hooks: {reset_error}")
                torch.cuda.empty_cache()
                raise
            elapsed = time.perf_counter() - started

        encoded_images = []
        for image, image_seed in zip(images, seeds, strict=True):
            buffer = io.BytesIO()
            save_args: dict[str, Any] = {}
            if request["output_format"] in {"jpeg", "webp"}:
                save_args["quality"] = request["quality"]
            if request["output_format"] == "jpeg" and image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffer, format=request["output_format"].upper(), **save_args)
            encoded_images.append(
                {
                    "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "seed": image_seed,
                    "format": request["output_format"],
                }
            )

        return {
            "images": encoded_images,
            "generation_time_seconds": round(elapsed, 3),
            "model": MODEL_ID,
            "loras": [
                {"name": item.name, "scale": scale, "order": index}
                for index, (item, scale) in enumerate(request["loras"])
            ],
        }


_worker: KreaWorker | None = None
_worker_lock = threading.Lock()


def get_worker() -> KreaWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = KreaWorker(REGISTRY)
    return _worker
