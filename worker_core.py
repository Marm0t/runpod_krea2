from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """An error that can safely be returned to the API client."""


@dataclass(frozen=True)
class LoraInfo:
    name: str
    path: Path
    adapter_name: str
    aliases: tuple[str, ...]
    base_model: str | None
    trigger_words: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "file": self.path.name,
            "base_model": self.base_model,
            "trigger_words": list(self.trigger_words),
        }


def _safetensors_metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as file:
        raw_length = file.read(8)
        if len(raw_length) != 8:
            raise ValueError("file is too short to be a safetensors checkpoint")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 100 * 1024 * 1024:
            raise ValueError("safetensors header is unexpectedly large")
        header = json.loads(file.read(header_length))
    metadata = header.get("__metadata__", {})
    return metadata if isinstance(metadata, dict) else {}


def _trigger_words(metadata: dict[str, str]) -> tuple[str, ...]:
    raw = metadata.get("ss_tag_frequency")
    if not raw:
        return ()
    try:
        frequencies = json.loads(raw)
        tags: list[str] = []
        for group in frequencies.values():
            if isinstance(group, dict):
                tags.extend(str(tag) for tag in group)
        return tuple(dict.fromkeys(tags))
    except (TypeError, ValueError):
        return ()


class LoraRegistry:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.items: list[LoraInfo] = []
        self._lookup: dict[str, LoraInfo] = {}

        if not self.directory.exists():
            return

        for index, path in enumerate(sorted(self.directory.glob("*.safetensors"))):
            try:
                metadata = _safetensors_metadata(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot read LoRA metadata from {path}: {exc}") from exc

            name = path.stem
            aliases = tuple(
                alias
                for alias in dict.fromkeys(
                    [metadata.get("name"), metadata.get("ss_output_name")]
                )
                if alias and alias != name
            )
            item = LoraInfo(
                name=name,
                path=path,
                adapter_name=f"local_lora_{index}",
                aliases=aliases,
                base_model=metadata.get("ss_base_model_version"),
                trigger_words=_trigger_words(metadata),
            )
            for key in (name, path.name, *aliases):
                if key in self._lookup:
                    other = self._lookup[key]
                    raise RuntimeError(
                        f"Duplicate LoRA name/alias {key!r}: {other.path.name} and {path.name}"
                    )
                self._lookup[key] = item
            self.items.append(item)

    def resolve(self, name: str) -> LoraInfo:
        try:
            return self._lookup[name]
        except KeyError as exc:
            available = ", ".join(item.name for item in self.items) or "none"
            raise InputError(f"Unknown LoRA {name!r}. Available: {available}") from exc

    def public_list(self) -> list[dict[str, Any]]:
        return [item.public_dict() for item in self.items]


def adapter_configuration(
    selected_loras: list[tuple[LoraInfo, float]],
) -> tuple[list[str], list[float]]:
    """Return aligned adapter names and weights without changing request order."""
    return (
        [item.adapter_name for item, _ in selected_loras],
        [scale for _, scale in selected_loras],
    )


def _bounded_int(data: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InputError(f"{key} must be an integer from {low} to {high}")
    return value


def _finite_float(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{label} must be a number from {low} to {high}")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise InputError(f"{label} must be a number from {low} to {high}")
    return result


def parse_request(
    data: Any,
    registry: LoraRegistry,
    *,
    turbo: bool,
    max_pixels: int,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InputError("input must be a JSON object")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputError("prompt must be a non-empty string")
    if len(prompt) > 10_000:
        raise InputError("prompt is too long (maximum 10000 characters)")

    width = _bounded_int(data, "width", 1024, 256, 2048)
    height = _bounded_int(data, "height", 1024, 256, 2048)
    if width % 16 or height % 16:
        raise InputError("width and height must be multiples of 16")
    if width * height > max_pixels:
        raise InputError(f"width * height must not exceed {max_pixels}")

    num_images = _bounded_int(data, "num_images", 1, 1, 4)
    steps = _bounded_int(data, "num_inference_steps", 8 if turbo else 28, 1, 100)
    guidance = _finite_float(
        data.get("guidance_scale", 0.0 if turbo else 4.5),
        "guidance_scale",
        0.0,
        20.0,
    )

    negative_prompt = data.get("negative_prompt")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise InputError("negative_prompt must be a string")
    if turbo and negative_prompt:
        raise InputError("negative_prompt is not supported by Krea 2 Turbo")

    seed = data.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise InputError("seed must be an integer from 0 to 2^63-1")

    raw_loras = data.get("loras", [])
    if not isinstance(raw_loras, list):
        raise InputError("loras must be an array")
    if raw_loras and not registry.items:
        raise InputError(f"No LoRA files found in {registry.directory}")

    selected_loras: list[tuple[LoraInfo, float]] = []
    seen: set[str] = set()
    for index, raw_lora in enumerate(raw_loras):
        if not isinstance(raw_lora, dict):
            raise InputError(f"loras[{index}] must be an object")
        name = raw_lora.get("name")
        if not isinstance(name, str) or not name:
            raise InputError(f"loras[{index}].name must be a non-empty string")
        item = registry.resolve(name)
        if item.name in seen:
            raise InputError(f"LoRA {item.name!r} is selected more than once")
        seen.add(item.name)
        scale = _finite_float(raw_lora.get("scale", 1.0), f"loras[{index}].scale", -2.0, 2.0)
        selected_loras.append((item, scale))

    output_format = data.get("output_format", "png").lower()
    if output_format not in {"png", "jpeg", "webp"}:
        raise InputError("output_format must be png, jpeg, or webp")
    quality = _bounded_int(data, "quality", 95, 1, 100)

    return {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_images": num_images,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "loras": selected_loras,
        "output_format": output_format,
        "quality": quality,
    }
