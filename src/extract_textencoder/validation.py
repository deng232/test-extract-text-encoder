from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from .constants import (
    EXPECTED_EMBEDDING_SHAPE,
    EXPECTED_MODEL_TENSOR_COUNT,
    EXPECTED_TEXT_CONFIG,
    EXPECTED_TOTAL_TENSOR_COUNT,
    FORBIDDEN_OUTPUT_KEYS,
    FORBIDDEN_OUTPUT_PREFIXES,
    LAYER_SUFFIX_SHAPES,
    MAX_KEPT_LAYER,
    SOURCE_EMBED_KEY,
    SOURCE_LANGUAGE_PREFIX,
)
from .errors import ConversionError

LAYER_PATTERN = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Failed to read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def validate_source_config(config: Mapping[str, Any]) -> None:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or (
        "Mistral3ForConditionalGeneration" not in architectures
    ):
        raise ConversionError(
            "Source must declare Mistral3ForConditionalGeneration."
        )
    if config.get("model_type") != "mistral3":
        raise ConversionError(
            f"Expected model_type='mistral3', got {config.get('model_type')!r}."
        )
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ConversionError("config.json does not contain a text_config object.")

    errors: list[str] = []
    for field, expected in EXPECTED_TEXT_CONFIG.items():
        actual = text_config.get(field)
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and float(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")
    if errors:
        raise ConversionError(
            "Source text architecture is incompatible:\n  - " + "\n  - ".join(errors)
        )


def selected_source_key(key: str) -> bool:
    if key == SOURCE_EMBED_KEY:
        return True
    match = LAYER_PATTERN.fullmatch(key)
    return match is not None and int(match.group(1)) <= MAX_KEPT_LAYER


def output_key_for(source_key: str) -> str:
    if not selected_source_key(source_key):
        raise ConversionError(f"Unexpected selected source key: {source_key}")
    return source_key.removeprefix(SOURCE_LANGUAGE_PREFIX)


def find_selected_weights(index: Mapping[str, Any]) -> dict[str, str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConversionError("Checkpoint index has no valid weight_map object.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
        raise ConversionError("Checkpoint weight_map keys and values must be strings.")

    selected = {key: shard for key, shard in weight_map.items() if selected_source_key(key)}
    expected_keys = {SOURCE_EMBED_KEY}
    for layer in range(MAX_KEPT_LAYER + 1):
        for suffix in LAYER_SUFFIX_SHAPES:
            expected_keys.add(f"language_model.model.layers.{layer}.{suffix}")

    missing = expected_keys - selected.keys()
    unexpected = selected.keys() - expected_keys
    if missing or unexpected or len(selected) != EXPECTED_MODEL_TENSOR_COUNT:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        details.append(
            f"expected {EXPECTED_MODEL_TENSOR_COUNT} selected tensors, got {len(selected)}"
        )
        raise ConversionError("Incompatible checkpoint tensor layout:\n  - " + "\n  - ".join(details))
    return selected


def validate_tekken_bytes(data: bytes) -> None:
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        raise ConversionError("tekken.json is a Git LFS pointer, not tokenizer data.")
    if len(data) < 1_000_000:
        raise ConversionError("tekken.json is unexpectedly small.")
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError("tekken.json is not valid UTF-8 JSON.") from exc
    if not isinstance(parsed, dict):
        raise ConversionError("tekken.json must contain a JSON object.")


def validate_output_tensors(tensors: Mapping[str, torch.Tensor]) -> None:
    errors: list[str] = []
    expected_keys = {"model.embed_tokens.weight", "tekken_model"}
    expected_shapes = {"model.embed_tokens.weight": EXPECTED_EMBEDDING_SHAPE}
    for layer in range(MAX_KEPT_LAYER + 1):
        for suffix, shape in LAYER_SUFFIX_SHAPES.items():
            key = f"model.layers.{layer}.{suffix}"
            expected_keys.add(key)
            expected_shapes[key] = shape

    actual_keys = set(tensors)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected keys: {sorted(extra)}")

    for key, shape in expected_shapes.items():
        tensor = tensors.get(key)
        if tensor is not None:
            if tuple(tensor.shape) != shape:
                errors.append(f"{key}: expected shape {shape}, got {tuple(tensor.shape)}")
            if not tensor.is_floating_point():
                errors.append(f"{key} must be floating point, got {tensor.dtype}")

    tekken = tensors.get("tekken_model")
    if tekken is not None:
        if tekken.dtype != torch.uint8:
            errors.append(f"tekken_model must be uint8, got {tekken.dtype}")
        if tekken.ndim != 1:
            errors.append(f"tekken_model must be one-dimensional, got {tuple(tekken.shape)}")

    if len(actual_keys) != EXPECTED_TOTAL_TENSOR_COUNT:
        errors.append(
            f"expected {EXPECTED_TOTAL_TENSOR_COUNT} tensors, got {len(actual_keys)}"
        )
    if errors:
        raise ConversionError("Output validation failed:\n  - " + "\n  - ".join(errors))


def inspect_saved_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ConversionError(f"Checkpoint does not exist: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        errors: list[str] = []
        if len(keys) != EXPECTED_TOTAL_TENSOR_COUNT:
            errors.append(f"expected {EXPECTED_TOTAL_TENSOR_COUNT} tensors, got {len(keys)}")
        if any(key.startswith(FORBIDDEN_OUTPUT_PREFIXES) for key in keys):
            errors.append("checkpoint contains layers above layer 29")
        forbidden = keys & FORBIDDEN_OUTPUT_KEYS
        if forbidden:
            errors.append(f"forbidden keys present: {sorted(forbidden)}")

        expected_keys = {"model.embed_tokens.weight", "tekken_model"}
        for layer in range(MAX_KEPT_LAYER + 1):
            for suffix in LAYER_SUFFIX_SHAPES:
                expected_keys.add(f"model.layers.{layer}.{suffix}")
        missing = expected_keys - keys
        extra = keys - expected_keys
        if missing:
            errors.append(f"missing keys: {sorted(missing)}")
        if extra:
            errors.append(f"unexpected keys: {sorted(extra)}")

        expected_shapes = {"model.embed_tokens.weight": EXPECTED_EMBEDDING_SHAPE}
        for layer in range(MAX_KEPT_LAYER + 1):
            for suffix, shape in LAYER_SUFFIX_SHAPES.items():
                expected_shapes[f"model.layers.{layer}.{suffix}"] = shape

        model_dtypes: set[str] = set()
        for key, expected_shape in expected_shapes.items():
            if key not in keys:
                continue
            shape = tuple(reader.get_slice(key).get_shape())
            if shape != expected_shape:
                errors.append(f"{key}: expected shape {expected_shape}, got {shape}")
            tensor = reader.get_tensor(key)
            if not tensor.is_floating_point():
                errors.append(f"{key} must be floating point, got {tensor.dtype}")
            model_dtypes.add(str(tensor.dtype))
            del tensor
        tekken_size = 0
        if "tekken_model" in keys:
            tekken = reader.get_tensor("tekken_model")
            tekken_size = tekken.numel()
            if tekken.dtype != torch.uint8 or tekken.ndim != 1:
                errors.append("tekken_model must be a one-dimensional uint8 tensor")
            else:
                validate_tekken_bytes(tekken.numpy().tobytes())

        if errors:
            raise ConversionError("Checkpoint inspection failed:\n  - " + "\n  - ".join(errors))
        metadata = reader.metadata() or {}

    return {
        "path": path,
        "size": path.stat().st_size,
        "tensor_count": len(keys),
        "tekken_bytes": tekken_size,
        "model_dtypes": sorted(model_dtypes),
        "metadata": metadata,
    }
