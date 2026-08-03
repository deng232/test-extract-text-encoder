from __future__ import annotations

import os
import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .constants import EXPECTED_TOTAL_TENSOR_COUNT
from .errors import ConversionError
from .sources import Source
from .validation import (
    find_selected_weights,
    inspect_saved_file,
    load_json_object,
    output_key_for,
    validate_output_tensors,
    validate_source_config,
    validate_tekken_bytes,
)


def human_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{byte_count} B"


def convert(
    *,
    source: Source,
    output: Path,
    overwrite: bool = False,
    allow_non_bf16: bool = False,
) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise ConversionError(
            f"Output already exists: {output}\nUse --overwrite to replace it."
        )
    if output.exists() and not output.is_file():
        raise ConversionError(f"Output path is not a regular file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source:   {source.description}")
    print(f"Revision: {source.revision}")
    print(f"Output:   {output}")
    print("Resolving checkpoint metadata and tokenizer...")

    config = load_json_object(source.get("config.json"))
    validate_source_config(config)
    index = load_json_object(source.get("model.safetensors.index.json"))
    selected = find_selected_weights(index)

    tekken_bytes = source.get("tekken.json").read_bytes()
    validate_tekken_bytes(tekken_bytes)

    shard_names = sorted(set(selected.values()))
    shard_paths: dict[str, Path] = {}
    for position, shard_name in enumerate(shard_names, start=1):
        print(f"Resolving shard {position}/{len(shard_names)}: {shard_name}")
        shard_paths[shard_name] = source.get(shard_name)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.incomplete")
    try:
        with ExitStack() as stack:
            readers = {
                name: stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                for name, path in shard_paths.items()
            }
            output_tensors: dict[str, torch.Tensor] = {}
            for source_key, shard_name in selected.items():
                reader = readers[shard_name]
                if source_key not in reader.keys():
                    raise ConversionError(
                        f"Index maps {source_key!r} to {shard_name!r}, but the "
                        "shard does not contain that tensor."
                    )
                tensor = reader.get_tensor(source_key)
                if not tensor.is_floating_point():
                    raise ConversionError(
                        f"{source_key} has non-floating dtype {tensor.dtype}."
                    )
                if (
                    tensor.dtype != torch.bfloat16
                    and not allow_non_bf16
                ):
                    raise ConversionError(
                        f"{source_key} has dtype {tensor.dtype}; BF16 is required. "
                        "Use --allow-non-bf16 to preserve source dtypes."
                    )
                destination_key = output_key_for(source_key)
                if destination_key in output_tensors:
                    raise ConversionError(f"Duplicate output key: {destination_key}")
                output_tensors[destination_key] = tensor

            output_tensors["tekken_model"] = torch.frombuffer(
                bytearray(tekken_bytes), dtype=torch.uint8
            )
            validate_output_tensors(output_tensors)

            output_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in output_tensors.values()
            )
            free_bytes = shutil.disk_usage(output.parent).free
            print(f"Estimated tensor data: {human_size(output_bytes)}")
            print(f"Free output space:     {human_size(free_bytes)}")
            if free_bytes < output_bytes:
                raise ConversionError(
                    f"Insufficient output space: need at least {human_size(output_bytes)}, "
                    f"have {human_size(free_bytes)}."
                )

            metadata = {
                "format": "pt",
                "model_family": "mistral3",
                "purpose": "flux2_text_encoder",
                "source": source.description,
                "source_revision": source.revision,
                "kept_layers": "0-29",
                "output_hidden_states": "10,20,30",
            }
            print(f"Writing temporary checkpoint: {temporary}")
            save_file(output_tensors, str(temporary), metadata=metadata)

        print("Validating saved checkpoint...")
        result = inspect_saved_file(temporary)
        print("Installing validated checkpoint...")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise

    final_size = output.stat().st_size
    print(f"Completed: {output}")
    print(f"Size:      {human_size(final_size)}")
    print(f"Tensors:   {EXPECTED_TOTAL_TENSOR_COUNT}")
    result["path"] = output
    result["size"] = final_size
    return result
