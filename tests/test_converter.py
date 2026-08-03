from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import extract_textencoder.validation as validation
from extract_textencoder.constants import EXPECTED_TEXT_CONFIG, LAYER_SUFFIX_SHAPES
from extract_textencoder.converter import convert
from extract_textencoder.sources import LocalSource


def test_miniature_sharded_checkpoint_is_converted_atomically(
    tmp_path: Path, monkeypatch,
) -> None:
    # Preserve the real key count and layer topology while shrinking every
    # model tensor so this integration fixture stays small.
    miniature_shapes = {suffix: (1,) for suffix in LAYER_SUFFIX_SHAPES}
    monkeypatch.setattr(validation, "LAYER_SUFFIX_SHAPES", miniature_shapes)
    monkeypatch.setattr(validation, "EXPECTED_EMBEDDING_SHAPE", (2, 2))

    config = {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "text_config": dict(EXPECTED_TEXT_CONFIG),
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "tekken.json").write_text(
        json.dumps({"padding": "x" * 1_000_000}), encoding="utf-8"
    )

    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        "language_model.model.embed_tokens.weight": torch.ones(
            (2, 2), dtype=torch.bfloat16
        )
    }
    for layer in range(30):
        for suffix in miniature_shapes:
            tensors[f"language_model.model.layers.{layer}.{suffix}"] = torch.ones(
                (1,), dtype=torch.bfloat16
            )
    save_file(tensors, str(tmp_path / shard_name))
    index = {"weight_map": {key: shard_name for key in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    output = tmp_path / "output.safetensors"
    result = convert(source=LocalSource(tmp_path), output=output)

    assert output.is_file()
    assert result["path"] == output.resolve()
    assert not list(tmp_path.glob("*.incomplete"))

