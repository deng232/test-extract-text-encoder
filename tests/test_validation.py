from __future__ import annotations

import pytest

from extract_textencoder.constants import (
    EXPECTED_MODEL_TENSOR_COUNT,
    EXPECTED_TEXT_CONFIG,
    LAYER_SUFFIX_SHAPES,
    MAX_KEPT_LAYER,
    SOURCE_EMBED_KEY,
)
from extract_textencoder.errors import ConversionError
from extract_textencoder.validation import (
    find_selected_weights,
    output_key_for,
    selected_source_key,
    validate_source_config,
    validate_tekken_bytes,
)


def compatible_config() -> dict[str, object]:
    return {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "text_config": dict(EXPECTED_TEXT_CONFIG),
    }


def compatible_index() -> dict[str, object]:
    weight_map = {SOURCE_EMBED_KEY: "model-00001.safetensors"}
    for layer in range(MAX_KEPT_LAYER + 1):
        for suffix in LAYER_SUFFIX_SHAPES:
            weight_map[f"language_model.model.layers.{layer}.{suffix}"] = (
                f"model-{layer // 4 + 1:05d}.safetensors"
            )
    weight_map["language_model.model.layers.30.self_attn.q_proj.weight"] = (
        "model-00009.safetensors"
    )
    weight_map["language_model.lm_head.weight"] = "model-00010.safetensors"
    return {"weight_map": weight_map}


def test_accepts_expected_configuration() -> None:
    validate_source_config(compatible_config())


def test_rejects_changed_architecture() -> None:
    config = compatible_config()
    text_config = config["text_config"]
    assert isinstance(text_config, dict)
    text_config["hidden_size"] = 4096
    with pytest.raises(ConversionError, match="hidden_size"):
        validate_source_config(config)


def test_selects_embedding_and_layers_zero_through_twenty_nine() -> None:
    assert selected_source_key(SOURCE_EMBED_KEY)
    assert selected_source_key("language_model.model.layers.29.mlp.down_proj.weight")
    assert not selected_source_key("language_model.model.layers.30.mlp.down_proj.weight")
    assert not selected_source_key("language_model.lm_head.weight")


def test_finds_exact_expected_tensor_set() -> None:
    selected = find_selected_weights(compatible_index())
    assert len(selected) == EXPECTED_MODEL_TENSOR_COUNT
    assert not any("layers.30." in key for key in selected)


def test_rejects_missing_layer_tensor() -> None:
    index = compatible_index()
    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict)
    del weight_map["language_model.model.layers.12.self_attn.k_proj.weight"]
    with pytest.raises(ConversionError, match="missing keys"):
        find_selected_weights(index)


def test_rewrites_only_selected_keys() -> None:
    assert output_key_for(SOURCE_EMBED_KEY) == "model.embed_tokens.weight"
    assert output_key_for(
        "language_model.model.layers.3.self_attn.q_proj.weight"
    ) == "model.layers.3.self_attn.q_proj.weight"
    with pytest.raises(ConversionError):
        output_key_for("language_model.model.layers.31.self_attn.q_proj.weight")


def test_rejects_pointer_and_invalid_tokenizer() -> None:
    with pytest.raises(ConversionError, match="LFS pointer"):
        validate_tekken_bytes(b"version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ConversionError, match="unexpectedly small"):
        validate_tekken_bytes(b"{}")

