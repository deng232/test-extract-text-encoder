from __future__ import annotations

from typing import Final

SOURCE_EMBED_KEY: Final = "language_model.model.embed_tokens.weight"
SOURCE_LAYER_PREFIX: Final = "language_model.model.layers."
SOURCE_LANGUAGE_PREFIX: Final = "language_model."
MAX_KEPT_LAYER: Final = 29

LAYER_SUFFIX_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "input_layernorm.weight": (5120,),
    "post_attention_layernorm.weight": (5120,),
    "self_attn.q_proj.weight": (4096, 5120),
    "self_attn.k_proj.weight": (1024, 5120),
    "self_attn.v_proj.weight": (1024, 5120),
    "self_attn.o_proj.weight": (5120, 4096),
    "mlp.gate_proj.weight": (32768, 5120),
    "mlp.up_proj.weight": (32768, 5120),
    "mlp.down_proj.weight": (5120, 32768),
}

EXPECTED_MODEL_TENSOR_COUNT: Final = 1 + 30 * len(LAYER_SUFFIX_SHAPES)
EXPECTED_TOTAL_TENSOR_COUNT: Final = EXPECTED_MODEL_TENSOR_COUNT + 1
EXPECTED_EMBEDDING_SHAPE: Final = (131072, 5120)

EXPECTED_TEXT_CONFIG: Final[dict[str, int | float]] = {
    "hidden_size": 5120,
    "intermediate_size": 32768,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 131072,
    "rope_theta": 1_000_000_000.0,
}

FORBIDDEN_OUTPUT_PREFIXES: Final = (
    "model.layers.30.",
    "model.layers.31.",
    "model.layers.32.",
    "model.layers.33.",
    "model.layers.34.",
    "model.layers.35.",
    "model.layers.36.",
    "model.layers.37.",
    "model.layers.38.",
    "model.layers.39.",
)

FORBIDDEN_OUTPUT_KEYS: Final = {
    "model.norm.weight",
    "language_model.lm_head.weight",
    "model.lm_head.weight",
    "lm_head.weight",
}

