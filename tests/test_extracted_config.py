from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_extracted_flux2_encoder import (
    EXPECTED_HIDDEN_SIZE,
    TestConfigurationError,
    TestFailure,
    load_config,
    output_paths,
    validate_output_targets,
    validate_prompt_embeddings,
)


def write_config(path: Path, *, overrides: str = "") -> Path:
    path.write_text(
        """
[model]
checkpoint = "encoder.safetensors"
source_model_repo = "owner/source"
processor_repo = "owner/processor"
pipeline_repo = "owner/pipeline"
device = "cuda:0"
dtype = "bfloat16"

[generation]
output_directory = "outputs"
seed = 42
max_sequence_length = 512
hidden_state_layers = [10, 20, 30]
height = 768
width = 768
num_inference_steps = 28
guidance_scale = 4.0
overwrite_outputs = false

[[prompts]]
name = "red_cube"
text = "a red cube"

[[prompts]]
name = "blue_sphere"
text = "a blue sphere"
"""
        + overrides,
        encoding="utf-8",
    )
    return path


def test_loads_config_and_resolves_paths_relative_to_toml(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    assert config.checkpoint == (tmp_path / "encoder.safetensors").resolve()
    assert config.output_directory == (tmp_path / "outputs").resolve()
    assert config.hidden_state_layers == (10, 20, 30)
    assert [prompt.name for prompt in config.prompts] == [
        "red_cube",
        "blue_sphere",
    ]


def test_rejects_missing_configuration(tmp_path: Path) -> None:
    with pytest.raises(TestConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('dtype = "bfloat16"', 'dtype = "float16"', "bfloat16"),
        (
            "hidden_state_layers = [10, 20, 30]",
            "hidden_state_layers = [9, 19, 29]",
            "10, 20, 30",
        ),
        ('name = "blue_sphere"', 'name = "../unsafe"', "Prompt name"),
        ('text = "a blue sphere"', 'text = "a red cube"', "distinct"),
        ("height = 768", "height = 767", "divisible by 16"),
    ],
)
def test_rejects_invalid_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = write_config(tmp_path / "test.toml")
    text = path.read_text(encoding="utf-8").replace(old, new)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TestConfigurationError, match=message):
        load_config(path)


def test_output_names_are_derived_from_safe_prompt_names(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    assert output_paths(config) == (
        (tmp_path / "outputs" / "red_cube.png").resolve(),
        (tmp_path / "outputs" / "blue_sphere.png").resolve(),
    )


def test_existing_outputs_are_protected_by_default(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    config.output_directory.mkdir()
    output_paths(config)[0].write_bytes(b"existing")
    with pytest.raises(TestConfigurationError, match="already exist"):
        validate_output_targets(config)


def test_overwrite_can_be_enabled_explicitly(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    config.output_directory.mkdir()
    output_paths(config)[0].write_bytes(b"existing")
    validate_output_targets(replace(config, overwrite_outputs=True))


def embedding_fixture(config, *, fill: float = 0.0) -> torch.Tensor:
    return torch.full(
        (
            len(config.prompts),
            config.max_sequence_length,
            EXPECTED_HIDDEN_SIZE * len(config.hidden_state_layers),
        ),
        fill,
        dtype=torch.bfloat16,
    )


def test_accepts_finite_distinct_prompt_embeddings(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    embeddings = embedding_fixture(config)
    embeddings[1].fill_(1.0)
    assert validate_prompt_embeddings(embeddings, config) == [1.0]


def test_rejects_wrong_embedding_shape(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    embeddings = torch.zeros((2, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(TestFailure, match="shape"):
        validate_prompt_embeddings(embeddings, config)


def test_rejects_nonfinite_embeddings(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    embeddings = embedding_fixture(config)
    embeddings[0, 0, 0] = float("nan")
    with pytest.raises(TestFailure, match="NaN"):
        validate_prompt_embeddings(embeddings, config)


def test_rejects_identical_embeddings(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "test.toml"))
    with pytest.raises(TestFailure, match="identical"):
        validate_prompt_embeddings(embedding_fixture(config), config)
