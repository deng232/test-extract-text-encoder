#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROMPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_LAYER_SELECTION = (10, 20, 30)


class TestConfigurationError(ValueError):
    """Raised when the test TOML is incomplete or unsafe."""


class TestFailure(RuntimeError):
    """Raised when the extracted encoder fails an end-to-end check."""


@dataclass(frozen=True)
class PromptConfig:
    name: str
    text: str


@dataclass(frozen=True)
class TestConfig:
    checkpoint: Path
    source_model_repo: str
    processor_repo: str
    pipeline_repo: str
    device: str
    dtype: str
    output_directory: Path
    seed: int
    max_sequence_length: int
    hidden_state_layers: tuple[int, ...]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    overwrite_outputs: bool
    prompts: tuple[PromptConfig, ...]


def _table(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TestConfigurationError(f"{name} must be a TOML table.")
    return value


def _string(table: Mapping[str, Any], field: str) -> str:
    value = table.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TestConfigurationError(f"{field} must be a non-empty string.")
    return value.strip()


def _integer(table: Mapping[str, Any], field: str) -> int:
    value = table.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TestConfigurationError(f"{field} must be an integer.")
    return value


def _number(table: Mapping[str, Any], field: str) -> float:
    value = table.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TestConfigurationError(f"{field} must be a number.")
    return float(value)


def _boolean(table: Mapping[str, Any], field: str) -> bool:
    value = table.get(field)
    if not isinstance(value, bool):
        raise TestConfigurationError(f"{field} must be true or false.")
    return value


def _relative_path(config_directory: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    return path.resolve()


def load_config(path: Path) -> TestConfig:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise TestConfigurationError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TestConfigurationError(
            f"Failed to read TOML configuration {config_path}: {exc}"
        ) from exc

    model = _table(document.get("model"), "model")
    generation = _table(document.get("generation"), "generation")
    prompt_values = document.get("prompts")
    if not isinstance(prompt_values, list) or len(prompt_values) < 2:
        raise TestConfigurationError("prompts must contain at least two prompt tables.")

    prompts: list[PromptConfig] = []
    names: set[str] = set()
    texts: set[str] = set()
    for index, prompt_value in enumerate(prompt_values, start=1):
        prompt = _table(prompt_value, f"prompts[{index}]")
        name = _string(prompt, "name")
        text = _string(prompt, "text")
        if PROMPT_NAME_PATTERN.fullmatch(name) is None:
            raise TestConfigurationError(
                f"Prompt name {name!r} may contain only letters, digits, '_' and '-'."
            )
        if name in names:
            raise TestConfigurationError(f"Duplicate prompt name: {name}")
        if text in texts:
            raise TestConfigurationError("Prompt texts must be distinct.")
        names.add(name)
        texts.add(text)
        prompts.append(PromptConfig(name=name, text=text))

    dtype = _string(model, "dtype")
    if dtype != "bfloat16":
        raise TestConfigurationError("The GPU test currently supports dtype='bfloat16' only.")

    layer_values = generation.get("hidden_state_layers")
    if not isinstance(layer_values, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in layer_values
    ):
        raise TestConfigurationError("hidden_state_layers must be an integer array.")
    hidden_state_layers = tuple(layer_values)
    if hidden_state_layers != EXPECTED_LAYER_SELECTION:
        raise TestConfigurationError(
            "hidden_state_layers must be exactly [10, 20, 30] for FLUX.2."
        )

    seed = _integer(generation, "seed")
    max_sequence_length = _integer(generation, "max_sequence_length")
    height = _integer(generation, "height")
    width = _integer(generation, "width")
    num_inference_steps = _integer(generation, "num_inference_steps")
    guidance_scale = _number(generation, "guidance_scale")

    if seed < 0:
        raise TestConfigurationError("seed must be non-negative.")
    if max_sequence_length <= 0:
        raise TestConfigurationError("max_sequence_length must be positive.")
    if height <= 0 or height % 16 != 0:
        raise TestConfigurationError("height must be positive and divisible by 16.")
    if width <= 0 or width % 16 != 0:
        raise TestConfigurationError("width must be positive and divisible by 16.")
    if num_inference_steps <= 0:
        raise TestConfigurationError("num_inference_steps must be positive.")
    if guidance_scale <= 0:
        raise TestConfigurationError("guidance_scale must be positive.")

    directory = config_path.parent
    return TestConfig(
        checkpoint=_relative_path(directory, _string(model, "checkpoint")),
        source_model_repo=_string(model, "source_model_repo"),
        processor_repo=_string(model, "processor_repo"),
        pipeline_repo=_string(model, "pipeline_repo"),
        device=_string(model, "device"),
        dtype=dtype,
        output_directory=_relative_path(
            directory, _string(generation, "output_directory")
        ),
        seed=seed,
        max_sequence_length=max_sequence_length,
        hidden_state_layers=hidden_state_layers,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        overwrite_outputs=_boolean(generation, "overwrite_outputs"),
        prompts=tuple(prompts),
    )


def output_paths(config: TestConfig) -> tuple[Path, ...]:
    return tuple(config.output_directory / f"{prompt.name}.png" for prompt in config.prompts)


def validate_output_targets(config: TestConfig) -> None:
    if config.output_directory.exists() and not config.output_directory.is_dir():
        raise TestConfigurationError(
            f"Output path is not a directory: {config.output_directory}"
        )
    existing = [path for path in output_paths(config) if path.exists()]
    if existing and not config.overwrite_outputs:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise TestConfigurationError(
            "Output images already exist. Set overwrite_outputs=true or move them:\n"
            f"{formatted}"
        )


def validate_prompt_embeddings(prompt_embeds: Any, config: TestConfig) -> list[float]:
    import torch

    expected_shape = (
        len(config.prompts),
        config.max_sequence_length,
        EXPECTED_HIDDEN_SIZE * len(config.hidden_state_layers),
    )
    if tuple(prompt_embeds.shape) != expected_shape:
        raise TestFailure(
            f"Unexpected prompt embedding shape: expected {expected_shape}, "
            f"got {tuple(prompt_embeds.shape)}."
        )
    if not torch.isfinite(prompt_embeds).all().item():
        raise TestFailure("Prompt embeddings contain NaN or infinity.")

    differences = [
        (prompt_embeds[0] - prompt_embeds[index]).abs().mean().item()
        for index in range(1, len(prompt_embeds))
    ]
    if not differences or any(value <= 0 for value in differences):
        raise TestFailure("Distinct prompts produced identical embeddings.")
    return differences


def run_gpu_test(config: TestConfig) -> None:
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from diffusers import Flux2Pipeline
    from safetensors import safe_open
    from torch import nn
    from transformers import AutoConfig, AutoProcessor, MistralConfig, MistralModel

    from extract_textencoder.validation import inspect_saved_file

    if not torch.cuda.is_available():
        raise TestFailure("CUDA is not available.")
    try:
        device = torch.device(config.device)
    except (RuntimeError, ValueError) as exc:
        raise TestConfigurationError(f"Invalid device {config.device!r}.") from exc
    if device.type != "cuda":
        raise TestConfigurationError("The full image test requires a CUDA device.")

    dtype = torch.bfloat16
    print("Inspecting extracted checkpoint...")
    inspection = inspect_saved_file(config.checkpoint)
    print(
        f"Checkpoint valid: {inspection['tensor_count']} tensors, "
        f"{inspection['model_dtypes']}"
    )

    with safe_open(
        str(config.checkpoint), framework="pt", device="cpu"
    ) as reader:
        tekken_shape = tuple(reader.get_slice("tekken_model").get_shape())

    complete_config = AutoConfig.from_pretrained(config.source_model_repo)
    text_config = complete_config.text_config
    if isinstance(text_config, dict):
        text_config = MistralConfig(**text_config)
    elif not isinstance(text_config, MistralConfig):
        text_config = MistralConfig(**text_config.to_dict())

    for field, expected in {
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "intermediate_size": 32768,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 131072,
    }.items():
        actual = getattr(text_config, field)
        if actual != expected:
            raise TestFailure(
                f"Source configuration has incompatible {field}: "
                f"expected {expected}, got {actual}."
            )
    text_config.num_hidden_layers = 30
    text_config.use_cache = False
    text_config.output_hidden_states = True

    class ExtractedFlux2TextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = MistralModel(text_config)
            self.model.norm = nn.Identity()
            self.register_buffer(
                "tekken_model",
                torch.empty(tekken_shape, dtype=torch.uint8),
                persistent=True,
            )

        @property
        def dtype(self) -> torch.dtype:
            return self.model.embed_tokens.weight.dtype

        @property
        def device(self) -> torch.device:
            return self.model.embed_tokens.weight.device

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_hidden_states: bool = True,
            use_cache: bool = False,
            **kwargs: Any,
        ) -> Any:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                **kwargs,
            )

    print(f"Loading extracted encoder on {device}...")
    # Keep the 35+ GiB of parameters on the meta device until Accelerate loads
    # their checkpoint values. Buffers must remain real CPU tensors because
    # Transformers creates non-persistent runtime buffers (for example RoPE
    # state) that are intentionally absent from the safetensors checkpoint.
    # Making those buffers meta tensors leaves Accelerate with no data to move
    # when it dispatches the completed module to CUDA.
    with init_empty_weights(include_buffers=False):
        encoder = ExtractedFlux2TextEncoder()
    encoder = load_checkpoint_and_dispatch(
        encoder,
        checkpoint=str(config.checkpoint),
        device_map={"": str(device)},
        dtype=dtype,
        no_split_module_classes=["MistralDecoderLayer"],
    )
    encoder.eval()

    print(f"Loading processor: {config.processor_repo}")
    processor = AutoProcessor.from_pretrained(config.processor_repo)
    embedding_helper = getattr(
        Flux2Pipeline, "_get_mistral_3_small_prompt_embeds", None
    )
    if embedding_helper is None:
        raise TestFailure(
            "Installed Diffusers does not provide the FLUX.2 Mistral prompt "
            "embedding helper expected by this test."
        )

    prompt_texts = [prompt.text for prompt in config.prompts]
    print(f"Encoding {len(prompt_texts)} prompts...")
    with torch.inference_mode():
        prompt_embeds = embedding_helper(
            text_encoder=encoder,
            tokenizer=processor,
            prompt=prompt_texts,
            dtype=dtype,
            device=device,
            max_sequence_length=config.max_sequence_length,
            hidden_states_layers=config.hidden_state_layers,
        )

    differences = validate_prompt_embeddings(prompt_embeds, config)
    print(f"Prompt embedding shape: {tuple(prompt_embeds.shape)}")
    print(f"Mean prompt differences: {differences}")

    prompt_embeds = prompt_embeds.detach().to(device="cpu", dtype=dtype)
    del encoder
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    print("Extracted encoder unloaded.")
    print(f"Loading image pipeline: {config.pipeline_repo}")
    pipe = Flux2Pipeline.from_pretrained(
        config.pipeline_repo,
        text_encoder=None,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    config.output_directory.mkdir(parents=True, exist_ok=True)
    print("Generating images...")
    for index, (prompt, path) in enumerate(zip(config.prompts, output_paths(config))):
        generator = torch.Generator(device=device).manual_seed(config.seed)
        embeddings = prompt_embeds[index : index + 1].to(device=device, dtype=dtype)
        with torch.inference_mode():
            image = pipe(
                prompt_embeds=embeddings,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            ).images[0]
        image.save(path)
        print(f"Saved {prompt.name}: {path}")

    print("End-to-end FLUX.2 encoder test passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an extracted FLUX.2 text encoder through image generation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("test_extracted.toml"),
        help="TOML configuration path (default: test_extracted.toml beside this script)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = load_config(build_parser().parse_args(argv).config)
        validate_output_targets(config)
        run_gpu_test(config)
        return 0
    except (TestConfigurationError, TestFailure) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
