
# End-to-End Test of an Extracted FLUX.2 Text Encoder

## What this test proves

This test determines whether the extracted checkpoint is usable as a **FLUX.2 text encoder**, independently of ComfyUI.

The complete path being tested is:

```text
prompt
  ↓
Hugging Face Mistral processor
  ↓
extracted embedding + layers 0–29
  ↓
hidden states 10, 20, and 30
  ↓
[B, 512, 15360] prompt embeddings
  ↓
Hugging Face Diffusers Flux2Pipeline
  ↓
generated image
```

A successful image generation proves that the extracted model produces conditioning that the actual FLUX.2 diffusion transformer can consume.

It does not test:

```text
ComfyUI model detection
ComfyUI key conversion
ComfyUI's embedded tekken_model loader
ComfyUI offloading or quantization
```

Those are separate packaging tests.

---

## Why `text_encoder=None` is used

The extracted file is not a complete Hugging Face:

```text
Mistral3ForConditionalGeneration
```

It intentionally lacks:

```text
vision tower
multimodal projector
layers 30–39
final language-model normalization
LM head
generation support
```

Therefore, it should not be passed directly to:

```python
Mistral3ForConditionalGeneration.from_pretrained(...)
```

Instead, load the extracted transformer with a small adapter, calculate the prompt embeddings, and give those embeddings directly to `Flux2Pipeline`.

Diffusers officially supports this:

```python
pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=None,
)

image = pipe(
    prompt_embeds=prompt_embeds,
).images[0]
```

The official BFL Diffusers guide uses this path for remote text encoding, and the pipeline bypasses its local text encoder whenever `prompt_embeds` is supplied.

---

## Hardware requirement

The extracted BF16 encoder is approximately 35–36 GB. The simplest test loads it onto one sufficiently large GPU, computes the embeddings, unloads it, and then loads a quantized FLUX.2 diffusion pipeline.

Recommended for the simplest test:

```text
GPU VRAM: 48 GB minimum
Safer:     80 GB
CPU RAM:   64 GB or more
Disk:      approximately 80 GB free
```

The text encoder and diffusion transformer do not need to remain loaded simultaneously.

---

## Create the remote-cloud environment with uv

The GPU validator is documentation-only and is not installed as a CLI command.
The `cloud` dependency extra contains Diffusers, Transformers, Accelerate,
bitsandbytes, and Pillow for this walkthrough.

```bash
uv sync --extra cloud
```

Authenticate with Hugging Face:

```bash
hf auth login
```

You must have accepted the FLUX.2-dev repository license before downloading its weights.

---

## Test script

Save this as:

```text
test_extracted_flux2_encoder.py
```

```python
#!/usr/bin/env python3

from __future__ import annotations

import gc
from pathlib import Path

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from diffusers import Flux2Pipeline
from safetensors import safe_open
from torch import nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    MistralConfig,
    MistralModel,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXTRACTED_CHECKPOINT = Path(
    "./mistral_3_small_flux2_huihui_bf16.safetensors"
)

# Use the repository from which the extracted weights originated.
SOURCE_MODEL_REPO = (
    "huihui-ai/"
    "Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated"
)

# The base repository can be used for the processor when the derivative
# repository does not carry all processor files.
PROCESSOR_REPO = (
    "mistralai/"
    "Mistral-Small-3.2-24B-Instruct-2506"
)

# This official Diffusers repository contains a quantized FLUX.2 transformer.
PIPELINE_REPO = "diffusers/FLUX.2-dev-bnb-4bit"

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
SEED = 42

OUTPUT_DIRECTORY = Path("./flux2_encoder_test")

PROMPTS = [
    (
        "A studio photograph of a bright red metal cube standing on a "
        "white table, plain gray background, soft shadow, centered."
    ),
    (
        "A studio photograph of a bright blue glass sphere standing on a "
        "white table, plain gray background, soft shadow, centered."
    ),
]


# ---------------------------------------------------------------------------
# Extracted encoder adapter
# ---------------------------------------------------------------------------

class ExtractedFlux2TextEncoder(nn.Module):
    """
    Minimal Hugging Face-compatible wrapper around the extracted Mistral
    transformer.

    The checkpoint is expected to contain:

        model.embed_tokens.weight
        model.layers.0.*
        ...
        model.layers.29.*
        tekken_model

    It must not require:

        model.norm.weight
        language-model head
        vision tower
        layers 30-39
    """

    def __init__(
        self,
        config: MistralConfig,
        tekken_shape: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.model = MistralModel(config)

        # A normal 30-layer MistralModel would apply its final RMSNorm after
        # layer 29. FLUX.2 needs the raw output after layer 29 because that
        # corresponds to hidden_states[30] in the original 40-layer model.
        #
        # Replacing the final norm with Identity also makes the module's state
        # dictionary match an extraction that deliberately omitted
        # model.norm.weight.
        self.model.norm = nn.Identity()

        # Keep the checkpoint namespace complete so Accelerate does not see
        # tekken_model as an unexpected tensor. The processor used by this
        # test is still loaded separately through AutoProcessor.
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
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Validation and loading
# ---------------------------------------------------------------------------

def inspect_checkpoint(
    checkpoint: Path,
) -> tuple[int, ...]:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Extracted checkpoint not found: {checkpoint}"
        )

    with safe_open(
        str(checkpoint),
        framework="pt",
        device="cpu",
    ) as reader:
        keys = set(reader.keys())

        required_keys = {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.9.self_attn.q_proj.weight",
            "model.layers.19.self_attn.q_proj.weight",
            "model.layers.29.self_attn.q_proj.weight",
            "tekken_model",
        }

        missing = required_keys - keys

        if missing:
            raise RuntimeError(
                "Extracted checkpoint is missing required keys:\n"
                + "\n".join(f"  {key}" for key in sorted(missing))
            )

        forbidden_prefixes = (
            "model.layers.30.",
            "model.layers.31.",
            "model.layers.39.",
        )

        for key in keys:
            if key.startswith(forbidden_prefixes):
                raise RuntimeError(
                    f"Checkpoint was not pruned correctly: {key}"
                )

        if "model.norm.weight" in keys:
            raise RuntimeError(
                "The checkpoint contains model.norm.weight. This test "
                "expects the final normalization to have been removed."
            )

        embedding_shape = tuple(
            reader.get_slice(
                "model.embed_tokens.weight"
            ).get_shape()
        )

        expected_embedding_shape = (131072, 5120)

        if embedding_shape != expected_embedding_shape:
            raise RuntimeError(
                "Unexpected embedding shape:\n"
                f"  expected: {expected_embedding_shape}\n"
                f"  actual:   {embedding_shape}"
            )

        tekken_shape = tuple(
            reader.get_slice("tekken_model").get_shape()
        )

    return tekken_shape


def load_text_config() -> MistralConfig:
    complete_config = AutoConfig.from_pretrained(
        SOURCE_MODEL_REPO,
    )

    text_config = complete_config.text_config

    if isinstance(text_config, dict):
        text_config = MistralConfig(**text_config)

    if not isinstance(text_config, MistralConfig):
        text_config = MistralConfig(
            **text_config.to_dict()
        )

    expected_values = {
        "hidden_size": 5120,
        "intermediate_size": 32768,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 131072,
    }

    for field, expected in expected_values.items():
        actual = getattr(text_config, field)

        if actual != expected:
            raise RuntimeError(
                f"Incompatible {field}: "
                f"expected {expected}, got {actual}"
            )

    # Only the first 30 layers exist in the extracted checkpoint.
    text_config.num_hidden_layers = 30
    text_config.use_cache = False
    text_config.output_hidden_states = True

    return text_config


def load_extracted_encoder(
    checkpoint: Path,
    text_config: MistralConfig,
    tekken_shape: tuple[int, ...],
) -> ExtractedFlux2TextEncoder:
    # Construct the architecture without allocating tens of gigabytes of
    # randomly initialized weights.
    with init_empty_weights(include_buffers=True):
        encoder = ExtractedFlux2TextEncoder(
            config=text_config,
            tekken_shape=tekken_shape,
        )

    # Simplest configuration: load the complete extracted encoder on one GPU.
    encoder = load_checkpoint_and_dispatch(
        encoder,
        checkpoint=str(checkpoint),
        device_map={"": DEVICE},
        dtype=DTYPE,
        no_split_module_classes=["MistralDecoderLayer"],
    )

    encoder.eval()

    return encoder


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_prompts(
    encoder: ExtractedFlux2TextEncoder,
    processor: AutoProcessor,
    prompts: list[str],
) -> torch.Tensor:
    prompt_embeds = (
        Flux2Pipeline._get_mistral_3_small_prompt_embeds(
            text_encoder=encoder,
            tokenizer=processor,
            prompt=prompts,
            dtype=DTYPE,
            device=torch.device(DEVICE),
            max_sequence_length=512,
            hidden_states_layers=(10, 20, 30),
        )
    )

    expected_shape = (
        len(prompts),
        512,
        15360,
    )

    if tuple(prompt_embeds.shape) != expected_shape:
        raise RuntimeError(
            "Text encoder produced an unexpected shape:\n"
            f"  expected: {expected_shape}\n"
            f"  actual:   {tuple(prompt_embeds.shape)}"
        )

    if not torch.isfinite(prompt_embeds).all():
        raise RuntimeError(
            "Text encoder produced NaN or infinity values."
        )

    difference = (
        prompt_embeds[0] - prompt_embeds[1]
    ).abs().mean().item()

    if difference == 0:
        raise RuntimeError(
            "The two different prompts produced identical embeddings."
        )

    print(f"Prompt embedding shape: {tuple(prompt_embeds.shape)}")
    print(f"Mean embedding difference: {difference:.8f}")

    # Move the relatively small conditioning tensors to CPU before deleting
    # the large text encoder.
    return prompt_embeds.detach().to(
        device="cpu",
        dtype=DTYPE,
    )


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def load_image_pipeline() -> Flux2Pipeline:
    pipe = Flux2Pipeline.from_pretrained(
        PIPELINE_REPO,
        text_encoder=None,
        torch_dtype=DTYPE,
    )

    pipe = pipe.to(DEVICE)

    return pipe


@torch.inference_mode()
def generate_images(
    pipe: Flux2Pipeline,
    prompt_embeds: torch.Tensor,
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, prompt in enumerate(PROMPTS):
        # Recreate the generator with the same seed for every image. Therefore
        # the initial latent noise is identical, and the prompt conditioning
        # is the primary intended difference.
        generator = torch.Generator(
            device=DEVICE,
        ).manual_seed(SEED)

        current_embeddings = prompt_embeds[
            index : index + 1
        ].to(
            device=DEVICE,
            dtype=DTYPE,
        )

        image = pipe(
            prompt_embeds=current_embeddings,
            height=768,
            width=768,
            num_inference_steps=28,
            guidance_scale=4.0,
            generator=generator,
        ).images[0]

        output_path = OUTPUT_DIRECTORY / (
            f"prompt_{index + 1}.png"
        )

        image.save(output_path)

        print()
        print(f"Prompt {index + 1}: {prompt}")
        print(f"Saved: {output_path}")


def main() -> None:
    tekken_shape = inspect_checkpoint(
        EXTRACTED_CHECKPOINT
    )

    text_config = load_text_config()

    print("Loading extracted text encoder...")

    encoder = load_extracted_encoder(
        checkpoint=EXTRACTED_CHECKPOINT,
        text_config=text_config,
        tekken_shape=tekken_shape,
    )

    print("Loading Mistral processor...")

    processor = AutoProcessor.from_pretrained(
        PROCESSOR_REPO,
    )

    print("Encoding prompts...")

    prompt_embeds = encode_prompts(
        encoder=encoder,
        processor=processor,
        prompts=PROMPTS,
    )

    # The text encoder is no longer needed. Free its VRAM before loading the
    # diffusion transformer.
    del encoder
    del processor

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print()
    print("Extracted text encoder unloaded.")
    print("Loading FLUX.2 image pipeline...")

    pipe = load_image_pipeline()

    print("Generating images...")

    generate_images(
        pipe=pipe,
        prompt_embeds=prompt_embeds,
    )

    print()
    print("Test completed.")
    print(f"Inspect the images in: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
```

---

## Run the test

```bash
uv run python test_extracted_flux2_encoder.py
```

Expected files:

```text
flux2_encoder_test/
├── prompt_1.png
└── prompt_2.png
```

The two generations use the same seed but different prompt embeddings:

```text
prompt 1:
bright red metal cube

prompt 2:
bright blue glass sphere
```

---

## How to judge the result

### Definite pass

The encoder is usable when all of these are true:

```text
[1] Prompt embedding shape is (2, 512, 15360).
[2] Embeddings contain no NaN or infinity.
[3] The two prompts produce different embeddings.
[4] Flux2Pipeline completes without a tensor-shape error.
[5] Both outputs are recognizable, non-noise images.
[6] Image one depicts a red cube.
[7] Image two depicts a blue sphere.
```

This demonstrates that the extracted model is producing meaningful conditioning consumed by the actual image model.

### Partial pass

If both images are coherent but prompt adherence is weak:

```text
the encoder is technically operational
```

but its altered representations may not align well with the FLUX.2 diffusion transformer.

This can happen with an abliterated or heavily fine-tuned Mistral checkpoint even when extraction is structurally correct.

### Failed conditioning

If both images are nearly identical despite different prompts and identical seeds:

```text
the diffusion pipeline is running,
but the extracted encoder conditioning is probably ineffective
```

Check:

```text
hidden-state layer indices
final normalization removal
weight-key conversion
tokenizer compatibility
whether prompt_embeds were moved to the pipeline GPU
```

### Pure noise or severe corruption

If both outputs are noise or badly corrupted, likely causes include:

```text
wrong source architecture
incorrect tensor-key mapping
incorrect hidden-state selection
final RMSNorm incorrectly retained
corrupted BF16 weights
wrong FLUX.2 diffusion model
```

### Pipeline crashes before denoising

A shape such as this is required:

```text
[batch, 512, 15360]
```

The Diffusers FLUX.2 implementation explicitly stacks hidden-state outputs `10`, `20`, and `30` and reshapes them into the final prompt embeddings.

---

## Testing against the official encoder

After the extracted encoder passes, the stronger comparison is:

```text
same prompt
same seed
same pipeline
same sampler settings

A: official Mistral encoder
B: extracted replacement encoder
```

This determines whether the replacement is merely executable or is also competitively aligned with FLUX.2.

It is not necessary for the initial usability test.

---

## Important distinction

This test proves:

```text
the extracted transformer is a usable FLUX.2 text encoder
```

It does not prove:

```text
the .safetensors file is already a valid ComfyUI text-encoder package
```

The latter additionally depends on ComfyUI’s:

```text
key namespace
architecture detection
tekken_model handling
dtype and quantization metadata
model-loading implementation
```
