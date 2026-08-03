# Extracting Mistral Small 3.2 for FLUX.2

## What the tool does

`extract-textencoder` converts a full Hugging Face checkpoint derived from
`Mistral-Small-3.2-24B-Instruct-2506` into a standalone text encoder accepted
by FLUX.2 workflows in ComfyUI.

FLUX.2 consumes hidden states 10, 20, and 30. Because hidden state zero is the
embedding output, only these language-model components are required:

```text
token embedding
transformer layers 0-29
```

The converter removes layers 30-39, final language-model normalization, LM
head, vision tower, and multimodal projector. It embeds the source
`tekken.json` as a top-level uint8 tensor named `tekken_model`.

This is pruning and repackaging, not training. The resulting transformer is
still causal and is used as a frozen hidden-state feature extractor.

## Supported input

The source must be a full, unquantized checkpoint with:

- `Mistral3ForConditionalGeneration` and `model_type: mistral3`;
- the original 5120-wide, 40-layer Mistral Small 3.2 architecture;
- Hugging Face keys under `language_model.model.*`;
- `model.safetensors.index.json`, `config.json`, and `tekken.json`;
- the original compatible tokenizer.

GGUF, GPTQ, AWQ, EXL2, bitsandbytes repositories, LoRA-only adapters, other
Mistral sizes, and unrelated LLM families are rejected. BF16 is required by
default. `--allow-non-bf16` preserves other source dtypes but does not perform
quantization.

## Storage

Expect roughly 35-36 GB for a BF16 output and tens of gigabytes of source
shards. Use a filesystem with at least 85-100 GB free when the Hugging Face
cache and output share a disk. Conversion is CPU and disk intensive; it does
not require a GPU.

## Set up with uv

```bash
uv sync
```

This repository intentionally has no generated `uv.lock` because its initial
implementation was prepared without executing package-management commands.
Generate one on the conversion machine when reproducible dependency resolution
is required:

```bash
uv lock
```

## Convert a Hugging Face repository

```bash
uv run extract-textencoder convert \
  --repo-id huihui-ai/Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated \
  --revision FULL_COMMIT_HASH \
  --cache-dir /mnt/models/huggingface \
  --output /mnt/models/mistral_3_small_flux2_huihui_bf16.safetensors
```

Use a commit hash for reproducible conversion. Authentication follows normal
Hugging Face behavior, including `HF_TOKEN` and `hf auth login`.

Only shards containing the embedding and layers 0-29 are downloaded.

## Convert a local snapshot

```bash
uv run extract-textencoder convert \
  --local-dir /mnt/models/mistral-small-snapshot \
  --output /mnt/models/mistral_3_small_flux2_bf16.safetensors
```

The local directory must contain every file referenced by its checkpoint
index. Paths in the index are constrained to that directory.

## Safety and validation

The converter validates the source configuration and the exact 271 selected
model tensors before writing. It then adds `tekken_model`, producing 272 total
tensors. It checks every retained tensor shape and source dtype.

Output is written beside the destination using a unique `.incomplete` name,
reopened and structurally inspected, then installed atomically. Existing files
are not replaced unless `--overwrite` is supplied. A failed conversion removes
only its own incomplete temporary file.

Inspect a completed file without running its transformer:

```bash
uv run extract-textencoder inspect \
  /mnt/models/mistral_3_small_flux2_bf16.safetensors
```

## ComfyUI installation

Place the validated file in:

```text
ComfyUI/models/text_encoders/
```

Use it in a FLUX.2 Load CLIP/text-encoder node. Standalone output keys begin
with `model.embed_tokens` and `model.layers`; they must not retain the source
`language_model` prefix or the nested all-in-one checkpoint prefix.

The output architecture is:

```text
prompt -> embedded Tekken tokenizer -> embedding -> layers 0-29
       -> hidden states 10, 20, 30 -> 15360-wide FLUX.2 conditioning
```

Configure `test_extracted.toml` and run `test_extracted_flux2_encoder.py` on the
GPU machine before judging semantic quality; `test_extracted.md` describes the
acceptance criteria. A structurally valid fine-tuned or abliterated model may
still align worse with the diffusion transformer than the official encoder.
