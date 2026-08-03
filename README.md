# extract-textencoder

`extract-textencoder` extracts the embedding and first 30 transformer layers
from a compatible Hugging Face Mistral Small 3.2 24B checkpoint and packages
them as a standalone FLUX.2 text encoder for ComfyUI.

This is intentionally not a universal LLM converter. The first release only
supports full checkpoints derived from
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` that retain its architecture,
tokenizer, and Hugging Face tensor layout.

## Environment

Use [`uv`](https://docs.astral.sh/uv/) for environment and package management:

```bash
uv sync
uv run extract-textencoder --help
```

No lockfile is committed yet because this repository was prepared under a
static-only constraint. Run `uv lock` on the machine that will perform the
conversion if a reproducible environment is required.

## Convert from Hugging Face

```bash
uv run extract-textencoder convert \
  --repo-id huihui-ai/Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated \
  --revision COMMIT_HASH \
  --output mistral_3_small_flux2_huihui_bf16.safetensors
```

The normal Hugging Face credential sources are honored, including `HF_TOKEN`
and credentials saved by `hf auth login`.

## Convert a local snapshot

The directory must contain `config.json`, `tekken.json`,
`model.safetensors.index.json`, and all referenced shard files.

```bash
uv run extract-textencoder convert \
  --local-dir /mnt/models/mistral-small-snapshot \
  --output /mnt/models/mistral_3_small_flux2_bf16.safetensors
```

Existing output files are protected by default. Pass `--overwrite` only when
replacement is intended. BF16 is required by default; `--allow-non-bf16`
preserves floating-point source dtypes and does not quantize them.

## Inspect an output

```bash
uv run extract-textencoder inspect mistral_3_small_flux2_bf16.safetensors
```

The inspector checks tensor names, shapes, dtypes, layer coverage, tokenizer
packaging, and forbidden weights without loading or running the transformer.

For design details and the remote GPU validation walkthrough, see
[`extract.md`](extract.md) and [`test_extracted.md`](test_extracted.md).

