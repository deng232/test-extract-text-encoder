# End-to-End Test of an Extracted FLUX.2 Text Encoder

## Purpose

This test determines whether a structurally valid extracted checkpoint also
produces meaningful conditioning that FLUX.2 can consume.

```text
prompt
  -> Mistral processor
  -> extracted embedding and layers 0-29
  -> hidden states 10, 20, and 30
  -> [batch, 512, 15360] prompt embeddings
  -> FLUX.2 diffusion pipeline
  -> generated image
```

The structural `extract-textencoder inspect` command proves that keys, tensor
shapes, dtypes, and tokenizer packaging are correct. This GPU test additionally
checks runtime loading, prompt encoding, and image conditioning.

## Hardware

The extracted BF16 encoder is about 32-36 GiB. The test loads it, creates prompt
embeddings, unloads it, and only then loads the quantized diffusion pipeline.

```text
GPU VRAM: 48 GB minimum; 80 GB is safer
CPU RAM:  64 GB or more
Disk:     approximately 80 GB free, depending on cached models
```

## Configure the test

Edit `test_extracted.toml`. Paths are resolved relative to that TOML file.

The default model section points to the checkpoint created by this repository:

```toml
[model]
checkpoint = "mistral_3_small_flux2_huihui_bf16.safetensors"
pipeline_repo = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
dtype = "bfloat16"
```

The generation section controls output location, seed, resolution, and sampling:

```toml
[generation]
output_directory = "flux2_encoder_test"
seed = 42
max_sequence_length = 512
hidden_state_layers = [10, 20, 30]
height = 768
width = 768
num_inference_steps = 28
guidance_scale = 4.0
overwrite_outputs = false
```

Each `[[prompts]]` entry requires a unique filesystem-safe name and distinct
prompt text. Its name becomes the PNG filename. The committed configuration
uses a red cube and a blue sphere so prompt adherence is easy to judge.

Existing output images are protected by default. Set
`overwrite_outputs = true` only when replacing an earlier run is intended.

## Run

```bash
python test_extracted_flux2_encoder.py
```

To use another configuration:

```bash
python test_extracted_flux2_encoder.py --config /path/to/test.toml
```

## Automated checks

The script reads `tekken_model` directly from the converted checkpoint and uses
it through Mistral's text-only tokenizer. It does not download the original LLM
repository or a separate Hugging Face processor.

Before image generation, the script requires:

1. A structurally valid 272-tensor extracted checkpoint.
2. CUDA and the configured CUDA device.
3. A compatible Mistral Small 3.2 text configuration.
4. Prompt embeddings shaped `(prompt count, 512, 15360)`.
5. No NaN or infinity in the embeddings.
6. Different embeddings for the distinct prompts.

It then moves the embeddings to CPU, unloads the large text encoder, clears its
CUDA allocations, and loads the FLUX.2 pipeline with `text_encoder=None`.
Every image uses a newly initialized generator with the same seed, so prompt
conditioning is the intended source of the visible difference.

With the default configuration, successful execution creates:

```text
flux2_encoder_test/
├── red_cube.png
└── blue_sphere.png
```

## How to judge the result

A definite pass requires both coherent images and recognizable prompt
adherence: the first should depict a red cube and the second a blue sphere.

If embeddings pass but the images are coherent with weak adherence, the
encoder is operational but its modified representations may align poorly with
the FLUX.2 diffusion transformer. If both images are nearly identical, inspect
the hidden-state selection and conditioning path. Noise, corruption, or tensor
shape failures indicate an incompatible source, incorrect model loading, or an
incompatible pipeline version.

This test validates Diffusers consumption of the extracted transformer. It does
not independently test ComfyUI architecture detection or model offloading.
