# HF Image Generation

This package is the extendable image-generation workspace for
`black-forest-labs/FLUX.1-Krea-dev`. It is designed to support two execution
paths:

- local Diffusers inference, for full artifact capture including per-step
  latents;
- Hugging Face Inference Provider/FAL fallback, for provider-hosted final-image
  generation when local inference is not available.

No inference has been run for this implementation pass. The commands below are
documented as dry-run checks or guarded real-run commands only.

## Folder Structure

Package files:

```text
hf_image_gen/
  __init__.py
  config.py
  decode_latents.py
  generate_fal_provider.py
  generate_flux_krea.py
  latent_capture.py
  output_paths.py
  README.md
  image/
```

Run output layout:

```text
hf_image_gen/image/
  <run_id>/
    metadata.json
    prompt.txt
    latents/
      step_000.pt
      step_001.pt
      ...
    decoded_latents/
      step_000.png
      step_001.png
      ...
    final image/
      image.png
```

Both modes create the same top-level run directory shape. Local Diffusers mode
can populate `latents/` and `decoded_latents/`; provider mode cannot expose
Diffusers step latents, so those directories may stay empty and metadata should
record `latents_available: false`.

## Modes

### Local Diffusers

`generate_flux_krea.py` is the local runner for
`diffusers.FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev")`.
This path is the only path that can capture per-step latents because it runs the
Diffusers denoising loop directly and can attach a
`callback_on_step_end` callback with `callback_on_step_end_tensor_inputs`.

Use local mode when you need:

- `latents/step_*.pt` tensors;
- decoded latent previews in `decoded_latents/step_*.png`;
- final image output in `final image/image.png`;
- full local control over dtype, device, seed, dimensions, and step count.

Local mode requires the image-generation dependencies and enough local compute
for FLUX.1 Krea. Install the optional dependency group before any real local
run:

```bash
pip install -e ".[image-gen]"
```

### HF/FAL Provider

`generate_fal_provider.py` is the provider fallback. It uses the Hugging Face
Inference Provider interface with the FAL provider (`fal-ai`) when configured
that way. This path is useful when local GPU inference is unavailable.

Use provider mode when final-image output is enough. Provider text-to-image APIs
return the final image and response metadata; they do not expose Diffusers
per-step latents or VAE-decoded latent previews.

## Environment

Set secrets in the environment. Do not commit them.

```bash
export HF_TOKEN="hf_..."
export FAL_KEY="fal_..."
```

`HF_TOKEN` is used for Hugging Face model access and for Hugging Face Inference
Provider calls. The token must belong to an account that can access the model.

`FAL_KEY` is used for direct FAL-backed provider access or fallback behavior.

`black-forest-labs/FLUX.1-Krea-dev` is treated as a gated model for this
workflow. Before any real local or provider inference, open the model page while
logged in to Hugging Face, accept the model terms if prompted, and use an
`HF_TOKEN` from that account:

```text
https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
```

## Dry-Run Commands

Dry runs are for path, metadata, prompt, and CLI validation. They must not load
FLUX weights or call a remote image provider.

Local Diffusers dry run:

```bash
python -m hf_image_gen.generate_flux_krea \
  --dry-run \
  --run-id dry_local
```

FAL provider dry run:

```bash
python -m hf_image_gen.generate_fal_provider \
  --run-id dry_fal \
  --overwrite
```

Repository smoke check for the local dry-run layout:

```bash
python scripts/check_hf_image_gen_dry_run.py
```

## Real-Run Commands

Real inference must be explicitly guarded with `--run-inference`. Do not remove
that flag from real-run commands, and do not run real inference as part of
documentation or CI validation.

Local Diffusers real run:

```bash
export HF_TOKEN="hf_..."

python -m hf_image_gen.generate_flux_krea \
  --run-inference \
  --run-id flux_krea_seed0 \
  --seed 0 \
  --width 1024 \
  --height 1536 \
  --num-inference-steps 50 \
  --guidance-scale 4.5 \
  --save-decoded-latents
```

FAL provider real run:

```bash
export HF_TOKEN="hf_..."
export FAL_KEY="fal_..."

python -m hf_image_gen.generate_fal_provider \
  --run-inference \
  --run-id flux_krea_fal_seed0 \
  --provider fal-ai \
  --seed 0 \
  --width 1024 \
  --height 1536 \
  --steps 50 \
  --guidance-scale 4.5 \
  --overwrite
```

If a checkout exposes a local CLI that can run inference without
`--run-inference`, treat that as an implementation gap and do not use it for a
real run until the guard is in place.

## Expected Artifacts

Local Diffusers metadata should identify:

- model ID and prompt;
- seed, dimensions, guidance scale, and inference steps;
- device, dtype, and package versions;
- `inference_ran: true` only after a guarded real run;
- all saved latent and decoded-preview paths.

Provider metadata should identify:

- model ID, provider, prompt, and generation parameters;
- final image path;
- `latents_available: false`;
- an explanation that provider calls do not return Diffusers step latents.

For this documentation pass, `inference_ran` remains false because no inference
has been executed.
