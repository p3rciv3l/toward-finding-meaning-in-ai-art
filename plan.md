# FLUX.1 Krea Dev Inference Plan

## Goal

Run one 50-step generation on `black-forest-labs/FLUX.1-Krea-dev`, save the final image, save per-step latents, and save decoded latent previews for each step.

## Source Notes

- Hugging Face model card: `black-forest-labs/FLUX.1-Krea-dev` is tagged for `FluxPipeline`, supports Diffusers, and can be used as a drop-in replacement for `FLUX.1 [dev]`.
- Hugging Face Diffusers FLUX docs: guidance-distilled FLUX models use `FluxPipeline`; 50 sampling steps are the quality-oriented baseline for dev-style FLUX models.
- Hugging Face Diffusers callback docs: per-step tensors are captured with `callback_on_step_end` and `callback_on_step_end_tensor_inputs=["latents"]`.
- Hugging Face Inference Providers / FAL docs: `InferenceClient(provider="fal-ai")` can run text-to-image and returns a final `PIL.Image` object. Provider APIs do not expose Diffusers internal per-step latents through that client path.

## Important Implementation Constraint

Saving latents requires access to the running Diffusers pipeline internals. A normal Hugging Face Inference Provider call through FAL returns the final image only, so it cannot by itself produce per-step latents or decoded latent previews.

Implementation should therefore use one of these paths:

- Preferred for this project: run a Python generation script under `hf_image_gen/` with `diffusers.FluxPipeline` and Diffusers callbacks. Use FAL only if the execution environment is a custom FAL job/endpoint that runs this Python code and returns the saved artifacts.
- Fallback for FAL-hosted provider only: use `huggingface_hub.InferenceClient(provider="fal-ai")` for the final image and mark latent capture as unavailable for that execution mode.

## Prompt

```text
hyperreal close portrait of a woman meeting the viewer's gaze head-on, eyes steady and lucid beneath a helmet cropped tight at the frame's edge. her face carries a dusting of white frost along the lashes and temples, lips slightly parted as if mid-breath, fine moisture glinting where the cold meets warmth. the light comes from behind a low golden sun cutting through icy air wrapping her in a soft halo while highlights cling delicately to the frost edges. her skin reveals honest micro detail: faint pores, subsurface glow on the cheeks, and a natural sheen from the chill. the palette moves between warm amber and arctic blue. every surface behaves realistically matte skin diffusing light, frost refracting it, the atmosphere crisp yet breathable. the mood is tender intensity, a quiet warmth radiating through the ice. captured on an 85mm lens at f/2.0, focus locked to her eyes, shallow depth isolating her face in luminous realism.
--v 7 --ar 2:3 --raw --profile
```

## Target Layout

```text
hf_image_gen/
  generate_flux_krea.py
  config.py
  README.md
  image/
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

Use a timestamped or seed-based `<run_id>` so repeated runs do not overwrite prior outputs.

## Generation Settings

- Model: `black-forest-labs/FLUX.1-Krea-dev`
- Pipeline: `diffusers.FluxPipeline`
- Inference steps: `50`
- Image count: `1`
- Aspect ratio: `2:3`
- Initial target size: `1024x1536`, unless FAL or GPU limits require a lower 2:3 size.
- Guidance scale: start with `4.5` from the Krea model card example; keep configurable.
- Dtype: `torch.bfloat16` when GPU supports it.
- Device/offload: start with `pipe.enable_model_cpu_offload()`; allow full CUDA placement when enough VRAM is available.
- Seed: explicit configurable seed for reproducibility; write it to `metadata.json`.

## Live Checklist

Status updated by Worker 10 on 2026-04-23 after inspecting the implemented files
and running dry-run/unit verification only. No real local Diffusers inference,
HF Inference Provider request, or FAL request has been run by Worker 10.

### Implemented

- [x] Create root-level `hf_image_gen/` package.
- [x] Implement `hf_image_gen/image/` as the default runtime output root.
- [x] Add generation config for model ID, prompt, steps, aspect ratio,
  dimensions, guidance scale, seed, dtype, execution mode, provider settings,
  save flags, and output paths in `hf_image_gen/config.py`.
- [x] Add local Diffusers runner in `hf_image_gen/generate_flux_krea.py`.
- [x] Guard local inference behind `--run-inference`; dry-run is the default and
  does not load FLUX weights.
- [x] Use `FluxPipeline.from_pretrained(...)` for
  `black-forest-labs/FLUX.1-Krea-dev` in the guarded local inference path.
- [x] Support configurable torch dtype, including `bfloat16` and auto dtype
  selection.
- [x] Document gated-model access and required Hugging Face token setup in
  `hf_image_gen/README.md`.
- [x] Implement run directory creation under the configured image output root.
- [x] Create per-run folders: `latents/`, `decoded_latents/`, and
  `final image/`.
- [x] Implement Diffusers callback helpers that read
  `callback_kwargs["latents"]`.
- [x] Pass `callback_on_step_end` and
  `callback_on_step_end_tensor_inputs=["latents"]` to the local pipeline call.
- [x] Save callback latent tensors as `latents/step_<step>.pt`.
- [x] Implement latent preview/decoded-preview saving as
  `decoded_latents/step_<step>.png`, with VAE decode when available and a
  normalized preview fallback.
- [x] Save the final local Diffusers image as `final image/image.png` after a
  guarded real local run.
- [x] Save `prompt.txt` with the resolved prompt for each run.
- [x] Save `metadata.json` with model, seed, dimensions, step count, guidance
  scale, runtime, package versions, execution mode, output paths, and
  inference flags.
- [x] Add `hf_image_gen/README.md` explaining local Diffusers mode versus
  HF/FAL provider fallback mode.
- [x] Add optional dependency notes for `diffusers`, `transformers`,
  `accelerate`, `torch`, `sentencepiece`, `protobuf`, `huggingface_hub`, and
  `Pillow` in `pyproject.toml` and README documentation.
- [x] Add HF/FAL provider fallback runner in
  `hf_image_gen/generate_fal_provider.py`.
- [x] Ensure provider fallback records `latents_available: false` because
  hosted provider text-to-image calls do not expose Diffusers per-step latents.
- [x] Add tests for config, output paths, local dry-run behavior, provider
  dry-run behavior, latent capture, and latent previews.

### Dry-Run Verification

- [x] Verify the local Diffusers runner can run a dry path that creates run
  directories, `metadata.json`, and `prompt.txt` without loading the model.
- [x] Verify the HF/FAL provider runner can run a dry path that creates run
  directories, `metadata.json`, and `prompt.txt` without making a network or
  provider inference call.
- [x] Run repository smoke check:
  `python scripts/check_hf_image_gen_dry_run.py`.
- [x] Run package unit tests:
  `python -m unittest discover -s tests`.

### Real Inference Still Pending

- [ ] Run one real local generation with `num_inference_steps=50`.
- [ ] Confirm exactly one final image exists for the real run.
- [ ] Confirm latents were saved for each denoising callback step in the real
  run.
- [ ] Confirm decoded latent previews were saved for each denoising callback
  step in the real run.
- [ ] Confirm `metadata.json` and `prompt.txt` match the actual real run.
- [ ] Optionally run a real HF/FAL provider fallback request and confirm it
  saves a final image while keeping latent capture marked unavailable.

## References

- https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
- https://huggingface.co/docs/diffusers/en/api/pipelines/flux
- https://huggingface.co/docs/diffusers/en/using-diffusers/callback
- https://huggingface.co/docs/inference-providers/main/providers/fal-ai
- https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
