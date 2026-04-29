# Toward Finding Meaning in AI Art

This workspace contains small provider harnesses plus a FLUX.1 Krea
image-generation experiment setup for capturing and decoding denoising-step
latents.

This repository exposes compact client entrypoints for both OpenRouter and
Gemini. Model aliases live in `prod_env/model_deployments.yaml`; each harness
resolves an alias first and falls back to a raw provider model ID when no alias
matches.

```python
from clients import get_model_client, get_gemini_client

openrouter = get_model_client("gpt-5.1")  # YAML alias or raw OpenRouter model ID
reply = openrouter.generate(
    [{"role": "user", "content": "Write a one-line test response."}]
)
print(reply["choices"][0]["message"]["content"])

gemini = get_gemini_client("gemini-2.5-flash-image")  # YAML alias or raw Gemini model ID
result = gemini.generate(
    contents=[{"role": "user", "parts": [{"text": "Describe this image briefly."}]}]
)
print(result["candidates"][0]["content"]["parts"][0]["text"])
```

## HF Image Generation

The `hf_image_gen/` package contains the FLUX.1 Krea image-generation workspace
for `black-forest-labs/FLUX.1-Krea-dev`. See `hf_image_gen/README.md` for the
local Diffusers mode, HF/FAL provider fallback mode, exact run-output folder
structure, required `HF_TOKEN`/`FAL_KEY` environment variables, gated model
terms, and dry-run versus guarded `--run-inference` commands.

`scripts/decode_flux_latents.py` decodes saved FLUX packed latent tensors back
into PNGs through the model VAE. It supports selecting individual steps or
ranges, CPU/CUDA/MPS devices, and resumable output folders.

## Repository Policy

The git repo tracks source, tests, notebooks, and reproducibility scripts. Large
generated outputs are intentionally ignored:

- `flux_unzipped/` for raw latent tensors and decoded PNGs;
- `flux_latents_only.zip` for the raw-latent archive;
- `flux_vae_decoded_latents_vast.zip` for the decoded-image archive;
- `artifacts/` for provider scan and experiment outputs;
- `.env` and local virtual environments.

Keep those outputs locally or move them to external storage. If they need to be
versioned later, use Git LFS rather than normal git blobs.
