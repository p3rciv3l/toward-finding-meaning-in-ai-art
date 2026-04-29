#!/usr/bin/env python3
"""Decode saved FLUX packed latents into PNG images with the model VAE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_HEIGHT = 1536
DEFAULT_WIDTH = 1024
DEFAULT_VAE_SCALE_FACTOR = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-dir", type=Path, default=Path("flux_unzipped/latents"))
    parser.add_argument("--output-dir", type=Path, default=Path("flux_unzipped/vae_decoded_latents"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--vae-scale-factor", type=int, default=DEFAULT_VAE_SCALE_FACTOR)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--step", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    latents = select_latents(args)
    if not latents:
        raise SystemExit(f"No latent tensors matched under {args.latents_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    records: list[dict[str, Any]] = manifest.setdefault("records", [])

    device = resolve_device(torch, args.device)
    dtype = resolve_torch_dtype(torch, args.torch_dtype)
    print(f"decoding {len(latents)} latent tensor(s) on {device}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    vae.to(device)
    vae.eval()
    processor = VaeImageProcessor(vae_scale_factor=args.vae_scale_factor)

    pending = []
    for latent_path in latents:
        output_path = args.output_dir / f"step_{step_from_path(latent_path):03d}.png"
        if output_path.exists() and not args.overwrite:
            print(f"{latent_path.name} skipped; image exists")
        else:
            pending.append(latent_path)

    started = time.time()
    for batch_index, batch_paths in enumerate(chunks(pending, args.batch_size), start=1):
        batch_started = time.time()
        batch_steps = [step_from_path(path) for path in batch_paths]
        try:
            packed = torch.cat(
                [torch.load(path, map_location="cpu").to(dtype=dtype) for path in batch_paths],
                dim=0,
            )
            unpacked = FluxPipeline._unpack_latents(
                packed,
                args.height,
                args.width,
                args.vae_scale_factor,
            )
            prepared = (unpacked / vae.config.scaling_factor) + vae.config.shift_factor
            prepared = prepared.to(device=device, dtype=dtype)
            with torch.inference_mode():
                decoded = vae.decode(prepared, return_dict=False)[0]
            images = processor.postprocess(decoded.detach().cpu(), output_type="pil")
            batch_elapsed = round(time.time() - batch_started, 3)
            for latent_path, step, image in zip(batch_paths, batch_steps, images, strict=True):
                output_path = args.output_dir / f"step_{step:03d}.png"
                image.save(output_path, format="PNG")
                record = {
                    "step": step,
                    "latent_path": str(latent_path),
                    "image_path": str(output_path),
                    "mode": "vae",
                    "width": image.width,
                    "height": image.height,
                    "duration_seconds": batch_elapsed,
                    "batch_size": len(batch_paths),
                }
                upsert_record(records, record)
            print(
                f"[batch {batch_index:02d}] steps {batch_steps[0]:03d}-{batch_steps[-1]:03d} "
                f"-> {len(batch_paths)} image(s) ({batch_elapsed}s)"
            )
        except Exception as exc:  # noqa: BLE001 - record and continue.
            batch_elapsed = round(time.time() - batch_started, 3)
            for latent_path, step in zip(batch_paths, batch_steps, strict=True):
                output_path = args.output_dir / f"step_{step:03d}.png"
                record = {
                    "step": step,
                    "latent_path": str(latent_path),
                    "image_path": str(output_path),
                    "mode": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_seconds": batch_elapsed,
                    "batch_size": len(batch_paths),
                }
                upsert_record(records, record)
            print(f"[batch {batch_index:02d}] steps {batch_steps} failed: {type(exc).__name__}: {exc}")

        manifest.update(
            {
                "source_latents_dir": str(args.latents_dir),
                "output_dir": str(args.output_dir),
                "model_id": args.model_id,
                "width": args.width,
                "height": args.height,
                "vae_scale_factor": args.vae_scale_factor,
                "device": device,
                "dtype": str(dtype),
                "duration_seconds": round(time.time() - started, 3),
            }
        )
        write_manifest(manifest_path, manifest)
        empty_device_cache(torch, device)

    return 0


def chunks(paths: list[Path], batch_size: int) -> list[list[Path]]:
    return [paths[index : index + batch_size] for index in range(0, len(paths), batch_size)]


def select_latents(args: argparse.Namespace) -> list[Path]:
    candidates = sorted(args.latents_dir.glob("step_*.pt"))
    requested = set(args.step or [])
    selected = []
    for path in candidates:
        step = step_from_path(path)
        if requested and step not in requested:
            continue
        if args.start_step is not None and step < args.start_step:
            continue
        if args.end_step is not None and step > args.end_step:
            continue
        selected.append(path)
    return selected


def step_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(torch: Any, requested: str) -> Any:
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    return torch.float32


def empty_device_cache(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def upsert_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    for index, existing in enumerate(records):
        if existing.get("step") == record.get("step"):
            records[index] = record
            return
    records.append(record)
    records.sort(key=lambda item: int(item.get("step", -1)))


if __name__ == "__main__":
    raise SystemExit(main())
