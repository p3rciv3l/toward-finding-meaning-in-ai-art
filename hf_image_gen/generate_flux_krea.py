#!/usr/bin/env python3
"""Local Diffusers entrypoint for FLUX.1-Krea-dev image generation.

The dry-run path intentionally avoids importing Torch or Diffusers so it can be
used to validate paths and metadata on machines that cannot run the model.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_PROMPT = """hyperreal close portrait of a woman meeting the viewer's gaze head-on, eyes steady and lucid beneath a helmet cropped tight at the frame's edge. her face carries a dusting of white frost along the lashes and temples, lips slightly parted as if mid-breath, fine moisture glinting where the cold meets warmth. the light comes from behind a low golden sun cutting through icy air wrapping her in a soft halo while highlights cling delicately to the frost edges. her skin reveals honest micro detail: faint pores, subsurface glow on the cheeks, and a natural sheen from the chill. the palette moves between warm amber and arctic blue. every surface behaves realistically matte skin diffusing light, frost refracting it, the atmosphere crisp yet breathable. the mood is tender intensity, a quiet warmth radiating through the ice. captured on an 85mm lens at f/2.0, focus locked to her eyes, shallow depth isolating her face in luminous realism.
--v 7 --ar 2:3 --raw --profile"""
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "image"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1536
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_IMAGES = 1
DEFAULT_MAX_SEQUENCE_LENGTH = 512
_OPTIONAL_IMPORT_ERRORS: list[dict[str, str]] = []


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if name == exc.name or name.startswith(f"{exc.name}."):
            return None
        raise
    except Exception as exc:
        _OPTIONAL_IMPORT_ERRORS.append(
            {
                "module": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None


def _first_optional_module(names: Iterable[str]) -> Any | None:
    for name in names:
        module = _optional_import(name)
        if module is not None:
            return module
    return None


_config = _first_optional_module(("hf_image_gen.config", "config"))
_output_paths_module = _first_optional_module(("hf_image_gen.output_paths",))
_latent_capture_module = _first_optional_module(("hf_image_gen.latent_capture",))
_decode_latents_module = _first_optional_module(("hf_image_gen.decode_latents",))


def _config_value(*names: str, default: Any) -> Any:
    if _config is None:
        return default
    for name in names:
        if hasattr(_config, name):
            return getattr(_config, name)
    return default


MODEL_ID = _config_value("MODEL_ID", "DEFAULT_MODEL_ID", "FLUX_KREA_MODEL_ID", default=MODEL_ID)
DEFAULT_PROMPT = _config_value("PROMPT", "DEFAULT_PROMPT", "FLUX_KREA_PROMPT", default=DEFAULT_PROMPT)
DEFAULT_OUTPUT_ROOT = Path(
    _config_value(
        "OUTPUT_ROOT",
        "DEFAULT_OUTPUT_ROOT",
        "IMAGE_OUTPUT_ROOT",
        default=DEFAULT_OUTPUT_ROOT,
    )
)
DEFAULT_WIDTH = int(_config_value("WIDTH", "DEFAULT_WIDTH", default=DEFAULT_WIDTH))
DEFAULT_HEIGHT = int(_config_value("HEIGHT", "DEFAULT_HEIGHT", default=DEFAULT_HEIGHT))
DEFAULT_NUM_INFERENCE_STEPS = int(
    _config_value(
        "NUM_INFERENCE_STEPS",
        "DEFAULT_NUM_INFERENCE_STEPS",
        "DEFAULT_STEPS",
        default=DEFAULT_NUM_INFERENCE_STEPS,
    )
)
DEFAULT_GUIDANCE_SCALE = float(
    _config_value("GUIDANCE_SCALE", "DEFAULT_GUIDANCE_SCALE", default=DEFAULT_GUIDANCE_SCALE)
)
DEFAULT_NUM_IMAGES = int(
    _config_value("NUM_IMAGES", "DEFAULT_NUM_IMAGES", default=DEFAULT_NUM_IMAGES)
)
DEFAULT_MAX_SEQUENCE_LENGTH = _config_value(
    "MAX_SEQUENCE_LENGTH",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    default=DEFAULT_MAX_SEQUENCE_LENGTH,
)
if DEFAULT_MAX_SEQUENCE_LENGTH is not None:
    DEFAULT_MAX_SEQUENCE_LENGTH = int(DEFAULT_MAX_SEQUENCE_LENGTH)
DEFAULT_SEED = _config_value("SEED", "DEFAULT_SEED", default=None)
if DEFAULT_SEED is not None:
    DEFAULT_SEED = int(DEFAULT_SEED)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    latents_dir: Path
    decoded_latents_dir: Path
    final_image_dir: Path
    metadata_path: Path
    prompt_path: Path
    final_image_path: Path

    @property
    def run_root(self) -> Path:
        return self.run_dir

    def latent_path(self, step: int) -> Path:
        return self.latents_dir / f"step_{int(step):03d}.pt"

    def decoded_latent_path(self, step: int) -> Path:
        return self.decoded_latents_dir / f"step_{int(step):03d}.png"


class LatentSaver:
    def __init__(
        self,
        *,
        paths: RunPaths,
        torch_module: Any,
        save_decoded_latents: bool,
        height: int,
        width: int,
    ) -> None:
        self.paths = paths
        self.torch = torch_module
        self.save_decoded_latents = save_decoded_latents
        self.height = height
        self.width = width
        self.saved_steps: list[dict[str, Any]] = []

    def __call__(self, pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        filename = f"step_{int(step):03d}.pt"
        latent_path = self.paths.latents_dir / filename
        self.torch.save(_detach_to_cpu(latents), latent_path)

        record: dict[str, Any] = {
            "step": int(step),
            "timestep": _json_safe(timestep),
            "latents_path": str(latent_path),
        }
        if self.save_decoded_latents:
            preview_path = self.paths.decoded_latents_dir / f"step_{int(step):03d}.png"
            if _save_decoded_preview(pipe, latents, preview_path, self.height, self.width):
                record["decoded_latents_path"] = str(preview_path)

        self.saved_steps.append(record)
        return callback_kwargs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one FLUX.1-Krea-dev image locally with Diffusers.",
    )
    parser.add_argument("--prompt", help="Prompt text. Defaults to the configured Krea portrait prompt.")
    parser.add_argument("--prompt-file", type=Path, help="Read prompt text from this file.")
    parser.add_argument("--model-id", default=MODEL_ID, help=f"Diffusers model ID. Default: {MODEL_ID}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for run outputs.")
    parser.add_argument("--run-id", help="Run directory name. Defaults to a timestamp and seed.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed. Defaults to config value or a random seed.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Output width. Default: {DEFAULT_WIDTH}")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Output height. Default: {DEFAULT_HEIGHT}")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Denoising steps. Default: {DEFAULT_NUM_INFERENCE_STEPS}",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=f"Guidance scale. Default: {DEFAULT_GUIDANCE_SCALE}",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Images per prompt. Default: {DEFAULT_NUM_IMAGES}",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help=f"FLUX text sequence length. Default: {DEFAULT_MAX_SEQUENCE_LENGTH}",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Torch dtype for pipeline weights. Default: auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--cpu-offload",
        dest="cpu_offload",
        action="store_true",
        default=True,
        help="Use Diffusers model CPU offload when running on CUDA. Default: enabled",
    )
    parser.add_argument(
        "--no-cpu-offload",
        dest="cpu_offload",
        action="store_false",
        help="Place the pipeline directly on the selected device.",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached model files.",
    )
    parser.add_argument(
        "--save-decoded-latents",
        action="store_true",
        default=True,
        help="Try to decode and save a PNG preview for each captured latent step. Default: enabled.",
    )
    parser.add_argument(
        "--no-save-decoded-latents",
        dest="save_decoded_latents",
        action="store_false",
        help="Disable decoded latent preview PNGs while still saving raw latent tensors.",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Actually load the model and run generation. Without this flag, only a dry run is performed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create run directories and metadata without importing Diffusers, loading the model, or running inference.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an existing explicit --run-id directory.")
    args = parser.parse_args(argv)

    if args.prompt and args.prompt_file:
        parser.error("Use either --prompt or --prompt-file, not both.")
    if args.run_inference and args.dry_run:
        parser.error("Use either --run-inference or --dry-run, not both.")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive integers.")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be a positive integer.")
    if args.num_images <= 0:
        parser.error("--num-images must be a positive integer.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dry_run = not args.run_inference
    prompt = _resolve_prompt(args)
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**32 - 1)
    paths = _prepare_run_paths(args.output_root, args.run_id, seed, overwrite=args.overwrite)
    started_at = time.time()

    metadata = _base_metadata(args, prompt, seed, paths, started_at)
    _write_run_files(paths, prompt, metadata)

    if args.dry_run:
        metadata.update(
            {
                "status": "dry_run_complete",
                "duration_seconds": round(time.time() - started_at, 3),
                "model_loaded": False,
                "inference_ran": False,
            }
        )
        _write_metadata(paths.metadata_path, metadata)
        print(json.dumps({"run_dir": str(paths.run_dir), "metadata": str(paths.metadata_path)}, indent=2))
        return 0

    latent_saver: Any | None = None
    try:
        result, latent_saver = _run_pipeline(args, prompt, seed, paths)
        image = _first_image(result)
        image.save(paths.final_image_path)
        metadata.update(
            {
                "status": "complete",
                "duration_seconds": round(time.time() - started_at, 3),
                "model_loaded": True,
                "inference_ran": True,
                "final_image_path": str(paths.final_image_path),
                "latents_saved": _latent_records(latent_saver, paths),
            }
        )
        _write_metadata(paths.metadata_path, metadata)
        print(json.dumps({"run_dir": str(paths.run_dir), "image": str(paths.final_image_path)}, indent=2))
        return 0
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "duration_seconds": round(time.time() - started_at, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "latents_saved": _latent_records(latent_saver, paths),
            }
        )
        _write_metadata(paths.metadata_path, metadata)
        raise


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return str(DEFAULT_PROMPT).strip()


def _prepare_run_paths(output_root: Path, run_id: str | None, seed: int, *, overwrite: bool) -> RunPaths:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    actual_run_id = run_id or _default_run_id(seed)
    run_dir = output_root / actual_run_id
    if run_dir.exists() and run_id and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if run_dir.exists() and not overwrite:
        run_dir = _dedupe_run_dir(output_root, actual_run_id)

    latents_dir = run_dir / "latents"
    decoded_latents_dir = run_dir / "decoded_latents"
    final_image_dir = run_dir / "final image"
    for path in (latents_dir, decoded_latents_dir, final_image_dir):
        path.mkdir(parents=True, exist_ok=overwrite)

    return RunPaths(
        run_dir=run_dir,
        latents_dir=latents_dir,
        decoded_latents_dir=decoded_latents_dir,
        final_image_dir=final_image_dir,
        metadata_path=run_dir / "metadata.json",
        prompt_path=run_dir / "prompt.txt",
        final_image_path=final_image_dir / "image.png",
    )


def _default_run_id(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"flux-krea-{stamp}-seed-{seed}"


def _dedupe_run_dir(output_root: Path, run_id: str) -> Path:
    for index in range(1, 1000):
        candidate = output_root / f"{run_id}-{index:02d}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not create a unique run directory under {output_root}")


def _write_run_files(paths: RunPaths, prompt: str, metadata: dict[str, Any]) -> None:
    paths.prompt_path.write_text(prompt + "\n", encoding="utf-8")
    _write_metadata(paths.metadata_path, metadata)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_metadata(
    args: argparse.Namespace,
    prompt: str,
    seed: int,
    paths: RunPaths,
    started_at: float,
) -> dict[str, Any]:
    metadata = {
        "status": "initialized",
        "created_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "execution_mode": "dry_run" if args.dry_run else "local_diffusers",
        "model_loaded": False,
        "inference_ran": False,
        "model_id": args.model_id,
        "prompt": prompt,
        "seed": seed,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_images": args.num_images,
        "max_sequence_length": args.max_sequence_length,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "local_files_only": args.local_files_only,
        "save_decoded_latents": args.save_decoded_latents,
        "output": {
            "run_dir": str(paths.run_dir),
            "prompt_path": str(paths.prompt_path),
            "metadata_path": str(paths.metadata_path),
            "latents_dir": str(paths.latents_dir),
            "decoded_latents_dir": str(paths.decoded_latents_dir),
            "final_image_dir": str(paths.final_image_dir),
            "final_image_path": str(paths.final_image_path),
        },
        "versions": _package_versions(
            "python",
            "diffusers",
            "torch",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "Pillow",
        ),
    }
    metadata["optional_helpers"] = {
        "config": _config is not None,
        "output_paths": _output_paths_module is not None,
        "latent_capture": _latent_capture_module is not None,
        "decode_latents": _decode_latents_module is not None,
    }
    if _OPTIONAL_IMPORT_ERRORS:
        metadata["optional_module_errors"] = list(_OPTIONAL_IMPORT_ERRORS)
    return metadata


def _run_pipeline(args: argparse.Namespace, prompt: str, seed: int, paths: RunPaths) -> tuple[Any, Any]:
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError as exc:
        raise ImportError(
            "Local FLUX generation requires diffusers and torch. Install the Diffusers "
            "runtime dependencies before running without --dry-run."
        ) from exc

    dtype = _resolve_torch_dtype(torch, args.torch_dtype)
    target_device = _resolve_device(torch, args.device)
    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    from_pretrained_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if token:
        from_pretrained_kwargs["token"] = token

    pipe = FluxPipeline.from_pretrained(args.model_id, **from_pretrained_kwargs)
    if target_device == "cuda" and args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(target_device)

    latent_saver = _make_latent_saver(args, paths, torch)
    generator = _make_generator(torch, target_device, seed, use_cpu=args.cpu_offload)
    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": args.num_images,
        "generator": generator,
        "callback_on_step_end": latent_saver,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if args.max_sequence_length is not None:
        call_kwargs["max_sequence_length"] = args.max_sequence_length

    return pipe(**call_kwargs), latent_saver


def _make_latent_saver(args: argparse.Namespace, paths: RunPaths, torch: Any) -> Any:
    helper_class = getattr(_latent_capture_module, "LatentCapture", None)
    if helper_class is not None:
        try:
            return helper_class(
                paths=paths,
                save_latents=True,
                save_decoded=args.save_decoded_latents,
                height=args.height,
                width=args.width,
            )
        except TypeError:
            pass
    return LatentSaver(
        paths=paths,
        torch_module=torch,
        save_decoded_latents=args.save_decoded_latents,
        height=args.height,
        width=args.width,
    )


def _latent_records(latent_saver: Any, paths: RunPaths) -> list[dict[str, Any]]:
    if latent_saver is None:
        return []

    saved_steps = getattr(latent_saver, "saved_steps", None)
    if saved_steps is not None:
        return list(saved_steps)

    latent_steps = list(getattr(latent_saver, "saved_latent_steps", []) or [])
    preview_steps = set(getattr(latent_saver, "saved_preview_steps", []) or [])
    preview_methods = dict(getattr(latent_saver, "preview_methods", {}) or {})
    records: list[dict[str, Any]] = []
    for step in latent_steps:
        record = {
            "step": int(step),
            "latents_path": str(paths.latent_path(int(step))),
        }
        if step in preview_steps:
            record["decoded_latents_path"] = str(paths.decoded_latent_path(int(step)))
            if step in preview_methods:
                record["decoded_latents_method"] = preview_methods[step]
        records.append(record)
    return records


def _resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if _mps_is_available(torch):
        return torch.float16
    return torch.float32


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if _mps_is_available(torch):
        return "mps"
    return "cpu"


def _mps_is_available(torch: Any) -> bool:
    return bool(
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _make_generator(torch: Any, target_device: str, seed: int, *, use_cpu: bool) -> Any:
    generator_device = "cpu" if use_cpu or target_device == "cpu" else target_device
    try:
        return torch.Generator(device=generator_device).manual_seed(seed)
    except (RuntimeError, ValueError):
        return torch.Generator().manual_seed(seed)


def _first_image(result: Any) -> Any:
    if hasattr(result, "images") and result.images:
        return result.images[0]
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, (list, tuple)) and first:
            return first[0]
        return first
    raise RuntimeError("Pipeline did not return an image.")


def _detach_to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    return value


def _try_save_decoded_latents(
    pipe: Any,
    latents: Any,
    output_path: Path,
    height: int,
    width: int,
    torch: Any,
) -> bool:
    helper = getattr(_decode_latents_module, "save_decoded_latents", None)
    if helper is not None:
        try:
            helper(pipe, latents, output_path, height=height, width=width)
            return True
        except Exception as exc:
            print(f"Package latent decoder failed for {output_path}: {exc}", file=sys.stderr)

    if not hasattr(pipe, "_unpack_latents") or not hasattr(pipe, "vae"):
        return False
    try:
        with torch.no_grad():
            vae = pipe.vae
            device = getattr(vae, "device", None) or getattr(latents, "device", None)
            dtype = getattr(vae, "dtype", None) or getattr(latents, "dtype", None)
            preview_latents = latents.detach()
            if device is not None or dtype is not None:
                preview_latents = preview_latents.to(device=device, dtype=dtype)
            preview_latents = pipe._unpack_latents(
                preview_latents,
                height,
                width,
                pipe.vae_scale_factor,
            )
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
            shift_factor = getattr(vae.config, "shift_factor", 0.0)
            preview_latents = (preview_latents / scaling_factor) + shift_factor
            decoded = vae.decode(preview_latents, return_dict=False)[0]
            images = pipe.image_processor.postprocess(decoded, output_type="pil")
            images[0].save(output_path)
        return True
    except Exception as exc:
        print(f"Could not save decoded latent preview {output_path}: {exc}", file=sys.stderr)
        return False


def _save_decoded_preview(
    pipe: Any,
    latents: Any,
    output_path: Path,
    height: int,
    width: int,
) -> bool:
    try:
        from hf_image_gen.decode_latents import save_decoded_latent_preview

        save_decoded_latent_preview(
            latents,
            output_path,
            pipe=pipe,
            height=height,
            width=width,
            allow_fallback=True,
        )
        return True
    except Exception as exc:
        print(f"Could not save decoded latent preview {output_path}: {exc}", file=sys.stderr)
        return False


def _package_versions(*package_names: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in package_names:
        if name == "python":
            versions[name] = platform.python_version()
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (RuntimeError, TypeError, ValueError):
            pass
    if hasattr(value, "shape"):
        return {
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", None)),
            "device": str(getattr(value, "device", None)),
        }
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
