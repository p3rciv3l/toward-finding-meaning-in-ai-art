#!/usr/bin/env python3
"""HF Inference Provider/FAL final-image fallback for FLUX.1-Krea-dev.

This entrypoint is intentionally dry-run first. It creates the run directory,
prompt, and metadata without network access unless ``--run-inference`` is
provided. The real provider path saves only the final image because
``huggingface_hub.InferenceClient.text_to_image`` does not expose Diffusers
per-step latents.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_PROVIDER = "fal-ai"
DEFAULT_OUTPUT_ROOT = Path("hf_image_gen/image")
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1536
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_SEED = 0
DEFAULT_OUTPUT_FORMAT = "png"

DEFAULT_PROMPT = """hyperreal close portrait of a woman meeting the viewer's gaze head-on, eyes steady and lucid beneath a helmet cropped tight at the frame's edge. her face carries a dusting of white frost along the lashes and temples, lips slightly parted as if mid-breath, fine moisture glinting where the cold meets warmth. the light comes from behind a low golden sun cutting through icy air wrapping her in a soft halo while highlights cling delicately to the frost edges. her skin reveals honest micro detail: faint pores, subsurface glow on the cheeks, and a natural sheen from the chill. the palette moves between warm amber and arctic blue. every surface behaves realistically matte skin diffusing light, frost refracting it, the atmosphere crisp yet breathable. the mood is tender intensity, a quiet warmth radiating through the ice. captured on an 85mm lens at f/2.0, focus locked to her eyes, shallow depth isolating her face in luminous realism.
--v 7 --ar 2:3 --raw --profile"""


JsonDict = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-inference",
        action="store_true",
        help="Actually call the selected HF Inference Provider and save the final image.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Create prompt and metadata only. This is also the default when --run-inference is omitted.",
    )
    parser.add_argument("--model", "--model-id", dest="model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--prompt", help="Prompt text. Defaults to the project FLUX.1 Krea prompt.")
    parser.add_argument("--prompt-file", type=Path, help="Read prompt text from a UTF-8 file.")
    parser.add_argument("--negative-prompt", help="Optional negative prompt.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Run directory name. Defaults to a UTC timestamp and seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow an existing explicit run directory to be reused.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--steps", "--num-inference-steps", dest="steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scheduler", help="Optional scheduler override accepted by the selected provider.")
    parser.add_argument("--output-format", choices=("png", "jpeg", "jpg"), default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--image-filename", help="Final image filename. Defaults to image.<output-format>.")
    parser.add_argument(
        "--extra-body-json",
        help="JSON object passed as InferenceClient.text_to_image(extra_body=...).",
    )
    parser.add_argument(
        "--extra-body-file",
        type=Path,
        help="Path to a JSON object merged into provider-specific extra_body before --extra-body-json.",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "Provider token/API key. Defaults to HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, "
            "FAL_KEY, or FAL_API_KEY from the environment."
        ),
    )
    parser.add_argument("--timeout", type=float, help="Optional InferenceClient timeout in seconds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    started_at = time.time()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        prompt = resolve_prompt(args.prompt, args.prompt_file)
        extra_body = load_extra_body(args.extra_body_json, args.extra_body_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    dry_run = not args.run_inference
    run_id, run_dir = resolve_run_dir(args.output_root, args.run_id, args.seed, args.overwrite)
    final_image_path = run_dir / "final image" / resolve_image_filename(args.output_format, args.image_filename)
    prompt_path = run_dir / "prompt.txt"
    metadata_path = run_dir / "metadata.json"
    request_kwargs = build_request_kwargs(args, prompt, extra_body)
    api_key = resolve_api_key(args.api_key)
    api_key_source = resolve_api_key_source(args.api_key, api_key)

    try:
        (run_dir / "latents").mkdir(parents=True, exist_ok=True)
        (run_dir / "decoded_latents").mkdir(parents=True, exist_ok=True)
        (run_dir / "final image").mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: could not prepare run directory: {exc}", file=sys.stderr)
        return 1

    metadata = build_metadata(
        args=args,
        dry_run=dry_run,
        run_id=run_id,
        run_dir=run_dir,
        prompt=prompt,
        prompt_path=prompt_path,
        metadata_path=metadata_path,
        final_image_path=final_image_path,
        request_kwargs=request_kwargs,
        extra_body=extra_body,
        api_key_source=api_key_source,
        started_at=started_at,
    )

    if dry_run:
        metadata["status"] = "dry_run_created"
        metadata["inference_ran"] = False
        metadata["result"] = {
            "saved": False,
            "reason": "dry-run mode; no network request or provider inference was performed",
        }
        metadata["artifacts"]["final_image_path"] = None
        write_json(metadata_path, metadata)
        print(json.dumps({"status": "dry_run_created", "run_dir": str(run_dir)}, indent=2))
        return 0

    try:
        metadata["provider_call_attempted"] = True
        result = run_provider_call(
            provider=args.provider,
            api_key=api_key,
            timeout=args.timeout,
            request_kwargs=request_kwargs,
        )
        image_info = save_result_image(result, final_image_path)
        metadata["status"] = "completed"
        metadata["inference_ran"] = True
        metadata["result"] = image_info
        metadata["duration_seconds"] = round(time.time() - started_at, 3)
        write_json(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "final_image_path": str(final_image_path),
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # pragma: no cover - intentionally not exercised by worker verification.
        metadata["status"] = "failed"
        metadata["inference_ran"] = False
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        metadata["duration_seconds"] = round(time.time() - started_at, 3)
        write_json(metadata_path, metadata)
        print(f"error: {exc}", file=sys.stderr)
        print(f"metadata written to {metadata_path}", file=sys.stderr)
        return 1


def validate_args(args: argparse.Namespace) -> None:
    if args.prompt and args.prompt_file:
        raise ValueError("use either --prompt or --prompt-file, not both")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers")
    if args.steps <= 0:
        raise ValueError("--steps must be a positive integer")
    if args.guidance_scale < 0:
        raise ValueError("--guidance-scale must be non-negative")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")


def resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt_file is not None:
        resolved = prompt_file.read_text(encoding="utf-8").strip()
    elif prompt is not None:
        resolved = prompt.strip()
    else:
        resolved = DEFAULT_PROMPT.strip()
    if not resolved:
        raise ValueError("prompt cannot be empty")
    return resolved


def parse_json_object(value: str, *, source: str) -> JsonDict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{source} must decode to a JSON object")
    return parsed


def load_extra_body(json_text: str | None, json_file: Path | None) -> JsonDict:
    extra_body: JsonDict = {}
    if json_file is not None:
        extra_body.update(parse_json_object(json_file.read_text(encoding="utf-8"), source=str(json_file)))
    if json_text:
        extra_body.update(parse_json_object(json_text, source="--extra-body-json"))
    return extra_body


def resolve_run_dir(output_root: Path, run_id: str | None, seed: int, overwrite: bool) -> tuple[str, Path]:
    output_root = Path(output_root)
    if run_id:
        candidate = output_root / run_id
        if candidate.exists() and not overwrite:
            raise ValueError(f"run directory already exists: {candidate}; pass --overwrite to reuse it")
        return run_id, candidate

    base_run_id = f"fal_provider_{utc_stamp()}_seed{seed}"
    candidate = output_root / base_run_id
    if overwrite or not candidate.exists():
        return base_run_id, candidate
    for index in range(1, 1000):
        deduped_run_id = f"{base_run_id}_{index:02d}"
        candidate = output_root / deduped_run_id
        if not candidate.exists():
            return deduped_run_id, candidate
    raise ValueError(f"could not create a unique run directory under {output_root}")


def resolve_image_filename(output_format: str, image_filename: str | None) -> str:
    if image_filename:
        return image_filename
    extension = "jpg" if output_format == "jpg" else output_format
    return f"image.{extension}"


def build_request_kwargs(args: argparse.Namespace, prompt: str, extra_body: JsonDict) -> JsonDict:
    request_kwargs: JsonDict = {
        "prompt": prompt,
        "model": args.model_id,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    if args.negative_prompt:
        request_kwargs["negative_prompt"] = args.negative_prompt
    if args.scheduler:
        request_kwargs["scheduler"] = args.scheduler
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


def build_metadata(
    *,
    args: argparse.Namespace,
    dry_run: bool,
    run_id: str,
    run_dir: Path,
    prompt: str,
    prompt_path: Path,
    metadata_path: Path,
    final_image_path: Path,
    request_kwargs: JsonDict,
    extra_body: JsonDict,
    api_key_source: str,
    started_at: float,
) -> JsonDict:
    return {
        "schema_version": 1,
        "created_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started_at, 3),
        "execution_mode": "hf_fal_provider" if not dry_run else "hf_fal_provider_dry_run",
        "dry_run": dry_run,
        "inference_ran": False,
        "provider_call_attempted": False,
        "model": {
            "model_id": args.model_id,
            "provider": args.provider,
        },
        "prompt": prompt,
        "generation": {
            "width": args.width,
            "height": args.height,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt,
            "scheduler": args.scheduler,
            "image_count": 1,
            "output_format": args.output_format,
        },
        "provider_request": {
            "client": "huggingface_hub.InferenceClient",
            "client_kwargs": {
                "provider": args.provider,
                "api_key_source": api_key_source,
                "timeout": args.timeout,
            },
            "method": "text_to_image",
            "method_kwargs": request_kwargs,
            "extra_body": extra_body,
        },
        "latents_available": False,
        "latents_unavailable_reason": (
            "Hugging Face Inference Provider/FAL text-to-image calls return the final image only; "
            "Diffusers per-step latent tensors and decoded latent previews are not exposed."
        ),
        "latents": {
            "available": False,
            "decoded_previews_available": False,
            "latents_dir": str(run_dir / "latents"),
            "decoded_latents_dir": str(run_dir / "decoded_latents"),
            "expected_latent_files": 0,
            "expected_decoded_preview_files": 0,
            "reason": (
                "Provider-hosted InferenceClient.text_to_image does not expose callback "
                "latents from the underlying Diffusers denoising loop."
            ),
        },
        "artifacts": {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "prompt_path": str(prompt_path),
            "metadata_path": str(metadata_path),
            "latents_dir": str(run_dir / "latents"),
            "decoded_latents_dir": str(run_dir / "decoded_latents"),
            "final_image_dir": str(final_image_path.parent),
            "final_image_path": str(final_image_path),
        },
        "versions": dependency_versions(),
    }


def run_provider_call(
    *,
    provider: str,
    api_key: str | None,
    timeout: float | None,
    request_kwargs: JsonDict,
) -> Any:
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for provider inference. Install huggingface_hub "
            "and Pillow before running with --run-inference."
        ) from exc

    client_kwargs: JsonDict = {"provider": provider}
    if api_key:
        client_kwargs["api_key"] = api_key
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    client = InferenceClient(**client_kwargs)
    return client.text_to_image(**request_kwargs)


def save_result_image(result: Any, final_image_path: Path) -> JsonDict:
    final_image_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "save"):
        image = result
        suffix = final_image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"} and getattr(image, "mode", None) in {"RGBA", "LA", "P"}:
            image = image.convert("RGB")
        image.save(final_image_path)
        return {
            "saved": True,
            "path": str(final_image_path),
            "result_type": type(result).__name__,
            "mode": getattr(image, "mode", None),
            "size": list(getattr(image, "size", [])) or None,
        }
    if isinstance(result, (bytes, bytearray)):
        final_image_path.write_bytes(bytes(result))
        return {
            "saved": True,
            "path": str(final_image_path),
            "result_type": type(result).__name__,
            "mode": None,
            "size": None,
        }
    raise TypeError(
        "InferenceClient.text_to_image returned an unsupported result type "
        f"{type(result).__name__}; expected a PIL.Image-like object or bytes."
    )


def resolve_api_key(explicit_api_key: str | None) -> str | None:
    return (
        explicit_api_key
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("FAL_KEY")
        or os.getenv("FAL_API_KEY")
    )


def resolve_api_key_source(explicit_api_key: str | None, resolved_api_key: str | None) -> str:
    if explicit_api_key:
        return "argument"
    if not resolved_api_key:
        return "not_provided"
    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "FAL_KEY", "FAL_API_KEY"):
        if os.getenv(env_name) == resolved_api_key:
            return f"environment:{env_name}"
    return "environment"


def dependency_versions() -> JsonDict:
    return {
        "python": sys.version.split()[0],
        "huggingface_hub": distribution_version("huggingface-hub"),
        "Pillow": distribution_version("Pillow"),
    }


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
