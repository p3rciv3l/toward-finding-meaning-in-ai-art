#!/usr/bin/env python3
"""Create temporary local and HF/FAL dry-run layouts without inference."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_image_gen.generate_fal_provider import main as fal_main
from hf_image_gen.generate_flux_krea import main as flux_main


DRY_PROMPT = "dry-run verification prompt"


def main() -> int:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_summary = check_local_diffusers_dry_run(root)
            provider_summary = check_hf_fal_provider_dry_run(root)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "checks": [local_summary, provider_summary],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke script should report any failed invariant.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


def check_local_diffusers_dry_run(root: Path) -> dict[str, str]:
    output_root = root / "local"
    run_id = "dry_local_check"
    _call_entrypoint(
        flux_main,
        [
            "--dry-run",
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--prompt",
            DRY_PROMPT,
            "--seed",
            "123",
        ],
    )

    run_root = output_root / run_id
    required = [
        run_root / "metadata.json",
        run_root / "prompt.txt",
        run_root / "latents",
        run_root / "decoded_latents",
        run_root / "final image",
    ]
    _require_paths(required)
    _require((run_root / "prompt.txt").read_text(encoding="utf-8").strip() == DRY_PROMPT, "local prompt mismatch")
    metadata = _read_json(run_root / "metadata.json")
    _require(metadata["status"] == "dry_run_complete", "local dry-run status mismatch")
    _require(metadata["execution_mode"] == "dry_run", "local execution mode mismatch")
    _require(metadata["model_loaded"] is False, "local dry-run loaded a model")
    _require(metadata["inference_ran"] is False, "local dry-run ran inference")
    return {"mode": "local_diffusers", "run_root": str(run_root)}


def check_hf_fal_provider_dry_run(root: Path) -> dict[str, str]:
    output_root = root / "provider"
    run_id = "dry_provider_check"
    _call_entrypoint(
        fal_main,
        [
            "--dry-run",
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--prompt",
            DRY_PROMPT,
            "--seed",
            "456",
        ],
    )

    run_root = output_root / run_id
    required = [
        run_root / "metadata.json",
        run_root / "prompt.txt",
        run_root / "final image",
    ]
    _require_paths(required)
    _require((run_root / "prompt.txt").read_text(encoding="utf-8").strip() == DRY_PROMPT, "provider prompt mismatch")
    metadata = _read_json(run_root / "metadata.json")
    _require(metadata["status"] == "dry_run_created", "provider dry-run status mismatch")
    _require(metadata["execution_mode"] == "hf_fal_provider_dry_run", "provider execution mode mismatch")
    _require(metadata["dry_run"] is True, "provider dry_run flag mismatch")
    _require(metadata["inference_ran"] is False, "provider dry-run ran inference")
    _require(metadata["latents_available"] is False, "provider dry-run reported latent capture")
    _require(metadata["latents"]["available"] is False, "provider latent detail mismatch")
    return {"mode": "hf_fal_provider", "run_root": str(run_root)}


def _call_entrypoint(entrypoint, argv: list[str]) -> None:
    with redirect_stdout(StringIO()):
        exit_code = entrypoint(argv)
    if exit_code != 0:
        raise RuntimeError(f"entrypoint returned {exit_code}: {entrypoint.__module__}")


def _require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing dry-run paths: {missing}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
