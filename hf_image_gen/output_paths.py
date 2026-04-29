"""Run-directory management for image generation artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import GenerationConfig

LATENTS_DIR_NAME = "latents"
DECODED_LATENTS_DIR_NAME = "decoded_latents"
FINAL_IMAGE_DIR_NAME = "final image"
FINAL_IMAGE_FILE_NAME = "image.png"
PROMPT_FILE_NAME = "prompt.txt"
METADATA_FILE_NAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_root: Path
    latents_dir: Path
    decoded_latents_dir: Path
    final_image_dir: Path
    final_image_path: Path
    prompt_path: Path
    metadata_path: Path

    @property
    def run_dir(self) -> Path:
        return self.run_root

    def latent_path(self, step: int) -> Path:
        return self.latents_dir / f"step_{step:03d}.pt"

    def decoded_latent_path(self, step: int) -> Path:
        return self.decoded_latents_dir / f"step_{step:03d}.png"


def prepare_run_paths(config: GenerationConfig) -> RunPaths:
    config.validate()
    paths = paths_for_run(
        config.resolved_run_id(),
        image_root=config.output_root,
        output_format=config.output_format,
    )
    if paths.run_root.exists() and not config.overwrite:
        raise FileExistsError(f"run directory already exists: {paths.run_root}")

    for directory in (
        paths.run_root,
        paths.latents_dir,
        paths.decoded_latents_dir,
        paths.final_image_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def paths_for_run(
    run_id: str,
    *,
    image_root: str | Path | None = None,
    output_format: str = "png",
) -> RunPaths:
    safe_run_id = _validate_run_id(run_id)
    root = Path(image_root if image_root is not None else GenerationConfig().output_root).resolve()
    run_root = root / safe_run_id
    return RunPaths(
        run_root=run_root,
        latents_dir=run_root / LATENTS_DIR_NAME,
        decoded_latents_dir=run_root / DECODED_LATENTS_DIR_NAME,
        final_image_dir=run_root / FINAL_IMAGE_DIR_NAME,
        final_image_path=run_root / FINAL_IMAGE_DIR_NAME / f"image.{_normalized_extension(output_format)}",
        prompt_path=run_root / PROMPT_FILE_NAME,
        metadata_path=run_root / METADATA_FILE_NAME,
    )


def prepare_run_output(
    *,
    prompt: str,
    metadata: Mapping[str, Any],
    run_id: str,
    image_root: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> RunPaths:
    paths = paths_for_run(run_id, image_root=image_root)
    if dry_run:
        return paths
    if paths.run_root.exists() and not overwrite:
        raise FileExistsError(f"run directory already exists: {paths.run_root}")
    for directory in (
        paths.run_root,
        paths.latents_dir,
        paths.decoded_latents_dir,
        paths.final_image_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    write_prompt(paths, prompt, overwrite=overwrite)
    write_metadata(paths, metadata, overwrite=overwrite)
    return paths


def build_run_id(*, seed: int | None = None, prefix: str = "flux_krea") -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    seed_part = "seedauto" if seed is None else f"seed{seed}"
    return f"{_validate_run_id(prefix)}_{stamp}_{seed_part}"


def write_prompt(paths: RunPaths, prompt: str, *, overwrite: bool = False) -> None:
    atomic_write_text(paths.prompt_path, prompt, overwrite=overwrite)


def write_metadata(paths: RunPaths, metadata: Mapping[str, Any], *, overwrite: bool = False) -> None:
    atomic_write_text(
        paths.metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def atomic_write_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def _normalized_extension(output_format: str) -> str:
    value = output_format.lower()
    return "jpg" if value == "jpeg" else value


def _validate_run_id(run_id: str) -> str:
    if not run_id:
        raise ValueError("run_id must not be empty")
    path = Path(run_id)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"run_id must be a single safe path component: {run_id!r}")
    return run_id
