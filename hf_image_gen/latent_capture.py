"""Diffusers callback utilities for saving intermediate latents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decode_latents import save_decoded_latent_preview
from .output_paths import RunPaths

DIFFUSERS_CALLBACK_TENSOR_INPUTS = ["latents"]


@dataclass(slots=True)
class LatentCapture:
    paths: RunPaths
    save_latents: bool = True
    save_decoded: bool = True
    preview_every: int = 1
    height: int | None = None
    width: int | None = None
    saved_latent_steps: list[int] = field(default_factory=list)
    saved_preview_steps: list[int] = field(default_factory=list)
    preview_methods: dict[int, str] = field(default_factory=dict)

    def __call__(self, pipe: Any, step: int, timestep: int, callback_kwargs: dict[str, Any]):
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        cloned = clone_latents_for_save(latents)
        if self.save_latents:
            self._save_latent_tensor(step, cloned)
        if self.save_decoded and step % self.preview_every == 0:
            save_decoded_latent_preview(
                cloned,
                self.paths.decoded_latent_path(step),
                pipe=pipe,
                height=self.height,
                width=self.width,
            )
            self.saved_preview_steps.append(step)
            self.preview_methods[step] = "decoded_latent_preview"
        return callback_kwargs

    def _save_latent_tensor(self, step: int, latents: Any) -> None:
        import torch

        torch.save(latents, self.paths.latent_path(step))
        self.saved_latent_steps.append(step)


@dataclass(frozen=True, slots=True)
class LatentCaptureRecord:
    step_index: int
    timestep: Any
    latent_path: Path | None = None
    preview_path: Path | None = None


@dataclass(slots=True)
class LatentCaptureCallback:
    """Standalone latent callback with testable save and preview controls."""

    latent_dir: Path | str
    save_latents: bool = True
    latent_save_every: int = 1
    max_latent_saves: int | None = None
    save_previews: bool = False
    preview_dir: Path | str | None = None
    preview_save_every: int = 1
    max_preview_saves: int | None = None
    preview_callback: Any | None = None
    torch_module: Any | None = None
    records: list[LatentCaptureRecord] = field(default_factory=list)
    latent_paths: list[Path] = field(default_factory=list)
    preview_paths: list[Path] = field(default_factory=list)

    @property
    def callback_on_step_end_tensor_inputs(self) -> list[str]:
        return list(DIFFUSERS_CALLBACK_TENSOR_INPUTS)

    def __call__(self, pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]):
        if "latents" not in callback_kwargs:
            raise KeyError(
                "latents missing from callback_kwargs; pass "
                "callback_on_step_end_tensor_inputs=['latents'] to the Diffusers pipeline"
            )
        latents = detach_cpu_clone(callback_kwargs["latents"])
        latent_path = self._maybe_save_latents(step_index, latents)
        preview_path = self._maybe_save_preview(pipe, step_index, timestep, latents)
        if latent_path is not None or preview_path is not None:
            self.records.append(
                LatentCaptureRecord(
                    step_index=step_index,
                    timestep=timestep,
                    latent_path=latent_path,
                    preview_path=preview_path,
                )
            )
        return callback_kwargs

    def _maybe_save_latents(self, step_index: int, latents: Any) -> Path | None:
        if not self.save_latents:
            return None
        if step_index % self.latent_save_every != 0:
            return None
        if self.max_latent_saves is not None and len(self.latent_paths) >= self.max_latent_saves:
            return None

        latent_dir = Path(self.latent_dir)
        latent_dir.mkdir(parents=True, exist_ok=True)
        path = latent_dir / f"step_{step_index:03d}.pt"
        torch = self.torch_module or _import_torch()
        torch.save(latents, path)
        self.latent_paths.append(path)
        return path

    def _maybe_save_preview(self, pipe: Any, step_index: int, timestep: Any, latents: Any) -> Path | None:
        if not self.save_previews:
            return None
        if step_index % self.preview_save_every != 0:
            return None
        if self.max_preview_saves is not None and len(self.preview_paths) >= self.max_preview_saves:
            return None
        if self.preview_dir is None:
            raise ValueError("preview_dir is required when save_previews=True")

        preview_dir = Path(self.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        output_path = preview_dir / f"step_{step_index:03d}.png"
        if self.preview_callback is not None:
            self.preview_callback(
                pipe=pipe,
                step_index=step_index,
                timestep=timestep,
                latents=latents,
                output_path=output_path,
            )
        else:
            save_decoded_latent_preview(latents, output_path, pipe=pipe)
        self.preview_paths.append(output_path)
        return output_path


def detach_cpu_clone(tensor: Any) -> Any:
    return tensor.detach().cpu().clone()


def clone_latents_for_save(latents: Any):
    return detach_cpu_clone(latents)


def make_latent_capture_callback(latent_dir: str | Path, **kwargs: Any) -> LatentCaptureCallback:
    return LatentCaptureCallback(latent_dir=latent_dir, **kwargs)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to save latent tensors") from exc
    return torch
