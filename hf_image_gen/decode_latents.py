"""Latent preview helpers for FLUX/Diffusers runs.

This module deliberately does not load models or import Diffusers at module
import time. Pass an already constructed pipeline when VAE decoding is wanted;
otherwise the normalized preview path can render dry-run tensors/lists.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


class LatentPreviewError(RuntimeError):
    """Raised when a latent preview cannot be produced."""


@dataclass(frozen=True)
class LatentPreviewResult:
    """A rendered preview and metadata about the path used to produce it."""

    image: Any
    mode: str
    fallback_reason: str | None = None


def decode_latents_to_pil(
    latents: Any,
    *,
    pipe: Any | None = None,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    prefer_vae: bool = True,
    allow_fallback: bool = True,
) -> Any:
    """Return a PIL image preview for a latent tensor.

    If ``pipe`` has a usable ``vae.decode`` method, this first tries the real
    VAE decode path. For FLUX packed latents, known unpack hooks such as
    ``pipe._unpack_latents`` are used before decode when available. If VAE
    decode is unavailable or fails and ``allow_fallback`` is true, this returns
    a deterministic min/max-normalized tensor preview instead.
    """

    return make_latent_preview(
        latents,
        pipe=pipe,
        height=height,
        width=width,
        image_index=image_index,
        prefer_vae=prefer_vae,
        allow_fallback=allow_fallback,
    ).image


def decode_latents(
    latents: Any,
    *,
    pipe: Any | None = None,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    prefer_vae: bool = True,
    allow_fallback: bool = True,
) -> Any:
    """Compatibility alias for ``decode_latents_to_pil``."""

    return decode_latents_to_pil(
        latents,
        pipe=pipe,
        height=height,
        width=width,
        image_index=image_index,
        prefer_vae=prefer_vae,
        allow_fallback=allow_fallback,
    )


def make_latent_preview(
    latents: Any,
    *,
    pipe: Any | None = None,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    prefer_vae: bool = True,
    allow_fallback: bool = True,
) -> LatentPreviewResult:
    """Build a latent preview image without loading any models."""

    if prefer_vae and pipe is not None and _has_vae_decode(pipe):
        try:
            image = _decode_with_pipe_vae(
                latents,
                pipe=pipe,
                height=height,
                width=width,
                image_index=image_index,
            )
            return LatentPreviewResult(image=image, mode="vae")
        except Exception as exc:  # noqa: BLE001 - fallback is the point here.
            if not allow_fallback:
                raise LatentPreviewError("VAE latent decode failed") from exc
            fallback_reason = f"{exc.__class__.__name__}: {exc}"
    else:
        fallback_reason = "VAE decode unavailable"
        if prefer_vae and pipe is None:
            fallback_reason = "No pipeline supplied"

    if not allow_fallback:
        raise LatentPreviewError(fallback_reason)

    image = normalized_tensor_preview(
        latents,
        height=height,
        width=width,
        image_index=image_index,
        vae_scale_factor=_get_vae_scale_factor(pipe),
    )
    return LatentPreviewResult(
        image=image,
        mode="normalized",
        fallback_reason=fallback_reason,
    )


def save_latent_preview(
    latents: Any,
    output_path: str | Path,
    *,
    pipe: Any | None = None,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    prefer_vae: bool = True,
    allow_fallback: bool = True,
    compress_level: int = 6,
) -> Path:
    """Render a latent preview and save it as a PNG."""

    result = make_latent_preview(
        latents,
        pipe=pipe,
        height=height,
        width=width,
        image_index=image_index,
        prefer_vae=prefer_vae,
        allow_fallback=allow_fallback,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(path, format="PNG", compress_level=compress_level)
    return path


def save_decoded_latent_preview(
    latents: Any,
    output_path: str | Path,
    *,
    pipe: Any | None = None,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    prefer_vae: bool = True,
    allow_fallback: bool = True,
    compress_level: int = 6,
) -> Path:
    """Compatibility wrapper for callers that use decoded-preview naming."""

    return save_latent_preview(
        latents,
        output_path,
        pipe=pipe,
        height=height,
        width=width,
        image_index=image_index,
        prefer_vae=prefer_vae,
        allow_fallback=allow_fallback,
        compress_level=compress_level,
    )


def save_step_latent_preview(
    latents: Any,
    output_dir: str | Path,
    *,
    step: int,
    pipe: Any | None = None,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    prefix: str = "step",
    prefer_vae: bool = True,
    allow_fallback: bool = True,
) -> Path:
    """Save a step-indexed latent preview PNG under ``output_dir``."""

    filename = f"{prefix}_{step:03d}.png"
    return save_latent_preview(
        latents,
        Path(output_dir) / filename,
        pipe=pipe,
        height=height,
        width=width,
        image_index=image_index,
        prefer_vae=prefer_vae,
        allow_fallback=allow_fallback,
    )


def normalized_tensor_preview(
    latents: Any,
    *,
    height: int | None = None,
    width: int | None = None,
    image_index: int = 0,
    vae_scale_factor: int | None = None,
) -> Any:
    """Render a normalized RGB preview from a tensor-like object or nested list."""

    data, shape = _to_nested_float_data(latents)
    pixels, image_size = _preview_pixels_from_data(
        data,
        shape,
        height=height,
        width=width,
        image_index=image_index,
        vae_scale_factor=vae_scale_factor,
        value_mode="minmax",
    )
    return _pil_from_pixels(pixels, image_size)


def _decode_with_pipe_vae(
    latents: Any,
    *,
    pipe: Any,
    height: int | None,
    width: int | None,
    image_index: int,
) -> Any:
    vae = pipe.vae
    decode_latents_for_vae = prepare_latents_for_vae_decode(
        latents,
        pipe=pipe,
        height=height,
        width=width,
    )
    decode_latents_for_vae = _move_to_vae_device_dtype(decode_latents_for_vae, vae)

    with _maybe_no_grad():
        try:
            decoded = vae.decode(decode_latents_for_vae, return_dict=False)
        except TypeError:
            decoded = vae.decode(decode_latents_for_vae)

    decoded = _unwrap_decode_output(decoded)

    image_processor = getattr(pipe, "image_processor", None)
    postprocess = getattr(image_processor, "postprocess", None)
    if callable(postprocess):
        images = postprocess(decoded, output_type="pil")
        return _select_image(images, image_index)

    return _decoded_tensor_to_pil(decoded, image_index=image_index)


def prepare_latents_for_vae_decode(
    latents: Any,
    *,
    pipe: Any,
    height: int | None = None,
    width: int | None = None,
) -> Any:
    """Clone/detach, FLUX-unpack, and scale latents for ``pipe.vae.decode``."""

    prepared = _clone_detach(latents)
    prepared = unpack_flux_latents_if_needed(
        prepared,
        pipe=pipe,
        height=height,
        width=width,
    )

    vae_config = getattr(getattr(pipe, "vae", None), "config", None)
    scaling_factor = getattr(vae_config, "scaling_factor", None)
    shift_factor = getattr(vae_config, "shift_factor", None)

    if scaling_factor not in (None, 0):
        prepared = prepared / scaling_factor
    if shift_factor is not None:
        prepared = prepared + shift_factor

    return prepared


def unpack_flux_latents_if_needed(
    latents: Any,
    *,
    pipe: Any | None,
    height: int | None = None,
    width: int | None = None,
) -> Any:
    """Unpack FLUX packed latents when shape and hooks indicate it is needed."""

    shape = _shape_tuple(latents)
    if len(shape) != 3:
        return latents

    scale = _get_vae_scale_factor(pipe)
    resolved_height, resolved_width = _resolve_target_size(
        pipe,
        height=height,
        width=width,
        vae_scale_factor=scale,
        packed_shape=shape,
    )

    for hook_name in ("_unpack_latents", "unpack_latents"):
        hook = getattr(pipe, hook_name, None) if pipe is not None else None
        if not callable(hook):
            continue
        for args in (
            (latents, resolved_height, resolved_width, scale),
            (latents, resolved_height, resolved_width),
            (latents,),
        ):
            try:
                return hook(*args)
            except TypeError:
                continue

    return _local_flux_unpack_if_shape_matches(
        latents,
        shape=shape,
        height=resolved_height,
        width=resolved_width,
        vae_scale_factor=scale,
    )


def _local_flux_unpack_if_shape_matches(
    latents: Any,
    *,
    shape: tuple[int, ...],
    height: int | None,
    width: int | None,
    vae_scale_factor: int | None,
) -> Any:
    if height is None or width is None or vae_scale_factor is None:
        return latents
    if len(shape) != 3:
        return latents

    batch_size, num_patches, channels = shape
    if channels % 4 != 0:
        return latents

    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    expected_patches = (latent_height // 2) * (latent_width // 2)
    if expected_patches != num_patches:
        return latents

    try:
        return (
            latents.view(
                batch_size,
                latent_height // 2,
                latent_width // 2,
                channels // 4,
                2,
                2,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch_size, channels // 4, latent_height, latent_width)
        )
    except AttributeError:
        return latents


def _decoded_tensor_to_pil(decoded: Any, *, image_index: int) -> Any:
    data, shape = _to_nested_float_data(decoded)
    pixels, image_size = _preview_pixels_from_data(
        data,
        shape,
        height=None,
        width=None,
        image_index=image_index,
        vae_scale_factor=None,
        value_mode="decoded",
    )
    return _pil_from_pixels(pixels, image_size)


def _preview_pixels_from_data(
    data: Any,
    shape: tuple[int, ...],
    *,
    height: int | None,
    width: int | None,
    image_index: int,
    vae_scale_factor: int | None,
    value_mode: str,
) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    if not shape:
        return _pixels_from_scalar(float(data), value_mode=value_mode), (1, 1)

    if len(shape) == 4:
        sample = _select_sequence(data, shape[0], image_index)
        pixels, size = _pixels_from_chw(sample, value_mode=value_mode)
        return pixels, size

    if len(shape) == 3:
        if _looks_like_batched_sequence(shape):
            sample = _select_sequence(data, shape[0], image_index)
            return _pixels_from_sequence_channels(
                sample,
                height=height,
                width=width,
                vae_scale_factor=vae_scale_factor,
                value_mode=value_mode,
            )
        if _looks_like_hwc(shape):
            return _pixels_from_hwc(data, value_mode=value_mode)
        return _pixels_from_chw(data, value_mode=value_mode)

    if len(shape) == 2:
        if _looks_like_sequence_channels(shape):
            return _pixels_from_sequence_channels(
                data,
                height=height,
                width=width,
                vae_scale_factor=vae_scale_factor,
                value_mode=value_mode,
            )
        return _pixels_from_hw(data, value_mode=value_mode)

    flat = _flatten_floats(data)
    grid_height, grid_width = _factor_grid(len(flat))
    rows = []
    index = 0
    for _ in range(grid_height):
        row = []
        for _ in range(grid_width):
            row.append(flat[index] if index < len(flat) else 0.0)
            index += 1
        rows.append(row)
    return _pixels_from_hw(rows, value_mode=value_mode)


def _pixels_from_scalar(value: float, *, value_mode: str) -> list[tuple[int, int, int]]:
    pixel = _normalize_pixel_channels([[value]], value_mode=value_mode)[0]
    return [pixel]


def _pixels_from_hw(data: Any, *, value_mode: str) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    height = len(data)
    width = len(data[0]) if height else 0
    raw_pixels = []
    for y in range(height):
        for x in range(width):
            value = _safe_float(data[y][x])
            raw_pixels.append((value, value, value))
    return _normalize_pixel_channels(raw_pixels, value_mode=value_mode), (width, height)


def _pixels_from_hwc(data: Any, *, value_mode: str) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    height = len(data)
    width = len(data[0]) if height else 0
    raw_pixels = []
    for y in range(height):
        for x in range(width):
            raw_pixels.append(_rgb_triplet(data[y][x]))
    return _normalize_pixel_channels(raw_pixels, value_mode=value_mode), (width, height)


def _pixels_from_chw(data: Any, *, value_mode: str) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    channels = len(data)
    height = len(data[0]) if channels else 0
    width = len(data[0][0]) if height else 0
    raw_pixels = []
    for y in range(height):
        for x in range(width):
            values = [data[channel][y][x] for channel in range(channels)]
            raw_pixels.append(_rgb_triplet(values))
    return _normalize_pixel_channels(raw_pixels, value_mode=value_mode), (width, height)


def _pixels_from_sequence_channels(
    data: Any,
    *,
    height: int | None,
    width: int | None,
    vae_scale_factor: int | None,
    value_mode: str,
) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    length = len(data)
    grid_height, grid_width = _sequence_grid_dimensions(
        length,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
    )
    raw_pixels = []
    for index in range(grid_height * grid_width):
        if index < length:
            raw_pixels.append(_rgb_triplet(data[index]))
        else:
            raw_pixels.append((0.0, 0.0, 0.0))
    return _normalize_pixel_channels(raw_pixels, value_mode=value_mode), (grid_width, grid_height)


def _normalize_pixel_channels(
    raw_pixels: list[tuple[float, float, float]],
    *,
    value_mode: str,
) -> list[tuple[int, int, int]]:
    if value_mode == "decoded":
        return [
            (
                _unit_to_byte((r + 1.0) / 2.0),
                _unit_to_byte((g + 1.0) / 2.0),
                _unit_to_byte((b + 1.0) / 2.0),
            )
            for r, g, b in raw_pixels
        ]

    finite_values = [value for pixel in raw_pixels for value in pixel if math.isfinite(value)]
    if not finite_values:
        return [(127, 127, 127) for _ in raw_pixels]

    min_value = min(finite_values)
    max_value = max(finite_values)
    span = max_value - min_value
    if span <= 1e-12:
        return [(127, 127, 127) for _ in raw_pixels]

    normalized = []
    for r, g, b in raw_pixels:
        normalized.append(
            (
                _unit_to_byte((r - min_value) / span),
                _unit_to_byte((g - min_value) / span),
                _unit_to_byte((b - min_value) / span),
            )
        )
    return normalized


def _rgb_triplet(values: Any) -> tuple[float, float, float]:
    sequence = list(values) if _is_sequence(values) else [values]
    if not sequence:
        return (0.0, 0.0, 0.0)
    if len(sequence) == 1:
        value = _safe_float(sequence[0])
        return (value, value, value)
    if len(sequence) == 2:
        first = _safe_float(sequence[0])
        second = _safe_float(sequence[1])
        return (first, second, (first + second) / 2.0)
    return (_safe_float(sequence[0]), _safe_float(sequence[1]), _safe_float(sequence[2]))


def _pil_from_pixels(pixels: list[tuple[int, int, int]], size: tuple[int, int]) -> Any:
    if size[0] <= 0 or size[1] <= 0:
        raise LatentPreviewError("Cannot create a preview from an empty latent tensor")

    Image = _import_pil_image()
    image = Image.new("RGB", size)
    image.putdata(pixels)
    return image


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise LatentPreviewError(
            "Pillow is required to save latent preview PNGs. Install it with `pip install Pillow`."
        ) from exc
    return Image


def _to_nested_float_data(value: Any) -> tuple[Any, tuple[int, ...]]:
    tensor = _clone_detach(value)
    for method_name in ("float", "cpu"):
        method = getattr(tensor, method_name, None)
        if callable(method):
            try:
                tensor = method()
            except TypeError:
                pass

    shape = _shape_tuple(tensor)
    tolist = getattr(tensor, "tolist", None)
    if callable(tolist):
        data = tolist()
        if not shape:
            shape = _infer_shape(data)
        return data, shape

    if _is_sequence(tensor):
        return tensor, _infer_shape(tensor)

    return _safe_float(tensor), ()


def _clone_detach(value: Any) -> Any:
    cloned = value
    detach = getattr(cloned, "detach", None)
    if callable(detach):
        cloned = detach()
    clone = getattr(cloned, "clone", None)
    if callable(clone):
        cloned = clone()
    return cloned


def _move_to_vae_device_dtype(latents: Any, vae: Any) -> Any:
    to_method = getattr(latents, "to", None)
    if not callable(to_method):
        return latents

    device = None
    dtype = None
    parameters = getattr(vae, "parameters", None)
    if callable(parameters):
        try:
            first_param = next(parameters())
            device = getattr(first_param, "device", None)
            dtype = getattr(first_param, "dtype", None)
        except (StopIteration, TypeError):
            pass

    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if not kwargs:
        return latents

    try:
        return to_method(**kwargs)
    except TypeError:
        return latents


def _maybe_no_grad() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


def _unwrap_decode_output(decoded: Any) -> Any:
    sample = getattr(decoded, "sample", None)
    if sample is not None:
        return sample
    if isinstance(decoded, (tuple, list)) and decoded:
        return decoded[0]
    return decoded


def _select_image(images: Any, image_index: int) -> Any:
    if isinstance(images, (tuple, list)):
        return _select_sequence(images, len(images), image_index)
    return images


def _select_sequence(sequence: Any, length: int, index: int) -> Any:
    if index < 0 or index >= length:
        raise LatentPreviewError(f"image_index {index} is outside batch size {length}")
    return sequence[index]


def _has_vae_decode(pipe: Any) -> bool:
    vae = getattr(pipe, "vae", None)
    return callable(getattr(vae, "decode", None))


def _get_vae_scale_factor(pipe: Any | None) -> int | None:
    scale = getattr(pipe, "vae_scale_factor", None) if pipe is not None else None
    if scale is not None:
        try:
            return int(scale)
        except (TypeError, ValueError):
            pass

    vae = getattr(pipe, "vae", None) if pipe is not None else None
    config = getattr(vae, "config", None)
    channels = getattr(config, "block_out_channels", None)
    if channels is not None:
        try:
            return 2 ** (len(channels) - 1)
        except TypeError:
            return None
    return None


def _resolve_target_size(
    pipe: Any | None,
    *,
    height: int | None,
    width: int | None,
    vae_scale_factor: int | None,
    packed_shape: tuple[int, ...],
) -> tuple[int | None, int | None]:
    if height is not None and width is not None:
        return height, width

    default_sample_size = getattr(pipe, "default_sample_size", None) if pipe is not None else None
    if default_sample_size is not None and vae_scale_factor is not None:
        resolved = int(default_sample_size) * int(vae_scale_factor)
        return height or resolved, width or resolved

    if vae_scale_factor is not None and len(packed_shape) == 3:
        _, num_patches, _ = packed_shape
        grid_height, grid_width = _factor_grid(num_patches)
        inferred_height = grid_height * vae_scale_factor * 2
        inferred_width = grid_width * vae_scale_factor * 2
        return height or inferred_height, width or inferred_width

    return height, width


def _sequence_grid_dimensions(
    length: int,
    *,
    height: int | None,
    width: int | None,
    vae_scale_factor: int | None,
) -> tuple[int, int]:
    if height is not None and width is not None and vae_scale_factor:
        grid_height = int(height) // (int(vae_scale_factor) * 2)
        grid_width = int(width) // (int(vae_scale_factor) * 2)
        if grid_height > 0 and grid_width > 0 and grid_height * grid_width == length:
            return grid_height, grid_width
    return _factor_grid(length)


def _factor_grid(length: int) -> tuple[int, int]:
    if length <= 0:
        return 0, 0
    root = int(math.sqrt(length))
    for height in range(root, 0, -1):
        if length % height == 0:
            return height, length // height
    return root, math.ceil(length / root)


def _looks_like_batched_sequence(shape: tuple[int, ...]) -> bool:
    if len(shape) != 3:
        return False
    batch, tokens, channels = shape
    return batch <= 16 and channels >= 8 and tokens > channels


def _looks_like_sequence_channels(shape: tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    tokens, channels = shape
    return channels >= 8 and tokens > channels


def _looks_like_hwc(shape: tuple[int, ...]) -> bool:
    if len(shape) != 3:
        return False
    return shape[-1] in (1, 2, 3, 4) and shape[0] > 4 and shape[1] > 4


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return ()


def _infer_shape(value: Any) -> tuple[int, ...]:
    if not _is_sequence(value):
        return ()
    shape = []
    cursor = value
    while _is_sequence(cursor):
        shape.append(len(cursor))
        if not cursor:
            break
        cursor = cursor[0]
    return tuple(shape)


def _flatten_floats(value: Any) -> list[float]:
    if not _is_sequence(value):
        return [_safe_float(value)]
    flattened = []
    for item in value:
        flattened.extend(_flatten_floats(item))
    return flattened


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _unit_to_byte(value: float) -> int:
    if not math.isfinite(value):
        value = 0.0
    value = min(1.0, max(0.0, value))
    return int(round(value * 255))
