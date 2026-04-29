"""
Configuration primitives for FLUX/Krea image generation.

This module intentionally has no inference dependencies. It describes and
validates a run so local Diffusers, Hugging Face provider, or FAL-backed runners
can share one configuration shape.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


JsonDict = Dict[str, Any]

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
DEFAULT_FAL_ENDPOINT_ID = "fal-ai/flux/krea"
DEFAULT_HF_PROVIDER = "fal-ai"
DEFAULT_OUTPUT_ROOT = Path("hf_image_gen/image")
DEFAULT_ASPECT_RATIO = "2:3"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1536
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_STEPS = 50
DEFAULT_NUM_INFERENCE_STEPS = DEFAULT_STEPS
DEFAULT_DTYPE = "bfloat16"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_RUN_ID_PREFIX = "flux_krea"
CONFIG_SCHEMA_VERSION = 1

DEFAULT_PROMPT = (
    "hyperreal close portrait of a woman meeting the viewer's gaze head-on, "
    "eyes steady and lucid beneath a helmet cropped tight at the frame's edge. "
    "her face carries a dusting of white frost along the lashes and temples, "
    "lips slightly parted as if mid-breath, fine moisture glinting where the "
    "cold meets warmth. the light comes from behind a low golden sun cutting "
    "through icy air wrapping her in a soft halo while highlights cling "
    "delicately to the frost edges. her skin reveals honest micro detail: faint "
    "pores, subsurface glow on the cheeks, and a natural sheen from the chill. "
    "the palette moves between warm amber and arctic blue. every surface behaves "
    "realistically matte skin diffusing light, frost refracting it, the "
    "atmosphere crisp yet breathable. the mood is tender intensity, a quiet "
    "warmth radiating through the ice. captured on an 85mm lens at f/2.0, focus "
    "locked to her eyes, shallow depth isolating her face in luminous realism.\n"
    "--v 7 --ar 2:3 --raw --profile"
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUPPORTED_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
_LOCAL_LATENT_MODES = frozenset({"local_diffusers", "fal_custom"})

# Backward-compatible aliases used by early runner modules.
MODEL_ID = DEFAULT_MODEL_ID
PROMPT = DEFAULT_PROMPT
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
WIDTH = DEFAULT_WIDTH
HEIGHT = DEFAULT_HEIGHT
GUIDANCE_SCALE = DEFAULT_GUIDANCE_SCALE
NUM_INFERENCE_STEPS = DEFAULT_NUM_INFERENCE_STEPS


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ExecutionMode(str, Enum):
    """Supported execution backends for later runner implementations."""

    LOCAL_DIFFUSERS = "local_diffusers"
    HF_INFERENCE = "hf_inference"
    HF_FAL_PROVIDER = "hf_fal_provider"
    FAL_PROVIDER = "fal_provider"
    FAL_CUSTOM = "fal_custom"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class ProviderSettings:
    """
    Provider-side controls without secrets.

    Tokens are represented as environment variable names so metadata can be
    persisted safely.
    """

    hf_provider: str = DEFAULT_HF_PROVIDER
    fal_endpoint_id: str = DEFAULT_FAL_ENDPOINT_ID
    hf_token_env: str = "HF_TOKEN"
    fal_key_env: str = "FAL_KEY"
    timeout_seconds: Optional[float] = 120.0
    start_timeout_seconds: Optional[float] = None
    client_timeout_seconds: Optional[float] = None
    use_queue: bool = True
    poll_interval_seconds: float = 0.5
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hf_provider", validate_optional_name("hf_provider", self.hf_provider))
        object.__setattr__(
            self,
            "fal_endpoint_id",
            validate_optional_name("fal_endpoint_id", self.fal_endpoint_id),
        )
        object.__setattr__(
            self,
            "hf_token_env",
            validate_optional_name("hf_token_env", self.hf_token_env),
        )
        object.__setattr__(
            self,
            "fal_key_env",
            validate_optional_name("fal_key_env", self.fal_key_env),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            validate_optional_float("timeout_seconds", self.timeout_seconds, minimum=0.0),
        )
        object.__setattr__(
            self,
            "start_timeout_seconds",
            validate_optional_float("start_timeout_seconds", self.start_timeout_seconds, minimum=0.0),
        )
        object.__setattr__(
            self,
            "client_timeout_seconds",
            validate_optional_float("client_timeout_seconds", self.client_timeout_seconds, minimum=0.0),
        )
        object.__setattr__(
            self,
            "poll_interval_seconds",
            validate_float("poll_interval_seconds", self.poll_interval_seconds, minimum=0.0),
        )
        object.__setattr__(self, "extra", dict(self.extra or {}))

    def to_dict(self) -> JsonDict:
        return json_ready(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderSettings":
        return cls(**dict(data))


@dataclass(frozen=True)
class SaveFlags:
    """Artifact switches used by the runner layer."""

    save_prompt: bool = True
    save_metadata: bool = True
    save_final_image: bool = True
    save_latents: bool = True
    save_decoded_latents: bool = True
    save_provider_response: bool = True
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "save_prompt",
            "save_metadata",
            "save_final_image",
            "save_latents",
            "save_decoded_latents",
            "save_provider_response",
            "overwrite_existing",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    def to_dict(self) -> JsonDict:
        return json_ready(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SaveFlags":
        return cls(**dict(data))


@dataclass(frozen=True)
class FluxImageGenConfig:
    """
    Validated, JSON-serializable run configuration for FLUX image generation.
    """

    model_id: str = DEFAULT_MODEL_ID
    prompt: str = DEFAULT_PROMPT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: Optional[str] = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    steps: int = DEFAULT_STEPS
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    dtype: str = DEFAULT_DTYPE
    execution_mode: ExecutionMode = ExecutionMode.LOCAL_DIFFUSERS
    provider: Any = DEFAULT_HF_PROVIDER
    provider_settings: ProviderSettings = field(default_factory=ProviderSettings)
    save: SaveFlags = field(default_factory=SaveFlags)
    save_latents: Optional[bool] = None
    save_decoded_latents: Optional[bool] = None
    output_format: str = DEFAULT_OUTPUT_FORMAT
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    overwrite: bool = False
    metadata_extra: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", validate_model_id(self.model_id))
        object.__setattr__(self, "prompt", validate_prompt(self.prompt))
        object.__setattr__(self, "output_root", validate_output_root(self.output_root))
        object.__setattr__(self, "width", validate_dimension("width", self.width))
        object.__setattr__(self, "height", validate_dimension("height", self.height))
        object.__setattr__(self, "aspect_ratio", normalize_aspect_ratio(self.aspect_ratio))
        validate_dimensions_match_aspect_ratio(self.width, self.height, self.aspect_ratio)
        object.__setattr__(
            self,
            "guidance_scale",
            validate_float("guidance_scale", self.guidance_scale, minimum=0.0),
        )
        steps = self.num_inference_steps if self.num_inference_steps is not None else self.steps
        steps = validate_int("num_inference_steps", steps, minimum=1)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "num_inference_steps", steps)
        object.__setattr__(self, "seed", validate_seed(self.seed))
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(self, "execution_mode", normalize_execution_mode(self.execution_mode))

        provider_settings = self.provider_settings
        if isinstance(provider_settings, Mapping):
            provider_settings = ProviderSettings.from_dict(provider_settings)
        elif provider_settings is None:
            provider_settings = ProviderSettings()
        elif not isinstance(provider_settings, ProviderSettings):
            raise TypeError("provider_settings must be ProviderSettings, a mapping, or None")

        provider = self.provider
        if isinstance(provider, ProviderSettings):
            provider_settings = provider
            provider_name = provider_settings.hf_provider
        elif isinstance(provider, Mapping):
            provider_settings = ProviderSettings.from_dict(provider)
            provider_name = provider_settings.hf_provider
        elif provider is None:
            provider_name = provider_settings.hf_provider
        else:
            provider_name = validate_optional_name("provider", provider)

        if provider_name != provider_settings.hf_provider:
            provider_settings = replace(provider_settings, hf_provider=provider_name)
        object.__setattr__(self, "provider", provider_name)
        object.__setattr__(self, "provider_settings", provider_settings)

        save = self.save
        if isinstance(save, Mapping):
            save = SaveFlags.from_dict(save)
        elif save is None:
            save = SaveFlags()
        elif not isinstance(save, SaveFlags):
            raise TypeError("save must be SaveFlags, a mapping, or None")

        if self.save_latents is not None:
            save = replace(save, save_latents=bool(self.save_latents))
        if self.save_decoded_latents is not None:
            save = replace(save, save_decoded_latents=bool(self.save_decoded_latents))
        overwrite = bool(self.overwrite or save.overwrite_existing)
        if overwrite != save.overwrite_existing:
            save = replace(save, overwrite_existing=overwrite)
        object.__setattr__(self, "save", save)
        object.__setattr__(self, "save_latents", save.save_latents)
        object.__setattr__(self, "save_decoded_latents", save.save_decoded_latents)
        object.__setattr__(self, "overwrite", overwrite)
        object.__setattr__(self, "output_format", validate_output_format(self.output_format))
        object.__setattr__(self, "extra_body", dict(self.extra_body or {}))

        run_id = self.run_id or make_run_id(seed=self.seed)
        object.__setattr__(self, "run_id", validate_run_id(run_id))
        object.__setattr__(self, "metadata_extra", dict(self.metadata_extra or {}))
        object.__setattr__(self, "created_at", validate_timestamp(self.created_at))

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def run_dir(self) -> Path:
        return self.output_root / str(self.run_id)

    @property
    def prompt_path(self) -> Path:
        return self.run_dir / "prompt.txt"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def latents_dir(self) -> Path:
        return self.run_dir / "latents"

    @property
    def decoded_latents_dir(self) -> Path:
        return self.run_dir / "decoded_latents"

    @property
    def final_image_dir(self) -> Path:
        return self.run_dir / "final image"

    @property
    def final_image_path(self) -> Path:
        return self.final_image_dir / f"image.{normalized_output_extension(self.output_format)}"

    @property
    def latent_capture_supported(self) -> bool:
        return self.execution_mode.value in _LOCAL_LATENT_MODES

    def unsupported_save_flags(self) -> Tuple[str, ...]:
        if self.latent_capture_supported:
            return ()

        unsupported = []
        if self.save.save_latents:
            unsupported.append("save_latents")
        if self.save.save_decoded_latents:
            unsupported.append("save_decoded_latents")
        return tuple(unsupported)

    def planned_output_paths(self) -> JsonDict:
        return json_ready(
            {
                "run_dir": self.run_dir,
                "prompt": self.prompt_path,
                "metadata": self.metadata_path,
                "latents_dir": self.latents_dir,
                "decoded_latents_dir": self.decoded_latents_dir,
                "final_image_dir": self.final_image_dir,
                "final_image": self.final_image_path,
            }
        )

    def generation_kwargs(self) -> JsonDict:
        data = {
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.seed is not None:
            data["seed"] = self.seed
        return data

    def to_dict(self) -> JsonDict:
        return json_ready(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "model_id": self.model_id,
                "prompt": self.prompt,
                "output_root": self.output_root,
                "run_id": self.run_id,
                "width": self.width,
                "height": self.height,
                "aspect_ratio": self.aspect_ratio,
                "guidance_scale": self.guidance_scale,
                "steps": self.steps,
                "num_inference_steps": self.num_inference_steps,
                "seed": self.seed,
                "dtype": self.dtype,
                "execution_mode": self.execution_mode,
                "provider": self.provider,
                "provider_settings": self.provider_settings,
                "save": self.save,
                "save_latents": self.save_latents,
                "save_decoded_latents": self.save_decoded_latents,
                "output_format": self.output_format,
                "extra_body": self.extra_body,
                "overwrite": self.overwrite,
                "metadata_extra": self.metadata_extra,
                "created_at": self.created_at,
            }
        )

    def to_metadata(self, extra: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> JsonDict:
        metadata = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "model": {
                "model_id": self.model_id,
                "dtype": self.dtype,
                "execution_mode": self.execution_mode,
            },
            "prompt": self.prompt,
            "generation": {
                "width": self.width,
                "height": self.height,
                "aspect_ratio": self.aspect_ratio,
                "guidance_scale": self.guidance_scale,
                "steps": self.steps,
                "num_inference_steps": self.num_inference_steps,
                "seed": self.seed,
                "output_format": self.output_format,
            },
            "provider": {
                "name": self.provider,
                "settings": self.provider_settings,
                "extra_body": self.extra_body,
            },
            "save": self.save,
            "paths": self.planned_output_paths(),
            "unsupported_save_flags": self.unsupported_save_flags(),
        }

        merged_extra = dict(self.metadata_extra)
        if extra:
            merged_extra.update(extra)
        if merged_extra:
            metadata["extra"] = merged_extra
        if kwargs:
            metadata.update(kwargs)
        return json_ready(metadata)

    def metadata_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_metadata(), indent=indent, sort_keys=True) + "\n"

    def with_overrides(self, **kwargs: Any) -> "FluxImageGenConfig":
        return replace(self, **kwargs)

    def validate(self) -> "FluxImageGenConfig":
        return self

    def resolved_run_id(self) -> str:
        return str(self.run_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FluxImageGenConfig":
        values = dict(data)
        values.pop("schema_version", None)
        return cls(**values)


GenerationConfig = FluxImageGenConfig


def make_run_id(
    prefix: str = DEFAULT_RUN_ID_PREFIX,
    *,
    seed: Optional[int] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    timestamp = timestamp or datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    suffix = f"_seed{seed}" if seed is not None else ""
    return validate_run_id(f"{sanitize_run_id(prefix)}_{stamp}{suffix}")


def sanitize_run_id(value: str) -> str:
    text = str(value).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or DEFAULT_RUN_ID_PREFIX


def validate_model_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("model_id must be a string")
    text = value.strip()
    if not text:
        raise ValueError("model_id cannot be empty")
    if any(char.isspace() for char in text):
        raise ValueError("model_id cannot contain whitespace")
    return text


def validate_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt must be a string")
    text = value.strip()
    if not text:
        raise ValueError("prompt cannot be empty")
    return text


def validate_output_root(value: Any) -> Path:
    path = Path(value).expanduser()
    if not str(path):
        raise ValueError("output_root cannot be empty")
    return path


def validate_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    text = value.strip()
    if not text:
        raise ValueError("run_id cannot be empty")
    if "/" in text or "\\" in text:
        raise ValueError("run_id cannot contain path separators")
    if not _RUN_ID_RE.match(text):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    return text


def validate_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("created_at must be an ISO timestamp string")
    text = value.strip()
    if not text:
        raise ValueError("created_at cannot be empty")
    return text


def validate_dimension(name: str, value: int, *, multiple_of: int = 8) -> int:
    number = validate_int(name, value, minimum=64, maximum=4096)
    if number % multiple_of != 0:
        raise ValueError(f"{name} must be divisible by {multiple_of}")
    return number


def validate_dimensions_match_aspect_ratio(width: int, height: int, aspect_ratio: str) -> None:
    ratio_width, ratio_height = parse_aspect_ratio(aspect_ratio)
    if width * ratio_height != height * ratio_width:
        raise ValueError(
            f"dimensions {width}x{height} do not match aspect_ratio {aspect_ratio}"
        )


def parse_aspect_ratio(value: str) -> Tuple[int, int]:
    if not isinstance(value, str):
        raise TypeError("aspect_ratio must be a string like '2:3'")
    text = value.strip().lower().replace("x", ":")
    if ":" not in text:
        raise ValueError("aspect_ratio must use ':' or 'x', for example '2:3'")
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("aspect_ratio must have exactly two parts")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValueError("aspect_ratio parts must be positive integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError("aspect_ratio parts must be positive integers")
    divisor = math.gcd(width, height)
    return width // divisor, height // divisor


def normalize_aspect_ratio(value: str) -> str:
    width, height = parse_aspect_ratio(value)
    return f"{width}:{height}"


def dimensions_from_aspect_ratio(
    aspect_ratio: str,
    *,
    long_edge: int = DEFAULT_HEIGHT,
    multiple_of: int = 8,
) -> Tuple[int, int]:
    ratio_width, ratio_height = parse_aspect_ratio(aspect_ratio)
    long_edge = validate_dimension("long_edge", long_edge, multiple_of=multiple_of)
    if ratio_width >= ratio_height:
        width = long_edge
        height = round_to_multiple(long_edge * ratio_height / ratio_width, multiple_of)
    else:
        height = long_edge
        width = round_to_multiple(long_edge * ratio_width / ratio_height, multiple_of)
    return (
        validate_dimension("width", width, multiple_of=multiple_of),
        validate_dimension("height", height, multiple_of=multiple_of),
    )


def round_to_multiple(value: float, multiple_of: int) -> int:
    return int(round(value / multiple_of) * multiple_of)


def validate_int(
    name: str,
    value: Any,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def validate_float(
    name: str,
    value: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def validate_optional_float(
    name: str,
    value: Optional[Any],
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    return validate_float(name, value, minimum=minimum, maximum=maximum)


def validate_optional_name(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    return text


def validate_seed(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return validate_int("seed", value, minimum=0, maximum=2**63 - 1)


def normalize_dtype(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("dtype must be a string")
    text = value.strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "none": "auto",
    }
    text = aliases.get(text, text)
    if text not in _SUPPORTED_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_DTYPES))
        raise ValueError(f"dtype must be one of: {supported}")
    return text


def normalize_execution_mode(value: Any) -> ExecutionMode:
    if isinstance(value, ExecutionMode):
        return value
    if not isinstance(value, str):
        raise TypeError("execution_mode must be a string or ExecutionMode")
    text = value.strip().lower().replace("-", "_")
    aliases = {
        "diffusers": ExecutionMode.LOCAL_DIFFUSERS,
        "local": ExecutionMode.LOCAL_DIFFUSERS,
        "hf": ExecutionMode.HF_INFERENCE,
        "huggingface": ExecutionMode.HF_INFERENCE,
        "hf_fal": ExecutionMode.HF_FAL_PROVIDER,
        "fal_hf": ExecutionMode.HF_FAL_PROVIDER,
        "fal": ExecutionMode.FAL_PROVIDER,
        "fal_ai": ExecutionMode.FAL_PROVIDER,
        "fal_provider_dry_run": ExecutionMode.DRY_RUN,
        "hf_fal_provider_dry_run": ExecutionMode.DRY_RUN,
        "custom_fal": ExecutionMode.FAL_CUSTOM,
        "dry": ExecutionMode.DRY_RUN,
    }
    if text in aliases:
        return aliases[text]
    try:
        return ExecutionMode(text)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in ExecutionMode)
        raise ValueError(f"execution_mode must be one of: {supported}") from exc


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return str(value)


def validate_output_format(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_format must be a string")
    text = value.strip().lower()
    if text == "jpg":
        text = "jpeg"
    if text not in {"png", "jpeg"}:
        raise ValueError("output_format must be 'png', 'jpeg', or 'jpg'")
    return text


def normalized_output_extension(value: str) -> str:
    text = validate_output_format(value)
    return "jpg" if text == "jpeg" else text


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_DTYPE",
    "DEFAULT_FAL_ENDPOINT_ID",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_HEIGHT",
    "DEFAULT_HF_PROVIDER",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_FORMAT",
    "DEFAULT_PROMPT",
    "DEFAULT_RUN_ID_PREFIX",
    "DEFAULT_STEPS",
    "DEFAULT_WIDTH",
    "GUIDANCE_SCALE",
    "HEIGHT",
    "ExecutionMode",
    "FluxImageGenConfig",
    "GenerationConfig",
    "JsonDict",
    "MODEL_ID",
    "NUM_INFERENCE_STEPS",
    "OUTPUT_ROOT",
    "PROMPT",
    "ProviderSettings",
    "SaveFlags",
    "WIDTH",
    "dimensions_from_aspect_ratio",
    "json_ready",
    "make_run_id",
    "normalized_output_extension",
    "normalize_aspect_ratio",
    "normalize_dtype",
    "normalize_execution_mode",
    "parse_aspect_ratio",
    "sanitize_run_id",
    "utc_timestamp",
    "validate_dimension",
    "validate_dimensions_match_aspect_ratio",
    "validate_float",
    "validate_int",
    "validate_model_id",
    "validate_output_root",
    "validate_output_format",
    "validate_prompt",
    "validate_run_id",
    "validate_seed",
]
