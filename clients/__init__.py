"""
Public client exports for the repository harnesses.
"""

from importlib import import_module

from clients.openrouter_client import (
    OpenRouterClient,
    ProviderPreferences,
    load_model_deployments,
    load_model_config,
    get_model_client,
)

_GEMINI_EXPORT_MAP = {
    "GeminiClient": ("GeminiClient",),
    "get_gemini_client": ("get_gemini_client", "get_model_client"),
    "load_gemini_model_deployments": (
        "load_gemini_model_deployments",
        "load_model_deployments",
    ),
    "load_gemini_model_config": ("load_gemini_model_config", "load_model_config"),
}

_FAL_EXPORT_MAP = {
    "FALClient": ("FALClient",),
    "get_fal_client": ("get_fal_client", "get_model_client"),
    "load_fal_model_deployments": ("load_fal_model_deployments", "load_model_deployments"),
    "load_fal_model_config": ("load_fal_model_config", "load_model_config"),
}

__all__ = [
    "OpenRouterClient",
    "ProviderPreferences",
    "load_model_deployments",
    "load_model_config",
    "get_model_client",
    "GeminiClient",
    "get_gemini_client",
    "load_gemini_model_deployments",
    "load_gemini_model_config",
    "FALClient",
    "get_fal_client",
    "load_fal_model_deployments",
    "load_fal_model_config",
]


def _load_gemini_module():
    try:
        return import_module("clients.gemini_client")
    except ModuleNotFoundError as exc:
        if exc.name != "clients.gemini_client":
            raise
        return None


def _load_fal_module():
    try:
        return import_module("clients.fal_client")
    except ModuleNotFoundError as exc:
        if exc.name != "clients.fal_client":
            raise
        return None


def _resolve_gemini_export(name: str):
    module = _load_gemini_module()
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            "clients.gemini_client is not available in this checkout"
        )

    for candidate in _GEMINI_EXPORT_MAP[name]:
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise AttributeError(
        f"module 'clients.gemini_client' does not define a compatible export for {name!r}"
    )


def _resolve_fal_export(name: str):
    module = _load_fal_module()
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            "clients.fal_client is not available in this checkout"
        )

    for candidate in _FAL_EXPORT_MAP[name]:
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise AttributeError(
        f"module 'clients.fal_client' does not define a compatible export for {name!r}"
    )


class GeminiClient:
    def __new__(cls, *args, **kwargs):
        target = _resolve_gemini_export("GeminiClient")
        return target(*args, **kwargs)


def get_gemini_client(*args, **kwargs):
    target = _resolve_gemini_export("get_gemini_client")
    return target(*args, **kwargs)


def load_gemini_model_deployments(*args, **kwargs):
    target = _resolve_gemini_export("load_gemini_model_deployments")
    return target(*args, **kwargs)


def load_gemini_model_config(*args, **kwargs):
    target = _resolve_gemini_export("load_gemini_model_config")
    return target(*args, **kwargs)


class FALClient:
    def __new__(cls, *args, **kwargs):
        target = _resolve_fal_export("FALClient")
        return target(*args, **kwargs)


def get_fal_client(*args, **kwargs):
    target = _resolve_fal_export("get_fal_client")
    return target(*args, **kwargs)


def load_fal_model_deployments(*args, **kwargs):
    target = _resolve_fal_export("load_fal_model_deployments")
    return target(*args, **kwargs)


def load_fal_model_config(*args, **kwargs):
    target = _resolve_fal_export("load_fal_model_config")
    return target(*args, **kwargs)


def __dir__():
    return sorted(set(globals()) | set(__all__))
