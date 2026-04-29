"""
Standalone OpenRouter Client

A simple client for OpenRouter's unified AI API that provides intelligent
provider routing across multiple model providers with built-in defaults and
easy parameter overrides.

References:
    - Parameters: https://openrouter.ai/docs/api/reference/parameters
    - Provider Routing: https://openrouter.ai/docs/guides/routing/provider-selection
"""

import os
import inspect
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return False


load_dotenv()

# Note: We use requests directly instead of the OpenAI SDK to avoid import hang issues
# The OpenAI SDK can hang on import in some environments due to httpx/HTTP2 issues

# Lazy imports to avoid any module-level execution
_yaml = None
_requests = None


def _get_yaml():
    """Lazy import of yaml."""
    global _yaml
    if _yaml is None:
        try:
            import yaml

            _yaml = yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml package is required. Install with: pip install pyyaml"
            ) from exc
    return _yaml


def _get_requests():
    """Lazy import of requests."""
    global _requests
    if _requests is None:
        try:
            import requests

            _requests = requests
        except ImportError as exc:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            ) from exc
    return _requests


class ProviderPreferences(TypedDict, total=False):
    """
    OpenRouter provider routing configuration.

    Controls how requests are routed across providers for optimal cost,
    performance, and reliability.

    Reference: https://openrouter.ai/docs/guides/routing/provider-selection
    """

    order: List[str]
    allow_fallbacks: bool
    require_parameters: bool
    data_collection: Literal["allow", "deny"]
    zdr: bool
    enforce_distillable_text: bool
    only: List[str]
    ignore: List[str]
    quantizations: List[str]
    sort: Literal["price", "throughput", "latency"]
    max_price: Dict[str, float]


class OpenRouterClient:
    """
    Standalone client for OpenRouter's unified AI API.

    Provides intelligent routing across multiple AI providers (OpenAI, Anthropic,
    Google, etc.) with support for cost optimization, latency preferences, and
    privacy controls.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY"),
        timeout: Optional[float] = 30.0,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 0,
        frequency_penalty: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        repetition_penalty: Optional[float] = 1.0,
        min_p: Optional[float] = 0.0,
        top_a: Optional[float] = 0.0,
        seed: Optional[int] = 42069,
        max_tokens: Optional[int] = 5120,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        verbosity: Optional[Literal["low", "medium", "high"]] = "medium",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = True,
        provider: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.timeout = timeout
        self._defaults = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
            "top_a": top_a,
            "seed": seed,
            "max_tokens": max_tokens,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "verbosity": verbosity,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "reasoning": reasoning if reasoning is not None else {"effort": "medium"},
        }
        self._default_provider = self._build_provider_config(provider)

    def _build_provider_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        config = {
            "require_parameters": False,
            "allow_fallbacks": True,
            "quantizations": ["fp16", "bf16", "fp8"],
            "data_collection": "deny",
        }
        if user_config:
            config.update(user_config)
        return config

    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        session: Optional[Any] = None,
        **override_params,
    ):
        params = self._build_request_params(**override_params)
        params["model"] = self.model
        params["messages"] = messages
        params["stream"] = stream

        requester = session if session is not None else _get_requests()
        response = requester.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _build_request_params(self, **override_params) -> Dict[str, Any]:
        params = {}
        for key, value in self._defaults.items():
            if value is not None and key not in override_params:
                params[key] = value

        params.update(override_params)

        provider_config = dict(self._default_provider)
        if "provider" in override_params:
            provider_config.update(override_params["provider"])
        params["provider"] = provider_config

        return {key: value for key, value in params.items() if value is not None}


def load_model_deployments(yaml_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load model deployments from YAML file.
    """
    if yaml_path is None:
        current_dir = Path(__file__).parent.parent
        yaml_path = current_dir / "prod_env" / "model_deployments.yaml"

    yaml = _get_yaml()
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Model deployments file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        raise ValueError(f"Invalid YAML file: {yaml_path} is empty or contains no data")
    if "models" not in data:
        raise ValueError(f"Invalid YAML structure: expected 'models' key in {yaml_path}")

    configs: Dict[str, Dict[str, Any]] = {}
    for model_config in data["models"]:
        if "name" not in model_config:
            raise ValueError(f"Model config missing 'name' field: {model_config}")
        if "model" not in model_config:
            raise ValueError(f"Model config missing 'model' field: {model_config}")

        client_name = model_config.get("client")
        backend_name = model_config.get("backend")
        if client_name not in {None, "openrouter"}:
            continue
        if backend_name not in {None, "openrouter"}:
            continue

        name = model_config["name"]
        configs[name] = model_config

    return configs


def load_model_config(name: str, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    configs = load_model_deployments(yaml_path)
    if name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Model '{name}' not found. Available models: {available}")
    return configs[name]


def get_model_client(
    name: str,
    api_key: Optional[str] = None,
    yaml_path: Optional[str] = None,
    **override_params,
) -> OpenRouterClient:
    """
    Get a configured OpenRouterClient instance from YAML configuration.
    Falls back to treating `name` as a raw OpenRouter model ID when no alias exists.
    """
    try:
        config = dict(load_model_config(name, yaml_path))
        model = config.pop("model")
        provider = config.pop("provider", None)
        config.pop("name", None)
        config.pop("backend", None)
        config.pop("client", None)
        config.pop("fal", None)

        allowed = set(inspect.signature(OpenRouterClient.__init__).parameters)
        allowed.discard("self")
        client_params = {
            key: value for key, value in {**config, **override_params}.items() if key in allowed
        }
        if provider is not None and "provider" not in client_params:
            client_params["provider"] = provider
    except (ValueError, FileNotFoundError):
        model = name
        allowed = set(inspect.signature(OpenRouterClient.__init__).parameters)
        allowed.discard("self")
        client_params = {
            key: value for key, value in override_params.items() if key in allowed
        }

    return OpenRouterClient(model=model, api_key=api_key, **client_params)
