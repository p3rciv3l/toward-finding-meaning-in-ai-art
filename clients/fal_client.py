"""
Standalone FAL AI client.

This harness mirrors the repository's OpenRouter/Gemini pattern:
- constructor defaults live in code
- YAML deployments override those defaults
- per-call kwargs override both

The client composes the official `fal_client` Python SDK for direct run/stream/
realtime/upload flows, while using plain HTTP for queue-specific controls that
the SDK does not expose directly, such as `fal_max_queue_length`.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv():
        return False

load_dotenv()

_yaml = None
_requests = None
_fal_sdk = None
_crypto_ed25519 = None
_crypto_serialization = None

JsonDict = Dict[str, Any]

def _get_yaml():
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


def _get_fal_sdk():
    global _fal_sdk
    if _fal_sdk is None:
        try:
            import fal_client

            _fal_sdk = fal_client
        except ImportError as exc:
            raise ImportError(
                "fal-client package is required. Install with: pip install fal-client"
            ) from exc
    return _fal_sdk


def _get_crypto():
    global _crypto_ed25519, _crypto_serialization
    if _crypto_ed25519 is None or _crypto_serialization is None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            _crypto_ed25519 = Ed25519PublicKey
            _crypto_serialization = serialization
        except ImportError as exc:
            raise ImportError(
                "cryptography package is required for webhook verification. "
                "Install with: pip install cryptography"
            ) from exc
    return _crypto_ed25519, _crypto_serialization


def _coerce_fal_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key

    env_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if env_key:
        return env_key

    key_id = os.getenv("FAL_KEY_ID")
    key_secret = os.getenv("FAL_KEY_SECRET")
    if key_id and key_secret:
        return f"{key_id}:{key_secret}"

    return None


def _strip_slashes(value: str) -> str:
    return value.strip("/")


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    return value if value.startswith("/") else f"/{value}"


def _bool_header(value: bool) -> str:
    return "1" if value else "0"


@dataclass
class FALResponse:
    data: Any
    request_id: Optional[str] = None
    status_code: Optional[int] = None
    endpoint_id: Optional[str] = None
    mode: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return {
            "data": self.data,
            "request_id": self.request_id,
            "status_code": self.status_code,
            "endpoint_id": self.endpoint_id,
            "mode": self.mode,
            "headers": dict(self.headers),
            "url": self.url,
        }


class FALClientError(Exception):
    def __init__(
        self,
        message: str,
        *,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        endpoint_id: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        response_body: Any = None,
        error_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.request_id = request_id
        self.status_code = status_code
        self.endpoint_id = endpoint_id
        self.headers = dict(headers or {})
        self.response_body = response_body
        self.error_type = error_type


class FALRequestError(FALClientError):
    pass


class FALModelError(FALClientError):
    def __init__(self, message: str, *, detail: Optional[List[JsonDict]] = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.detail = detail or []


class FALTransportError(FALClientError):
    pass


class FALWebhookVerificationError(FALClientError):
    pass


class FALClient:
    """
    General-purpose FAL client covering text, image, audio, video, and mixed I/O.

    Transport controls are explicit constructor/per-call parameters. Model-specific
    arguments remain modality-agnostic and are merged from:
    OpenAPI defaults < constructor defaults < per-call explicit values.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        default_timeout: float = 120.0,
        timeout: Optional[float] = None,
        use_batch_api: bool = True,
        batch_model: Optional[str] = None,
        request_mode: str = "submit",
        path: str = "",
        stream_path: str = "/stream",
        realtime_path: str = "/realtime",
        start_timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        hint: Optional[str] = None,
        priority: str = "normal",
        webhook_url: Optional[str] = None,
        with_logs: bool = False,
        poll_interval: float = 0.1,
        queue_status_mode: str = "polling",
        fal_max_queue_length: Optional[int] = None,
        disable_retries: bool = False,
        disable_fallback: bool = False,
        store_io: Optional[bool] = True,
        object_lifecycle_preference: Optional[JsonDict] = None,
        headers: Optional[JsonDict] = None,
        upload_repository: str = "fal_v3",
        upload_fallback_repository: Optional[Tuple[str, ...]] = ("cdn", "fal"),
        upload_image_format: str = "jpeg",
        use_jwt: bool = True,
        token_expiration: int = 120,
        max_buffering: Optional[int] = None,
        use_openapi_defaults: bool = True,
        webhook_verify: bool = True,
        webhook_jwks_url: str = "https://rest.fal.ai/.well-known/jwks.json",
        webhook_timestamp_leeway_seconds: int = 300,
        webhook_jwks_cache_seconds: int = 86400,
        prompt: Optional[str] = None,
        input: Optional[Any] = None,
        text: Optional[str] = None,
        texts: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        model_arg: Optional[str] = None,
        seed: Optional[int] = None,
        num_images: Optional[int] = None,
        image_size: Optional[Any] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        duration: Optional[Any] = None,
        negative_prompt: Optional[str] = None,
        enable_safety_checker: Optional[bool] = None,
        enable_safety_checks: Optional[bool] = None,
        enable_prompt_expansion: Optional[bool] = None,
        expand_prompt: Optional[bool] = None,
        output_format: Optional[str] = None,
        sync_mode: Optional[bool] = None,
        limit_generations: Optional[bool] = None,
        enable_web_search: Optional[bool] = None,
        thinking_level: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        sample_rate: Optional[str] = None,
        voice: Optional[str] = None,
        temperature_audio: Optional[float] = None,
        temperature_text: Optional[float] = None,
        top_k_audio: Optional[int] = None,
        top_k_text: Optional[int] = None,
        num_extra_steps: Optional[int] = None,
        acoustic_cfg_scale: Optional[float] = None,
        noise_temperature: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        speed_up_factor: Optional[float] = None,
        audio_url: Optional[str] = None,
        video_url: Optional[str] = None,
        image_url: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        images_data_url: Optional[str] = None,
        transcript: Optional[str] = None,
        language: Optional[str] = None,
        language_code: Optional[str] = None,
        use_pnc: Optional[bool] = None,
        tag_audio_events: Optional[bool] = None,
        diarize: Optional[bool] = None,
        txt_color: Optional[str] = None,
        txt_font: Optional[str] = None,
        font_size: Optional[int] = None,
        stroke_width: Optional[int] = None,
        left_align: Optional[int] = None,
        top_align: Optional[int] = None,
        refresh_interval: Optional[int] = None,
        audio: Optional[Any] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        file_url: Optional[str] = None,
        request_overrides: Optional[JsonDict] = None,
    ):
        self._api_key = _coerce_fal_key(api_key)
        if not self._api_key:
            raise ValueError(
                "FAL API key not provided. Set FAL_KEY or FAL_KEY_ID/FAL_KEY_SECRET, "
                "or pass api_key explicitly."
            )

        self.model = model
        self.default_timeout = default_timeout
        self.timeout = timeout if timeout is not None else default_timeout
        self.use_batch_api = use_batch_api
        self.batch_model = batch_model
        self.request_mode = request_mode
        self.path = _normalize_path(path)
        self.stream_path = _normalize_path(stream_path)
        self.realtime_path = _normalize_path(realtime_path)
        self.start_timeout = start_timeout
        self.client_timeout = client_timeout
        self.hint = hint
        self.priority = priority
        self.webhook_url = webhook_url
        self.with_logs = with_logs
        self.poll_interval = poll_interval
        self.queue_status_mode = queue_status_mode
        self.fal_max_queue_length = fal_max_queue_length
        self.disable_retries = disable_retries
        self.disable_fallback = disable_fallback
        self.store_io = store_io
        self.object_lifecycle_preference = object_lifecycle_preference
        self.headers = dict(headers or {})
        self.upload_repository = upload_repository
        self.upload_fallback_repository = upload_fallback_repository
        self.upload_image_format = upload_image_format
        self.use_jwt = use_jwt
        self.token_expiration = token_expiration
        self.max_buffering = max_buffering
        self.use_openapi_defaults = use_openapi_defaults
        self.webhook_verify = webhook_verify
        self.webhook_jwks_url = webhook_jwks_url
        self.webhook_timestamp_leeway_seconds = webhook_timestamp_leeway_seconds
        self.webhook_jwks_cache_seconds = webhook_jwks_cache_seconds
        self._request_overrides = dict(request_overrides or {})
        self._schema_cache: Dict[str, JsonDict] = {}
        self._jwks_cache: Dict[str, Tuple[float, JsonDict]] = {}
        self._request_url_cache: Dict[str, Dict[str, str]] = {}

        self._model_defaults: JsonDict = {
            "prompt": prompt,
            "input": input,
            "text": text,
            "texts": texts,
            "prompts": prompts,
            "model": model_arg,
            "seed": seed,
            "num_images": num_images,
            "image_size": image_size,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "negative_prompt": negative_prompt,
            "enable_safety_checker": enable_safety_checker,
            "enable_safety_checks": enable_safety_checks,
            "enable_prompt_expansion": enable_prompt_expansion,
            "expand_prompt": expand_prompt,
            "output_format": output_format,
            "sync_mode": sync_mode,
            "limit_generations": limit_generations,
            "enable_web_search": enable_web_search,
            "thinking_level": thinking_level,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "sample_rate": sample_rate,
            "voice": voice,
            "temperature_audio": temperature_audio,
            "temperature_text": temperature_text,
            "top_k_audio": top_k_audio,
            "top_k_text": top_k_text,
            "num_extra_steps": num_extra_steps,
            "acoustic_cfg_scale": acoustic_cfg_scale,
            "noise_temperature": noise_temperature,
            "num_inference_steps": num_inference_steps,
            "speed_up_factor": speed_up_factor,
            "audio_url": audio_url,
            "video_url": video_url,
            "image_url": image_url,
            "image_urls": image_urls,
            "images_data_url": images_data_url,
            "transcript": transcript,
            "language": language,
            "language_code": language_code,
            "use_pnc": use_pnc,
            "tag_audio_events": tag_audio_events,
            "diarize": diarize,
            "txt_color": txt_color,
            "txt_font": txt_font,
            "font_size": font_size,
            "stroke_width": stroke_width,
            "left_align": left_align,
            "top_align": top_align,
            "refresh_interval": refresh_interval,
            "audio": audio,
            "image": image,
            "video": video,
            "file_url": file_url,
        }

        sdk = _get_fal_sdk()
        self._sdk = sdk.SyncClient(key=self._api_key, default_timeout=self.timeout)

    def invoke(
        self,
        arguments: Optional[JsonDict] = None,
        *,
        input: Optional[Any] = None,
        request_mode: Optional[str] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        start_timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        hint: Optional[str] = None,
        priority: Optional[str] = None,
        webhook_url: Optional[str] = None,
        with_logs: Optional[bool] = None,
        poll_interval: Optional[float] = None,
        on_enqueue: Optional[Callable[[FALResponse], None]] = None,
        on_queue_update: Optional[Callable[[FALResponse], None]] = None,
        headers: Optional[JsonDict] = None,
        fal_max_queue_length: Optional[int] = None,
        disable_retries: Optional[bool] = None,
        disable_fallback: Optional[bool] = None,
        store_io: Optional[bool] = None,
        object_lifecycle_preference: Optional[JsonDict] = None,
        use_openapi_defaults: Optional[bool] = None,
        **model_params: Any,
    ) -> Any:
        mode = (request_mode or self.request_mode).lower()
        call_args = self._build_arguments(
            arguments=arguments,
            input_value=input,
            use_openapi_defaults=use_openapi_defaults,
            use_batch_api=use_batch_api,
            batch_model=batch_model,
            **model_params,
        )

        common = {
            "path": path,
            "timeout": timeout,
            "start_timeout": start_timeout,
            "client_timeout": client_timeout,
            "hint": hint,
            "priority": priority,
            "webhook_url": webhook_url,
            "with_logs": with_logs,
            "poll_interval": poll_interval,
            "headers": headers,
            "fal_max_queue_length": fal_max_queue_length,
            "disable_retries": disable_retries,
            "disable_fallback": disable_fallback,
            "store_io": store_io,
            "object_lifecycle_preference": object_lifecycle_preference,
        }

        if mode == "run":
            return self.run(call_args, **common)
        if mode == "submit":
            return self.submit(call_args, **common)
        if mode == "subscribe":
            return self.subscribe(
                call_args,
                on_enqueue=on_enqueue,
                on_queue_update=on_queue_update,
                **common,
            )
        if mode == "stream":
            return self.stream(call_args, path=path, timeout=timeout)
        if mode == "realtime":
            return self.realtime()
        raise ValueError(
            "Unsupported request_mode. Expected one of: run, submit, subscribe, stream, realtime."
        )

    def run(
        self,
        arguments: JsonDict,
        *,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        start_timeout: Optional[float] = None,
        hint: Optional[str] = None,
        headers: Optional[JsonDict] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        disable_retries: Optional[bool] = None,
        disable_fallback: Optional[bool] = None,
        store_io: Optional[bool] = None,
        object_lifecycle_preference: Optional[JsonDict] = None,
    ) -> FALResponse:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        request_headers = self._build_request_headers(
            headers=headers,
            disable_retries=disable_retries,
            disable_fallback=disable_fallback,
            store_io=store_io,
            object_lifecycle_preference=object_lifecycle_preference,
            hint=hint,
            start_timeout=start_timeout,
            priority=None,
            include_priority=False,
        )
        try:
            data = self._sdk.run(
                endpoint_id,
                arguments=arguments,
                path=_normalize_path(path) if path is not None else self.path,
                timeout=timeout if timeout is not None else self.timeout,
                start_timeout=start_timeout if start_timeout is not None else self.start_timeout,
                hint=hint if hint is not None else self.hint,
                headers=request_headers,
            )
        except Exception as exc:
            raise self._normalize_exception(exc, endpoint_id=endpoint_id) from exc
        return FALResponse(data=data, endpoint_id=endpoint_id, mode="run")

    def submit(
        self,
        arguments: JsonDict,
        *,
        path: Optional[str] = None,
        start_timeout: Optional[float] = None,
        hint: Optional[str] = None,
        priority: Optional[str] = None,
        webhook_url: Optional[str] = None,
        headers: Optional[JsonDict] = None,
        fal_max_queue_length: Optional[int] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        disable_retries: Optional[bool] = None,
        disable_fallback: Optional[bool] = None,
        store_io: Optional[bool] = None,
        object_lifecycle_preference: Optional[JsonDict] = None,
        timeout: Optional[float] = None,
        **_: Any,
    ) -> FALResponse:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        request_headers = self._build_request_headers(
            headers=headers,
            disable_retries=disable_retries,
            disable_fallback=disable_fallback,
            store_io=store_io,
            object_lifecycle_preference=object_lifecycle_preference,
            hint=hint,
            start_timeout=start_timeout,
            priority=priority,
        )
        params = self._build_queue_params(
            webhook_url=webhook_url,
            fal_max_queue_length=fal_max_queue_length,
        )
        response = self._request_json(
            "POST",
            self._queue_url(endpoint_id, path),
            json_body=arguments,
            params=params,
            headers=request_headers,
            timeout=timeout if timeout is not None else self.timeout,
            endpoint_id=endpoint_id,
        )
        return self._envelope(response, endpoint_id=endpoint_id, mode="submit")

    def subscribe(
        self,
        arguments: JsonDict,
        *,
        path: Optional[str] = None,
        start_timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        hint: Optional[str] = None,
        priority: Optional[str] = None,
        webhook_url: Optional[str] = None,
        with_logs: Optional[bool] = None,
        poll_interval: Optional[float] = None,
        on_enqueue: Optional[Callable[[FALResponse], None]] = None,
        on_queue_update: Optional[Callable[[FALResponse], None]] = None,
        headers: Optional[JsonDict] = None,
        fal_max_queue_length: Optional[int] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        disable_retries: Optional[bool] = None,
        disable_fallback: Optional[bool] = None,
        store_io: Optional[bool] = None,
        object_lifecycle_preference: Optional[JsonDict] = None,
        timeout: Optional[float] = None,
        **_: Any,
    ) -> FALResponse:
        effective_client_timeout = (
            client_timeout if client_timeout is not None else self.client_timeout
        )
        effective_start_timeout = start_timeout
        if effective_start_timeout is None and effective_client_timeout is not None:
            effective_start_timeout = effective_client_timeout

        submitted = self.submit(
            arguments,
            path=path,
            start_timeout=effective_start_timeout,
            hint=hint,
            priority=priority,
            webhook_url=webhook_url,
            headers=headers,
            fal_max_queue_length=fal_max_queue_length,
            use_batch_api=use_batch_api,
            batch_model=batch_model,
            disable_retries=disable_retries,
            disable_fallback=disable_fallback,
            store_io=store_io,
            object_lifecycle_preference=object_lifecycle_preference,
            timeout=timeout,
        )
        if on_enqueue is not None:
            on_enqueue(submitted)

        deadline = None
        if effective_client_timeout is not None:
            deadline = time.time() + effective_client_timeout

        interval = poll_interval if poll_interval is not None else self.poll_interval
        request_id = submitted.request_id
        if not request_id:
            raise FALTransportError(
                "Queue submission did not return a request_id.",
                endpoint_id=submitted.endpoint_id,
                response_body=submitted.data,
            )

        while True:
            if deadline is not None and time.time() > deadline:
                raise FALTransportError(
                    "Timed out waiting for the queued FAL request to complete.",
                    request_id=request_id,
                    endpoint_id=submitted.endpoint_id,
                )

            status_response = self.status(
                request_id,
                with_logs=with_logs,
                path=path,
                use_batch_api=use_batch_api,
                batch_model=batch_model,
                timeout=timeout,
            )
            if on_queue_update is not None:
                on_queue_update(status_response)

            status_value = str(status_response.data.get("status", "")).upper()
            if status_value == "COMPLETED":
                return self.result(
                    request_id,
                    path=path,
                    use_batch_api=use_batch_api,
                    batch_model=batch_model,
                    timeout=timeout,
                )
            if self._is_terminal_queue_status(status_value):
                self._raise_terminal_queue_status(status_response)
            time.sleep(interval)

    def stream(
        self,
        arguments: JsonDict,
        *,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
    ) -> Iterable[Any]:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        try:
            return self._sdk.stream(
                endpoint_id,
                arguments=arguments,
                path=_normalize_path(path) if path is not None else self.stream_path,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except Exception as exc:
            raise self._normalize_exception(exc, endpoint_id=endpoint_id) from exc

    def realtime(self, *, use_batch_api: Optional[bool] = None, batch_model: Optional[str] = None):
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        try:
            return self._sdk.realtime(
                endpoint_id,
                use_jwt=self.use_jwt,
                path=self.realtime_path,
                max_buffering=self.max_buffering,
                token_expiration=self.token_expiration,
            )
        except Exception as exc:
            raise self._normalize_exception(exc, endpoint_id=endpoint_id) from exc

    def status(
        self,
        request_id: str,
        *,
        with_logs: Optional[bool] = None,
        path: Optional[str] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> FALResponse:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        params = {"logs": "1" if (with_logs if with_logs is not None else self.with_logs) else "0"}
        response = self._request_json(
            "GET",
            self._request_url_for(
                request_id,
                endpoint_id=endpoint_id,
                path=path,
                kind="status_url",
                fallback_suffix="/status",
            ),
            params=params,
            timeout=timeout if timeout is not None else self.timeout,
            endpoint_id=endpoint_id,
        )
        return self._envelope(response, endpoint_id=endpoint_id, mode="status")

    def stream_status(
        self,
        request_id: str,
        *,
        with_logs: bool = False,
        mode: str = "polling",
        path: Optional[str] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[FALResponse]:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        if mode == "polling":
            while True:
                current = self.status(
                    request_id,
                    with_logs=with_logs,
                    path=path,
                    use_batch_api=use_batch_api,
                    batch_model=batch_model,
                    timeout=timeout,
                )
                yield current
                status_value = str(current.data.get("status", "")).upper()
                if status_value == "COMPLETED":
                    return
                if self._is_terminal_queue_status(status_value):
                    self._raise_terminal_queue_status(current)
                time.sleep(self.poll_interval)
            return

        if mode != "streaming":
            raise ValueError("mode must be 'polling' or 'streaming'")

        response = self._request(
            "GET",
            self._request_url_for(
                request_id,
                endpoint_id=endpoint_id,
                path=path,
                kind="status_url",
                fallback_suffix="/status/stream",
            ),
            params={"logs": "1" if with_logs else "0"},
            timeout=timeout if timeout is not None else self.timeout,
            stream=True,
        )
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            payload = json.loads(line)
            yield FALResponse(
                data=payload,
                request_id=request_id,
                endpoint_id=endpoint_id,
                mode="stream_status",
                headers=dict(response.headers),
                status_code=response.status_code,
                url=response.url,
            )

    def result(
        self,
        request_id: str,
        *,
        path: Optional[str] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> FALResponse:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        response = self._request_json(
            "GET",
            self._request_url_for(
                request_id,
                endpoint_id=endpoint_id,
                path=path,
                kind="response_url",
                fallback_suffix="",
            ),
            timeout=timeout if timeout is not None else self.timeout,
            endpoint_id=endpoint_id,
        )
        return self._envelope(response, endpoint_id=endpoint_id, mode="result")

    def cancel(
        self,
        request_id: str,
        *,
        path: Optional[str] = None,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> FALResponse:
        endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
        response = self._request_json(
            "PUT",
            self._request_url_for(
                request_id,
                endpoint_id=endpoint_id,
                path=path,
                kind="cancel_url",
                fallback_suffix="/cancel",
            ),
            timeout=timeout if timeout is not None else self.timeout,
            endpoint_id=endpoint_id,
        )
        return self._envelope(response, endpoint_id=endpoint_id, mode="cancel")

    def upload(
        self,
        data: Any,
        content_type: str,
        file_name: Optional[str] = None,
        *,
        repository: Optional[str] = None,
        fallback_repository: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        try:
            return self._sdk.upload(
                data,
                content_type=content_type,
                file_name=file_name,
                repository=repository or self.upload_repository,
                fallback_repository=fallback_repository or self.upload_fallback_repository,
            )
        except Exception as exc:
            raise self._normalize_exception(exc, endpoint_id=self.model) from exc

    def upload_file(
        self,
        path: str,
        *,
        repository: Optional[str] = None,
        fallback_repository: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        try:
            return self._sdk.upload_file(
                path,
                repository=repository or self.upload_repository,
                fallback_repository=fallback_repository or self.upload_fallback_repository,
            )
        except Exception as exc:
            raise self._normalize_exception(exc, endpoint_id=self.model) from exc

    def upload_image(
        self,
        image: Any,
        *,
        format: Optional[str] = None,
        repository: Optional[str] = None,
        fallback_repository: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        try:
            return self._sdk.upload_image(
                image,
                format=format or self.upload_image_format,
                repository=repository or self.upload_repository,
                fallback_repository=fallback_repository or self.upload_fallback_repository,
            )
        except Exception as exc:
            raise self._normalize_exception(exc, endpoint_id=self.model) from exc

    def inspect_model_schema(
        self,
        endpoint_id: Optional[str] = None,
        *,
        refresh: bool = False,
        include_enterprise_status: bool = True,
    ) -> JsonDict:
        target = endpoint_id or self.model
        if not refresh and target in self._schema_cache:
            return self._schema_cache[target]

        expands = ["openapi-3.0"]
        if include_enterprise_status:
            expands.append("enterprise_status")

        params: List[Tuple[str, str]] = [("endpoint_id", target)]
        for item in expands:
            params.append(("expand", item))

        response = self._request_json(
            "GET",
            "https://api.fal.ai/v1/models",
            params=params,
            timeout=self.timeout,
            endpoint_id=target,
        )
        models = response["data"].get("models", [])
        if not models:
            raise ValueError(f"No schema metadata returned for endpoint '{target}'.")
        model_data = models[0]
        self._schema_cache[target] = model_data
        return model_data

    def get_model_argument_defaults(
        self,
        endpoint_id: Optional[str] = None,
        *,
        refresh: bool = False,
    ) -> JsonDict:
        schema_record = self.inspect_model_schema(endpoint_id=endpoint_id, refresh=refresh)
        openapi = schema_record.get("openapi") or {}
        target = schema_record.get("endpoint_id") or endpoint_id or self.model
        path_key = f"/{target}"
        request_body_schema = (
            openapi.get("paths", {})
            .get(path_key, {})
            .get("post", {})
            .get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        resolved = self._resolve_openapi_schema(request_body_schema, openapi)
        properties = resolved.get("properties", {})
        defaults: JsonDict = {}
        for key, value in properties.items():
            if "default" in value:
                defaults[key] = value["default"]
        return defaults

    def delete_request_payloads(
        self,
        request_id: str,
        *,
        idempotency_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> FALResponse:
        headers = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        response = self._request_json(
            "DELETE",
            f"https://api.fal.ai/v1/models/requests/{request_id}/payloads",
            headers=headers,
            timeout=timeout if timeout is not None else self.timeout,
            endpoint_id=self.model,
        )
        return self._envelope(response, endpoint_id=self.model, mode="delete_request_payloads")

    def verify_webhook(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
        *,
        jwks_url: Optional[str] = None,
        timestamp_leeway_seconds: Optional[int] = None,
    ) -> bool:
        if not self.webhook_verify:
            return True

        request_id = headers.get("X-Fal-Webhook-Request-Id") or headers.get(
            "x-fal-webhook-request-id"
        )
        user_id = headers.get("X-Fal-Webhook-User-Id") or headers.get("x-fal-webhook-user-id")
        timestamp = headers.get("X-Fal-Webhook-Timestamp") or headers.get(
            "x-fal-webhook-timestamp"
        )
        signature = headers.get("X-Fal-Webhook-Signature") or headers.get(
            "x-fal-webhook-signature"
        )
        if not all([request_id, user_id, timestamp, signature]):
            raise FALWebhookVerificationError("Missing required FAL webhook verification headers.")

        try:
            timestamp_value = int(timestamp)
        except ValueError as exc:
            raise FALWebhookVerificationError("Invalid FAL webhook timestamp.") from exc

        leeway = (
            timestamp_leeway_seconds
            if timestamp_leeway_seconds is not None
            else self.webhook_timestamp_leeway_seconds
        )
        if abs(int(time.time()) - timestamp_value) > leeway:
            raise FALWebhookVerificationError("FAL webhook timestamp is outside the allowed window.")

        body_hash = hashlib.sha256(raw_body).hexdigest()
        message = "\n".join([request_id, user_id, timestamp, body_hash]).encode("utf-8")
        signature_bytes = self._decode_webhook_signature(signature)

        jwks = self._get_jwks(jwks_url or self.webhook_jwks_url)
        Ed25519PublicKey, serialization = _get_crypto()

        last_error: Optional[Exception] = None
        for key in jwks.get("keys", []):
            if key.get("kty") != "OKP" or key.get("crv") != "Ed25519" or "x" not in key:
                continue
            public_bytes = base64.urlsafe_b64decode(f"{key['x']}==")
            public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
            try:
                public_key.verify(signature_bytes, message)
                public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
                return True
            except Exception as exc:  # pragma: no cover - verification loop
                last_error = exc

        raise FALWebhookVerificationError(
            "FAL webhook signature verification failed.",
            response_body={"jwks_url": jwks_url or self.webhook_jwks_url},
        ) from last_error

    def _build_arguments(
        self,
        *,
        arguments: Optional[JsonDict],
        input_value: Optional[Any],
        use_openapi_defaults: Optional[bool],
        use_batch_api: Optional[bool],
        batch_model: Optional[str],
        **model_params: Any,
    ) -> JsonDict:
        merged: JsonDict = {}
        should_use_openapi_defaults = (
            use_openapi_defaults
            if use_openapi_defaults is not None
            else self.use_openapi_defaults
        )
        if should_use_openapi_defaults:
            endpoint_id = self._resolve_model(use_batch_api=use_batch_api, batch_model=batch_model)
            try:
                merged.update(self.get_model_argument_defaults(endpoint_id))
            except Exception:
                pass

        merged.update({key: value for key, value in self._model_defaults.items() if value is not None})
        merged.update({key: value for key, value in self._request_overrides.items() if value is not None})

        if arguments is not None:
            merged.update(arguments)
        if input_value is not None:
            if "input" in merged and merged["input"] is not None and merged["input"] != input_value:
                raise ValueError("Provide either arguments['input'] or input=, not conflicting values.")
            merged["input"] = input_value

        extra = {key: value for key, value in model_params.items() if value is not None}
        merged.update(extra)
        return merged

    def _resolve_model(
        self,
        *,
        use_batch_api: Optional[bool] = None,
        batch_model: Optional[str] = None,
    ) -> str:
        should_use_batch = self.use_batch_api if use_batch_api is None else use_batch_api
        configured_batch = batch_model or self.batch_model
        if not should_use_batch:
            return self.model
        if configured_batch:
            return configured_batch
        discovered = self._discover_batch_model()
        return discovered or self.model

    def _discover_batch_model(self) -> Optional[str]:
        try:
            schema = self.inspect_model_schema(self.model)
        except Exception:
            return None

        group = ((schema.get("metadata") or {}).get("group") or {}).get("key")
        if not group:
            return None

        response = self._request_json(
            "GET",
            "https://api.fal.ai/v1/models",
            params={"q": group, "limit": 100},
            timeout=self.timeout,
            endpoint_id=self.model,
        )
        for item in response["data"].get("models", []):
            endpoint_id = item.get("endpoint_id")
            item_group = ((item.get("metadata") or {}).get("group") or {}).get("key")
            if item_group != group or not endpoint_id or endpoint_id == self.model:
                continue
            if endpoint_id.endswith("/batch") or endpoint_id.endswith("/batched"):
                return endpoint_id
        return None

    def _build_request_headers(
        self,
        *,
        headers: Optional[JsonDict],
        disable_retries: Optional[bool],
        disable_fallback: Optional[bool],
        store_io: Optional[bool],
        object_lifecycle_preference: Optional[JsonDict],
        hint: Optional[str],
        start_timeout: Optional[float],
        priority: Optional[str],
        include_priority: bool = True,
    ) -> JsonDict:
        merged = dict(self.headers)
        merged.update(headers or {})

        effective_disable_retries = (
            self.disable_retries if disable_retries is None else disable_retries
        )
        effective_disable_fallback = (
            self.disable_fallback if disable_fallback is None else disable_fallback
        )
        effective_store_io = self.store_io if store_io is None else store_io
        effective_object_lifecycle = (
            self.object_lifecycle_preference
            if object_lifecycle_preference is None
            else object_lifecycle_preference
        )
        effective_hint = self.hint if hint is None else hint
        effective_start_timeout = self.start_timeout if start_timeout is None else start_timeout
        effective_priority = self.priority if priority is None else priority

        if effective_start_timeout is not None:
            if effective_start_timeout <= 1.0:
                raise ValueError("FAL start_timeout must be greater than 1.0 seconds.")
            merged["X-Fal-Request-Timeout"] = str(effective_start_timeout)
        if effective_hint is not None:
            merged["X-Fal-Runner-Hint"] = str(effective_hint)
        if include_priority and effective_priority is not None:
            merged["X-Fal-Queue-Priority"] = str(effective_priority)
        if effective_disable_retries:
            merged["X-Fal-No-Retry"] = "1"
        if effective_disable_fallback:
            merged["x-app-fal-disable-fallback"] = "1"
        if effective_store_io is not None:
            merged["X-Fal-Store-IO"] = _bool_header(bool(effective_store_io))
        if effective_object_lifecycle is not None:
            merged["X-Fal-Object-Lifecycle-Preference"] = json.dumps(effective_object_lifecycle)
        return merged

    def _build_queue_params(
        self,
        *,
        webhook_url: Optional[str],
        fal_max_queue_length: Optional[int],
    ) -> JsonDict:
        params: JsonDict = {}
        effective_webhook = self.webhook_url if webhook_url is None else webhook_url
        effective_max_queue_length = (
            self.fal_max_queue_length if fal_max_queue_length is None else fal_max_queue_length
        )
        if effective_webhook is not None:
            params["fal_webhook"] = effective_webhook
        if effective_max_queue_length is not None:
            params["fal_max_queue_length"] = int(effective_max_queue_length)
        return params

    def _queue_url(self, endpoint_id: str, path: Optional[str]) -> str:
        suffix = self.path if path is None else _normalize_path(path)
        return f"https://queue.fal.run/{_strip_slashes(endpoint_id)}{suffix}"

    def _queue_request_url(
        self,
        endpoint_id: str,
        request_id: str,
        *,
        path: Optional[str] = None,
        suffix: str = "",
    ) -> str:
        base = self._queue_url(endpoint_id, path)
        return f"{base}/requests/{request_id}{suffix}"

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Any] = None,
        json_body: Optional[Any] = None,
        headers: Optional[JsonDict] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
        include_auth: bool = True,
        endpoint_id: Optional[str] = None,
    ):
        requester = _get_requests()
        request_headers = dict(headers or {})
        if include_auth:
            request_headers.setdefault("Authorization", f"Key {self._api_key}")
        try:
            response = requester.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=request_headers,
                timeout=timeout,
                stream=stream,
            )
        except Exception as exc:
            raise FALTransportError(
                str(exc),
                endpoint_id=endpoint_id,
                response_body={"url": url},
            ) from exc
        if response.status_code >= 400:
            self._raise_from_response(response, endpoint_id=endpoint_id)
        return response

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Any] = None,
        json_body: Optional[Any] = None,
        headers: Optional[JsonDict] = None,
        timeout: Optional[float] = None,
        endpoint_id: Optional[str] = None,
        include_auth: bool = True,
    ) -> JsonDict:
        response = self._request(
            method,
            url,
            params=params,
            json_body=json_body,
            headers=headers,
            timeout=timeout,
            include_auth=include_auth,
            endpoint_id=endpoint_id,
        )
        try:
            data = response.json()
        except ValueError:
            data = {"raw_text": response.text}
        return {
            "data": data,
            "headers": dict(response.headers),
            "status_code": response.status_code,
            "url": response.url,
            "endpoint_id": endpoint_id,
        }

    def _envelope(self, response: JsonDict, *, endpoint_id: str, mode: str) -> FALResponse:
        data = response["data"]
        request_id = (
            data.get("request_id")
            if isinstance(data, dict)
            else None
        ) or response["headers"].get("x-fal-request-id")
        if request_id and isinstance(data, dict):
            self._cache_request_urls(request_id, data)
        return FALResponse(
            data=data,
            request_id=request_id,
            status_code=response["status_code"],
            endpoint_id=endpoint_id,
            mode=mode,
            headers=response["headers"],
            url=response["url"],
        )

    def _raise_from_response(self, response: Any, *, endpoint_id: Optional[str] = None) -> None:
        request_id = response.headers.get("x-fal-request-id")
        error_header = response.headers.get("X-Fal-Error-Type") or response.headers.get(
            "x-fal-error-type"
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text}

        if isinstance(payload, dict) and isinstance(payload.get("detail"), list):
            detail = payload.get("detail") or []
            message = "; ".join(item.get("msg", "Model error") for item in detail if isinstance(item, dict))
            raise FALModelError(
                message or "FAL model validation/content error.",
                detail=detail,
                request_id=request_id,
                status_code=response.status_code,
                endpoint_id=endpoint_id or self.model,
                headers=response.headers,
                response_body=payload,
                error_type=error_header,
            )

        message = ""
        if isinstance(payload, dict):
            message = str(payload.get("detail") or payload.get("message") or response.text)
        if not message:
            message = response.text or f"FAL request failed with status {response.status_code}"
        raise FALRequestError(
            message,
            request_id=request_id,
            status_code=response.status_code,
            endpoint_id=endpoint_id or self.model,
            headers=response.headers,
            response_body=payload,
            error_type=(payload.get("error_type") if isinstance(payload, dict) else None) or error_header,
        )

    def _cache_request_urls(self, request_id: str, payload: JsonDict) -> None:
        cached = self._request_url_cache.setdefault(request_id, {})
        for key in ("response_url", "status_url", "cancel_url"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                cached[key] = value

    def _request_url_for(
        self,
        request_id: str,
        *,
        endpoint_id: str,
        path: Optional[str],
        kind: str,
        fallback_suffix: str,
    ) -> str:
        cached = self._request_url_cache.get(request_id, {})
        if kind in cached:
            base_url = cached[kind]
            if fallback_suffix == "/status/stream" and kind == "status_url":
                return f"{base_url}/stream"
            return base_url
        return self._queue_request_url(endpoint_id, request_id, path=path, suffix=fallback_suffix)

    def _normalize_exception(self, exc: Exception, *, endpoint_id: str) -> FALClientError:
        if isinstance(exc, FALClientError):
            return exc

        status_code = getattr(exc, "status_code", None)
        headers = dict(getattr(exc, "response_headers", {}) or {})
        response_body = getattr(exc, "response", None)
        request_id = headers.get("x-fal-request-id") or headers.get("X-Fal-Request-Id")
        error_type = getattr(exc, "error_type", None) or headers.get("X-Fal-Error-Type")
        message = str(exc)

        if isinstance(response_body, dict) and isinstance(response_body.get("detail"), list):
            return FALModelError(
                message,
                detail=response_body.get("detail"),
                request_id=request_id,
                status_code=status_code,
                endpoint_id=endpoint_id,
                headers=headers,
                response_body=response_body,
                error_type=error_type,
            )

        if status_code is not None:
            return FALRequestError(
                message,
                request_id=request_id,
                status_code=status_code,
                endpoint_id=endpoint_id,
                headers=headers,
                response_body=response_body,
                error_type=error_type,
            )

        return FALTransportError(
            message,
            request_id=request_id,
            endpoint_id=endpoint_id,
            headers=headers,
            response_body=response_body,
            error_type=error_type,
        )

    def _resolve_openapi_schema(self, schema: JsonDict, openapi: JsonDict) -> JsonDict:
        if "$ref" not in schema:
            return schema
        ref = schema["$ref"]
        if not ref.startswith("#/components/schemas/"):
            return schema
        name = ref.rsplit("/", 1)[-1]
        return openapi.get("components", {}).get("schemas", {}).get(name, schema)

    def _is_terminal_queue_status(self, status_value: str) -> bool:
        return status_value in {"FAILED", "CANCELLED"}

    def _raise_terminal_queue_status(self, response: FALResponse) -> None:
        payload = response.data if isinstance(response.data, dict) else {"raw": response.data}
        status_value = str(payload.get("status", "UNKNOWN")).upper()
        message = str(payload.get("error") or payload.get("detail") or f"FAL request ended with status {status_value}.")
        raise FALRequestError(
            message,
            request_id=response.request_id,
            status_code=response.status_code,
            endpoint_id=response.endpoint_id,
            headers=response.headers,
            response_body=payload,
            error_type=payload.get("error_type") or status_value.lower(),
        )

    def _get_jwks(self, jwks_url: str) -> JsonDict:
        cached = self._jwks_cache.get(jwks_url)
        if cached is not None:
            cached_at, jwks = cached
            if time.time() - cached_at < self.webhook_jwks_cache_seconds:
                return jwks

        response = self._request_json(
            "GET",
            jwks_url,
            timeout=self.timeout,
            include_auth=False,
            endpoint_id=self.model,
        )
        self._jwks_cache[jwks_url] = (time.time(), response["data"])
        return response["data"]

    def _decode_webhook_signature(self, value: str) -> bytes:
        candidate = value.strip()
        if "=" in candidate and "," in candidate:
            for part in candidate.split(","):
                if "=" in part:
                    _, candidate = part.split("=", 1)
                    candidate = candidate.strip()
                    break
        candidate = candidate.replace("-", "+").replace("_", "/")
        padding = "=" * (-len(candidate) % 4)
        try:
            return base64.b64decode(candidate + padding)
        except Exception as exc:
            raise FALWebhookVerificationError("Invalid FAL webhook signature encoding.") from exc


def load_model_deployments(yaml_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    if yaml_path is None:
        current_dir = Path(__file__).parent.parent
        yaml_path = current_dir / "prod_env" / "model_deployments.yaml"

    yaml = _get_yaml()
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Model deployments file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None or "models" not in data:
        raise ValueError(f"Invalid model deployments YAML: {yaml_path}")

    configs: Dict[str, Dict[str, Any]] = {}
    for model_config in data["models"]:
        if "name" not in model_config or "model" not in model_config:
            continue
        client_name = model_config.get("client")
        backend_name = model_config.get("backend")
        model_name = str(model_config["model"])
        if client_name not in {None, "fal"} and backend_name not in {None, "fal"}:
            continue
        if (
            backend_name == "fal"
            or client_name == "fal"
            or model_name.startswith("fal-")
            or model_name.startswith("fal-ai/")
            or model_name.startswith("workflows/")
        ):
            configs[model_config["name"]] = model_config

    return configs


def load_model_config(name: str, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    configs = load_model_deployments(yaml_path)
    if name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"FAL model '{name}' not found. Available models: {available}")
    return configs[name]


def get_model_client(
    name: str,
    api_key: Optional[str] = None,
    yaml_path: Optional[str] = None,
    **override_params: Any,
) -> FALClient:
    try:
        config = dict(load_model_config(name, yaml_path))
        model = config.pop("model")
        config.pop("name", None)
        client_name = config.pop("client", None)
        backend_name = config.pop("backend", None)
        if client_name not in {None, "fal"}:
            raise ValueError(f"Model '{name}' is not configured for the FAL client.")
        if backend_name not in {None, "fal"}:
            raise ValueError(f"Model '{name}' is not configured for the FAL client.")

        fal_config = config.pop("fal", {})
        if not isinstance(fal_config, dict):
            raise ValueError(f"Model '{name}' has an invalid 'fal' config block.")

        allowed = set(inspect.signature(FALClient.__init__).parameters)
        allowed.discard("self")
        merged = {**config, **fal_config, **override_params}
        client_params = {key: value for key, value in merged.items() if key in allowed}
    except (ValueError, FileNotFoundError):
        model = name
        client_params = override_params

    return FALClient(model=model, api_key=api_key, **client_params)


load_fal_model_deployments = load_model_deployments
load_fal_model_config = load_model_config
get_fal_client = get_model_client
