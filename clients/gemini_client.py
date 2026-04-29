"""
Standalone Gemini Developer API client.

The client mirrors the repo's OpenRouter harness style: direct HTTP requests,
lazy third-party imports, YAML-driven defaults, and broad parameter exposure.
"""

import base64
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from types import ModuleType
from typing import IO, Any, BinaryIO, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast


_load_dotenv: Callable[..., bool]

try:
    _load_dotenv = cast(
        Callable[..., bool],
        importlib.import_module("dotenv").load_dotenv,
    )
except ImportError:  # pragma: no cover - optional dependency
    def _fallback_load_dotenv(
        dotenv_path: str | os.PathLike[str] | None = None,
        stream: IO[str] | None = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: str | None = "utf-8",
    ) -> bool:
        del dotenv_path, stream, verbose, override, interpolate, encoding
        return False

    _load_dotenv = _fallback_load_dotenv


_ = _load_dotenv()

_yaml: ModuleType | None = None
_requests: ModuleType | None = None


def _get_yaml():
    global _yaml
    if _yaml is None:
        try:
            _yaml = importlib.import_module("yaml")
        except ImportError as exc:
            raise ImportError(
                "pyyaml package is required. Install with: pip install pyyaml"
            ) from exc
    return _yaml


def _get_requests():
    global _requests
    if _requests is None:
        try:
            _requests = importlib.import_module("requests")
        except ImportError as exc:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            ) from exc
    return _requests


JsonDict = Dict[str, Any]
ContentInput = Union[str, JsonDict, Sequence[Any]]


class GeminiClient:
    """
    General-purpose Gemini Developer API client.

    The request builder is modality-agnostic: text, image, audio, video, file
    references, tool parts, and mixed prompts all flow through `contents`.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: Optional[float] = 180.0,
        api_version: str = "v1beta",
        base_url: str = "https://generativelanguage.googleapis.com",
        use_model_defaults: bool = True,
        use_batch_api: bool = True,
        batch_mode: str = "force",
        batch_input_mode: str = "auto",
        batch_priority: int = 0,
        batch_display_name: Optional[str] = None,
        batch_poll_interval_seconds: float = 30.0,
        batch_enable_fallback: bool = True,
        batch_inline_max_bytes: int = 20 * 1024 * 1024,
        batch_input_file_max_bytes: int = 2 * 1024 * 1024 * 1024,
        system_instruction: Optional[Union[str, JsonDict]] = None,
        tools: Optional[List[JsonDict]] = None,
        tool_config: Optional[JsonDict] = None,
        safety_settings: Optional[List[JsonDict]] = None,
        cached_content: Optional[str] = None,
        store: Optional[bool] = None,
        service_tier: Optional[str] = None,
        generation_config: Optional[JsonDict] = None,
        stop_sequences: Optional[List[str]] = None,
        response_mime_type: Optional[str] = "text/plain",
        response_schema: Optional[JsonDict] = None,
        response_json_schema: Optional[JsonDict] = None,
        _response_json_schema: Optional[JsonDict] = None,
        response_modalities: Optional[List[str]] = None,
        candidate_count: Optional[int] = 1,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_logprobs: Optional[bool] = None,
        logprobs: Optional[int] = None,
        enable_enhanced_civic_answers: Optional[bool] = None,
        speech_config: Optional[JsonDict] = None,
        thinking_config: Optional[JsonDict] = {
            "include_thoughts": True,
            "thinking_level": "high",
        },
        image_config: Optional[JsonDict] = None,
        media_resolution: Optional[str] = None,
        request_overrides: Optional[JsonDict] = None,
    ):
        self._api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                "or pass api_key explicitly."
            )

        self.model = self._normalize_model_name(model)
        self.timeout = timeout
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self.use_model_defaults = use_model_defaults
        self.use_batch_api = use_batch_api
        self.batch_mode = batch_mode
        self.batch_input_mode = batch_input_mode
        self.batch_priority = batch_priority
        self.batch_display_name = batch_display_name
        self.batch_poll_interval_seconds = batch_poll_interval_seconds
        self.batch_enable_fallback = batch_enable_fallback
        self.batch_inline_max_bytes = batch_inline_max_bytes
        self.batch_input_file_max_bytes = batch_input_file_max_bytes
        self._model_info_cache: Optional[JsonDict] = None
        self._request_overrides = request_overrides or {}

        self._defaults = {
            "system_instruction": system_instruction,
            "tools": tools,
            "tool_config": tool_config,
            "safety_settings": safety_settings,
            "cached_content": cached_content,
            "store": store,
            "service_tier": service_tier,
            "generation_config": generation_config,
            "stop_sequences": stop_sequences,
            "response_mime_type": response_mime_type,
            "response_schema": response_schema,
            "response_json_schema": response_json_schema,
            "_response_json_schema": _response_json_schema,
            "response_modalities": response_modalities,
            "candidate_count": candidate_count,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "response_logprobs": response_logprobs,
            "logprobs": logprobs,
            "enable_enhanced_civic_answers": enable_enhanced_civic_answers,
            "speech_config": speech_config,
            "thinking_config": thinking_config,
            "image_config": image_config,
            "media_resolution": media_resolution,
        }

    def generate(
        self,
        contents: ContentInput,
        *,
        use_batch_api: Optional[bool] = None,
        wait_for_batch: bool = True,
        display_name: Optional[str] = None,
        metadata: Optional[JsonDict] = None,
        **override_params: Any,
    ) -> JsonDict:
        should_use_batch = self._resolve_batch_usage(use_batch_api)
        if should_use_batch:
            try:
                batch = self.create_batch(
                    requests_or_file=[
                        {
                            "request": self._build_generate_request(
                                contents=contents,
                                include_model=True,
                                **override_params,
                            ),
                            "metadata": metadata,
                        }
                    ],
                    display_name=display_name,
                )
            except Exception as exc:
                if (
                    self.batch_mode == "force"
                    or not self.batch_enable_fallback
                    or not self._should_fallback_from_batch_exception(exc)
                ):
                    raise
            else:
                if wait_for_batch:
                    completed = self.wait_for_batch(batch)
                    return self.extract_batch_responses(completed)
                return batch

        payload = self._build_generate_request(contents=contents, **override_params)
        return self._request_json(
            "POST",
            f"/{self.api_version}/{self.model}:generateContent",
            json_body=payload,
        )

    def stream_generate(
        self,
        contents: ContentInput,
        **override_params: Any,
    ) -> Iterable[JsonDict]:
        payload = self._build_generate_request(contents=contents, **override_params)
        response = self._request(
            "POST",
            f"/{self.api_version}/{self.model}:streamGenerateContent?alt=sse",
            json_body=payload,
            stream=True,
        )
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            yield json.loads(line)

    def create_batch(
        self,
        requests_or_file: Union[str, Path, Sequence[JsonDict]],
        *,
        display_name: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> JsonDict:
        input_config: JsonDict
        if isinstance(requests_or_file, (str, Path)):
            input_config = {"file_name": str(requests_or_file)}
        else:
            input_config = {"requests": {"requests": list(requests_or_file)}}

        payload = {
            "display_name": display_name or self.batch_display_name or self._default_batch_name(),
            "input_config": input_config,
            "priority": self.batch_priority if priority is None else priority,
            "model": self.model,
        }

        return self._request_json(
            "POST",
            f"/{self.api_version}/{self.model}:batchGenerateContent",
            json_body={"batch": self._to_api_dict(payload)},
        )

    def get_batch(self, batch_name: Union[str, JsonDict]) -> JsonDict:
        return self._request_json("GET", self._batch_path(batch_name))

    def list_batches(
        self,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> JsonDict:
        params = self._clean_none({"pageSize": page_size, "pageToken": page_token})
        return self._request_json("GET", f"/{self.api_version}/batches", params=params)

    def cancel_batch(self, batch_name: Union[str, JsonDict]) -> JsonDict:
        return self._request_json(
            "POST", f"{self._batch_path(batch_name)}:cancel", json_body={}
        )

    def delete_batch(self, batch_name: Union[str, JsonDict]) -> JsonDict:
        return self._request_json("DELETE", self._batch_path(batch_name))

    def wait_for_batch(
        self,
        batch_name: Union[str, JsonDict],
        *,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ) -> JsonDict:
        interval = poll_interval_seconds or self.batch_poll_interval_seconds
        deadline = None if timeout_seconds is None else time.time() + timeout_seconds
        current = self.get_batch(batch_name)

        while not self._is_batch_terminal(current):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for batch {self._batch_name(batch_name)}")
            time.sleep(interval)
            current = self.get_batch(batch_name)

        return current

    def extract_batch_responses(self, batch: JsonDict) -> JsonDict:
        output = batch.get("output", {})
        if "inlinedResponses" in output:
            return output["inlinedResponses"]
        if "responsesFile" in output:
            return output
        return batch

    def count_tokens(
        self,
        *,
        contents: Optional[ContentInput] = None,
        generate_content_request: Optional[JsonDict] = None,
        **override_params: Any,
    ) -> JsonDict:
        if contents is not None and generate_content_request is not None:
            raise ValueError("Provide either contents or generate_content_request, not both.")

        if generate_content_request is not None:
            payload = {"generateContentRequest": self._to_api_dict(generate_content_request)}
        else:
            if override_params:
                payload = {
                    "generateContentRequest": self._build_generate_request(
                        contents=contents if contents is not None else "",
                        **override_params,
                    )
                }
            else:
                payload = {"contents": self._to_api_dict(self._normalize_contents(contents or ""))}

        return self._request_json(
            "POST",
            f"/{self.api_version}/{self.model}:countTokens",
            json_body=payload,
        )

    def upload_file(
        self,
        file: Union[str, Path, BinaryIO],
        *,
        mime_type: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> JsonDict:
        file_name, binary_data = self._read_file_bytes(file)
        effective_mime = mime_type or self._guess_mime_type(file_name)

        start_headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(binary_data)),
            "X-Goog-Upload-Header-Content-Type": effective_mime,
            "Content-Type": "application/json",
        }
        metadata = {"file": self._clean_none({"displayName": display_name or file_name})}
        start_response = self._request(
            "POST",
            f"/upload/{self.api_version}/files",
            json_body=metadata,
            extra_headers=start_headers,
        )
        upload_url = start_response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Gemini upload did not return X-Goog-Upload-URL.")

        finalize_headers = {
            "Content-Length": str(len(binary_data)),
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Type": effective_mime,
        }
        requests = _get_requests()
        response = requests.post(
            upload_url,
            headers={**self._auth_headers(), **finalize_headers},
            data=binary_data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_file(self, file_name: str) -> JsonDict:
        return self._request_json("GET", self._ensure_resource_name(file_name, "files"))

    def list_files(
        self,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> JsonDict:
        params = self._clean_none({"pageSize": page_size, "pageToken": page_token})
        return self._request_json("GET", f"/{self.api_version}/files", params=params)

    def delete_file(self, file_name: str) -> JsonDict:
        return self._request_json(
            "DELETE", self._ensure_resource_name(file_name, "files")
        )

    def create_cache(
        self,
        *,
        model: Optional[str] = None,
        contents: Optional[ContentInput] = None,
        ttl: Optional[str] = None,
        expire_time: Optional[str] = None,
        display_name: Optional[str] = None,
        system_instruction: Optional[Union[str, JsonDict]] = None,
        tools: Optional[List[JsonDict]] = None,
        tool_config: Optional[JsonDict] = None,
    ) -> JsonDict:
        payload = self._clean_none(
            {
                "model": self._normalize_model_name(model or self.model),
                "contents": self._normalize_contents(contents or []),
                "ttl": ttl,
                "expire_time": expire_time,
                "display_name": display_name,
                "system_instruction": system_instruction,
                "tools": tools,
                "tool_config": tool_config,
            }
        )
        return self._request_json(
            "POST",
            f"/{self.api_version}/cachedContents",
            json_body=self._to_api_dict(payload),
        )

    def get_cache(self, cache_name: str) -> JsonDict:
        return self._request_json(
            "GET", self._ensure_resource_name(cache_name, "cachedContents")
        )

    def list_caches(
        self,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> JsonDict:
        params = self._clean_none({"pageSize": page_size, "pageToken": page_token})
        return self._request_json("GET", f"/{self.api_version}/cachedContents", params=params)

    def update_cache_ttl(
        self,
        cache_name: str,
        *,
        ttl: Optional[str] = None,
        expire_time: Optional[str] = None,
    ) -> JsonDict:
        payload = self._clean_none({"ttl": ttl, "expire_time": expire_time})
        return self._request_json(
            "PATCH",
            self._ensure_resource_name(cache_name, "cachedContents"),
            params={"updateMask": ",".join(sorted(self._to_api_dict(payload).keys()))},
            json_body=self._to_api_dict(payload),
        )

    def delete_cache(self, cache_name: str) -> JsonDict:
        return self._request_json(
            "DELETE", self._ensure_resource_name(cache_name, "cachedContents")
        )

    def get_model_info(self, model: Optional[str] = None) -> JsonDict:
        normalized = self._normalize_model_name(model or self.model)
        response = self._request_json("GET", f"/{self.api_version}/{normalized}")
        if normalized == self.model:
            self._model_info_cache = response
        return response

    def list_models(
        self,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> JsonDict:
        params = self._clean_none({"pageSize": page_size, "pageToken": page_token})
        return self._request_json("GET", f"/{self.api_version}/models", params=params)

    def _build_generate_request(
        self,
        *,
        contents: ContentInput,
        include_model: bool = False,
        **override_params: Any,
    ) -> JsonDict:
        params = dict(self._defaults)
        params.update(override_params)

        model_defaults = self._get_model_defaults() if self.use_model_defaults else {}
        generation_config = {}
        raw_generation_config = params.pop("generation_config", None)
        if raw_generation_config:
            generation_config.update(raw_generation_config)

        generation_fields = {
            "stop_sequences": None,
            "response_mime_type": "text/plain",
            "response_schema": None,
            "response_json_schema": None,
            "_response_json_schema": None,
            "response_modalities": None,
            "candidate_count": 1,
            "max_output_tokens": model_defaults.get("max_output_tokens"),
            "temperature": model_defaults.get("temperature"),
            "top_p": model_defaults.get("top_p"),
            "top_k": model_defaults.get("top_k"),
            "seed": None,
            "presence_penalty": None,
            "frequency_penalty": None,
            "response_logprobs": None,
            "logprobs": None,
            "enable_enhanced_civic_answers": None,
            "speech_config": None,
            "thinking_config": None,
            "image_config": None,
            "media_resolution": None,
        }
        for key, default_value in generation_fields.items():
            value = params.pop(key, default_value)
            if value is not None:
                generation_config[key] = value

        payload = self._clean_none(
            {
                "contents": self._normalize_contents(contents),
                "system_instruction": params.pop("system_instruction", None),
                "tools": params.pop("tools", None),
                "tool_config": params.pop("tool_config", None),
                "safety_settings": params.pop("safety_settings", None),
                "cached_content": params.pop("cached_content", None),
                "store": params.pop("store", None),
                "service_tier": params.pop("service_tier", None),
                "generation_config": generation_config or None,
            }
        )
        if include_model:
            payload["model"] = self.model
        payload.update(self._request_overrides)
        payload.update(params)
        return self._to_api_dict(payload)

    def _normalize_contents(self, contents: ContentInput) -> List[JsonDict]:
        if isinstance(contents, str):
            return [{"role": "user", "parts": [{"text": contents}]}]

        if isinstance(contents, dict):
            if "contents" in contents:
                nested = contents["contents"]
                return self._normalize_contents(nested)
            if "parts" in contents:
                return [self._normalize_content(contents)]
            return [{"role": "user", "parts": [self._normalize_part(contents)]}]

        normalized_parts: List[JsonDict] = []
        normalized_contents: List[JsonDict] = []
        for item in contents:
            if isinstance(item, dict) and "parts" in item:
                if normalized_parts:
                    normalized_contents.append({"role": "user", "parts": normalized_parts})
                    normalized_parts = []
                normalized_contents.append(self._normalize_content(item))
            else:
                normalized_parts.append(self._normalize_part(item))

        if normalized_parts:
            normalized_contents.append({"role": "user", "parts": normalized_parts})

        return normalized_contents

    def _normalize_content(self, content: JsonDict) -> JsonDict:
        role = content.get("role", "user")
        raw_parts = content.get("parts", [])
        return {
            "role": role,
            "parts": [self._normalize_part(part) for part in raw_parts],
        }

    def _normalize_part(self, part: Any) -> JsonDict:
        if isinstance(part, str):
            return {"text": part}

        if isinstance(part, Path):
            return self._file_path_to_inline_part(part)

        if isinstance(part, dict):
            if "text" in part and len(part) == 1:
                return {"text": part["text"]}
            if "mime_type" in part and "data" in part:
                return {"inline_data": {"mime_type": part["mime_type"], "data": part["data"]}}
            if "mimeType" in part and "data" in part:
                return {"inline_data": {"mime_type": part["mimeType"], "data": part["data"]}}
            if "file_uri" in part or "fileUri" in part:
                return {
                    "file_data": {
                        "file_uri": part.get("file_uri") or part.get("fileUri"),
                        "mime_type": part.get("mime_type") or part.get("mimeType"),
                    }
                }
            if "inline_data" in part or "inlineData" in part or "file_data" in part or "fileData" in part:
                return part
            return part

        raise TypeError(f"Unsupported Gemini content part: {type(part)!r}")

    def _file_path_to_inline_part(self, path: Path) -> JsonDict:
        mime_type = self._guess_mime_type(path.name)
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    def _resolve_batch_usage(self, requested: Optional[bool]) -> bool:
        if requested is not None:
            return requested
        if self.batch_mode == "never":
            return False
        return self.use_batch_api

    def _should_fallback_from_batch_exception(self, exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {404, 405, 501}:
            return True

        body_text = ""
        if response is not None:
            try:
                payload = response.json()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                body_text = json.dumps(payload).lower()
            else:
                body_text = (getattr(response, "text", "") or "").lower()

        message = str(exc).lower()
        combined = f"{message} {body_text}"
        return any(
            needle in combined
            for needle in (
                "batch not supported",
                "batch unsupported",
                "not implemented",
                "method not allowed",
                "not found",
            )
        )

    def _get_model_defaults(self) -> JsonDict:
        if self._model_info_cache is None:
            try:
                self._model_info_cache = self.get_model_info(self.model)
            except Exception:
                return {}

        return {
            "max_output_tokens": self._model_info_cache.get("outputTokenLimit"),
            "temperature": self._model_info_cache.get("temperature"),
            "top_p": self._model_info_cache.get("topP"),
            "top_k": self._model_info_cache.get("topK"),
        }

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[JsonDict] = None,
        params: Optional[JsonDict] = None,
    ) -> JsonDict:
        response = self._request(method, path, json_body=json_body, params=params)
        if not response.content:
            return {}
        return response.json()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[JsonDict] = None,
        params: Optional[JsonDict] = None,
        extra_headers: Optional[JsonDict] = None,
        stream: bool = False,
    ):
        requests = _get_requests()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        response = requests.request(
            method=method,
            url=url,
            headers={**self._auth_headers(), **(extra_headers or {})},
            json=json_body,
            params=params,
            timeout=self.timeout,
            stream=stream,
        )
        response.raise_for_status()
        return response

    def _auth_headers(self) -> JsonDict:
        return {"x-goog-api-key": self._api_key}

    def _normalize_model_name(self, model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    def _ensure_resource_name(self, name: str, resource_prefix: str) -> str:
        normalized = name if name.startswith(f"{resource_prefix}/") else f"{resource_prefix}/{name}"
        return f"/{self.api_version}/{normalized}"

    def _batch_path(self, batch_name: Union[str, JsonDict]) -> str:
        return self._ensure_resource_name(self._batch_name(batch_name), "batches")

    def _batch_name(self, batch_name: Union[str, JsonDict]) -> str:
        if isinstance(batch_name, dict):
            name = batch_name.get("name")
            if not name and "response" in batch_name:
                name = batch_name["response"].get("name")
            if not name:
                raise ValueError("Batch object does not contain a name field.")
            return name
        return batch_name

    def _is_batch_terminal(self, batch: JsonDict) -> bool:
        state = (batch.get("state") or "").replace("BATCH_STATE_", "").replace("JOB_STATE_", "")
        return state in {"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"}

    def _default_batch_name(self) -> str:
        return f"{self.model.rsplit('/', 1)[-1]}-{int(time.time())}"

    def _read_file_bytes(self, file: Union[str, Path, BinaryIO]) -> Tuple[str, bytes]:
        if isinstance(file, (str, Path)):
            path = Path(file)
            with open(path, "rb") as handle:
                return path.name, handle.read()
        if hasattr(file, "read"):
            filename = getattr(file, "name", "upload.bin")
            data = file.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return Path(filename).name, data
        raise TypeError(f"Unsupported file input: {type(file)!r}")

    def _guess_mime_type(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".heic": "image/heic",
            ".heif": "image/heif",
            ".gif": "image/gif",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".mp4": "video/mp4",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".json": "application/json",
        }.get(suffix, "application/octet-stream")

    def _to_api_dict(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._to_api_dict(item) for item in value]
        if isinstance(value, dict):
            converted = {}
            for key, item_value in value.items():
                converted[self._snake_to_camel(key)] = self._to_api_dict(item_value)
            return self._clean_none(converted)
        return value

    def _snake_to_camel(self, key: str) -> str:
        special = {
            "_response_json_schema": "_responseJsonSchema",
            "system_instruction": "systemInstruction",
            "tool_config": "toolConfig",
            "safety_settings": "safetySettings",
            "cached_content": "cachedContent",
            "service_tier": "serviceTier",
            "generation_config": "generationConfig",
            "stop_sequences": "stopSequences",
            "response_mime_type": "responseMimeType",
            "response_schema": "responseSchema",
            "response_json_schema": "responseJsonSchema",
            "response_modalities": "responseModalities",
            "candidate_count": "candidateCount",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
            "presence_penalty": "presencePenalty",
            "frequency_penalty": "frequencyPenalty",
            "response_logprobs": "responseLogprobs",
            "enable_enhanced_civic_answers": "enableEnhancedCivicAnswers",
            "speech_config": "speechConfig",
            "thinking_config": "thinkingConfig",
            "image_config": "imageConfig",
            "media_resolution": "mediaResolution",
            "function_declarations": "functionDeclarations",
            "google_search": "googleSearch",
            "google_search_retrieval": "googleSearchRetrieval",
            "google_maps": "googleMaps",
            "file_search": "fileSearch",
            "url_context": "urlContext",
            "computer_use": "computerUse",
            "mcp_servers": "mcpServers",
            "function_calling_config": "functionCallingConfig",
            "allowed_function_names": "allowedFunctionNames",
            "retrieval_config": "retrievalConfig",
            "include_server_side_tool_invocations": "includeServerSideToolInvocations",
            "inline_data": "inlineData",
            "file_data": "fileData",
            "mime_type": "mimeType",
            "file_uri": "fileUri",
            "thought_signature": "thoughtSignature",
            "part_metadata": "partMetadata",
            "video_metadata": "videoMetadata",
            "display_name": "displayName",
            "input_config": "inputConfig",
            "file_name": "fileName",
            "expire_time": "expireTime",
        }
        if key in special:
            return special[key]
        if "_" not in key:
            return key
        head, *tail = key.split("_")
        return head + "".join(piece.capitalize() for piece in tail)

    def _clean_none(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._clean_none(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [self._clean_none(item) for item in value if item is not None]
        return value


def load_model_deployments(
    yaml_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    if yaml_path is None:
        current_dir = Path(__file__).parent.parent
        resolved_yaml_path = current_dir / "prod_env" / "model_deployments.yaml"
    else:
        resolved_yaml_path = Path(yaml_path)

    yaml = _get_yaml()
    if not resolved_yaml_path.exists():
        raise FileNotFoundError(
            f"Model deployments file not found: {resolved_yaml_path}"
        )

    with open(resolved_yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None or "models" not in data:
        raise ValueError(f"Invalid model deployments YAML: {resolved_yaml_path}")

    configs: Dict[str, Dict[str, Any]] = {}
    for model_config in data["models"]:
        if "name" not in model_config or "model" not in model_config:
            continue
        if model_config.get("client") == "gemini" or str(model_config["model"]).startswith("gemini"):
            configs[model_config["name"]] = model_config

    return configs


def load_model_config(
    name: str, yaml_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    configs = load_model_deployments(yaml_path)
    if name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Gemini model '{name}' not found. Available models: {available}")
    return configs[name]


def get_model_client(
    name: str,
    api_key: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
    **override_params: Any,
) -> GeminiClient:
    try:
        config = dict(load_model_config(name, yaml_path))
        model = config.pop("model")
        config.pop("name", None)
        config.pop("api_provider", None)
        config.pop("provider", None)
        config.pop("backend", None)
        client_name = config.pop("client", None)
        if client_name not in {None, "gemini"}:
            raise ValueError(f"Model '{name}' is not configured for the Gemini client.")

        api_base_url = config.pop("api_base_url", None)
        if api_base_url is not None and "base_url" not in override_params:
            config["base_url"] = api_base_url

        allowed = set(inspect.signature(GeminiClient.__init__).parameters)
        allowed.discard("self")
        client_params = {
            key: value for key, value in {**config, **override_params}.items() if key in allowed
        }
    except (ValueError, FileNotFoundError):
        model = name
        client_params = override_params
    return GeminiClient(model=model, api_key=api_key, **client_params)


load_gemini_model_deployments = load_model_deployments
load_gemini_model_config = load_model_config
get_gemini_client = get_model_client
