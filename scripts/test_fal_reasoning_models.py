#!/usr/bin/env python3
"""
Run three FAL image models through the local harness and capture queue logs.

The queue logs are the closest thing the public harness exposes to model
"reasoning" for image generation/editing. Results and logs are written to
artifacts/fal_reasoning_runs/.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clients.fal_client import FALResponse, get_fal_client

PROMPT = """_aidap
a hyperreal close portrait of a woman meeting the viewer's gaze head-on, eyes steady and lucid beneath a helmet cropped tight at the frame's edge. her face carries a dusting of white frost along the lashes and temples, lips slightly parted as if mid-breath, fine moisture glinting where the cold meets warmth. the light comes from behind a low golden sun cutting through icy air wrapping her in a soft halo while highlights cling delicately to the frost edges. her skin reveals honest micro detail: faint pores, subsurface glow on the cheeks, and a natural sheen from the chill. the palette moves between warm amber and arctic blue. every surface behaves realistically matte skin diffusing light, frost refracting it, the atmosphere crisp yet breathable. the mood is tender intensity, a quiet warmth radiating through the ice. captured on an 85mm lens at f/2.0, focus locked to her eyes, shallow depth isolating her face in luminous realism.
--v 7 --ar 2:3 --raw --profile"""

RUN_DIR = ROOT / "artifacts" / "fal_reasoning_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def read_fal_api_key() -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file at {env_path}")

    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FAL_API_KEY="):
            return line.split("=", 1)[1].strip()

    value = os.getenv("FAL_API_KEY") or os.getenv("FAL_KEY")
    if value:
        return value
    raise RuntimeError("FAL_API_KEY not found in .env or environment.")


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def extract_logs(payload: Any) -> List[Any]:
    if isinstance(payload, dict):
        logs = payload.get("logs")
        if isinstance(logs, list):
            return logs
    return []


def extract_first_media_url(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("images", "image", "data", "output"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                candidate = value[0]
                if isinstance(candidate, dict) and isinstance(candidate.get("url"), str):
                    return candidate["url"]
            if isinstance(value, dict) and isinstance(value.get("url"), str):
                return value["url"]

        for value in payload.values():
            url = extract_first_media_url(value)
            if url:
                return url
    elif isinstance(payload, list):
        for item in payload:
            url = extract_first_media_url(item)
            if url:
                return url
    return None


def download_media(url: str, target: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)


def persist_run(run: Dict[str, Any]) -> Dict[str, Any]:
    model_name = run["model_name"]
    result_url = extract_first_media_url(run["result"])
    json_path = RUN_DIR / f"{sanitize_filename(model_name)}.json"
    json_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    image_path: Optional[Path] = None
    if result_url:
        extension = ".png" if result_url.lower().endswith(".png") else ".jpg"
        image_path = RUN_DIR / f"{sanitize_filename(model_name)}{extension}"
        download_media(result_url, image_path)

    return {
        "model_name": model_name,
        "result_url": result_url,
        "artifact_json": str(json_path.relative_to(ROOT)),
        "artifact_image": str(image_path.relative_to(ROOT)) if image_path else None,
        "queue_update_count": len(run["queue_updates"]),
    }


def run_model(
    model_name: str,
    api_key: str,
    *,
    prompt: str,
    image_url: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = get_fal_client(model_name, api_key=api_key)
    queue_updates: List[Dict[str, Any]] = []

    def on_queue_update(update: FALResponse) -> None:
        queue_updates.append(update.to_dict())

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "on_queue_update": on_queue_update,
    }
    if image_url is not None:
        kwargs["image_url"] = image_url
    if overrides:
        kwargs.update(overrides)

    result = client.invoke(**kwargs)
    result_dict = result.to_dict() if isinstance(result, FALResponse) else {"data": result}

    artifact = {
        "model_name": model_name,
        "prompt": prompt,
        "image_url_input": image_url,
        "request_kwargs": kwargs,
        "queue_updates": queue_updates,
        "result": result_dict,
        "logs": [extract_logs(item.get("data")) for item in queue_updates],
    }
    return artifact


def main() -> None:
    api_key = read_fal_api_key()
    summary: List[Dict[str, Any]] = []

    z_run = run_model(
        "z-image-turbo",
        api_key,
        prompt=PROMPT,
        overrides={
            "image_size": {"width": 512, "height": 768},
            "num_images": 1,
            "num_inference_steps": 4,
            "enable_prompt_expansion": True,
            "acceleration": "none",
            "output_format": "png",
            "with_logs": True,
        },
    )
    z_url = extract_first_media_url(z_run["result"])
    if not z_url:
        raise RuntimeError("z-image-turbo did not return an image URL.")
    summary.append(persist_run(z_run))
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hunyuan_run = run_model(
        "HunyuanImage-3.0-Instruct",
        api_key,
        prompt=PROMPT,
        overrides={
            "image_size": {"width": 512, "height": 768},
            "num_images": 1,
            "guidance_scale": 3.5,
            "output_format": "png",
            "with_logs": True,
        },
    )
    hunyuan_url = extract_first_media_url(hunyuan_run["result"])
    if not hunyuan_url:
        raise RuntimeError("HunyuanImage-3.0-Instruct did not return an image URL.")
    summary.append(persist_run(hunyuan_run))
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stepx_run = run_model(
        "stepx-edit2",
        api_key,
        prompt=PROMPT,
        image_url=z_url,
        overrides={
            "num_inference_steps": 20,
            "guidance_scale": 5.5,
            "enable_thinking_mode": True,
            "enable_reflection_mode": True,
            "output_format": "png",
            "with_logs": True,
        },
    )
    stepx_url = extract_first_media_url(stepx_run["result"])
    if not stepx_url:
        raise RuntimeError("stepx-edit2 did not return an image URL.")
    summary.append(persist_run(stepx_run))
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_path = RUN_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
