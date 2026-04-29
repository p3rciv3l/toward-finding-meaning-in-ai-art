#!/usr/bin/env python3

import json
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/api"
AUTHOR = "black-forest-labs"
OUTPUT_PATH = Path("artifacts/hf_flux_provider_scan.json")


def fetch_json(url: str):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def list_author_models():
    url = f"{BASE_URL}/models?author={AUTHOR}&limit=200"
    data = fetch_json(url)
    return [item["id"] for item in data if item["id"].startswith("black-forest-labs/FLUX.")]


def fetch_model_mapping(model_id: str):
    url = f"{BASE_URL}/models/{model_id}?expand=inferenceProviderMapping"
    data = fetch_json(url)
    return {
        "model_id": model_id,
        "pipeline_tag": data.get("pipeline_tag"),
        "tags": data.get("tags", []),
        "inference_provider_mapping": data.get("inferenceProviderMapping") or {},
        "model_api_url": url,
        "model_page_url": f"https://huggingface.co/{model_id}",
    }


def main():
    model_ids = sorted(list_author_models())
    report = {
        "source": "huggingface_hub_api",
        "author": AUTHOR,
        "models": [fetch_model_mapping(model_id) for model_id in model_ids],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
