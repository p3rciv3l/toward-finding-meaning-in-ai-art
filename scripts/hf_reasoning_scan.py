#!/usr/bin/env python3
"""Scan Hugging Face model catalogs for trace/reasoning-style signals.

This script crawls selected Hugging Face pipeline catalogs, follows pagination,
keeps only models with an inference provider mapping, and optionally scans each
model card README for trace-like keywords such as `think_info` or `best_info`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://huggingface.co/api/models"
RAW_README_URL = "https://huggingface.co/{model_id}/raw/main/README.md"
USER_AGENT = "Mozilla/5.0 (compatible; hf-reasoning-scan/1.0)"
DEFAULT_OUTPUT = Path("artifacts/hf_reasoning_scan.json")
PIPELINES = (
    "text-to-image",
    "image-to-image",
    "image-text-to-image",
    "image-text-to-text",
)
KEYWORDS = (
    "think_info",
    "best_info",
    "reformat_prompt",
    "reasoning_content",
    "reasoning_effort",
    "thinking mode",
    "thinking",
    "reasoning",
    "prompt enhancer",
    "prompt rewrite",
)


def fetch(url: str, timeout: float = 30.0) -> tuple[str, dict[str, str]]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
        headers = {key.lower(): value for key, value in response.headers.items()}
    return body, headers


def fetch_json(url: str, timeout: float = 30.0):
    body, headers = fetch(url, timeout=timeout)
    return json.loads(body), headers


def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
    return match.group(1) if match else None


def iter_pipeline_models(pipeline: str, limit: int = 100) -> Iterable[dict]:
    url = (
        f"{BASE_URL}?pipeline_tag={quote(pipeline)}&limit={limit}"
        "&full=true&expand=inferenceProviderMapping&sort=downloads&direction=-1"
    )
    while url:
        data, headers = fetch_json(url)
        for item in data:
            yield item
        url = parse_next_link(headers.get("link"))


def provider_mapping_names(item: dict) -> list[str]:
    mappings = item.get("inferenceProviderMapping") or []
    if isinstance(mappings, dict):
        mappings = [mappings]
    names = sorted(
        {
            mapping.get("provider")
            for mapping in mappings
            if isinstance(mapping, dict) and mapping.get("provider")
        }
    )
    return names


def fetch_readme(model_id: str) -> str | None:
    url = RAW_README_URL.format(model_id=model_id)
    try:
        body, _ = fetch(url)
    except (HTTPError, URLError):
        return None
    return body


def keyword_matches(text: str | None, keywords: Iterable[str]) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return sorted({kw for kw in keywords if kw.lower() in lowered})


def build_record(item: dict, include_readme: bool) -> dict:
    model_id = item["id"]
    readme = fetch_readme(model_id) if include_readme else None
    matches = keyword_matches(readme, KEYWORDS)
    return {
        "model_id": model_id,
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "pipeline_tag": item.get("pipeline_tag"),
        "provider_names": provider_mapping_names(item),
        "provider_mapping": item.get("inferenceProviderMapping") or [],
        "model_page_url": f"https://huggingface.co/{model_id}",
        "model_api_url": f"https://huggingface.co/api/models/{model_id}?expand=inferenceProviderMapping",
        "keyword_matches": matches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the JSON report. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--limit-per-pipeline",
        type=int,
        default=None,
        help="Optional cap for models per pipeline after pagination order is applied.",
    )
    parser.add_argument(
        "--skip-readmes",
        action="store_true",
        help="Skip README fetching and keyword scanning.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent workers for README fetching. Default: 12",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    records_by_pipeline: dict[str, list[dict]] = {}

    for pipeline in PIPELINES:
        raw_items: list[dict] = []
        seen = 0
        for item in iter_pipeline_models(pipeline):
            seen += 1
            mappings = item.get("inferenceProviderMapping") or []
            if mappings:
                raw_items.append(item)
            if seen % 100 == 0:
                print(
                    f"{pipeline}: scanned {seen} models, candidates with providers {len(raw_items)}",
                    file=sys.stderr,
                )
            if args.limit_per_pipeline is not None and seen >= args.limit_per_pipeline:
                break
        records: list[dict] = []
        if args.skip_readmes:
            records = [build_record(item, include_readme=False) for item in raw_items]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(build_record, item, True) for item in raw_items]
                for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    records.append(future.result())
                    if index % 50 == 0 or index == len(futures):
                        print(
                            f"{pipeline}: scanned READMEs for {index}/{len(futures)} routed models",
                            file=sys.stderr,
                        )
        records_by_pipeline[pipeline] = records
        print(
            f"{pipeline}: scanned {seen} models, kept {len(records)} with provider mappings",
            file=sys.stderr,
        )

    all_records = [record for records in records_by_pipeline.values() for record in records]
    deduped: dict[str, dict] = {}
    for record in all_records:
        model_id = record["model_id"]
        if model_id not in deduped:
            deduped[model_id] = {
                **record,
                "pipelines_seen": [record["pipeline_tag"]],
            }
        else:
            existing = deduped[model_id]
            existing["pipelines_seen"] = sorted(
                set(existing["pipelines_seen"]) | {record["pipeline_tag"]}
            )
            existing["provider_names"] = sorted(
                set(existing["provider_names"]) | set(record["provider_names"])
            )
            existing["keyword_matches"] = sorted(
                set(existing["keyword_matches"]) | set(record["keyword_matches"])
            )

    report = {
        "generated_at_epoch": time.time(),
        "duration_seconds": round(time.time() - started_at, 2),
        "pipelines": list(PIPELINES),
        "keywords": list(KEYWORDS),
        "records_by_pipeline": records_by_pipeline,
        "unique_model_count": len(deduped),
        "models_with_keyword_matches": [
            model for model in sorted(deduped.values(), key=lambda item: item["model_id"])
            if model["keyword_matches"]
        ],
        "all_unique_models": sorted(deduped.values(), key=lambda item: item["downloads"] or 0, reverse=True),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "unique_model_count": report["unique_model_count"],
                "models_with_keyword_matches": len(report["models_with_keyword_matches"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
