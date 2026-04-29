#!/usr/bin/env python3
"""Scan fal.ai model API pages for schema fields related to reasoning traces.

This script fetches the public FAL sitemap, filters to model API pages, and
checks each page for specific field names. It writes a JSON report so the
results can be inspected later without re-running the crawl.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "Mozilla/5.0 (compatible; fal-schema-scan/1.0)"
SITEMAP_INDEX_URL = "https://fal.ai/sitemap.xml"
DEFAULT_OUTPUT = Path("artifacts/fal_schema_scan.json")
FIELDS = (
    "reformat_prompt",
    "think_info",
    "best_info",
    "enable_thinking_mode",
    "enable_reflection_mode",
    "thinking_level",
    "reasoning",
)


@dataclass
class PageScan:
    url: str
    status: int | None
    title: str | None
    model_id: str | None
    matches: list[str]
    error: str | None = None


def fetch_text(url: str, timeout: float = 20.0) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def get_sitemap_urls() -> list[str]:
    index_xml = fetch_text(SITEMAP_INDEX_URL)
    sitemap_urls = re.findall(r"<loc>(.*?)</loc>", index_xml)
    all_urls: list[str] = []
    for sitemap_url in sitemap_urls:
        xml = fetch_text(sitemap_url)
        all_urls.extend(re.findall(r"<loc>(.*?)</loc>", xml))
    return all_urls


def extract_title(html: str) -> str | None:
    match = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title or None


def extract_model_id(url: str, html: str) -> str | None:
    parts = url.split("/models/", 1)
    if len(parts) == 2:
        tail = parts[1].removesuffix("/api")
        if tail:
            return tail
    header_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.I | re.S)
    if header_match:
        header = re.sub(r"<.*?>", "", header_match.group(1))
        header = re.sub(r"\s+", " ", header).strip()
        return header or None
    return None


def find_matches(html: str, fields: Iterable[str]) -> list[str]:
    lower_html = html.lower()
    matches = [field for field in fields if field.lower() in lower_html]
    return sorted(matches)


def scan_page(url: str) -> PageScan:
    try:
        html = fetch_text(url)
    except HTTPError as exc:
        return PageScan(url=url, status=exc.code, title=None, model_id=None, matches=[], error=f"HTTP {exc.code}")
    except URLError as exc:
        return PageScan(url=url, status=None, title=None, model_id=None, matches=[], error=f"URL error: {exc.reason}")
    except Exception as exc:  # pragma: no cover - defensive fallback
        return PageScan(url=url, status=None, title=None, model_id=None, matches=[], error=str(exc))

    return PageScan(
        url=url,
        status=200,
        title=extract_title(html),
        model_id=extract_model_id(url, html),
        matches=find_matches(html, FIELDS),
    )


def build_report(scans: list[PageScan], api_urls: list[str], started_at: float) -> dict:
    matched = [scan for scan in scans if scan.matches]
    field_to_models: dict[str, list[dict[str, str]]] = {field: [] for field in FIELDS}
    for scan in matched:
        for field in scan.matches:
            field_to_models[field].append(
                {
                    "model_id": scan.model_id or "",
                    "url": scan.url,
                    "title": scan.title or "",
                }
            )

    for models in field_to_models.values():
        models.sort(key=lambda item: item["model_id"])

    return {
        "generated_at_epoch": time.time(),
        "duration_seconds": round(time.time() - started_at, 2),
        "api_pages_scanned": len(api_urls),
        "pages_with_any_match": len(matched),
        "fields_checked": list(FIELDS),
        "field_to_models": field_to_models,
        "matched_pages": [
            {
                "model_id": scan.model_id,
                "url": scan.url,
                "title": scan.title,
                "matches": scan.matches,
            }
            for scan in sorted(matched, key=lambda item: item.model_id or item.url)
        ],
        "errors": [
            {
                "url": scan.url,
                "status": scan.status,
                "error": scan.error,
            }
            for scan in scans
            if scan.error
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON report. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Concurrent workers for page fetches. Default: 16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of API pages to scan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    all_urls = get_sitemap_urls()
    api_urls = sorted({url for url in all_urls if "/models/" in url and url.endswith("/api")})
    if args.limit is not None:
        api_urls = api_urls[: args.limit]

    scans: list[PageScan] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_url = {executor.submit(scan_page, url): url for url in api_urls}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_url), start=1):
            scan = future.result()
            scans.append(scan)
            if index % 100 == 0 or index == len(api_urls):
                print(
                    f"scanned {index}/{len(api_urls)} pages; "
                    f"matches so far: {sum(1 for item in scans if item.matches)}",
                    file=sys.stderr,
                )

    report = build_report(scans, api_urls, started_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "api_pages_scanned": report["api_pages_scanned"],
            "pages_with_any_match": report["pages_with_any_match"],
            "output": str(args.output),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
