#!/usr/bin/env python3
"""
Dev-only helper to scrape ETA text from a Google Maps directions web page.

Usage:
  python scripts/scrape_google_maps_eta.py "<google maps directions url>"

Optional:
  python scripts/scrape_google_maps_eta.py "<url>" --headed
  python scripts/scrape_google_maps_eta.py "<url>" --timeout 45

Notes:
- This is for debugging/prototyping only.
- Requires Playwright:
    pip install playwright
    playwright install chromium
- Google Maps DOM can change, so this script is best-effort rather than guaranteed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, List


DURATION_RE = re.compile(r"^(?:(?:\d+\s*hr\s*)?\d+\s*min|\d+\s*hr)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape ETA from Google Maps directions page")
    parser.add_argument("url", help="Google Maps directions URL")
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--timeout",
        type=int,
        default=35,
        help="Navigation/render timeout in seconds (default: 35)",
    )
    return parser.parse_args()


def import_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Missing dependency: playwright\n"
            "Install it with:\n"
            "  pip install playwright\n"
            "  playwright install chromium",
            file=sys.stderr,
        )
        sys.exit(2)
    return sync_playwright, PlaywrightTimeoutError


def scrape_candidates(page) -> List[dict[str, Any]]:
    return page.evaluate(
        """
        () => {
          const durationRe = /^(?:(?:\\d+\\s*hr\\s*)?\\d+\\s*min|\\d+\\s*hr)$/i;
          const nodes = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);

          while (walker.nextNode()) {
            const textNode = walker.currentNode;
            const text = (textNode.textContent || "").replace(/\\s+/g, " ").trim();
            if (!durationRe.test(text)) continue;

            const parent = textNode.parentElement;
            if (!parent) continue;

            const style = window.getComputedStyle(parent);
            if (style.visibility === "hidden" || style.display === "none") continue;

            const rect = parent.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) continue;

            nodes.push({
              text,
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height,
              tag: parent.tagName,
              ariaLabel: parent.getAttribute("aria-label") || "",
            });
          }

          const deduped = [];
          const seen = new Set();
          for (const node of nodes) {
            const key = `${node.text}|${Math.round(node.x)}|${Math.round(node.y)}`;
            if (seen.has(key)) continue;
            seen.add(key);
            deduped.push(node);
          }

          deduped.sort((a, b) => {
            if (Math.abs(a.y - b.y) > 12) return a.y - b.y;
            return a.x - b.x;
          });

          return deduped;
        }
        """
    )


def duration_to_minutes(text: str) -> int:
    normalized = text.lower().replace("hours", "hr").replace("hour", "hr").replace("minutes", "min").replace("minute", "min")
    hours = 0
    minutes = 0
    hr_match = re.search(r"(\d+)\s*hr", normalized)
    min_match = re.search(r"(\d+)\s*min", normalized)
    if hr_match:
        hours = int(hr_match.group(1))
    if min_match:
        minutes = int(min_match.group(1))
    return hours * 60 + minutes


def pick_best_guess(candidates: List[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    enriched = []
    for item in candidates:
        text = str(item.get("text", "")).strip()
        if not DURATION_RE.match(text):
            continue
        item = dict(item)
        item["minutes"] = duration_to_minutes(text)
        enriched.append(item)

    if not enriched:
        return None

    # Best-effort heuristic:
    # - prefer items near the upper-left directions panel
    # - among those, take the first visible duration candidate
    directions_panel = [c for c in enriched if c["x"] < 700 and c["y"] < 900]
    if directions_panel:
        return directions_panel[0]
    return enriched[0]


def main() -> int:
    args = parse_args()
    sync_playwright, PlaywrightTimeoutError = import_playwright()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.set_default_timeout(args.timeout * 1000)

        try:
            page.goto(args.url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
        except PlaywrightTimeoutError:
            print("Timed out waiting for Google Maps to load.", file=sys.stderr)
            browser.close()
            return 1

        candidates = []
        deadline = time.monotonic() + min(float(args.timeout), 8.0)
        while time.monotonic() < deadline:
            candidates = scrape_candidates(page)
            if candidates:
                break
            page.wait_for_timeout(700)
        best = pick_best_guess(candidates)

        result = {
            "best_guess": best,
            "all_candidates": candidates[:25],
            "candidate_count": len(candidates),
            "url": args.url,
        }
        print(json.dumps(result, indent=2))
        browser.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
