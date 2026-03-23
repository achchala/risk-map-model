"""
Standalone Playwright worker for Google Maps ETA scraping.
Runs in a separate process to avoid asyncio loop conflicts.
"""

import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

_GOOGLE_DURATION_RE = re.compile(
    r"^(?:(?:\d+\s*(?:day|days|d)\s+)?)?(?:(?:\d+\s*hr\s*)?\d+\s*min|\d+\s*hr|\d+\s*h)$",
    re.IGNORECASE,
)

_worker_browser = None
_worker_playwright = None
_worker_timeout_error = None


def _worker_init():
    """Run in worker process: launch Playwright (no asyncio in fresh process)."""
    global _worker_browser, _worker_playwright, _worker_timeout_error
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage"],
    )
    _worker_playwright = pw
    _worker_browser = browser
    _worker_timeout_error = PlaywrightTimeoutError
    logging.info("[google-eta] Worker process: browser launched")


def _worker_duration_text_to_seconds(text: str) -> Optional[float]:
    normalized = (
        str(text)
        .lower()
        .replace("days", "d")
        .replace("day", "d")
        .replace("hours", "hr")
        .replace("hour", "hr")
        .replace(" h ", " hr ")
        .replace(" h,", " hr,")
        .replace(" h.", " hr.")
        .replace(" h", " hr")
        .replace("minutes", "min")
        .replace("minute", "min")
        .strip()
    )
    if not normalized:
        return None
    days = hours = minutes = 0
    day_match = re.search(r"(\d+)\s*d\b", normalized)
    hr_match = re.search(r"(\d+)\s*hr", normalized)
    min_match = re.search(r"(\d+)\s*min", normalized)
    if day_match:
        days = int(day_match.group(1))
    if hr_match:
        hours = int(hr_match.group(1))
    if min_match:
        minutes = int(min_match.group(1))
    total = float(days * 86400 + hours * 3600 + minutes * 60)
    return total if total > 0 else None


def _block_heavy(route):
    if route.request.resource_type in {"image", "media", "font"}:
        route.abort()
    else:
        route.continue_()


def _scrape_candidates(page):
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
            nodes.push({text, x: rect.x, y: rect.y, width: rect.width, height: rect.height,
              tag: parent.tagName, ariaLabel: parent.getAttribute("aria-label") || ""});
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


def _pick_best(candidates):
    enriched = []
    for item in candidates or []:
        text = str(item.get("text", "")).strip()
        normalized = (
            text.lower()
            .replace("days", "d")
            .replace("day", "d")
            .replace("hours", "hr")
            .replace("hour", "hr")
            .replace(" h ", " hr ").replace(" h", " hr")
            .replace("minutes", "min")
            .replace("minute", "min")
        )
        if not _GOOGLE_DURATION_RE.match(text):
            continue
        minutes = 0
        day_match = re.search(r"(\d+)\s*d\b", normalized)
        hr_match = re.search(r"(\d+)\s*hr", normalized)
        min_match = re.search(r"(\d+)\s*min", normalized)
        if day_match:
            minutes += int(day_match.group(1)) * 24 * 60
        if hr_match:
            minutes += int(hr_match.group(1)) * 60
        if min_match:
            minutes += int(min_match.group(1))
        enriched.append({**item, "minutes": minutes})
    if not enriched:
        return None
    directions_panel = [c for c in enriched if c["x"] < 700 and c["y"] < 900]
    return directions_panel[0] if directions_panel else enriched[0]


def worker_scrape_one(name: str, url: str, timeout_seconds: int = 10) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Scrape one Google Maps URL. Runs in worker process.
    Returns (name, result_dict, error_str). result_dict or error_str is None.
    """
    global _worker_browser, _worker_timeout_error
    if _worker_browser is None:
        return name, None, "Worker not initialized (Playwright not launched)"

    context = _worker_browser.new_context(viewport={"width": 1440, "height": 1200})
    context.route("**/*", _block_heavy)
    page = context.new_page()
    page.set_default_timeout(10000)

    try:
        logging.info("[google-eta] START %s scrape", name)
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(700)
        except _worker_timeout_error:
            return name, None, "Timed out waiting for Google Maps to load"

        candidates = []
        deadline = time.monotonic() + min(float(timeout_seconds), 4.0)
        while time.monotonic() < deadline:
            candidates = _scrape_candidates(page)
            if candidates:
                break
            page.wait_for_timeout(250)

        best_guess = _pick_best(candidates)
        seconds = None
        if best_guess and best_guess.get("minutes") is not None:
            parsed = float(best_guess["minutes"]) * 60.0
            if parsed > 0:
                seconds = parsed
        if seconds is None:
            seconds = _worker_duration_text_to_seconds((best_guess or {}).get("text", ""))

        if seconds is None:
            return name, None, f"No ETA found in Google Maps page (candidate_count={len(candidates)})"

        logging.info("[google-eta] SUCCESS %s scrape: %ss", name, int(seconds))
        return name, {
            "seconds": seconds,
            "best_guess": best_guess,
            "candidate_count": len(candidates),
            "url": url,
        }, None
    finally:
        try:
            page.close()
        except Exception:
            pass
        try:
            context.close()
        except Exception:
            pass
