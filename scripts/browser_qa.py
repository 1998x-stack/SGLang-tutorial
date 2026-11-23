#!/usr/bin/env python3
"""Chromium QA for all thirty chapters at desktop, tablet and mobile widths."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (1440, 768, 390, 360)
PAGES = ["index.html"] + [f"{i:02}.html" for i in range(1, 31)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/browser-qa")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for name in PAGES:
            for width in WIDTHS:
                page = browser.new_page(viewport={"width": width, "height": 900})
                page_errors, request_errors = [], []
                page.on("pageerror", lambda err: page_errors.append(str(err)))
                page.on("requestfailed", lambda req: request_errors.append(req.url))
                page.goto((ROOT / name).as_uri(), wait_until="load")
                page.evaluate("document.querySelectorAll('img').forEach(img => img.loading = 'eager')")
                page.wait_for_function("Array.from(document.images).every(img => img.complete)", timeout=10000)
                shape = page.evaluate("""() => ({
                    overflow: document.documentElement.scrollWidth - innerWidth,
                    images: [...document.images].map(x => ({
                        src: x.getAttribute('src'), complete: x.complete, width: x.naturalWidth
                    })),
                    math: [...document.querySelectorAll('math')].map(x => ({
                        namespace: x.namespaceURI, width: x.getBoundingClientRect().width,
                        height: x.getBoundingClientRect().height
                    })),
                    overflowNodes: [...document.querySelectorAll('body *')].map(x => {
                        const r = x.getBoundingClientRect(), s = getComputedStyle(x);
                        return {
                            tag: x.tagName.toLowerCase(), cls: String(x.className || ''),
                            text: (x.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 90),
                            left: Math.round(r.left), right: Math.round(r.right),
                            width: Math.round(r.width), scrollWidth: x.scrollWidth,
                            clientWidth: x.clientWidth, overflowX: s.overflowX
                        };
                    }).filter(x => x.width > 0 && (x.left < -1 || x.right > innerWidth + 1))
                      .sort((a, b) => b.right - a.right).slice(0, 12)
                })""")
                fail = []
                if shape["overflow"] > 1:
                    fail.append(f"document overflows horizontally: {shape['overflow']}px")
                for img in shape["images"]:
                    if not img["complete"] or img["width"] < 1:
                        fail.append(f"image not loaded: {img['src']}")
                if name != "index.html":
                    if not shape["math"]:
                        fail.append("MathML missing")
                    for item in shape["math"]:
                        if item["namespace"] != "http://www.w3.org/1998/Math/MathML":
                            fail.append("incorrect MathML namespace")
                        if item["width"] < 10 or item["height"] < 10:
                            fail.append("zero-sized MathML")
                    buttons = page.locator("[data-stepper] [data-step]")
                    if buttons.count() < 3:
                        fail.append("missing interactive controls")
                    else:
                        last = buttons.last
                        last.click()
                        if last.get_attribute("aria-pressed") != "true":
                            fail.append("stage button did not activate")
                        panels = page.locator("[data-stepper] [data-panel]")
                        if not panels.last.is_visible() or panels.first.is_visible():
                            fail.append("stage did not change visible explanation")
                else:
                    search = page.locator("#course-search")
                    search.fill("Chunked Prefill")
                    if page.locator(".course-card:not([hidden])").count() != 1:
                        fail.append("catalog search failed")
                    search.fill("")
                    page.locator('[data-filter="kernel"]').click()
                    if page.locator(".course-card:not([hidden])").count() != 1:
                        fail.append("category filter failed")
                    page.locator("[data-filter='mooncake']").click()
                    if page.locator(".course-card:not([hidden])").count() != 1:
                        fail.append("Mooncake category filter failed")
                    page.locator("[data-filter='all']").click()
                    if page.locator(".course-grid .course-card").count() != 30:
                        fail.append("course grid must contain all 30 chapter cards")
                    for category in ("quantization", "kv-quantization", "cuda-graph", "multimodal-pipeline", "observability", "structured-output", "tool-parser", "deterministic", "adaptive-spec", "checkpoint-engine"):
                        page.locator(f"[data-filter='{category}']").click()
                        if page.locator(".course-card:not([hidden])").count() != 1:
                            fail.append(f"{category} category filter failed")
                    page.locator("[data-filter='all']").click()
                fail.extend(page_errors)
                fail.extend(f"request failed: {url}" for url in request_errors)
                page.screenshot(path=str(out / f"{name[:-5] or 'index'}-{width}.png"), full_page=True)
                records.append({"page": name, "width": width, "passed": not fail, "errors": fail,
                                "images": len(shape["images"]), "math": len(shape["math"])})
                errors.extend(f"{name}@{width}: {message}" for message in fail)
                page.close()
        browser.close()
    (out / "report.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(records)} browser views, {len(errors)} issues; see {out}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
