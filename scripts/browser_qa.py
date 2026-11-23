#!/usr/bin/env python3
"""Real Chromium QA for every chapter at desktop/tablet/mobile widths."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (1440, 768, 390, 360)
PAGES = ["index.html"] + [f"{i:02}.html" for i in range(1, 6)]


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
                page_errors = []
                page.on("pageerror", lambda err: page_errors.append(str(err)))
                page.goto((ROOT / name).as_uri(), wait_until="load")
                shape = page.evaluate("""() => ({
                    overflow: document.documentElement.scrollWidth - innerWidth,
                    images: [...document.images].map(x => ({
                        src: x.getAttribute('src'), complete: x.complete, width: x.naturalWidth
                    })),
                    math: [...document.querySelectorAll('math')].map(x => ({
                        namespace: x.namespaceURI, width: x.getBoundingClientRect().width,
                        height: x.getBoundingClientRect().height
                    }))
                })""")
                fail = []
                if shape["overflow"] > 1:
                    fail.append(f"document overflows horizontally: {shape['overflow']}px")
                for image in shape["images"]:
                    if not image["complete"] or image["width"] < 1:
                        fail.append(f"image not loaded: {image['src']}")
                if name != "index.html":
                    if not shape["math"]:
                        fail.append("MathML missing")
                    for m in shape["math"]:
                        if m["namespace"] != "http://www.w3.org/1998/Math/MathML":
                            fail.append("incorrect MathML namespace")
                        if m["width"] < 10 or m["height"] < 10:
                            fail.append("zero-sized MathML")
                    buttons = page.locator("[data-stepper] [data-step]")
                    if buttons.count() < 3:
                        fail.append("missing interaction")
                    else:
                        last = buttons.last
                        last.click()
                        if last.get_attribute("aria-pressed") != "true":
                            fail.append("stage button not active")
                        panels = page.locator("[data-stepper] [data-panel]")
                        if not panels.last.is_visible() or panels.first.is_visible():
                            fail.append("stage did not alter visible explanation")
                else:
                    search = page.locator("#course-search")
                    search.fill("Radix")
                    if page.locator(".course-card:not([hidden])").count() != 1:
                        fail.append("catalog search failed")
                    search.fill("")
                    page.locator('[data-filter="scheduler"]').click()
                    if page.locator(".course-card:not([hidden])").count() != 1:
                        fail.append("category filter failed")
                fail.extend(page_errors)
                page.screenshot(path=str(out / f"{name[:-5]}-{width}.png"), full_page=True)
                records.append({"page": name, "width": width, "passed": not fail, "errors": fail,
                                "images": len(shape["images"]), "math": len(shape["math"])})
                errors.extend(f"{name}@{width}: {e}" for e in fail)
                page.close()
        browser.close()
    (out / "report.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(records)} browser views, {len(errors)} issues; see {out}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
