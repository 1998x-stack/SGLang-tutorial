"""Check every published course page and generated asset after Pages deploy."""
import argparse
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
FILES = ["index.html"] + [f"{i:02}.html" for i in range(1, 46)]
FILES += ["assets/css/site.css", "assets/js/site.js"]
FILES += [f"assets/diagrams/{p.name}" for p in sorted((ROOT / "assets/diagrams").glob("*.svg"))]
FILES += [f"examples/{p.name}" for p in sorted((ROOT / "examples").glob("*.py"))]
FILES += [f"evidence/{i:02}.json" for i in range(1, 46)]

def verify(base_url):
    for path in FILES:
        url = urljoin(base_url.rstrip("/") + "/", path)
        for attempt in range(5):
            try:
                with urlopen(url, timeout=20) as response:
                    data = response.read()
                    if response.status != 200 or len(data) < 20:
                        raise ValueError("Empty or non-200 response")
                print("OK", path, len(data), "bytes")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    verify(parser.parse_args().base_url)
