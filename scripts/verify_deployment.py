"""Check public GitHub Pages after successful Actions deploy."""
import argparse
import time
from urllib.request import urlopen
from urllib.parse import urljoin

FILES = [
    "index.html", "01.html", "05.html", "06.html", "10.html",
    "assets/css/site.css", "assets/js/site.js",
    "assets/diagrams/06-memory-mapping.svg",
    "assets/diagrams/10-eagle-draft-tree.svg",
    "examples/prefix_match.py", "examples/page_allocator.py",
    "examples/chunk_budget.py", "examples/online_softmax.py",
    "examples/speculative_sampling.py",
    "evidence/01.json", "evidence/06.json", "evidence/10.json",
]


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
