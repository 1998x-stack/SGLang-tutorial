"""Check public GitHub Pages after successful Actions deploy."""
import argparse
import time
from urllib.request import urlopen
from urllib.parse import urljoin

FILES = ["index.html", "01.html", "05.html", "assets/css/site.css",
         "assets/js/site.js", "assets/diagrams/02-radix-tree.svg"]


def verify(base_url):
    for path in FILES:
        url = urljoin(base_url.rstrip("/") + "/", path)
        for attempt in range(5):
            try:
                with urlopen(url, timeout=20) as response:
                    data = response.read()
                    if response.status != 200 or len(data) < 200:
                        raise ValueError("Empty or non-200 response")
                print("OK", path, len(data), "bytes")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    verify(p.parse_args().base_url)
