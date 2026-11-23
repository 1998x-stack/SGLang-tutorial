"""Structural, evidence, SVG and deterministic prefix tests. Offline: no GPU."""
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from prefix_match import Namespace, RequestKey, longest_common_prefix, reusable_prefix  # noqa

REV = "0a8693fbe3a2e46ccc06470a461a2b85865672fd"


class Reader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = {}
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.setdefault(tag, []).append(a)
        if a.get("id"):
            self.ids.add(a["id"])
        self.links.extend(a[k] for k in ("href", "src") if k in a)


def read(name):
    p = Reader()
    p.feed((ROOT / name).read_text(encoding="utf-8"))
    return p


class ArticleTest(unittest.TestCase):
    def test_course_map(self):
        p = read("index.html")
        self.assertEqual(len(p.tags.get("h1", [])), 1)
        self.assertEqual(sum("data-category" in a for a in p.tags.get("article", [])), 5)
        self.assertEqual(sum("data-filter" in a for a in p.tags.get("button", [])), 6)

    def test_depth_and_interaction_contract(self):
        for i in range(1, 6):
            name = f"{i:02}.html"
            with self.subTest(article=name):
                p = read(name)
                self.assertEqual(len(p.tags.get("h1", [])), 1)
                self.assertGreaterEqual(len(p.tags.get("h2", [])), 8)
                self.assertGreaterEqual(len(p.tags.get("section", [])), 8)
                self.assertGreaterEqual(len(p.tags.get("img", [])), 2)
                self.assertGreaterEqual(len(p.tags.get("math", [])), 1)
                self.assertTrue(all(a.get("alt") for a in p.tags.get("img", [])))
                self.assertTrue(all(a.get("aria-label") for a in p.tags.get("math", [])))
                buttons = [a["data-step"] for a in p.tags.get("button", []) if "data-step" in a]
                panels = [a["data-panel"] for a in p.tags.get("div", []) if "data-panel" in a]
                self.assertGreaterEqual(len(buttons), 3)
                self.assertEqual(sorted(buttons), sorted(panels))
                html = (ROOT / name).read_text(encoding="utf-8")
                self.assertNotIn("<svg", html)
                self.assertNotIn("原创图解 · 单文件离线 HTML", html)
                self.assertNotIn("<script src=\"https:", html)

    def test_links_are_resolved(self):
        for name in ["index.html"] + [f"{i:02}.html" for i in range(1, 6)]:
            p = read(name)
            for link in p.links:
                ref = urlsplit(link)
                if ref.scheme or ref.netloc:
                    continue
                if not ref.path:
                    if ref.fragment:
                        self.assertIn(unquote(ref.fragment), p.ids, f"{name}: {link}")
                    continue
                target = (ROOT / unquote(ref.path)).resolve()
                self.assertTrue(target.is_relative_to(ROOT), f"unsafe local path {link}")
                self.assertTrue(target.is_file(), f"{name}: missing {link}")

    def test_standalone_svg(self):
        files = list((ROOT / "assets/diagrams").glob("*.svg"))
        self.assertEqual(len(files), 10)
        ns = "{http://www.w3.org/2000/svg}"
        for f in files:
            with self.subTest(file=f.name):
                root = ET.parse(f).getroot()
                self.assertEqual(root.tag, ns + "svg")
                self.assertTrue(root.get("viewBox"))
                self.assertTrue(root.find(ns + "title").text.strip())
                self.assertTrue(root.find(ns + "desc").text.strip())
                self.assertGreaterEqual(len(root.findall(".//" + ns + "text")), 5)
                self.assertFalse(root.findall(".//" + ns + "script"))

    def test_claims(self):
        for i in range(1, 6):
            n = f"{i:02}"
            ledger = json.loads((ROOT / "evidence" / (n + ".json")).read_text())
            self.assertEqual(ledger["article"], n + ".html")
            self.assertEqual(ledger["source_revision"], REV)
            self.assertGreaterEqual(len(ledger["claims"]), 4)
            for claim in ledger["claims"]:
                self.assertTrue(claim["locator"])
                self.assertTrue(claim["source_url"].startswith("https://"))
                self.assertIn(claim["status"], {"verified", "derived", "illustrative", "unverified"})


class PrefixTest(unittest.TestCase):
    def setUp(self):
        self.ns = Namespace("model", "tok", tenant="A")
        self.base = RequestKey((11, 12, 13, 91, 92), self.ns)

    def test_lcp(self):
        self.assertEqual(longest_common_prefix(self.base.tokens, (11, 12, 13, 73)), 3)
        self.assertEqual(reusable_prefix(self.base, RequestKey(self.base.tokens, self.ns)), 5)
        self.assertEqual(reusable_prefix(self.base, RequestKey((42, 11), self.ns)), 0)
        self.assertEqual(reusable_prefix(self.base, RequestKey((), self.ns)), 0)

    def test_alignment(self):
        candidate = RequestKey((11, 12, 13, 73), self.ns)
        self.assertEqual(reusable_prefix(self.base, candidate, 2), 2)
        self.assertEqual(reusable_prefix(self.base, candidate, 4), 0)
        with self.assertRaises(ValueError):
            reusable_prefix(self.base, candidate, 0)

    def test_namespace(self):
        tenant = Namespace("model", "tok", tenant="B")
        adapter = Namespace("model", "tok", adapter="lora", tenant="A")
        self.assertEqual(reusable_prefix(self.base, RequestKey(self.base.tokens, tenant)), 0)
        self.assertEqual(reusable_prefix(self.base, RequestKey(self.base.tokens, adapter)), 0)


if __name__ == "__main__":
    unittest.main()
