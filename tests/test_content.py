"""Structural, evidence, SVG and deterministic CPU example tests."""
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from fractions import Fraction
from math import isclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from prefix_match import Namespace, RequestKey, longest_common_prefix, reusable_prefix
from page_allocator import ToyPagedAllocator
from chunk_budget import ChunkBudget, choose_chunk
from online_softmax import online_weighted_sum, full_weighted_sum
from speculative_sampling import acceptance_probability, residual_distribution, verify_candidate
from parallel_topology import model_parallel_groups, attention_tp_width, moe_tp_width
from moe_dispatch import dispatch_topk, combine_outputs, imbalance_ratio
from hicache_tiers import HiCacheToy, write_levels
from lora_pool import AdapterPool, PoolFull
from epd_pipeline import StageCapacity, end_to_end_latency_ms
from router_policy import Worker, power_of_two, cache_aware
from pd_lifecycle import Request as PDRequest, TransitionError
from mooncake_planes import ObjectRef, PlaneStore
from deepep_overlap import mode_for_phase, overlapped_time, skew
from lora_scheduler import AdapterPool as SchedulerAdapterPool, PoolFull as SchedulerPoolFull

REV_FIRST = "0a8693fbe3a2e46ccc06470a461a2b85865672fd"
REV_ADVANCED = "6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd"
REV_THIRD = "ac0380c7faf9cece3ca32ffe75815af7c585acea"
REV_FOURTH = "ac0243723b0dcba177f8792979daaa2f8d6ee6ea"


class Reader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = {}
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.setdefault(tag, []).append(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        self.links.extend(attrs[k] for k in ("href", "src") if k in attrs)


def read(name):
    reader = Reader()
    reader.feed((ROOT / name).read_text(encoding="utf-8"))
    return reader


class ArticleTest(unittest.TestCase):
    def test_course_map(self):
        page = read("index.html")
        self.assertEqual(len(page.tags.get("h1", [])), 1)
        self.assertEqual(sum("data-category" in x for x in page.tags.get("article", [])), 20)
        self.assertEqual(sum("data-filter" in x for x in page.tags.get("button", [])), 21)

    def test_depth_and_interaction_contract(self):
        for i in range(1, 21):
            name = f"{i:02}.html"
            with self.subTest(article=name):
                page = read(name)
                html = (ROOT / name).read_text(encoding="utf-8")
                self.assertEqual(len(page.tags.get("h1", [])), 1)
                self.assertGreaterEqual(len(page.tags.get("h2", [])), 8)
                self.assertGreaterEqual(len(page.tags.get("section", [])), 8)
                self.assertGreaterEqual(len(page.tags.get("img", [])), 2)
                self.assertGreaterEqual(len(page.tags.get("math", [])), 1)
                self.assertTrue(all(x.get("alt") for x in page.tags.get("img", [])))
                self.assertTrue(all(x.get("aria-label") for x in page.tags.get("math", [])))
                triggers = [x["data-step"] for x in page.tags.get("button", []) if "data-step" in x]
                panels = [x["data-panel"] for x in page.tags.get("div", []) if "data-panel" in x]
                self.assertGreaterEqual(len(triggers), 3)
                self.assertEqual(sorted(triggers), sorted(panels))
                self.assertNotIn("<svg", html)
                self.assertNotIn("MathJax", html)
                self.assertNotIn("<script src=\"https:", html)

    def test_local_links_are_resolved(self):
        names = ["index.html"] + [f"{i:02}.html" for i in range(1, 21)]
        for name in names:
            page = read(name)
            for link in page.links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                if not parsed.path:
                    if parsed.fragment:
                        self.assertIn(unquote(parsed.fragment), page.ids, f"{name}: {link}")
                    continue
                target = (ROOT / unquote(parsed.path)).resolve()
                self.assertTrue(target.is_relative_to(ROOT), f"unsafe local path: {link}")
                self.assertTrue(target.is_file(), f"{name}: missing {link}")

    def test_svg_manifest(self):
        files = list((ROOT / "assets/diagrams").glob("*.svg"))
        self.assertEqual(len(files), 40)
        ns = "{http://www.w3.org/2000/svg}"
        for path in files:
            with self.subTest(svg=path.name):
                root = ET.parse(path).getroot()
                self.assertEqual(root.tag, ns + "svg")
                self.assertTrue(root.get("viewBox"))
                self.assertTrue(root.find(ns + "title").text.strip())
                self.assertTrue(root.find(ns + "desc").text.strip())
                text_nodes = root.findall(".//" + ns + "text")
                self.assertGreaterEqual(len(text_nodes), 5)
                font_sizes = [float(node.get("font-size", "16")) for node in text_nodes]
                self.assertGreaterEqual(min(font_sizes), 14)
                self.assertFalse(root.findall(".//" + ns + "script"))

    def test_evidence_revision_and_claims(self):
        for i in range(1, 11):
            n = f"{i:02}"
            evidence = json.loads((ROOT / "evidence" / f"{n}.json").read_text())
            self.assertEqual(evidence["article"], n + ".html")
            expected = REV_FIRST if i <= 5 else (REV_ADVANCED if i <= 10 else (REV_THIRD if i <= 15 else REV_FOURTH))
            self.assertEqual(evidence["source_revision"], expected)
            self.assertGreaterEqual(len(evidence["claims"]), 4)
            for claim in evidence["claims"]:
                self.assertTrue(claim["claim"])
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

    def test_page_alignment(self):
        candidate = RequestKey((11, 12, 13, 73), self.ns)
        self.assertEqual(reusable_prefix(self.base, candidate, 2), 2)
        self.assertEqual(reusable_prefix(self.base, candidate, 4), 0)
        with self.assertRaises(ValueError):
            reusable_prefix(self.base, candidate, 0)

    def test_namespace_isolation(self):
        tenant = Namespace("model", "tok", tenant="B")
        adapter = Namespace("model", "tok", adapter="lora", tenant="A")
        self.assertEqual(reusable_prefix(self.base, RequestKey(self.base.tokens, tenant)), 0)
        self.assertEqual(reusable_prefix(self.base, RequestKey(self.base.tokens, adapter)), 0)


class PageAllocatorTest(unittest.TestCase):
    def test_request_row_is_not_physical_page(self):
        pool = ToyPagedAllocator(page_size=4)
        row_id = pool.create_request("A")
        self.assertEqual(row_id, 1)
        self.assertEqual(pool.append("A", 6), [8, 9, 10, 11, 16, 17])
        self.assertEqual(pool.append("A", 2), [18, 19])
        row = pool.row("A")
        self.assertEqual(row.page_ids, [2, 4])
        self.assertEqual(row.slots, [8, 9, 10, 11, 16, 17, 18, 19])
        self.assertEqual(pool.release("A"), [2, 4])
        self.assertEqual(pool.free_pages, [2, 4, 7, 9])

    def test_exhaustion_and_duplicate_requests(self):
        pool = ToyPagedAllocator(page_size=2, free_pages=(2,), rows=1)
        pool.create_request("A")
        with self.assertRaises(ValueError):
            pool.create_request("A")
        with self.assertRaises(MemoryError):
            pool.append("A", 3)
        self.assertEqual(pool.row("A").slots, [])
        self.assertEqual(pool.row("A").page_ids, [])
        self.assertEqual(pool.free_pages, [2])


class ChunkBudgetTest(unittest.TestCase):
    def test_variable_chunk_sizes(self):
        budgets = [
            ChunkBudget(12, 6, 4, 8),
            ChunkBudget(8, 5, 3, 8),
            ChunkBudget(5, 8, 6, 8),
        ]
        self.assertEqual([choose_chunk(x) for x in budgets], [4, 3, 5])

    def test_alignment_and_zero_progress(self):
        self.assertEqual(choose_chunk(ChunkBudget(10, 7, 7, alignment=4)), 4)
        self.assertEqual(choose_chunk(ChunkBudget(3, 1, 1, alignment=4)), 0)
        with self.assertRaises(ValueError):
            choose_chunk(ChunkBudget(3, 2, 2, alignment=0))


class OnlineSoftmaxTest(unittest.TestCase):
    def test_tiled_matches_full_softmax(self):
        scores = [2.0, -1.0, 3.5, 0.25]
        values = [10.0, 20.0, -4.0, 7.0]
        self.assertTrue(isclose(online_weighted_sum(scores, values),
                                full_weighted_sum(scores, values), rel_tol=1e-12, abs_tol=1e-12))


class SpeculativeSamplingTest(unittest.TestCase):
    def test_acceptance_and_residual(self):
        p = (Fraction(3, 5), Fraction(2, 5))
        q = (Fraction(2, 5), Fraction(3, 5))
        self.assertEqual(acceptance_probability(p[1], q[1]), Fraction(2, 3))
        self.assertFalse(verify_candidate(p, q, 1, Fraction(4, 5)))
        self.assertEqual(residual_distribution(p, q), (Fraction(1), Fraction(0)))
        self.assertTrue(verify_candidate(p, q, 0, Fraction(9, 10)))



class ParallelTopologyTest(unittest.TestCase):
    def test_documented_groups(self):
        g=model_parallel_groups(8,2,4)
        self.assertEqual(g["tp"],((0,1),(2,3),(4,5),(6,7)))
        self.assertEqual(g["pp"],((0,2,4,6),(1,3,5,7)))
    def test_nested_widths(self):
        self.assertEqual(attention_tp_width(8,1,2),4)
        self.assertEqual(moe_tp_width(8,4,2),1)
        with self.assertRaises(ValueError): model_parallel_groups(8,3,2)
class MoEDispatchTest(unittest.TestCase):
    def test_dispatch_and_combine(self):
        routes=[[(0,.7),(2,.3)],[(1,.4),(3,.6)]]
        buckets,loads=dispatch_topk(routes,{0:0,1:0,2:1,3:1})
        self.assertEqual(len(buckets[0]),2)
        combined=combine_outputs(routes,{(0,0):2,(0,2):4,(1,1):1,(1,3):3})
        self.assertAlmostEqual(combined[0],2.6)
        self.assertAlmostEqual(combined[1],2.2)
        self.assertEqual(imbalance_ratio(loads),1.0)
    def test_invalid_weights(self):
        with self.assertRaises(ValueError): dispatch_topk([[(0,.2)]],{0:0})
class HiCacheTierTest(unittest.TestCase):
    def test_l2_private_l3_shared_namespace(self):
        c=HiCacheToy(); c.put("A","doc","L2"); c.put("A","global","L3","shared")
        self.assertEqual(c.lookup("B","doc","shared")[0],"MISS")
        self.assertEqual(c.lookup("B","global","shared")[0],"L3")
    def test_write_policy(self):
        self.assertEqual(write_levels("write_through"),("L2","L3"))
        self.assertEqual(write_levels("write_through_selective",False),())
        self.assertEqual(write_levels("write_through_selective",True),("L2","L3"))
class LoRAPoolTest(unittest.TestCase):
    def test_lru_eviction(self):
        p=AdapterPool(3); p.acquire("A"); p.acquire("B")
        self.assertEqual(p.acquire("C"),"A")
        self.assertEqual(p.active,("<base>","B","C"))
    def test_pinned_blocks_eviction(self):
        p=AdapterPool(2); p.acquire("A",pinned=True)
        with self.assertRaises(PoolFull): p.acquire("B")
class EPDStageTest(unittest.TestCase):
    def test_bottleneck_after_scaling(self):
        self.assertEqual(StageCapacity(8,1,10,2,3,4).bottleneck(),("encoder",8))
        self.assertEqual(StageCapacity(8,2,10,2,3,4).bottleneck(),("decode",12))
    def test_latency(self):
        self.assertEqual(end_to_end_latency_ms(10,2,20,3,30,5),70)
        with self.assertRaises(ValueError): end_to_end_latency_ms(1,-1,1,1,1)


class RouterPolicyTest(unittest.TestCase):
    def test_cache_owner_and_admission(self):
        workers=[Worker("A","prefill",2048,800),Worker("B","prefill",512,100),Worker("C","prefill",0,500)]
        self.assertEqual(cache_aware(workers,min_prefix=1024,mode="prefer"),"A")
        blocked=[Worker("A","prefill",4096,9000),Worker("B","prefill",0,200)]
        self.assertEqual(cache_aware(blocked),"B")
    def test_balanced_switch_requires_both_thresholds(self):
        workers=[Worker("A","prefill",2048,3000),Worker("B","prefill",0,200)]
        self.assertEqual(cache_aware(workers,mode="balanced",factor=2,gap=1024),"B")
        workers=[Worker("A","prefill",2048,500),Worker("B","prefill",0,200)]
        self.assertEqual(cache_aware(workers,mode="balanced",factor=2,gap=1024),"A")
    def test_power_of_two_and_admission(self):
        workers=[Worker("A","decode",waiting=10),Worker("B","decode",waiting=2),Worker("C","decode",waiting=5)]
        self.assertEqual(power_of_two(workers),"B")

class PDLifecycleTest(unittest.TestCase):
    def test_early_reserves_before_prefill(self):
        r=PDRequest("early"); r.handshake()
        self.assertTrue(r.destination_allocated)
        r.finish_prefill(); r.transfer()
        self.assertTrue(r.transferred)
    def test_prefill_complete_delays_allocation(self):
        r=PDRequest("prefill_complete"); r.handshake()
        self.assertFalse(r.destination_allocated)
        with self.assertRaises(TransitionError): r.allocate_destination()
        r.finish_prefill(); r.allocate_destination(); r.transfer()
        self.assertTrue(r.transferred)
    def test_abort_prevents_transfer(self):
        r=PDRequest(); r.handshake(); r.abort()
        with self.assertRaises(TransitionError): r.finish_prefill()

class MooncakePlaneTest(unittest.TestCase):
    def test_payload_types_and_planes_do_not_alias(self):
        s=PlaneStore()
        pd=ObjectRef("pd-transfer","same","key","language-kv")
        hicache=ObjectRef("hicache-l3","same","key","kv-pages")
        mm=ObjectRef("epd-embedding","same","key","vit-embedding")
        s.put(pd,"request handoff"); s.put(hicache,"reusable pages"); s.put(mm,"encoder output")
        self.assertEqual(s.get(pd),"request handoff")
        self.assertEqual(s.get(hicache),"reusable pages")
        self.assertEqual(s.get(mm),"encoder output")
        self.assertEqual(len(s.objects),3)

class DeepEPOverlapTest(unittest.TestCase):
    def test_mode_by_phase(self):
        self.assertEqual(mode_for_phase("prefill"),"normal")
        self.assertEqual(mode_for_phase("decode"),"low_latency")
        self.assertEqual(mode_for_phase("mixed"),"auto")
    def test_overlap_bounds(self):
        self.assertEqual(overlapped_time(4,7,3),8)
        with self.assertRaises(ValueError): overlapped_time(4,7,5)
    def test_expert_skew(self):
        self.assertEqual(skew([10,10,10]),1.0)
        self.assertEqual(skew([10,10,40]),2.0)

class AdvancedLoRASchedulerTest(unittest.TestCase):
    def test_none_slot_and_lru_eviction(self):
        p=SchedulerAdapterPool(3); p.load("A",pinned=True); p.load("B"); p.load("C")
        self.assertEqual(list(p.slots.keys()),[None,"A","C"])
        self.assertTrue(p.distinct_ids_fit([None,"A","C"]))
        self.assertFalse(p.distinct_ids_fit([None,"A","C","D"]))
    def test_pinned_limit_and_exhaustion(self):
        p=SchedulerAdapterPool(2)
        p.load("A",pinned=True)
        with self.assertRaises(SchedulerPoolFull): p.load("B",pinned=True)
        with self.assertRaises(SchedulerPoolFull): p.load("C")

if __name__ == "__main__":
    unittest.main()
