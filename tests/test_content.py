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
from weight_quantization import quantize_row, quantize_matrix_per_row, estimate_packed_bytes, mse
from kv_cache_quantization import bf16_bytes, fp8_bytes, nvfp4_bytes, block_quantize, capacity_multiplier, attention_access
from cuda_graph_capture import GraphKey, GraphCache, replay_segments
from multimodal_pipeline import MediaItem, expand_placeholders, transport_target, token_count
from serving_metrics import request_metrics, percentile, throughput, benchmark_layer
from grammar_mask import ToyGrammar, GrammarCache
from tool_parser import ToyToolStreamParser
from deterministic_inference import reduce_float32, batch_invariant_sum, seeded_choice
from adaptive_spec import AdaptiveSlot, AdaptivePolicy, RuntimeStatePool
from checkpoint_engine import split_shards, transfer_plan, WeightReadiness

REV_FIRST = "0a8693fbe3a2e46ccc06470a461a2b85865672fd"
REV_ADVANCED = "6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd"
REV_THIRD = "ac0380c7faf9cece3ca32ffe75815af7c585acea"
REV_FOURTH = "ac0243723b0dcba177f8792979daaa2f8d6ee6ea"
REV_FIFTH = "563813c0d60282fa848382fec600b40d28ddec98"
REV_SIXTH = "563813c0d60282fa848382fec600b40d28ddec98"
REV_SEVENTH = "75ed36d976113c693f8e6f64925e119573a93fda"
REV_EIGHTH = "b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22"
REV_NINTH = "b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7"
REV_TENTH = "b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7"
REV_ELEVENTH = "0fde0356cc44bca9a7727a3a2e2c2d10cd5bd30f"


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
        self.assertEqual(sum("data-category" in x for x in page.tags.get("article", [])), 55)
        self.assertEqual(sum("data-filter" in x for x in page.tags.get("button", [])), 64)

    def test_depth_and_interaction_contract(self):
        for i in range(1, 56):
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
        names = ["index.html"] + [f"{i:02}.html" for i in range(1, 56)]
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
        self.assertEqual(len(files), 110)
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
        for i in range(1, 56):
            n = f"{i:02}"
            evidence = json.loads((ROOT / "evidence" / f"{n}.json").read_text())
            self.assertEqual(evidence["article"], n + ".html")
            expected = REV_FIRST if i <= 5 else (REV_ADVANCED if i <= 10 else (REV_THIRD if i <= 15 else (REV_FOURTH if i <= 20 else (REV_FIFTH if i <= 25 else (REV_SIXTH if i <= 30 else (REV_SEVENTH if i <= 35 else (REV_EIGHTH if i <= 40 else (REV_NINTH if i <= 45 else (REV_TENTH if i <= 50 else REV_ELEVENTH)))))))))
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


class WeightQuantizationTest(unittest.TestCase):
    def test_lower_bits_trade_storage_for_error(self):
        values = [0.1, -0.7, 1.2, 0.4, -1.8, 0.25]
        _, _, r4 = quantize_row(values, 4)
        _, _, r8 = quantize_row(values, 8)
        self.assertLess(mse(values, r8), mse(values, r4))
        self.assertLess(estimate_packed_bytes(128, 4), estimate_packed_bytes(128, 8))

    def test_per_row_scales_and_invalid_input(self):
        matrix = [[0.1, 0.2], [10.0, -5.0]]
        q, scales, restored = quantize_matrix_per_row(matrix, 4)
        self.assertEqual(len(q), 2)
        self.assertNotEqual(scales[0], scales[1])
        self.assertEqual(len(restored[1]), 2)
        with self.assertRaises(ValueError):
            quantize_row([], 4)


class KVCacheQuantizationTest(unittest.TestCase):
    def test_scale_overhead_is_included(self):
        self.assertEqual(bf16_bytes(1024), 2048)
        self.assertEqual(fp8_bytes(1024), 1028)
        self.assertEqual(nvfp4_bytes(1024, 16, 4), 580)
        self.assertAlmostEqual(capacity_multiplier(2, 0.5 + 1/16), 3.5555555, places=5)

    def test_block_quantization_and_backend_access(self):
        values = [0.1, -0.4, 0.7, 1.2] * 8
        restored, scales = block_quantize(values, 8)
        self.assertEqual(len(restored), len(values))
        self.assertEqual(len(scales), 4)
        self.assertEqual(attention_access(True, False), "fused")
        self.assertEqual(attention_access(False, True), "dequant-workspace")
        with self.assertRaises(RuntimeError):
            attention_access(False, False)


class CudaGraphCaptureTest(unittest.TestCase):
    def test_graph_key_and_static_buffer_identity(self):
        cache = GraphCache()
        key = GraphKey((8, 1024), (0, 512, 1024))
        self.assertEqual(cache.run(key, "static-A"), "capture")
        self.assertEqual(cache.run(key, "static-A"), "replay")
        with self.assertRaises(RuntimeError):
            cache.run(key, "new-address")
        other = GraphKey((16, 1024), (0, 1024))
        self.assertEqual(cache.run(other, "static-B"), "capture")
        self.assertEqual(cache.graph_count, 2)

    def test_breakable_replay_order(self):
        self.assertEqual(
            replay_segments(["attention", "mlp"], ["dynamic-op"]),
            ["graph:attention", "eager:dynamic-op", "graph:mlp"],
        )
        with self.assertRaises(ValueError):
            replay_segments(["one"], ["extra-break"])


class MultimodalPipelineTest(unittest.TestCase):
    def test_placeholder_expansion_preserves_order(self):
        ids = [1, 900, 2, 901, 3]
        expanded = expand_placeholders(ids, {900: [[10, 11]], 901: [[20, 21, 22]]})
        self.assertEqual(expanded, [1, 10, 11, 2, 20, 21, 22, 3])

    def test_partial_prefix_and_transport(self):
        ids = [10, 11, 900, 12]
        self.assertEqual(expand_placeholders(ids, {900: [[1, 2]]}, start_len=2), [10, 11, 1, 2, 12])
        items = [MediaItem("image", 900, (1, 2, 3)), MediaItem("video", 901, (4, 5))]
        self.assertEqual(token_count(items), 5)
        self.assertEqual(transport_target("cuda_ipc"), "gpu")
        self.assertEqual(transport_target("cpu"), "cpu")
        with self.assertRaises(ValueError):
            transport_target("rdma")


class ServingMetricsTest(unittest.TestCase):
    def test_request_latency_definitions(self):
        result = request_metrics(0.0, 0.2, 1.2, 6)
        self.assertAlmostEqual(result["ttft"], 0.2)
        self.assertAlmostEqual(result["tpot"], 0.2)
        self.assertAlmostEqual(result["e2e"], 1.2)
        self.assertIsNone(request_metrics(0.0, 0.2, 0.5, 1)["tpot"])

    def test_percentile_throughput_and_benchmark_layers(self):
        self.assertEqual(percentile([0.1, 0.2, 0.3, 0.4], 0.95), 0.4)
        self.assertEqual(throughput(100, 5), 20)
        self.assertEqual(benchmark_layer("bench_serving")[0], "http")
        self.assertEqual(benchmark_layer("bench_one_batch")[1], "ModelRunner")
        with self.assertRaises(ValueError):
            benchmark_layer("unknown")



class GrammarMaskTest(unittest.TestCase):
    def test_allowed_tokens_rejection_and_rollback(self):
        grammar = ToyGrammar()
        self.assertEqual(grammar.allowed_tokens(), frozenset({"{"}))
        self.assertFalse(grammar.accept("true"))
        self.assertTrue(grammar.accept("{"))
        self.assertEqual(grammar.allowed_tokens(), frozenset({'"ok"'}))
        self.assertTrue(grammar.accept('"ok"'))
        grammar.rollback()
        self.assertEqual(grammar.allowed_tokens(), frozenset({'"ok"'}))

    def test_complete_sequence_and_cache(self):
        grammar = ToyGrammar()
        for token in ['{', '"ok"', ':', 'true', ',', '"count"', ':', '2', '}']:
            self.assertIn(token, grammar.allowed_tokens())
            self.assertTrue(grammar.accept(token))
        self.assertTrue(grammar.terminated)
        cache = GrammarCache()
        self.assertEqual(cache.get("schema-A"), cache.get("schema-A"))
        self.assertEqual(cache.compiles, 1)


class ToolParserTest(unittest.TestCase):
    def test_partial_marker_json_and_final_flush(self):
        parser = ToyToolStreamParser()
        normal, calls = parser.feed("Searching <tool_")
        self.assertEqual(normal, "Searching ")
        self.assertEqual(calls, [])
        normal, calls = parser.feed('call>{"name":"search",')
        self.assertEqual(normal, "")
        self.assertEqual(calls, [])
        normal, calls = parser.feed('"arguments":{"q":"SGLang"}}</tool_call> done.')
        self.assertEqual(normal, " done.")
        self.assertEqual(calls, [{"name": "search", "arguments": {"q": "SGLang"}}])
        self.assertEqual(parser.finish(), "")

    def test_multiple_calls_and_malformed_fallback(self):
        parser = ToyToolStreamParser()
        normal, calls = parser.feed(
            '<tool_call>{"name":"a","arguments":{}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
        )
        self.assertEqual(normal, "")
        self.assertEqual([x["name"] for x in calls], ["a", "b"])
        parser = ToyToolStreamParser()
        normal, calls = parser.feed("<tool_call>{bad}</tool_call>")
        self.assertEqual(calls, [])
        self.assertIn("{bad}", normal)


class DeterministicInferenceTest(unittest.TestCase):
    def test_float32_reduction_order_changes_result(self):
        values = [1e20, -1e20, 3.14]
        left = reduce_float32(values, [0, 1, 2])
        right = reduce_float32(values, [1, 2, 0])
        self.assertNotEqual(left, right)
        self.assertEqual(batch_invariant_sum(values), left)

    def test_per_request_seed_is_reproducible(self):
        self.assertEqual(seeded_choice("ABC", 42), seeded_choice("ABC", 42))
        self.assertIn(seeded_choice("ABC", 42), "ABC")


class AdaptiveSpecTest(unittest.TestCase):
    def test_ema_moves_up_and_down_with_hysteresis(self):
        slot = AdaptiveSlot(candidate_steps=(1, 3, 5, 7), current_steps=3, alpha=1)
        self.assertTrue(slot.update([4, 4, 4]))
        self.assertEqual(slot.current_steps, 5)
        self.assertTrue(slot.update([0, 0, 0]))
        self.assertEqual(slot.current_steps, 1)

    def test_batch_slots_and_prebuilt_state_pool(self):
        small = AdaptiveSlot(candidate_steps=(1, 3, 5), current_steps=1, alpha=1)
        large = AdaptiveSlot(candidate_steps=(1, 3), current_steps=1, alpha=1)
        policy = AdaptivePolicy({1: small, 8: large})
        self.assertIs(policy.slot_for(2), small)
        self.assertIs(policy.slot_for(16), large)
        pool = RuntimeStatePool([1, 3, 5])
        pool.activate(3)
        pool.activate(3)
        self.assertEqual(pool.swaps, 1)
        with self.assertRaises(KeyError):
            pool.activate(7)


class CheckpointEngineTest(unittest.TestCase):
    def test_shard_split_and_transfer_modes(self):
        assignments = split_shards(["c", "a", "b", "d"], 2)
        self.assertEqual(assignments, [["a", "c"], ["b", "d"]])
        self.assertEqual(transfer_plan("broadcast", 2, 4), [("broadcast", 2, 4)])
        self.assertEqual(len(transfer_plan("all", 2, 4)), 2)
        with self.assertRaises(ValueError):
            transfer_plan("unknown", 2, 4)

    def test_admin_auth_and_rank_readiness_barrier(self):
        readiness = WeightReadiness(expected_ranks=2, admin_key="secret")
        with self.assertRaises(PermissionError):
            readiness.accept_rank(0, "weights", "wrong")
        readiness.accept_rank(0, "weights-0", "secret")
        self.assertFalse(readiness.ready)
        readiness.accept_rank(1, "weights-1", "secret")
        self.assertTrue(readiness.ready)
        with self.assertRaises(ValueError):
            readiness.accept_rank(2, "weights-2", "secret")




class SeventhBatchTest(unittest.TestCase):
    def test_hisparse_working_set(self):
        from hisparse_working_set import selftest
        selftest()

    def test_hybrid_pool_budget(self):
        from hybrid_pool_budget import selftest
        selftest()

    def test_kv_sharing_pages(self):
        from kv_sharing_pages import selftest
        selftest()

    def test_paged_expert_waves(self):
        from paged_expert_waves import selftest
        selftest()

    def test_rl_rollout_versions(self):
        from rl_rollout_versions import selftest
        selftest()



class EighthBatchTest(unittest.TestCase):
    def test_prefill_cp_layout(self):
        from prefill_cp_layout import selftest
        selftest()

    def test_dcp_lse_merge(self):
        from dcp_lse_merge import selftest
        selftest()

    def test_session_eviction(self):
        from session_eviction import selftest
        selftest()

    def test_unified_arena(self):
        from unified_arena import selftest
        selftest()

    def test_rac_calibration(self):
        from rac_calibration import selftest
        selftest()


class NinthBatchTest(unittest.TestCase):
    def test_pipeline_chunking(self):
        from pipeline_chunking import selftest
        selftest()

    def test_engram_lookup(self):
        from engram_lookup import selftest
        selftest()

    def test_radix_policy(self):
        from radix_policy import selftest
        selftest()

    def test_rfork_readiness(self):
        from rfork_readiness import selftest
        selftest()

    def test_serving_simulation(self):
        from serving_simulation import selftest
        selftest()


class TenthBatchTest(unittest.TestCase):
    def test_sampling_distribution(self):
        from sampling_distribution import selftest
        selftest()

    def test_checkpoint_shards(self):
        from checkpoint_shards import selftest
        selftest()

    def test_streaming_detokenize(self):
        from streaming_detokenize import selftest
        selftest()

    def test_reasoning_budget(self):
        from reasoning_budget import selftest
        selftest()

    def test_fleet_routing(self):
        from fleet_routing import selftest
        selftest()


class EleventhBatchTest(unittest.TestCase):
    def test_kernel_contract(self):
        from kernel_contract import selftest
        selftest()

    def test_layer_boundary_contract(self):
        from layer_boundary_contract import selftest
        selftest()

    def test_hybrid_checkpoint_reuse(self):
        from hybrid_checkpoint_reuse import selftest
        selftest()

    def test_storage_lifecycle(self):
        from storage_lifecycle import selftest
        selftest()

    def test_object_streaming_budget(self):
        from object_streaming_budget import selftest
        selftest()

if __name__ == "__main__":
    unittest.main()
