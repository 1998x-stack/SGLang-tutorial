# SGLang Tutorial · 从源码到可视化推理系统

**读懂 SGLang：从请求生命周期、KV 与调度，深入分布式并行、MoE 专家通信、HiCache、多 LoRA 与多模态 EPD。**

[开始阅读](https://1998x-stack.github.io/SGLang-tutorial/) · [官方 SGLang](https://github.com/sgl-project/sglang) · [官方文档](https://docs.sglang.io/) · [原始论文](https://arxiv.org/abs/2312.07104)

> 本仓库是独立编写的中文源码导读教程，并非 SGLang 官方文档。内容以可追溯的论文与上游源码为依据；与运行版本相关的细节必须重新核对。

## 课程地图

| 章 | 核心问题 | 机制图 / 实验 | 证据 |
|---|---|---|---|
| [01 · 请求生命周期](01.html) | 一次生成请求经历哪些阶段？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [02 · RadixAttention](02.html) | 多请求如何共享一段已计算的前缀？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [03 · 动态调度与 overlap](03.html) | CPU 与 GPU 如何在迭代间并发工作？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [04 · Frontend 与结构化生成](04.html) | fork/join 与 grammar 如何改变生成？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [05 · Prefill–Decode 分离](05.html) | Prefill/Decode 分离引入哪些成本？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [06 · KV Pool 与物理页](06.html) | 请求行如何映射物理 KV 页？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [07 · RadixCache 内部机制](07.html) | RadixCache 如何匹配、拆分、锁定和淘汰？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [08 · Chunked Prefill](08.html) | 长 Prompt 如何根据预算切成 chunk？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [09 · Attention Backend](09.html) | 后端如何把页索引送进 attention kernel？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [10 · EAGLE 推测解码](10.html) | Draft Tree 如何被 Target 验证？ | 两幅独立 SVG + MathML + 交互 / CPU 实验 | 论文、官方文档、固定 commit 源码 |
| [11 · 分布式并行](11.html) | TP/PP/DP/EP 与 Attention-DP 如何切分 rank？ | 双 SVG + MathML + 交互 / CPU rank-group 实验 | 固定 commit 源码与 Megatron-LM |
| [12 · MoE Expert Parallelism](12.html) | Top-K token 如何跨 GPU dispatch、计算与 combine？ | 双 SVG + MathML + 交互 / CPU 路由实验 | 官方 EP 文档、DeepEP 与源码 |
| [13 · HiCache](13.html) | KV 如何在 GPU、Host 与 Storage 间保留和回迁？ | 双 SVG + MathML + 交互 / CPU tier 实验 | 官方设计文档、工程博客与源码 |
| [14 · Multi-LoRA](14.html) | 多 adapter 如何共享 base model 并管理 GPU slots？ | 双 SVG + MathML + 交互 / CPU pool 实验 | S-LoRA、Punica、官方文档与源码 |
| [15 · EPD](15.html) | Encoder、Prefill、Decode 如何独立部署？ | 双 SVG + MathML + 交互 / CPU capacity 实验 | 官方 EPD 文档与源码 |

## 第二批源码基线

第二批（06–10）以 SGLang 上游 `sgl-project/sglang@6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd` 为实现快照。该 SHA 仅表示核查时的主分支状态，不代表后续版本的默认行为。

关键源码入口：
- [ReqToTokenPool / KVCache / MHA pool](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/mem_cache/memory_pool.py)
- [PagedTokenToKVPoolAllocator](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/mem_cache/allocator/paged.py)
- [RadixCache](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/mem_cache/radix_cache.py)
- [PrefillAdder / SchedulePolicy](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/managers/schedule_policy.py)
- [Scheduler](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/managers/scheduler.py)
- [Attention backend resolver](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py)
- [EAGLE worker](https://github.com/sgl-project/sglang/blob/6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd/python/sglang/srt/speculative/eagle_worker_v2.py)

## 第三批源码基线

第三批（11–15）以 SGLang 上游 `sgl-project/sglang@ac0380c7faf9cece3ca32ffe75815af7c585acea` 为实现快照（2026-11-23）。本批覆盖并行 group、MoE EP、HiCache、LoRA 与 EPD；官方文档、运行时源码和论文历史贡献分开记录在 `evidence/11.json`–`evidence/15.json`。

关键入口：
- [Parallel groups](https://github.com/sgl-project/sglang/blob/ac0380c7faf9cece3ca32ffe75815af7c585acea/python/sglang/srt/distributed/parallel_state.py)
- [Expert Parallelism](https://github.com/sgl-project/sglang/blob/ac0380c7faf9cece3ca32ffe75815af7c585acea/docs/docs/advanced_features/expert_parallelism.mdx)
- [HiCache design](https://github.com/sgl-project/sglang/blob/ac0380c7faf9cece3ca32ffe75815af7c585acea/docs/docs/advanced_features/hicache_design.mdx)
- [LoRA manager](https://github.com/sgl-project/sglang/blob/ac0380c7faf9cece3ca32ffe75815af7c585acea/python/sglang/srt/lora/lora_manager.py)
- [EPD docs](https://github.com/sgl-project/sglang/blob/ac0380c7faf9cece3ca32ffe75815af7c585acea/docs/docs/advanced_features/epd_disaggregation.mdx)

## 可视化规范

- 每图回答一个明确问题；图形先定义数据依赖，再定义形状、边和颜色。
- 蓝绿色表示有效路径 / 输出，紫色表示缓存与复用，橙色表示计算 / 新分配，灰色表示等待 / 未采用。颜色必须与文字、线型同时表达语义。
- SVG 均以独立文件存放在 `assets/diagrams/`，包含 `viewBox/title/desc`；文章用 `img + figcaption`，默认状态无需 JS 也能理解。
- 公式使用原生 MathML；离线可读。交互必须改变真实教学状态，不仅改变颜色。
- 示例数值均为 **toy/derived**；性能数字必须给出模型、硬件、工作负载、指标和原始来源。

## 本地验证

```bash
python3 -m http.server 8000
python3 -m unittest discover -s tests -v
python3 examples/page_allocator.py
python3 examples/chunk_budget.py
python3 examples/online_softmax.py
python3 examples/speculative_sampling.py
python3 examples/parallel_topology.py
python3 examples/moe_dispatch.py
python3 examples/hicache_tiers.py
python3 examples/lora_pool.py
python3 examples/epd_pipeline.py
python3 scripts/browser_qa.py --out reports/browser-qa
```

## 研究资料

1. Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
2. Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
3. Ye et al., [FlashInfer](https://arxiv.org/abs/2501.01005), MLSys 2025.
4. Dao et al., [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023.
5. Li et al., [EAGLE-2](https://arxiv.org/abs/2406.16858), EMNLP 2024.
6. Li et al., [EAGLE-3](https://arxiv.org/abs/2503.01840), 2025.
7. Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2023.
8. Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), 2019.
9. Sheng et al., [S-LoRA](https://arxiv.org/abs/2311.03285), 2023.
10. Chen et al., [Punica](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html), MLSys 2024.
11. [SGLang HiCache engineering blog](https://www.sglang.io/blog/hicache) and [EPD documentation](https://docs.sglang.io/advanced_features/epd_disaggregation.html).

## 生产流程

先读 [AGENTS.md](AGENTS.md) 和本地 Skill；新选题先去重，再查原论文、官方文档和固定 commit 源码；先写证据清单和故事板，再做文章、SVG 和 CPU 实验。PR 应附静态检查、浏览器 QA、截图和版本说明；未运行的检查标记为 not verified。Pages 必须独立检查公开资源后才能称为已发布。
