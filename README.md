# SGLang Tutorial · 从源码到可视化推理系统

**读懂 SGLang：从请求生命周期、KV 与调度，深入分布式并行、MoE 专家通信、HiCache、多 LoRA、量化、CUDA Graph、结构化生成、Tool Parser、确定性推理与分布式 Checkpoint Engine。**

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
| [16 · KV-aware Router](16.html) | Bucket、Admission、Power-of-Two 与 cache/session affinity 如何协同？ | 双 SVG + MathML + 交互 / CPU policy 实验 | 固定 Router 源码、架构与 metrics |
| [17 · PD lifecycle](17.html) | Bootstrap、early/prefill_complete reservation、transfer 与 abort 如何协作？ | 双 SVG + MathML + 交互 / CPU 状态机 | 固定 PD docs、prefill/decode 与 backend 源码 |
| [18 · Mooncake data planes](18.html) | PD KV transfer、HiCache L3 与 EPD embeddings 有何区别？ | 双 SVG + MathML + 交互 / CPU object-plane 实验 | Mooncake Store、HiCache 与 EPD 源码 |
| [19 · DeepEP V2](19.html) | Normal/Low-Latency、ElasticBuffer、TBO/SBO 与 EPLB 如何协作？ | 双 SVG + MathML + 交互 / CPU overlap 实验 | SGLang EP docs、DeepEP 与固定 adapter |
| [20 · Multi-LoRA scheduler](20.html) | GPU slots、None base ID、pinning、drain 与 dynamic load 如何调度？ | 双 SVG + MathML + 交互 / CPU pool 实验 | LoRA docs、manager、memory pool 与 S-LoRA/Punica |
| [21 · Weight Quantization](21.html) | AWQ/GPTQ/FP8/FP4 如何从 checkpoint metadata 进入 layer method 与 GEMM backend？ | 双 SVG + MathML + 交互 / CPU per-row quant 实验 | Quantization docs、checkpoint resolver、method registry 与 AWQ/GPTQ |
| [22 · Quantized KV Cache](22.html) | FP8/FP4 KV 如何处理 scale、page layout 与 attention backend 读取？ | 双 SVG + MathML + 交互 / CPU block quant 与容量估算 | Quantized KV docs、KV cache method、FP4 access registry 与 KIVI |
| [23 · CUDA Graph](23.html) | Capture、Replay、Breakable segments 与 ViT graph cache 如何约束 shape/address？ | 双 SVG + MathML + 交互 / CPU graph-key 实验 | Breakable Graph、ViT graph、Foundry persistence 与 PyTorch CUDA semantics |
| [24 · Multimodal Processing](24.html) | Raw image/video/audio 如何经过 processor、token expansion 与 feature transport？ | 双 SVG + MathML + 交互 / CPU placeholder expansion | TokenSpaceProcessor、processor executor、expansion 与 feature transport |
| [25 · Observability & Benchmarking](25.html) | TTFT、TPOT、ITL、Prometheus 与四类 benchmark 应如何解释？ | 双 SVG + MathML + 交互 / CPU metrics 计算 | Production metrics、benchmark guide、metrics reporter 与 observability docs |
| [26 · XGrammar / Structured Generation](26.html) | JSON Schema / EBNF 如何编译成 grammar state 与 token mask？ | 双 SVG + MathML + 交互 / CPU grammar-mask 实验 | Structured Outputs docs、XGrammar backend、GrammarManager 与 MLSys 2025 |
| [27 · Function Calling / Tool Parser](27.html) | 模型专属 tool-call protocol 如何增量解析并归一化为 ToolCallItem？ | 双 SVG + MathML + 交互 / CPU streaming parser | Tool Parser docs、FunctionCallParser、BaseFormatDetector、Toolformer 与 Gorilla |
| [28 · Deterministic Inference](28.html) | Dynamic batch 为什么改变浮点 reduction，如何验证 batch-invariant execution？ | 双 SVG + MathML + 交互 / CPU float32 与 seeded sampler | Deterministic Inference docs、batch_invariant_ops、LLM-42 与 Vosti |
| [29 · Adaptive Speculative Decoding](29.html) | 接受长度 EMA、batch-size slots 与预构建 graph state 如何共同调整 EAGLE steps？ | 双 SVG + MathML + 交互 / CPU adaptive policy | Adaptive Speculative docs、adaptive_spec_params、AdaptiveController、BanditSpec |
| [30 · Distributed Checkpoint Engine](30.html) | 多节点 checkpoint 如何经 Parameter Server、Broadcast/P2P 和 IPC 进入 Ready 状态？ | 双 SVG + MathML + 交互 / CPU shard/readiness 实验 | Checkpoint Engine docs、checkpoint worker、weight updater 与 IPC endpoint |
| [31 · HiSparse](31.html) | 稀疏模型怎样按 top-k 调入活跃 KV，并把搬运藏在计算之后？ | 双 SVG + MathML + CPU hisparse_working_set | 固定 commit 官方文档与源码 |
| [32 · Hybrid KV Cache](32.html) | 混合模型为什么要分别为 token KV 和 recurrent checkpoints 分配内存？ | 双 SVG + MathML + CPU hybrid_pool_budget | 固定 commit 官方文档与源码 |
| [33 · KV Cache Sharing](33.html) | 相同 prefix 如何跨请求复用？CP ranks 如何只保存每页的一份持久副本？ | 双 SVG + MathML + CPU kv_sharing_pages | 固定 commit 官方文档与源码 |
| [34 · Paged Experts](34.html) | MoE 专家超出 GPU 容量时，如何按路由装载权重而不改变计算？ | 双 SVG + MathML + CPU paged_expert_waves | 固定 commit 官方文档与源码 |
| [35 · SGLang for RL](35.html) | 从采样到 trainer 再到 weight refit，如何保护 logprob、KV 与策略版本的一致性？ | 双 SVG + MathML + CPU rl_rollout_versions | 固定 commit 官方文档与源码 |
| [36 · Prefill Context Parallelism](36.html) | 长 Prompt 的 query 计算如何沿序列维分布，同时保留完整因果语义？ | 双 SVG + MathML + CPU prefill_cp_layout | 固定 commit 源码与官方文档 |
| [37 · Decode Context Parallelism](37.html) | 每个 GPU 只读取局部 KV，为什么还能得到完整 attention 输出？ | 双 SVG + MathML + CPU dcp_lse_merge | 固定 commit 源码与官方文档 |
| [38 · Session-Aware Radix Cache](38.html) | 如何保护活跃 Agent 的缓存，同时不把整个 KV 池锁死？ | 双 SVG + MathML + CPU session_eviction | 固定 commit 源码与官方文档 |
| [39 · Unified Memory](39.html) | 如何让 Hybrid 模型的 full KV、window KV 和 recurrent state 动态共享显存？ | 双 SVG + MathML + CPU unified_arena | 固定 commit 源码与官方文档 |
| [40 · Reasoning-Aware Compression](40.html) | 为什么剪枝后的模型可能更慢？怎样用实际生成轨迹校准激活？ | 双 SVG + MathML + CPU rac_calibration | 固定 commit 源码与官方文档 |
| [41 · Pipeline Parallelism](41.html) | 同一个长 Prompt 怎样让多个模型 stage 同时工作，而不破坏因果依赖？ | 双 SVG + MathML + CPU pipeline_chunking | 固定 commit 源码与官方文档 |
| [42 · Engram](42.html) | 大规模静态记忆如何从 token 历史得到 row IDs，并在 GPU/Host/Peer 间查表？ | 双 SVG + MathML + CPU engram_lookup | 固定 commit 源码与官方文档 |
| [43 · Radix Eviction Policies](43.html) | 缓存不足时应该先回收哪个 leaf？如何把重算代价和延迟目标变成可解释策略？ | 双 SVG + MathML + CPU radix_policy | 固定 commit 源码与官方文档 |
| [44 · R-Fork](44.html) | 如何从已经运行的模型实例装载同一权重，而不是重新经过完整 checkpoint 读取？ | 双 SVG + MathML + CPU rfork_readiness | 固定 commit 源码与官方文档 |
| [45 · SGLang Simulator](45.html) | 如何用真实 scheduler/cache 与替代 forward latency，在无模型权重情况下比较 serving 策略？ | 双 SVG + MathML + CPU serving_simulation | 固定 commit 源码与官方文档 |
| [46 · Logits Processor & Sampler](46.html) | 从 LM head 的 logits 到实际 token，采样分布与返回的 logprob 为什么可能不同？ | 双 SVG + MathML + CPU sampling_distribution | 固定 commit 源码与官方文档 |
| [47 · Model Loading](47.html) | 磁盘 checkpoint 怎样变成各 TP/PP rank 可运行的模型参数，为什么读取 shard 不等于 TP 切分？ | 双 SVG + MathML + CPU checkpoint_shards | 固定 commit 源码与官方文档 |
| [48 · Tokenizer Manager & Detokenizer](48.html) | 如何避免 CPU tokenization 堵塞事件循环，并保证流式文本不乱码、不重复？ | 双 SVG + MathML + CPU streaming_detokenize | 固定 commit 源码与官方文档 |
| [49 · Reasoning Control](49.html) | 怎样区分提示模型少思考、强制 token budget、结构化答案和输出解析？ | 双 SVG + MathML + CPU reasoning_budget | 固定 commit 源码与官方文档 |
| [50 · SGLang × Kubernetes / llm-d](50.html) | 多个 SGLang replicas 如何在 Gateway、Endpoint Picker、KV locality 与 autoscaler 之间协作？ | 双 SVG + MathML + CPU fleet_routing | 固定 commit 源码与官方文档 |
| [51 · Unified Kernel Architecture](51.html) | 一个算子怎样从 Python 调用进入适配的设备实现，并保证 shape、dtype、stride 与 stream 契约？ | 双 SVG + MathML + CPU kernel_contract | 固定 commit 源码与官方文档 |
| [52 · Layer Boundary](52.html) | Attention、FFN 与 MoE 之间如何保证 token rows、partial sums 和 residual 更新正确衔接？ | 双 SVG + MathML + CPU layer_boundary_contract | 固定 commit 源码与官方文档 |
| [53 · Hybrid Cache Checkpoints](53.html) | Full KV 命中更长前缀时，为什么 recurrent-state checkpoint 仍可能限制实际可复用长度？ | 双 SVG + MathML + CPU hybrid_checkpoint_reuse | 固定 commit 源码与官方文档 |
| [54 · HiCache Dynamic Storage](54.html) | 怎样在不重启的前提下安全切换 L3 后端，避免未完成写入、host lock 泄漏与 DP 状态分裂？ | 双 SVG + MathML + CPU storage_lifecycle | 固定 commit 源码与官方文档 |
| [55 · Model Object Storage](55.html) | 怎样从对象存储流式读取模型权重，控制启动时延、并发与峰值内存？ | 双 SVG + MathML + CPU object_streaming_budget | 固定 commit 源码与官方文档 |

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

## 第四批源码基线

第四批（16–20）以 SGLang 上游 sgl-project/sglang@ac0243723b0dcba177f8792979daaa2f8d6ee6ea 为实现快照（2026-11-23）。外部依赖分别固定为 SGLang DeepEP V2 / ElasticBuffer 分支 ece7e40b2f4e53a2910a442d2bbc33fe4130c3e0、独立的 DeepEP V2.5 / EPBuffer 分支 d30642958b04c4b1bd7b607ba7c551ea39e9140a，以及 Mooncake 8cd2bd5b45c36dc7520cc28267018aa8ae0645c1。Router、PD transfer、Mooncake L3、EPD embedding cache、DeepEP V1/V2 与 Multi-LoRA 的数据对象和运行路径分开记录在 evidence/16.json–evidence/20.json。

关键入口：
- [sgl-router README](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/sgl-router/README.md) · [architecture](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/sgl-router/ARCHITECTURE.md) · [metrics](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/sgl-router/monitoring/README.md)
- [PD disaggregation](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/docs/docs/advanced_features/pd_disaggregation.mdx) · [prefill.py](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/disaggregation/prefill.py) · [decode.py](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/disaggregation/decode.py)
- [Mooncake Store integration](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/mem_cache/storage/mooncake_store/README.md) · [EPD docs](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/docs/docs/advanced_features/epd_disaggregation.mdx)
- [Expert parallelism docs](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/docs/docs/advanced_features/expert_parallelism.mdx) · [DeepEP V1 adapter](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) · [DeepEP V2 adapter](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py)\n- [sgl-deep-ep / ElasticBuffer](https://github.com/sgl-project/DeepEP/tree/ece7e40b2f4e53a2910a442d2bbc33fe4130c3e0) · [sgl-deep-ep-v25 / EPBuffer family](https://github.com/sgl-project/DeepEP/tree/d30642958b04c4b1bd7b607ba7c551ea39e9140a)
- [LoRA docs](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/docs/docs/advanced_features/lora.mdx) · [manager](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/lora/lora_manager.py) · [memory pool](https://github.com/sgl-project/sglang/blob/ac0243723b0dcba177f8792979daaa2f8d6ee6ea/python/sglang/srt/lora/mem_pool.py)

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
python3 examples/router_policy.py
python3 examples/pd_lifecycle.py
python3 examples/mooncake_planes.py
python3 examples/deepep_overlap.py
python3 examples/lora_scheduler.py
python3 examples/weight_quantization.py
python3 examples/kv_cache_quantization.py
python3 examples/cuda_graph_capture.py
python3 examples/multimodal_pipeline.py
python3 examples/serving_metrics.py
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
11. [SGLang HiCache engineering blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/) and [EPD documentation](https://docs.sglang.io/advanced_features/epd_disaggregation.html).
12. [Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079), OSDI 2024.
13. [DeepEP repository](https://github.com/deepseek-ai/DeepEP) and [SGLang Large-Scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/).
14. [S-LoRA](https://arxiv.org/abs/2311.03285) and [Punica](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html).

## 生产流程

先读 [AGENTS.md](AGENTS.md) 和本地 Skill；新选题先去重，再查原论文、官方文档和固定 commit 源码；先写证据清单和故事板，再做文章、SVG 和 CPU 实验。PR 应附静态检查、浏览器 QA、截图和版本说明；未运行的检查标记为 not verified。Pages 必须独立检查公开资源后才能称为已发布。


## 第五批源码基线

第五批（21–25）固定上游实现快照为 `sgl-project/sglang@563813c0d60282fa848382fec600b40d28ddec98`（2026-11-23）。本批以 quantization / quantized KV、CUDA Graph、multimodal token-space processing、observability / benchmark 为主线；历史论文与当前实现分别记录在 `evidence/21.json`–`evidence/25.json`。

关键入口：
- [Quantization docs](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/quantization.mdx) · [checkpoint metadata resolver](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/model_loader/checkpoint_quantization.py) · [quantization registry](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/layers/quantization/__init__.py)
- [Quantized KV Cache docs](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/quantized_kv_cache.mdx) · [FP4 KV access registry](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py)
- [Breakable CUDA Graph](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/breakable_cuda_graph.mdx) · [ViT CUDA Graph](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx) · [Graph persistence](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/cuda_graph_persistence.mdx)
- [TokenSpaceProcessor](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/multimodal/token_space_mm_process/processor.py) · [placeholder expansion](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/multimodal/token_space_mm_process/expansion.py) · [feature transport](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/multimodal/transport/feature_transport.py)
- [Production metrics](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/references/production_metrics.mdx) · [benchmark and profiling](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/developer_guide/benchmark_and_profiling.mdx)

CPU-only teaching labs: `weight_quantization.py`, `kv_cache_quantization.py`, `cuda_graph_capture.py`, `multimodal_pipeline.py`, `serving_metrics.py`. They model invariants and arithmetic only; they do not implement SGLang CUDA kernels, media decoders, CUDA Graphs or a running metrics server.

## 第六批源码基线

第六批（26–30）固定上游实现快照为 `sgl-project/sglang@563813c0d60282fa848382fec600b40d28ddec98`（2026-11-23）。本批先对照 01–25 课程去重：第 26 章从第 04 章的前端 grammar 概念下沉到 XGrammar compiler/matcher 与 scheduler queue；第 29 章从第 10 章的 EAGLE 算法下沉到 runtime adaptive policy；其余主题分别补齐 Tool Parser、Deterministic Inference 与 Checkpoint Engine。

关键入口：
- [Structured Outputs](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/structured_outputs.mdx) · [reasoning model constraints](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/structured_outputs_for_reasoning_models.mdx) · [XGrammar backend](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/constrained/xgrammar_backend.py) · [GrammarManager](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/constrained/grammar_manager.py)
- [Tool Parser docs](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/tool_parser.mdx) · [FunctionCallParser](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/function_call/function_call_parser.py) · [BaseFormatDetector](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/function_call/base_format_detector.py) · [serving_chat](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/entrypoints/openai/serving_chat.py)
- [Deterministic Inference](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/deterministic_inference.mdx) · [batch invariant ops](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py)
- [Adaptive Speculative Decoding](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/adaptive_speculative_decoding.mdx) · [adaptive policy](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/speculative/adaptive_spec_params.py) · [runtime state](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/speculative/adaptive_runtime_state.py)
- [Checkpoint Engine](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/docs/docs/advanced_features/checkpoint_engine.mdx) · [IPC worker](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py) · [server-side weight updater](https://github.com/sgl-project/sglang/blob/563813c0d60282fa848382fec600b40d28ddec98/python/sglang/srt/managers/scheduler_components/weight_updater.py)

CPU-only labs: `grammar_mask.py`, `tool_parser.py`, `deterministic_inference.py`, `adaptive_spec.py`, `checkpoint_engine.py`. They verify finite-state constraints, chunk-boundary parsing, float32 order effects, EMA/hysteresis and rank readiness; they do not implement XGrammar, SGLang CUDA kernels, external tool execution, GPU IPC or a distributed Parameter Server.

## 第六批本地实验

```bash
python3 examples/grammar_mask.py
python3 examples/tool_parser.py
python3 examples/deterministic_inference.py
python3 examples/adaptive_spec.py
python3 examples/checkpoint_engine.py
```

这些实验均为 CPU-only 教学抽象，不依赖 GPU、真实 SGLang server、ZMQ/IPC 或外部工具执行。


## 第七批（31–35）

固定上游 SGLang 快照 `75ed36d976113c693f8e6f64925e119573a93fda`。HiSparse 仅卸载合适稀疏 KV；Hybrid 区分 SWA/full 与 recurrent state；官方 KV Sharing 指 prefill CP 页分片，不等于跨请求 radix prefix reuse；Paged Experts 管理权重而非 activation；RL rollout 需要 token-level behavior logprob、版本区间和缓存一致性。教学数字不代表实测。所有历史 evidence 保留原快照。

- 31 [HiSparse：从完整 Host KV 到 GPU Working Set](31.html) · [上游文档](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/docs/docs/advanced_features/hisparse_guide.mdx) · [实验](examples/hisparse_working_set.py) · [证据](evidence/31.json)
- 32 [Hybrid KV Cache：全局 KV、滑窗与循环状态](32.html) · [上游文档](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/docs/docs/advanced_features/hybrid_model_memory.mdx) · [实验](examples/hybrid_pool_budget.py) · [证据](evidence/32.json)
- 33 [KV Cache Sharing：前缀复用与 CP 页分片](33.html) · [上游文档](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/docs/docs/advanced_features/kv_cache_sharing.mdx) · [实验](examples/kv_sharing_pages.py) · [证据](evidence/33.json)
- 34 [Paged Experts：逻辑专家、物理槽与 Waves](34.html) · [上游文档](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/docs/docs/advanced_features/paged_experts.mdx) · [实验](examples/paged_expert_waves.py) · [证据](evidence/34.json)
- 35 [SGLang for RL：Rollout 数据、权重版本与引擎协同](35.html) · [上游文档](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/docs/docs/advanced_features/sglang_for_rl.mdx) · [实验](examples/rl_rollout_versions.py) · [证据](evidence/35.json)

回归目标：35 HTML、70 SVG、35 evidence、36 个首页筛选按钮、30 CPU labs；Chromium 4 个宽度共 144 views。GPU benchmark 未执行；CI 和 Pages 结果以实际运行记录为准。


## 第八批（36–40）

固定上游 SGLang `b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22`；前 35 章历史 evidence 保留原 revision。PCP 切 query 计算；DCP 切 KV 位置并精确 LSE 合并；session references 是软保护；Unified Memory 是共享 device arena 非 CUDA UVM；RAC 是离线剪枝校准。

- [Prefill Context Parallelism：Zigzag 与 Interleave](36.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22/docs/docs/advanced_features/prefill_cp.mdx) · [CPU 实验](examples/prefill_cp_layout.py) · [evidence](evidence/36.json)
- [Decode Context Parallelism：MLA 分片与精确 LSE 合并](37.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22/docs/docs/advanced_features/dcp.mdx) · [CPU 实验](examples/dcp_lse_merge.py) · [evidence](evidence/37.json)
- [Session-Aware Radix Cache：多轮会话的软保护与淘汰](38.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22/docs/docs/advanced_features/session_radix_cache.mdx) · [CPU 实验](examples/session_eviction.py) · [evidence](evidence/38.json)
- [Unified Memory：共享 Arena、动态 Bands 与安全 Compaction](39.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22/docs/docs/advanced_features/unified_memory.mdx) · [CPU 实验](examples/unified_arena.py) · [evidence](evidence/39.json)
- [Reasoning-Aware Compression：On-policy CoT 校准与模型剪枝](40.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22/docs/docs/advanced_features/reasoning_aware_compression.mdx) · [CPU 实验](examples/rac_calibration.py) · [evidence](evidence/40.json)

回归目标：40 HTML、80 SVG、40 evidence、41 filters、35 CPU labs；Chromium 1440/768/390/360 × 41 pages = 164 views。真实 GPU benchmark 未执行；CI / Pages 以实际运行记录为准。


## 第九批（41–45）

固定上游 SGLang `b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7`；前 40 章历史 evidence 保留原 revision。PP 研究 stage/chunk 依赖；Engram 是静态表；eviction 先筛可回收叶，再评分；R-Fork 是 remote weight loader；Simulator 复用调度并替换计算时延。

- [Pipeline Parallelism：Microbatch、动态 Chunking 与异步 P2P](41.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/pipeline_parallelism.mdx) · [CPU 实验](examples/pipeline_chunking.py) · [evidence](evidence/41.json)
- [Engram：条件记忆表、N-gram Hash 与分布式 Lookup](42.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/engram.mdx) · [CPU 实验](examples/engram_lookup.py) · [evidence](evidence/42.json)
- [Radix Eviction Policies：从 LRU 到 Tail Latency 与 State Cost](43.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/radix_eviction_policy.mdx) · [CPU 实验](examples/radix_policy.py) · [evidence](evidence/43.json)
- [R-Fork：Seed 实例、GPU-to-GPU 权重传输与 Ready Gate](44.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/rfork.mdx) · [CPU 实验](examples/rfork_readiness.py) · [evidence](evidence/44.json)
- [SGLang Simulator：真实调度逻辑、预测时钟与容量规划](45.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/sglang_simulator.mdx) · [CPU 实验](examples/serving_simulation.py) · [evidence](evidence/45.json)

回归目标：45 HTML、90 SVG、45 evidence、46 filters、40 CPU labs；Chromium 1440/768/390/360 × 46 pages = 184 views。GPU 和实际多卡仿真未实测；CI / Pages 以运行记录为准。


## 第十批（46–50）

固定上游 SGLang `b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7`，llm-d `9059ad4050a3e1713d14b954685dac87cfe932cc`；历史 evidence 保留原 revision。Sampling distribution 与 API logprob 区分；checkpoint 文件与 TP layout 区分；token、文本和 SSE 片段区分；reasoning 生成约束与输出 parser 分离；集群级 EPP 与引擎内部 scheduler 分离。

- [Logits Processor & Sampler：概率分布、截断与 Logprob 语义](46.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/python/sglang/srt/layers/logits_processor.py#L680-L838) · [CPU 实验](examples/sampling_distribution.py) · [evidence](evidence/46.json)
- [Model Loading：Checkpoint、Safetensors、Weight Loader 与分片](47.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/model_loading.mdx) · [CPU 实验](examples/checkpoint_shards.py) · [evidence](evidence/47.json)
- [Tokenizer Manager & Detokenizer：异步分词、增量解码与流式响应](48.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/python/sglang/srt/managers/tokenizer_manager.py#L1065-L1161) · [CPU 实验](examples/streaming_detokenize.py) · [evidence](evidence/48.json)
- [Reasoning Control：Thinking Budget、Reasoning Parser 与 Grammar 状态](49.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/thinking_budget.mdx) · [CPU 实验](examples/reasoning_budget.py) · [evidence](evidence/49.json)
- [SGLang × Kubernetes / llm-d：集群级推理调度与弹性部署](50.html) · [上游入口](https://github.com/sgl-project/sglang/blob/b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7/docs/docs/advanced_features/llm-d.mdx) · [CPU 实验](examples/fleet_routing.py) · [evidence](evidence/50.json)

回归目标：50 HTML、100 SVG、50 evidence、51 filters、45 CPU labs；Chromium 1440/768/390/360 × 51 pages = 204 views。GPU、真实模型和 Kubernetes 未部署/实测；CI / Pages 以运行记录为准。


## 第十一批（51–55）

固定上游 0fde0356cc44bca9a7727a3a2e2c2d10cd5bd30f。统一 kernel namespace、boundary contract、hybrid checkpoint、runtime L3 和 object streaming；保留历史证据版本。

- [Unified Kernel Architecture：AOT、JIT、Registry 与 Backend Selection](51.html) · [CPU lab](examples/kernel_contract.py) · [evidence](evidence/51.json)
- [Layer Boundary：布局契约、延迟归约与 Residual Stream](52.html) · [CPU lab](examples/layer_boundary_contract.py) · [evidence](evidence/52.json)
- [Hybrid Cache Checkpoints：Level 2/3、分叉与 Recurrent State 恢复](53.html) · [CPU lab](examples/hybrid_checkpoint_reuse.py) · [evidence](evidence/53.json)
- [HiCache Dynamic Storage：运行时 Attach/Detach 与一致性门禁](54.html) · [CPU lab](examples/storage_lifecycle.py) · [evidence](evidence/54.json)
- [Model Object Storage：Metadata、Safetensors Streaming 与分布式 I/O](55.html) · [CPU lab](examples/object_streaming_budget.py) · [evidence](evidence/55.json)

回归目标：55 HTML、110 SVG、55 evidence、56 filters、50 CPU labs、224 browser views。GPU、真实存储及 Kubernetes 未实测。
