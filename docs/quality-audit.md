# 全仓库质量治理 · 2026-11-25

基线 9722139a25a04c410dc8b6f4396fa7bbd1cf3b78。逐章提取了 55 页的 MathML、图片、依赖和实验入口；结构检查不等同于逐 claim 语义复核。

- 8 个一级领域，55 个细分标签（默认折叠），共64 filters；课程图为 DAG，跨领域依赖允许。
- 每章自动展示前置与反向依赖。新增脚本检查公式 XML arity、重复 ID、图归属、历史证据 SHA 与移动链接、依赖和 manifest 同步；Chromium 逐 SVG 检查文字 bbox 和 marker path length。
- 历史 SHA 保留，51 章记录统一 kernel 命名空间迁移；36/37 已存在有效箭头坐标。第40章 ICLR URL 已与 arXiv:2509.12464 交叉核验。

| 章 | 领域 | 前置章节 | 审查状态 |
|---|---|---|---|
| 01 从请求到 token | 推理基础与服务生命周期 | — | 公式/图清单已读；完整逐 claim 审查待办 |
| 02 RadixAttention：共享前缀不是魔法 | KV Cache 与内存系统 | 1 | 公式/图清单已读；完整逐 claim 审查待办 |
| 03 动态批处理与 CPU–GPU Overlap | 推理基础与服务生命周期 | 1 | 公式/图清单已读；完整逐 claim 审查待办 |
| 04 SGLang Frontend 与结构化生成 | 推理基础与服务生命周期 | 1 | 公式/图清单已读；完整逐 claim 审查待办 |
| 05 Prefill–Decode 分离 | 并行与分布式执行 | 1, 3 | 公式/图清单已读；完整逐 claim 审查待办 |
| 06 KV Pool：从请求行到物理页 | KV Cache 与内存系统 | 1, 2 | 公式/图清单已读；完整逐 claim 审查待办 |
| 07 RadixCache：拆分、锁引用与淘汰 | KV Cache 与内存系统 | 2, 6 | 公式/图清单已读；完整逐 claim 审查待办 |
| 08 Chunked Prefill：预算驱动的长输入切片 | 推理基础与服务生命周期 | 3, 6 | 公式/图清单已读；完整逐 claim 审查待办 |
| 09 Attention Backend：从页表到 Kernel | 算子与 GPU 执行 | 6, 8 | 公式/图清单已读；完整逐 claim 审查待办 |
| 10 EAGLE：Draft Tree 与 Target Verification | 生成、采样与约束 | 1, 3, 9 | 公式/图清单已读；完整逐 claim 审查待办 |
| 11 TP、PP、DP、EP 与 Attention-DP | 并行与分布式执行 | 1, 5 | 公式/图清单已读；完整逐 claim 审查待办 |
| 12 MoE Expert Parallelism 与 DeepEP | 并行与分布式执行 | 11 | 公式/图清单已读；完整逐 claim 审查待办 |
| 13 HiCache：GPU、Host 与 Storage | KV Cache 与内存系统 | 6, 7 | 公式/图清单已读；完整逐 claim 审查待办 |
| 14 Multi-LoRA Serving | 模型与扩展运行时 | 1, 11 | 公式/图清单已读；完整逐 claim 审查待办 |
| 15 EPD：Encoder、Prefill、Decode 分离 | 并行与分布式执行 | 5, 11 | 公式/图清单已读；完整逐 claim 审查待办 |
| 16 SGLang Router：Admission、Prefix 与 Session | 路由、RL 与集群运维 | 2, 3, 5 | 公式/图清单已读；完整逐 claim 审查待办 |
| 17 PD：Bootstrap、KV Reservation 与恢复 | 并行与分布式执行 | 5, 6, 16 | 公式/图清单已读；完整逐 claim 审查待办 |
| 18 Mooncake：Transfer、HiCache L3 与 Embeddings | KV Cache 与内存系统 | 13, 17 | 公式/图清单已读；完整逐 claim 审查待办 |
| 19 DeepEP V2：ElasticBuffer、Overlap 与 EPLB | 并行与分布式执行 | 11, 12 | 公式/图清单已读；完整逐 claim 审查待办 |
| 20 Multi-LoRA：GPU Slots、Pinning 与 Dynamic Load | 模型与扩展运行时 | 14, 3 | 公式/图清单已读；完整逐 claim 审查待办 |
| 21 Weight Quantization：从 checkpoint 到 GEMM Kernel | 模型与扩展运行时 | 9, 11 | 公式/图清单已读；完整逐 claim 审查待办 |
| 22 Quantized KV Cache：FP8、FP4 与 Attention Backend | KV Cache 与内存系统 | 6, 9, 21 | 公式/图清单已读；完整逐 claim 审查待办 |
| 23 CUDA Graph：Capture、Replay 与 Breakable Execution | 算子与 GPU 执行 | 3, 9 | 公式/图清单已读；完整逐 claim 审查待办 |
| 24 Multimodal Processing：Media I/O 与 Token Expansion | 模型与扩展运行时 | 15, 1 | 公式/图清单已读；完整逐 claim 审查待办 |
| 25 Observability & Benchmarking：TTFT、TPOT 与性能剖析 | 推理基础与服务生命周期 | 1, 3 | 公式/图清单已读；完整逐 claim 审查待办 |
| 26 XGrammar：Structured Generation 的编译与 Token Mask | 生成、采样与约束 | 4, 9 | 公式/图清单已读；完整逐 claim 审查待办 |
| 27 Function Calling：Tool Parser 与 Streaming Protocol | 生成、采样与约束 | 4, 26 | 公式/图清单已读；完整逐 claim 审查待办 |
| 28 Deterministic Inference：Batch-Invariant Kernels | 算子与 GPU 执行 | 9, 23 | 公式/图清单已读；完整逐 claim 审查待办 |
| 29 Adaptive Speculative Decoding：EMA 与 Runtime State | 生成、采样与约束 | 10, 23, 28 | 公式/图清单已读；完整逐 claim 审查待办 |
| 30 Distributed Checkpoint Engine：并行加载与 IPC | 权重加载与存储 I/O | 11, 21 | 公式/图清单已读；完整逐 claim 审查待办 |
| 31 HiSparse：从完整 Host KV 到 GPU Working Set | KV Cache 与内存系统 | 6, 9, 13 | 公式/图清单已读；完整逐 claim 审查待办 |
| 32 Hybrid KV Cache：全局 KV、滑窗与循环状态 | KV Cache 与内存系统 | 6, 7, 13 | 公式/图清单已读；完整逐 claim 审查待办 |
| 33 KV Cache Sharing：前缀复用与 CP 页分片 | KV Cache 与内存系统 | 7, 11, 13 | 公式/图清单已读；完整逐 claim 审查待办 |
| 34 Paged Experts：逻辑专家、物理槽与 Waves | 模型与扩展运行时 | 12, 21 | 公式/图清单已读；完整逐 claim 审查待办 |
| 35 SGLang for RL：Rollout 数据、权重版本与引擎协同 | 路由、RL 与集群运维 | 10, 25, 28, 30 | 公式/图清单已读；完整逐 claim 审查待办 |
| 36 Prefill Context Parallelism：Zigzag 与 Interleave | 并行与分布式执行 | 8, 11, 33 | 公式/图清单已读；完整逐 claim 审查待办 |
| 37 Decode Context Parallelism：MLA 分片与精确 LSE 合并 | 并行与分布式执行 | 9, 11, 36 | 公式/图清单已读；完整逐 claim 审查待办 |
| 38 Session-Aware Radix Cache：多轮会话的软保护与淘汰 | KV Cache 与内存系统 | 7, 13, 16 | 公式/图清单已读；完整逐 claim 审查待办 |
| 39 Unified Memory：共享 Arena、动态 Bands 与安全 Compaction | KV Cache 与内存系统 | 32, 33, 38 | 公式/图清单已读；完整逐 claim 审查待办 |
| 40 Reasoning-Aware Compression：On-policy CoT 校准与模型剪枝 | 模型与扩展运行时 | 21, 25, 35 | 公式/图清单已读；完整逐 claim 审查待办 |
| 41 Pipeline Parallelism：Microbatch、动态 Chunking 与异步 P2P | 并行与分布式执行 | 3, 8, 11, 17, 36 | 公式/图清单已读；完整逐 claim 审查待办 |
| 42 Engram：条件记忆表、N-gram Hash 与分布式 Lookup | 模型与扩展运行时 | 11, 21, 32, 34 | 公式/图清单已读；完整逐 claim 审查待办 |
| 43 Radix Eviction Policies：从 LRU 到 Tail Latency 与 State Cost | KV Cache 与内存系统 | 7, 13, 25, 32, 38, 39 | 公式/图清单已读；完整逐 claim 审查待办 |
| 44 R-Fork：Seed 实例、GPU-to-GPU 权重传输与 Ready Gate | 权重加载与存储 I/O | 17, 18, 30, 35 | 公式/图清单已读；完整逐 claim 审查待办 |
| 45 SGLang Simulator：真实调度逻辑、预测时钟与容量规划 | 路由、RL 与集群运维 | 3, 8, 13, 25, 39, 41 | 公式/图清单已读；完整逐 claim 审查待办 |
| 46 Logits Processor & Sampler：概率分布、截断与 Logprob 语义 | 生成、采样与约束 | 9, 10, 26, 28, 35 | 公式/图清单已读；完整逐 claim 审查待办 |
| 47 Model Loading：Checkpoint、Safetensors、Weight Loader 与分片 | 权重加载与存储 I/O | 11, 21, 30, 44 | 公式/图清单已读；完整逐 claim 审查待办 |
| 48 Tokenizer Manager & Detokenizer：异步分词、增量解码与流式响应 | 推理基础与服务生命周期 | 1, 3, 24, 27 | 公式/图清单已读；完整逐 claim 审查待办 |
| 49 Reasoning Control：Thinking Budget、Reasoning Parser 与 Grammar 状态 | 生成、采样与约束 | 26, 27, 28, 40, 46, 48 | 公式/图清单已读；完整逐 claim 审查待办 |
| 50 SGLang × Kubernetes / llm-d：集群级推理调度与弹性部署 | 路由、RL 与集群运维 | 5, 13, 16, 17, 25, 38, 44, 45 | 公式/图清单已读；完整逐 claim 审查待办 |
| 51 Unified Kernel Architecture：AOT、JIT、Registry 与 Backend Selection | 算子与 GPU 执行 | 9, 21, 23, 28, 46 | 公式/图清单已读；完整逐 claim 审查待办 |
| 52 Layer Boundary：布局契约、延迟归约与 Residual Stream | 并行与分布式执行 | 11, 12, 19, 36, 37, 41 | 公式/图清单已读；完整逐 claim 审查待办 |
| 53 Hybrid Cache Checkpoints：Level 2/3、分叉与 Recurrent State 恢复 | KV Cache 与内存系统 | 7, 13, 32, 38, 39, 43 | 公式/图清单已读；完整逐 claim 审查待办 |
| 54 HiCache Dynamic Storage：运行时 Attach/Detach 与一致性门禁 | KV Cache 与内存系统 | 13, 18, 38, 39, 50 | 公式/图清单已读；完整逐 claim 审查待办 |
| 55 Model Object Storage：Metadata、Safetensors Streaming 与分布式 I/O | 权重加载与存储 I/O | 30, 44, 47, 50 | 公式/图清单已读；完整逐 claim 审查待办 |

## 验收
公式定义域、维度、退化边界、例子和语义；SVG 图文一致性、标签碰撞、箭头与移动端；源码 SHA/path/symbol/line range；前置知识关系。CI artifact: reports/quality-audit.json、browser-qa/svg-geometry.json。未逐个联网核验所有历史证据 URL，未逐像素人工核看110张图，未执行真实 GPU、集群或存储实验。


## P0 deep review (01–10)

- All ten article sections and 20 SVG source labels/arrows read. Historical code files and symbol locations reviewed; paper abstracts cross-checked, without claiming full PDF reread. Six SVGs adjusted, including first-token lifecycle and PD control/data separation. Actual screenshots and full QA pending CI.
- Core mathematical boundaries: cached payload, lookup cap vs forward reuse, grammar support mass, non-overlap PD accounting, page indices, split vs insert, root sentinel, alignment and zero progress, masked online softmax, speculative conservation. Added five regression tests and strengthened existing CPU online-softmax demo.
- Read evidence records and pinned source paths. Exact per-claim recheck status tracked in P0 review record; no GPU benchmark claimed.
