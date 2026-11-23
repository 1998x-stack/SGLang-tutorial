# SVG 视觉故事板 · Chapters 01–25

图不是装饰，而是具有输入、输出、状态和因果方向的微型技术解释。

## 第一批：全景机制（01–05）

| 图 | 问题 | 视觉不变量 |
|---|---|---|
| 01 inference path | 请求如何从 Prompt 走到多个 token？ | Prefill 先于 Decode；历史 K/V 被复用 |
| 01 KV memory | 每个 token 需要多少 KV 内存？ | K 与 V 两份，按层和 KV heads 计 |
| 02 radix tree | 公共前缀如何分叉？ | token ID 路径，分叉后后缀独立 |
| 02 cache lifecycle | 命中后怎样锁定和回收？ | lock_ref 保护使用中路径 |
| 03 batch timeline | 为什么批次成员会变化？ | 每次 forward 的集合可以变化 |
| 03 overlap | CPU/GPU 如何重叠？ | 不消除自回归依赖 |
| 04 fork/join | 程序分支如何共享父前缀？ | join 汇总输出，不合并不同后缀 KV |
| 04 grammar FSM | grammar 如何缩小候选？ | 状态限制候选，不保证语义正确 |
| 05 P/D topology | KV 为什么跨 worker 传输？ | Prefill → transfer → Decode |
| 05 latency path | 分离新增什么成本？ | queue/compute/transfer/decode 分开 |

## 第二批：源码内部（06–10）

| 图 | 问题 | 关键对象 / 不变量 | 状态 |
|---|---|---|---|
| 06 memory mapping | request row 如何定位 K/V？ | ReqToTokenPool → physical slot → KVCache | row / slot / buffer |
| 06 page lifecycle | 物理页如何申请与释放？ | page_id × P + offset；页不重复释放 | free / allocate / extend / free |
| 07 radix split | 部分命中时怎样拆压缩边？ | prefix node 与两个后缀；共享段不复制 | before / after split |
| 07 lock & evict | 哪些节点可回收？ | lock_ref 0→1 protected；仅 evictable leaf 可驱逐 | locked / unlocked / evicted |
| 08 chunk flow | 长 Prompt 如何逐段推进？ | min(remaining, token budget, memory budget, policy cap) | chunk 1/2/3 |
| 08 mixed timeline | Prefill 如何与 Decode 混合？ | MIXED forward 中两类行并存 | iteration t0/t1/t2 |
| 09 backend dispatch | 如何选择 Prefill/Decode kernel？ | resolver → hybrid wrapper → backend metadata | same / hybrid |
| 09 paged attention | Kernel 如何读取分散 KV？ | request row → page indices → K/V tile loop | plan / read / accumulate |
| 10 EAGLE draft tree | Draft 如何产生候选树？ | parent topology + candidate tokens + tree mask | draft / verify / commit |
| 10 verification | chain 与 tree verification 有何区别？ | token-wise rejection 与文档所述 topk=1 block verification 的 prefix check / correction 不混淆 | accept / reject / residual |

## 设计约束

- 每张图都先写 caption 和可验证断言，再画图。
- 箭头必须对应数据或控制流；颜色不能成为唯一语义。
- 文字最小建议 14px；移动端允许图内横向滚动，不缩小到不可读。
- 独立 SVG 必须包含 `xmlns`、`viewBox`、`title`、`desc`，并通过 XML 解析。
- 每图必须在不开 JS 时可读；交互状态改变解释结论。
- 不把页号画成 token ID，不把 RadixCache 画成物理 allocator，不把 attention backend 画成 KV pool。
- Chunked Prefill 不画成固定等长切片；Speculative chain rejection 与 block verification 分开画；当前文档所述 stochastic EAGLE/EAGLE3/NEXTN block 路径要求 topk=1，不能标成 topk>1 tree。

## 第三批（11–15）：分布式与多租户 serving
| 章 | 学习问题 | Figure A | Figure B | 交互 / CPU 实验 |
|---|---|---|---|---|
| 11 | TP/PP/DP/EP/CP 分别切分什么？ | 五种并行轴 | TP=2、PP=4 rank 网格 | 并行轴选择器；parallel_topology.py |
| 12 | Top-K token 如何跨 rank 到达专家并返回？ | Router / All-to-All | TBO 双 microbatch 时间线 | Token lifecycle；moe_dispatch.py |
| 13 | 被驱逐的 KV 如何从 Host / Storage 恢复？ | L1/L2/L3 与共享边界 | local match / prefetch / promotion | tier-hit explorer；hicache_tiers.py |
| 14 | 多 adapter 如何共享 base model？ | 异构 batch 与 LoRA branch | Host/GPU adapter pool | pool lifecycle；lora_pool.py |
| 15 | 多模态 Encoder、Prefill、Decode 如何解耦？ | Unified / PD / EPD | embeddings 与 KV 两类数据流 | topology selector；epd_pipeline.py |

并行图须区分逻辑 rank group 与物理 GPU 拓扑；HiCache 图须标注 L2 instance-private 与 L3 backend-dependent；EPD 图须分开画 embedding cache 和 language KV cache。

## 第四批（16–20）：路由、PD 状态机、数据平面与多租户调度
| 章 | 学习问题 | Figure A | Figure B | 交互 / CPU 实验 |
|---|---|---|---|---|
| 16 | Bucket、Admission、Policy 与 Retry 如何串联？ | Router request lifecycle | cache-aware owner vs load fallback | policy selector；router_policy.py |
| 17 | Bootstrap、KV reservation、transfer 与 abort 如何协作？ | Prefill/Decode 双侧状态机 | early vs prefill_complete 时间线 | allocation policy；pd_lifecycle.py |
| 18 | Mooncake 的 PD、HiCache、EPD 三种角色如何区分？ | 三种 payload / consumer 平面 | Master/Metadata/Store 控制与数据面 | role selector；mooncake_planes.py |
| 19 | DeepEP normal/low-latency、V2 buffer、overlap、EPLB 如何配合？ | dispatch mode comparison | TBO microbatch overlap | mode selector；deepep_overlap.py |
| 20 | Multi-LoRA 的 adapter IDs、CPU/GPU residency 与 pinning 如何调度？ | heterogeneous batch rows | GPU slot pool / LRU / pinned | pool selector；lora_scheduler.py |

关键不变量：Router admission 不是 reservation；PD bootstrap 控制消息不等于 KV payload；Mooncake 三类 payload 不共享 key；DeepEP A2A backend 与 MoE runner 分离；SGLang ElasticBuffer wheel 与独立 V2.5 EPBuffer wheel 不能混用；LoRA None 是一个 distinct ID，pinned adapters 不能占满 GPU slots。


## 第五批（21–25）：数值表示、执行图、多模态输入与可观测性
| 章 | 学习问题 | Figure A | Figure B | 交互 / CPU 实验 |
|---|---|---|---|---|
| 21 | checkpoint quant metadata 如何落到 per-layer quant method 与 GEMM backend？ | metadata → registry → layer method → kernel | W/A precision matrix | offline / online / backend selector；weight_quantization.py |
| 22 | 量化 KV 如何写入 pool 并被不同 attention backend 读取？ | quantize → paged pool → native/workspace read | BF16/FP8/NVFP4 scale-aware capacity | native / workspace / unsupported selector；kv_cache_quantization.py |
| 23 | Graph capture/replay 为什么依赖静态 shape 与地址？ | warmup/capture/static buffer/replay | breakable graph/eager segments | full / breakable / eager selector；cuda_graph_capture.py |
| 24 | raw image/video/audio 如何变成 token spans 与 scheduler features？ | media I/O → processor → expansion → transport | placeholder → variable token runs | modality selector；multimodal_pipeline.py |
| 25 | 生产指标与 benchmark 分别观察哪一层？ | TTFT/TPOT/ITL/E2E timeline | four benchmark tools and stack boundaries | serving/engine/kernel selector；serving_metrics.py |

关键不变量：W/A quantization 与 KV quantization 是不同对象；NVFP4 ratio 计入 scale overhead 但不代表吞吐；graph replay 更新 buffer 内容而非替换地址；multimodal placeholder expansion 不重复扫描插入 token；request-level TPOT 与 token-weighted ITL 统计口径不同；single-batch kernel benchmark 不能冒充 steady-state serving。
