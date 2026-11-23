---
name: sglang-visual-tutorial
description: Deep source-grounded Chinese visual explainers on SGLang inference, KV caches, scheduling and deployment. Compose original SVGs, equations and deterministic interactive examples with Olah-style narrative and explicit evidence.
---

# SGLang Visual Tutorial Skill

以 [Olah-style-paper-posts](https://github.com/1998x-stack/Olah-style-posts/tree/main/skills/olah-style-paper-posts) 七道门为上层规范。本地 Skill 仅补充 **SGLang 专属语义、知识拓扑与可视化测试**。

## Gate 0 选题去重与上游锚点
- 先阅读 README/课程首页/最近变动，查找同一概念是否已建章。选题结果须能回答一个明确的学习问题。
- 搜索原论文 + 官方文档 + 运行时源码；将论文年代的机制和当前实现独立对照。
- 上游源码参照：\`sgl-project/sglang@0a8693fbe3a2e46ccc06470a461a2b85865672fd\`（2026-11-23 观察值）。任何后续修改时都应重新固定并审视变更。
- 建立 claims：只把查到的工程位置写成事实；实验性能不能用想象值。

## Gate 1 画故事板
每节先写：**具体困惑 / 输入 / 输出 / 不变量 / 版本适用性 / 反例**。
首批必备 10 张不同的机制图（见 \`docs/visual-storyboard.md\`）；一个章节至少两种拓扑，不允许两张换色卡片。

## Gate 2 技术推导
- Prefix：令 a,b 为 token ID 序列，LCP(a,b) = 首个不等位置前的长度；page 对齐时，缓存可命中长度一般至多为 \`floor(LCP / page_size) * page_size\`，真实策略视上游实现。
- KV：按标准 MHA/GQA 缓存估算 \`2 * layers * kv_heads * head_dim * tokens * bytes_per_element\`，排除权重/碎片/运行时开销。
- Batch：令 \`B_t\` 为时刻 t 参与 forward 的请求集合；有动态加入/退出，调度是否接纳受 token budget、KV 和策略限制。
- P/D：关键路径应分别展示 prefill、transfer、decode；低延迟与吞吐不能从一张拓扑图直接证明。
- Grammar：候选 token 集合与状态转换分开描述；不要声称 FSM 可以验证任意语义。

## Gate 3 页面与文字
复用 \`assets/css/site.css\` 的讲义风排版。正文提供 6+ 有层次的 h2，至少一处真实 native MathML；不写默认值等易变参数为永久事实。长代码/公式局部滚动，网页整体不得横向溢出。

## Gate 4 SVG
每图有 title、desc、viewBox；使用 \`assets/diagrams/NN-*.svg\`。过程：边列表 → 坐标标注 → 检查逻辑 → SVG XML 解析 → 1440/390/360 浏览器实图 → 独立文件截图 → compare。
所有 SVG 默认状态可解释；交互用页面上的原生按钮改变**真实教学结论**，不仅换背景色。图例、标签和 figcaption 共同承担语义。

## Gate 5 可执行最小实验
优先 Python 标准库/纯 JS 的确定性 toy 示例。包括正常、空前缀、部分匹配、不相容命名空间等边界测试。明确标注 toy 不等于生产内核。

## Gate 6 QA
\`python3 -m unittest discover -s tests -v\`；浏览器检查桌面/平板/手机的公式排版、SVG 标签、断链、CSS/JS 404、按钮实际引起状态变化、ARIA 状态、键盘 focus、JavaScript error、水平溢出。没有跑浏览器时记录 not verified。

## Gate 7 发布
PR 比较文件并检查 CI；仅在检查通过后合并。用实际 GitHub Pages URL + 资源 URL 验证部署，不允许因 push 成功声称线上已发布。


## Gate 0.5 · 第二批源码基线
- Chapters 01–05 保留原 evidence 快照 `0a8693fbe3a2e46ccc06470a461a2b85865672fd`。
- Chapters 06–10 以 `sgl-project/sglang@6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd` 为固定快照。关键路径：`memory_pool.py`、`allocator/paged.py`、`radix_cache.py`、`schedule_policy.py`、`scheduler.py`、`schedule_batch.py`、`attention_backend_setup.py`、`flashinfer_backend.py`、`speculative/eagle_worker_v2.py`。

## Gate 2.5 · 五个高级机制的正确性
- **KV address:** request row 是逻辑映射；physical slot 由 allocator 提供；每层 K/V buffer 再解释 slot。页号与 offset 展开为 `slot = page_id * P + offset`。toy allocator 必须测试 page boundary、free/reuse、double allocation 与 pool exhaustion。
- **Radix internals:** `match_prefix` 可能触发 edge split；split 的 key/value index slices 不等于复制底层 KV tensor。lock_ref 从 0→1 将路径长度从 evictable 移入 protected；eviction 由策略选择可回收 leaf，并通过 allocator 释放。
- **Chunked prefill:** 仅用 `min(remaining, token budget, memory budget, policy cap)` 作为教学上界，不宣称是实现的完整单公式。解释 `available_chunk_tokens`、`fit_chunk`、alignment、continuation/stash 和 mixed forward。注意模型/后端可能禁用 chunking。
- **Attention:** 画出 resolver → prefill/decode backend (or hybrid) → metadata → kernel。把 paged KV index traversal 与 attention math 分开。Online softmax 只作为算法教学，必须注明 backend-specific implementation differences。
- **Speculative:** 标准 chain sampling 的 `alpha=min(1,p/q)` 与 residual distribution 只适用于对应 token-wise chain；EAGLE draft topology、token-wise rejection 与 block verification 是不同维度。当前官方文档所述 stochastic EAGLE/EAGLE3/NEXTN block-verification 路径要求 `--speculative-eagle-topk 1`，不得写成 `topk>1` 的树验证。支持设备、DFlash/DSpark proposal 条件和 correction 规则以固定版本官方文档为准。
- 数值例子全部标为 toy/derived；不把论文 speedup 或某个 GPU 的结果移植为仓库自己的 benchmark。

## Gate 5.5 · 第二批 CPU 实验
运行：
```bash
python3 examples/page_allocator.py
python3 examples/chunk_budget.py
python3 examples/online_softmax.py
python3 examples/speculative_sampling.py
```
每个示例都要有确定输出、可复现断言和负面边界测试；它们验证概念，不声称运行 SGLang GPU serving。

## Gate 6.5 · 扩展课程回归
- 10 篇章节、20 个 SVG、10 个 evidence JSON、首页 10 张卡片与 11 个分类筛选项。
- Chromium 1440/768/390/360 × 11 pages = 44 个 viewport；验证 lazy SVG 实际加载、MathML namespace/geometry、stepper 状态、搜索/筛选、JS error、请求失败与页面横向溢出。
- 所有内部章节链接、examples 链接和图资源均应解析；独立 SVG 通过 XML 解析，标题/描述非空。

## 第三批（11–15）源码与正确性约束
- 固定源码快照 `sgl-project/sglang@ac0380c7faf9cece3ca32ffe75815af7c585acea`；入口为 `parallel_state.py`、`dp_attention.py`、`expert_parallelism.mdx`、`deepep.py`、`moe_runner/runner.py`、`hicache_design.mdx`、`lora_manager.py`、`lora/mem_pool.py`、`epd_disaggregation.mdx` 与 `encoder/server.py`。
- TP×PP 描述主 model-parallel world；Attention-DP/CP 与 MoE-EP/DP 是子组。MoE dispatch / combine 与 compute runner 分层；TBO/SBO 不消除依赖。
- HiCache 的 L2 是 instance-private，L3 backend-dependent；LoRA `max_loras_per_batch` 计入 `None` base ID；EPD embedding cache 与 KV cache 必须分开画。
- CPU labs：`parallel_topology.py`、`moe_dispatch.py`、`hicache_tiers.py`、`lora_pool.py`、`epd_pipeline.py`。覆盖非法拓扑、路由权重、L2/L3 共享边界、pinned slot 与 pipeline bottleneck。
- 回归基线：15 篇、30 SVG、15 evidence、16 个筛选按钮；Chromium 1440/768/390/360 × 16 pages；Pages 必须发布 examples 与 evidence。

## Gate 0.75 · 第四批源码与外部依赖锚点（16–20）
- SGLang ac0243723b0dcba177f8792979daaa2f8d6ee6ea；DeepEP f6aa38fe06cd4a0489e7730365d4e0114970a2f3；Mooncake 8cd2bd5b45c36dc7520cc28267018aa8ae0645c1。各 claim 必须指向该 snapshot 的真实文件与 locator。
- Router docs / Rust reorg policy、PD sender/receiver state machine、Mooncake Store + HiCache + EPD embedding cache、DeepEP V1/V2 adapter、LoRA manager / memory pool 分别核验。
- 区分控制面与数据面，区分不同 Mooncake payload，区分 DeepEP dispatcher 与 runner，区分 LoRA CPU residency / GPU slot / adapter ID。
- prefill_complete 的适用限制与 worker version consistency 必须按官方 docs 原文检查，不得写成通用开关。

## Gate 2.75 · Router / PD / Mooncake / EP / LoRA 机制边界
- Router：bucket fit → admission → policy → dispatch/retry。Power-of-Two 用阶段性 lexicographic load；cache-aware 的 prefix owner 需通过 admission；session affinity 是 router-local state。
- PD：bootstrap metadata 不等于 KV payload transfer。early 与 prefill_complete 改变每请求目的 slot reservation 时机；必须描述 sender/receiver abort、timeouts、retraction 和 unsupported configurations。
- Mooncake：PD request KV handoff、HiCache L3 KV objects、EPD multimodal embeddings 的 key/type/lifecycle 分开。
- DeepEP：normal / low_latency、V1 Buffer / V2 ElasticBuffer、FP8 scales / capacity checks、TBO/SBO、EPLB 的统计窗口和负载目标分开。
- Multi-LoRA：distinct IDs includes None；CPU loaded count 与 GPU slot count 不同；pinned <= max_loras_per_batch - 1；dynamic load shape contract 与 admin authentication。

## Gate 5.75 · 第四批 CPU 实验
- router_policy.py：cache owner、admission、balanced factor+gap、Power-of-Two fallback。
- pd_lifecycle.py：early / prefill_complete、非法状态转移、abort。
- mooncake_planes.py：PD KV / HiCache pages / EPD embeddings 类型隔离。
- deepep_overlap.py：phase mode、overlap bounds、expert skew。
- lora_scheduler.py：base None slot、LRU、pinned limit、slot exhaustion。
- 全部只验证教学抽象，不模拟 Rust router、GPU collectives、RDMA 或真实吞吐。

## Gate 6.75 · 第四批回归
- 20 篇文章、40 张 SVG、20 份 evidence、21 个首页 filter buttons。
- Chromium 1440/768/390/360 × 21 pages（首页 + 20 章）。
- 首页 20 张课程卡片必须全部在 .course-grid 内；搜索、原有类别与新增 Mooncake / Router 等分类均要交互验证。
- Pages staging 和 public smoke check 覆盖全部 20 HTML、全部 SVG、examples 与 evidence。
