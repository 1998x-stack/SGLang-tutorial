# Repository instructions · SGLang visual course

本项目是**证据驱动的研究生级 SGLang 源码课程**。使用 [Olah-style-paper-posts](https://github.com/1998x-stack/Olah-style-posts/blob/main/skills/olah-style-paper-posts/SKILL.md) 的顺序工作流，参考 [vllm-tutorial](https://github.com/1998x-stack/vllm-tutorial) 的工程检查，但不复制其页面文案与图形。

## 编辑前
1. 从 README 课程地图和 evidence/* 检查知识重复，审阅相邻篇和本站 \`skills/sglang-visual-tutorial/SKILL.md\`。
2. 到 [官方仓库](https://github.com/sgl-project/sglang) 与 [官方文档](https://docs.sglang.io/) 验证，固定要追踪的上游 commit，而不是只记 main。
3. 将每个可核验的断言登记为 evidence ledger：claim、source_url、locator、kind、status、notes。论文历史贡献、当代实现、教学推导和性能评测必须各自标明。
4. 按“困惑 → 反例 → 机制图 → 方程与张量 → 源码路径 → 实验 → 限制”创建文章。

## 永久约束
- RadixAttention 是前缀 KV **缓存/调度系统策略**，不是名为 RadixAttention 的单独 GPU kernel。区分 radix tree、物理 KV 池、attention backend 和 page 映射。
- 分清 **token 字符串/ID 前缀命中** 与 KV 复用条件。示意图中的共享命中仅在相容模型、tokenizer、适配器/缓存命名空间等前提下成立；不要把纯文本相同视为自动安全共享。
- Prefill/Decode 是一次请求中的逻辑阶段；P/D disaggregation 是跨独立 worker 的部署拓扑。KV 传输和同步有实际成本，绝不保证更快。
- overlap scheduling 表示迭代间 CPU 控制工作与设备执行可能并行，不等于模型计算无依赖或所有版本默认开启。
- 约束解码限制合法下一 token 集合，不保证语义真实、JSON 事实正确、安全授权或完全无成本。
- 源码路径与 API 随版本演进；每个技术断言须附来源，实验需注明硬件、模型、batch、输入/输出长度及冷/热缓存状态。
- 独立 SVG 必须有 xmlns/viewBox/title/desc，最小文字建议 ≥14px，箭头方向与数据关系一致；一图一问，关键图应可在不开 JS 时读懂。
- 公式采用 native MathML (mfrac, msub, msup 等)，不得以伪 Unicode 串替代数学结构。SVG 内禁止 MathML。
- 不依赖 CDN、外部字体、分析脚本；维护键盘可用性、可读 focus、reduced motion、离线和打印版。
- 把构建状态放在 README/PR，不把“由 AI 生成”“SVG 原创”“工程过程”之类元叙事放在文章正文。
- **不得声称没有执行过的 QA 通过。** 静态、浏览器、GPU、Pages 是四种不同验收。

## 视觉语义
蓝绿代表请求进展/有效路径；紫色代表共享 KV / 缓存；橙色代表计算/新分配；灰色代表等待/失效。颜色必须搭配明文与线型。所有机制 SVG 用图注说明输入、执行方向和限制；不要把 SIMD、批处理、CPU/GPU 并行画成错误的依赖图。

## 每篇最低质量
1 h1；6+ 顺序性 h2；至少 2 个不同问题的独立 SVG；公式与来源定位；明确的教学反例；一个有意义的可交互状态或可执行最小例子；浏览器无水平溢出；本地链接可解析；证据 JSON 不为空。

## 运行检查
\`\`\`bash
python3 -m unittest discover -s tests -v
python3 examples/prefix_match.py
# 浏览器 QA：1440、768、390、360；验证图示、按键、MathML、控制台、溢出
\`\`\`
将快照与测试报告放 PR，不要事后编造。先 CI 后合并；主分支 Pages 需要独立核验。


## 第二批专属工程约束（06–10）
- **KV Pool：**严格区分 request row、logical token position、physical slot ID、page ID/offset 和每层 K/V tensor。页号与 token ID 不是同一概念；公式 `slot = page_id * page_size + offset` 仅描述页内展开。
- **Allocator：**分别说明 alloc / extend / decode / free；涉及 staged pages、reuse guard 时必须讲清异步写入与过早复用的风险。教学 allocator 必须标为 toy，不能称为 SGLang 实现。
- **RadixCache：**节点 value 是 KV slot 索引；_split_node 拆的是压缩边与索引片段，不是复制 K/V tensor。lock_ref 影响 protected/evictable，eviction strategy 不可一概写成 LRU。
- **Chunked Prefill：**chunk 由剩余输入、token budget、KV memory budget、策略上限及页/状态对齐共同决定。mixed batch 只在启用且满足约束时出现；不能将可变 chunk 画成固定等长切片。
- **Attention Backend：**拆分 resolver、metadata planner 和 kernel。Prefill/Decode 可用不同 backend；兼容矩阵随模型、设备、page size、quantization 和 speculative mode 变化。Online softmax 教学算法不代表每个 backend 的实现完全相同。
- **Speculative Decoding：**区分 EAGLE draft topology、target verification、accepted-prefix commit、token-wise rejection sampling 与 block verification。当前官方文档所述 stochastic EAGLE/EAGLE3/NEXTN block-verification 路径要求 `--speculative-eagle-topk 1`；不能将 block verification 误画成 `topk>1` 树。链式 acceptance/residual 公式不能冒充 EAGLE tree 的完整算法；所有 speedup 必须有受控 benchmark。
- 新增 CPU-only 示例：`page_allocator.py`、`chunk_budget.py`、`online_softmax.py`、`speculative_sampling.py`；验证正常、空预算、页耗尽、对齐、概率归一化等边界。
- 第二批实现快照：`sgl-project/sglang@6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd`。第一批 evidence 仍保留其原始快照，不要无证据地批量重写历史 claim。

## 第三批专属约束（11–15）
- Parallelism：TP 切层内权重，PP 切层深度，DP 切请求副本，EP 切 MoE experts，CP/DCP 切上下文。`initialize_model_parallel()` 的主 world-size 为 TP×PP；Attention-DP/CP 与 MoE-EP/DP 是子组，禁止无条件相乘。
- MoE EP：区分 Router Top-K IDs / gate weights、dispatch、expert GEMM 与 combine。通信 backend 和 runner backend 是不同维度；TBO/SBO 不会消除依赖。
- HiCache：L1/L2 为单 instance 私有；L3 是否共享取决于 backend/namespace。区分 metadata lookup、prefetch、write-back 与 Host layout。
- Multi-LoRA：`max_loras_per_batch` 包含 base-model `None` ID；CPU loaded adapters 与 GPU slots 不同。区分 pinning、eviction、dynamic / overlap loading。
- EPD：Encoder embeddings、Prefill 生成的 language KV 是不同数据；global multimodal embedding cache 不等于 HiCache。EPD 会增加 transfer/router/故障边界，不能声称默认更快。
- 第三批实现快照：`sgl-project/sglang@ac0380c7faf9cece3ca32ffe75815af7c585acea`；旧章节 evidence 保留各自历史 commit。

## 第四批专属工程约束（16–20）
- Router：区分 bucket fit、worker admission、policy pick 与 dispatch retry。Admission 基于 load reports，不是原子 KV reservation；cache-aware 只适用于 plain/prefill 组，session bindings 是 router-local state，副本扩展需要 sticky routing。Power-of-Two 比较阶段相关 load vector，不能把不同量纲随意相加。
- PD lifecycle：区分 bootstrap control plane 与 KV data plane。early 在 Prefill 完成前 reserve Decode slots；prefill_complete 延后 per-request reservation，不改变 global pool 容量，且不是普遍吞吐优化。按 pinned docs 核对支持条件、timeout 起点、abort ACK、retraction、同版本/同策略要求。
- Mooncake：严格分开 PD request KV transfer、HiCache L3 reusable KV pages、EPD global multimodal embeddings。Master/Metadata 管理控制面；Store 与 SGLang client 直接传 payload。HiCache page 只有所有所需 TP ranks 完成后才可提交 radix tree。
- DeepEP：A2A dispatcher 与 MoE runner 是两条配置轴；Normal / Low-Latency 是不同 batch regime。区分 SGLang DeepEP V2 的 ElasticBuffer package 与独立 V2.5 EPBuffer family。SGLang snapshot 的 pyproject 固定 sgl-deep-ep，release workflow 从 sgl-deepep 分支构建；sgl-deep-ep-v25 使用另一条 branch / wheel，不能假设可直接替换。TBO/SBO 只隐藏可重叠通信，不消除 collective ordering 和 buffer lifetime。EPLB 的 token-count balance 不等于端到端 latency balance。
- Multi-LoRA：max_loras_per_batch 是 distinct adapter IDs，包含 base None；CPU max_loaded_loras 与 GPU slots 分开。Pinned adapters 上限 max_loras_per_batch - 1；pinned 不参与常规 eviction/drain。动态 load 是 admin API，需要 API key，rank/target-module shape contract 固定。Prefix KV reuse 必须尊重 adapter namespace。
- 第四批固定源码：SGLang ac0243723b0dcba177f8792979daaa2f8d6ee6ea；SGLang DeepEP V2 branch ece7e40b2f4e53a2910a442d2bbc33fe4130c3e0；DeepEP V2.5 branch d30642958b04c4b1bd7b607ba7c551ea39e9140a；Mooncake 8cd2bd5b45c36dc7520cc28267018aa8ae0645c1。旧章节 evidence 保留历史 commit，不得批量改写。
- 第四批 CPU labs：router_policy.py、pd_lifecycle.py、mooncake_planes.py、deepep_overlap.py、lora_scheduler.py。回归必须覆盖全部 20 篇文章与 evidence。


## 第五批专属工程约束（21–25）

- Weight quantization：严格区分 checkpoint 已量化（offline）与 runtime online quantization；区分 weight / activation / KV 三种对象。量化注册表按平台变化，不能把 CUDA 的方法名/兼容矩阵推广到 CPU、ROCm、XPU 或 NPU。说明 create_weights、checkpoint load、process_weights_after_loading、apply 与 GEMM backend 的阶段边界。AWQ/GPTQ 论文算法不能等同于 SGLang 的每一个 runtime kernel。
- Quantized KV：FP8 scale 与 FP4 recipe 分开；必须把 scale metadata、packed layout、page size、head dimension、prefill/decode attention backend 和临时 workspace 一起核对。NVFP4 的 3.56× 是含 block-scale 的理论 scalar capacity ratio，不是实测 throughput。native NVFP4 的 SM100/shape/algorithm 限制不能泛化到所有 KV formats。
- CUDA Graph：Capture/Replay 依赖固定 operation sequence、shape、control flow 与 memory addresses；新输入应 copy 到 static buffers。Full graph 与 breakable graph 分开，eager break 仍每次 replay 执行。ViT graph key 除 sequence length 还依赖 segmentation pattern。Foundry persistence 是可选且有 model/flags/parallel/GPU identity 与 backend 限制的功能。
- Multimodal request path：区分 API content parts、raw media I/O、processor worker clone、per-item preprocessing、placeholder token expansion、M-RoPE/span metadata 与 CPU/CUDA IPC/VMM feature transport。token-space expansion 只替换 caller placeholders，已展开 prefix 保留，插入 tokens 不重复扫描。与第 15 章 EPD worker topology、第 18 章 Mooncake payload plane 严格区分。
- Observability：Counter/Gauge/Histogram 不能混用。TTFT、request-level TPOT、token-weighted ITL 与 E2E latency 定义不同；TPOT 排除 first token 且有请求纳入条件。metrics labels 要关注 cardinality。默认不记录 request contents；request dump/crash dump 需说明敏感数据和访问控制。
- Benchmark：bench_serving、bench_one_batch_server、bench_offline_throughput、bench_one_batch 覆盖不同 stack layers，不能互相替代。single-batch 结果不等于 steady state；固定 model/hardware/prompt-output distribution/concurrency/arrival rate/cache state/warmup/measurement window。Profiling admin endpoints 必须使用 admin key；PD prefill/decode profiler 分开。
- 第五批源码基线：`sgl-project/sglang@563813c0d60282fa848382fec600b40d28ddec98`。evidence/21.json–evidence/25.json 中每条 claim 要有可定位 URL 与 locator；历史章节保留各自 commit。
- 第五批 CPU labs：weight_quantization.py、kv_cache_quantization.py、cuda_graph_capture.py、multimodal_pipeline.py、serving_metrics.py。只验证通用教学模型和边界，不模拟真实 SGLang kernel、CUDA capture、媒体解码、CUDA IPC 或 server throughput。
- 第五批回归：25 篇文章、50 张独立 SVG、25 份 evidence、26 个首页 filter buttons；Chromium 1440/768/390/360 × 26 pages（首页 + 25 章）。Pages staging 与 public smoke check 必须覆盖 25 HTML、50 SVG、全部 examples 与 25 evidence。

## 第六批专属工程约束（26–30）

- Structured generation：严格区分 request constraint、grammar compiler、matcher state、vocabulary bitmask、logits masking 与 sampler。一个请求只选择 JSON Schema / regex / EBNF / structural tag 中适用的约束参数；backend 能力与 tokenizer 支持要逐项核对。语法合法不等于语义真实或业务授权。GrammarManager 异步编译、缓存与 DP/TP/PP rank sync 必须分层解释。第 04 章是 frontend/DSL；第 26 章是 runtime backend。
- Tool parser：parser registry 选择 model-format detector；non-stream 与 streaming 路径分开。必须覆盖 partial marker、chunk boundary、incomplete JSON、multiple calls、final flush 和普通文本保留。ToolCallItem 是 serving protocol normalization，不是工具执行。工具执行、授权、参数验证、sandbox、审计与 prompt-injection 防护属于应用层。
- Deterministic inference：说明浮点非结合性、batch-shape-dependent reduction order、batch-invariant operators 与 per-request RNG state。后端兼容矩阵、Radix Cache / chunked prefill / CUDA Graph / non-greedy sampling 约束必须基于固定源码文档。temperature=0 不等于完整确定性；确定性也不自动证明 training-inference parity。
- Adaptive speculative decoding：第 10 章讲 EAGLE 算法；第 29 章讲自适应运行时策略。当前文档限制为 EAGLE/EAGLE3 且 speculative_eagle_topk=1；不得泛化到任意 speculative algorithm/top-k tree。区分 per-BS EMA slot、warmup、update interval、hysteresis、candidate steps 与 prebuilt SpecRuntimeState。切换是完整 state reference swap，不是在线 graph recapture。候选状态数增加会影响启动时间与显存。
- Checkpoint Engine：区分 model loader、Checkpoint Engine worker、Parameter Server、Broadcast/P2P/All、IPC endpoint 与 post-load hooks。dummy load + wait-weights-before-ready 的 Server 在真实权重到齐和 post-load 完成前不得标记 Ready。`/update_weights_from_ipc` 是 admin control endpoint，server/update worker 使用相同 SGLANG_ADMIN_API_KEY；不能写入日志或公开。Rank mapping、TP size、device UUID、quantization post hook、barrier 与 weight consistency 都是关键正确性条件。
- 第六批源码基线：`sgl-project/sglang@563813c0d60282fa848382fec600b40d28ddec98`。evidence/26.json–evidence/30.json 的每条 claim 必须含可定位 URL、locator、kind 和 status；研究论文贡献与当前 SGLang 实现分开写。
- 第六批 CPU labs：grammar_mask.py、tool_parser.py、deterministic_inference.py、adaptive_spec.py、checkpoint_engine.py。只验证有限状态 grammar、增量 parser、float32 reduction order、EMA policy 和 rank readiness；不得声称实现真实 GPU bitmask、真实模型 tool parser、CUDA batch-invariant kernel、EAGLE worker 或 IPC transport。
- 第六批回归：30 篇文章、60 张独立 SVG、30 份 evidence、31 个首页 filter buttons；Chromium 1440/768/390/360 × 31 pages（首页 + 30 章），共 124 views。Pages staging 与 public smoke check 覆盖 30 HTML、60 SVG、全部 examples 与 30 evidence。

## 第七批专属工程约束（31–35）
- HiSparse：模型原生 sparse top-k 与设备驻留分开。PD decode-only；DeepSeek V4 C4 direct-to-host 不等同于所有 KV/索引；共享 index 预取 eligibility 单独核对。任何 miss plan 必须在消费前完成。
- Hybrid：SWA full-to-window 映射与 Mamba/GDN/KDA recurrent state 是不同对象。state:KV ratio 是字节比；复用需同一边界的 KV 和 checkpoint，计入 idle session 和 draft slots。
- KV Sharing：明确 request prefix reuse、L3 cross-instance sharing、prefill CP page sharding 三种范围。owner/local row 公式必须保留 page_size；gather scratch 与通信成本计入，PD 与 HiCache ownership 遵守 pinned docs。
- Paged Experts：逻辑 expert ID 与物理 slot 分开，wave 不可重复或漏算 routed entry；原始 gate weights 不在 wave 内重新归一化。K 与 KV 预算竞争；tokens×top_k≤K 是有界 graph 路径的充分条件。
- RL：行为 logprob 与采样语义及 token mask 对齐；weight_versions 为 output 半开区间，prefill attribution 独立。pause/refit/resume 区分 abort/retract/in_place；更新返回失败不保证权重回滚。Admin endpoints 不暴露凭证。确定性不等于训练推理一致。
- 固定上游 75ed36d976113c693f8e6f64925e119573a93fda；新增 5 个 CPU toy labs、10 SVG、5 evidence。回归 35 章、70 SVG、35 evidence、36 filters、144 browser views、30 labs。CI 通过后方合并；独立验证 Pages；未执行的检查不得声称通过。


## 第八批专属工程约束（36–40）
- PCP：zigzag 每请求重启 2C blocks；interleave 按 flattened new-token index，保留请求内 position 和 causal extent。区分 query layout、KV page ownership、DSA LayerSplit、DCP 与 TP/DP/EP。
- DCP：virtual v → owner=v mod c、physical=floor(v/c)；LSE merge 使用 backend 对应指数底，空 shard 权重 0，不能平均局部 softmax。DCP 必须嵌套 attention TP replica，KDA state 不因此扩容。
- Session cache：session_id 不重建 prompt；引用为软保护非 pin，close 取消引用不删除 KV；Full 与 SWA/Mamba 淘汰规则及 cascade 分开，拒绝旧 generation 的迟到注册。
- Unified Memory：共享 arena 不是 CUDA UVM；区分 virtual/physical ID、vector demand、capacity estimate 和可达布局。Pending、network、fixed page 的 hazard clearance 必须先满足；PP 各 stage 共同可行。具体支持组合见 pinned docs。
- RAC：prompt + dense on-policy decode calibration 与 pruning solver 分离；不能把权重剪枝等同于生成 token 剪枝或 RL。比较 accuracy、completion length、wall clock；toy 不是 SparseGPT benchmark。
- 固定上游 b1ef3a60797aa6c29bcfa4cc3bc135dd2f19ec22，历史 claims 不重写。40 章、80 SVG、40 evidence、41 filters、35 CPU labs、164 browser views。所有 QA 未执行时标 not verified；CI 通过后合并并检查 Pages。


## 第九批专属工程约束（41–45）
- PP：stage 与 chunk 依赖均保留；理想 bubble 公式不含通信；动态 chunk 的二次方程、平滑、对齐、容量门分开。P2PWork 需在消费/缓冲复用前等待。
- Engram：静态 table rows 与请求 KV、MoE 权重分开。token 窗口、boundary pad、hash IDs、placement、replication、lookup 独立解释；不得承诺 collision-free 或任意 topology。
- Eviction：先筛 eligible leaves，再排序；parent 重新入候选。T-LRU high-water history 不随 tail trim 下降；state_cost 仅 recurrent checkpoints，joint_lru 和 session 兼容性按固定文档。
- R-Fork：remote_instance loading 不复制进程/KV；NCCL 对应 ranks、TransferEngine registration、ModelExpress discovery 区分。ready gate 示例是工程不变量，不声称完整事务；admin endpoints 不暴露凭证。
- Simulator：真实 scheduler/cache 与 mock forward/sample 区分；logical time、wall clock、预测误差独立。当前 single-worker runtime 不等于 predictor 目标拓扑；不得把模拟数当真实 GPU 性能。
- 固定上游 b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7；45 章、90 SVG、45 evidence、46 filters、40 CPU labs、184 browser views。CI 通过后合并并检查 Pages；未执行检查如实标记。


## 第十批专属工程约束（46–50）
- Sampler：固定实现的 penalty / grammar / bias / watermark / custom processor / sanitize 顺序和旁路分别标注。temperature=0 用 greedy；严格说明 top-p 边界和后端差异；API logprob 不假定等于截断后 behavior distribution。
- Loader：checkpoint file shard 与 TP/PP 参数布局分开；default iterator 与 sharded_state load 顺序不同；memory peak 分 host/device；校验 revision、dtype、shape、missing keys。可信 checkpoint 与代码边界不可省略。
- Tokenizer：单字符串动态批处理空队列立即处理；kwargs 不一致逐条执行。增量解码区分 token offsets、sent offsets、UTF-8 holdback、stop trim 与取消。非 beam continuity 生产 warning 与 CI exception 不混淆。
- Reasoning：模板 hint、grammar hard budget、custom processor、parser 分层。多 token terminator、partial match、reasoning_in_grammar 交集、total output limit 分开；native require_reasoning 与 chat 参数映射须依据固定文档。
- llm-d：SGLang scheduler 与 EPP 集群层分开；precise index 不是 KV tensor；SGLang reference dedicated render pool；区分 community 与 validated，不能移用 vLLM 实测。固定 llm-d 9059ad4050a3e1713d14b954685dac87cfe932cc 推荐 KEDA+EPP，WVA deprecated；不声称真实集群已部署。
- 固定 SGLang b323e8e6e3aacfe606b11a2ef51ae73bc8b283a7；50 章、100 SVG、50 evidence、51 filters、45 CPU labs、204 browser views。CI 通过方合并，独立验证 Pages；未执行检查如实标记。


## 第十一批专属工程约束（51–55）
- Kernel：固定快照统一 sglang.kernels，历史 jit_kernel 不重写旧证据；AOT/JIT、registry/selector、FFI、stream、dtype/stride、尾部 guard 分开。
- Boundary：token rows 与 sum group 分开，exactly-once completion 与 residual update，EP A2A 属 compute。opaque handle 不作为普通 tensor。
- Checkpoints：共同可用 state/KV boundary；level2/3 与 SWA 例外；软 cap 不等于 slot 上限，CoW 与 host-backup 单独核对。
- HiCache attach/detach：strict idle + admin；DP partial success 无自动 rollback；drain→stop→release→drain；失败保留 bookkeeping；detach 不删远端数据。
- Object storage：metadata 与 weight URI 分离；streamer buffer 非全局峰值，distributed world 与 stage-local fallback 分开；不泄露凭证。
- 固定上游 0fde0356cc44bca9a7727a3a2e2c2d10cd5bd30f；55 章、110 SVG、55 evidence、56 filters、50 CPU labs、224 browser views；通过 CI 后再合并与验证 Pages。


## 55 章质量治理
- assets/course-graph.json 是结构化 DAG；site.js 内嵌镜像由 audit 检查。8 一级领域、55 细分专题、全部，共64 filters。
- 公式需明确量纲、定义域、假设、边界和数值；SVG 箭头实际长度、文字边界、图文一致性分别检查；流程 MathML 不冒充推导。
- 保留历史 SHA，迁移单独补充。每个 claim 的 path/symbol/line 必须复核，自动清单不代表语义通过。
- CI 输出 reports/quality-audit.json 与 browser-qa/svg-geometry.json；结构与零长度箭头阻断，bbox 越界和移动来源进入审阅队列。详见 docs/quality-audit.md。


## P0 基础章审查规范
- 普通生成按 Prefill 产首 token、Decode 消费前一 token 绘图；逻辑 LCP 与 forward reuse 上限分开；根哨兵和普通引用计数、match split 与后续 insert 分开。
- 每个 CPU 教学示例与图、页号、输出断言一致；公式必须覆盖退化边界：全 mask、空支持、对齐零推进、p=q 无 correction；不要把 block verification 当 tree topology。
- Evidence 增加逐 claim `p0_review` 与具体历史行号；论文摘要核对不能宣称全文复核，源码文件/定位核查不能宣称 GPU 复现。详见 docs/p0-core-review.md。
