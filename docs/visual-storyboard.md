# 10 张机制图的视觉故事板

**目标**：图不是标题装饰，而是可验证、具有输入/输出/因果方向的微型技术解释。图在普通屏宽可直接读懂；移动端提供文本复述与可滚动细节，不依赖颜色传达全部信息。

| 图 | 想回答的问题 | 不变量 / 关键边 | 交互状态 | 源 |
|---|---|---|---|---|
| 01 · inference-path | 一次请求如何从 Prompt 走到多个 token？ | Prefill 先于新 token 的后续 Decode；KV 写/读路径区分 | Prefill/Decode A/B | 官方 runtime |
| 01 · kv-memory | 每个 token 产生多少 KV 内存？ | 2 × layers × KV heads × head dim × bytes | 不同 T 的计算 | 推导 |
| 02 · radix-tree | A/B 分叉如何共享已算 KV？ | 所有边是 token 序列，查找最长公共前缀 | 选择候选序列 | 原论文 §3, radix_cache.py |
| 02 · cache-lifecycle | 命中→锁定→扩展→释放→回收的顺序？ | 使用中节点不可按空闲路径随意逐出；toy 不宣称精确实现 | 命中/分裂/回收 | radix_cache.py |
| 03 · batch-timeline | 为什么批次的成员会变化？ | 结束的请求退出，新请求在迭代边界加入 | t0/t1/t2 | scheduler.py |
| 03 · cpu-gpu-overlap | CPU 调度和 GPU forward 能否部分重叠？ | GPU token 依赖不被消除；仅时间线交错 | 串行/重叠 | scheduler.py |
| 04 · fork-join | 一条语言模型程序如何复用并行前缀？ | fork 后共享父前缀，join 需等待分支产物 | 原路径/并行路径 | Frontend DSL |
| 04 · grammar-fsm | 语法规则怎样收缩 token 候选？ | 状态决定合法候选，过滤不能保证事实 | 状态 q0/q1/q2 | 原论文/官方 docs |
| 05 · pd-topology | P/D 分离时 KV 为什么要传输？ | prefill workers → KV transfer → decode workers | colocated/disaggregated | 官方 P/D |
| 05 · latency-path | 拆分部署可能改善什么、额外付出什么？ | critical path 包含排队/计算/传输/解码 | 关键路径分段 | 成本模型推导 |

## 每图 QA 清单

- [ ] SVG 独立 XML 可解析；\`xmlns\`、\`viewBox\`、\`title\`、\`desc\` 均非空。
- [ ] 图上每个实线箭头都对应真实数据路径/控制流程；虚线用于可选边或跨迭代关系。
- [ ] 标签≥14px，箭头不要压住文字，移动端不靠缩小到不可读来塞满。
- [ ] 色盲场景仍可由标签、线型和 caption 识别；白底打印足够对比。
- [ ] 不把假想 throughput / GPU 数值标成测量结果。
- [ ] 有意义的状态选择会更新文字结论，且不用 JS 仍有默认图。

## 动态讨论页

交互是 progressive enhancement：图片本身为完整默认知识图；按钮与状态描述位于 figure 附近，按键切换的是“共享路径长度 / 调度批成员 / 合法候选 / 拆分成本”等概念，而不是只更换主题色。控制器使用 \`data-stepper\` 并同步 \`aria-pressed\`、\`hidden\`、\`aria-live\`。

## 五个应该主动发现的错误

1. 画成 \`radix tree → 模型权重\`：树索引的是 KV 缓存，不是 weight。
2. 画成 \`Decode 没有 Prefill\`：P/D 分离只改变部署/传输，不改变依赖。
3. 画成 \`overlap 时两轮 token 同时完成\`：重叠的是 CPU 工作与 GPU 计算时段。
4. 画成 \`JSON 结构合法 = 事实正确\`：合法性不是事实核查。
5. 画成 \`缓存键仅为可见字符串\`：真实 KV 复用还受 token ID、模型、适配器、隔离命名空间等限制。
