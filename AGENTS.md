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
- **Speculative Decoding：**区分 EAGLE draft tree、target verification、accepted-prefix commit、token-wise rejection sampling 与 block verification。链式 acceptance/residual 公式不能冒充 EAGLE tree 的完整算法；所有 speedup 必须有受控 benchmark。
- 新增 CPU-only 示例：`page_allocator.py`、`chunk_budget.py`、`online_softmax.py`、`speculative_sampling.py`；验证正常、空预算、页耗尽、对齐、概率归一化等边界。
- 第二批实现快照：`sgl-project/sglang@6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd`。第一批 evidence 仍保留其原始快照，不要无证据地批量重写历史 claim。
