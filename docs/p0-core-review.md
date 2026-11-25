# P0 · 01–10 章深度审查

基线：PR #12 merge 43976e84f5f0f86a6ce338489897ef68f3e34f8f；其 main Pages run 45596708580 已通过 public smoke。保留 01–05 的 0a8693fbe3a2e46ccc06470a461a2b85865672fd、06–10 的 6a4e10a0ee643dfbe5f59a98b60293bc11fc1bdd。不新增章节。

| 章 | 修复 | 图解/测试 | Evidence（总/固定代码） |
|---|---|---|---|
| 01 | Prefill 首 token 与 Decode 时序；TTFT 时钟边界 | 重绘 inference path；修订 KV desc | 5/3 |
| 02 | LCP 与 input_len−1、page alignment | 读两张 SVG；prefix-cap 回归 | 5/4 |
| 03 | 正常/overlap event loop、WAR、延迟采样 | 读时间线/箭头；对照 event loop | 4/4 |
| 04 | 合法概率质量、非空 JSON 子语法 | 读 fork/FSM；Fraction 概率回归 | 4/2 |
| 05 | 控制 vs KV 数据；串行关键路径及 early/prefill_complete | 重绘 PD topology | 5/5 |
| 06 | page4 vs page7 实验/正文一致 | 读 mapping/lifecycle；对照已有 allocator 测试 | 6/5 |
| 07 | match split vs insert；root sentinel、非根锁计数 | 修订 split/lock 两图 | 6/6 |
| 08 | 预算、对齐、零推进 | 读两张图；既有 chunk_budget | 7/7 |
| 09 | 局部统计、全 mask 与多 tile 数值 | 修订 online state 标签；masked tile CPU 回归 | 6/4 |
| 10 | chain proof、q>0/p=q、block 非 tree | 读两张图；概率守恒回归 | 9/3 |

## 验收范围

- 十篇正文各 section 与 20 张 SVG 文本、路径、描述已阅读；六张 SVG 改动。CI 将重新检查全部 224 page viewports 与 110 SVG geometry（文字越界是 review queue）。并非对所有截图做人工像素级审查。
- 历史实现路径和关键符号已重新打开；重点行范围实读，其他 locator 按符号位置核对。每份 evidence 记录本轮 scope；论文摘要、固定文档及独立数学推导分开。未重新通读全部论文 PDF，也未执行真实 GPU/分布式实验。
- 5 新回归：KV bytes、prefix cap、grammar mass、masked online softmax、speculative probability conservation。保持 55 章、110 SVG、50 CPU labs、167 DAG edges，新增证据条目。
- 仅完整 CI 通过后合并，再独立确认 Pages；无 GPU 性能承诺。
