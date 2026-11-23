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
