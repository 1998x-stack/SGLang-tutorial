# SGLang Tutorial · 从源码到可视化推理系统

**读懂 SGLang：请求如何执行、共享前缀如何复用、调度如何让 GPU 忙起来，以及生产系统为什么要分离 Prefill 与 Decode。**

[开始阅读](https://1998x-stack.github.io/SGLang-tutorial/) · [官方 SGLang](https://github.com/sgl-project/sglang) · [官方文档](https://docs.sglang.io/) · [原始论文](https://arxiv.org/abs/2312.07104)

> 本仓库是独立编写的中文源码导读教程，并非 SGLang 官方文档。内容以可追溯的论文与上游源码为依据；与运行版本相关的细节必须重新核对。

## 课程设计

| 章 | 核心问题 | 机制图 / 实验 | 证据 |
| --- | --- | --- | --- |
| [01 · 请求生命周期](01.html) | 一次生成请求经历哪些阶段？ | Prefill / Decode token 路径 + KV 显存公式 | SGLang 官方文档、源码 |
| [02 · RadixAttention](02.html) | 多个请求如何共享一段已计算的前缀？ | 压缩基数树 + 最长公共前缀交互 | NeurIPS 2024 论文、radix_cache.py |
| [03 · 调度与重叠](03.html) | CPU 与 GPU 如何在迭代间并发工作？ | 动态批次 + overlap timeline | scheduler.py |
| [04 · 前端编程与约束输出](04.html) | fork / join 与 grammar 如何改变生成过程？ | 分支 DAG + 有限状态机 | 官方前端教程、原始论文 |
| [05 · Prefill–Decode 分离](05.html) | 为什么把输入处理与逐 token 生成拆到不同实例？ | KV 传输拓扑 + 延迟成本模型 | 官方 P/D 文档 |

后续专题路线：KV pool / page 分配、chunked prefill、attention backend、speculative decoding、TP/DP/EP、MoE、量化、HiCache、多模态、Router、可观测性、基准测试与安全部署。严格遵守 **先证据、后扩篇**，避免无意义重复。

## 可视化工程规范

- 每图回答一个明确问题。图形先定义数据依赖，再定义形状、边和颜色，不允许仅为了美观添加假箭头。
- **蓝绿色 = 有效数据/命中；紫色 = 缓存与复用；橙色 = 新计算；灰色 = 未采用或等待**。图形的文字与线型应能在黑白打印和色觉差异下区分语义。
- SVG 均以可独立打开的文件存放在 `assets/diagrams/`，有 `viewBox/title/desc`；文章使用 `img + figcaption`，交互是渐进增强，不依赖 JS 才能理解。
- 每篇先写直觉、示意图，再给数学推导、具体例子、源码位置、误区、限制和参考资料。公式采用原生 MathML，离线可读。
- 所有演示数值是 **toy/derived**，不得标成真实 GPU 基准测试数据；论文结果必须定位表格/实验条件。
- 支持 1440、768、390、360px；不使用 CDN、外部字体或遥测；缩放与键盘聚焦可用。

## 目录

```text
index.html                       课程首页与主题检索
01.html ... 05.html              首批章节
assets/css/site.css              首页与正文通用样式
assets/js/site.js                 无依赖交互
assets/diagrams/*.svg            独立、可复用的机制图
evidence/01.json ... 05.json     claim → 原始来源定位
examples/prefix_match.py         CPU-only 最长公共前缀教学实验
tests/test_content.py            HTML/SVG/证据/离线约束静态检查
.github/workflows/validate.yml   CI 与发布准备
AGENTS.md                        仓库内容生产合同
skills/sglang-visual-tutorial/SKILL.md  课程专用工作流
docs/visual-storyboard.md        SVG 一图一问设计稿
```

## 本地阅读与验证

```bash
python3 -m http.server 8000
# 浏览 http://localhost:8000/
python3 -m unittest discover -s tests -v
python3 examples/prefix_match.py
```

浏览器截图、交互和移动端检查采用 [Olah-style-posts](https://github.com/1998x-stack/Olah-style-posts/tree/main/skills/olah-style-paper-posts) 的浏览器 QA 规约。只有实际完成的测试才能标记通过。Pages 启用方式与部署状态请以 GitHub Actions 和 Pages 设置为准。

## 研究资料（按可信度优先级）

1. Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
2. [SGLang 官方仓库](https://github.com/sgl-project/sglang) 及 [官方文档](https://docs.sglang.io/)；版本敏感的 API/路径以具体 commit 为准。
3. [SGLang 调度器](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)，[RadixCache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)。
4. [SGLang 官方 P/D Disaggregation 文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)，[Frontend DSL](https://github.com/sgl-project/sglang/blob/main/docs/docs/references/frontend/frontend_tutorial.mdx)。
5. [vllm-tutorial](https://github.com/1998x-stack/vllm-tutorial) —— 参考其课程组织与 QA 工程；[Olah-style-posts](https://github.com/1998x-stack/Olah-style-posts) —— 参考其论文图解工作流。内容、绘图和代码均为独立创作，不复制原始素材。

## 编辑与贡献

先读 [AGENTS.md](AGENTS.md)，再读本仓库 Skill；新增主题前检查是否与已有章节重复，建立事实清单与 SVG storyboard，并提供证据 JSON。PR 应附静态检查结果、浏览器 QA 结果、修改前后图示与版本说明；未跑的检查明确标记“未验证”。本仓库不默认拥有第三方素材的版权或许可。
