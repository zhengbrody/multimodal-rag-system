# 项目求职展示整理与改进路线

## 当前定位

这个项目最适合包装成 **Recruiter-facing Personal AI Search + Production RAG Case Study**，而不是单纯的 Streamlit 聊天 demo。

核心叙事应该是：

> 我用 RAG 做了一个可嵌入个人网站的智能问答系统，让 recruiter / 面试官可以直接问我的经历、项目、技能、求职状态，并且每个回答都有来源、置信度和反幻觉控制。

这比“我做了一个 RAG demo”更强，因为它同时展示了三件事：

- **产品意识**：解决个人网站/求职展示中信息查找成本高的问题。
- **ML 系统能力**：知识库处理、检索、生成、验证、置信度、来源追踪。
- **工程能力**：FastAPI、Streamlit、测试、CI、Docker、Kubernetes 配置、监控、缓存、部署路径。

## 你的强展示点

### 1. Production ML Engineer 叙事清晰

你的资料里最强的主线不是“会很多工具”，而是：

> 能把 ML 从实验、比赛、研究推进到可服务真实用户/业务的系统。

可以反复围绕这些证据讲：

- Allianz：200K+ daily transactions，fraud detection，precision +15%，latency 150ms -> 105ms。
- Penn State：NLP/entity extraction，10K+ docs/day，92% F1，MLflow/Airflow/Optuna。
- Kaggle：LLM human preference / inference optimization，QLoRA、Gemma/Llama、ensemble、quantization。
- Personal RAG：把自己的求职资料变成可检索、可验证、可扩展的 AI agent-facing knowledge base。

### 2. 项目本身能展示系统设计

当前代码已经具备不错的面试讨论材料：

- Retriever 抽象：mock/OpenAI/Pinecone/CLIP 可切换。
- Pipeline 抽象：普通 RAG 和 conversational RAG。
- Anti-hallucination：低温度、严格 prompt、confidence、source tracing、verification。
- API 能力：`/ask`、`/health`、`/stats`、`/metrics`、`/feedback`。
- 运维意识：structured logging、monitoring、cache、Docker、K8s manifests。

面试时可以把它讲成一个小型生产系统，而不是课设。

## 目前主要问题

### 1. README 叙事偏泛

README 现在更像通用开源项目介绍，还没有把 “Zheng Dong 的个人求职展示系统” 放到第一屏中心。建议第一屏直接回答：

- 这个项目为谁解决什么问题？
- 为什么它能证明你适合 MLE/SDE？
- demo 能问什么？
- 哪些指标是可复现的，哪些只是目标或扩展？

### 2. 指标需要分层，避免过度承诺

项目里有 `92% Recall@5`、`500 concurrent users`、`99.9% uptime` 这类强指标。它们对简历有吸引力，但公开仓库里需要配套证据：

- evaluation dataset
- load test command
- hardware / cloud config
- run date
- result artifact

否则建议在 README 中写成 “designed/evaluated hooks for ...” 或 “load-tested in local/simulated setup ...”，不要写成绝对生产 SLA。

### 3. UI 还像 demo，不像 portfolio 产品

Streamlit UI 有功能，但第一眼还是“聊天 demo”。更强的求职展示体验应该是：

- 首屏就是 “Ask Zheng's AI Portfolio”。
- 默认问题围绕 recruiter 场景：work authorization、availability、strongest project、Allianz deep dive、why MLE。
- Sources 面板展示知识库证据，而不是只展示技术源文档。
- About tab 改成 “Why this project matters” 和 “System design”。

### 4. 知识库需要权威来源策略

我已经根据本机 `Desktop/Generateresuem/RESUME_RULES.md` 更新了知识库中的权威联系方式和时间线：

- Email: `zhengdong0317@gmail.com`
- LinkedIn: `linkedin.com/in/zhengdong17`
- Allianz: `Jun-Aug 2024`
- Penn State RA: `Sep 2024-Jan 2025`
- Qingdao: `Jun-Sep 2023`

后续建议把 resume rules、简历 JSON、项目 README 都作为 ingestion sources，并在 metadata 中标注 `source_type` 和 `authority_level`。

## 下一步优先级

### P0: 可信度与一致性

- 添加 `data/evaluation/career_questions.jsonl`，至少 50 个 recruiter / 面试问题。
- 添加 `scripts/evaluate_retrieval.py`，输出 Recall@K、MRR、hit rate。
- 给 load test 保存固定结果：JMeter CSV、日期、机器配置。
- README 中只保留有证据的指标。

### P1: 招聘展示体验

- 重写 README 顶部 150 行，改成 portfolio-first。
- Streamlit 首屏改名为 “Ask Zheng's AI Portfolio”。
- 默认问题改成求职问题，而不是通用 RAG 示例。
- 增加 screenshots / demo GIF / 2 分钟 walkthrough。

### P2: 知识库工程化

- 增加 resume/project README ingestion script。
- 给每个 document 加 `source_file`、`authority_level`、`last_updated`。
- 对冲突信息做 priority：`RESUME_RULES.md > resume_content.json > knowledge_base.json > generated notes`。
- 增加敏感字段过滤，例如 tax、visa documents、password exports 永不 ingestion。

### P3: 技术加分项

- 增加 faithfulness checker 或 LLM-as-judge evaluation。
- 引入 reranking，对 recruiter-style questions 提升排序质量。
- 增加 admin 页面更新知识库。
- 公网部署并加 monitoring snapshot。

## 建议简历 bullet

**MLE 版本**

> Built a recruiter-facing personal RAG system with FastAPI, Streamlit, OpenAI embeddings, and pluggable retrievers; implemented source tracing, confidence scoring, feedback collection, and anti-hallucination verification for grounded Q&A over resume/project knowledge.

**SDE / backend 版本**

> Developed a production-style Q&A service with FastAPI, typed schemas, structured logging, metrics, Docker deployment, and CI tests; designed retriever/pipeline abstractions supporting mock, OpenAI, Pinecone, and CLIP-backed retrieval modes.

**更强但需要评测支撑的版本**

> Evaluated the RAG system on a curated recruiter-question set and load-tested API latency under simulated concurrent traffic, publishing reproducible scripts and result artifacts for retrieval quality and service performance.

## 面试 Demo 脚本

按这个顺序演示最稳：

1. “Who are you?”
2. “What is your strongest ML systems project?”
3. “Tell me about your Allianz experience.”
4. “Why are you qualified for MLE roles?”
5. “What sources support this answer?”
6. “How would you scale this system beyond a personal website?”

这个顺序能展示：个人品牌、经历、技术深度、RAG grounding、系统设计。
