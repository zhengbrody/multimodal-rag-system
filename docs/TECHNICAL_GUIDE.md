# Personal RAG Portfolio 技术说明

本文档用于说明这个 project 如何启动、包含哪些模块、RAG 如何工作、CI/CD 如何验证，以及面试时可以重点讲什么。

## 1. 项目定位

这个项目最适合定位为：

**Recruiter-facing Personal AI Portfolio + Production-minded RAG Case Study**

它不是普通聊天 demo，而是一个面向招聘者的个人知识问答系统。招聘者可以直接问：

- What is Zheng's strongest machine learning project?
- Why is Zheng a fit for Machine Learning Engineer roles?
- What did Zheng build at Allianz?
- What is Zheng's work authorization and availability?

系统会基于本地知识库检索相关内容，再生成带 confidence 和 sources 的回答，展示你对 RAG、后端 API、前端 demo、测试、CI/CD、部署和工程权衡的理解。

## 2. 项目包含什么

核心目录：

```text
multimodal-rag-system/
├── data/raw/knowledge_base.json      # 个人知识库，简历/项目/FAQ 的事实来源
├── frontend/personal_app.py          # Streamlit 求职展示界面
├── src/api/personal_api.py           # FastAPI 后端，暴露 ask/health/metrics 等接口
├── src/rag/knowledge_processor.py    # 把 JSON 知识库转换成可检索 documents
├── src/rag/mock_retriever.py         # 无外部依赖的关键词检索器，CI 和 demo 默认使用
├── src/rag/retriever.py              # OpenAI embedding + FAISS 语义检索器
├── src/rag/mock_pipeline.py          # 无 LLM 成本的本地回答生成管线
├── src/rag/pipeline.py               # OpenAI LLM RAG 生成管线
├── src/utils/cache.py                # 简单 TTL cache
├── src/utils/monitoring.py           # 轻量监控和性能记录
├── tests/                            # API 和 retriever 测试
├── .github/workflows/ci.yml          # GitHub Actions CI
├── docker-compose.yml                # API + frontend 容器化启动
└── docs/                             # 项目文档
```

关键设计：

- **Streamlit-only mode**：没有后端也能跑，适合 Streamlit Cloud 免费公开 demo。
- **Mock mode**：默认不需要 OpenAI key，便于测试、CI 和面试演示。
- **OpenAI mode**：可切换到 embedding + LLM 的真实语义检索和生成。
- **Source tracing**：回答带 sources，避免像黑盒聊天机器人。
- **Confidence scoring**：回答显示 high/medium/low，体现 RAG 质量控制。

## 3. 如何启动

### 3.1 最轻量本地启动：只跑 Streamlit

适合快速展示 UI，不需要 OpenAI key，不需要后端。

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/personal_app.py
```

打开：

```text
http://localhost:8501
```

如果没有配置 `API_URL`，前端会使用 `data/raw/knowledge_base.json` 和本地 mock RAG 逻辑回答问题。

### 3.2 本地完整启动：FastAPI + Streamlit

适合展示系统结构和 API。

```bash
pip install -r requirements_simple.txt
USE_MOCK=true python run.py
```

打开：

```text
Frontend: http://localhost:8501
API docs: http://localhost:8000/docs
Health:   http://localhost:8000/health
```

也可以分开启动：

```bash
USE_MOCK=true uvicorn src.api.personal_api:app --reload --port 8000
API_URL=http://localhost:8000 streamlit run frontend/personal_app.py
```

### 3.3 OpenAI 语义模式

适合展示真实 embedding 和 LLM 生成效果。

```bash
cp .env.example .env
```

在 `.env` 中设置：

```bash
OPENAI_API_KEY=sk-...
USE_MOCK=false
LLM_MODEL=gpt-3.5-turbo
```

然后启动：

```bash
USE_MOCK=false python run.py
```

如果修改了知识库，建议重建索引：

```bash
curl -X POST http://localhost:8000/rebuild-index
```

### 3.4 Docker 启动

```bash
docker-compose up --build
```

默认 `USE_MOCK=true`，所以不加 OpenAI key 也可以启动。

## 4. RAG 请求链路

一次问答的流程：

```text
User question
  -> Streamlit frontend
  -> FastAPI /ask endpoint, or local fallback when API is unavailable
  -> Retriever gets top-k relevant documents
  -> RAG pipeline builds grounded answer
  -> Response returns answer + confidence + sources
  -> UI renders answer, source evidence, and system metadata
```

这个设计的重点不是“让模型自由发挥”，而是把回答限制在 personal knowledge base 中。对于求职项目，这一点很重要，因为招聘者关心的是可验证事实，而不是泛泛的 AI 回答。

## 5. 知识库如何更新

主要文件：

```text
data/raw/knowledge_base.json
```

建议把内容维护成事实化、可追溯的 Q&A 或结构化条目：

- `personal_info`：身份、地点、联系方式、work authorization、availability。
- `skills`：技能栈，最好按 ML/backend/frontend/cloud 分组。
- `projects`：重点项目，包含 problem/action/result/tech stack。
- `experience`：实习和工作经历，强调真实产出。
- `education`：学校、专业、课程、毕业时间。
- `faq`：招聘者最可能问的问题。

更新后：

- Streamlit-only mode 会在当前进程中重新读取本地知识库。
- FastAPI mode 可调用 `/rebuild-index`。
- 如果使用 OpenAI/FAISS mode，需要确保 processed index 重新生成。

内容建议：

- 优先写具体事实，例如项目规模、技术选择、结果、约束和 tradeoff。
- 避免夸张指标，除非有日志、测试或公开材料支持。
- 把最能体现 MLE/backend/RAG 能力的项目放在 FAQ 和 projects 里，提升检索命中率。

## 6. CI/CD 如何保证没有问题

当前 CI 文件：

```text
.github/workflows/ci.yml
```

每次 push 到 `main` 或 `develop`，以及 PR 到这两个分支时，会运行：

- Python 3.11 测试
- Python 3.12 测试
- `pytest tests/ -v --cov=src --cov-report=term-missing`
- `black --check src/ tests/`
- `ruff check src/ tests/`
- `mypy --ignore-missing-imports src/ || true`

本地提交前建议运行：

```bash
python -m pytest
black --check src/ tests/
ruff check src/ tests/
python -m py_compile run.py frontend/personal_app.py src/api/personal_api.py src/rag/*.py src/utils/*.py
```

推送后建议确认 GitHub Actions：

```bash
gh run list --branch main --limit 5
gh run watch <run-id> --exit-status
```

## 7. Streamlit Cloud 部署要点

Streamlit Cloud 配置：

```text
Repository: your GitHub repository
Branch: main
Main file path: frontend/personal_app.py
Requirements file: frontend/requirements.txt
```

Secrets：

- 没有独立后端时，不需要设置 `API_URL`，前端会自动使用本地 fallback。
- 如果部署了 FastAPI 后端，再设置：

```toml
API_URL = "https://your-backend-api-url.com"
```

公开访问检查：

```bash
curl -I https://your-app.streamlit.app/
```

如果返回跳转到 `share.streamlit.io/-/auth/app`，说明 app 还不是公开访问。需要在 Streamlit Cloud 的 app settings 中把访问权限改成 public 或 anyone with the link，否则招聘者会看到登录页。

## 8. 面试可以讲的技术点

可以重点讲：

- 如何把个人简历和项目经历转换成可检索知识库。
- 为什么求职场景需要 source tracing 和 confidence，而不是只给一个聊天回答。
- Mock retriever 和 OpenAI retriever 的 tradeoff：成本、稳定性、语义能力、CI 可测性。
- 为什么 Streamlit-only fallback 对 portfolio demo 很重要：部署更稳、没有后端也可访问。
- 如何通过 pytest、black、ruff、GitHub Actions 保证改动不破坏核心逻辑。
- 如何设计 anti-hallucination：低温度、严格 prompt、只基于 retrieved context、无法回答时明确说明。
- 未来如何扩展：evaluation set、retrieval metrics、reranker、authority metadata、多数据源 ingestion。

## 9. 当前限制和下一步

当前限制：

- 公开 demo 的访问权限依赖 Streamlit Cloud 设置，代码无法代替平台权限配置。
- Mock mode 更适合稳定展示和 CI，不等于真实 embedding 语义检索质量。
- 还没有独立的 retrieval evaluation dataset，例如 recall@k、MRR、answer groundedness。
- 知识库仍主要手写维护，后续可以从 resume、project README、LinkedIn export 等来源自动 ingestion。

建议下一步：

- 增加 `eval/questions.json`，固定 20-30 个招聘者问题并记录期望 sources。
- 给每条 knowledge document 增加 `authority_level`、`source_file`、`last_updated`。
- 在 UI 中增加 “Why this answer?” 面板，展示检索分数和证据片段。
- 把最强项目做成单独 case study 页面，和 RAG 问答互相链接。

## 10. 一句话项目介绍

Built a recruiter-facing personal RAG system with FastAPI, Streamlit, pluggable retrievers, source tracing, confidence scoring, and CI/CD validation to answer grounded questions over resume and project knowledge.
