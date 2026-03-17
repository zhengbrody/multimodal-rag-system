## 📊 Metrics & Evaluation Notes

This project is intended as a **personal RAG / portfolio system**, so all metrics should be interpreted in that context rather than as hard production SLAs.

### 1. Runtime Metrics (built into the API)

The FastAPI backend exposes a `/metrics` endpoint and feeds data into the Streamlit Analytics tab. Out of the box it tracks:

- **`requests_total`** – total HTTP requests handled by the API
- **`questions_total`** – number of Q&A requests
- **`feedback_total`** – how many feedback events were submitted
- **`errors_total`** – number of failed requests (status code ≥ 400)
- **`average_latency_ms`** – mean latency since startup
- **`uptime_seconds`** – how long the API has been running
- **`error_rate_percent`** – error percentage over all requests

These numbers are **environment-dependent** (your laptop vs. a cloud VM) and are not pinned to any fixed target like “<500 ms p95”.

### 2. Retrieval / RAG Quality Metrics

The codebase is structured so that you can plug in your own evaluation scripts for:

- **Recall@K**
- **MRR / NDCG**
- **Simple hit-rate style checks for specific questions**

Typical workflow:

1. Prepare a small evaluation set of question–answer or question–document pairs that reflect your personal knowledge base.
2. Run them through the `/ask` endpoint or a small Python script using the same retriever.
3. Compute Recall@K / MRR / NDCG using your favourite library or a custom script.

Because this is a personal portfolio project, **no global “95%+ Recall@5” claim is hard-coded here**. Any numbers you publish should clearly state:

- the dataset (e.g., “50 manually curated interview-style questions about my CV”)
- the hardware (e.g., “local M1 MacBook Pro”)
- the configuration (mock vs. OpenAI mode, K value, temperature, verification on/off)

### 3. How to Report Numbers in a Resume or README

To keep things honest and reproducible, prefer wording like:

- “Implemented evaluation hooks (Recall@K/MRR) and a small benchmark script; results depend on dataset and hardware.”
- “On my local test set of 50 career questions, the RAG pipeline correctly surfaced a relevant document in the top‑3 ~X% of the time.”
- “Latency on my laptop is typically in the **Y–Z ms** range for mock mode; OpenAI mode adds network overhead.”

Avoid claiming universal guarantees (for example “99.9% uptime” or “95%+ recall”) unless you have long‑running logs and a public script that others can run to reproduce them.




