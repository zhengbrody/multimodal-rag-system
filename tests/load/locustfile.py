"""
Locust load test for the Personal RAG API — a pure-Python alternative to the JMeter
plan (``test_plan.jmx``), runnable without a Java/JMeter install. Mirrors the JMeter
traffic mix: 70% POST ``/ask``, 20% GET ``/health``, 10% GET ``/metrics``.

Run the API in mock mode (zero API cost), then drive it:

    USE_MOCK=true uvicorn api.personal_api:app --app-dir src --port 8000
    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
           --headless -u 100 -r 50 -t 30s --csv tests/load/results/locust

p50/p95/p99 latencies + RPS + failure rate are printed in the Locust summary and,
with ``--csv``, written to ``tests/load/results/locust_stats.csv``.
"""

import csv
import random
from pathlib import Path

from locust import HttpUser, between, task

_CSV = Path(__file__).parent / "questions.csv"
QUESTIONS = []
if _CSV.exists():
    with _CSV.open() as f:
        QUESTIONS = [row["question"] for row in csv.DictReader(f) if row.get("question")]
QUESTIONS = QUESTIONS or ["What are your main skills?"]


class RAGUser(HttpUser):
    """Simulated user: mostly asks questions, some health/metrics polling."""

    wait_time = between(0.1, 0.5)

    @task(7)
    def ask(self):
        self.client.post(
            "/ask",
            json={"question": random.choice(QUESTIONS), "k": 5},
            name="POST /ask",
        )

    @task(2)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="GET /metrics")
