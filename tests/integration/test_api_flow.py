import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure mock mode for integration tests
os.environ["USE_MOCK"] = "true"

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from api.personal_api import app

def test_end_to_end_question_flow():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What is your work experience?",
                "k": 5,
                "use_verification": False,
                "conversational": False,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["question"]
    assert data["answer"]
    assert data["confidence"] in {"high", "medium", "low"}
    assert isinstance(data.get("sources", []), list)
