from fastapi.testclient import TestClient

from api.index import app
from api.service import load_artifact

client = TestClient(app)
PAYLOAD = {"RevolvingUtilizationOfUnsecuredLines": 0.2, "age": 45, "NumberOfTime30_59DaysPastDueNotWorse": 0, "DebtRatio": 0.35, "MonthlyIncome": 5000, "NumberOfOpenCreditLinesAndLoans": 8, "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1, "NumberOfTime60_89DaysPastDueNotWorse": 0, "NumberOfDependents": 1}


def test_health_is_available():
    assert client.get("/api/health").status_code == 200


def test_prediction_is_informational_and_deterministic():
    first = client.post("/api/predict", json=PAYLOAD)
    second = client.post("/api/predict", json=PAYLOAD)
    assert first.status_code == second.status_code == 200
    assert 0 <= first.json()["probability_of_default"] <= 1
    assert first.json()["probability_of_default"] == second.json()["probability_of_default"]
    assert first.json()["recommendation"] == "requiere revisión"


def test_invalid_input_is_rejected():
    invalid = {**PAYLOAD, "age": 10}
    assert client.post("/api/predict", json=invalid).status_code == 422


def test_model_unavailable_returns_503(monkeypatch):
    load_artifact.cache_clear()
    monkeypatch.setattr("api.service.MODEL_PATH", __import__("pathlib").Path("missing.pkl"))
    assert client.post("/api/predict", json=PAYLOAD).status_code == 503
    monkeypatch.undo()
    load_artifact.cache_clear()
