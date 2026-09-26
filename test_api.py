from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_status():
    response = client.post(
        "/predict",
        json={"text": "claim your free cash prize now"}
    )
    assert response.status_code == 200


def test_predict_schema():
    response = client.post(
        "/predict",
        json={"text": "please send the college notes tomorrow"}
    )

    data = response.json()

    assert "text" in data
    assert "prediction" in data
    assert "spam_probability" in data
    assert "ham_probability" in data

    assert data["prediction"] in ["Spam", "Ham"]
    assert 0 <= data["spam_probability"] <= 1
    assert 0 <= data["ham_probability"] <= 1


def test_invalid_request():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422
