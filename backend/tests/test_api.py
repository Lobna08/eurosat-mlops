"""
Tests de l'API avec le TestClient de FastAPI (convention TP4) : les 4
endpoints, les erreurs 400 / 413 / 422 et le 503 quand le modele est absent.

Les tests d'inference utilisent le vrai modele ONNX (sortie DVC du stage
export) et sont ignores s'il n'a pas ete genere (dvc repro / dvc pull).
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import MAX_UPLOAD_BYTES, app, settings

MODEL_MISSING = not all(path.exists() for path in settings().values())
needs_model = pytest.mark.skipif(
    MODEL_MISSING, reason="Modele ONNX absent : lancer `dvc repro export`"
)


def make_image_bytes(fmt="JPEG", size=64, seed=0):
    """Image RGB aleatoire au format des tuiles EuroSAT (64 x 64)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@needs_model
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "EuroSAT-ResNet50" in data["model_version"]
    assert isinstance(data["gpu_available"], bool)
    assert "CPUExecutionProvider" in data["providers"]


@needs_model
def test_model_metadata_endpoint(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "EuroSAT-ResNet50"
    assert data["registry_version"] >= 1
    assert len(data["classes"]) == 10
    assert "test_f1_macro" in data["test_metrics"]


@needs_model
@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_predict_valid_image(client, fmt):
    files = {"file": (f"tile.{fmt.lower()}", make_image_bytes(fmt), "image/*")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    probs = data["probabilities"]
    assert len(probs) == 10
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert data["predicted_class"] == max(probs, key=probs.get)
    assert data["confidence"] == pytest.approx(probs[data["predicted_class"]])
    assert data["server_latency_ms"] >= data["inference_ms"] > 0


@needs_model
def test_predict_visualize_returns_png(client):
    files = {"file": ("tile.jpg", make_image_bytes(), "image/jpeg")}
    response = client.post("/predict/visualize", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-predicted-class"] in client.app.state.service.classes
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "PNG"
    assert image.size == (224, 224)


@needs_model
@pytest.mark.parametrize("endpoint", ["/predict", "/predict/visualize"])
def test_invalid_image_returns_400(client, endpoint):
    files = {"file": ("notes.txt", b"ceci n'est pas une image", "text/plain")}
    response = client.post(endpoint, files=files)
    assert response.status_code == 400


@needs_model
def test_empty_file_returns_400(client):
    response = client.post("/predict", files={"file": ("vide.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


@needs_model
def test_too_large_file_returns_413(client):
    files = {"file": ("big.jpg", b"0" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413


def test_predict_without_file_returns_422(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_model_not_loaded_returns_503(client, monkeypatch, tmp_path):
    """Demarrage avec un MODEL_PATH absent : l'API repond, mais en 503."""
    # Le lifespan remplace app.state.service : on le restaure apres le test
    monkeypatch.setattr(app.state, "service", getattr(app.state, "service", None))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.onnx"))
    with TestClient(app) as unavailable:
        assert unavailable.get("/health").status_code == 503
        assert unavailable.get("/model/metadata").status_code == 503
        files = {"file": ("tile.jpg", make_image_bytes(), "image/jpeg")}
        assert unavailable.post("/predict", files=files).status_code == 503
        assert unavailable.post("/predict/visualize", files=files).status_code == 503
