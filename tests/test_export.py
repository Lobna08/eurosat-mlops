"""
Tests du passage PyTorch -> API ONNX : le pretraitement NumPy du backend
reproduit build_transforms(train=False), le backend n'importe pas torch,
et les sorties ONNX permettent de recalculer les logits (base de la CAM).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from backend.app.inference import load_image, preprocess
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms

BACKEND_DIR = Path("backend")
MODEL_DIR = Path("models")
TEST_CSV = Path("data/processed/test.csv")


def assert_same_tensor(image, img_size=224):
    expected = build_transforms(img_size, train=False)(image).unsqueeze(0).numpy()
    actual = preprocess(image, img_size, IMAGENET_MEAN, IMAGENET_STD)
    assert actual.shape == expected.shape == (1, 3, img_size, img_size)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_numpy_preprocess_matches_torchvision_synthetic():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    assert_same_tensor(image)


@pytest.mark.skipif(not TEST_CSV.exists(), reason="Lancer dvc repro d'abord")
def test_numpy_preprocess_matches_torchvision_on_test_images():
    """Memes octets JPEG decodes comme dans l'API (load_image)."""
    paths = pd.read_csv(TEST_CSV)["path"].iloc[::400]
    for path in paths:
        assert_same_tensor(load_image(Path(path).read_bytes()))


def test_backend_does_not_import_torch():
    code = "import sys, app.main; sys.exit(int('torch' in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], cwd=BACKEND_DIR)
    assert result.returncode == 0


@pytest.mark.skipif(
    not (MODEL_DIR / "resnet50.onnx").exists(), reason="Lancer dvc repro export"
)
def test_onnx_features_reproduce_logits():
    """logits = W . GAP(features) + b : la CAM utilise bien la tete du modele."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(MODEL_DIR / "resnet50.onnx"), providers=["CPUExecutionProvider"]
    )
    fc = np.load(MODEL_DIR / "fc_weights.npz")
    x = np.random.default_rng(0).standard_normal((2, 3, 224, 224), dtype=np.float32)
    logits, features = session.run(["logits", "features"], {"image": x})
    assert features.shape == (2, 2048, 7, 7)
    recomputed = features.mean(axis=(2, 3)) @ fc["weight"].T + fc["bias"]
    np.testing.assert_allclose(recomputed, logits, atol=1e-4)
