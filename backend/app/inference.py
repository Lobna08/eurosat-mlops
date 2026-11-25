"""
Service d'inference ONNX Runtime (sans PyTorch) : chargement du modele,
pretraitement NumPy, prediction et carte d'activation (CAM).

Le pretraitement reproduit build_transforms(img_size, train=False) de
src/data/dataset.py : resize bilineaire PIL en img_size x img_size, passage
dans [0, 1], normalisation ImageNet. Un test verifie l'egalite des tenseurs.

CAM (Zhou et al., 2016) : le modele exporte renvoie aussi les feature maps de
layer4 (2048 x 7 x 7). Comme la tete est GAP + couche lineaire, la carte de la
classe c est sum_k W[c, k] * F_k. Pour cette architecture, Grad-CAM donne la
meme carte (le gradient du logit par rapport a F_k vaut W[c, k] / 49).
"""

import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Le fichier recu n'est pas une image lisible."""


def load_image(data):
    """Decode des octets en image PIL RGB, ou leve InvalidImageError."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Fichier illisible comme image") from exc
    return image.convert("RGB")


def preprocess(image, img_size, mean, std):
    """Image PIL RGB -> tenseur float32 (1, 3, img_size, img_size)."""
    resized = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[np.newaxis]


def softmax(logits):
    z = np.exp(logits - logits.max())
    return z / z.sum()


def jet_colormap(values):
    """Palette 'jet' (bleu -> rouge) pour des valeurs dans [0, 1] -> uint8 RGB."""
    v = values[..., np.newaxis]
    rgb = np.clip(1.5 - np.abs(4.0 * v - np.array([3.0, 2.0, 1.0])), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def overlay_cam(image, cam, size, alpha=0.45):
    """Superpose la CAM (normalisee dans [0, 1]) a l'image, renvoie un PNG."""
    base = np.asarray(
        image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
    )
    heat = Image.fromarray(cam.astype(np.float32)).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    colored = jet_colormap(np.clip(np.asarray(heat), 0.0, 1.0)).astype(np.float32)
    blended = ((1 - alpha) * base + alpha * colored).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format="PNG")
    return buffer.getvalue()


class InferenceService:
    """Encapsule le modele ONNX et ses metadonnees (ne connait pas HTTP)."""

    def __init__(self, model_path, metadata_path, cam_weights_path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.cam_weights_path = Path(cam_weights_path)
        self.session = None
        self.metadata = {}
        self.cam_weights = None
        self.load()

    def load(self):
        """Charge la session ONNX Runtime, les metadonnees et les poids de fc."""
        for path in (self.model_path, self.metadata_path, self.cam_weights_path):
            if not path.exists():
                raise FileNotFoundError(f"Fichier introuvable : {path}")
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.cam_weights = np.load(self.cam_weights_path)["weight"]

    def is_ready(self):
        return self.session is not None

    @property
    def classes(self):
        return self.metadata["classes"]

    @property
    def version(self):
        meta = self.metadata
        return (
            f"{meta['model_name']} v{meta['registry_version']} "
            f"({meta['registry_stage']})"
        )

    def run(self, image):
        """Image PIL -> (probabilites, feature maps, temps d'inference en ms)."""
        x = preprocess(
            image,
            self.metadata["img_size"],
            self.metadata["mean"],
            self.metadata["std"],
        )
        t0 = time.perf_counter()
        logits, features = self.session.run(["logits", "features"], {"image": x})
        inference_ms = (time.perf_counter() - t0) * 1000
        return softmax(logits[0]), features[0], inference_ms

    def predict(self, data):
        """Octets d'image -> dict (classe, probabilites, latences en ms)."""
        t0 = time.perf_counter()
        probs, _, inference_ms = self.run(load_image(data))
        best = int(probs.argmax())
        return {
            "predicted_class": self.classes[best],
            "confidence": float(probs[best]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "inference_ms": inference_ms,
            "server_latency_ms": (time.perf_counter() - t0) * 1000,
            "model_version": self.version,
        }

    def visualize(self, data):
        """Octets d'image -> (PNG de la CAM superposee, classe, latence en ms)."""
        t0 = time.perf_counter()
        image = load_image(data)
        probs, features, _ = self.run(image)
        best = int(probs.argmax())
        cam = np.maximum(np.tensordot(self.cam_weights[best], features, axes=1), 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        png = overlay_cam(image, cam, self.metadata["img_size"])
        return png, self.classes[best], (time.perf_counter() - t0) * 1000
