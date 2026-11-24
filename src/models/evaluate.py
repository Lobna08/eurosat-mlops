"""
Stage DVC 'evaluate' : metriques sur le TEST isole, mesures d'inference,
puis enregistrement du modele dans le Model Registry.

Premier et seul endroit ou la partition test est lue. Les resultats sont
ecrits dans metrics/scores.json (metrique DVC, commitee dans Git) et ajoutes
au meme run MLflow que l'entrainement (run_id lu dans models/run_info.json).

Ordre : entrainer (train.py) -> evaluer -> enregistrer en Staging. Le modele
n'entre dans le Registry qu'avec ses metriques test, citees dans la
description de la version. La version creee est ecrite dans
metrics/registry.json (models/run_info.json, sortie du stage train, n'est
jamais reecrit ici).
L'export ONNX + quantification seront ajoutes ici au checkpoint B.
"""

import json
import statistics
import time
from pathlib import Path

import matplotlib
import mlflow
import torch
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import PROCESSED_DIR, EuroSATDataset, build_transforms
from src.models.architecture import build_resnet50, count_parameters

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_PATH = Path("models/resnet50_linear_probe.pt")
METRICS_DIR = Path("metrics")


@torch.no_grad()
def predict(model, loader):
    preds, labels = [], []
    for images, targets in loader:
        preds.append(model(images).argmax(1))
        labels.append(targets)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def measure_latency_ms(model, img_size, warmup=5, runs=30):
    """Latence PyTorch CPU, batch 1, mediane en ms (reference avant ONNX Runtime)."""
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(warmup):
        model(x)
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        timings.append((time.perf_counter() - t0) * 1000)
    return statistics.median(timings)


def register_staging(run_id, model_name, scores):
    """Model Registry (convention TP3) : nouvelle version -> Staging.
    Appele seulement apres le log des metriques test dans le meme run.
    La promotion en Production se fera apres comparaison CNN vs ViT."""
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", model_name).version
    client.transition_model_version_stage(model_name, version, stage="Staging")
    client.update_model_version(
        model_name,
        version,
        description=(
            f"Baseline ResNet50 linear probe (CPU). Test : F1 macro "
            f"{scores['test_f1_macro']:.4f} sur {int(scores['test_samples'])} "
            "images. Candidat Staging en attente de la comparaison avec ViT-B/16."
        ),
    )
    return version


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    cfg, ml_cfg = params["train"], params["mlflow"]
    with open("models/run_info.json", "r", encoding="utf-8") as f:
        run_info = json.load(f)
    classes = run_info["classes"]

    model = build_resnet50(len(classes), freeze_backbone=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    test_set = EuroSATDataset(
        PROCESSED_DIR / "test.csv", build_transforms(cfg["img_size"], train=False)
    )
    loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)
    y_true, y_pred = predict(model, loader)

    total, _ = count_parameters(model)
    scores = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1_macro": f1_score(y_true, y_pred, average="macro"),
        "test_f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "test_precision_macro": precision_score(y_true, y_pred, average="macro"),
        "test_recall_macro": recall_score(y_true, y_pred, average="macro"),
        "latency_ms_cpu_torch_b1": measure_latency_ms(model, cfg["img_size"]),
        "model_size_mb": MODEL_PATH.stat().st_size / 1e6,
        "params_total": total,
        "test_samples": len(y_true),
    }
    scores = {k: round(float(v), 4) for k, v in scores.items()}

    METRICS_DIR.mkdir(exist_ok=True)
    with open(METRICS_DIR / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    with open(METRICS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=range(len(classes)),
        display_labels=classes,
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title("Matrice de confusion - test (ResNet50 linear probe)")
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metrics(scores)
        mlflow.log_metrics({f"test_f1_{c}": report[c]["f1-score"] for c in classes})
        mlflow.log_artifact(
            str(METRICS_DIR / "confusion_matrix.png"), "diagnostic_plots"
        )
        mlflow.log_artifact(str(METRICS_DIR / "classification_report.json"), "reports")

    model_name = ml_cfg["registered_model"]
    version = register_staging(run_info["run_id"], model_name, scores)
    registry = {
        "registered_model": model_name,
        "model_version": int(version),
        "stage": "Staging",
        "run_id": run_info["run_id"],
    }
    with open(METRICS_DIR / "registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    for k, v in scores.items():
        print(f"[evaluate] {k}: {v}")
    print(f"[evaluate] {model_name} v{version} -> Staging")


if __name__ == "__main__":
    main()
