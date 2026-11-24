"""
Stage DVC 'train' : baseline ResNet50 en linear probe + tracking MLflow.

Linear probe : le backbone ImageNet est gele, on extrait une seule fois les
features (2048-d) des images train/val, puis on entraine uniquement la couche
fc sur ces features. Choix impose par la machine temporaire (CPU 2 coeurs) :
un fine-tuning complet est reporte a la machine GPU (checkpoint C).
Consequence assumee : les augmentations ne sont pas utilisees ici (features
pre-calculees une fois), elles le seront lors du fine-tuning.

Le meilleur epoch est choisi sur le F1 macro de VALIDATION ; le test n'est
jamais lu dans ce script.

Le modele est logge dans le run MLflow mais N'EST PAS enregistre dans le
Model Registry ici : l'enregistrement (Staging) est fait par evaluate.py,
une fois les metriques test connues (entrainer -> evaluer -> enregistrer).
"""

import json
import platform
import random
import time
from pathlib import Path

import matplotlib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import PROCESSED_DIR, EuroSATDataset, build_transforms
from src.models.architecture import build_resnet50, count_parameters, extract_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_version():
    """Hash DVC de data/raw : relie le run MLflow a la version exacte des donnees."""
    with open("data/raw.dvc", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def make_loader(split, img_size, batch_size):
    # train=False : pas d'augmentation, les features sont calculees une seule fois.
    dataset = EuroSATDataset(
        PROCESSED_DIR / f"{split}.csv", build_transforms(img_size, train=False)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_head(head, feats, labels, criterion):
    head.eval()
    with torch.no_grad():
        logits = head(feats)
        loss = criterion(logits, labels).item()
    preds = logits.argmax(1)
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return loss, acc, f1


def train_head(head, train_feats, train_labels, val_feats, val_labels, cfg, seed):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed)
    history, best = [], {"val_f1_macro": -1.0}

    for epoch in range(1, cfg["epochs"] + 1):
        head.train()
        order = torch.randperm(len(train_feats), generator=generator)
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start : start + cfg["batch_size"]]
            optimizer.zero_grad()
            loss = criterion(head(train_feats[idx]), train_labels[idx])
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc, _ = evaluate_head(head, train_feats, train_labels, criterion)
        va_loss, va_acc, va_f1 = evaluate_head(head, val_feats, val_labels, criterion)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1_macro": va_f1,
        }
        history.append(row)
        mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(
            f"[train] epoch {epoch:02d} loss={tr_loss:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}"
        )
        if va_f1 > best["val_f1_macro"]:
            best = {
                **row,
                "state": {k: v.clone() for k, v in head.state_dict().items()},
            }

    head.load_state_dict(best.pop("state"))
    return pd.DataFrame(history), best


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(history["epoch"], history["train_acc"], label="train acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="val acc")
    axes[1].plot(history["epoch"], history["val_f1_macro"], label="val F1 macro")
    axes[1].set_title("Accuracy / F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def detect_device():
    """Device reellement disponible a l'execution (CPU sur la machine temporaire)."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), platform.processor() or platform.machine()


def main():
    params = load_params()
    seed, cfg, ml_cfg = params["seed"], params["train"], params["mlflow"]
    set_seed(seed)
    device, device_name = detect_device()

    with open(PROCESSED_DIR / "split_report.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment"])
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=f"{cfg['model']}-{cfg['strategy']}") as run:
        model = build_resnet50(len(classes), freeze_backbone=True)
        total, trainable = count_parameters(model)

        mlflow.log_params({**cfg, "seed": seed, "num_classes": len(classes)})
        mlflow.log_params(
            {f"subset_{k}": v for k, v in params["prepare"]["subset_per_class"].items()}
        )
        mlflow.log_params({"params_total": total, "params_trainable": trainable})
        mlflow.set_tags(
            {
                "dataset": "EuroSAT RGB",
                "dataset_dvc_md5": dataset_version(),
                "device": device.type,
                "device_name": device_name,
                "torch_version": torch.__version__,
            }
        )

        t0 = time.perf_counter()
        train_feats, train_labels = extract_features(
            model, make_loader("train", cfg["img_size"], cfg["batch_size"]), device
        )
        val_feats, val_labels = extract_features(
            model, make_loader("val", cfg["img_size"], cfg["batch_size"]), device
        )
        model.cpu()
        mlflow.log_metric("feature_extraction_s", time.perf_counter() - t0)
        print(
            f"[train] features : train {tuple(train_feats.shape)}, "
            f"val {tuple(val_feats.shape)}"
        )

        t0 = time.perf_counter()
        history, best = train_head(
            model.fc, train_feats, train_labels, val_feats, val_labels, cfg, seed
        )
        mlflow.log_metric("head_training_s", time.perf_counter() - t0)
        mlflow.log_metrics({f"best_{k}": v for k, v in best.items()})

        history.to_csv(METRICS_DIR / "train_history.csv", index=False)
        plot_history(history, METRICS_DIR / "learning_curves.png")
        mlflow.log_artifact(
            str(METRICS_DIR / "learning_curves.png"), "diagnostic_plots"
        )

        model.eval()
        torch.save(model.state_dict(), MODEL_DIR / "resnet50_linear_probe.pt")
        example = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"])
        with torch.no_grad():
            signature = infer_signature(example.numpy(), model(example).numpy())
        mlflow.pytorch.log_model(model, "model", signature=signature)

        run_info = {
            "run_id": run.info.run_id,
            "best_epoch": best["epoch"],
            "classes": classes,
        }
        with open(MODEL_DIR / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        print(f"[train] run {run.info.run_id} termine (modele logge, non enregistre)")


if __name__ == "__main__":
    main()
