"""
Stage DVC 'export' : PyTorch -> ONNX (FP32), quantification dynamique INT8,
puis mesures comparatives sous ONNX Runtime.

Le graphe ONNX a deux sorties :
  - logits   (N, 10)         : la prediction
  - features (N, 2048, 7, 7) : les feature maps de layer4, pour calculer une CAM
    en NumPy cote API (poids de fc, pas de gradients ni de PyTorch en prod).

Controles et mesures :
  1. Equivalence PyTorch vs ONNX Runtime FP32 sur des images reelles du test :
     logits a export.atol_logits pres (absolu), features a export.rtol_features
     pres (relatif) et meme classe predite, sinon le stage echoue.
  2. FP32 (modele servi par l'API) : accuracy et F1 macro sur le test complet.
  3. INT8 : accuracy et F1 macro sur un sous-ensemble stratifie du test
     (export.int8_eval_subset images, seed globale), compare au FP32 sur les
     memes images. ConvInteger coute ~1 s/image sur ce CPU 2 coeurs.
  4. Taille du fichier et latence mediane batch 1 pour les deux.
  Les chiffres sont rapportes tels quels, sans hypothese de gain.

Resultats dans metrics/export.json (metrique DVC) et dans le meme run MLflow
que train/evaluate. Le Model Registry n'est pas modifie ici.
"""

import gc
import json
import statistics
import time
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROCESSED_DIR,
    EuroSATDataset,
    build_transforms,
)
from src.models.architecture import build_resnet50

MODEL_DIR = Path("models")
TORCH_PATH = MODEL_DIR / "resnet50_linear_probe.pt"
FP32_PATH = MODEL_DIR / "resnet50.onnx"
INT8_PATH = MODEL_DIR / "resnet50_int8.onnx"
FC_WEIGHTS_PATH = MODEL_DIR / "fc_weights.npz"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_DIR = Path("metrics")


class LogitsAndFeatures(nn.Module):
    """ResNet50 qui renvoie aussi les feature maps de layer4 (entree de la CAM)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(torch.flatten(self.avgpool(features), 1))
        return logits, features


def export_fp32(wrapper, img_size, opset):
    """Export dynamo (torch.export) avec un batch dynamique, poids dans le fichier."""
    dummy = torch.randn(1, 3, img_size, img_size)
    batch = torch.export.Dim("batch", min=1, max=256)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(FP32_PATH),
        input_names=["image"],
        output_names=["logits", "features"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(str(FP32_PATH))


def make_session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@torch.no_grad()
def check_equivalence(wrapper, session, loader, atol_logits, rtol_features):
    """Ecart max PyTorch vs ONNX Runtime sur les memes tenseurs d'entree.

    Logits : tolerance absolue. Features : tolerance relative a leur valeur max
    (elles montent a ~25 apres layer4, et la CAM est normalisee dans [0, 1])."""
    max_logits, max_features, max_value, same_class, n = 0.0, 0.0, 0.0, 0, 0
    for images, _ in loader:
        t_logits, t_features = wrapper(images)
        o_logits, o_features = session.run(None, {"image": images.numpy()})
        max_logits = max(max_logits, float(np.abs(t_logits.numpy() - o_logits).max()))
        max_features = max(
            max_features, float(np.abs(t_features.numpy() - o_features).max())
        )
        max_value = max(max_value, float(t_features.abs().max()))
        same_class += int((t_logits.argmax(1).numpy() == o_logits.argmax(1)).sum())
        n += len(images)
    result = {
        "equiv_images": n,
        "equiv_max_abs_diff_logits": max_logits,
        "equiv_max_abs_diff_features": max_features,
        "equiv_max_rel_diff_features": max_features / max_value,
        "equiv_same_class_rate": same_class / n,
        "equiv_atol_logits": atol_logits,
        "equiv_rtol_features": rtol_features,
    }
    if (
        max_logits > atol_logits
        or max_features / max_value > rtol_features
        or same_class != n
    ):
        raise ValueError(f"Export ONNX non equivalent a PyTorch : {result}")
    return result


def predict_session(session, loader):
    preds, labels = [], []
    for images, targets in loader:
        logits = session.run(["logits"], {"image": images.numpy()})[0]
        preds.append(logits.argmax(1))
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def scores(prefix, y_true, y_pred):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def measure_latency_ms(session, x, warmup=5, runs=50):
    """Latence ONNX Runtime CPU, batch 1, mediane en ms (protocole de evaluate)."""
    for _ in range(warmup):
        session.run(None, {"image": x})
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"image": x})
        timings.append((time.perf_counter() - t0) * 1000)
    return statistics.median(timings)


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    cfg, exp_cfg, ml_cfg = params["train"], params["export"], params["mlflow"]
    with open("models/run_info.json", "r", encoding="utf-8") as f:
        run_info = json.load(f)
    with open(METRICS_DIR / "scores.json", "r", encoding="utf-8") as f:
        test_scores = json.load(f)
    with open(METRICS_DIR / "registry.json", "r", encoding="utf-8") as f:
        registry = json.load(f)
    classes = run_info["classes"]

    model = build_resnet50(len(classes), freeze_backbone=True, pretrained=False)
    model.load_state_dict(torch.load(TORCH_PATH, map_location="cpu", weights_only=True))
    wrapper = LogitsAndFeatures(model).eval()

    t0 = time.perf_counter()
    export_fp32(wrapper, cfg["img_size"], exp_cfg["opset"])
    export_s = time.perf_counter() - t0
    quantize_dynamic(str(FP32_PATH), str(INT8_PATH), weight_type=QuantType.QInt8)
    print(f"[export] {FP32_PATH} et {INT8_PATH} ecrits ({export_s:.1f} s)")

    np.savez(
        FC_WEIGHTS_PATH,
        weight=model.fc.weight.detach().numpy(),
        bias=model.fc.bias.detach().numpy(),
    )

    test_set = EuroSATDataset(
        PROCESSED_DIR / "test.csv", build_transforms(cfg["img_size"], train=False)
    )
    batch_size = exp_cfg["eval_batch_size"]
    # Images reparties sur tout le manifeste trie (donc sur toutes les classes)
    check_idx = np.linspace(0, len(test_set) - 1, exp_cfg["n_check"]).astype(int)
    check_loader = DataLoader(Subset(test_set, check_idx.tolist()), batch_size)

    fp32 = make_session(FP32_PATH)
    results = check_equivalence(
        wrapper,
        fp32,
        check_loader,
        exp_cfg["atol_logits"],
        exp_cfg["rtol_features"],
    )
    print(f"[export] Equivalence PyTorch / ONNX FP32 : {results}")
    # PyTorch n'est plus utile : liberer la RAM (16 Go sur cette machine)
    del wrapper, model
    gc.collect()

    x = test_set[0][0].unsqueeze(0).numpy()
    runs = exp_cfg["latency_runs"]

    # FP32 (modele servi par l'API) : test complet
    t0 = time.perf_counter()
    y_true, fp32_pred = predict_session(fp32, DataLoader(test_set, batch_size))
    results["fp32_eval_s"] = time.perf_counter() - t0
    results["fp32_eval_images"] = len(y_true)
    results.update(scores("fp32_full_test", y_true, fp32_pred))
    results["fp32_latency_ms_b1"] = measure_latency_ms(fp32, x, runs=runs)
    results["fp32_size_mb"] = FP32_PATH.stat().st_size / 1e6
    print(f"[export] fp32 : {len(y_true)} images test evaluees")
    del fp32
    gc.collect()

    # INT8 : sous-ensemble stratifie (ConvInteger ~1 s/image sur ce CPU)
    sub_idx, _ = train_test_split(
        np.arange(len(y_true)),
        train_size=exp_cfg["int8_eval_subset"],
        stratify=y_true,
        random_state=params["seed"],
    )
    sub_idx = np.sort(sub_idx)
    int8 = make_session(INT8_PATH)
    t0 = time.perf_counter()
    y_sub, int8_pred = predict_session(
        int8, DataLoader(Subset(test_set, sub_idx.tolist()), batch_size)
    )
    results["int8_eval_s"] = time.perf_counter() - t0
    results["int8_eval_images"] = len(y_sub)
    results.update(scores("int8_subset", y_sub, int8_pred))
    # FP32 sur exactement les memes images, pour une comparaison a armes egales
    results.update(scores("fp32_subset", y_sub, fp32_pred[sub_idx]))
    results["int8_fp32_agreement_subset"] = float(
        (int8_pred == fp32_pred[sub_idx]).mean()
    )
    results["int8_latency_ms_b1"] = measure_latency_ms(int8, x, runs=runs)
    results["int8_size_mb"] = INT8_PATH.stat().st_size / 1e6
    print(f"[export] int8 : {len(y_sub)} images test evaluees (sous-ensemble)")
    del int8
    gc.collect()

    results["export_s"] = export_s
    results = {k: round(float(v), 6) for k, v in results.items()}
    with open(METRICS_DIR / "export.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    metadata = {
        "model_name": registry["registered_model"],
        "architecture": "ResNet50 (ImageNet IMAGENET1K_V2), linear probe",
        "registry_version": registry["model_version"],
        "registry_stage": registry["stage"],
        "run_id": run_info["run_id"],
        "classes": classes,
        "img_size": cfg["img_size"],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "test_metrics": {
            k: test_scores[k]
            for k in (
                "test_accuracy",
                "test_f1_macro",
                "test_f1_weighted",
                "test_precision_macro",
                "test_recall_macro",
                "test_samples",
            )
        },
        "onnx_outputs": ["logits", "features"],
        "onnx_opset": exp_cfg["opset"],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metrics({f"onnx_{k}": v for k, v in results.items()})
        mlflow.set_tags(
            {"onnx_opset": exp_cfg["opset"], "onnxruntime_version": ort.__version__}
        )
        mlflow.log_artifact(str(METRICS_DIR / "export.json"), "export")
        mlflow.log_artifact(str(METADATA_PATH), "export")

    for k, v in results.items():
        print(f"[export] {k}: {v}")


if __name__ == "__main__":
    main()
