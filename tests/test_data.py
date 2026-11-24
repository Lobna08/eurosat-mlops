from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import build_transforms, check_no_duplicates, stratified_split
from src.models.architecture import build_resnet50, count_parameters

PROCESSED = Path("data/processed")


def fake_dataframe(n_classes=10, per_class=20):
    return pd.DataFrame(
        [
            {"path": f"{c}/{i}.jpg", "label": f"class{c}"}
            for c in range(n_classes)
            for i in range(per_class)
        ]
    )


def test_split_is_disjoint_and_stratified():
    df = fake_dataframe()
    splits = stratified_split(df, val_size=0.15, test_size=0.15, seed=42)
    paths = [set(part["path"]) for part in splits.values()]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert sum(len(p) for p in paths) == len(df)
    assert splits["test"]["label"].nunique() == 10


def test_split_is_reproducible():
    df = fake_dataframe()
    a = stratified_split(df, 0.15, 0.15, seed=42)["test"]["path"].tolist()
    b = stratified_split(df, 0.15, 0.15, seed=42)["test"]["path"].tolist()
    assert a == b


def test_cross_split_duplicates_stop_pipeline():
    check_no_duplicates({"train-val": 0, "train-test": 0, "val-test": 0})
    with pytest.raises(ValueError):
        check_no_duplicates({"train-val": 0, "train-test": 2, "val-test": 0})


def test_transforms_output_shape():
    image = Image.new("RGB", (64, 64))
    tensor = build_transforms(224, train=False)(image)
    assert tensor.shape == (3, 224, 224)


def test_linear_probe_only_trains_head():
    model = build_resnet50(10, freeze_backbone=True, pretrained=False)
    _, trainable = count_parameters(model)
    assert trainable == 2048 * 10 + 10
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 10)


@pytest.mark.skipif(not (PROCESSED / "test.csv").exists(), reason="dvc repro non lance")
def test_generated_splits_have_no_leakage():
    parts = {
        n: set(pd.read_csv(PROCESSED / f"{n}.csv")["path"])
        for n in ("train", "val", "test")
    }
    assert not (parts["train"] & parts["test"])
    assert not (parts["val"] & parts["test"])
    assert not (parts["train"] & parts["val"])
