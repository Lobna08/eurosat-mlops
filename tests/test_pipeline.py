from pathlib import Path

import yaml


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_params_validity():
    params = load_params()
    assert {"seed", "prepare", "train", "mlflow"} <= set(params)
    prep = params["prepare"]
    assert 0 < prep["val_size"] < 1 and 0 < prep["test_size"] < 1
    assert prep["val_size"] + prep["test_size"] < 1
    assert params["train"]["epochs"] >= 1


def test_mlflow_uses_sql_backend():
    # Le Model Registry exige un backend SQL (convention TP3).
    assert load_params()["mlflow"]["tracking_uri"].startswith("sqlite:///")


def test_dvc_files_exist():
    assert Path("dvc.yaml").exists()
    assert Path("data/raw.dvc").exists(), "data/raw doit etre suivi par DVC"
