# EuroSAT MLOps — Vision Transformer vs CNN

Classification de l'occupation des sols (EuroSAT RGB, 27 000 images Sentinel-2, 10 classes),
Axe 4 (Vision Transformers) du projet Deep Learning Avancé & MLOps.

## Installation (Windows, convention des TPs)

Prérequis : Python 3.11, Poetry ≥ 2, Git.

```powershell
poetry config virtualenvs.in-project true --local
poetry env use py -3.11
poetry install
```

## Décisions clés

- **Python 3.11 + Poetry** (`.venv` dans le projet, `poetry.lock`) : convention des TPs, qui acceptent Python 3.10 à 3.13.
- **PyTorch CPU** (`torch 2.15.0+cpu`) sur la machine actuelle, sans GPU exploitable. Le device est détecté à l'exécution ; la machine GPU utilisera une build cu128.
- **Baseline en linear probe** : ResNet50 ImageNet gelé, seule la couche finale est entraînée (fine-tuning impossible sur 2 cœurs CPU).
- **Anti-fuite** : split stratifié 70/15/15 (seed 42) avant toute transformation, normalisation ImageNet, sélection sur la validation, test lu uniquement par `evaluate`, arrêt du pipeline si une image est présente dans deux partitions.

## Données (DVC)

```powershell
poetry run dvc pull                     # récupère data/raw depuis le remote (C:/tmp/dvc_remote_eurosat)
# ou, depuis zéro :
poetry run python -m src.data.download  # télécharge EuroSAT RGB dans data/raw/EuroSAT_RGB
```

## Pipeline reproductible

```powershell
poetry run dvc repro        # prepare -> train -> evaluate
poetry run dvc dag          # graphe du pipeline
poetry run dvc metrics show # metrics/scores.json
```

Toujours passer par `poetry run`, ou activer `.venv\Scripts\activate.bat` dans le dossier du projet. Les stages appellent
`python src/...py`, qui doit être le Python du `.venv` (où `src/` est installé en package éditable).
Sur ce CPU (2 cœurs), un `dvc repro` complet prend environ 1 h 10 (extraction des features ResNet50).

Ordre des stages : `prepare` (split stratifié, contrôle des doublons) → `train` (linear probe, run MLflow)
→ `evaluate` (métriques test sur le même run, puis enregistrement en Staging ; version écrite dans `metrics/registry.json`).

## Résultats actuels

Run MLflow `3c0522e9491d4c19a541fb702346b18f`. ResNet50 ImageNet gelé + couche linéaire, CPU, sans augmentation.
Split stratifié 70/15/15 (seed 42) : train 18 899 · val 4 051 · test 4 050 images.

| Métrique (test, 4 050 images) | Valeur |
|---|---|
| Accuracy | 0.9662 |
| F1 macro | 0.9651 |
| F1 weighted | 0.9662 |
| Précision / rappel macro | 0.9652 / 0.9650 |
| F1 par classe (min – max) | 0.9282 (PermanentCrop) – 0.9912 (Residential) |
| Latence PyTorch CPU, batch 1 (médiane) | 138.0 ms |
| Taille du modèle / paramètres | 94.4 Mo / 23.5 M (20 490 entraînables) |

Ce baseline n'est pas final : le fine-tuning et la comparaison avec ViT-B/16 sont prévus sur GPU (checkpoint C).

## API

```powershell
cd backend
..\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger : http://127.0.0.1:8000/docs

- `GET /health` : état du service, modèle chargé, version du Registry, GPU disponible (503 si le modèle n'est pas chargé).
- `GET /model/metadata` : nom, architecture, version du Registry, classes, métriques sur le test.
- `POST /predict` : image → classe prédite, probabilités des 10 classes, latence serveur en ms (400 si le fichier n'est pas une image).
- `POST /predict/visualize` : image → PNG de la carte d'activation (CAM) de la classe prédite.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@data/raw/EuroSAT_RGB/AnnualCrop/AnnualCrop_10.jpg"
```

L'API sert le modèle ONNX FP32 (F1 macro 0.9651 sur le test complet, 59.1 ms en batch 1) plutôt que l'INT8 dynamique, qui est 4 fois plus petit (23.8 Mo contre 94.3 Mo) mais plus lent sur ce CPU (1 147.2 ms) et moins précis (F1 macro 0.8444 contre 0.9674 pour le FP32 sur les mêmes 500 images du test).

## Suivi des expériences (MLflow)

```powershell
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Interface : http://127.0.0.1:5000 (expérience `EuroSAT-Classification`, modèle `EuroSAT-ResNet50`).

## Qualité

```powershell
poetry run black --check src tests
poetry run isort --check-only src tests
poetry run flake8 src tests
poetry run pytest -v
```

## Structure

```
src/data/dataset.py        split stratifié (stage prepare) + Dataset PyTorch
src/data/download.py       téléchargement unique du dataset brut
src/models/architecture.py ResNet50 (ViT/Swin à venir)
src/models/train.py        linear probe + tracking MLflow (modèle loggé)
src/models/evaluate.py     métriques sur le test isolé + enregistrement Registry (Staging)
params.yaml / dvc.yaml     hyperparamètres et DAG DVC
```
