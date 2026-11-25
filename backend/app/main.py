"""
API FastAPI d'inference EuroSAT (ONNX Runtime, sans PyTorch).

Le modele est charge une seule fois au demarrage (lifespan). Les chemins
viennent de variables d'environnement : aucune dependance a mlflow.db ou a
DVC au runtime, les metadonnees du Registry sont figees par le stage export.
Si le chargement echoue, l'API demarre quand meme et repond 503.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.inference import InferenceService, InvalidImageError
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
)

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_DIR / "models"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def settings():
    """Lu au demarrage (et non a l'import) pour que les tests puissent le changer."""
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_DIR / "resnet50.onnx")))
    return {
        "model_path": model_path,
        "metadata_path": Path(
            os.getenv("METADATA_PATH", str(model_path.parent / "model_metadata.json"))
        ),
        "cam_weights_path": Path(
            os.getenv("CAM_WEIGHTS_PATH", str(model_path.parent / "fc_weights.npz"))
        ),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie : chargement unique du modele au demarrage."""
    print("[INIT] Chargement du modele ONNX...")
    try:
        app.state.service = InferenceService(**settings())
        print(f"[OK] Modele charge : {app.state.service.version}")
    except Exception as exc:  # l'API reste joignable et repond 503
        app.state.service = None
        print(f"[ERREUR] Modele non charge : {exc}")
    yield
    print("[SHUTDOWN] Liberation des ressources.")
    app.state.service = None


app = FastAPI(
    title="EuroSAT Inference API",
    description="""
## Classification d'occupation des sols (EuroSAT RGB, 10 classes)
Inference ONNX Runtime sur CPU d'un ResNet50 (linear probe) suivi dans MLflow.

* **/health** : sonde de sante (modele charge, version, GPU)
* **/model/metadata** : nom, architecture, version du Registry, classes, metriques test
* **/predict** : image -> classe + probabilites des 10 classes + latence serveur
* **/predict/visualize** : image -> PNG de la carte d'activation (CAM) superposee
""",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Predicted-Class", "X-Server-Latency-Ms"],
)

ERRORS_503 = {503: {"model": ErrorResponse, "description": "Modele non charge"}}
ERRORS_IMAGE = {
    400: {"model": ErrorResponse, "description": "Fichier qui n'est pas une image"},
    413: {"model": ErrorResponse, "description": "Fichier de plus de 10 Mo"},
    **ERRORS_503,
}


def get_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "service", None)
    if service is None or not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modele n'est pas charge.",
        )
    return service


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Fichier vide.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Fichier trop volumineux (max 10 Mo).",
        )
    return data


@app.get(
    "/health",
    response_model=HealthResponse,
    responses=ERRORS_503,
    tags=["Monitoring & Sante"],
    summary="Sonde de sante du service",
)
async def health(request: Request):
    """Verifie que l'API repond et que le modele ONNX est charge (sinon 503)."""
    service = get_service(request)
    providers = service.session.get_providers()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=service.version,
        model_file=service.model_path.name,
        gpu_available="CUDAExecutionProvider" in ort.get_available_providers(),
        providers=providers,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    responses=ERRORS_503,
    tags=["Gouvernance & Metadonnees"],
    summary="Metadonnees et metriques du modele",
)
async def model_metadata(request: Request):
    """Nom, architecture, version du Registry, classes et metriques sur le test."""
    service = get_service(request)
    meta = service.metadata
    return ModelMetadataResponse(
        model_name=meta["model_name"],
        architecture=meta["architecture"],
        registry_version=meta["registry_version"],
        registry_stage=meta["registry_stage"],
        run_id=meta["run_id"],
        classes=meta["classes"],
        test_metrics=meta["test_metrics"],
        img_size=meta["img_size"],
        model_file=service.model_path.name,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses=ERRORS_IMAGE,
    tags=["Inference"],
    summary="Classification d'une image",
)
async def predict(request: Request, file: UploadFile = File(...)):
    """Recoit une image (JPEG, PNG...), renvoie la classe predite, les
    probabilites des 10 classes et la latence serveur en ms."""
    service = get_service(request)
    data = await read_upload(file)
    try:
        # Inference CPU bloquante : executee hors de la boucle asyncio
        result = await run_in_threadpool(service.predict, data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post(
    "/predict/visualize",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Image annotee : CAM de la classe predite superposee",
        },
        **ERRORS_IMAGE,
    },
    tags=["Inference"],
    summary="Carte d'activation (CAM) de la classe predite",
)
async def predict_visualize(request: Request, file: UploadFile = File(...)):
    """Renvoie un PNG : l'image avec la carte d'activation de la classe predite
    (rouge = zones qui augmentent le plus son score). Classe et latence serveur
    dans les en-tetes X-Predicted-Class et X-Server-Latency-Ms."""
    service = get_service(request)
    data = await read_upload(file)
    try:
        png, label, latency_ms = await run_in_threadpool(service.visualize, data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Predicted-Class": label,
            "X-Server-Latency-Ms": f"{latency_ms:.2f}",
        },
    )
