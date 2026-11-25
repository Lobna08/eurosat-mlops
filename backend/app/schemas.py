"""
Schemas Pydantic v2 des reponses de l'API (contrats documentes dans Swagger).
Les entrees sont des fichiers (UploadFile) : leur validation (image lisible)
est faite par le service d'inference et renvoie HTTP 400.
"""

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class HealthResponse(BaseModel):
    """Etat de sante du service (sonde Docker / readiness)."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'healthy' si le modele est charge")
    model_loaded: bool = Field(..., description="Session ONNX Runtime prete")
    model_version: str = Field(
        ..., description="Modele du Registry MLflow, version et stage"
    )
    model_file: str = Field(..., description="Fichier ONNX servi (FP32 ou INT8)")
    gpu_available: bool = Field(
        ..., description="Execution provider CUDA disponible dans ONNX Runtime"
    )
    providers: List[str] = Field(..., description="Execution providers utilises")
    timestamp: str = Field(..., description="Horodatage UTC ISO 8601")


class ModelMetadataResponse(BaseModel):
    """Caracteristiques du modele servi (gouvernance)."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "model_name": "EuroSAT-ResNet50",
                "architecture": "ResNet50 (ImageNet IMAGENET1K_V2), linear probe",
                "registry_version": 2,
                "registry_stage": "Staging",
                "run_id": "3c0522e9491d4c19a541fb702346b18f",
                "classes": EXAMPLE_CLASSES,
                "test_metrics": {"test_accuracy": 0.9662, "test_f1_macro": 0.9651},
                "img_size": 224,
                "model_file": "resnet50.onnx",
            }
        },
    )

    model_name: str = Field(..., description="Nom dans le Model Registry MLflow")
    architecture: str = Field(..., description="Reseau et strategie d'entrainement")
    registry_version: int = Field(..., ge=1, description="Version du Registry")
    registry_stage: str = Field(..., description="Stage MLflow (Staging, Production)")
    run_id: str = Field(..., description="Run MLflow d'entrainement et d'evaluation")
    classes: List[str] = Field(
        ..., min_length=10, max_length=10, description="Les 10 classes EuroSAT"
    )
    test_metrics: Dict[str, float] = Field(
        ..., description="Metriques sur le test isole (4 050 images)"
    )
    img_size: int = Field(..., gt=0, description="Taille d'entree du reseau (pixels)")
    model_file: str = Field(..., description="Fichier ONNX servi")


class PredictionResponse(BaseModel):
    """Resultat de la classification d'une image."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_class": "Forest",
                "confidence": 0.9876,
                "probabilities": {c: 0.0014 for c in EXAMPLE_CLASSES}
                | {"Forest": 0.9876},
                "inference_ms": 52.3,
                "server_latency_ms": 55.1,
                "model_version": "EuroSAT-ResNet50 v2 (Staging)",
            }
        },
    )

    predicted_class: str = Field(..., description="Classe de probabilite maximale")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilite de la classe predite"
    )
    probabilities: Dict[str, float] = Field(
        ..., min_length=10, max_length=10, description="Softmax sur les 10 classes"
    )
    inference_ms: float = Field(
        ..., ge=0.0, description="Temps du seul appel ONNX Runtime (ms)"
    )
    server_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Temps serveur : decodage + pretraitement + inference (ms)",
    )
    model_version: str = Field(..., description="Modele et version du Registry")


class ErrorResponse(BaseModel):
    """Format des erreurs HTTP (400, 413, 503)."""

    detail: str = Field(..., description="Message d'erreur")
