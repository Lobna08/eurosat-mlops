"""
Definition des reseaux. Checkpoint A : baseline CNN ResNet50 pre-entraine ImageNet.
Le ViT-B/16 (et Swin en extension) sera ajoute ici pour le checkpoint C.
"""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(num_classes, freeze_backbone=True, pretrained=True):
    """ResNet50 ImageNet, tete fc (2048 -> 1000) remplacee par 2048 -> num_classes.

    freeze_backbone=True : linear probe, seuls les poids de `fc` sont entrainables.
    pretrained=False : poids aleatoires (tests unitaires, evite le telechargement).
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def feature_extractor(model):
    """Tout le ResNet sauf `fc` : image -> vecteur de 2048 features."""
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten())


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def extract_features(model, loader, device="cpu"):
    """Passe le backbone gele une seule fois sur un DataLoader (mode eval).
    Les features sont renvoyees sur CPU (la tete lineaire y est entrainee)."""
    extractor = feature_extractor(model).to(device).eval()
    feats, labels = [], []
    for images, targets in loader:
        feats.append(extractor(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)
