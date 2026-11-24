"""
Donnees EuroSAT RGB : split reproductible (stage DVC 'prepare') + Dataset PyTorch.

Stage 'prepare' (python src/data/dataset.py) :
  1. Liste les 27 000 images de data/raw/EuroSAT_RGB/<Classe>/*.jpg
  2. Split stratifie train / val / test avec une seed fixe
  3. Controle des doublons (hash MD5 du contenu) entre partitions : erreur si > 0
  4. Tire un sous-echantillon equilibre dans chaque partition (si demande)
  5. Ecrit des manifestes CSV (chemin, label) dans data/processed/

Anti-fuite : le split est fait AVANT toute transformation, la normalisation
utilise les statistiques ImageNet du pre-entrainement (rien n'est calcule sur
nos donnees), et l'augmentation n'est appliquee qu'a la partition train.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

PROCESSED_DIR = Path("data/processed")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_images(raw_dir):
    """Retourne un DataFrame (path, label) trie, donc deterministe."""
    rows = [
        {"path": p.as_posix(), "label": p.parent.name}
        for p in sorted(Path(raw_dir).glob("*/*.jpg"))
    ]
    return pd.DataFrame(rows)


def stratified_split(df, val_size, test_size, seed):
    """Split stratifie en deux temps : (train+val) / test, puis train / val."""
    trainval, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval, test_size=val_ratio, stratify=trainval["label"], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def count_cross_split_duplicates(splits):
    """Nombre d'images au contenu identique presentes dans deux partitions."""
    hashes = {name: set(part["path"].map(file_md5)) for name, part in splits.items()}
    names = list(hashes)
    return {
        f"{a}-{b}": len(hashes[a] & hashes[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def check_no_duplicates(duplicates):
    """Controle des doublons : arrete le pipeline si une image est partagee."""
    if any(duplicates.values()):
        raise ValueError(
            "Fuite de donnees : images identiques dans plusieurs partitions "
            f"{duplicates}. Corriger le dataset avant de continuer."
        )


def subsample(part, n_per_class, seed):
    if n_per_class is None:
        return part
    return part.groupby("label", group_keys=False).sample(
        n=n_per_class, random_state=seed
    )


def prepare():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    seed = params["seed"]
    cfg = params["prepare"]

    df = list_images(cfg["raw_dir"])
    classes = sorted(df["label"].unique())
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)})
    print(f"[prepare] {len(df)} images, {len(classes)} classes")

    splits = stratified_split(df, cfg["val_size"], cfg["test_size"], seed)
    duplicates = count_cross_split_duplicates(splits)
    print(f"[prepare] Doublons inter-partitions (MD5) : {duplicates}")
    check_no_duplicates(duplicates)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = {"classes": classes, "duplicates": duplicates, "sizes": {}}
    for name, part in splits.items():
        sub = subsample(part, cfg["subset_per_class"][name], seed)
        sub.sort_values("path").to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        report["sizes"][name] = {"full": len(part), "used": len(sub)}
        print(f"[prepare] {name}: {len(sub)} images utilisees / {len(part)}")

    with open(PROCESSED_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def build_transforms(img_size, train):
    """Pretraitement compatible ImageNet. Augmentation geometrique en train seulement
    (une image satellite n'a pas d'orientation canonique)."""
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class EuroSATDataset(Dataset):
    """Lit un manifeste CSV produit par le stage 'prepare'."""

    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label_idx"])


if __name__ == "__main__":
    prepare()
