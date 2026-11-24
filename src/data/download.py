"""
Telechargement unique du dataset EuroSAT RGB (27 000 images JPEG 64x64, 10 classes).

Ce script n'est PAS un stage DVC : comme dans le TP2, la donnee brute est
placee une fois dans data/raw/ puis prise sous controle par `dvc add data/raw`.
Source : miroir torchgeo sur HuggingFace (le meme que torchvision), controle MD5.
"""

import shutil
import tempfile
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive

URL = (
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/"
    "c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"
)
MD5 = "c8fa014336c82ac7804f0398fcb19387"
TARGET = Path("data/raw/EuroSAT_RGB")


def main():
    if TARGET.exists() and any(TARGET.iterdir()):
        print(f"[download] {TARGET} existe deja, rien a faire.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        download_and_extract_archive(URL, download_root=tmp, md5=MD5)
        # L'archive contient un dossier "2750/<Classe>/<Classe>_<n>.jpg"
        extracted = Path(tmp) / "2750"
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(TARGET))

    n_images = sum(1 for _ in TARGET.rglob("*.jpg"))
    print(f"[download] {n_images} images extraites dans {TARGET}")


if __name__ == "__main__":
    main()
