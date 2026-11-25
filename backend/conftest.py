"""Rend le package `app` importable quel que soit le dossier de lancement de
pytest (racine du depot ou backend/, comme dans le TP4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
