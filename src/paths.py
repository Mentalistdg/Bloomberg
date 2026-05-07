"""Rutas centralizadas del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "usa_fondos_pp.sqlite"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
APP_DATA_DIR = PROJECT_ROOT / "app" / "backend" / "data"
