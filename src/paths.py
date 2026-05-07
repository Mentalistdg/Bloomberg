"""Rutas centralizadas del proyecto.

Todas las rutas del pipeline se definen aquí para evitar paths
hardcodeados en múltiples archivos. Cualquier script o módulo de src/
importa sus rutas desde aquí.
"""

from pathlib import Path

# Raíz del proyecto (directorio que contiene src/, scripts/, artifacts/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base de datos sqlite con los datos crudos (~80 MB, no versionada en git)
DB_PATH = PROJECT_ROOT / "assets" / "usa_fondos_pp.sqlite"

# Directorio donde se guardan todos los artefactos intermedios y finales
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Subdirectorio para gráficos diagnósticos (EDA y señal)
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Directorio donde el script 05 deposita los JSONs que consume el dashboard
APP_DATA_DIR = PROJECT_ROOT / "app" / "backend" / "data"
