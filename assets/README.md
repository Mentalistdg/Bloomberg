# assets/

Esta carpeta debe contener el archivo fuente del pipeline:

- **`usa_fondos_pp.sqlite`** (~80 MB) — base SQLite con tablas `historico`, `fees`, `subyacentes`.

El archivo no se distribuye con el repositorio (ver `.gitignore`). Debe colocarse aquí antes de ejecutar `uv run python -m scripts.run_all`.
