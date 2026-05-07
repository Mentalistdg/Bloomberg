"""Explorador rápido del archivo usa_fondos_pp.sqlite.

Corre con:
    uv run explorar_db.py
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).parent / "usa_fondos_pp.sqlite"


def main() -> None:
    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tablas = [r[0] for r in cur.fetchall()]
    print(f"Archivo: {DB}")
    print(f"Tablas encontradas: {tablas}\n")

    for t in tablas:
        cur.execute(f'PRAGMA table_info("{t}")')
        cols = [(c[1], c[2]) for c in cur.fetchall()]
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        n = cur.fetchone()[0]

        print(f"=== {t}  ({n:,} filas) ===")
        print("Columnas:")
        for nombre, tipo in cols:
            print(f"  - {nombre} ({tipo})")

        print("Primeras 5 filas:")
        cur.execute(f'SELECT * FROM "{t}" LIMIT 5')
        nombres = [d[0] for d in cur.description]
        print("  " + " | ".join(nombres))
        for row in cur.fetchall():
            print("  " + " | ".join(str(x) for x in row))
        print()

    con.close()


if __name__ == "__main__":
    main()
