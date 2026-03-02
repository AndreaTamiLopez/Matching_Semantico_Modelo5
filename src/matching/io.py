from __future__ import annotations

import os
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    """
    Lee CSV o Excel según extensión.
    - CSV: utf-8 o utf-8-sig
    - Excel: .xlsx
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv"]:
        # intenta utf-8-sig primero (común cuando viene de Excel)
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, encoding="utf-8")
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .csv o .xlsx")


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
