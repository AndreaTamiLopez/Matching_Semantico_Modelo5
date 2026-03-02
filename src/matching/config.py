from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchingConfig:
    # Inputs
    politicas_path: str = "data/raw/politicas.xlsx"
    proyectos_path: str = "data/raw/proyectos.xlsx"

    # Columnas
    col_text_politica: str = "Indicador de Producto(MGA)"
    col_text_proyecto: str = "Indicadores de producto PATR"
    col_id_proyecto: str = "codigo_proyecto"

    # Modelo
    top_k: int = 10
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    device: str = "cpu"

    # Output
    output_csv: str = "data/outputs/matching_top10_embeddings.csv"
