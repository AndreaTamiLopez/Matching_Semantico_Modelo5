from __future__ import annotations

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


def match_proyecto_to_politicas_embeddings_only(
    df_politicas: pd.DataFrame,
    df_proyectos: pd.DataFrame,
    col_text_politica: str,
    col_text_proyecto: str,
    col_id_proyecto: str,
    top_k: int = 10,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Retorna Top-K políticas por proyecto usando SOLO embeddings (sin LLM).

    Salida: DataFrame en formato long con hasta (n_proyectos * top_k) filas.
    Columnas principales:
      - columnas originales de df_proyectos
      - matched_politica_text
      - similarity_score
      - rank
    """

    # Validaciones mínimas
    if col_text_politica not in df_politicas.columns:
        raise ValueError(f"df_politicas no tiene la columna: {col_text_politica}")
    for c in [col_text_proyecto, col_id_proyecto]:
        if c not in df_proyectos.columns:
            raise ValueError(f"df_proyectos no tiene la columna: {c}")

    # Modelo
    model = SentenceTransformer(model_name, device=device)

    # Textos
    pol_texts = df_politicas[col_text_politica].fillna("").astype(str).tolist()
    proy_texts = df_proyectos[col_text_proyecto].fillna("").astype(str).tolist()

    # Embeddings normalizados (coseno = dot)
    E_pol = model.encode(
        pol_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    E_proy = model.encode(
        proy_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Nearest Neighbors (PROYECTO -> POLÍTICAS)
    top_k = int(top_k)
    top_k = min(top_k, len(df_politicas))  # por seguridad
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine", algorithm="brute")
    nn.fit(E_pol)

    distances, indices = nn.kneighbors(E_proy, return_distance=True)
    scores = 1.0 - distances  # coseno

    # Salida
    out_rows = []
    for i in range(len(df_proyectos)):
        base = df_proyectos.iloc[i].to_dict()

        for rank in range(top_k):
            j = int(indices[i, rank])
            sc = float(scores[i, rank])

            out_rows.append(
                {
                    **base,
                    "matched_politica_text": df_politicas.iloc[j][col_text_politica],
                    "similarity_score": sc,
                    "rank": rank + 1,
                }
            )

    df_out = pd.DataFrame(out_rows)
    df_out = df_out.sort_values([col_id_proyecto, "rank"], ascending=[True, True]).reset_index(drop=True)
    return df_out
