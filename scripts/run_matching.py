from __future__ import annotations

from matching import (
    MatchingConfig,
    read_table,
    write_csv,
    match_proyecto_to_politicas_embeddings_only,
)


def main():
    cfg = MatchingConfig()

    df_politicas = read_table(cfg.politicas_path)
    df_proyectos = read_table(cfg.proyectos_path)

    df_out = match_proyecto_to_politicas_embeddings_only(
        df_politicas=df_politicas,
        df_proyectos=df_proyectos,
        col_text_politica=cfg.col_text_politica,
        col_text_proyecto=cfg.col_text_proyecto,
        col_id_proyecto=cfg.col_id_proyecto,
        top_k=cfg.top_k,
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )

    write_csv(df_out, cfg.output_csv)

    print("CSV guardado en:", cfg.output_csv)
    print("Filas:", len(df_out), "| Columnas:", df_out.shape[1])


if __name__ == "__main__":
    main()
