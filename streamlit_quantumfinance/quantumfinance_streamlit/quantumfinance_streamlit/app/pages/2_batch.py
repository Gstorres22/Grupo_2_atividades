"""Página: Predição em lote via upload de CSV."""

from __future__ import annotations
from io import StringIO

import pandas as pd
import streamlit as st

from app.utils.api_client import APIError
from app.utils.schema import required_columns, default_payload


def render(client) -> None:
    st.subheader("Predição em lote (CSV)")
    st.caption(
        "Faça upload de um arquivo .csv com uma linha por cliente. "
        "As colunas devem corresponder às features do modelo."
    )

    with st.expander("📑 Ver colunas esperadas / baixar template"):
        cols = required_columns()
        st.write(f"Total de colunas: **{len(cols)}**")
        st.code(", ".join(cols), language="text")

        template_df = pd.DataFrame([default_payload()])
        # Type_of_Loan é multiselect -> converter para string separada por ';'
        if "Type_of_Loan" in template_df.columns:
            template_df["Type_of_Loan"] = template_df["Type_of_Loan"].apply(
                lambda v: ";".join(v) if isinstance(v, list) else v
            )
        st.download_button(
            "⬇️ Baixar template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="template_credit_score.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Selecione um CSV", type=["csv"])

    if not uploaded:
        st.info("Aguardando upload de arquivo...")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return

    st.markdown(f"**Pré-visualização** — {len(df)} linhas")
    st.dataframe(df.head(10), use_container_width=True)

    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        st.error("Colunas obrigatórias ausentes: " + ", ".join(missing))
        return

    if st.button("🚀 Executar predição em lote", type="primary"):
        _run_batch(client, df)


def _run_batch(client, df: pd.DataFrame) -> None:
    # Trata Type_of_Loan: aceita string separada por ';' e converte para lista
    if "Type_of_Loan" in df.columns:
        df["Type_of_Loan"] = df["Type_of_Loan"].apply(
            lambda v: v.split(";") if isinstance(v, str) else v
        )

    rows = df[required_columns()].to_dict(orient="records")

    with st.spinner(f"Enviando {len(rows)} registros à API..."):
        try:
            results = client.predict_batch(rows)
        except APIError as e:
            st.error(f"❌ {e.message}")
            if e.details:
                with st.expander("Detalhes técnicos"):
                    st.code(str(e.details))
            return

    if not results:
        st.warning("A API não retornou predições.")
        return

    out = df.copy()
    out["credit_score"] = [r.get("credit_score") for r in results]
    out["prob_good"] = [r.get("probabilities", {}).get("Good", 0)
                        for r in results]
    out["prob_standard"] = [r.get("probabilities", {}).get("Standard", 0)
                            for r in results]
    out["prob_poor"] = [r.get("probabilities", {}).get("Poor", 0)
                        for r in results]
    out["model_version"] = [r.get("model_version") for r in results]

    st.success(f"✅ {len(results)} predições concluídas.")

    # Distribuição das classes
    dist = out["credit_score"].value_counts(normalize=True) * 100
    cols = st.columns(3)
    for col, cls in zip(cols, ["Good", "Standard", "Poor"]):
        with col:
            st.metric(
                {"Good": "🟢 Bom", "Standard": "🟡 Padrão",
                 "Poor": "🔴 Ruim"}[cls],
                f"{dist.get(cls, 0):.1f}%",
            )

    st.dataframe(out, use_container_width=True)
    st.download_button(
        "⬇️ Baixar resultados (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predicoes_credit_score.csv",
        mime="text/csv",
    )
