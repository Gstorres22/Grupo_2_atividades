"""Página: Histórico de predições da sessão atual."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils.styles import score_badge


def render(client) -> None:
    st.subheader("Histórico da sessão")
    st.caption(
        "Lista das predições feitas nesta sessão. Os dados são apagados "
        "ao fechar o navegador."
    )

    history = st.session_state.get("history", [])
    if not history:
        st.info("Nenhuma predição realizada ainda nesta sessão.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Total de predições", len(history))
    with col2:
        latencies = [h.get("latency_ms", 0) for h in history if h.get("latency_ms")]
        avg = sum(latencies) / len(latencies) if latencies else 0
        st.metric("Latência média", f"{avg:.1f} ms")

    st.divider()

    # Tabela resumida
    rows = []
    for h in reversed(history):  # mais recente primeiro
        rows.append({
            "Quando": h["timestamp"],
            "Resultado": h.get("credit_score"),
            "Versão modelo": h.get("model_version"),
            "Latência (ms)": h.get("latency_ms"),
            "Request ID": h.get("request_id"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Exportar
    full = pd.DataFrame([{
        "timestamp": h["timestamp"],
        "credit_score": h.get("credit_score"),
        "model_version": h.get("model_version"),
        "latency_ms": h.get("latency_ms"),
        "request_id": h.get("request_id"),
        **{f"prob_{k}": v for k, v in (h.get("probabilities") or {}).items()},
        **{f"input_{k}": v for k, v in (h.get("input") or {}).items()},
    } for h in history])

    col_dl, col_clear = st.columns([1, 1])
    with col_dl:
        st.download_button(
            "⬇️ Exportar histórico (CSV)",
            data=full.to_csv(index=False).encode("utf-8"),
            file_name="historico_predicoes.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_clear:
        if st.button("🗑️ Limpar histórico", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.divider()
    st.markdown("### Detalhes da última predição")
    last = history[-1]
    st.markdown(
        f"**Resultado:** {score_badge(last.get('credit_score', 'Standard'))}",
        unsafe_allow_html=True,
    )
    with st.expander("Ver input enviado"):
        st.json(last.get("input", {}))
    with st.expander("Ver probabilidades retornadas"):
        st.json(last.get("probabilities", {}))
