"""Página: Predição individual de Credit Score."""

from __future__ import annotations
from datetime import datetime

import streamlit as st

from app.components.form_builder import render_form
from app.components.charts import score_gauge, probability_bar
from app.utils.api_client import APIError
from app.utils.schema import CREDIT_SCORE_CLASSES, default_payload
from app.utils.styles import score_badge


def render(client) -> None:
    st.subheader("Predição individual")
    st.caption(
        "Preencha os dados do cliente nos grupos abaixo e clique em "
        "**Calcular score** para enviar à API de predição."
    )

    with st.container():
        # Botão para preencher com valores default (útil para demo rápida)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("📋 Preencher com valores de exemplo",
                         use_container_width=True):
                st.session_state["_seed_form"] = default_payload()
        with col_b:
            if st.button("🧹 Limpar formulário", use_container_width=True):
                st.session_state["_seed_form"] = default_payload()

    payload = render_form(
        key_prefix="single",
        initial=st.session_state.get("_seed_form", {}),
    )

    st.divider()
    submit = st.button("🚀 Calcular score", type="primary",
                       use_container_width=True)

    if submit:
        _run_prediction(client, payload)


def _run_prediction(client, payload: dict) -> None:
    with st.spinner("Consultando o modelo..."):
        try:
            result = client.predict(payload)
        except APIError as e:
            st.error(f"❌ {e.message}")
            if e.details:
                with st.expander("Ver detalhes técnicos"):
                    st.code(str(e.details))
            return

    _save_to_history(payload, result)
    _render_result(result)


def _render_result(result: dict) -> None:
    label = result.get("credit_score", "Standard")
    probs = result.get("probabilities", {})

    st.success("Predição concluída.")
    st.markdown("### Resultado")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"**Classificação:** {score_badge(label)}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Versão do modelo:** `{result.get('model_version', 'n/a')}`"
        )
        st.markdown(
            f"**Tempo de resposta:** `{result.get('latency_ms', 0)} ms`"
        )
        st.markdown(
            f"**Request ID:** `{result.get('request_id', 'n/a')}`"
        )

    with col2:
        # P(Good) como score 0-100 — proxy útil para o gauge
        good_prob = float(probs.get("Good", 0)) * 100 if probs else None
        st.plotly_chart(score_gauge(label, good_prob),
                        use_container_width=True)

    if probs:
        st.markdown("#### Distribuição de probabilidades")
        st.plotly_chart(probability_bar(probs), use_container_width=True)

    with st.expander("Ver resposta JSON completa"):
        st.json(result)


def _save_to_history(payload: dict, result: dict) -> None:
    history = st.session_state.setdefault("history", [])
    history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "credit_score": result.get("credit_score"),
        "model_version": result.get("model_version"),
        "request_id": result.get("request_id"),
        "latency_ms": result.get("latency_ms"),
        "input": payload,
        "probabilities": result.get("probabilities", {}),
    })
