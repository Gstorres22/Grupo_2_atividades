"""Página: Status da API e informações do modelo em produção."""

from __future__ import annotations
import time

import streamlit as st

from app.utils.api_client import APIError


def render(client) -> None:
    st.subheader("Status da API e do modelo")
    st.caption("Verifique a saúde do serviço e qual versão do modelo está "
               "atualmente em produção.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🩺 Health check")
        if st.button("Verificar saúde da API", use_container_width=True):
            _check_health(client)

    with col2:
        st.markdown("#### 🤖 Modelo em produção")
        if st.button("Consultar versão do modelo", use_container_width=True):
            _check_model(client)


def _check_health(client) -> None:
    started = time.perf_counter()
    try:
        data = client.health()
    except APIError as e:
        st.error(f"API offline ou inacessível: {e.message}")
        return
    elapsed = (time.perf_counter() - started) * 1000

    status = (data.get("status") or "").lower()
    if status == "ok":
        st.success(f"API saudável ✅ — latência: {elapsed:.1f} ms")
    else:
        st.warning(f"Resposta inesperada da API. Status: {status or 'n/a'}")
    st.json(data)


def _check_model(client) -> None:
    try:
        data = client.model_info()
    except APIError as e:
        st.error(f"Não foi possível obter info do modelo: {e.message}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Nome", data.get("name", "n/a"))
    with c2:
        st.metric("Versão", data.get("version", "n/a"))
    with c3:
        st.metric("Estágio", data.get("stage", "n/a"))

    if "algorithm" in data:
        st.markdown(f"**Algoritmo:** `{data['algorithm']}`")
    if "trained_at" in data:
        st.markdown(f"**Treinado em:** `{data['trained_at']}`")

    with st.expander("Ver resposta completa"):
        st.json(data)
