"""
QuantumFinance · Credit Score · POC Streamlit
=============================================

Entry point da aplicação. Para executar localmente:

    streamlit run app.py

Estrutura:
- app/utils/        -> cliente da API, schema, estilos, autenticação
- app/components/   -> formulário e gráficos reutilizáveis
- app/pages/        -> uma página por funcionalidade
- training/         -> treinamento + MLflow Registry
- inference/        -> Predictor que carrega modelo da Production
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from app.utils.api_client import APIConfig, CreditScoreAPIClient  # noqa: E402
from app.utils.auth import (  # noqa: E402
    render_login_screen, require_session, logout,
)
from app.utils.styles import inject_css, render_header, render_footer  # noqa: E402

# Páginas (arquivos começam com dígito -> importlib)
_1_predict = importlib.import_module("app.pages.1_predict")
_2_batch   = importlib.import_module("app.pages.2_batch")
_3_history = importlib.import_module("app.pages.3_history")
_4_status  = importlib.import_module("app.pages.4_status")
_5_about   = importlib.import_module("app.pages.5_about")


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QuantumFinance · Credit Score",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ---------------------------------------------------------------------------
# Helpers de secrets
# ---------------------------------------------------------------------------

def _read_default(key: str, fallback: str = "") -> str:
    try:
        return st.secrets["api"].get(key, fallback)  # type: ignore[attr-defined]
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Sidebar — configurações pós-login
# ---------------------------------------------------------------------------

def build_authenticated_sidebar(session) -> CreditScoreAPIClient:
    st.sidebar.markdown(
        f"### 👤 {session.username}\n"
        f"<small style='color:#9AA4B2'>"
        f"{'Modo demonstração' if session.is_demo else 'Sessão autenticada'}"
        f"</small>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sair", use_container_width=True):
        logout()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Configurações da API")

    base_url = st.sidebar.text_input(
        "URL da API",
        value=st.session_state.get("api_base_url",
                                   _read_default("base_url",
                                                 "http://localhost:8000")),
        help="Endereço base do serviço.",
    )
    timeout = st.sidebar.number_input(
        "Timeout (s)", value=30, min_value=1, max_value=120, step=1,
    )

    st.sidebar.markdown("### 🔌 Modo de inferência")
    mode = st.sidebar.radio(
        "Como obter as predições?",
        options=["API HTTP", "Modelo local (MLflow)", "Demo (simulado)"],
        index=2 if session.is_demo else 0,
        help=("API HTTP usa o serviço remoto. "
              "Modelo local carrega o modelo em Production do MLflow. "
              "Demo gera predições simuladas para validar a UI."),
    )
    fallback = st.sidebar.checkbox(
        "Fallback automático para modelo local",
        value=True,
        help="Se a API falhar/timeout, usa o modelo local sem interromper.",
    )

    use_local  = (mode == "Modelo local (MLflow)")
    demo_mode  = (mode == "Demo (simulado)")

    st.sidebar.divider()
    if demo_mode:
        st.sidebar.warning("⚠️ Predições simuladas.")
    elif use_local:
        st.sidebar.success("🤖 Usando modelo local (MLflow).")
    else:
        st.sidebar.success("📡 Usando API HTTP.")

    return CreditScoreAPIClient(
        APIConfig(
            base_url=base_url,
            api_key="",
            timeout=int(timeout),
            auth_token=session.token,
        ),
        demo_mode=demo_mode,
        use_local=use_local,
        fallback_local=fallback,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    session = require_session()

    if session is None:
        # Não logado -> tela de login (header simples)
        render_header(
            title="QuantumFinance · Credit Score",
            subtitle="Faça login para acessar o serviço de score",
        )
        default_api = _read_default("base_url", "http://localhost:8000")
        render_login_screen(default_api)
        render_footer()
        return

    # Logado -> app completo
    render_header(
        title="QuantumFinance · Credit Score",
        subtitle=f"Sessão de {session.username}",
    )
    client = build_authenticated_sidebar(session)

    tabs = st.tabs([
        "🎯 Predição",
        "📦 Predição em lote",
        "🕘 Histórico",
        "📡 Status",
        "ℹ️ Sobre",
    ])
    with tabs[0]: _1_predict.render(client)
    with tabs[1]: _2_batch.render(client)
    with tabs[2]: _3_history.render(client)
    with tabs[3]: _4_status.render(client)
    with tabs[4]: _5_about.render(client)

    render_footer()


if __name__ == "__main__":
    main()
