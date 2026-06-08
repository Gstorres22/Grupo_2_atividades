"""
Autenticação do app Streamlit.

Fluxo:
  1. O usuário informa username + password na tela de login.
  2. O app envia POST {API}/login com essas credenciais.
  3. A API retorna um JWT (access_token).
  4. O token é guardado em st.session_state e enviado em todas as
     requisições seguintes via header Authorization: Bearer <token>.

Em **Modo demonstração**, qualquer combinação válida (definida em
`DEMO_USERS`) libera o app, sem chamar a API. Útil enquanto a API
real ainda não está pronta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
import streamlit as st


# ----------------------------------------------------------------------
# Usuários de demonstração (modo offline)
# ----------------------------------------------------------------------

DEMO_USERS = {
    "admin":      "admin123",
    "analista":   "analista123",
    "parceiro":   "parceiro123",
}


# ----------------------------------------------------------------------
# Tipos
# ----------------------------------------------------------------------

@dataclass
class Session:
    username: str
    token: str
    expires_at: float            # epoch seconds
    is_demo: bool = False

    def is_valid(self) -> bool:
        return bool(self.token) and time.time() < self.expires_at


# ----------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------

def login_via_api(base_url: str, username: str, password: str,
                  timeout: int = 15) -> Session:
    """Chama POST {base_url}/login e devolve uma Session."""
    url = base_url.rstrip("/") + "/login"
    try:
        resp = requests.post(
            url,
            json={"username": username, "password": password},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Não foi possível conectar à API de autenticação. "
            "Verifique a URL ou ative o modo demonstração."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Tempo de resposta esgotado ao tentar logar.")

    if resp.status_code == 401:
        raise RuntimeError("Usuário ou senha inválidos.")
    if resp.status_code == 429:
        raise RuntimeError(
            "Limite de tentativas atingido. Aguarde antes de tentar de novo."
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Erro inesperado no login ({resp.status_code}): {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError("Resposta de login inválida (JSON malformado).")

    token = data.get("access_token") or data.get("token")
    if not token:
        raise RuntimeError("A API não retornou um token de acesso.")

    # expires_in em segundos (default 1h)
    expires_in = int(data.get("expires_in") or 3600)
    return Session(
        username=username,
        token=token,
        expires_at=time.time() + expires_in,
        is_demo=False,
    )


def login_demo(username: str, password: str) -> Session:
    """Autenticação local para o modo demonstração."""
    expected = DEMO_USERS.get(username.strip().lower())
    if expected is None or expected != password:
        raise RuntimeError(
            "Usuário ou senha inválidos.\n\n"
            "Em modo demonstração você pode usar:\n"
            "- admin / admin123\n"
            "- analista / analista123\n"
            "- parceiro / parceiro123"
        )
    return Session(
        username=username,
        token="demo-token-" + str(int(time.time())),
        expires_at=time.time() + 3600,
        is_demo=True,
    )


def logout() -> None:
    st.session_state.pop("session", None)
    # Limpa também o histórico para não vazar entre logins
    st.session_state.pop("history", None)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

def render_login_screen(default_api_url: str) -> None:
    """Renderiza a tela de login. Não retorna nada; ao logar com sucesso
    grava a sessão em st.session_state['session'] e dispara rerun."""

    # Espaçamento superior para centralizar
    st.markdown("<div style='height: 60px'></div>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            """
            <div class="qf-card" style="text-align:center; padding-top:32px;">
              <div class="qf-logo" style="margin: 0 auto 16px auto;
                   width:64px; height:64px; font-size:32px;">Q</div>
              <h2 style="margin: 8px 0 0 0;">QuantumFinance</h2>
              <p style="color:#9AA4B2; margin: 4px 0 24px 0;">
                 Acesso ao serviço de Credit Score
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            api_url = st.text_input(
                "URL da API",
                value=default_api_url,
                help="Endereço da API de Credit Score.",
            )
            username = st.text_input("Usuário", placeholder="ex.: admin")
            password = st.text_input("Senha", type="password")

            demo_mode = st.toggle(
                "Modo demonstração (sem API real)",
                value=False,
                help=("Use admin / admin123, analista / analista123, "
                      "ou parceiro / parceiro123."),
            )

            submitted = st.form_submit_button(
                "🔐 Entrar", type="primary", use_container_width=True,
            )

            if submitted:
                _handle_submit(api_url, username, password, demo_mode)

        with st.expander("ℹ️ Credenciais de teste (modo demonstração)"):
            st.markdown(
                "- **admin / admin123**\n"
                "- **analista / analista123**\n"
                "- **parceiro / parceiro123**"
            )


def _handle_submit(api_url: str, username: str, password: str,
                   demo_mode: bool) -> None:
    if not username or not password:
        st.error("Preencha usuário e senha.")
        return
    try:
        if demo_mode:
            session = login_demo(username, password)
        else:
            session = login_via_api(api_url, username, password)
    except RuntimeError as e:
        st.error(str(e))
        return

    st.session_state["session"] = session
    st.session_state["api_base_url"] = api_url
    st.session_state["demo_mode_login"] = demo_mode
    st.success(f"Bem-vindo(a), {session.username}!")
    time.sleep(0.4)
    st.rerun()


def require_session() -> Session | None:
    """Retorna a sessão atual se válida, senão None."""
    s: Session | None = st.session_state.get("session")
    if s is None:
        return None
    if not s.is_valid():
        # Token expirou
        logout()
        return None
    return s
