"""CSS customizado e helpers visuais para o app Streamlit."""

import streamlit as st


CUSTOM_CSS = """
<style>
/* ----- Layout geral ----- */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ----- Header / branding ----- */
.qf-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 24px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0F2A26 0%, #0E1117 60%);
    border: 1px solid rgba(0, 194, 168, 0.25);
    margin-bottom: 24px;
}
.qf-logo {
    width: 48px; height: 48px;
    border-radius: 12px;
    background: linear-gradient(135deg, #00C2A8, #0EA5E9);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; font-weight: 700; color: white;
    box-shadow: 0 4px 18px rgba(0, 194, 168, 0.35);
}
.qf-title {
    font-size: 22px; font-weight: 700; margin: 0;
    color: #FAFAFA;
}
.qf-subtitle {
    font-size: 13px; color: #9AA4B2; margin: 0;
}

/* ----- Cards ----- */
.qf-card {
    background: #1A1F2E;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.qf-card h4 { margin-top: 0; }

/* ----- Result badges ----- */
.qf-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
}
.qf-badge-good     { background: rgba(34,197,94,0.15);  color: #22C55E; border: 1px solid rgba(34,197,94,0.4); }
.qf-badge-standard { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.4); }
.qf-badge-poor     { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.4); }

/* ----- Metric cards ----- */
.qf-metric {
    background: #1A1F2E;
    border-radius: 12px;
    padding: 14px 18px;
    border-left: 4px solid #00C2A8;
}
.qf-metric-label { font-size: 12px; color: #9AA4B2; text-transform: uppercase; letter-spacing: 1px;}
.qf-metric-value { font-size: 22px; font-weight: 700; color: #FAFAFA; margin-top: 4px;}

/* ----- Botão primário ----- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00C2A8 0%, #0EA5E9 100%);
    border: 0;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }

/* ----- Sidebar ----- */
section[data-testid="stSidebar"] {
    background: #0B0E14;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ----- Footer ----- */
.qf-footer {
    text-align: center; color: #6B7280; font-size: 12px;
    margin-top: 40px; padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* ----- Tabs ----- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #1A1F2E;
    border-radius: 8px 8px 0 0;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    background: #0F2A26;
    border-bottom: 2px solid #00C2A8;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="qf-header">
            <div class="qf-logo">Q</div>
            <div>
                <p class="qf-title">{title}</p>
                <p class="qf-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="qf-footer">
            QuantumFinance · Credit Score Service · POC Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_badge(label: str) -> str:
    """Retorna o HTML de um badge colorido para o resultado da predição."""
    key = label.lower()
    css_class = {
        "good": "qf-badge-good",
        "standard": "qf-badge-standard",
        "poor": "qf-badge-poor",
    }.get(key, "qf-badge-standard")
    ptbr = {"good": "Bom", "standard": "Padrão", "poor": "Ruim"}.get(key, label)
    return f'<span class="qf-badge {css_class}">{ptbr.upper()}</span>'
