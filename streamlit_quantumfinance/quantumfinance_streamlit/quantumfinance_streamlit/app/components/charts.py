"""Gráficos reutilizáveis: gauge de score e barras de probabilidade."""

from __future__ import annotations

import plotly.graph_objects as go

from app.utils.schema import CREDIT_SCORE_CLASSES


def score_gauge(label: str, score_value: float | None = None) -> go.Figure:
    """Gauge mostrando o nível de risco do cliente.

    Se a API retornar `probabilities`, usamos P(Good) como score numérico
    (0 a 100). Caso contrário, mapeamos a classe para uma faixa fixa.
    """
    if score_value is None:
        score_value = {"Poor": 20, "Standard": 55, "Good": 85}.get(label, 50)

    color = CREDIT_SCORE_CLASSES.get(label, {}).get("color", "#9CA3AF")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_value,
        number={"suffix": "/100", "font": {"size": 36, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9AA4B2"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],   "color": "rgba(239,68,68,0.25)"},
                {"range": [33, 66],  "color": "rgba(245,158,11,0.25)"},
                {"range": [66, 100], "color": "rgba(34,197,94,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    return fig


def probability_bar(probabilities: dict[str, float]) -> go.Figure:
    """Barras horizontais com a probabilidade de cada classe."""
    order = ["Good", "Standard", "Poor"]
    labels_pt = [CREDIT_SCORE_CLASSES[k]["label"] for k in order]
    values = [probabilities.get(k, 0) * 100 for k in order]
    colors = [CREDIT_SCORE_CLASSES[k]["color"] for k in order]

    fig = go.Figure(go.Bar(
        x=values, y=labels_pt, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#FAFAFA"),
    ))
    fig.update_layout(
        height=240, margin=dict(l=10, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], showgrid=False, color="#9AA4B2",
                   ticksuffix="%"),
        yaxis=dict(color="#FAFAFA"),
        showlegend=False,
    )
    return fig
