"""Construtor dinâmico do formulário a partir do schema."""

from __future__ import annotations
from typing import Any

import streamlit as st

from app.utils.schema import Feature, get_feature_groups


def render_form(key_prefix: str = "form",
                initial: dict[str, Any] | None = None) -> dict[str, Any]:
    """Renderiza o formulário em abas/expanders e retorna o payload."""
    initial = initial or {}
    payload: dict[str, Any] = {}

    groups = get_feature_groups()
    group_names = list(groups.keys())
    tabs = st.tabs(group_names)

    for tab, gname in zip(tabs, group_names):
        with tab:
            features = groups[gname]
            # Distribui em 2 colunas para melhor uso do espaço
            cols = st.columns(2)
            for i, feat in enumerate(features):
                with cols[i % 2]:
                    payload[feat.name] = _render_field(feat, key_prefix, initial)
    return payload


def _render_field(feat: Feature, key_prefix: str,
                  initial: dict[str, Any]) -> Any:
    key = f"{key_prefix}_{feat.name}"
    default = initial.get(feat.name, feat.default)

    if feat.kind == "int":
        return st.number_input(
            feat.label, min_value=int(feat.min_value) if feat.min_value is not None else None,
            max_value=int(feat.max_value) if feat.max_value is not None else None,
            value=int(default), step=int(feat.step or 1),
            help=feat.help or None, key=key,
        )
    if feat.kind == "number":
        return st.number_input(
            feat.label, min_value=float(feat.min_value) if feat.min_value is not None else None,
            max_value=float(feat.max_value) if feat.max_value is not None else None,
            value=float(default), step=float(feat.step or 1.0),
            help=feat.help or None, key=key, format="%.2f",
        )
    if feat.kind == "select":
        options = feat.options or []
        idx = options.index(default) if default in options else 0
        return st.selectbox(feat.label, options, index=idx,
                            help=feat.help or None, key=key)
    if feat.kind == "multiselect":
        options = feat.options or []
        default_list = default if isinstance(default, list) else [default]
        default_list = [d for d in default_list if d in options]
        return st.multiselect(feat.label, options, default=default_list,
                              help=feat.help or None, key=key)
    # fallback
    return st.text_input(feat.label, value=str(default), key=key)
