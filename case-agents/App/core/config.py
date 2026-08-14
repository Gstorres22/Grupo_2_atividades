"""Configuracao central da camada de aplicacao.

Le o `App/.env` e expoe as configuracoes como um objeto tipado. Nenhum outro
modulo deve ler `os.environ` diretamente — assim existe um unico lugar para
auditar o que e configuravel e qual e o padrao de cada coisa.

PRINCIPIO DE DESIGN: degradacao graciosa.
Se nao houver `OPENAI_API_KEY`, nada quebra. `llm_enabled` vira False e todo o
fluxo roda no modo local (lexico puro), que e exatamente o modo em que os
testes e o `run_case.py` do nucleo rodam. Isso e proposital: a solucao do case
nao pode depender de rede para ser avaliada.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).resolve().parent.parent
CASE_DIR = APP_DIR.parent
CACHE_DIR = APP_DIR / "cache"


def _load_dotenv() -> None:
    """Carrega App/.env se o python-dotenv estiver instalado.

    O import e opcional de proposito: o nucleo nao depende de dotenv, e quem
    rodar so o case nao precisa te-lo instalado.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = APP_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "sim"}


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    model_small: str
    model_judge: str
    embedding_model: str
    enable_llm_escalation: bool
    router_confidence_threshold: float
    retriever_margin_threshold: float

    @property
    def llm_enabled(self) -> bool:
        """Ha chave E o escalonamento esta ligado?"""
        return bool(self.openai_api_key) and self.enable_llm_escalation

    def describe(self) -> str:
        """Resumo legivel do modo de operacao — util no topo dos notebooks."""
        if not self.openai_api_key:
            return "MODO LOCAL (sem OPENAI_API_KEY): busca lexica pura, sem escalonamento."
        if not self.enable_llm_escalation:
            return "MODO LOCAL (ENABLE_LLM_ESCALATION=false): chave presente, mas desligada."
        return (
            f"MODO CASCATA: embeddings={self.embedding_model}, "
            f"escalonamento={self.model_small}, "
            f"limiares router<{self.router_confidence_threshold} / "
            f"margem<{self.retriever_margin_threshold}"
        )


def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        model_small=os.environ.get("OPENAI_MODEL_SMALL", "gpt-4o-mini"),
        model_judge=os.environ.get("OPENAI_MODEL_JUDGE", "gpt-4o-mini"),
        embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        enable_llm_escalation=_as_bool(os.environ.get("ENABLE_LLM_ESCALATION"), default=True),
        router_confidence_threshold=_as_float(os.environ.get("ROUTER_CONFIDENCE_THRESHOLD"), 0.65),
        retriever_margin_threshold=_as_float(os.environ.get("RETRIEVER_MARGIN_THRESHOLD"), 0.05),
    )
