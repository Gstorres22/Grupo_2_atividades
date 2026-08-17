"""
Configurações do sistema, carregadas de variáveis de ambiente e .env.

E subagentes de testes, (personas que eu simulei como teste)


"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from App import APP_DIR, CASE_DIR

CACHE_DIR = APP_DIR / "cache"


def _load_dotenv() -> None:
    """
    Coloquei isso aqui mas não é obrigatoriom, se for só o case não precisa ter isso instalado, agora se for rodar tudo tem que ter
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

    # --- V1.0.1: orquestrador por LLM ---
    orchestrator_model: str
    orchestrator_reasoning_effort: str
    """Quanto o modelo deve "pensar" antes de responder: none/low/medium/high.

    So se aplica a modelos de raciocinio (familia GPT-5, serie o). O padrao de
    FABRICA da familia 5.6 e "medium", que gasta tokens invisiveis cobrados como
    saida. Para classificacao usamos "none": medimos 23 tokens de raciocinio por
    chamada no default contra 0 com "none", sem ganho de acerto que justifique."""

    # --- Camada de testes com subagentes (personas e avaliadores) ---
    agents_model: str
    """Modelo usado pelos subagentes que TESTAM o sistema.

    Deliberadamente separado de `orchestrator_model`: quem avalia nao deve ser
    o mesmo modelo que e avaliado. LLMs tendem a preferir as proprias saidas
    (vies de auto-preferencia), entao usar o mesmo modelo nos dois papeis
    inflaria a nota. Ver ADR-16 em V1_0_1_DECISIONS.md."""

    @property
    def llm_enabled(self) -> bool:
        """Ha chave E o escalonamento esta ligado?"""
        return bool(self.openai_api_key) and self.enable_llm_escalation

    @property
    def agents_enabled(self) -> bool:
        """
        Subagentes não particinpam do pipeline principal, mas podem ser chamados em testes.
        """
        return bool(self.openai_api_key)

    def describe(self) -> str:
        if not self.openai_api_key:
            return
        if not self.enable_llm_escalation:
            return
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
        orchestrator_model=os.environ.get("OPENAI_MODEL_ORCHESTRATOR", "gpt-4o-mini"),
        orchestrator_reasoning_effort=os.environ.get("ORCHESTRATOR_REASONING_EFFORT", "none"),
        agents_model=os.environ.get("OPENAI_MODEL_AGENTS", "gpt-5.6-sol"),
    )
