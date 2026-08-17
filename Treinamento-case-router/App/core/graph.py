"""Orquestracao do cerebro de roteamento como grafo de estado (LangGraph).

===============================================================================
Aplicação do LangGra´ph
===============================================================================

A ideia do fluxo é uma sequencia de DECISOES de custo. O LangGraph modela exatamente isso: 
nos que executam e arestas condicionais que escolhem o proximo no. Ganhos concretos:

  * As decisoes de custo viram estrutura explicita e auditavel, em vez de
    `if/else` espalhados. Da para desenhar o grafo e discutir a arquitetura.
  * O estado carrega a TELEMETRIA (custo e latencia acumulados por no), que e
    justamente o que o Pilar 3 precisa medir.
  * Extensibilidade real: adicionar human-in-the-loop, retry, ou um segundo
    especialista e adicionar um no — nao e reescrever o fluxo.

E por que o nucleo (`candidate_starter/`) NAO usa LangGraph: o entregavel do
case precisa rodar com `pip install -r requirements.txt` (numpy, pandas,
scikit-learn, pytest) e sem rede. O grafo aqui CONSOME as mesmas classes
`QueryRouter` e `ToolRetriever` — nao reimplementa nada. A dependencia aponta
sempre de App/ para o nucleo, nunca o contrario.

===============================================================================
O FLUXO
===============================================================================

    query
      |
      v
   [route] --------- confianca baixa? ---------> [escalate_route]
      |                                                 |
      |<------------------------------------------------|
      |
      +-- FAST_PATH --> [fast_path_answer] --------------+
      |                                                  |
      +-- AGENT -----> [select_tools]                    |
                            |                            |
                    margem pequena? --> [rerank_llm]      |
                            |                |            |
                            |<---------------+            |
                            v                             |
                      [execute_tool] ---------------------+
                                                          |
                                                          v
                                                        fim
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from typing_extensions import TypedDict

from App.core.config import Settings
from App.core.escalation import LLMEscalator
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.mock_llm import (
    COST_AGENT_LLM_CALL_USD,
    COST_RETRIEVAL_USD,
    COST_ROUTER_USD,
    fast_path_answer,
    mock_tool_execution,
)
from common.schemas import Tool


class PipelineState(TypedDict, total=False):
    """Estado que atravessa o grafo.

    `total=False` porque cada no preenche apenas a sua parte — nos de FAST_PATH
    nunca definem `tools`, por exemplo.
    """

    query: str
    route: str
    confidence: Optional[float]
    route_escalated: bool
    tools: List[str]
    tool_scores: List[float]
    rerank_escalated: bool
    executed_tool: Optional[str]
    answer: str
    cost_usd: float
    latency_ms: float
    trace: List[str]


def build_graph(
    router: QueryRouter,
    retriever: ToolRetriever,
    catalog: Sequence[Tool],
    escalator: LLMEscalator,
    settings: Settings,
    k: int = 2,
):
    """Compila o grafo. Os componentes entram por injecao de dependencia.

    Passar router/retriever prontos (em vez de construi-los aqui dentro) mantem
    o grafo testavel: da para injetar dublês em teste e trocar a implementacao
    sem tocar na orquestracao.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "LangGraph nao esta instalado. Rode: pip install -r App/requirements-app.txt\n"
            "(O nucleo do case em candidate_starter/ nao precisa dele.)"
        ) from error

    # -- Nos -----------------------------------------------------------------
    def node_route(state: PipelineState) -> Dict[str, Any]:
        """Classificacao local. Sempre roda, sempre barata."""
        result = router.predict(state["query"])
        return {
            "route": result.route,
            "confidence": result.confidence,
            "cost_usd": state.get("cost_usd", 0.0) + COST_ROUTER_USD,
            "latency_ms": state.get("latency_ms", 0.0) + result.latency_ms,
            "route_escalated": False,
            "trace": state.get("trace", []) + [f"route={result.route} (conf={result.confidence:.2f})"],
        }

    def node_escalate_route(state: PipelineState) -> Dict[str, Any]:
        """Desempate por LLM. So e alcancado pela aresta condicional."""
        from common.schemas import RouteResult

        start = time.perf_counter()
        local = RouteResult(state["route"], 0.0, state.get("confidence"))
        resolved = escalator.resolve_route(state["query"], local, is_uncertain=True)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "route": resolved.route,
            "route_escalated": True,
            "latency_ms": state.get("latency_ms", 0.0) + elapsed,
            "trace": state.get("trace", []) + [f"escalate_route -> {resolved.route}"],
        }

    def node_fast_path(state: PipelineState) -> Dict[str, Any]:
        """Resposta de prateleira: custo de LLM zero."""
        return {
            "answer": fast_path_answer(state["query"]),
            "trace": state.get("trace", []) + ["fast_path (custo de LLM = 0)"],
        }

    def node_select_tools(state: PipelineState) -> Dict[str, Any]:
        """Busca hibrida (lexical + vetorial). Guarda a lista LONGA de candidatas.

        Guardamos N candidatas (e nao so k) porque o rerank opcional precisa de
        material para reordenar. O corte em k acontece depois.
        """
        result = retriever.search_candidates(state["query"])
        return {
            "tools": [m.name for m in result.matches],
            "tool_scores": [m.score for m in result.matches],
            "cost_usd": state.get("cost_usd", 0.0) + COST_RETRIEVAL_USD,
            "latency_ms": state.get("latency_ms", 0.0) + result.latency_ms,
            "rerank_escalated": False,
            "trace": state.get("trace", []) + [f"select_tools -> {result.matches[0].name} (+{len(result.matches)-1})"],
        }

    def node_rerank(state: PipelineState) -> Dict[str, Any]:
        """Rerank por LLM sobre as N candidatas — nunca sobre as 285 tools."""
        from common.schemas import RetrievalResult, ToolMatch

        start = time.perf_counter()
        candidates = RetrievalResult(
            matches=[ToolMatch(n, s) for n, s in zip(state["tools"], state["tool_scores"])],
            latency_ms=0.0,
        )
        reranked = escalator.rerank_tools(
            state["query"], candidates, catalog, k=k, is_ambiguous=True
        )
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "tools": [m.name for m in reranked.matches],
            "tool_scores": [m.score for m in reranked.matches],
            "rerank_escalated": True,
            "latency_ms": state.get("latency_ms", 0.0) + elapsed,
            "trace": state.get("trace", []) + [f"rerank_llm -> {reranked.matches[0].name}"],
        }

    def node_execute(state: PipelineState) -> Dict[str, Any]:
        """Executa a tool escolhida + 1 chamada de LLM com contexto reduzido."""
        selected = state["tools"][:k]
        execution = mock_tool_execution(selected[0], state["query"])
        return {
            "tools": selected,
            "executed_tool": selected[0],
            "answer": execution["result"],
            "cost_usd": state.get("cost_usd", 0.0) + COST_AGENT_LLM_CALL_USD,
            "trace": state.get("trace", []) + [f"execute_tool={selected[0]}"],
        }

    # -- Arestas condicionais (onde moram as decisoes de custo) ---------------
    def decide_after_route(state: PipelineState) -> str:
        confidence = state.get("confidence")
        uncertain = (
            escalator.active
            and confidence is not None
            and confidence < settings.router_confidence_threshold
        )
        if uncertain:
            return "escalate"
        return "fast" if state["route"] == "FAST_PATH" else "agent"

    def decide_after_escalation(state: PipelineState) -> str:
        return "fast" if state["route"] == "FAST_PATH" else "agent"

    def decide_after_select(state: PipelineState) -> str:
        """Margem pequena entre 1o e 2o = ambiguidade real = vale o LLM."""
        scores = state.get("tool_scores") or []
        if not escalator.active or len(scores) < 2 or scores[0] <= 0:
            return "execute"
        margin = (scores[0] - scores[1]) / scores[0]
        return "rerank" if margin < settings.retriever_margin_threshold else "execute"

    # -- Montagem -------------------------------------------------------------
    builder = StateGraph(PipelineState)
    builder.add_node("route", node_route)
    builder.add_node("escalate_route", node_escalate_route)
    builder.add_node("fast_path", node_fast_path)
    builder.add_node("select_tools", node_select_tools)
    builder.add_node("rerank", node_rerank)
    builder.add_node("execute_tool", node_execute)

    builder.set_entry_point("route")
    builder.add_conditional_edges(
        "route",
        decide_after_route,
        {"escalate": "escalate_route", "fast": "fast_path", "agent": "select_tools"},
    )
    builder.add_conditional_edges(
        "escalate_route",
        decide_after_escalation,
        {"fast": "fast_path", "agent": "select_tools"},
    )
    builder.add_conditional_edges(
        "select_tools",
        decide_after_select,
        {"rerank": "rerank", "execute": "execute_tool"},
    )
    builder.add_edge("rerank", "execute_tool")
    builder.add_edge("fast_path", END)
    builder.add_edge("execute_tool", END)

    return builder.compile()


def run_query(graph, query: str) -> PipelineState:
    """Atalho para rodar uma query e receber o estado final."""
    return graph.invoke({"query": query, "cost_usd": 0.0, "latency_ms": 0.0, "trace": []})
