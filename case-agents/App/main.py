"""Ponto de entrada local do cerebro de roteamento.

    python -m App.main                    # roda a demonstracao
    python -m App.main "quero meu saldo"  # roda uma query especifica

Monta o pipeline completo (nucleo + camada de aplicacao) e mostra, para cada
query, o caminho percorrido no grafo, o custo e a latencia acumulados.

Este arquivo e tambem o esqueleto do futuro handler Lambda: `build_pipeline()`
faz o trabalho caro (treinar o router, indexar o catalogo, carregar embeddings)
UMA vez, e `run_query()` atende cada requisicao. Em Lambda, `build_pipeline()`
iria para o escopo do modulo, aproveitando o container reaproveitado entre
invocacoes (cold start paga o treino; warm start nao paga nada).
"""
from __future__ import annotations

import sys
from typing import Tuple

from App.core.config import get_settings
from App.core.embeddings import build_dense_provider
from App.core.escalation import LLMEscalator
from App.core.graph import build_graph, run_query
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data, load_tools

DEMO_QUERIES = [
    "Bom dia, qual o horário de atendimento?",
    "Quanto eu tenho disponível na conta agora?",
    "Perdi meu cartão na rua, preciso bloquear agora",
    "Bom dia, preciso do meu saldo",  # saudação + intenção acionável -> AGENT
    "Manda o pdf da minha fatura atual",
]


def build_pipeline() -> Tuple[object, object]:
    """Monta o pipeline inteiro. Devolve (grafo_compilado, escalator)."""
    settings = get_settings()
    print(f"[config] {settings.describe()}")

    tools = load_tools()
    texts, labels = load_router_training_data()

    router = QueryRouter(confidence_threshold=settings.router_confidence_threshold).fit(texts, labels)
    retriever = ToolRetriever()

    # Busca hibrida: o estagio vetorial so entra se houver chave de API.
    # Sem chave, `build_dense_provider` devolve None e a busca segue lexica.
    dense_provider = build_dense_provider(settings)
    retriever.set_dense_provider(dense_provider)
    retriever.fit(tools)
    print(
        f"[indice] {retriever.n_tools} tools indexadas | "
        f"modo={'hibrido (lexical + vetorial)' if dense_provider else 'lexical'}"
    )
    if dense_provider:
        print(f"[embeddings] {dense_provider.stats()}")

    escalator = LLMEscalator(settings)
    graph = build_graph(router, retriever, tools, escalator, settings)
    return graph, escalator


def main() -> None:
    graph, escalator = build_pipeline()
    queries = sys.argv[1:] or DEMO_QUERIES

    for query in queries:
        state = run_query(graph, query)
        print("\n" + "-" * 70)
        print(f"Query    : {query}")
        print(f"Rota     : {state['route']} (confiança {state.get('confidence', 0):.2f})")
        if state.get("tools"):
            print(f"Tools    : {state['tools']}")
        print(f"Resposta : {state.get('answer', '')[:100]}")
        print(f"Custo    : US$ {state.get('cost_usd', 0):.6f} | "
              f"Latência: {state.get('latency_ms', 0):.2f} ms")
        print(f"Caminho  : {' -> '.join(state.get('trace', []))}")

    if escalator.active:
        print("\n" + "=" * 70)
        print(f"Telemetria da cascata: {escalator.telemetry.as_dict()}")


if __name__ == "__main__":
    main()
