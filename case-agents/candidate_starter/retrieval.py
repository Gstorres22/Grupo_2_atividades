"""Pilar 2 — Seleção de Tools Relevantes.

O catálogo de tools está em `data/tools_registry.json`. Passar todas as tools no prompt
de um LLM não escala (estoura contexto, confunde o modelo, aumenta custo e latência).

`search(query, k=2)` deve retornar as `k` tools mais relevantes do catálogo para a query,
antes de qualquer chamada ao LLM. A estratégia de seleção/ranking é livre — escolha o que
fizer sentido e esteja preparado para justificar os trade-offs.
"""
import time
from typing import List

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool


class ToolRetriever(BaseToolRetriever):
    def __init__(self) -> None:
        self._tools: List[Tool] = []
        self._fitted = False
        # TODO: inicialize aqui suas estruturas de índice

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo de tools para busca."""
        self._tools = tools
        # TODO: construa seu(s) índice(s) a partir de `tools`
        # (ex: use tool.name + tool.description + tool.category como texto do documento)
        # marque self._fitted = True ao final
        raise NotImplementedError("Implemente a indexação do ToolRetriever.")

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para `query`."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()
        # TODO: calcule os scores de relevância e retorne o top-k, ex:
        #   latency_ms = (time.perf_counter() - start) * 1000
        #   return RetrievalResult(matches=[...], latency_ms=latency_ms)
        raise NotImplementedError("Implemente a busca de tools.")
