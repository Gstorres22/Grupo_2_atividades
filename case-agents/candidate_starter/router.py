"""Pilar 1 — Router de Queries.

Implemente um componente que decide, para cada `query` do usuário, se ela deve ir para:
  - "FAST_PATH": queries simples (saudação, FAQ) -> resposta local, sem LLM.
  - "AGENT": queries que precisam de uma tool + raciocínio -> vai para o Pilar 2.

A técnica é livre (ex: modelo clássico de ML, embeddings + classificador, regras, etc.)
— escolha o que fizer sentido e esteja preparado para justificar.
"""
import time
from typing import List

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self) -> None:
        self._fitted = False
        # TODO: instancie aqui o seu modelo/estratégia de classificação

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Treina o router com os exemplos de `data/router_training_data.json`."""
        # TODO: treine seu modelo com (texts, labels) e marque self._fitted = True
        raise NotImplementedError("Implemente o treinamento do QueryRouter.")

    def predict(self, query: str) -> RouteResult:
        """Classifica `query` e retorna a rota escolhida + latência medida (em ms)."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()
        # TODO: substitua pela predição real do seu modelo.
        # Dica: meça a latência com time.perf_counter() e monte um RouteResult, ex:
        #   latency_ms = (time.perf_counter() - start) * 1000
        #   return RouteResult(route=route, latency_ms=latency_ms, confidence=confidence)
        raise NotImplementedError("Implemente a predição do QueryRouter.")
