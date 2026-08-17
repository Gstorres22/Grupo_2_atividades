"""V1 — Machine Learning classico (a versao ja publicada no GitHub).

===============================================================================
O QUE ESTA VERSAO FAZ, EM UMA FRASE
===============================================================================

Decide tudo LOCALMENTE, sem chamar IA nenhuma para tomar a decisao: um
classificador treinado escolhe a rota, e uma busca por similaridade de texto
escolhe as ferramentas.

Este arquivo NAO reimplementa nada. Ele apenas EMBRULHA (`wrapper`) as classes
que ja estao em `candidate_starter/` no contrato comum `BasePipeline`, para que
o comparador consiga rodar V1 e V1.0.1 pelo mesmo caminho de codigo.

Por que embrulhar em vez de alterar o original: `candidate_starter/` e o
entregavel do case, ja publicado. Mexer nele para acomodar a comparacao
misturaria duas preocupacoes diferentes e quebraria a promessa de que aquele
codigo roda sem dependencia nenhuma alem do `requirements.txt` original.
"""
from __future__ import annotations

import time
from typing import Optional

from App.core.config import Settings, get_settings
from App.core.embeddings import build_dense_provider
from App.versions.base import BasePipeline, PipelineResult
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data, load_tools
from common.mock_llm import COST_AGENT_LLM_CALL_USD, COST_RETRIEVAL_USD, COST_ROUTER_USD


class V1ClassicPipeline(BasePipeline):
    """Roteador de ML classico + busca hibrida local."""

    name = "V1"
    description = "ML classico: TF-IDF + Regressao Logistica no roteador, busca lexical/vetorial nas ferramentas"

    def __init__(self, settings: Optional[Settings] = None, use_embeddings: bool = True):
        """
        Args:
            settings: configuracao ja carregada. Se None, le do `.env`.
                Recebemos por parametro (em vez de sempre chamar `get_settings()`)
                para o comparador poder rodar as duas versoes com EXATAMENTE a
                mesma configuracao, sem risco de uma reler o arquivo no meio.
            use_embeddings: liga o estagio vetorial da busca. Deixamos
                configuravel porque a V1.0.1 precisa usar o MESMO estagio de
                recuperacao de candidatas — se uma versao tivesse embeddings e a
                outra nao, a comparacao mediria o embedding, nao o orquestrador.
        """
        self.settings = settings or get_settings()
        self.use_embeddings = use_embeddings
        self._router: Optional[QueryRouter] = None
        self._retriever: Optional[ToolRetriever] = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> "V1ClassicPipeline":
        """Trabalho caro, feito uma unica vez."""
        # 1) Carrega os 53 exemplos rotulados e treina o classificador.
        #    O treino leva ~1 segundo e acontece so aqui.
        texts, labels = load_router_training_data()
        self._router = QueryRouter(
            confidence_threshold=self.settings.router_confidence_threshold
        ).fit(texts, labels)

        # 2) Indexa as 285 ferramentas.
        self._retriever = ToolRetriever()
        if self.use_embeddings:
            # `build_dense_provider` devolve None se nao houver chave de API.
            # O `set_dense_provider` aceita None sem reclamar, entao esta linha
            # funciona igual com ou sem credencial — so muda a qualidade.
            self._retriever.set_dense_provider(build_dense_provider(self.settings))
        self._retriever.fit(load_tools())
        return self

    # ---------------------------------------------------------------- process
    def process(self, query: str, k: int = 2) -> PipelineResult:
        """Processa uma mensagem. Duas etapas, ambas locais."""
        if self._router is None or self._retriever is None:
            raise RuntimeError("Chame setup() antes de process().")

        resultado = PipelineResult(query=query, route="FAST_PATH")
        inicio = time.perf_counter()

        try:
            # ETAPA 1 — Classificar a rota.
            # `predict` devolve rota + latencia propria + confianca (a
            # probabilidade da classe vencedora na regressao logistica).
            decisao = self._router.predict(query)
            resultado.route = decisao.route
            resultado.confidence = decisao.confidence
            resultado.cost_usd += COST_ROUTER_USD
            resultado.trace.append(f"router_local -> {decisao.route} (conf={decisao.confidence:.2f})")

            # ETAPA 2 — Se precisa de ferramenta, buscar.
            # No FAST_PATH nao ha etapa 2: e exatamente dai que vem a economia.
            if decisao.route == "AGENT":
                busca = self._retriever.search(query, k=k)
                resultado.tools = [m.name for m in busca.matches]
                resultado.cost_usd += COST_RETRIEVAL_USD + COST_AGENT_LLM_CALL_USD
                resultado.trace.append(f"busca_local -> {resultado.tools}")

        except Exception as erro:
            # Nunca deixamos a excecao subir: uma mensagem problematica vira uma
            # linha com erro, e o lote de 300 continua rodando.
            resultado.error = f"{type(erro).__name__}: {erro}"
            resultado.trace.append(f"ERRO: {resultado.error}")

        # A latencia e medida por fora das duas etapas, cobrindo o caminho
        # inteiro — que e o que o cliente sente.
        resultado.latency_ms = (time.perf_counter() - inicio) * 1000
        return resultado
