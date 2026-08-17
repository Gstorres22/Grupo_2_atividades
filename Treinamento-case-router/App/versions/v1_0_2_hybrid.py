"""V1.0.2 — Cascata hibrida: o local resolve o facil, o LLM resolve o resto.

===============================================================================
A IDEIA EM UMA FRASE
===============================================================================

Nem toda mensagem precisa de um LLM. "Bom dia" nao precisa. "Qual o horario de
atendimento?" nao precisa. Essas o classificador local resolve em 1 milissegundo
e custo zero. O LLM entra so onde o local nao tem base para opinar.

    V1     : local decide TUDO          -> 3 ms, mas erra 45% das ferramentas
    V1.0.1 : LLM decide TUDO            -> acerta 89%, mas custa 1,1 s em toda mensagem
    V1.0.2 : local decide o QUE SABE    -> tenta ficar com o acerto do LLM
             LLM decide o resto            e a latencia do local onde da

===============================================================================
O CRITERIO DE DESVIO: DOIS SINAIS QUE PRECISAM CONCORDAR
===============================================================================

Uma mensagem so pula o LLM se passar em DOIS testes ao mesmo tempo:

  1. CONFIANCA >= limiar
     O quanto o classificador acredita na propria resposta.

  2. FAMILIARIDADE >= limiar
     O quanto a mensagem se parece com algum exemplo do treino.

**Por que dois e nao so a confianca.** Medimos a precisao da confianca nas 150
mensagens de persona e ela NAO e monotonica:

    confianca [0,60-0,70) -> 92% de precisao
    confianca [0,70-0,80) -> 71%   <- confianca MAIOR, precisao MENOR
    confianca [0,80-0,90) -> 100%

Um modelo treinado com 53 exemplos nao produz probabilidade bem calibrada. Ele
consegue soar confiante sobre uma mensagem de um tipo que nunca viu — foi
exatamente assim que a V1 mandou "solicito a emissao do informe de rendimentos"
para o FAST_PATH com confianca 0,754.

A familiaridade cobre esse buraco. E uma deteccao simples de "fora da
distribuicao": se a mensagem nao se parece com nada do treino, o modelo nao tem
base para opinar, por mais confiante que soe. E o antidoto direto para o modo de
falha que derrubou a V1 com registro formal.

===============================================================================
POR QUE A CASCATA SO DESVIA FAST_PATH
===============================================================================

Poderiamos tambem desviar mensagens AGENT quando a busca local parecesse segura.
Nao fazemos, e o motivo e medido:

  * O problema da V1 nao esta principalmente na ROTA (84% de acerto nas
    personas), esta na ESCOLHA DA FERRAMENTA (54,9%).
  * Desviar um AGENT significa aceitar a escolha de ferramenta da V1 — ou seja,
    herdar justamente o que a V1.0.1 veio consertar.
  * O sinal de "margem" do buscador, que seria o candidato natural, discrimina
    mal: 68% das falhas ficam abaixo do limiar, mas 54% dos acertos tambem.

Ja o FAST_PATH nao tem escolha de ferramenta nenhuma. A resposta e uma string
pronta. Desviar ali nao arrisca qualidade de ferramenta — so a rota.

===============================================================================
DE ONDE VEM O GANHO, E DE ONDE NAO VEM
===============================================================================

A economia da cascata e proporcional a FRACAO DE FAST_PATH no trafego:

    trafego com 10% de FAST_PATH -> economia pequena
    trafego com 50% de FAST_PATH -> metade das mensagens nao chama LLM

Nosso conjunto de personas e pesado em AGENT (113 de 150 tem ferramenta
esperada), entao ele SUBESTIMA o ganho que a cascata teria num atendimento real,
onde saudacao e FAQ costumam ser fatia grande. O relatorio reporta a taxa de
desvio observada junto da composicao do trafego, para o numero poder ser lido
corretamente.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from App.core.config import Settings, get_settings
from App.versions.base import BasePipeline, PipelineResult
from App.versions.v1_0_1_orchestrator import V101OrchestratorPipeline
from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data
from common.mock_llm import COST_AGENT_LLM_CALL_USD, COST_ROUTER_USD

#: Limiares vindos de App/eval/calibrar_cascata.py. Nenhum conjunto de teste foi
#: consultado para escolhe-los.
#:
#: CONFIANCA (0,65) — validacao cruzada 5-fold sobre os 53 exemplos de treino.
#:
#: FAMILIARIDADE (0,418) — percentil 90 da similaridade dos exemplos de treino
#: ENTRE SI. E um criterio RELATIVO: a mensagem nova precisa se parecer com
#: algum exemplo de treino mais do que o treino se parece consigo mesmo.
#:
#: Por que relativo e nao calibrado por validacao cruzada: medimos que a
#: primeira tentativa (limiar 0,20, vindo da validacao cruzada) DESVIAVA
#: "solicito a emissao do informe de rendimentos" — exatamente o modo de falha
#: que a cascata deveria evitar. Motivo: n-grama de caractere mede sobreposicao
#: de LETRAS, nao de registro linguistico, e essa frase formal tem
#: familiaridade 0,347, acima da MEDIANA do treino (0,336). A validacao cruzada
#: nao consegue detectar isso por construcao: todas as dobras vem da mesma
#: distribuicao estreita.
LIMIAR_CONFIANCA_PADRAO = 0.65
LIMIAR_FAMILIARIDADE_PADRAO = 0.418


class V102HybridPipeline(BasePipeline):
    """Cascata: classificador local na frente, orquestrador LLM atras."""

    name = "V1.0.2"
    description = "Cascata hibrida: local resolve FAST_PATH familiar; LLM resolve o resto"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        limiar_confianca: float = LIMIAR_CONFIANCA_PADRAO,
        limiar_familiaridade: float = LIMIAR_FAMILIARIDADE_PADRAO,
        use_embeddings: bool = True,
    ):
        self.settings = settings or get_settings()
        self.limiar_confianca = limiar_confianca
        self.limiar_familiaridade = limiar_familiaridade

        self._router: Optional[QueryRouter] = None
        self._vec_familiaridade: Optional[TfidfVectorizer] = None
        self._matriz_treino = None

        # A V1.0.2 nao reimplementa o orquestrador: ela COMPOE a V1.0.1.
        # Isso importa para a comparacao — o caminho do LLM e literalmente o
        # mesmo codigo, entao a unica diferenca medida entre as duas versoes e
        # a cascata em si, e nao alguma variacao acidental de implementacao.
        self._orquestrador = V101OrchestratorPipeline(
            settings=self.settings, use_embeddings=use_embeddings
        )

        # Instrumentacao: sem estes contadores nao da para responder "a cascata
        # valeu a pena?", que e a unica pergunta que esta versao existe para
        # responder.
        self.desvios = 0
        """Mensagens resolvidas localmente, sem chamar LLM."""
        self.chamadas_llm = 0
        """Mensagens que foram para o orquestrador."""
        self.desvios_por_motivo = {"confianca_baixa": 0, "nao_familiar": 0, "rota_agent": 0}
        """Por que cada mensagem NAO foi desviada. Diz qual dos dois sinais esta
        barrando o desvio, e portanto onde mexer para aumentar a economia."""

    # ------------------------------------------------------------------ setup
    def setup(self) -> "V102HybridPipeline":
        textos, rotulos = load_router_training_data()
        self._router = QueryRouter(
            confidence_threshold=self.limiar_confianca
        ).fit(textos, rotulos)

        # Indice de familiaridade: os proprios exemplos de treino, vetorizados
        # com os MESMOS n-gramas de caractere do classificador. A coerencia
        # importa: se o modelo enxerga o texto assim, a nocao de "parecido"
        # tem de ser a mesma.
        self._vec_familiaridade = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
        )
        self._matriz_treino = normalize(self._vec_familiaridade.fit_transform(textos))

        self._orquestrador.setup()
        return self

    # ---------------------------------------------------- sinais de decisao
    def _familiaridade(self, query: str) -> float:
        """Maior similaridade entre a mensagem e QUALQUER exemplo de treino.

        Devolve um valor de 0 a 1. Perto de 0 significa "nunca vi nada parecido"
        — e o sinal de que o classificador esta fora do terreno dele.
        """
        try:
            vetor = normalize(self._vec_familiaridade.transform([query]))
            return float((vetor @ self._matriz_treino.T).toarray().max())
        except Exception:
            # Entrada degenerada (vazia, so pontuacao) nao vetoriza. Tratar como
            # totalmente desconhecida e o comportamento seguro: manda para o LLM.
            return 0.0

    def _pode_desviar(self, rota: str, confianca: Optional[float], familiaridade: float) -> bool:
        """Decide se a mensagem pode pular o LLM. Registra o motivo da recusa."""
        if rota != "FAST_PATH":
            self.desvios_por_motivo["rota_agent"] += 1
            return False
        if confianca is None or confianca < self.limiar_confianca:
            self.desvios_por_motivo["confianca_baixa"] += 1
            return False
        if familiaridade < self.limiar_familiaridade:
            self.desvios_por_motivo["nao_familiar"] += 1
            return False
        return True

    # ---------------------------------------------------------------- process
    def process(self, query: str, k: int = 2) -> PipelineResult:
        if self._router is None:
            raise RuntimeError("Chame setup() antes de process().")

        inicio = time.perf_counter()

        # ETAPA 1 — O classificador local sempre roda. Custa ~1 ms e produz os
        # dois sinais de que a cascata precisa.
        try:
            decisao = self._router.predict(query)
            rota, confianca = decisao.route, decisao.confidence
        except Exception as erro:
            # Se o classificador falhar, mandamos para o LLM: e o caminho que
            # aguenta entrada estranha. Falhar para o lado caro e o certo aqui.
            resultado = self._orquestrador.process(query, k=k)
            resultado.trace.insert(0, f"classificador local falhou ({type(erro).__name__}) -> LLM")
            self.chamadas_llm += 1
            resultado.latency_ms = (time.perf_counter() - inicio) * 1000
            return resultado

        familiaridade = self._familiaridade(query)

        # ETAPA 2 — A bifurcacao da cascata.
        if self._pode_desviar(rota, confianca, familiaridade):
            self.desvios += 1
            resultado = PipelineResult(
                query=query,
                route="FAST_PATH",
                confidence=confianca,
                cost_usd=COST_ROUTER_USD,   # sem custo de LLM: e essa a economia
                llm_calls=0,
            )
            resultado.trace.append(
                f"DESVIO local: FAST_PATH (conf={confianca:.2f}, "
                f"familiaridade={familiaridade:.2f}) — LLM nao foi chamado"
            )
            resultado.latency_ms = (time.perf_counter() - inicio) * 1000
            return resultado

        # ETAPA 3 — Nao deu para desviar: vai para o orquestrador.
        # O trabalho do classificador local nao e desperdicado — ele ja produziu
        # os sinais que explicam POR QUE esta mensagem precisou do LLM, o que
        # aparece no rastro e permite ajustar os limiares depois.
        self.chamadas_llm += 1
        resultado = self._orquestrador.process(query, k=k)
        resultado.trace.insert(
            0,
            f"sem desvio (rota_local={rota}, conf={confianca:.2f}, "
            f"familiaridade={familiaridade:.2f}) -> orquestrador",
        )
        # A latencia do classificador local soma: na cascata, a mensagem paga os
        # dois estagios. Ignorar isso subestimaria o custo real do desenho.
        resultado.latency_ms = (time.perf_counter() - inicio) * 1000
        return resultado

    # -------------------------------------------------------------- telemetria
    @property
    def taxa_desvio(self) -> float:
        """Fracao das mensagens que nao chamaram LLM. E a economia da cascata."""
        total = self.desvios + self.chamadas_llm
        return self.desvios / total if total else 0.0

    def telemetria(self) -> dict:
        return {
            "desvios": self.desvios,
            "chamadas_llm": self.chamadas_llm,
            "taxa_desvio": self.taxa_desvio,
            "motivos_de_nao_desvio": dict(self.desvios_por_motivo),
            "limiar_confianca": self.limiar_confianca,
            "limiar_familiaridade": self.limiar_familiaridade,
            "fallbacks_do_orquestrador": self._orquestrador.fallbacks,
            "nota": (
                "A taxa de desvio e proporcional a fracao de FAST_PATH no trafego. "
                "Um conjunto pesado em AGENT subestima o ganho que a cascata teria "
                "num atendimento real."
            ),
        }
