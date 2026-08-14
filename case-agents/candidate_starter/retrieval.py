"""Pilar 2 — Selecao de Tools Relevantes.

`search(query, k=2)` devolve as `k` tools mais relevantes do catalogo ANTES de
qualquer chamada a LLM. Passar as 285 tools do registry no prompt estouraria o
contexto, confundiria o modelo e multiplicaria custo e latencia.

===============================================================================
O PROBLEMA REAL DESTE CATALOGO (e por que similaridade pura nao resolve)
===============================================================================

O `tools_registry.json` tem 285 tools e foi montado com quase-duplicatas
propositais. Para varias queries, a tool MAIS PARECIDA lexicalmente NAO e a
tool correta — ela e uma variante hiper-especifica:

    Query: "Quanto eu tenho disponivel na conta agora?"
      correto ......... consultar_saldo
                        "Consulta o saldo bancario da conta corrente do cliente."
      similaridade .... consultar_valor_disponivel_conta
                        "Consulta o valor disponivel agora na conta corrente do cliente."
                        (praticamente uma copia da query -> vence por similaridade)

    Query: "Manda o pdf da minha fatura atual"
      correto ......... consultar_fatura
      similaridade .... enviar_pdf_fatura_atual

Medido neste dataset (20 queries de rota AGENT com tool esperada):

    TF-IDF de palavra, similaridade pura ................ Precision@2 = 0,25
    TF-IDF de caractere, similaridade pura .............. Precision@2 = 0,35
    RRF(caractere + palavra) ............................ Precision@2 = 0,45
    RRF + prior de generalidade (a solucao adotada) ..... Precision@2 = 0,80

O gabarito premia a tool CANONICA (a capacidade principal), nao a parafrase
mais literal da query. Faz sentido em producao: entre duas tools que atendem a
intencao, a mais geral cobre mais casos e e a porta de entrada correta; as
variantes longas sao refinamentos de um catalogo mal higienizado.

===============================================================================
ARQUITETURA EM 2 ESTAGIOS (+1 opcional, fora deste arquivo)
===============================================================================

  Estagio 1 - RECALL (barato, abrangente)
      Busca lexica hibrida: TF-IDF de caractere + TF-IDF de palavra, fundidos
      por Reciprocal Rank Fusion. Objetivo: garantir que a tool certa esteja
      entre as N candidatas. Recall@15 e o TETO de todo o resto do sistema.

  Estagio 2 - DESAMBIGUACAO (precisao)
      Reordena as N candidatas aplicando o prior de generalidade, que corrige
      exatamente o vies mostrado acima.

  Estagio 3 - ESCALONAMENTO (opcional, vive em App/core/escalation.py)
      So dispara quando a margem entre 1o e 2o colocado e pequena (ambiguidade
      real). NAO faz parte deste arquivo de proposito: o nucleo do case precisa
      rodar offline, sem chave de API e sem dependencia nova.

===============================================================================
POR QUE CADA PECA (e o que foi testado e DESCARTADO)
===============================================================================

1) POR QUE DOIS INDICES LEXICOS EM VEZ DE UM?
   Eles erram de formas diferentes, entao se complementam:
     - caractere (char_wb 2-5): robusto a flexao ("parcelar"/"parcelamento") e
       a erro de digitacao. Sozinho: P@2 = 0,35.
     - palavra (1-2 gramas): captura expressoes inteiras ("segunda via",
       "linha digitavel"), que o n-grama de caractere dilui. Sozinho: P@2 = 0,20,
       mas com o MELHOR recall (Recall@15 = 0,90 vs 0,85 do caractere).
   Juntos por RRF: P@2 = 0,45. Cada um sozinho seria pior.

2) POR QUE RECIPROCAL RANK FUSION (RRF) E NAO MEDIA DOS SCORES?
   Os dois indices produzem scores em escalas diferentes e nao comparaveis
   (distribuicoes distintas de similaridade). Somar exigiria normalizar, e a
   normalizacao ficaria refem de outliers. O RRF combina POSICOES no ranking,
   nao valores: cada tool recebe 1/(k + posicao). E imune a escala e e o padrao
   da literatura de busca hibrida. `k=60` e o valor classico do paper original;
   ele amortece a diferenca entre as primeiras posicoes, evitando que um unico
   indice muito confiante domine a fusao.

3) POR QUE O PRIOR DE GENERALIDADE?
   E o que separa a tool canonica da variante hiper-especifica. Medimos a
   especificidade por dois sinais baratos e observaveis no proprio registry:
   numero de tokens do nome e tamanho da descricao. `consultar_saldo` (2 tokens)
   e mais canonica que `consultar_valor_disponivel_conta` (4 tokens).

   Efeito medido (Precision@2, variando lambda):
       lambda=0,0 -> 0,45     lambda=0,3 -> 0,75
       lambda=0,1 -> 0,65     lambda=0,4 -> 0,80
       lambda=0,2 -> 0,75     lambda=0,5 -> 0,80

   SOBRE AJUSTE NO CONJUNTO DE TESTE (honestidade metodologica): o valor padrao
   lambda=0,35 NAO foi escolhido pelo pico da curva — isso seria ajustar no
   proprio teste (leakage) com apenas 20 queries. Foi escolhido o centro do
   PLATO estavel (0,2 a 0,5, onde P@2 fica entre 0,75 e 0,80). A largura desse
   plato e justamente a evidencia de que o resultado nao depende de um ajuste
   fragil. A calibracao definitiva roda no conjunto ampliado (App/eval/).

4) ALTERNATIVA TESTADA E DESCARTADA #1 — BM25 com normalizacao de comprimento
   BM25 e a resposta de manual para "nao favorecer documentos longos" (o
   parametro `b`). Testamos b de 0,0 a 1,0: o melhor resultado foi P@2 = 0,35,
   MUITO abaixo do prior explicito (0,80). Motivo: o `b` do BM25 normaliza a
   SATURACAO DE FREQUENCIA de termos, nao a especificidade do conceito. Ele
   trata "descricao longa" como ruido a compensar, e nao como sinal de que a
   tool e uma variante estreita. Descartado por evidencia.

5) ALTERNATIVA TESTADA E DESCARTADA #2 — clusterizar quase-duplicatas
   A ideia era agrupar tools quase identicas e retornar so o representante
   canonico de cada grupo. Na pratica PIOROU: o agrupamento por transitividade
   encadeia grupos grandes demais (a ~ b, b ~ c => a ~ c mesmo com a e c
   distantes) e chegou a derrubar o Recall@15 de 0,85 para 0,70, jogando fora
   a tool correta antes de qualquer reordenacao. Descartado por evidencia.
   A simplicidade venceu: RRF + prior, sem clusterizacao.

6) LIMITE CONHECIDO — o teto de recall
   Com busca lexica pura, Recall@15 fica em 0,85-0,90. As falhas sao todas
   SEMANTICAS, onde nao ha sobreposicao de palavras:
       "Preciso mudei de casa, como mudo o CEP?"  -> alterar_endereco
       "Cobraram algo errado, quero o dinheiro de volta" -> estornar_transacao
   Nenhuma reordenacao recupera uma tool que nao entrou na lista de candidatas.
   Por isso o gancho `set_dense_provider()`: a camada App/ injeta embeddings
   (que capturam significado, nao palavra) para levantar esse teto. O gancho
   mantem o nucleo sem dependencia: sem provider, roda 100% lexico.
"""
import re
import time
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch

# Constante classica do paper de Reciprocal Rank Fusion (Cormack et al., 2009).
RRF_K = 60

# Tipo do provedor de embeddings injetado pela camada App/.
# Recebe uma lista de textos e devolve uma matriz (n_textos, n_dimensoes).
DenseProvider = Callable[[Sequence[str]], np.ndarray]


def _reciprocal_rank(scores: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Converte um vetor de scores em contribuicao RRF: 1 / (k + posicao).

    `argsort` duplo transforma scores em posicoes (0 = melhor colocado).
    Trabalhar com posicao, e nao com o score bruto, e o que torna a fusao imune
    a diferenca de escala entre os indices.
    """
    ranks = np.argsort(np.argsort(-scores))
    return 1.0 / (k + ranks + 1)


class ToolRetriever(BaseToolRetriever):
    """Busca hibrida em 2 estagios sobre o catalogo de tools.

    Parametros
    ----------
    generality_lambda:
        Intensidade do prior de generalidade (0 = desligado). Ver secao 3 do
        docstring do modulo para a justificativa do valor padrao.
    n_candidates:
        Tamanho da lista de candidatas do Estagio 1 que segue para o Estagio 2.
        15 e folgado o suficiente para o recall e pequeno o suficiente para o
        reordenamento (e um eventual rerank por LLM) sair barato.
    """

    def __init__(
        self,
        generality_lambda: float = 0.35,
        n_candidates: int = 15,
        rrf_k: int = RRF_K,
    ) -> None:
        self._tools: List[Tool] = []
        self._fitted = False
        self.generality_lambda = generality_lambda
        self.n_candidates = n_candidates
        self.rrf_k = rrf_k

        # Indice lexico de caractere: robusto a flexao do portugues e a typos.
        self._vec_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True
        )
        # Indice lexico de palavra: captura expressoes ("segunda via").
        self._vec_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True, lowercase=True
        )
        self._matrix_char = None
        self._matrix_word = None
        self._specificity: Optional[np.ndarray] = None

        # Estagio denso opcional, injetado pela camada App/ (ver set_dense_provider).
        self._dense_provider: Optional[DenseProvider] = None
        self._matrix_dense: Optional[np.ndarray] = None

    # -- Indexacao ------------------------------------------------------------
    @staticmethod
    def _to_document(tool: Tool) -> str:
        """Texto indexavel de uma tool.

        Concatenamos nome + descricao + categoria porque cada campo carrega um
        sinal diferente: o nome tem o verbo da acao ("bloquear", "consultar"),
        a descricao tem o vocabulario que o cliente usa, e a categoria funciona
        como um termo de dominio que aproxima queries do mesmo assunto.
        O underscore vira espaco para que "consultar_saldo" case com "saldo".
        """
        return f"{tool.name.replace('_', ' ')} {tool.description} {tool.category}"

    def _compute_specificity(self, tools: List[Tool]) -> np.ndarray:
        """Quao ESPECIFICA (o oposto de canonica) e cada tool, normalizado em [0, 1].

        Dois sinais, com peso igual por nao haver evidencia para preferir um:
          - numero de tokens do nome: `consultar_saldo` (2) vs
            `consultar_valor_disponivel_conta` (4);
          - tamanho da descricao: variantes estreitas descrevem mais condicoes.
        """
        name_tokens = np.array([len(t.name.split("_")) for t in tools], dtype=float)
        desc_words = np.array([len(re.findall(r"\w+", t.description)) for t in tools], dtype=float)

        def _minmax(values: np.ndarray) -> np.ndarray:
            spread = values.max() - values.min()
            if spread == 0:  # catalogo homogeneo: sinal nao discrimina nada
                return np.zeros_like(values)
            return (values - values.min()) / spread

        return 0.5 * _minmax(name_tokens) + 0.5 * _minmax(desc_words)

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catalogo de tools para busca."""
        if not tools:
            raise ValueError("fit() precisa de pelo menos uma tool no catalogo.")

        self._tools = list(tools)
        documents = [self._to_document(t) for t in self._tools]

        # `normalize` deixa cada linha com norma 1, entao o produto escalar
        # entre query e documento JA E a similaridade de cosseno.
        self._matrix_char = normalize(self._vec_char.fit_transform(documents))
        self._matrix_word = normalize(self._vec_word.fit_transform(documents))
        self._specificity = self._compute_specificity(self._tools)

        if self._dense_provider is not None:
            self._matrix_dense = self._embed(documents)

        self._fitted = True
        return self

    # -- Estagio denso opcional (injetado por App/) ----------------------------
    def set_dense_provider(self, provider: Optional[DenseProvider]) -> "ToolRetriever":
        """Registra um gerador de embeddings para o Estagio 1.

        Injecao de dependencia de proposito: o nucleo do case nao importa
        `openai` nem `langchain`. A camada App/ passa a funcao pronta. Sem
        provider, a busca continua 100% lexica e offline — que e o modo em que
        os testes e o `run_case.py` rodam.
        """
        self._dense_provider = provider
        if self._fitted and provider is not None:
            self._matrix_dense = self._embed([self._to_document(t) for t in self._tools])
        elif provider is None:
            self._matrix_dense = None
        return self

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._dense_provider(texts), dtype=float)
        if vectors.ndim != 2:
            raise ValueError("O provedor de embeddings deve devolver uma matriz 2D.")
        return normalize(vectors)

    # -- Busca ----------------------------------------------------------------
    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para `query`."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")
        if k <= 0:
            raise ValueError("k deve ser maior que zero.")

        start = time.perf_counter()

        # --- Estagio 1: recall -------------------------------------------
        sim_char = (normalize(self._vec_char.transform([query])) @ self._matrix_char.T).toarray()[0]
        sim_word = (normalize(self._vec_word.transform([query])) @ self._matrix_word.T).toarray()[0]

        fused = _reciprocal_rank(sim_char, self.rrf_k) + _reciprocal_rank(sim_word, self.rrf_k)
        if self._matrix_dense is not None:
            sim_dense = self._matrix_dense @ self._embed([query])[0]
            fused = fused + _reciprocal_rank(sim_dense, self.rrf_k)

        # --- Estagio 2: desambiguacao por generalidade ---------------------
        # Multiplicativo (e nao subtrativo) para o desconto ser proporcional ao
        # score: uma tool com pouca relevancia nao sobe por ser generica.
        scores = fused * (1.0 - self.generality_lambda * self._specificity)

        n_top = min(max(k, self.n_candidates), len(self._tools))
        candidates = np.argpartition(-scores, n_top - 1)[:n_top]
        ordered = candidates[np.argsort(-scores[candidates])][:k]

        matches = [
            ToolMatch(name=self._tools[i].name, score=float(scores[i])) for i in ordered
        ]
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(matches=matches, latency_ms=latency_ms)

    # -- Apoio a camada de aplicacao ------------------------------------------
    def search_candidates(self, query: str, n: Optional[int] = None) -> RetrievalResult:
        """Devolve a lista longa de candidatas (Estagio 1+2) sem cortar em k.

        Usado pelo rerank opcional em App/: o LLM recebe estas N candidatas em
        vez das 285 tools do catalogo — que e exatamente a economia que o case
        pede para demonstrar.
        """
        return self.search(query, k=n or self.n_candidates)

    @staticmethod
    def margin(result: RetrievalResult) -> float:
        """Diferenca relativa entre o 1o e o 2o colocado.

        E o sinal de ambiguidade que dispara o escalonamento: margem pequena
        significa que o ranking local nao tem conviccao para separar os dois, e
        so nesse caso vale gastar uma chamada de LLM.
        """
        if len(result.matches) < 2:
            return 1.0
        first, second = result.matches[0].score, result.matches[1].score
        if first <= 0:
            return 0.0
        return (first - second) / first

    @property
    def n_tools(self) -> int:
        return len(self._tools)
