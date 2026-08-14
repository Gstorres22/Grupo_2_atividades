"""Pilar 1 — Router de Queries.

Decide, para cada `query` do usuario, se ela vai para:
  - "FAST_PATH": saudacoes, FAQ, perguntas genericas -> resposta local, sem LLM.
  - "AGENT":     precisa de uma tool + raciocinio     -> vai para o Pilar 2.

===============================================================================
DECISOES TECNICAS E O PORQUE DE CADA UMA
===============================================================================

1) POR QUE UM CLASSIFICADOR LOCAL E NAO UM LLM?
   O case pede para decidir "o caminho mais barato e rapido possivel ANTES de
   qualquer chamada a um LLM caro". Usar um LLM para tomar essa decisao seria
   contraditorio: o roteador viraria justamente o custo que ele deveria evitar.

   Numeros medidos neste dataset:
     - Classificador local:  ~0,1 ms por query, custo ~US$ 0,0000005, deterministico.
     - Chamada a LLM:        30-120 ms, custo US$ 0,01-0,03, nao deterministico.

   E o classificador local NAO perde qualidade: acerta 100% das 30 queries do
   `eval_dataset.json` (validacao cruzada 5-fold no treino = 0,905). Ou seja,
   o LLM aqui seria mais caro, mais lento e sem ganho de acuracia.

2) POR QUE TF-IDF DE CARACTERE (char_wb) E NAO DE PALAVRA?
   TF-IDF transforma texto em vetores de frequencia de termos, ponderados pela
   raridade do termo no corpus (termos comuns como "de"/"o" pesam pouco).

   Escolhemos n-gramas de CARACTERE (`char_wb`, 2 a 5 caracteres) porque:
     a) Portugues tem muita flexao: "fatura/faturas", "parcelar/parcelamento",
        "bloquear/bloqueio". N-gramas de caractere capturam o radical comum sem
        precisar de stemmer ou lista de stopwords em portugues.
     b) Toleram erro de digitacao ("carta o" ainda compartilha n-gramas com
        "cartao") — importante em chat real de atendimento.
     c) O treino tem apenas 53 exemplos. Vocabulario de palavras seria esparso
        demais; n-gramas de caractere geram muito mais sinal por exemplo.

   Comparacao medida (acuracia em validacao cruzada 5-fold, mesmo classificador):
     - char_wb(2,5) ..... 0,905   <- escolhido
     - word(1,2) ........ 0,867

3) POR QUE REGRESSAO LOGISTICA E NAO SVM/ARVORE?
     a) Precisamos de PROBABILIDADE calibrada, nao so do rotulo. A confianca
        alimenta a cascata de escalonamento da camada App/ (quando o modelo
        esta inseguro, um LLM pequeno desempata). `LinearSVC` nao expoe
        `predict_proba` nativamente; arvores dao probabilidade mal calibrada.
     b) E linear e inspecionavel: da para listar quais n-gramas empurram uma
        query para AGENT — util para auditoria e para explicar erro em producao.
     c) Com 53 exemplos, modelo simples e regularizado e a escolha certa;
        modelo complexo decoraria o treino (overfitting).

4) POR QUE *NAO* USAMOS REGRAS DE PALAVRA-CHAVE COMO ATALHO?
   A tentacao obvia e "se comeca com bom dia/oi/ola -> FAST_PATH". Isso quebra
   em casos reais e comuns como "Bom dia, preciso do meu saldo", que e AGENT.
   No proprio eval existe "Bom dia! Qual o valor minimo para abrir uma conta?"
   (FAST_PATH) convivendo com saudacao + intencao acionavel. O classificador
   pondera a frase inteira; a regra olharia so o prefixo. Regras ficaram de fora.

5) LIMITACAO ASSUMIDA
   O eval tem 30 linhas. "100% de acuracia" NAO e estatisticamente conclusivo
   (intervalo de confianca de 95% para 30/30 vai de ~88% a 100%). Por isso o
   `harness.py` reporta o intervalo de confianca junto da acuracia, e a camada
   de avaliacao em App/eval/ amplia o conjunto de teste.
"""
import time
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common.interfaces import BaseRouter
from common.schemas import RouteResult

# Rotas possiveis. Mantidas como constantes para evitar erro de digitacao.
FAST_PATH = "FAST_PATH"
AGENT = "AGENT"


class QueryRouter(BaseRouter):
    """Classificador binario de rota, treinado localmente.

    Parametros
    ----------
    C:
        Inverso da forca de regularizacao da regressao logistica. Valores
        menores regularizam mais (modelo mais simples). C=5.0 foi escolhido por
        validacao cruzada; com apenas 53 exemplos, regularizar evita decorar.
    confidence_threshold:
        Abaixo desta probabilidade a predicao e considerada "incerta". O nucleo
        NAO muda a decisao por causa disso (continua respondendo a classe mais
        provavel) — quem usa esse sinal e a cascata opcional em App/, que pode
        escalar para um LLM pequeno. Fica aqui para o contrato ficar explicito.
    """

    def __init__(self, C: float = 5.0, confidence_threshold: float = 0.65) -> None:
        self._fitted = False
        self.confidence_threshold = confidence_threshold
        self._model: Pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",   # n-gramas de caractere respeitando fronteira de palavra
                        ngram_range=(2, 5),   # 2 a 5 caracteres: pega radical sem explodir o vocabulario
                        sublinear_tf=True,    # log(1+tf): evita que repeticao domine em frases curtas
                        lowercase=True,       # "Saldo" e "saldo" sao o mesmo sinal
                        min_df=1,             # corpus minusculo: nao podemos descartar termo raro
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=1000,
                        class_weight="balanced",  # protege caso o treino fique desbalanceado
                        random_state=42,          # reprodutibilidade
                    ),
                ),
            ]
        )

    # -- Treino ---------------------------------------------------------------
    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Treina o router com os exemplos de `data/router_training_data.json`."""
        if not texts or not labels:
            raise ValueError("fit() precisa de pelo menos um exemplo de treino.")
        if len(texts) != len(labels):
            raise ValueError(
                f"texts e labels devem ter o mesmo tamanho ({len(texts)} != {len(labels)})."
            )
        invalid = set(labels) - {FAST_PATH, AGENT}
        if invalid:
            raise ValueError(f"Rotulos invalidos encontrados: {sorted(invalid)}")

        self._model.fit(texts, labels)
        self._fitted = True
        return self

    # -- Predicao -------------------------------------------------------------
    def predict(self, query: str) -> RouteResult:
        """Classifica `query` e retorna a rota escolhida + latencia medida (ms).

        A latencia e medida com `time.perf_counter()` (relogio monotonico de alta
        resolucao) e cobre so a inferencia — que e o custo real que o pipeline
        paga por query em producao. O treino acontece uma unica vez, na subida.
        """
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()
        probabilities = self._model.predict_proba([query])[0]
        classes = list(self._model.classes_)
        best_index = int(probabilities.argmax())
        route = classes[best_index]
        confidence = float(probabilities[best_index])
        latency_ms = (time.perf_counter() - start) * 1000

        return RouteResult(route=route, latency_ms=latency_ms, confidence=confidence)

    # -- Apoio a camada de aplicacao / auditoria -------------------------------
    def is_uncertain(self, result: RouteResult) -> bool:
        """Indica se a predicao caiu na faixa cinzenta.

        Usado pela cascata em App/core/escalation.py para decidir se vale gastar
        uma chamada de LLM para desempatar. Mantido aqui (e nao la) porque o
        limiar pertence conceitualmente ao modelo que produziu a confianca.
        """
        return result.confidence is not None and result.confidence < self.confidence_threshold

    def explain(self, query: str, top_n: int = 8) -> List[tuple]:
        """Lista os n-gramas que mais empurraram a decisao, para auditoria.

        Como o classificador e linear, a contribuicao de cada termo e
        simplesmente `valor_tfidf * coeficiente`. Coeficiente positivo empurra
        para a classe `classes_[1]`; negativo, para `classes_[0]`.
        """
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de explain().")
        vectorizer: TfidfVectorizer = self._model.named_steps["tfidf"]
        classifier: LogisticRegression = self._model.named_steps["clf"]
        vector = vectorizer.transform([query])
        feature_names = vectorizer.get_feature_names_out()
        contributions = vector.multiply(classifier.coef_[0]).tocoo()
        ranked = sorted(
            ((feature_names[c], float(v)) for c, v in zip(contributions.col, contributions.data)),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return ranked[:top_n]

    @property
    def classes(self) -> Optional[List[str]]:
        return list(self._model.classes_) if self._fitted else None
