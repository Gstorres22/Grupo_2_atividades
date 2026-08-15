"""Contrato comum entre a V1 e a V1.0.1.

===============================================================================
POR QUE ESTE ARQUIVO EXISTE
===============================================================================

Queremos responder a uma pergunta: "o orquestrador por LLM e melhor que o
classificador classico?". Para essa comparacao valer alguma coisa, as duas
versoes precisam ser:

  1. Alimentadas pelas MESMAS mensagens de teste.
  2. Medidas com as MESMAS metricas, calculadas do MESMO jeito.
  3. Intercambiaveis no codigo do comparador.

Se cada versao tivesse sua propria funcao de execucao e seu proprio formato de
saida, qualquer diferenca nos numeros poderia vir do arnes de medicao em vez do
sistema medido. Este arquivo elimina essa possibilidade: define UM formato de
resultado (`PipelineResult`) e UMA interface (`BasePipeline`) que as duas
implementam. O comparador so conhece esta interface.

Em engenharia isso se chama **inversao de dependencia**: o codigo de alto nivel
(o comparador) nao depende dos detalhes de baixo nivel (sklearn ou OpenAI);
os dois dependem de uma abstracao comum.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineResult:
    """Tudo o que uma versao produz ao processar UMA mensagem.

    Cada campo existe por um motivo de medicao especifico:
    """

    # --- A decisao em si ---------------------------------------------------
    query: str
    """A mensagem original do cliente, guardada para o relatorio ser auditavel
    sem precisar cruzar com o arquivo de entrada."""

    route: str
    """"FAST_PATH" (resposta pronta, sem IA) ou "AGENT" (precisa de ferramenta)."""

    tools: List[str] = field(default_factory=list)
    """As k ferramentas escolhidas, em ordem de relevancia. Vazio no FAST_PATH."""

    confidence: Optional[float] = None
    """Quao seguro o sistema esta da rota, de 0 a 1.

    Na V1 vem da probabilidade da regressao logistica. Na V1.0.1 vem do proprio
    LLM, que reporta sua confianca. IMPORTANTE: os dois numeros NAO sao
    diretamente comparaveis entre si — probabilidade de modelo calibrado e
    autoavaliacao de LLM sao coisas diferentes. Usamos cada um so dentro da
    sua propria versao."""

    # --- O custo da decisao -------------------------------------------------
    latency_ms: float = 0.0
    """Tempo total gasto para decidir, em milissegundos. Medido com
    `time.perf_counter()`, que e um relogio monotonico (nao anda para tras se o
    relogio do sistema for ajustado no meio da medicao)."""

    cost_usd: float = 0.0
    """Custo estimado em dolares. Na V1 e praticamente zero (so as constantes do
    mock). Na V1.0.1 inclui os tokens realmente gastos, convertidos pela tabela
    de precos. E o numero central da comparacao."""

    llm_calls: int = 0
    """Quantas chamadas de rede a um LLM foram feitas. Separado do custo porque
    o numero de chamadas domina a LATENCIA (cada uma paga ida e volta de rede),
    enquanto o numero de tokens domina o CUSTO. Sao gargalos diferentes."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    """Tokens de entrada e saida. Guardados separados porque tem precos
    diferentes — saida costuma custar varias vezes mais que entrada."""

    reasoning_tokens: int = 0
    """Tokens de "pensamento" que o modelo gerou e NAO devolveu na resposta.

    Existem apenas nos modelos de raciocinio (familia GPT-5, serie o). Sao
    invisiveis mas COBRADOS como saida — que e o preco mais caro. Medimos
    separadamente porque um valor alto aqui numa tarefa de classificacao
    significa dinheiro jogado fora, e o unico jeito de descobrir e olhando."""

    cached_tokens: int = 0
    """Tokens de entrada que vieram do cache do provedor e custam ate 90% menos.

    So acontece quando o prefixo do prompt se repete entre chamadas E passa do
    tamanho minimo exigido pelo provedor. Medir isso mostra se o cache esta de
    fato funcionando — um cache que silenciosamente para de funcionar multiplica
    a fatura sem nenhum aviso."""

    # --- Diagnostico --------------------------------------------------------
    trace: List[str] = field(default_factory=list)
    """Passo a passo do que aconteceu. Serve para entender POR QUE uma mensagem
    falhou, sem precisar reexecutar com depurador."""

    error: Optional[str] = None
    """Preenchido se algo deu errado. O pipeline nunca levanta excecao para
    fora: uma mensagem problematica vira uma linha com `error`, e o lote
    continua. Um teste de 300 mensagens nao pode morrer na mensagem 47."""

    def as_dict(self) -> Dict:
        """Converte para dicionario, para virar JSON no relatorio."""
        return {
            "query": self.query,
            "route": self.route,
            "tools": self.tools,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "trace": self.trace,
            "error": self.error,
        }


class BasePipeline(ABC):
    """O que toda versao precisa saber fazer.

    Sao apenas dois metodos, e a separacao entre eles e proposital:

      `setup()`  — trabalho CARO, feito UMA vez (treinar o modelo, indexar as
                   285 ferramentas, carregar embeddings do cache).
      `process()`— trabalho por mensagem, que e o que medimos.

    Se o treino acontecesse dentro de `process()`, a latencia medida incluiria
    um custo que em producao e pago so na subida do servico. A medicao ficaria
    inflada e sem sentido. Essa separacao tambem e o que permite empacotar o
    sistema em AWS Lambda depois: `setup()` roda no cold start, `process()` em
    cada invocacao.
    """

    #: Nome curto usado nos relatorios ("V1", "V1.0.1").
    name: str = "base"

    #: Descricao de uma linha, exibida no cabecalho do relatorio comparativo.
    description: str = ""

    @abstractmethod
    def setup(self) -> "BasePipeline":
        """Prepara tudo o que e caro. Deve devolver `self` para permitir
        encadear: `pipeline = MinhaVersao().setup()`."""

    @abstractmethod
    def process(self, query: str, k: int = 2) -> PipelineResult:
        """Processa UMA mensagem e devolve o resultado padronizado.

        Args:
            query: a mensagem do cliente.
            k: quantas ferramentas devolver quando a rota for AGENT.

        Nunca deve levantar excecao — erros vao no campo `error`.
        """
