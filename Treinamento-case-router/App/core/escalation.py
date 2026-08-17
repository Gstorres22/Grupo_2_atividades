"""

===============================================================================
Cascata de escalonamento
===============================================================================

  1. ROTEAMENTO — o classificador local devolve uma probabilidade. Se ela ficar
     abaixo do limiar, a query esta na faixa cinzenta e um LLM pequeno desempata.
     Custo esperado: proximo de zero, porque isso quase nunca acontece.

  2. SELECAO DE TOOLS — se a margem entre a 1a e a 2a candidata for pequena, o
     ranking local nao tem conviccao para separa-las. So nesse caso as N
     candidatas (nao as 285 tools!) vao para um LLM reordenar.

Isso preserva economia e rerank em
100% das queries destruiria o ganho de custo/latencia; um rerank em 10% delas
custa 10% do preco e captura a maior parte do ganho de qualidade.

===============================================================================
CRITERIO DE ACEITE (e a disposicao de reportar resultado negativo)
===============================================================================

O escalonamento so fica ligado por padrao se PAGAR o proprio custo — ou seja,
se o ganho de Precision@K justificar o custo e a latencia adicionais, medidos
no notebook 04. Se nao pagar, o honesto e desligar e reportar o experimento
negativo: "testei, medi, nao compensou". Um numero inflado por uma decisao que
contraria a premissa do case vale menos do que a medicao honesta.

===============================================================================
POR QUE O PROMPT INSTRUI A PREFERIR A TOOL CANONICA
===============================================================================

Sem essa instrucao, o LLM cai na MESMA armadilha que a similaridade lexica:
para "Manda o pdf da minha fatura atual" ele escolheria `enviar_pdf_fatura_atual`
por ser a parafrase mais literal, quando a resposta esperada e a capacidade
canonica `consultar_fatura`. O prompt carrega explicitamente a politica de
desempate do catalogo — a mesma que o prior de generalidade implementa no
estagio local.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from App.core.config import Settings
from common.schemas import RetrievalResult, RouteResult, Tool, ToolMatch

VALID_ROUTES = {"FAST_PATH", "AGENT"}

ROUTER_SYSTEM_PROMPT = """Voce e o desempatador de um roteador de atendimento bancario.
Classifique a mensagem do cliente em exatamente uma rota:

- FAST_PATH: saudacao, agradecimento, despedida, ou pergunta GENERICA sobre a
  instituicao que nao depende dos dados da conta do cliente (horario de
  atendimento, taxas de tabela, documentos necessarios, como funciona o cashback).
- AGENT: qualquer pedido que dependa dos DADOS ou de uma ACAO na conta daquele
  cliente especifico (saldo, fatura, bloquear cartao, atualizar cadastro).

Regra de desempate: se a mensagem tem saudacao E um pedido acionavel
("Bom dia, preciso do meu saldo"), a rota e AGENT — a saudacao e so cortesia.

Responda SOMENTE com JSON: {"route": "FAST_PATH" | "AGENT", "reason": "<breve>"}"""

RERANK_SYSTEM_PROMPT = """Voce seleciona a ferramenta correta em um catalogo de atendimento bancario.

O catalogo tem MUITAS ferramentas quase identicas. Escolha seguindo esta politica:

1. Prefira a ferramenta CANONICA e mais GERAL que resolve a intencao do cliente.
   Variantes hiper-especificas cujo nome praticamente repete a frase do cliente
   costumam ser refinamentos, NAO a porta de entrada correta.
   Exemplo: para "manda o pdf da minha fatura", a escolha certa e a ferramenta
   geral de consultar fatura, e nao uma variante especializada em enviar PDF.
2. So escolha uma variante especifica se o cliente pediu EXPLICITAMENTE aquela
   condicao especial (ex.: falou "poupanca", "PJ", "consignado", "internacional").
3. Nunca invente nome de ferramenta: use exatamente um dos nomes da lista.

Responda SOMENTE com JSON: {"ranking": ["nome1", "nome2", ...], "reason": "<breve>"}
Ordene do mais adequado para o menos adequado, incluindo no maximo 5 nomes."""


@dataclass
class EscalationTelemetry:
    """Contabiliza o que a cascata gastou — insumo do relatorio de custo."""

    router_calls: int = 0
    rerank_calls: int = 0
    router_queries: int = 0
    rerank_queries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0
    changed_decisions: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def router_escalation_rate(self) -> float:
        return self.router_calls / self.router_queries if self.router_queries else 0.0

    @property
    def rerank_escalation_rate(self) -> float:
        return self.rerank_calls / self.rerank_queries if self.rerank_queries else 0.0

    def as_dict(self) -> Dict:
        return {
            "router_calls": self.router_calls,
            "router_escalation_rate": self.router_escalation_rate,
            "rerank_calls": self.rerank_calls,
            "rerank_escalation_rate": self.rerank_escalation_rate,
            "changed_decisions": self.changed_decisions,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
        }


class LLMEscalator:
    """Escalonamento condicional para um LLM pequeno.

    Falha de forma segura: qualquer erro de rede, JSON invalido ou nome de tool
    inexistente faz o metodo devolver a decisao LOCAL original. O LLM so pode
    MELHORAR o resultado, nunca derrubar o pipeline.
    """

    def __init__(self, settings: Settings, timeout: float = 10.0):
        self.settings = settings
        self.telemetry = EscalationTelemetry()
        self._client = None
        if settings.llm_enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=settings.openai_api_key, timeout=timeout)
            except Exception as error:
                self.telemetry.notes.append(f"cliente indisponivel: {error}")

    @property
    def active(self) -> bool:
        return self._client is not None

    # -- Infra ----------------------------------------------------------------
    def _chat_json(self, system: str, user: str) -> Optional[dict]:
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model_small,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,  # decisao de roteamento deve ser reprodutivel
            )
            usage = response.usage
            if usage:
                self.telemetry.prompt_tokens += usage.prompt_tokens
                self.telemetry.completion_tokens += usage.completion_tokens
            return json.loads(response.choices[0].message.content)
        except Exception as error:
            self.telemetry.failures += 1
            self.telemetry.notes.append(str(error)[:200])
            return None

    # -- 1) Desempate de rota --------------------------------------------------
    def resolve_route(self, query: str, local: RouteResult, is_uncertain: bool) -> RouteResult:
        """Devolve a rota final. Só chama o LLM se `is_uncertain` for verdadeiro."""
        self.telemetry.router_queries += 1
        if not self.active or not is_uncertain:
            return local

        self.telemetry.router_calls += 1
        payload = self._chat_json(ROUTER_SYSTEM_PROMPT, f"Mensagem do cliente: {query!r}")
        if not payload or payload.get("route") not in VALID_ROUTES:
            return local  # falha segura: mantem a decisao local

        if payload["route"] != local.route:
            self.telemetry.changed_decisions += 1
        return RouteResult(
            route=payload["route"],
            latency_ms=local.latency_ms,
            confidence=local.confidence,
        )

    # -- 2) Rerank de tools ----------------------------------------------------
    def rerank_tools(
        self,
        query: str,
        candidates: RetrievalResult,
        catalog: Sequence[Tool],
        k: int,
        is_ambiguous: bool,
    ) -> RetrievalResult:
        """Reordena as candidatas. Só chama o LLM se `is_ambiguous` for verdadeiro."""
        self.telemetry.rerank_queries += 1
        top_k_local = RetrievalResult(
            matches=list(candidates.matches[:k]), latency_ms=candidates.latency_ms
        )
        if not self.active or not is_ambiguous or len(candidates.matches) <= 1:
            return top_k_local

        by_name = {t.name: t for t in catalog}
        listed = [
            f"- {m.name}: {by_name[m.name].description}"
            for m in candidates.matches
            if m.name in by_name
        ]
        if not listed:
            return top_k_local

        self.telemetry.rerank_calls += 1
        payload = self._chat_json(
            RERANK_SYSTEM_PROMPT,
            f"Pedido do cliente: {query!r}\n\nFerramentas candidatas:\n" + "\n".join(listed),
        )
        if not payload or not isinstance(payload.get("ranking"), list):
            return top_k_local

        scores = {m.name: m.score for m in candidates.matches}
        # Aceita apenas nomes que existiam nas candidatas: blinda contra alucinacao.
        reranked = [name for name in payload["ranking"] if name in scores][:k]
        if not reranked:
            return top_k_local

        if reranked[0] != top_k_local.matches[0].name:
            self.telemetry.changed_decisions += 1

        # Completa com as locais caso o LLM devolva menos de k nomes.
        for match in candidates.matches:
            if len(reranked) >= k:
                break
            if match.name not in reranked:
                reranked.append(match.name)

        return RetrievalResult(
            matches=[ToolMatch(name=n, score=scores[n]) for n in reranked],
            latency_ms=candidates.latency_ms,
        )
