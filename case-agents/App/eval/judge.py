"""LLM-as-judge — Precision@K SEMANTICO, ao lado da metrica estrita.

===============================================================================
O PROBLEMA QUE ESTE MODULO RESOLVE
===============================================================================

A metrica estrita pergunta: "o nome exato do gabarito estava no top-k?".
Em um catalogo com 285 tools e quase-duplicatas propositais, essa pergunta as
vezes e injusta com o sistema. Exemplo real do `eval_dataset.json`:

    Query    : "Fiz uma compra que nao reconheco, quero contestar"
    Gabarito : estornar_transacao
    Retornado: contestar_compra_desconhecida
               "Abre uma contestacao para uma compra nao reconhecida pelo cliente."

A metrica estrita marca ERRO. Mas a tool retornada faz exatamente o que o
cliente pediu — o cliente seria atendido. Contar isso como falha subestima o
sistema; contar como acerto sem criterio o superestima.

===============================================================================
COMO RESOLVEMOS (e por que NAO substituimos a metrica estrita)
===============================================================================

Um LLM avalia a EQUIVALENCIA FUNCIONAL entre a tool retornada e a esperada,
no contexto da query. Produzimos uma metrica "branda" que roda AO LADO da
estrita — nunca no lugar dela.

  * Precision@K estrito  -> reprodutivel, auditavel, sem custo, sem LLM.
                            E a metrica oficial do relatorio.
  * Precision@K semantico -> mais proximo da experiencia real do cliente,
                            mas depende de um juiz que tambem erra.

REPORTAR AS DUAS, SEMPRE. Mostrar so a semantica seria escolher a metrica que
favorece o resultado depois de ver o numero — isso e viés de seleção de métrica,
e invalida a avaliacao.

===============================================================================
CUIDADOS DE DESIGN DO JUIZ
===============================================================================

1. `temperature=0`: a mesma entrada deve dar o mesmo veredito.
2. O juiz NAO ve qual tool o sistema escolheu vs qual e o gabarito de forma
   rotulada — ele recebe as duas descricoes e julga equivalencia funcional.
   Isso reduz o vies de concordar com "a resposta oficial".
3. O juiz responde em 3 niveis (equivalente / parcial / diferente), nao em
   binario. "Parcial" e contabilizado a parte, nunca somado ao acerto.
4. Toda falha (rede, JSON invalido) e contabilizada e NAO vira acerto.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from App.core.config import Settings, get_settings
from common.schemas import Tool

JUDGE_SYSTEM_PROMPT = """Voce avalia ferramentas de um sistema de atendimento bancario.

Receba o pedido de um cliente e DUAS ferramentas (A e B). Decida se, para
ATENDER AQUELE PEDIDO, elas sao funcionalmente equivalentes.

Criterio: se o cliente ficaria igualmente atendido por qualquer uma das duas,
elas sao "equivalente". Se uma atende parcialmente (resolve parte do pedido, ou
exige um passo extra), e "parcial". Se levam a resultados diferentes, ou uma
delas trata de um produto/condicao que o cliente NAO mencionou (poupanca, PJ,
consignado, internacional), e "diferente".

Seja rigoroso: na duvida, responda "diferente".

Responda SOMENTE com JSON:
{"veredito": "equivalente" | "parcial" | "diferente", "motivo": "<uma frase>"}"""


@dataclass
class JudgeResult:
    strict_precision: float
    semantic_precision: float
    partial_rate: float
    n_evaluated: int
    n_judge_calls: int
    n_failures: int
    verdicts: List[dict] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "strict_precision_at_k": self.strict_precision,
            "semantic_precision_at_k": self.semantic_precision,
            "partial_rate": self.partial_rate,
            "n_evaluated": self.n_evaluated,
            "n_judge_calls": self.n_judge_calls,
            "n_failures": self.n_failures,
            "note": (
                "semantic_precision conta como acerto os casos em que o juiz considerou "
                "a tool retornada funcionalmente equivalente a esperada. 'parcial' NAO "
                "entra no acerto. Reportar sempre junto da metrica estrita."
            ),
        }


class SemanticJudge:
    def __init__(self, settings: Optional[Settings] = None, timeout: float = 20.0):
        self.settings = settings or get_settings()
        self._client = None
        self._cache: Dict[str, dict] = {}
        if self.settings.openai_api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.openai_api_key, timeout=timeout)

    @property
    def active(self) -> bool:
        return self._client is not None

    def _judge_pair(self, query: str, tool_a: Tool, tool_b: Tool) -> Optional[dict]:
        key = f"{query}||{tool_a.name}||{tool_b.name}"
        if key in self._cache:
            return self._cache[key]
        user = (
            f"Pedido do cliente: {query!r}\n\n"
            f"Ferramenta A — {tool_a.name}: {tool_a.description}\n"
            f"Ferramenta B — {tool_b.name}: {tool_b.description}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model_judge,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = json.loads(response.choices[0].message.content)
            self._cache[key] = payload
            return payload
        except Exception:
            return None

    def evaluate(self, rows: Sequence[dict], catalog: Sequence[Tool], k: int = 2) -> JudgeResult:
        """Avalia linhas no formato produzido por `App.eval.run_batch`.

        Cada linha precisa de `query`, `tools` (top-k retornado) e `expected_tool`.
        O juiz so e acionado quando a metrica estrita ERROU — quando ela acerta,
        nao ha o que reavaliar, e isso economiza a maior parte das chamadas.
        """
        by_name = {t.name: t for t in catalog}
        strict_hits = semantic_hits = partials = 0
        evaluated = calls = failures = 0
        verdicts: List[dict] = []

        for row in rows:
            expected = row.get("expected_tool")
            retrieved = (row.get("tools") or [])[:k]
            if not expected or expected not in by_name:
                continue
            evaluated += 1

            if expected in retrieved:
                strict_hits += 1
                semantic_hits += 1
                continue

            # A metrica estrita errou: vale perguntar ao juiz se o cliente
            # ainda assim seria atendido por alguma das tools retornadas.
            best = None
            for name in retrieved:
                if name not in by_name or not self.active:
                    continue
                calls += 1
                payload = self._judge_pair(row["query"], by_name[name], by_name[expected])
                if payload is None:
                    failures += 1
                    continue
                verdict = payload.get("veredito")
                verdicts.append({
                    "query": row["query"],
                    "retornada": name,
                    "esperada": expected,
                    "veredito": verdict,
                    "motivo": payload.get("motivo", ""),
                })
                if verdict == "equivalente":
                    best = "equivalente"
                    break
                if verdict == "parcial" and best is None:
                    best = "parcial"

            if best == "equivalente":
                semantic_hits += 1
            elif best == "parcial":
                partials += 1

        return JudgeResult(
            strict_precision=strict_hits / evaluated if evaluated else 0.0,
            semantic_precision=semantic_hits / evaluated if evaluated else 0.0,
            partial_rate=partials / evaluated if evaluated else 0.0,
            n_evaluated=evaluated,
            n_judge_calls=calls,
            n_failures=failures,
            verdicts=verdicts,
        )
