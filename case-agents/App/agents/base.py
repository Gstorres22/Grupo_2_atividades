"""Agente base — a peca reutilizavel sobre a qual personas e avaliadores sao feitos.

===============================================================================
O QUE E UM "AGENTE" AQUI
===============================================================================

Nada de magico: e um LLM com (a) um papel fixo descrito num prompt de sistema,
(b) uma saida obrigatoriamente em JSON, e (c) contabilidade de quanto custou.

Nao usamos framework de agentes nesta camada de proposito. Frameworks agregam
valor quando ha ferramentas, memoria entre turnos e decisao de proximo passo.
Aqui cada agente faz UMA chamada e devolve um JSON. Um framework adicionaria
dependencia e indirecao sem resolver problema nenhum que exista.

===============================================================================
AS TRES COISAS QUE ESTA CLASSE RESOLVE
===============================================================================

1. **JSON valido de verdade.** Pedir "responda em JSON" no prompt nao basta:
   o modelo as vezes embrulha em ```json ... ```, ou corta no meio. Usamos
   `response_format={"type":"json_object"}` e, mesmo assim, tentamos de novo se
   o parse falhar. Um agente que devolve texto solto quebra o pipeline inteiro.

2. **Falha isolada.** Se um dos 5 agentes de persona falhar, os outros 4 nao
   podem cair junto. Todo erro vira um resultado com `error` preenchido.

3. **Custo visivel.** Cada agente reporta tokens e dolares. Sem isso, "rodamos
   uma bateria de testes com IA" vira uma despesa que ninguem sabe medir.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from App.core.config import Settings, get_settings
from App.versions.v1_0_1_orchestrator import calcular_custo, modelo_usa_raciocinio


@dataclass
class AgentRun:
    """O resultado de uma execucao de agente."""

    agent_name: str
    payload: Optional[dict] = None
    """O JSON devolvido pelo agente. None se falhou."""

    error: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None
    attempts: int = 0
    """Quantas tentativas foram necessarias. Mais de 1 indica que o modelo
    devolveu JSON invalido na primeira — sinal util de instabilidade."""

    @property
    def ok(self) -> bool:
        return self.payload is not None and self.error is None

    def as_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "ok": self.ok,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "attempts": self.attempts,
            "payload": self.payload,
        }


class Agent:
    """Um LLM com papel fixo e saida em JSON.

    Args:
        name: identificador curto, usado nos relatorios e nos logs.
        system_prompt: define o papel. Fica FIXO entre chamadas de proposito —
            e o que permite comparar as saidas de dois agentes diferentes
            sabendo que a unica variavel foi o prompt.
        settings: configuracao. Se None, le do `.env`.
        model: sobrescreve o modelo padrao dos agentes. Util para o controle
            cruzado de vies (rodar o mesmo julgamento com outro modelo).
        temperature: 0 para julgamento (queremos reprodutibilidade), mais alto
            para geracao de dados (queremos diversidade). Cada subclasse
            escolhe o que faz sentido para o seu papel.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_attempts: int = 3,
        timeout: float = 180.0,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.settings = settings or get_settings()
        self.model = model or self.settings.agents_model
        self.temperature = temperature
        self.max_attempts = max_attempts

        if not self.settings.openai_api_key:
            raise ValueError("Os subagentes precisam de OPENAI_API_KEY em App/.env.")
        from openai import OpenAI

        self._client = OpenAI(api_key=self.settings.openai_api_key, timeout=timeout)

    # ------------------------------------------------------------------ run
    def run(self, user_prompt: str) -> AgentRun:
        """Executa o agente uma vez e devolve o resultado estruturado."""
        resultado = AgentRun(agent_name=self.name)
        inicio = time.perf_counter()

        for tentativa in range(1, self.max_attempts + 1):
            resultado.attempts = tentativa
            try:
                extras: Dict[str, object] = {}
                if modelo_usa_raciocinio(self.model):
                    # Nos avaliadores o raciocinio VALE a pena: julgar exige
                    # comparar evidencias. E o oposto do orquestrador, onde
                    # desligamos porque classificar nao precisa pensar.
                    extras["reasoning_effort"] = "medium"
                else:
                    extras["temperature"] = self.temperature

                resposta = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    **extras,
                )

                # Contabiliza o gasto ANTES de validar o conteudo: mesmo uma
                # resposta invalida ja foi cobrada, e ignorar isso subestimaria
                # o custo real da bateria de testes.
                uso = resposta.usage
                if uso:
                    resultado.prompt_tokens += uso.prompt_tokens
                    resultado.completion_tokens += uso.completion_tokens
                    det = getattr(uso, "completion_tokens_details", None)
                    if det:
                        resultado.reasoning_tokens += getattr(det, "reasoning_tokens", 0) or 0

                resultado.payload = json.loads(resposta.choices[0].message.content)
                resultado.error = None
                break  # sucesso: sai do laco de tentativas

            except json.JSONDecodeError as erro:
                # JSON malformado: vale tentar de novo, o modelo costuma acertar.
                resultado.error = f"JSON invalido: {erro}"
            except Exception as erro:
                # Rede, credencial, limite de taxa. Tentamos de novo com uma
                # espera crescente (backoff): 1s, 2s, 4s. Isso evita martelar
                # a API justamente quando ela esta pedindo para desacelerar.
                resultado.error = f"{type(erro).__name__}: {str(erro)[:200]}"
                if tentativa < self.max_attempts:
                    time.sleep(2 ** (tentativa - 1))

        resultado.latency_ms = (time.perf_counter() - inicio) * 1000
        resultado.cost_usd = calcular_custo(
            self.model, resultado.prompt_tokens, resultado.completion_tokens
        )
        return resultado


def run_parallel(tarefas: List[Callable[[], AgentRun]], max_workers: int = 5) -> List[AgentRun]:
    """Roda varios agentes ao mesmo tempo.

    Por que paralelizar: cada chamada de LLM passa a maior parte do tempo
    ESPERANDO a rede, nao usando CPU. Cinco agentes em sequencia levariam a
    soma dos tempos; em paralelo, levam o tempo do mais lento.

    Usamos threads (e nao processos) exatamente por isso — o trabalho e de
    entrada e saida, nao de processamento. O GIL do Python, que atrapalharia
    calculo pesado, e liberado durante a espera de rede.

    Uma tarefa que levante excecao vira um `AgentRun` com erro, e as demais
    continuam: uma persona quebrada nao pode derrubar a bateria inteira.
    """
    resultados: List[AgentRun] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(t): i for i, t in enumerate(tarefas)}
        for futuro in as_completed(futuros):
            try:
                resultados.append(futuro.result())
            except Exception as erro:
                resultados.append(
                    AgentRun(agent_name=f"tarefa_{futuros[futuro]}",
                             error=f"{type(erro).__name__}: {erro}")
                )
    return resultados


def resumir_custo(execucoes: List[AgentRun]) -> Dict:
    """Agrega o gasto de um conjunto de execucoes, para o rodape do relatorio."""
    conhecidos = [e.cost_usd for e in execucoes if e.cost_usd is not None]
    return {
        "n_execucoes": len(execucoes),
        "n_sucesso": sum(1 for e in execucoes if e.ok),
        "n_falha": sum(1 for e in execucoes if not e.ok),
        "tokens_entrada": sum(e.prompt_tokens for e in execucoes),
        "tokens_saida": sum(e.completion_tokens for e in execucoes),
        "tokens_raciocinio": sum(e.reasoning_tokens for e in execucoes),
        "custo_total_usd": sum(conhecidos) if conhecidos else None,
        "custo_desconhecido": len(conhecidos) < len(execucoes),
        "latencia_total_s": sum(e.latency_ms for e in execucoes) / 1000,
    }
