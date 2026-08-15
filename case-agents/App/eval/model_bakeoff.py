"""Comparativo de modelos para o papel de orquestrador (V1.0.1).

===============================================================================
POR QUE ESTE ARQUIVO EXISTE
===============================================================================

A pesquisa de precos nos deu candidatos e uma recomendacao no papel. Mas preco
de tabela e benchmark publico nao dizem qual modelo acerta MAIS NA NOSSA TAREFA,
que e classificar mensagens de banco em portugues e escolher entre 20
ferramentas quase-duplicadas.

Este script roda o MESMO pipeline, com o MESMO prompt e as MESMAS mensagens,
trocando apenas o modelo. A unica variavel que muda e o modelo — que e o que
torna a comparacao valida.

Ele mede quatro coisas, porque nenhuma delas sozinha decide:

    ACERTO   -> acuracia de rota e Precision@2
    CUSTO    -> dolares reais, a partir dos tokens de fato gastos
    LATENCIA -> p50 e p95, porque a media esconde a cauda
    TOKENS   -> incluindo os de raciocinio, que sao invisiveis mas cobrados

USO
---
    python -m App.eval.model_bakeoff
    python -m App.eval.model_bakeoff --models gpt-5.6-luna gpt-4o-mini
    python -m App.eval.model_bakeoff --out App/reports/bakeoff.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

from App.core.config import get_settings
from App.versions.base import PipelineResult
from App.versions.v1_0_1_orchestrator import (
    MODELOS_DESCONTINUADOS,
    V101OrchestratorPipeline,
)
from App.versions.v1_classic import V1ClassicPipeline
from common.data_loader import load_eval_dataset

#: Candidatos padrao. Escolhidos a partir da pesquisa de precos:
#: - gpt-5.6-luna : o tier nano atual, substituto oficial dos nanos descontinuados
#: - gpt-5.4-nano : praticamente empatado em preco, geracao anterior
#: - gpt-4o-mini  : mais barato sem cache e SEM raciocinio (zero token invisivel)
MODELOS_PADRAO = ["gpt-4o-mini", "gpt-5.4-nano", "gpt-5.6-luna"]

#: Casos que a V1 erra, extraidos dos testes de persona da V1.
#: Nao substituem o dataset oficial — sao um teste dirigido aos modos de falha
#: conhecidos. Um modelo que acerte o dataset oficial e erre estes nao resolveu
#: o problema que motivou a V1.0.1.
CASOS_CRITICOS = [
    {"query": "quero bloquear meu cartao",
     "expected_route": "AGENT", "expected_tool": "bloquear_cartao",
     "falha_v1": "negacao: devolve desbloquear_cartao"},
    {"query": "qero blqouear meu cartaao",
     "expected_route": "AGENT", "expected_tool": "bloquear_cartao",
     "falha_v1": "negacao + erro de digitacao"},
    {"query": "solicito a emissao do informe de rendimentos para o IRPF",
     "expected_route": "AGENT", "expected_tool": "emitir_informe_rendimentos",
     "falha_v1": "registro formal vira FAST_PATH"},
    {"query": "oi, preciso de ajuda com o meu saldo",
     "expected_route": "AGENT", "expected_tool": "consultar_saldo",
     "falha_v1": "saudacao sequestra a decisao"},
    {"query": "precido do extrado bnacario",
     "expected_route": "AGENT", "expected_tool": "consultar_extrato",
     "falha_v1": "erro de digitacao vira FAST_PATH"},
    {"query": "bom dia, qual o horario de atendimento?",
     "expected_route": "FAST_PATH", "expected_tool": None,
     "falha_v1": "(controle: a V1 acerta este)"},
]


def avaliar(pipeline, casos: List[dict], k: int = 2) -> Dict:
    """Roda um pipeline sobre uma lista de casos e agrega as metricas.

    Recebe o pipeline pronto (ja com `setup()` chamado) porque o custo de
    preparacao nao deve entrar na medicao de latencia por mensagem.
    """
    linhas: List[PipelineResult] = []
    acertos_rota = 0
    acertos_tool = 0
    total_com_tool = 0

    for caso in casos:
        r = pipeline.process(caso["query"], k=k)
        linhas.append(r)
        if r.route == caso["expected_route"]:
            acertos_rota += 1
        # Precision@k so e contabilizado quando ha ferramenta esperada E o
        # pipeline de fato foi para AGENT. Se ele errou a rota, contamos como
        # erro de ferramenta tambem — senao um roteador ruim inflaria o P@k
        # ao remover da conta justamente as mensagens que ele errou.
        if caso.get("expected_tool"):
            total_com_tool += 1
            if caso["expected_tool"] in r.tools:
                acertos_tool += 1

    latencias = sorted(r.latency_ms for r in linhas)
    def _percentil(p: float) -> float:
        if not latencias:
            return 0.0
        return latencias[min(len(latencias) - 1, int(round(p * (len(latencias) - 1))))]

    return {
        "n": len(casos),
        "acuracia_rota": acertos_rota / len(casos) if casos else 0.0,
        "precision_at_k": acertos_tool / total_com_tool if total_com_tool else None,
        "n_com_tool": total_com_tool,
        "custo_total_usd": sum(r.cost_usd for r in linhas),
        "custo_por_mensagem_usd": sum(r.cost_usd for r in linhas) / len(linhas) if linhas else 0,
        "latencia_p50_ms": statistics.median(latencias) if latencias else 0,
        "latencia_p95_ms": _percentil(0.95),
        "tokens_entrada": sum(r.prompt_tokens for r in linhas),
        "tokens_saida": sum(r.completion_tokens for r in linhas),
        "tokens_raciocinio": sum(r.reasoning_tokens for r in linhas),
        "tokens_cacheados": sum(r.cached_tokens for r in linhas),
        "chamadas_llm": sum(r.llm_calls for r in linhas),
        "erros": sum(1 for r in linhas if r.error),
        "linhas": [r.as_dict() for r in linhas],
    }


def rodar_bakeoff(modelos: List[str], incluir_v1: bool = True) -> Dict:
    """Roda a V1 e cada modelo candidato da V1.0.1 sobre o mesmo conjunto."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("Este comparativo precisa de OPENAI_API_KEY em App/.env.")

    dataset = load_eval_dataset()
    resultados: Dict[str, Dict] = {}

    if incluir_v1:
        print("Preparando V1 (ML classico)...")
        v1 = V1ClassicPipeline(settings).setup()
        print("  rodando dataset oficial...")
        oficial = avaliar(v1, dataset)
        print("  rodando casos criticos...")
        criticos = avaliar(v1, CASOS_CRITICOS)
        resultados["V1 (ML classico)"] = {
            "modelo": "scikit-learn (local)",
            "oficial": oficial,
            "criticos": criticos,
            "fallbacks": 0,
        }
        _imprimir_linha("V1 (ML classico)", oficial, criticos)

    for modelo in modelos:
        print(f"\nPreparando V1.0.1 com {modelo}...")
        if modelo in MODELOS_DESCONTINUADOS:
            print(f"  AVISO: {modelo} tem desligamento em {MODELOS_DESCONTINUADOS[modelo]}")
        # Sobrescrevemos a variavel de ambiente e recarregamos a configuracao.
        # E o jeito mais simples de trocar so o modelo mantendo todo o resto
        # exatamente igual entre as rodadas.
        os.environ["OPENAI_MODEL_ORCHESTRATOR"] = modelo
        cfg = get_settings()
        pipeline = V101OrchestratorPipeline(cfg).setup()

        print("  rodando dataset oficial...")
        oficial = avaliar(pipeline, dataset)
        print("  rodando casos criticos...")
        criticos = avaliar(pipeline, CASOS_CRITICOS)
        resultados[f"V1.0.1 / {modelo}"] = {
            "modelo": modelo,
            "reasoning_effort": cfg.orchestrator_reasoning_effort,
            "oficial": oficial,
            "criticos": criticos,
            "fallbacks": pipeline.fallbacks,
            "custo_desconhecido": pipeline.custo_desconhecido,
            "descontinuado_em": MODELOS_DESCONTINUADOS.get(modelo),
        }
        _imprimir_linha(f"V1.0.1 / {modelo}", oficial, criticos, pipeline.fallbacks)

    return resultados


def _imprimir_linha(nome: str, oficial: Dict, criticos: Dict, fallbacks: int = 0) -> None:
    p2 = oficial["precision_at_k"]
    print(
        f"  -> rota {oficial['acuracia_rota']:.0%} | P@2 {p2:.0%} | "
        f"criticos {criticos['acuracia_rota']:.0%}/{criticos['precision_at_k']:.0%} | "
        f"${oficial['custo_por_mensagem_usd']:.6f}/msg | "
        f"p50 {oficial['latencia_p50_ms']:.0f}ms"
        + (f" | fallbacks={fallbacks}" if fallbacks else "")
    )


def imprimir_tabela(resultados: Dict) -> None:
    """Tabela final, ordenada pelo acerto nos casos criticos."""
    print("\n" + "=" * 108)
    print("COMPARATIVO DE MODELOS — orquestrador da V1.0.1")
    print("=" * 108)
    cab = (f"{'Versao / modelo':26s} {'rota':>6s} {'P@2':>6s} {'crit.P@2':>9s} "
           f"{'$/msg':>10s} {'$/1M msg':>10s} {'p50 ms':>8s} {'p95 ms':>8s} {'racioc.':>8s}")
    print(cab); print("-" * 108)
    for nome, d in resultados.items():
        o, c = d["oficial"], d["criticos"]
        print(f"{nome:26s} {o['acuracia_rota']:6.0%} {o['precision_at_k']:6.0%} "
              f"{c['precision_at_k']:9.0%} {o['custo_por_mensagem_usd']:10.6f} "
              f"{o['custo_por_mensagem_usd']*1_000_000:10.0f} "
              f"{o['latencia_p50_ms']:8.0f} {o['latencia_p95_ms']:8.0f} "
              f"{o['tokens_raciocinio']:8d}")
    print("=" * 108)
    print("crit.P@2 = acerto de ferramenta nos casos que a V1 erra (o teste que importa)")
    print("$/1M msg = projecao do custo mensal a 1 milhao de mensagens")
    print("racioc.  = tokens de raciocinio invisiveis, cobrados como saida")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara modelos no papel de orquestrador.")
    parser.add_argument("--models", nargs="*", default=MODELOS_PADRAO)
    parser.add_argument("--out", default="App/reports/model_bakeoff.json")
    parser.add_argument("--sem-v1", action="store_true", help="pula a linha de base V1")
    args = parser.parse_args()

    resultados = rodar_bakeoff(args.models, incluir_v1=not args.sem_v1)
    imprimir_tabela(resultados)

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatorio completo salvo em: {destino}")


if __name__ == "__main__":
    main()
