"""Executor em lote — roda muitas queries pelo pipeline e coleta resultados.

Serve para teste de estresse, teste de robustez e para avaliar o conjunto
ampliado. Funciona OFFLINE (modo lexical) e tambem no modo cascata, sem mudar
nada na chamada.

USO
---
    # Entrada: JSON com uma lista de objetos. So `query` e obrigatorio.
    #   [{"query": "quero meu saldo", "expected_route": "AGENT",
    #     "expected_tool": "consultar_saldo", "persona": "leigo"}]
    python -m App.eval.run_batch entrada.json --out resultado.json

    # Ou passando queries direto na linha de comando:
    python -m App.eval.run_batch --query "meu carta o nao funciona" --query "oi"

SAIDA
-----
JSON com uma linha por query (rota, confianca, tools, latencia, custo, acertos)
e um bloco `summary` com as metricas agregadas — incluindo recorte por persona,
quando o campo `persona` estiver presente na entrada.

POR QUE ESTE ARQUIVO EXISTE
---------------------------
O `run_case.py` do nucleo avalia o dataset oficial e nao deve ser alterado.
Este executor e a porta para tudo que NAO e o dataset oficial: queries com erro
de digitacao, fora de escopo, ambiguas, ou geradas por persona. Ele nao substitui
o harness — reaproveita as mesmas funcoes de metrica de `candidate_starter/harness.py`
para os numeros serem comparaveis.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from App.core.config import get_settings
from App.core.embeddings import build_dense_provider
from App.core.escalation import LLMEscalator
from candidate_starter.harness import (
    compute_precision_at_k,
    compute_router_metrics,
    latency_stats,
    wilson_interval,
)
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data, load_tools


def build_components(use_llm: bool = True):
    """Monta router + retriever (+ estagio vetorial, se houver chave)."""
    settings = get_settings()
    tools = load_tools()
    texts, labels = load_router_training_data()

    router = QueryRouter(confidence_threshold=settings.router_confidence_threshold).fit(texts, labels)
    retriever = ToolRetriever()
    if use_llm:
        retriever.set_dense_provider(build_dense_provider(settings))
    retriever.fit(tools)
    escalator = LLMEscalator(settings) if use_llm else None
    return settings, router, retriever, tools, escalator


def run_batch(items: List[dict], k: int = 2, use_llm: bool = True) -> dict:
    settings, router, retriever, tools, escalator = build_components(use_llm)

    rows: List[dict] = []
    y_true: List[str] = []
    y_pred: List[str] = []
    hits: List[int] = []
    latencies: List[float] = []
    by_persona: Dict[str, List[dict]] = defaultdict(list)

    for item in items:
        query = item["query"]
        route_result = router.predict(query)
        route = route_result.route
        escalated = False

        # Cascata do router: so escala se o classificador estiver inseguro.
        if escalator and escalator.active and router.is_uncertain(route_result):
            resolved = escalator.resolve_route(query, route_result, is_uncertain=True)
            escalated = resolved.route != route
            route = resolved.route

        row = {
            "query": query,
            "persona": item.get("persona"),
            "route": route,
            "confidence": route_result.confidence,
            "route_escalated": escalated,
            "latency_ms": route_result.latency_ms,
            "tools": [],
        }

        if route == "AGENT":
            result = retriever.search(query, k=k)
            row["tools"] = [m.name for m in result.matches]
            row["retrieval_margin"] = ToolRetriever.margin(result)
            row["latency_ms"] += result.latency_ms

        # Comparacoes so acontecem quando a entrada trouxe gabarito.
        expected_route = item.get("expected_route")
        expected_tool = item.get("expected_tool")
        if expected_route:
            y_true.append(expected_route)
            y_pred.append(route)
            row["expected_route"] = expected_route
            row["route_ok"] = route == expected_route
        if expected_tool and route == "AGENT":
            hit = int(expected_tool in row["tools"])
            hits.append(hit)
            row["expected_tool"] = expected_tool
            row["tool_ok"] = bool(hit)

        latencies.append(row["latency_ms"])
        rows.append(row)
        if item.get("persona"):
            by_persona[item["persona"]].append(row)

    summary: dict = {
        "n_queries": len(rows),
        "mode": "cascata (LLM ligado)" if (escalator and escalator.active) else "local (lexico)",
        "config": settings.describe(),
        "latency_ms": latency_stats(latencies),
        "route_distribution": {
            "FAST_PATH": sum(1 for r in rows if r["route"] == "FAST_PATH"),
            "AGENT": sum(1 for r in rows if r["route"] == "AGENT"),
        },
    }
    if y_true:
        metrics = compute_router_metrics(y_true, y_pred, ["FAST_PATH", "AGENT"])
        summary["router_accuracy"] = metrics["accuracy"]
        summary["router_accuracy_ci95"] = metrics["accuracy_ci95"]
        summary["confusion_matrix"] = metrics["confusion_matrix"]
        summary["router_error_analysis"] = metrics["error_analysis"]
    if hits:
        summary[f"precision_at_{k}"] = compute_precision_at_k(hits)
        summary["precision_ci95"] = wilson_interval(sum(hits), len(hits))
        summary["n_tool_evaluated"] = len(hits)

    # Recorte por persona: mostra se o sistema quebra com um perfil especifico.
    if by_persona:
        summary["by_persona"] = {}
        for persona, persona_rows in sorted(by_persona.items()):
            scored = [r for r in persona_rows if "route_ok" in r]
            tool_scored = [r for r in persona_rows if "tool_ok" in r]
            summary["by_persona"][persona] = {
                "n": len(persona_rows),
                "router_accuracy": (
                    sum(r["route_ok"] for r in scored) / len(scored) if scored else None
                ),
                "precision_at_k": (
                    sum(r["tool_ok"] for r in tool_scored) / len(tool_scored) if tool_scored else None
                ),
                "latency_p95_ms": latency_stats([r["latency_ms"] for r in persona_rows])["p95"],
            }

    if escalator and escalator.active:
        summary["escalation"] = escalator.telemetry.as_dict()

    return {"summary": summary, "rows": rows}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Roda um lote de queries pelo pipeline.")
    parser.add_argument("input", nargs="?", help="JSON com a lista de queries")
    parser.add_argument("--query", action="append", default=[], help="query avulsa (repetivel)")
    parser.add_argument("--out", help="caminho do JSON de saida")
    parser.add_argument("-k", type=int, default=2, help="quantas tools retornar (padrao 2)")
    parser.add_argument("--no-llm", action="store_true", help="forca o modo 100%% local")
    args = parser.parse_args(argv)

    items: List[dict] = []
    if args.input:
        items.extend(json.loads(Path(args.input).read_text(encoding="utf-8")))
    items.extend({"query": q} for q in args.query)
    if not items:
        parser.error("informe um arquivo JSON de entrada ou pelo menos um --query")

    result = run_batch(items, k=args.k, use_llm=not args.no_llm)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Resultado salvo em: {args.out}")

    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.out:
        for row in result["rows"]:
            flag = ""
            if "route_ok" in row:
                flag = "OK " if row["route_ok"] else "ERRO"
            print(f"{flag:4s} [{row['route']:9s}] {row['query'][:52]:54s} {row['tools'][:2]}")


if __name__ == "__main__":
    main(sys.argv[1:])
