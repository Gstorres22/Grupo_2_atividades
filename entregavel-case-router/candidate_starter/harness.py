"""Pilar 3 — Evaluation Harness (Evals & Benchmarking).

Metricas obrigatorias do case:
  1. Acuracia do Router (+ matriz de confusao).
  2. Precision@K do Retriever de Tools.
  3. Economia de custo e latencia do pipeline "inteligente" vs. baseline.

===============================================================================
POR QUE MEDIMOS MAIS DO QUE O MINIMO PEDIDO
===============================================================================

O conjunto de avaliacao tem 30 queries (20 com tool esperada). Nesse tamanho,
uma metrica pontual engana com facilidade: a diferenca entre 29/30 e 30/30 e
uma unica linha. Por isso o relatorio traz, alem do obrigatorio:

  * INTERVALO DE CONFIANCA (Wilson) — "100% de acuracia" em 30 amostras tem
    intervalo de ~88% a 100%. Reportar so o ponto seria vender certeza que o
    dado nao sustenta. Usamos Wilson (e nao a formula normal) porque ela
    continua valida quando a proporcao encosta em 0 ou 1, que e exatamente o
    nosso caso.

  * ERRO ASSIMETRICO — os dois erros do router NAO custam a mesma coisa:
      - AGENT classificado como FAST_PATH: o cliente pede o saldo e recebe uma
        resposta de prateleira. E falha de atendimento, o erro grave.
      - FAST_PATH classificado como AGENT: gasta-se uma chamada de LLM a toa.
        Custa dinheiro, mas o cliente e atendido.
    Uma acuracia unica esconde essa diferenca; reportamos as duas taxas.

  * MRR@k — Precision@K responde "acertou?"; MRR responde "acertou em que
    posicao?". Acertar na 1a posicao permite executar a tool direto; acertar na
    2a exige que o LLM escolha. Sao qualidades diferentes.

  * LATENCIA p50/p95 — media esconde cauda. SLA de atendimento se mede no p95.

  * SUCESSO PONTA A PONTA — rota certa E tool certa no top-k. Ver a ressalva
    importante sobre isso na secao seguinte.

===============================================================================
DUAS RESSALVAS HONESTAS SOBRE A MEDICAO (nao sao bugs, sao vieses do desenho)
===============================================================================

RESSALVA 1 — Precision@K e CONDICIONAL ao acerto do router.
  `run_harness` so chama o retriever quando o router decidiu AGENT. Se o router
  mandar erroneamente uma query AGENT para o FAST_PATH, essa query simplesmente
  nao entra no calculo de Precision@K. Ou seja: um router ruim pode INFLAR o
  Precision@K, porque as queries que ele errou saem da conta. Por isso
  reportamos tambem `end_to_end_success_rate`, que conta a query como falha
  nesse cenario.

RESSALVA 2 — a economia de latencia obrigatoria compara coisas diferentes.
  No caminho "inteligente", `run_harness` soma apenas router + retrieval. No
  baseline, soma a chamada simulada de LLM. Como a chamada de LLM e ordens de
  grandeza mais lenta, a economia obrigatoria sai perto de 99% — um numero
  verdadeiro, mas que compara "so a decisao" contra "decisao + resposta".
  Mantivemos a metrica obrigatoria como o case pede E adicionamos
  `latency_savings_pct_comparable`, que inclui a chamada de LLM tambem no
  caminho inteligente. Essa segunda e a comparacao maca-com-maca.
"""
import statistics
import time
from math import sqrt
from typing import Dict, List, Optional

from common.interfaces import BaseRouter, BaseToolRetriever
from common.mock_llm import (
    COST_RETRIEVAL_USD,
    COST_ROUTER_USD,
    fast_path_answer,
    mock_tool_execution,
    simulate_agent_llm_call,
    simulate_baseline_llm_call,
)
from common.schemas import Tool


# =============================================================================
# METRICAS OBRIGATORIAS
# =============================================================================
def compute_router_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Acuracia e matriz de confusao do router.

    A matriz e montada a mao (em vez de `sklearn.metrics.confusion_matrix`)
    porque o formato pedido e um dict aninhado rotulado
    `{real: {predito: contagem}}`, muito mais legivel no JSON do relatorio do
    que um array cujo significado depende da ordem das classes.

    Extras (chaves adicionais, nao usadas por `run_harness`, mas gravadas no
    relatorio): precision/recall/F1 por classe, intervalo de confianca e as
    duas taxas de erro assimetrico descritas no cabecalho do modulo.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")
    if not y_true:
        raise ValueError("Nao ha exemplos para avaliar.")

    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    correct = sum(matrix[label][label] for label in labels)
    total = len(y_true)
    accuracy = correct / total

    per_class = {}
    for label in labels:
        true_positive = matrix[label][label]
        false_negative = sum(matrix[label][p] for p in labels if p != label)
        false_positive = sum(matrix[a][label] for a in labels if a != label)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }

    # Erro grave: era AGENT (precisa de tool) e foi para resposta de prateleira.
    agent_support = per_class.get("AGENT", {}).get("support", 0)
    fast_support = per_class.get("FAST_PATH", {}).get("support", 0)
    missed_agent = matrix.get("AGENT", {}).get("FAST_PATH", 0)
    wasted_llm = matrix.get("FAST_PATH", {}).get("AGENT", 0)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "accuracy_ci95": wilson_interval(correct, total),
        "error_analysis": {
            "agent_sent_to_fast_path": missed_agent,
            "agent_sent_to_fast_path_rate": (missed_agent / agent_support) if agent_support else 0.0,
            "fast_path_sent_to_agent": wasted_llm,
            "fast_path_sent_to_agent_rate": (wasted_llm / fast_support) if fast_support else 0.0,
            "comment": (
                "agent_sent_to_fast_path e o erro grave (cliente nao e atendido); "
                "fast_path_sent_to_agent so desperdica custo de LLM."
            ),
        },
    }


def compute_precision_at_k(hits: List[int]) -> float:
    """Precision@K a partir de uma lista de 0/1 (a tool esperada estava no top-k?).

    Com um unico item relevante por query, Precision@K equivale a taxa de
    acerto no top-k (tambem chamada de Hit Rate@K ou Recall@K neste cenario).
    Mantemos o nome que o case pede.
    """
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def compute_savings(
    smart_cost_usd: float,
    smart_latency_ms: float,
    baseline_cost_usd: float,
    baseline_latency_ms: float,
) -> Dict:
    """Percentual de economia do pipeline inteligente em relacao ao baseline.

    A guarda contra baseline zerado evita ZeroDivisionError caso o harness rode
    com um dataset vazio ou com os custos do mock zerados: nesse caso nao ha
    economia a reportar, e 0.0 e a resposta correta.
    """
    def _saving(smart: float, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return (baseline - smart) / baseline * 100.0

    return {
        "cost_savings_pct": _saving(smart_cost_usd, baseline_cost_usd),
        "latency_savings_pct": _saving(smart_latency_ms, baseline_latency_ms),
    }


# =============================================================================
# METRICAS DE APOIO
# =============================================================================
def wilson_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    """Intervalo de confianca de 95% para uma proporcao (metodo de Wilson).

    Preferido a aproximacao normal porque continua correto quando a proporcao
    encosta em 0 ou 1 — situacao comum aqui, com o router acertando tudo.
    """
    if total == 0:
        return {"low": 0.0, "high": 0.0}
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return {"low": max(0.0, center - spread), "high": min(1.0, center + spread)}


def compute_mrr_at_k(ranks: List[Optional[int]]) -> float:
    """MRR a partir da posicao (1-indexada) da tool correta; None = fora do top-k."""
    if not ranks:
        return 0.0
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def latency_stats(values: List[float]) -> Dict[str, float]:
    """p50/p95/media/max de uma lista de latencias em ms."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
    ordered = sorted(values)
    index_95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[index_95],
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


# =============================================================================
# ORQUESTRACAO
# =============================================================================
def run_harness(
    router: BaseRouter,
    retriever: BaseToolRetriever,
    tools: List[Tool],
    eval_dataset: List[dict],
    k: int = 2,
) -> dict:
    labels = ["FAST_PATH", "AGENT"]

    y_true: List[str] = []
    y_pred: List[str] = []
    precision_hits: List[int] = []
    gold_ranks: List[Optional[int]] = []          # posicao da tool correta (para MRR)
    router_latencies: List[float] = []
    retrieval_latencies: List[float] = []
    end_to_end_hits: List[int] = []               # rota certa E tool certa

    smart_cost_total = 0.0
    smart_latency_ms_total = 0.0
    smart_latency_comparable_total = 0.0          # inclui a chamada de LLM (ver ressalva 2)
    baseline_cost_total = 0.0
    baseline_latency_ms_total = 0.0

    rows = []

    for item in eval_dataset:
        query = item["query"]
        expected_route = item["expected_route"]
        expected_tool = item.get("expected_tool")

        route_result = router.predict(query)
        y_true.append(expected_route)
        y_pred.append(route_result.route)
        router_latencies.append(route_result.latency_ms)

        smart_cost = COST_ROUTER_USD
        smart_latency_ms = route_result.latency_ms
        smart_latency_comparable = route_result.latency_ms

        row = {
            "query": query,
            "expected_route": expected_route,
            "predicted_route": route_result.route,
            "router_confidence": route_result.confidence,
        }

        route_is_correct = route_result.route == expected_route

        if route_result.route == "FAST_PATH":
            fast_path_answer(query)
            # Query que precisava de tool mas foi para resposta de prateleira:
            # nao aparece no Precision@K (ressalva 1), mas conta como falha aqui.
            if expected_tool:
                end_to_end_hits.append(0)
        else:
            retrieval_result = retriever.search(query, k=k)
            smart_cost += COST_RETRIEVAL_USD
            smart_latency_ms += retrieval_result.latency_ms
            smart_latency_comparable += retrieval_result.latency_ms
            retrieval_latencies.append(retrieval_result.latency_ms)

            top_k_names = [m.name for m in retrieval_result.matches]
            if expected_tool:
                hit = int(expected_tool in top_k_names)
                precision_hits.append(hit)
                gold_ranks.append(top_k_names.index(expected_tool) + 1 if hit else None)
                end_to_end_hits.append(int(hit and route_is_correct))

            row["retrieved_tools"] = top_k_names
            row["expected_tool"] = expected_tool

            if top_k_names:
                mock_tool_execution(top_k_names[0], query)
                agent_start = time.perf_counter()
                llm_result = simulate_agent_llm_call(query, top_k_names[0])
                smart_latency_comparable += (time.perf_counter() - agent_start) * 1000
                smart_cost += llm_result["cost_usd"]

        smart_cost_total += smart_cost
        smart_latency_ms_total += smart_latency_ms
        smart_latency_comparable_total += smart_latency_comparable

        baseline_start = time.perf_counter()
        baseline_result = simulate_baseline_llm_call(query)
        baseline_latency_ms_total += (time.perf_counter() - baseline_start) * 1000
        baseline_cost_total += baseline_result["cost_usd"]

        rows.append(row)

    router_metrics = compute_router_metrics(y_true, y_pred, labels)
    precision_at_k = compute_precision_at_k(precision_hits) if precision_hits else None
    savings = compute_savings(
        smart_cost_total, smart_latency_ms_total, baseline_cost_total, baseline_latency_ms_total
    )
    # Comparacao maca-com-maca da latencia: inclui a chamada de LLM nos DOIS lados.
    # Calculada direto (e nao via compute_savings) porque aqui so a latencia importa.
    latency_savings_comparable = (
        (baseline_latency_ms_total - smart_latency_comparable_total) / baseline_latency_ms_total * 100.0
        if baseline_latency_ms_total > 0 else 0.0
    )

    n_queries = len(eval_dataset)
    report = {
        "n_queries": n_queries,
        "router_accuracy": router_metrics["accuracy"],
        "confusion_matrix": router_metrics["confusion_matrix"],
        "precision_at_k": precision_at_k,
        "k": k,
        "smart_pipeline": {
            "total_cost_usd": smart_cost_total,
            "total_latency_ms": smart_latency_ms_total,
            "cost_per_query_usd": smart_cost_total / n_queries if n_queries else 0.0,
        },
        "baseline_always_llm": {
            "total_cost_usd": baseline_cost_total,
            "total_latency_ms": baseline_latency_ms_total,
            "cost_per_query_usd": baseline_cost_total / n_queries if n_queries else 0.0,
        },
        **savings,
        # --- metricas adicionais (ver cabecalho do modulo) ---
        "router_detail": {
            "per_class": router_metrics["per_class"],
            "accuracy_ci95": router_metrics["accuracy_ci95"],
            "error_analysis": router_metrics["error_analysis"],
        },
        "retriever_detail": {
            "mrr_at_k": compute_mrr_at_k(gold_ranks),
            "n_evaluated": len(precision_hits),
            "note": (
                "Precision@K so considera queries que o router mandou para AGENT "
                "(ver Ressalva 1 no cabecalho de harness.py)."
            ),
        },
        "end_to_end_success_rate": (
            sum(end_to_end_hits) / len(end_to_end_hits) if end_to_end_hits else None
        ),
        "latency_detail_ms": {
            "router": latency_stats(router_latencies),
            "retrieval": latency_stats(retrieval_latencies),
        },
        "latency_savings_pct_comparable": latency_savings_comparable,
        "smart_pipeline_latency_comparable_ms": smart_latency_comparable_total,
        "rows": rows,
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("HARNESS DE AVALIAÇÃO - Router & Tool Retrieval")
    print("=" * 60)
    print(f"Queries avaliadas: {report['n_queries']}")
    print(f"Acurácia do Router: {report['router_accuracy']:.1%}")
    ci = report.get("router_detail", {}).get("accuracy_ci95")
    if ci:
        print(f"  IC 95% (Wilson): [{ci['low']:.1%}, {ci['high']:.1%}]")
    print(f"Matriz de confusão: {report['confusion_matrix']}")
    errors = report.get("router_detail", {}).get("error_analysis")
    if errors:
        print(f"  AGENT -> FAST_PATH (erro grave): {errors['agent_sent_to_fast_path']} "
              f"({errors['agent_sent_to_fast_path_rate']:.1%})")
        print(f"  FAST_PATH -> AGENT (custo à toa): {errors['fast_path_sent_to_agent']} "
              f"({errors['fast_path_sent_to_agent_rate']:.1%})")
    if report["precision_at_k"] is not None:
        print(f"Precision@{report['k']} do Retriever: {report['precision_at_k']:.1%} "
              f"(n={report['retriever_detail']['n_evaluated']})")
        print(f"  MRR@{report['k']}: {report['retriever_detail']['mrr_at_k']:.3f}")
    if report.get("end_to_end_success_rate") is not None:
        print(f"Sucesso ponta a ponta (rota + tool): {report['end_to_end_success_rate']:.1%}")
    print("-" * 60)
    print(f"Custo pipeline inteligente: ${report['smart_pipeline']['total_cost_usd']:.5f}")
    print(f"Custo baseline (tudo pro LLM): ${report['baseline_always_llm']['total_cost_usd']:.5f}")
    print(f"Economia de custo: {report.get('cost_savings_pct', 0):.1f}%")
    print(f"Latência pipeline inteligente: {report['smart_pipeline']['total_latency_ms']:.1f} ms")
    print(f"Latência baseline: {report['baseline_always_llm']['total_latency_ms']:.1f} ms")
    print(f"Economia de latência: {report.get('latency_savings_pct', 0):.1f}%")
    if "latency_savings_pct_comparable" in report:
        print(f"  (comparação maçã-com-maçã, incluindo a chamada de LLM nos dois lados: "
              f"{report['latency_savings_pct_comparable']:.1f}%)")
    latency = report.get("latency_detail_ms", {})
    if latency:
        print(f"  Router p50/p95: {latency['router']['p50']:.3f} / {latency['router']['p95']:.3f} ms")
        if latency["retrieval"]["mean"] > 0:
            print(f"  Retrieval p50/p95: {latency['retrieval']['p50']:.3f} / "
                  f"{latency['retrieval']['p95']:.3f} ms")
    print("=" * 60)
