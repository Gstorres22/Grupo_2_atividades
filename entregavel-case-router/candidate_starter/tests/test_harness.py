"""Testes das funcoes de METRICA do harness.

Por que estes testes existem: o argumento central desta entrega e "as decisoes
foram tomadas com base em medicao". Se o codigo que MEDE estiver errado, todo o
resto desaba junto. Ironicamente, era a unica parte sem teste.

O foco esta nos casos DEGENERADOS — lista vazia, proporcao 0%, proporcao 100%,
divisao por zero. Sao onde formula de metrica costuma quebrar, e sao justamente
os que nao aparecem numa execucao normal do dataset.
"""
import pytest

from candidate_starter.harness import (
    compute_mrr_at_k,
    compute_precision_at_k,
    compute_router_metrics,
    compute_savings,
    latency_stats,
    wilson_interval,
)

LABELS = ["FAST_PATH", "AGENT"]


# ---------------------------------------------------------------- acuracia
def test_router_metrics_matriz_e_acuracia():
    """A matriz de confusao deve ser {real: {predito: contagem}}."""
    y_true = ["FAST_PATH", "FAST_PATH", "AGENT", "AGENT"]
    y_pred = ["FAST_PATH", "AGENT", "AGENT", "AGENT"]
    m = compute_router_metrics(y_true, y_pred, LABELS)

    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"]["FAST_PATH"]["FAST_PATH"] == 1
    assert m["confusion_matrix"]["FAST_PATH"]["AGENT"] == 1
    assert m["confusion_matrix"]["AGENT"]["AGENT"] == 2
    assert m["confusion_matrix"]["AGENT"]["FAST_PATH"] == 0


def test_router_metrics_erro_assimetrico():
    """As duas taxas de erro devem ser contadas SEPARADAMENTE.

    Mandar AGENT para FAST_PATH e falha de atendimento; o inverso so gasta
    dinheiro. Uma acuracia unica esconde essa diferenca.
    """
    y_true = ["AGENT", "AGENT", "FAST_PATH", "FAST_PATH"]
    y_pred = ["FAST_PATH", "AGENT", "AGENT", "FAST_PATH"]
    err = compute_router_metrics(y_true, y_pred, LABELS)["error_analysis"]

    assert err["agent_sent_to_fast_path"] == 1
    assert err["agent_sent_to_fast_path_rate"] == 0.5
    assert err["fast_path_sent_to_agent"] == 1
    assert err["fast_path_sent_to_agent_rate"] == 0.5


def test_router_metrics_classe_ausente_nao_quebra():
    """Se uma classe nunca aparece, a divisao por zero nao pode estourar."""
    m = compute_router_metrics(["AGENT"] * 3, ["AGENT"] * 3, LABELS)
    assert m["accuracy"] == 1.0
    assert m["per_class"]["FAST_PATH"]["support"] == 0
    assert m["per_class"]["FAST_PATH"]["precision"] == 0.0


def test_router_metrics_rejeita_entradas_invalidas():
    with pytest.raises(ValueError):
        compute_router_metrics(["AGENT"], ["AGENT", "AGENT"], LABELS)
    with pytest.raises(ValueError):
        compute_router_metrics([], [], LABELS)


# --------------------------------------------------------------- precision
def test_precision_at_k():
    assert compute_precision_at_k([1, 1, 0, 0]) == 0.5
    assert compute_precision_at_k([1, 1]) == 1.0
    assert compute_precision_at_k([0, 0]) == 0.0


def test_precision_at_k_lista_vazia_devolve_zero():
    """Sem nada avaliado, a resposta e 0.0 — nunca uma excecao no meio do lote."""
    assert compute_precision_at_k([]) == 0.0


# --------------------------------------------------------------------- MRR
def test_mrr_usa_a_posicao():
    """Acertar na 1a posicao vale 1; na 2a, 0,5. E o que diferencia MRR de P@K."""
    assert compute_mrr_at_k([1, 1]) == 1.0
    assert compute_mrr_at_k([2, 2]) == 0.5
    assert compute_mrr_at_k([1, 2]) == 0.75


def test_mrr_trata_none_como_erro():
    """None = a tool correta ficou fora do top-k. Contribui zero, nao quebra."""
    assert compute_mrr_at_k([1, None]) == 0.5
    assert compute_mrr_at_k([None, None]) == 0.0
    assert compute_mrr_at_k([]) == 0.0


# ---------------------------------------------------------------- economia
def test_savings_calcula_percentual():
    s = compute_savings(smart_cost_usd=25.0, smart_latency_ms=50.0,
                        baseline_cost_usd=100.0, baseline_latency_ms=100.0)
    assert s["cost_savings_pct"] == 75.0
    assert s["latency_savings_pct"] == 50.0


def test_savings_com_baseline_zero_nao_estoura():
    """Guarda contra divisao por zero: sem baseline nao ha economia a reportar."""
    s = compute_savings(0.0, 0.0, 0.0, 0.0)
    assert s["cost_savings_pct"] == 0.0
    assert s["latency_savings_pct"] == 0.0


def test_savings_aceita_economia_negativa():
    """Se o pipeline 'inteligente' custar MAIS, o numero tem de ficar negativo.

    Zerar aqui esconderia uma regressao — exatamente o que a metrica existe
    para revelar.
    """
    s = compute_savings(200.0, 0.0, 100.0, 100.0)
    assert s["cost_savings_pct"] == -100.0


# ------------------------------------------------------------------ Wilson
def test_wilson_em_100_por_cento_nao_chega_a_1():
    """O motivo de usarmos Wilson em vez da aproximacao normal.

    Com 30/30 acertos, a formula normal daria intervalo [1,0; 1,0] — certeza
    absoluta a partir de 30 amostras, o que e falso. Wilson devolve um limite
    inferior honesto, em torno de 0,88.
    """
    ic = wilson_interval(30, 30)
    assert ic["high"] == 1.0
    assert 0.85 < ic["low"] < 0.90


def test_wilson_em_zero_por_cento():
    ic = wilson_interval(0, 30)
    assert ic["low"] == 0.0
    assert 0.0 < ic["high"] < 0.15


def test_wilson_sem_amostra():
    assert wilson_interval(0, 0) == {"low": 0.0, "high": 0.0}


def test_wilson_encolhe_com_mais_amostra():
    """Mais dados, intervalo mais estreito. Se isso quebrar, a formula esta errada."""
    largura_30 = wilson_interval(15, 30)["high"] - wilson_interval(15, 30)["low"]
    largura_300 = wilson_interval(150, 300)["high"] - wilson_interval(150, 300)["low"]
    assert largura_300 < largura_30


# ---------------------------------------------------------------- latencia
def test_latency_stats():
    s = latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["p50"] == 30.0
    assert s["max"] == 50.0
    assert s["mean"] == 30.0
    assert s["p95"] >= s["p50"]


def test_latency_stats_lista_vazia():
    assert latency_stats([]) == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}


def test_latency_p95_pega_a_cauda():
    """A media esconde a cauda; o p95 tem de enxerga-la. E o motivo de reportar os dois.

    Usamos 10% de outliers (e nao 5%) de proposito: com exatamente 5% o p95 cai
    na fronteira, e o valor depende da convencao de arredondamento do percentil.
    Um teste nao deve depender dessa escolha.
    """
    valores = [1.0] * 90 + [1000.0] * 10
    s = latency_stats(valores)
    assert s["p50"] == 1.0
    assert s["p95"] > 100.0
    assert s["mean"] < 200.0  # a media dilui a cauda; o p95 nao
