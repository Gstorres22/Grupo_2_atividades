"""Testes do Pilar 1 — roteador de queries.

Alem do contrato, um teste registra o CONTRAEXEMPLO que motivou nao usar regra
de palavra-chave. Ele existe para que ninguem, no futuro, "simplifique" o
roteador trocando o classificador por um `startswith("bom dia")`.
"""
import pytest

from candidate_starter.router import AGENT, FAST_PATH, QueryRouter

TEXTOS = [
    "Bom dia, qual o horario de atendimento?",
    "Oi, tudo bem?",
    "Obrigado pela ajuda",
    "Vocês cobram taxa de manutencao?",
    "Quero saber meu saldo",
    "Preciso bloquear meu cartao perdido",
    "Qual o valor da minha fatura?",
    "Quero parcelar a fatura do cartao",
]
ROTULOS = [FAST_PATH, FAST_PATH, FAST_PATH, FAST_PATH, AGENT, AGENT, AGENT, AGENT]


# ------------------------------------------------------------------ contrato
def test_predict_antes_de_fit_levanta_erro():
    with pytest.raises(RuntimeError):
        QueryRouter().predict("bom dia")


def test_fit_devolve_self_para_encadear():
    r = QueryRouter()
    assert r.fit(TEXTOS, ROTULOS) is r


def test_fit_rejeita_tamanhos_diferentes():
    with pytest.raises(ValueError):
        QueryRouter().fit(TEXTOS, ROTULOS[:-1])


def test_fit_rejeita_conjunto_vazio():
    with pytest.raises(ValueError):
        QueryRouter().fit([], [])


def test_fit_rejeita_rotulo_invalido():
    """Um rotulo digitado errado ("AGENTE") tem de falhar no fit, nao virar uma
    terceira classe silenciosa que so aparece no relatorio."""
    with pytest.raises(ValueError):
        QueryRouter().fit(TEXTOS, [FAST_PATH] * 7 + ["AGENTE"])


def test_predict_devolve_rota_valida_e_confianca():
    r = QueryRouter().fit(TEXTOS, ROTULOS)
    resultado = r.predict("bom dia")
    assert resultado.route in {FAST_PATH, AGENT}
    assert resultado.latency_ms >= 0
    assert 0.0 <= resultado.confidence <= 1.0


def test_classes_expostas_apos_fit():
    r = QueryRouter()
    assert r.classes is None
    r.fit(TEXTOS, ROTULOS)
    assert set(r.classes) == {FAST_PATH, AGENT}


# ---------------------------------------------------- comportamento aprendido
def test_separa_os_dois_casos_obvios():
    r = QueryRouter().fit(TEXTOS, ROTULOS)
    assert r.predict("Oi, tudo bem?").route == FAST_PATH
    assert r.predict("Quero saber meu saldo").route == AGENT


def test_saudacao_com_pedido_acionavel_vai_para_agent():
    """CONTRAEXEMPLO que justifica NAO usar regra de palavra-chave (ADR-04).

    Uma regra "comeca com saudacao -> FAST_PATH" mandaria esta mensagem para a
    resposta de prateleira, e o cliente ficaria sem o saldo. O classificador
    pondera a frase inteira.

    Se alguem trocar o modelo por regras, este teste quebra — que e o objetivo.
    """
    r = QueryRouter().fit(TEXTOS, ROTULOS)
    assert r.predict("Bom dia, preciso do meu saldo").route == AGENT


def test_tolera_erro_de_digitacao():
    """Motivo de usarmos n-gramas de CARACTERE (ADR-02): a rota nao pode mudar
    porque o cliente digitou rapido no celular."""
    r = QueryRouter().fit(TEXTOS, ROTULOS)
    assert r.predict("Quero saber meu saldo").route == r.predict("qero sabr meu sald").route


# ------------------------------------------------------------- auxiliares
def test_is_uncertain_respeita_o_limiar():
    r = QueryRouter(confidence_threshold=0.99).fit(TEXTOS, ROTULOS)
    # Com limiar praticamente maximo, quase tudo vira "incerto".
    assert r.is_uncertain(r.predict("bom dia")) is True

    r_permissivo = QueryRouter(confidence_threshold=0.01).fit(TEXTOS, ROTULOS)
    assert r_permissivo.is_uncertain(r_permissivo.predict("bom dia")) is False


def test_explain_lista_termos_com_peso():
    """Auditabilidade: precisa dar para explicar QUALQUER decisao em producao."""
    r = QueryRouter().fit(TEXTOS, ROTULOS)
    termos = r.explain("Quero saber meu saldo", top_n=5)
    assert 0 < len(termos) <= 5
    assert all(isinstance(t, str) and isinstance(p, float) for t, p in termos)
    # Ordenado por relevancia absoluta decrescente.
    pesos = [abs(p) for _, p in termos]
    assert pesos == sorted(pesos, reverse=True)


def test_explain_antes_de_fit_levanta_erro():
    with pytest.raises(RuntimeError):
        QueryRouter().explain("bom dia")
