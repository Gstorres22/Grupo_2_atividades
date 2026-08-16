"""Testes do Pilar 2 — busca de ferramentas.

Cobrem tres coisas: o CONTRATO (o que a interface promete), os casos
DEGENERADOS (entrada vazia, k maior que o catalogo, catalogo homogeneo) e o
COMPORTAMENTO que da identidade a solucao — o prior de generalidade.

O teste do prior e o mais importante do arquivo: e ele que garante que uma
mudanca futura no ranking nao desfaca silenciosamente a correcao que leva o
Precision@2 de 0,25 para 0,85.
"""
import numpy as np
import pytest

from candidate_starter.retrieval import ToolRetriever, _reciprocal_rank
from common.data_loader import load_tools
from common.schemas import RetrievalResult, Tool, ToolMatch

CATALOGO_MINIMO = [
    Tool(name="consultar_saldo",
         description="Consulta o saldo bancario da conta corrente do cliente.",
         category="financeiro"),
    Tool(name="consultar_valor_disponivel_conta",
         description="Consulta o valor disponivel agora na conta corrente do cliente.",
         category="financeiro"),
    Tool(name="bloquear_cartao",
         description="Bloqueia o cartao de credito ou debito do cliente.",
         category="cartao"),
]


# ------------------------------------------------------------------ contrato
def test_search_antes_de_fit_levanta_erro():
    with pytest.raises(RuntimeError):
        ToolRetriever().search("qualquer coisa")


def test_fit_com_catalogo_vazio_levanta_erro():
    with pytest.raises(ValueError):
        ToolRetriever().fit([])


def test_fit_devolve_self_para_encadear():
    """`ToolRetriever().fit(tools)` precisa funcionar como uma expressao unica."""
    r = ToolRetriever()
    assert r.fit(CATALOGO_MINIMO) is r


def test_search_respeita_o_k():
    r = ToolRetriever().fit(load_tools())
    for k in (1, 2, 5):
        assert len(r.search("quero meu saldo", k=k).matches) == k


def test_k_maior_que_o_catalogo_nao_estoura():
    """Pedir mais ferramentas do que existem devolve todas, sem excecao."""
    r = ToolRetriever().fit(CATALOGO_MINIMO)
    assert len(r.search("saldo", k=99).matches) == len(CATALOGO_MINIMO)


def test_k_invalido_levanta_erro():
    r = ToolRetriever().fit(CATALOGO_MINIMO)
    with pytest.raises(ValueError):
        r.search("saldo", k=0)


def test_latencia_e_medida_e_positiva():
    r = ToolRetriever().fit(CATALOGO_MINIMO)
    assert r.search("saldo", k=2).latency_ms >= 0


# ------------------------------------------------- prior de generalidade
@pytest.mark.parametrize("query,esperado", [
    # Cada caso foi VERIFICADO: erra com lambda=0 e acerta com o padrao.
    # Sao os que o prior de generalidade de fato corrige.
    ("Manda o pdf da minha fatura atual", "consultar_fatura"),
    ("Quero parcelar minha fatura em 3 vezes", "parcelar_fatura"),
    ("Preciso saber o saldo disponivel pra pix", "consultar_saldo"),
    ("O aplicativo esta travando, quero abrir um chamado", "abrir_chamado_suporte"),
])
def test_prior_prefere_a_ferramenta_canonica(query, esperado):
    """O comportamento que define a solucao.

    Nestes casos existe uma variante hiper-especifica cujo nome quase copia a
    frase do cliente — por exemplo `enviar_pdf_fatura_atual` para "manda o pdf
    da minha fatura". Sem o prior, a isca vence. Com ele, a ferramenta canonica
    entra no top-2.

    Se estes testes quebrarem, o Precision@2 caiu de 0,85 para perto de 0,45.
    """
    r = ToolRetriever().fit(load_tools())
    assert esperado in [m.name for m in r.search(query, k=2).matches]


def test_prior_nao_resolve_tudo():
    """Documenta um limite CONHECIDO, para nao virar surpresa depois.

    "Quanto eu tenho disponivel na conta agora?" continua errando: a isca
    `consultar_valor_disponivel_conta` tem nome de 4 tokens, curto o suficiente
    para o prior nao a derrubar. E uma das 3 falhas restantes das 20 queries.

    Se algum dia este teste falhar (ou seja, o caso passar a acertar), otimo —
    mas queremos saber, porque significa que o ranking mudou.
    """
    r = ToolRetriever().fit(load_tools())
    nomes = [m.name for m in r.search("Quanto eu tenho disponivel na conta agora?", k=2).matches]
    assert "consultar_saldo" not in nomes, (
        "Este caso passou a acertar — atualize o teste e o numero de falhas conhecidas."
    )


def test_prior_desligado_muda_o_ranking():
    """Com lambda=0 o prior nao atua — confirma que ele e mesmo a causa do efeito."""
    tools = load_tools()
    query = "Manda o pdf da minha fatura atual"
    com = [m.name for m in ToolRetriever(generality_lambda=0.40).fit(tools).search(query, k=2).matches]
    sem = [m.name for m in ToolRetriever(generality_lambda=0.0).fit(tools).search(query, k=2).matches]
    assert com != sem
    assert "consultar_fatura" in com


def test_especificidade_com_catalogo_homogeneo_e_zero():
    """Se todos os nomes tem o mesmo tamanho, o sinal nao discrimina nada.

    A guarda contra divisao por zero (max == min) precisa devolver zeros, nao NaN.
    """
    homogeneo = [
        Tool(name="consultar_saldo", description="a", category="x"),
        Tool(name="bloquear_cartao", description="b", category="y"),
    ]
    r = ToolRetriever().fit(homogeneo)
    assert np.allclose(r._specificity, 0.0)
    assert not np.isnan(r._specificity).any()


# ------------------------------------------------------------------- RRF
def test_rrf_converte_posicao_em_peso_decrescente():
    """O 1o colocado recebe mais que o 2o, que recebe mais que o 3o."""
    pesos = _reciprocal_rank(np.array([0.9, 0.5, 0.1]))
    assert pesos[0] > pesos[1] > pesos[2]


def test_rrf_e_imune_a_escala():
    """A razao de ser do RRF: so a ORDEM importa, nunca a magnitude.

    Dois vetores com escalas totalmente diferentes, mas a mesma ordem, devem
    produzir contribuicoes identicas. E isso que permite fundir TF-IDF com
    similaridade de embedding sem normalizar nada.
    """
    a = _reciprocal_rank(np.array([0.9, 0.5, 0.1]))
    b = _reciprocal_rank(np.array([900.0, 500.0, 100.0]))
    assert np.allclose(a, b)


def test_rrf_com_empates_nao_quebra():
    pesos = _reciprocal_rank(np.array([0.5, 0.5, 0.5]))
    assert len(pesos) == 3
    assert np.all(pesos > 0)


# ---------------------------------------------------------------- margem
def test_margem_e_relativa_ao_primeiro():
    r = RetrievalResult(matches=[ToolMatch("a", 1.0), ToolMatch("b", 0.5)], latency_ms=0)
    assert ToolRetriever.margin(r) == 0.5


def test_margem_com_uma_unica_candidata():
    """Sem segundo colocado nao ha ambiguidade: margem maxima."""
    r = RetrievalResult(matches=[ToolMatch("a", 1.0)], latency_ms=0)
    assert ToolRetriever.margin(r) == 1.0


def test_margem_com_score_zero_nao_divide_por_zero():
    r = RetrievalResult(matches=[ToolMatch("a", 0.0), ToolMatch("b", 0.0)], latency_ms=0)
    assert ToolRetriever.margin(r) == 0.0


# ---------------------------------------------- estagio denso (opcional)
def test_sem_provider_denso_a_busca_continua_funcionando():
    """O nucleo TEM de rodar offline. E a promessa da arquitetura em camadas."""
    r = ToolRetriever().set_dense_provider(None).fit(CATALOGO_MINIMO)
    assert len(r.search("saldo", k=2).matches) == 2


def test_provider_denso_injetado_e_usado():
    """Com provider registrado, o indice denso e construido."""
    def provider_falso(textos):
        # Vetores deterministicos, so para exercitar o caminho de codigo.
        return np.array([[len(t) % 7, len(t) % 5, 1.0] for t in textos], dtype=float)

    r = ToolRetriever().set_dense_provider(provider_falso).fit(CATALOGO_MINIMO)
    assert r._matrix_dense is not None
    assert len(r.search("saldo", k=2).matches) == 2
