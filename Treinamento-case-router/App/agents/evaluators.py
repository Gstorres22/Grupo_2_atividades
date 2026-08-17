"""2 agentes avaliadores de Engenharia de IA.

===============================================================================
POR QUE DOIS AVALIADORES COM LENTES DIFERENTES
===============================================================================

Dois avaliadores com o MESMO prompt dariam praticamente a mesma resposta duas
vezes — gastando o dobro e sem cobrir nada a mais. A redundancia so vira valor
quando as perspectivas sao distintas.

Entao cada um recebe uma lente fechada e complementar:

    avaliador_metrico     -> "os numeros sustentam a conclusao?"
                             Rigor estatistico, vies de medicao, vazamento,
                             tamanho de amostra, se a comparacao e justa.

    avaliador_producao    -> "isso aguenta o mundo real?"
                             Latencia, custo em escala, modos de falha,
                             seguranca, dependencia de terceiro, operacao.

Sao os dois jeitos de uma decisao de arquitetura dar errado: (a) os numeros
mentem, ou (b) os numeros estao certos mas o sistema quebra em producao. Um
avaliador so cobriria metade.

===============================================================================
COMO EVITAMOS QUE O AVALIADOR PUXE A SARDINHA
===============================================================================

1. **Modelo diferente do avaliado.** Os avaliadores usam `OPENAI_MODEL_AGENTS`,
   nunca `OPENAI_MODEL_ORCHESTRATOR`. LLMs tendem a preferir as proprias
   saidas — se o mesmo modelo escolhesse a ferramenta e depois julgasse a
   escolha, a nota seria inflada.

2. **Instrucao explicita de ceticismo.** O prompt manda procurar motivos para
   a V1.0.1 NAO valer a pena. O caminho de menor esforco de um LLM e concordar
   com a narrativa que recebe; pedir o contrario compensa esse vies.

3. **Nomes neutros nos dados.** As versoes chegam identificadas apenas por
   nome, sem adjetivo. Nada de "a versao nova e melhorada".

4. **Saida estruturada com nota obrigatoria.** Forcar um numero de 0 a 10 por
   dimensao impede a resposta morna que elogia tudo sem se comprometer.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from App.agents.base import Agent, AgentRun, run_parallel
from App.core.config import Settings, get_settings

CONTEXTO_COMUM = """## O SISTEMA

Um "cerebro de roteamento" para o atendimento de um banco digital. Para cada
mensagem do cliente ele decide:

- "FAST_PATH": saudacao ou FAQ generica -> resposta pronta local, sem LLM
- "AGENT": precisa de ferramenta -> seleciona 2 de um catalogo de 285, depois
  chama um LLM caro para executar

O catalogo tem quase-duplicatas propositais: para varias mensagens, a ferramenta
mais parecida lexicalmente NAO e a correta.

## AS VERSOES COMPARADAS

- **V1**: decisao 100% local. Classificador scikit-learn (TF-IDF de n-gramas de
  caractere + Regressao Logistica, treinado com 53 exemplos) decide a rota;
  busca hibrida (lexical + embeddings, fundidos por Reciprocal Rank Fusion, com
  um fator que penaliza ferramentas hiper-especificas) escolhe as ferramentas.
  Custo de decisao ~US$ 0.

- **V1.0.1**: um LLM pequeno decide a rota E escolhe as ferramentas, em UMA
  unica chamada. O estagio local de recuperacao continua existindo e reduz 285
  para 20 candidatas antes do LLM entrar — o LLM nunca ve o catalogo inteiro.

- **V1.0.2**: cascata hibrida. O classificador local roda primeiro; se ele
  disser FAST_PATH com confianca acima do limiar E a mensagem for parecida com
  algum exemplo de treino ("familiaridade"), a resposta sai local e o LLM NAO e
  chamado. Todo o resto vai para o orquestrador da V1.0.1, que e literalmente o
  mesmo codigo. Os limiares foram calibrados por validacao cruzada APENAS nos
  53 exemplos de treino, sem consultar nenhum conjunto de teste.

  Ponto de atencao para a sua analise: a economia da cascata e proporcional a
  fracao de FAST_PATH no trafego. O conjunto das personas e pesado em AGENT
  (113 de 150 mensagens tem ferramenta esperada), entao a taxa de desvio
  observada SUBESTIMA o ganho que a cascata teria num atendimento real. Por
  outro lado, a calibracao por validacao cruzada num conjunto de treino
  linguisticamente estreito e OTIMISTA por construcao — ela nao consegue
  enxergar mudanca de registro, que e justamente o modo de falha da V1.

## LIMITACOES CONHECIDAS DOS DADOS

- O `eval_dataset.json` oficial tem 30 mensagens com rotulos HUMANOS, mas 40%
  delas tem sobreposicao alta com os 53 exemplos de treino da V1 (3 sao copias
  literais). Esse vazamento favorece a V1.
- O conjunto gerado pelas personas tem rotulos de LLM ("prata"), nao humanos.
  Foi gerado por um modelo da OpenAI, e a V1.0.1 tambem usa OpenAI — ha risco
  de vies compartilhado favorecendo a V1.0.1."""

PROMPT_METRICO = f"""Voce e um Engenheiro de IA senior especializado em AVALIACAO E METRICAS.

{CONTEXTO_COMUM}

## SUA LENTE

Voce NAO opina sobre arquitetura ou custo. Sua unica pergunta e:
**os numeros apresentados sustentam a conclusao que se quer tirar deles?**

Investigue com ceticismo:
1. A comparacao e JUSTA? As duas versoes receberam as mesmas mensagens, nas
   mesmas condicoes? Alguma diferenca alem do componente sob teste?
2. O tamanho da amostra sustenta as diferencas observadas? Uma diferenca de N
   pontos percentuais em M mensagens pode ser ruido — diga quando for.
3. Ha vazamento de dados favorecendo alguma das versoes?
4. Os rotulos "prata" (gerados por LLM) foram tratados com o ceticismo devido,
   ou estao sendo somados aos "ouro" como se valessem o mesmo?
5. Alguma metrica esta definida de um jeito que INFLA o resultado? Preste
   atencao especial a metricas condicionais (ex.: Precision@K calculado apenas
   sobre mensagens que o roteador ja acertou — isso remove da conta justamente
   os casos dificeis).
6. O que NAO foi medido e deveria ter sido?

IMPORTANTE: procure ativamente motivos para NAO acreditar na conclusao. Se os
numeros forem solidos, diga — mas so depois de tentar derruba-los."""

PROMPT_PRODUCAO = f"""Voce e um Engenheiro de IA senior especializado em SISTEMAS EM PRODUCAO.

{CONTEXTO_COMUM}

## SUA LENTE

Voce NAO reanalisa estatistica. Sua unica pergunta e:
**este sistema aguenta o mundo real, e a que preco?**

Investigue com ceticismo:
1. LATENCIA. A V1 decide em milissegundos; a V1.0.1 depende de rede. O que isso
   significa para o cliente esperando no chat? E para o p95, nao so a media?
2. CUSTO EM ESCALA. Projete para 100 mil e 1 milhao de mensagens/mes. O que
   parece irrelevante por mensagem vira o que por mes?
3. MODOS DE FALHA. A V1.0.1 depende de um provedor externo. O que acontece
   quando ele fica lento, indisponivel, ou limita a taxa? O plano B e adequado?
   Um plano B que devolve a resposta da V1 significa que, sob falha, o sistema
   volta a ter os bugs da V1 — isso e aceitavel?
4. SEGURANCA. Passar a mensagem do cliente para dentro de um prompt abre
   superficie para manipulacao (prompt injection). O que pode dar errado quando
   a ferramenta escolhida movimenta dinheiro?
5. OPERACAO. O que precisa ser monitorado? O que quebra silenciosamente?
   Como se percebe uma regressao de qualidade em producao?
6. DEPENDENCIA DE TERCEIRO. Modelos sao descontinuados. Qual o custo de trocar?

IMPORTANTE: procure ativamente o que quebra. Se a arquitetura for solida, diga —
mas so depois de tentar derruba-la."""

FORMATO_SAIDA = """

## FORMATO DA RESPOSTA

Responda SOMENTE com JSON valido:

{
  "notas": {
    "<dimensao_1>": {"nota": 0-10, "justificativa": "..."},
    "<dimensao_2>": {"nota": 0-10, "justificativa": "..."}
  },
  "achados": [
    {"gravidade": "critico" | "alto" | "medio" | "baixo",
     "titulo": "frase curta",
     "descricao": "o problema, com o numero ou evidencia que o sustenta",
     "recomendacao": "o que fazer"}
  ],
  "pontos_fortes": ["..."],
  "recomendacao_final": "V1" | "V1.0.1" | "hibrido" | "inconclusivo",
  "justificativa_final": "2 a 4 frases explicando a recomendacao",
  "confianca": 0.0 a 1.0,
  "o_que_falta_medir": ["..."]
}

Use de 3 a 6 dimensoes nas notas, escolhidas por voce dentro da SUA lente.
Liste de 3 a 8 achados, ordenados do mais grave para o menos grave.
Escreva tudo em portugues do Brasil."""


AVALIADORES = [
    ("avaliador_metrico", PROMPT_METRICO + FORMATO_SAIDA),
    ("avaliador_producao", PROMPT_PRODUCAO + FORMATO_SAIDA),
]


def montar_dossie(comparacao: Dict, max_falhas: int = 12) -> str:
    """Monta o texto que os avaliadores recebem.

    Por que resumir em vez de mandar as 300 linhas: o avaliador precisa das
    METRICAS e de uma AMOSTRA de falhas, nao do log inteiro. Mandar tudo
    aumentaria o custo e diluiria o sinal no meio do ruido.

    A amostra de falhas e escolhida por DIVERGENCIA — casos em que uma versao
    acertou e a outra errou. Sao os unicos que carregam informacao sobre a
    diferenca entre as duas; onde ambas acertam ou ambas erram, nao ha o que
    comparar.
    """
    partes: List[str] = ["# DADOS DA COMPARACAO\n"]

    for bloco, titulo in [("oficial", "Dataset oficial (rotulos HUMANOS, n=30)"),
                          ("personas", "Conjunto das personas (rotulos de LLM)")]:
        dados = comparacao.get(bloco)
        if not dados:
            continue
        partes.append(f"\n## {titulo}\n")
        for versao, m in dados.get("metricas", {}).items():
            partes.append(
                f"- **{versao}**: acuracia de rota {m.get('acuracia_rota', 0):.1%}, "
                f"Precision@2 {m.get('precision_at_k') or 0:.1%}, "
                f"custo US$ {m.get('custo_por_mensagem_usd', 0):.6f}/msg, "
                f"latencia p50 {m.get('latencia_p50_ms', 0):.0f}ms / "
                f"p95 {m.get('latencia_p95_ms', 0):.0f}ms, "
                f"n={m.get('n', 0)}"
            )
        por_persona = dados.get("por_persona")
        if por_persona:
            partes.append("\n### Recorte por persona\n")
            for persona, valores in por_persona.items():
                partes.append(f"- {persona}: " + ", ".join(
                    f"{v}={dados_v.get('acuracia_rota', 0):.0%}/"
                    f"{(dados_v.get('precision_at_k') or 0):.0%}"
                    for v, dados_v in valores.items()
                ) + "  (formato: rota/P@2)")

    # Placar par a par: quantas vezes cada versao ganhou da outra.
    # Vem antes da amostra porque e o dado QUANTITATIVO; a amostra e ilustrativa.
    divergencias = comparacao.get("divergencias", [])
    if divergencias:
        partes.append("\n## Placar de divergencias (quantitativo, todas as mensagens)\n")
        for par in sorted({d["par"] for d in divergencias}):
            do_par = [d for d in divergencias if d["par"] == par]
            a, b = par.split(" x ")
            partes.append(
                f"- **{par}**: {len(do_par)} divergencias | "
                f"so {b} acertou: {sum(1 for d in do_par if d['vantagem_b'] == 1)} | "
                f"so {a} acertou: {sum(1 for d in do_par if d['vantagem_b'] == -1)} | "
                f"ambas certas ou ambas erradas: {sum(1 for d in do_par if d['vantagem_b'] == 0)}"
            )

        # Amostra ilustrativa, priorizando os casos decisivos (uma acertou, a
        # outra nao) — sao os unicos que informam sobre a diferenca.
        amostra = [d for d in divergencias if d["vantagem_b"] != 0][:max_falhas]
        if amostra:
            partes.append(f"\n## Amostra de casos decisivos ({len(amostra)} de "
                          f"{sum(1 for d in divergencias if d['vantagem_b'] != 0)})\n")
            partes.append("_Selecao: apenas casos em que uma versao acertou e a outra errou._\n")
            for d in amostra:
                linha = [f"- [{d['par']}] {d.get('query','')[:88]!r}",
                         f"  esperado: rota={d.get('expected_route')} tool={d.get('expected_tool')}"]
                for chave, valor in d.items():
                    if chave.endswith("_route"):
                        versao = chave[:-6]
                        linha.append(f"  {versao}: rota={valor} tools={d.get(versao + '_tools')} "
                                     f"ok={d.get(versao + '_ok')}")
                partes.append("\n".join(linha))

    telemetria = comparacao.get("telemetria")
    if telemetria:
        partes.append("\n## Telemetria das versoes\n")
        for versao, dados in telemetria.items():
            partes.append(f"- **{versao}**: " + ", ".join(f"{k}={v}" for k, v in dados.items()
                                                          if k != "nota"))

    extra = comparacao.get("observacoes")
    if extra:
        partes.append("\n## Observacoes do time\n" + "\n".join(f"- {o}" for o in extra))

    return "\n".join(partes)


def avaliar(
    comparacao: Dict,
    settings: Optional[Settings] = None,
    model: Optional[str] = None,
) -> List[AgentRun]:
    """Roda os 2 avaliadores em paralelo sobre o mesmo dossie."""
    settings = settings or get_settings()
    dossie = montar_dossie(comparacao)

    agentes = [
        Agent(
            name=nome,
            system_prompt=prompt,
            settings=settings,
            model=model,
            # Zero: um julgamento deve ser reprodutivel. Se rodar duas vezes e
            # der notas diferentes, a nota nao significa nada.
            temperature=0.0,
        )
        for nome, prompt in AVALIADORES
    ]
    return run_parallel([lambda a=a: a.run(dossie) for a in agentes], max_workers=2)
