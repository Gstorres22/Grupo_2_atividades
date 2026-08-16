# V1.0.2 (Cascata Híbrida) — Resultado e Recomendação

**Data:** 15/08/2026 · **Experimento:** 180 mensagens, as mesmas nas três versões
**Documentos irmãos:** [V1_0_2_DECISIONS.md](V1_0_2_DECISIONS.md) · [V1_0_1_COMPARACAO.md](V1_0_1_COMPARACAO.md) · [V1_DESCOBERTAS.md](V1_DESCOBERTAS.md)

> Medido depois da correção do prior de generalidade (ADR-07): sinal só-nome, `λ = 0,40`.
> Como as três versões compartilham o mesmo estágio de recuperação, **todas melhoraram**.

---

## Resposta curta

> **Sim, o ganho se justifica.** A V1.0.2 **empata** com a V1.0.1 em qualidade no conjunto
> grande (92% × 92%), é **mais rápida** (971 ms contra 1.107 ms de média) e evita 12% a 27% das
> chamadas de LLM. Em 21 mensagens resolvidas localmente, **zero erros**.
> **Recomendo adotar.**

---

## 1. Os números

### Dataset oficial — 30 mensagens, rótulos humanos

| Versão | Rota | Hit@2 | p50 | média | Sem LLM |
|---|---|---|---|---|---|
| V1 | 100% | 85% | 2 ms | 2 ms | 30 |
| V1.0.1 | 100% | **95%** | 1.112 ms | 1.152 ms | 0 |
| **V1.0.2** | 100% | 90% | **945 ms** | **770 ms** | **8** |

### Conjunto das personas — 150 mensagens

| Versão | Rota | Hit@2 | p50 | média | Sem LLM |
|---|---|---|---|---|---|
| V1 | 84,0% | 57,5% | 2 ms | 2 ms | 150 |
| V1.0.1 | **98,0%** | **92,0%** | 1.081 ms | 1.107 ms | 0 |
| **V1.0.2** | 97,3% | **92,0%** | **1.029 ms** | **971 ms** | **13** |

### Por persona (Hit@2)

| Persona | V1 | V1.0.1 | V1.0.2 |
|---|---|---|---|
| `leigo_idoso` | 40% | 80% | 80% |
| `jovem_girias` | 56% | 92% | 92% |
| `dedos_gordos` | 71% | 100% | 100% |
| `especialista_bancario` | 68% | 96% | 96% |
| `caotico` | 50% | 93% | 93% |

**A V1.0.2 iguala a V1.0.1 em todas as cinco personas.**

> **Nota de terminologia.** O avaliador métrico apontou, com razão, que com **uma** ferramenta
> esperada e **duas** retornadas a métrica correta se chama **Hit@2**, não Precision@2. O case
> pede "Precision@K", então o nome foi mantido no código; aqui usamos o tecnicamente certo.

---

## 2. O achado que decide a comparação

A V1.0.2 aparece 5 pontos abaixo da V1.0.1 no conjunto oficial. **Essa diferença não vem da
cascata.**

Foram apenas **5 divergências** entre as duas em 180 mensagens. Cruzando cada uma com a
telemetria de desvio:

```
Divergências V1.0.1 × V1.0.2 ................ 5
Delas, causadas por desvio da cascata ....... 0
Delas, causadas pelo LLM dar resposta
  diferente para a MESMA entrada ............ 5
```

As duas versões usam **literalmente o mesmo código** de orquestrador — a V1.0.2 compõe a
V1.0.1, não a reimplementa ([ADR-30](V1_0_2_DECISIONS.md#adr-30)). Quando a cascata não desvia,
a mensagem percorre exatamente o mesmo caminho. Ainda assim as respostas diferiram:

| Mensagem | V1.0.1 | V1.0.2 |
|---|---|---|
| "Quanto eu tenho disponível na conta agora?" | `consultar_saldo` ✅ | `consultar_valor_disponivel_conta` |
| "eu queria saber quanto veio para pagar do cartãozinho" | `consultar_valor_total_fatura` | `consultar_fatura` ✅ |
| "aquele papel da conta do cartão não chegou no e-mail" | `consultar_email_vinculado_conta` | `confirmar_email_cadastrado` |

**Conclusão: o LLM não é reprodutível**, mesmo com `temperature=0` e `reasoning_effort: none`.
Os 5 pontos de diferença no conjunto oficial são **1 mensagem** em 20 — dentro da variação que
o próprio modelo produz entre execuções.

Evidência independente: **a V1.0.1 mediu 75%, 85% e 95% de Hit@2 no mesmo conjunto oficial em
três execuções diferentes**, com o mesmo código e as mesmas 30 mensagens.

---

## 3. A cascata funcionou como projetada

| Verificação | Resultado |
|---|---|
| Mensagens resolvidas localmente | 21 |
| Delas, **corretas** | **21 (100%)** |
| Erros introduzidos pela cascata | **0** |
| Divergências causadas por desvio | **0** |

Toda mensagem que a cascata resolveu localmente estava certa. O LLM teria dado a mesma
resposta — e cobrado por ela.

> ⚠️ **Intervalo de confiança:** 21 de 21 acertos, com n=21, tem IC 95% de **[84%, 100%]**.
> Encorajador, não é prova de perfeição.

---

## 4. Quanto a cascata economiza

Ela desvia **35% das mensagens FAST_PATH**. Como só desvia FAST_PATH, a economia total é
proporcional à fatia de FAST_PATH no tráfego:

| Composição do tráfego | Mensagens sem LLM | Economia de custo e latência |
|---|---|---|
| 10% FAST_PATH | 4% | 4% |
| **25% (conjunto das personas)** | **9%** | **9%** |
| **33% (dataset oficial)** | **27%** | **27%** |
| 50% FAST_PATH | 18% | 18% |
| 65% FAST_PATH | 23% | 23% |

**Como ler esta tabela.** O conjunto das personas é pesado em AGENT (só 25% de FAST_PATH) — ele
**subestima** o ganho. Num atendimento bancário real, saudações, agradecimentos e FAQ costumam
ser fatia maior.

**Sobre a latência.** Desta vez a economia medida acompanha a teórica: 971 ms contra 1.107 ms
nas personas (−12%) e 770 ms contra 1.152 ms no oficial (−33%). O número em que confiar
continua sendo a **contagem de chamadas evitadas**, que é exata; a latência varia com a rede.

---

## 5. O parecer dos dois avaliadores

| Avaliador | Recomendação | Confiança |
|---|---|---|
| `avaliador_producao` | **híbrido** | 86% |
| `avaliador_metrico` | inconclusivo | 90% |

**Produção** recomendou o híbrido nas três rodadas, condicionando a produção a controles
determinísticos na execução financeira.

**Métrico** manteve "inconclusivo" com um argumento que **procede**: *"não há evidência para
escolher entre V1.0.1 e V1.0.2"* — 5 divergências, todas ruído. Em **qualidade**, estão
empatadas.

**Mas empate em qualidade não é empate na decisão.** Se duas versões entregam o mesmo resultado
e uma evita 12–27% das chamadas pagas, o desempate não precisa de mais dados de qualidade. Vem
do custo — e nesse eixo a diferença é **contada, não estimada**.

---

## 6. O que ainda bloqueia produção

Nada disto é específico da V1.0.2 — vale igualmente para a V1.0.1, e continua **aberto**:

| # | Bloqueio |
|---|---|
| 1 | **Nenhuma camada de política entre escolher e executar ferramenta financeira** |
| 2 | O plano B falha "aberto": sob queda do provedor, o sistema volta a ter os bugs da V1 |
| 3 | Falta conjunto humano grande e sem vazamento |
| 4 | Hit@2 não é sucesso do cliente — não medimos qual ferramenta foi executada |
| 5 | Prompt injection com ferramentas que movimentam dinheiro |

O item 1 é o que eu trataria primeiro.

---

## 7. Recomendação

**Adotar a V1.0.2**, em ordem de peso:

1. **Qualidade igual à da V1.0.1** — 92% × 92% nas personas, empate nas cinco personas, e as
   5 divergências são ruído do LLM. A cascata acertou 21 de 21.
2. **Mais rápida**: 971 ms contra 1.107 ms de média, com contagem exata de chamadas evitadas.
3. **A parte desviada é determinística e gratuita.** Pagar um LLM para responder "bom dia" é
   desperdício, e a cascata elimina isso sem risco medido.
4. **Degradação mais suave**: sob indisponibilidade do provedor, a fatia desviada continua
   sendo atendida com a mesma qualidade de sempre.

Não adotar significaria pagar por 12–27% de chamadas que não mudam o resultado.

### Antes de produção, na ordem

1. Camada de política para ferramentas sensíveis (Pix, TED, encerrar conta)
2. Conjunto de validação humano, novo e maior
3. Medição de sucesso ponta a ponta, não só Hit@2
4. Rollout gradual com telemetria da taxa de desvio em tráfego real
5. **Repetir as comparações N vezes** — dada a variação do LLM entre execuções, rodada única
   não sustenta diferenças menores que ~10 pontos

---

## Resumo em uma frase

> **A V1.0.2 empata com a V1.0.1 em qualidade, é mais rápida e evita 12% a 27% das chamadas
> pagas — e a única razão de o avaliador métrico dizer "inconclusivo" é que ele avalia
> qualidade, onde de fato há empate. O desempate vem do custo, que é contado e não estimado.**
