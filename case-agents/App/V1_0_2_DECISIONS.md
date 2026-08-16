# Registro de Decisões — V1.0.2 (Cascata Híbrida)

> Continuação de [DECISIONS.md](DECISIONS.md) (ADR-01 a 14, a V1) e
> [V1_0_1_DECISIONS.md](V1_0_1_DECISIONS.md) (ADR-15 a 26, a V1.0.1).
> Mesmo formato: **contexto → decisão → alternativas → evidência → consequência**.

---

## Sumário

| # | Decisão | Veredito |
|---|---|---|
| [27](#adr-27) | A cascata só desvia FAST_PATH | Adotado |
| [28](#adr-28) | Dois sinais que precisam concordar | Adotado |
| [29](#adr-29) | **Limiar de familiaridade por critério relativo** | Corrigido após falha |
| [30](#adr-30) | V1.0.2 compõe a V1.0.1 em vez de reimplementar | Adotado |
| [31](#adr-31) | Reaproveitar o conjunto de personas | Adotado |
| [32](#adr-32) | **O LLM não é reprodutível** | Achado, não decisão |

---

<a name="adr-27"></a>
## ADR-27 — A cascata só desvia FAST_PATH

**Contexto.** A cascata poderia também desviar mensagens AGENT quando a busca local parecesse
segura, o que aumentaria muito a economia.

**Decisão.** Só FAST_PATH é desviado. Todo AGENT vai para o LLM.

**Por quê — três razões medidas:**

1. **O problema da V1 não está na rota, está na ferramenta.** Nas personas: rota 84%,
   escolha de ferramenta 54,9%. Desviar um AGENT significa aceitar a escolha de ferramenta da
   V1 — ou seja, herdar exatamente o que a V1.0.1 veio consertar.
2. **O sinal natural de confiança do buscador discrimina mal.** Medimos: 68% das falhas ficam
   abaixo do limiar de margem, mas **54% dos acertos também**. Como detector de "estou em
   dúvida", é quase ruído.
3. **FAST_PATH não tem escolha de ferramenta.** A resposta é uma string pronta. Desviar ali não
   arrisca qualidade de ferramenta — só a rota, que é a parte que a V1 faz bem.

**Consequência.** A economia fica limitada à fatia de FAST_PATH do tráfego. É um teto
arquitetural assumido, não um descuido.

---

<a name="adr-28"></a>
## ADR-28 — Dois sinais que precisam concordar

**Decisão.** Uma mensagem só pula o LLM se passar em **dois** testes: `confiança ≥ 0,65`
**e** `familiaridade ≥ 0,418`.

**Por que não só a confiança.** Medimos a precisão da confiança nas 150 mensagens de persona e
ela **não é monotônica**:

| Faixa de confiança | Precisão |
|---|---|
| [0,60 – 0,70) | 92% |
| [0,70 – 0,80) | **71%** ← confiança maior, precisão menor |
| [0,80 – 0,90) | 100% |

Um modelo treinado com 53 exemplos não produz probabilidade bem calibrada. Ele consegue soar
confiante sobre uma mensagem de um tipo que nunca viu — foi assim que a V1 mandou
`"solicito a emissão do informe de rendimentos"` para FAST_PATH com **confiança 0,754**.

**O que a familiaridade acrescenta.** É uma detecção simples de "fora da distribuição": se a
mensagem não se parece com nada do treino, o modelo não tem base para opinar, por mais
confiante que soe. É o antídoto direto para o modo de falha que derrubou a V1.

---

<a name="adr-29"></a>
## ADR-29 — Limiar de familiaridade por critério relativo

**Este ADR registra um erro que quase entrou em produção, e como foi corrigido.**

### A primeira tentativa, que falhou

Calibramos o limiar por validação cruzada 5-fold sobre os 53 exemplos de treino — método
correto, sem tocar nos conjuntos de teste. Resultado: `familiaridade ≥ 0,20`, com **100% de
precisão** e 68% de cobertura. Números excelentes.

**E errados.** No teste de fumaça, `"solicito a emissão do informe de rendimentos"` foi
**desviada**: respondida localmente como FAST_PATH, sem o LLM ser consultado. É exatamente o
bug que a V1.0.2 existe para evitar.

### Por que falhou

| Mensagem | Familiaridade |
|---|---|
| `"bom dia"` | 1,000 |
| `"qual o horário de atendimento?"` | 0,769 |
| `"solicito a emissão do informe de rendimentos"` | **0,347** |
| *mediana da familiaridade interna do treino* | *0,336* |

**N-grama de caractere mede sobreposição de letras, não de registro linguístico.** Duas frases
em português compartilham muitos trigramas mesmo com vocabulário totalmente diferente — então a
frase formal parecia **mais familiar que metade dos próprios exemplos de treino**.

E a validação cruzada **não tinha como detectar isso, por construção**: todas as dobras vêm da
mesma distribuição estreita. Ela mede generalização *dentro* do registro, nunca *entre*
registros. Foi o mesmo ponto cego que produziu a V1.

### A correção

Limiar por critério **relativo**, ainda derivado só do treino:

> A mensagem precisa se parecer com algum exemplo de treino **mais do que os exemplos de treino
> se parecem entre si**.

Implementado como o **percentil 90** da distribuição de similaridade interna do treino:

```
p25 = 0,275   p50 = 0,336   p75 = 0,368   p90 = 0,418   máx = 0,502
```

Limiar = **0,418**. A frase formal (0,347) não passa; `"bom dia"` (1,000) passa. O teste de
fumaça foi de 5/6 para 6/6.

**Validação posterior:** na bateria completa, **21 de 21 desvios corretos**, zero erros.

**Lição registrada.** Validação cruzada não detecta mudança de distribuição. Quando se sabe que
a distribuição de produção é mais larga que a de treino — e aqui sabíamos, estava documentado
no [V1_DESCOBERTAS.md](V1_DESCOBERTAS.md) — o critério tem de ser relativo à estrutura do
próprio treino, não absoluto.

---

<a name="adr-30"></a>
## ADR-30 — A V1.0.2 compõe a V1.0.1 em vez de reimplementar

**Decisão.** `V102HybridPipeline` contém uma instância de `V101OrchestratorPipeline` e delega a
ela tudo que não for desviado.

**Por quê.** Se a V1.0.2 tivesse a própria cópia da lógica do orquestrador, qualquer diferença
medida entre as duas poderia vir de uma variação acidental de implementação — um prompt levemente
diferente, um parâmetro esquecido. Compondo, o caminho do LLM é **literalmente o mesmo código**,
então a única diferença possível é a cascata.

**Consequência que provou o valor da decisão.** Foi exatamente isso que permitiu concluir o
[ADR-32](#adr-32): como o código é o mesmo, qualquer divergência entre V1.0.1 e V1.0.2 em
mensagens não desviadas **só pode** vir do modelo. Sem essa garantia, teríamos passado horas
procurando um bug que não existe.

---

<a name="adr-31"></a>
## ADR-31 — Reaproveitar o conjunto de personas

**Decisão.** A bateria da V1.0.2 usa o **mesmo** `dataset_personas.json` gerado para a V1.0.1.
Gerar mensagens novas exige `--gerar-dataset` explícito.

**Por quê.** A V1.0.1 já foi medida e publicada sobre aquelas 150 mensagens. Gerar mensagens
novas tornaria os dois resultados incomparáveis — qualquer diferença poderia vir do conjunto de
teste em vez da arquitetura. O padrão do código é o comportamento correto; quebrar a comparação
exige um passo deliberado.

**Efeito colateral positivo.** A rodada custou **US$ 0** de geração, porque as mensagens já
existiam. A bateria inteira das três versões saiu por ~US$ 0,31.

---

<a name="adr-32"></a>
## ADR-32 — O LLM não é reprodutível

**Não é uma decisão — é um achado que muda como lemos todas as métricas.**

**O que observamos.** V1.0.1 e V1.0.2 compartilham o mesmo código de orquestrador
([ADR-30](#adr-30)). Em mensagens **não desviadas**, deveriam produzir resultados idênticos.
Produziram 5 respostas diferentes em 180 mensagens.

| Mensagem | V1.0.1 | V1.0.2 |
|---|---|---|
| "Preciso mudei de casa, como mudo o CEP?" | `atualizar_cep_entrega` | `alterar_endereco` |
| "vai chover amanhã em recife?" | FAST_PATH | `consultar_previsao_tempo` |

Isso com `temperature=0` e `reasoning_effort: none`.

**Evidência independente.** A V1.0.1 mediu **85%** de Hit@2 no dataset oficial na rodada de
ontem e **75%** hoje — mesmo código, mesmas 30 mensagens. Variação de 10 pontos, que são
3 mensagens.

**Consequências práticas:**

1. **Diferenças pequenas entre versões com LLM são ruído.** A V1.0.2 aparecer 0,9 ponto abaixo
   da V1.0.1 não significa nada.
2. **n=30 é insuficiente** para comparar qualquer coisa que envolva LLM. Uma mensagem vale
   3,3 pontos percentuais.
3. **Comparações precisam de repetição**, não de rodada única. Não fizemos isso nesta bateria —
   fica registrado como limitação.
4. **`temperature=0` não é determinismo.** É apenas amostragem gulosa; empates entre tokens,
   roteamento interno e mudanças no lado do provedor continuam variando a saída.

**O que faríamos diferente com mais tempo:** rodar cada versão N vezes e reportar média com
intervalo, em vez de um número por rodada.

---

## Limitações assumidas na V1.0.2

1. **21 desvios é amostra pequena.** 21 de 21 corretos tem IC 95% de [84%, 100%].
2. **A economia depende do tráfego.** Medimos 9% num conjunto pesado em AGENT e 27% no oficial.
   Só a produção dirá qual é a composição real.
3. **A latência medida está confundida por ruído de rede** — as versões rodaram em sequência,
   não intercaladas. O número confiável é a contagem de chamadas evitadas, que é exata.
4. **Os bloqueios de produção continuam abertos** e são comuns às três versões: camada de
   política para ferramentas financeiras, conjunto humano limpo, medição ponta a ponta.
