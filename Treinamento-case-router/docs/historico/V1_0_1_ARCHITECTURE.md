# Arquitetura — V1.0.1 (Agente Orquestrador)

> **O que é este documento:** como a V1.0.1 é montada, o que mudou em relação à V1, e como
> as duas são comparadas.
> **Documentos irmãos:** [V1_0_1_DECISIONS.md](V1_0_1_DECISIONS.md) (o porquê de cada escolha) ·
> [ARCHITECTURE.md](ARCHITECTURE.md) (a V1) · [V1_DESCOBERTAS.md](V1_DESCOBERTAS.md) (os bugs que motivaram esta versão)

---

## 1. O que muda, em uma imagem

```mermaid
flowchart TD
    subgraph V1["🟦 V1 — decisão 100% local"]
        direction TB
        A1[Mensagem] --> A2["<b>Classificador</b><br/>TF-IDF + Regressão Logística<br/>~1 ms · US$ 0"]
        A2 -->|FAST_PATH| A3[Resposta pronta]
        A2 -->|AGENT| A4["<b>Busca híbrida</b><br/>lexical + vetorial + RRF<br/>escolhe 2 de 285"]
        A4 --> A5[Executa ferramenta]
    end

    subgraph V101["🟩 V1.0.1 — um LLM decide as duas coisas"]
        direction TB
        B1[Mensagem] --> B2["<b>Recuperação local</b><br/>reduz 285 → 20 candidatas<br/>~2 ms · US$ 0"]
        B2 --> B3["<b>1 chamada de LLM</b><br/>decide a rota E escolhe a ferramenta<br/>~1,2 s · ~US$ 0,0002"]
        B3 -->|FAST_PATH| B4[Resposta pronta]
        B3 -->|AGENT| B5[Executa ferramenta]
        B3 -.->|"resposta inválida<br/>ou rede caiu"| B6["<b>Plano B</b><br/>decisão local da V1<br/><i>registrado e contado</i>"]
    end

    style V1 fill:#e8f4f8,stroke:#2c7a99,stroke-width:2px
    style V101 fill:#e9f7ef,stroke:#2e7d52,stroke-width:2px
    style B3 fill:#fff3cd,stroke:#b8860b
    style B6 fill:#ffeaea,stroke:#c44
```

**O que NÃO muda:** o estágio de recuperação local continua existindo nas duas. O LLM da
V1.0.1 **nunca vê as 285 ferramentas** — ele escolhe entre 20. Isso preserva a tese central do
projeto: o componente barato faz a triagem, o caro faz o julgamento fino.

---

## 2. Por que uma chamada e não duas

O caminho óbvio seria: uma chamada para decidir a rota, outra para escolher a ferramenta.
Rejeitamos, e o motivo é como o custo de um LLM se divide:

| Componente | O que domina |
|---|---|
| Tokens | o **preço em dólares** |
| Ida e volta de rede | a **latência** (~200–800 ms, mesmo para 5 tokens de resposta) |

Duas chamadas pagariam **dois** tempos de rede. Como as duas decisões dependem da mesma leitura
da mensagem, separá-las gastaria o dobro da latência sem ganhar informação.

Isso só é possível porque a busca local custa ~2 ms e zero dólar — então rodamos ela **antes**
de saber a rota, e mandamos as candidatas junto no mesmo prompt. Se a resposta for FAST_PATH,
descartamos as candidatas e não perdemos nada relevante.

---

## 3. O prompt carrega as duas armadilhas conhecidas

Os testes da V1 revelaram dois modos de falha. O prompt trata os dois de forma explícita,
porque um LLM genérico cairia nos mesmos:

```mermaid
flowchart LR
    P["<b>Prompt de sistema</b>"] --> R1["<b>Regra de NEGAÇÃO</b><br/>bloquear ≠ desbloquear<br/>ativar ≠ desativar"]
    P --> R2["<b>Regra CANÔNICA</b><br/>prefira a ferramenta geral<br/>à variante que copia a frase"]
    P --> R3["<b>Regra de SAUDAÇÃO</b><br/>'bom dia' + pedido = AGENT"]
    P --> R4["<b>Regra de REGISTRO</b><br/>formalidade e erro de digitação<br/>não mudam a intenção"]

    R1 --> F1["corrige: <i>quero bloquear</i><br/>→ desbloquear_cartao"]
    R2 --> F2["corrige: <i>pdf da fatura</i><br/>→ enviar_pdf_fatura_atual"]
    R3 --> F3["corrige: <i>oi, preciso do saldo</i><br/>→ FAST_PATH"]
    R4 --> F4["corrige: <i>solicito o informe</i><br/>→ FAST_PATH"]

    style P fill:#fff3cd,stroke:#b8860b
```

Cada regra existe por causa de um bug **medido** na V1, não por precaução genérica.

---

## 4. A camada de testes com subagentes

```mermaid
flowchart TD
    subgraph GER["[1] GERAÇÃO — 5 personas em paralelo"]
        direction LR
        P1[leigo_idoso] ~~~ P2[jovem_gírias] ~~~ P3[dedos_gordos] ~~~ P4[especialista] ~~~ P5[caótico]
    end

    GER --> DS[("<b>Um único conjunto</b><br/>de mensagens de teste")]

    DS --> EXE

    subgraph EXE["[2] EXECUÇÃO — as MESMAS mensagens nas duas versões"]
        direction LR
        E1[V1] ~~~ E2[V1.0.1]
    end

    EXE --> MET["<b>[3] Métricas + divergências</b><br/><i>código, não LLM</i><br/>contar acerto é operação exata"]

    MET --> AVA

    subgraph AVA["[4] AVALIAÇÃO — 2 especialistas, lentes diferentes"]
        direction LR
        A1["<b>métrico</b><br/>os números sustentam<br/>a conclusão?"] ~~~ A2["<b>produção</b><br/>isso aguenta<br/>o mundo real?"]
    end

    AVA --> REL[Relatório em App/reports/]

    style GER fill:#e8f4f8,stroke:#2c7a99
    style EXE fill:#e9f7ef,stroke:#2e7d52
    style MET fill:#fef6e4,stroke:#b8860b
    style AVA fill:#f3e8f8,stroke:#7a4899
```

### As quatro decisões de método embutidas nesse desenho

| Decisão | Por quê |
|---|---|
| **As personas geram UMA vez** | Se cada versão fosse testada com mensagens diferentes, a diferença de acerto poderia vir do conjunto de mensagens em vez do sistema. Mesmas entradas = a única variável é a versão |
| **As métricas são código, não LLM** | Contar acerto é operação exata. Pedir a um LLM que calcule métricas introduz erro onde não precisa haver nenhum. LLM entra só onde há julgamento subjetivo |
| **5 personas com perfis fechados** | Um único agente gerando 300 mensagens produz 300 variações do mesmo registro — exatamente o ponto cego que derrubou a V1 |
| **2 avaliadores com lentes distintas** | Dois avaliadores idênticos dariam a mesma resposta duas vezes. As lentes cobrem as duas formas de uma decisão dar errado: os números mentem, ou o sistema quebra em produção |

### As 5 personas e o que cada uma expõe

| Persona | Escrita | Desempenho na V1 |
|---|---|---|
| `leigo_idoso` | indireta, sem jargão, com rodeio | 39% |
| `jovem_girias` | abreviação, gíria, sem pontuação | 54% ← o melhor |
| `dedos_gordos` | 2 a 4 erros de digitação por mensagem | 32% |
| `especialista_bancario` | jargão correto e formal | **25%** ← o pior |
| `caotico` | fora de escopo, adversarial, degenerado | robustez |

As duas **pontas** do espectro falharam mais. Não é coincidência: ambas se afastam do registro
médio dos 53 exemplos de treino, em direções opostas.

---

## 5. Estrutura de pastas (o que foi acrescentado)

```
Treinamento-case-router/App/
│
├── versions/                      🆕 as duas versões sob um contrato comum
│   ├── base.py                       BasePipeline + PipelineResult
│   ├── v1_classic.py                 V1: embrulha candidate_starter/
│   └── v1_0_1_orchestrator.py        V1.0.1: orquestrador + tabela de preços
│
├── agents/                        🆕 subagentes de TESTE (SDK da OpenAI)
│   ├── base.py                       Agent reutilizável + execução paralela
│   ├── personas.py                   as 5 personas
│   ├── evaluators.py                 os 2 avaliadores
│   └── run_suite.py                  orquestra as 4 etapas
│
├── eval/
│   └── model_bakeoff.py           🆕 comparativo de modelos candidatos
│
├── reports/                       🆕 saídas dos subagentes (versionadas)
│   ├── model_bakeoff.json
│   ├── dataset_personas.json
│   └── bateria_v1_x_v101.json
│
└── core/, notebooks/                 (inalterados da V1)
```

### Por que `versions/` embrulha em vez de alterar

`candidate_starter/` é o entregável do case, **já publicado no GitHub**. Alterá-lo para
acomodar a comparação misturaria duas preocupações e quebraria a promessa de que aquele código
roda com o `requirements.txt` original, sem rede.

`v1_classic.py` não reimplementa nada — apenas adapta as classes existentes ao contrato comum.

---

## 6. Como rodar

Os comandos abaixo rodam a partir de `Treinamento-case-router/`.

**Comparativo de modelos candidatos:**

```bash
python -m App.eval.model_bakeoff
```

**Bateria completa (personas geram → as versões rodam → avaliadores julgam):**

```bash
python -m App.agents.run_suite --gerar-dataset --n-por-persona 30
```

**Rodada padrão, reaproveitando o conjunto já gerado.** É o modo normal: gerar mensagens novas
troca o conjunto de teste e quebra a comparação com as rodadas anteriores, por isso reaproveitar
é o default e gerar é que precisa de flag explícita.

```bash
python -m App.agents.run_suite
```

> ⚠️ Tudo aqui precisa de `OPENAI_API_KEY` em `App/.env`. O núcleo do case em
> `candidate_starter/` continua rodando offline, sem nenhuma dessas dependências.
