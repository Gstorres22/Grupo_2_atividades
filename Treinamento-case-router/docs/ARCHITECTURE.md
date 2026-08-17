# Arquitetura — Cérebro de Roteamento

> Documento de arquitetura: **o que existe, onde mora e por que está separado assim**.
> Para o raciocínio por trás de cada escolha técnica, veja [DECISIONS.md](DECISIONS.md).

---

## 1. Visão geral do fluxo

```mermaid
flowchart TD
    Q[Query do usuário] --> R["<b>1. Router</b><br/>TF-IDF char_wb + Regressão Logística<br/>~0,9 ms · US$ 0,0000005"]

    R -->|"confiança &lt; 0,65<br/>(faixa cinzenta)"| ESC["<b>Escalonamento</b><br/>LLM pequeno desempata"]
    ESC --> D{rota final}

    R -->|FAST_PATH| F["<b>Resposta local</b><br/>mock_llm.fast_path_answer<br/>custo de LLM = 0"]
    R -->|AGENT| S

    D -->|FAST_PATH| F
    D -->|AGENT| S["<b>2. Seleção de Tools</b><br/>Busca híbrida sobre 285 tools<br/>~1,7 ms"]

    S -->|"margem 1º-2º &lt; 5%<br/>(ambíguo)"| RR["<b>Rerank por LLM</b><br/>15 candidatas, não 285"]
    S -->|margem OK| X["<b>Executa a tool</b><br/>+ 1 chamada de LLM<br/>com contexto reduzido"]
    RR --> X

    F --> H["<b>3. Evaluation Harness</b><br/>Acurácia · Precision@K · Custo · Latência"]
    X --> H

    style R fill:#e8f4f8,stroke:#2c7a99
    style S fill:#e8f4f8,stroke:#2c7a99
    style H fill:#fef6e4,stroke:#b8860b
    style ESC fill:#ffeaea,stroke:#c44
    style RR fill:#ffeaea,stroke:#c44
```

Os blocos em **vermelho** são os únicos que gastam LLM antes da resposta — e ambos só
são alcançados por aresta condicional, quando o componente barato admite que não sabe.

---

## 2. As duas camadas e por que estão separadas

```mermaid
flowchart LR
    subgraph NUCLEO["🟦 NÚCLEO — candidate_starter/ + common/"]
        direction TB
        N1["router.py · retrieval.py · harness.py"]
        N2["Dependências: numpy, pandas,<br/>scikit-learn, pytest"]
        N3["Roda offline · sem chave de API<br/>determinístico · testável"]
    end

    subgraph APP["🟩 APLICAÇÃO — App/"]
        direction TB
        A1["graph.py · embeddings.py · escalation.py"]
        A2["Dependências: langgraph, openai,<br/>langchain-openai, dotenv"]
        A3["Precisa de rede e credencial<br/>não determinístico"]
    end

    APP -->|"importa e reutiliza"| NUCLEO
    NUCLEO -.->|"NUNCA importa"| APP

    style NUCLEO fill:#e8f4f8,stroke:#2c7a99,stroke-width:2px
    style APP fill:#e9f7ef,stroke:#2e7d52,stroke-width:2px
```

### Por que essa separação

| Motivo | O que garante na prática |
|---|---|
| **O avaliador precisa conseguir rodar** | `pip install -r requirements.txt` e `pytest` funcionam sem LangGraph, sem chave e sem internet. O `requirements.txt` original ficou **intocado** |
| **A premissa do case é custo** | Tudo que é caro, lento ou não determinístico fica fisicamente isolado numa camada opcional. Não há como uma dependência de LLM vazar para o caminho barato por descuido |
| **Falha isolada** | Se a API da OpenAI cair ou a chave expirar, o núcleo continua entregando resultado — degrada para modo lexical, não quebra |
| **Direção única de dependência** | `App/ → candidate_starter/`, nunca o contrário. O núcleo não sabe que a camada de aplicação existe |
| **Reuso sem duplicação** | O grafo LangGraph **consome** `QueryRouter` e `ToolRetriever` como nós. Não existe uma segunda implementação para manter em sincronia |
| **Caminho para o Lambda** | O deploy futuro empacota `candidate_starter/` + `App/core/`, deixando notebooks e avaliação de fora |

### O ponto de contato entre as camadas

A ligação é **um gancho de injeção de dependência**, não um import:

```python
# App/main.py — a camada de aplicação injeta o estágio vetorial
dense_provider = build_dense_provider(settings)   # None se não houver chave
retriever.set_dense_provider(dense_provider)      # o núcleo aceita None sem reclamar
retriever.fit(tools)
```

O núcleo declara o gancho e funciona sem ele. Quem tem a chave de API decide plugar.

---

## 3. Estrutura de pastas

A separação é **física, não só conceitual**: as duas camadas são pastas irmãs no repositório. O
entregável não contém nem menciona nada da camada de aplicação em código — só em comentários — e
por isso é entregue e avaliado sozinho.

```
<repositório>/
│
├── entregavel-case-router/         🟦 ENTREGÁVEL DO CASE
│   │
│   ├── candidate_starter/          núcleo puro · sklearn + stdlib
│   │   ├── router.py                  Pilar 1 · classificação de rota
│   │   ├── retrieval.py               Pilar 2 · busca híbrida em 2 estágios
│   │   ├── harness.py                 Pilar 3 · métricas e benchmark
│   │   ├── run_case.py                (original) orquestra e salva o relatório
│   │   └── tests/                     55 testes
│   │
│   ├── common/                     ⬜ INTOCADO — contratos e mocks fornecidos
│   │   ├── interfaces.py              ABCs que router e retriever implementam
│   │   ├── schemas.py                 RouteResult, ToolMatch, RetrievalResult
│   │   ├── mock_llm.py                custos e latências simuladas
│   │   └── data_loader.py             leitura dos JSONs
│   │
│   ├── data/                       ⬜ INTOCADO
│   │   ├── tools_registry.json        285 tools (com quase-duplicatas propositais)
│   │   ├── router_training_data.json  53 exemplos rotulados
│   │   └── eval_dataset.json          30 queries de avaliação
│   │
│   ├── reports/candidate_report.json  📄 saída do harness (entregável nº2)
│   ├── README.md                      comentário técnico (entregável nº3)
│   └── requirements.txt            ⬜ INTOCADO
│
└── Treinamento-case-router/        🟩 MATERIAL DE APOIO — fora do entregável
    │
    ├── Case_Storytelling_Revisao.docx  a jornada do projeto, em narrativa
    │
    ├── docs/                       DOCUMENTAÇÃO ATUAL
    │   ├── ARCHITECTURE.md            este documento
    │   ├── DECISIONS.md               ADR-01 a 14 · o porquê de cada escolha
    │   ├── V1_DESCOBERTAS.md          bugs achados com personas de usuário
    │   ├── V1_0_1_ARCHITECTURE.md     o orquestrador por LLM
    │   ├── V1_0_1_DECISIONS.md        ADR-15 a 26
    │   ├── V1_0_2_DECISIONS.md        ADR-27 a 32
    │   ├── V1_0_2_RESULTADO.md     ⭐ o resultado que decidiu a arquitetura
    │   └── historico/
    │       └── V1_0_1_COMPARACAO.md   números anteriores à correção do prior
    │
    └── App/                        CAMADA DE APLICAÇÃO (código)
        ├── .env.example               modelo de credenciais (versionado, sem valores)
        ├── .env                       credenciais reais (NUNCA versionado)
        ├── requirements-app.txt       dependências só desta camada
        ├── __init__.py                acha o entregável e o põe no sys.path
        ├── main.py                    entrypoint local · esqueleto do futuro Lambda
        │
        ├── core/
        │   ├── config.py              único lugar que lê variáveis de ambiente
        │   ├── embeddings.py          estágio vetorial + cache em disco
        │   ├── escalation.py          cascata: quando vale gastar um LLM
        │   └── graph.py               StateGraph do LangGraph
        │
        ├── versions/                  V1 clássica, V1.0.1 LLM, V1.0.2 híbrida
        ├── agents/                    5 personas + 2 avaliadores especialistas
        │
        ├── eval/
        │   ├── generate_eval_set.py   amplia o conjunto de teste via LLM
        │   ├── judge.py               LLM-as-judge · Precision@K semântico
        │   ├── run_batch.py           executa lotes de mensagens pelo pipeline
        │   ├── model_bakeoff.py       compara modelos candidatos a orquestrador
        │   └── calibrar_cascata.py    calibra os limiares da V1.0.2
        │
        ├── reports/                   JSON dos experimentos (versionados)
        │   └── historico/
        ├── notebooks/                 desenvolvimento passo a passo
        └── cache/                     embeddings persistidos (não versionado)
```

> **Por que a camada de aplicação saiu de dentro do entregável.** Enquanto ela morava lá, quem
> abrisse o entregável via primeiro nove arquivos de documentação e uma pasta `App/` maior que o
> próprio case — e precisava descobrir sozinho o que era pedido e o que era extra. A direção de
> dependência sempre foi `App/ → candidate_starter/`; mover a pasta apenas tornou isso visível na
> estrutura. O preço é que o `App/` não herda mais o `sys.path` do case, e por isso o
> `App/__init__.py` localiza o entregável sozinho — **procurando por conteúdo, não por nome**,
> justamente para sobreviver a renomeações como esta.

---

## 4. O Pilar 2 em detalhe (onde está a dificuldade real)

```mermaid
flowchart TD
    Q[Query] --> E1

    subgraph E1["ESTÁGIO 1 — RECALL (barato e abrangente)"]
        direction LR
        L1["TF-IDF caractere<br/>char_wb 2-5<br/><i>flexão e typos</i>"]
        L2["TF-IDF palavra<br/>1-2 gramas<br/><i>expressões</i>"]
        L3["Embeddings<br/>OpenAI<br/><i>sentido</i>"]
    end

    E1 --> RRF["<b>Reciprocal Rank Fusion</b><br/>combina POSIÇÕES, não scores<br/><i>imune à diferença de escala</i>"]
    RRF --> C["15 candidatas<br/><i>Recall@15 é o teto do sistema</i>"]

    C --> E2["<b>ESTÁGIO 2 — DESAMBIGUAÇÃO</b><br/>prior de generalidade<br/><i>prefere a tool canônica<br/>à variante hiper-específica</i>"]

    E2 --> K["top-k final"]
    K -.->|"só se ambíguo"| E3["ESTÁGIO 3 — rerank por LLM<br/><i>opcional, vive em App/</i>"]

    style E1 fill:#e8f4f8,stroke:#2c7a99
    style E2 fill:#fef6e4,stroke:#b8860b
    style E3 fill:#ffeaea,stroke:#c44
```

### O problema que o Estágio 2 ataca

O catálogo tem 285 tools com quase-duplicatas **propositais**. Para várias queries, a tool
mais parecida lexicalmente **não** é a correta:

| Query | Tool correta | Sem o prior | Com o prior |
|---|---|---|---|
| "Manda o pdf da minha fatura atual" | `consultar_fatura` | `enviar_pdf_fatura_atual` | ✅ corrigido |
| "Quero parcelar minha fatura em 3 vezes" | `parcelar_fatura` | `parcelar_fatura_numero_vezes_escolhido` | ✅ corrigido |
| "Preciso saber o saldo disponível pra pix" | `consultar_saldo` | `consultar_saldo_disponivel_pix` | ✅ corrigido |
| "Quanto eu tenho disponível na conta agora?" | `consultar_saldo` | `consultar_valor_disponivel_conta` | ❌ **ainda falha** |
| "Cobraram algo errado, quero o dinheiro de volta" | `estornar_transacao` | `reclamar_cobranca_errada...` | ❌ **ainda falha** |

O prior corrige **8 das 20** queries com ferramenta esperada, levando o Precision@2 de 0,45 a
0,85. **Três continuam falhando** — as duas acima e `"Preciso mudei de casa, como mudo o CEP?"`.

Os casos que sobram têm duas causas distintas, e nenhuma delas o prior resolve:

- **A isca também tem nome curto.** `consultar_valor_disponivel_conta` tem 4 tokens — não é
  longa o bastante para o prior derrubá-la.
- **Não há sobreposição de palavras.** Em "mudei de casa / CEP" → `alterar_endereco`, a palavra
  "endereço" simplesmente não aparece. Nenhum reordenamento recupera o que não entrou nas
  candidatas; isso é problema de **recall**, e é o que o estágio vetorial ataca.

Há testes dedicados a essas falhas conhecidas (`test_prior_nao_resolve_tudo`), para que elas
sejam documentadas em vez de virarem surpresa.

Efeito medido de cada camada da solução:

| Estratégia | Precision@2 |
|---|---|
| TF-IDF de palavra, similaridade pura | 0,25 |
| TF-IDF de caractere, similaridade pura | 0,35 |
| RRF(caractere + palavra) | 0,45 |
| **RRF + prior de generalidade** ← adotado | **0,85** |

---

## 5. Como rodar

**Núcleo (o que o avaliador roda) — a partir de `entregavel-case-router/`, sem chave e sem rede:**

```bash
pip install -r requirements.txt
```

```bash
python -m pytest candidate_starter/tests -v
```

```bash
python -m candidate_starter.run_case
```

**Camada de aplicação — a partir de `Treinamento-case-router/`, com LangGraph e OpenAI:**

```bash
pip install -r App/requirements-app.txt
```

```bash
python -m App.main
```

> O diretório de trabalho é diferente em cada bloco, e é de propósito: são duas camadas
> independentes. O `App/` acha o `entregavel-case-router/` sozinho (ver `App/__init__.py`), então basta
> estar em `Treinamento-case-router/` — não é preciso configurar `PYTHONPATH`.

Sem `OPENAI_API_KEY`, `App/main.py` roda igual, em modo lexical, e avisa no console.

---

## 6. Caminho para produção (AWS Lambda) — melhoria futura

```mermaid
flowchart LR
    API[API Gateway] --> LMB["Lambda handler"]

    subgraph COLD["Cold start · uma vez por container"]
        T1["treina o router<br/>(53 exemplos, ~1 s)"]
        T2["indexa 285 tools"]
        T3["carrega embeddings<br/>do cache"]
    end

    subgraph WARM["Warm · por requisição"]
        W1["route ~0,9 ms"]
        W2["search ~1,7 ms"]
    end

    LMB --> COLD
    COLD --> WARM
    T3 -.-> EFS[("Layer ou EFS<br/>embeddings.npz")]

    style COLD fill:#fef6e4,stroke:#b8860b
    style WARM fill:#e9f7ef,stroke:#2e7d52
```

`App/main.py:build_pipeline()` já isola o trabalho caro justamente para esse formato: em
Lambda ele sobe para o escopo do módulo, e só `run_query()` roda por invocação. **Não
implementado neste escopo** — registrado como próximo passo.
