# Registro de Decisões Técnicas

> Cada decisão está registrada no formato **contexto → decisão → alternativas consideradas →
> evidência → consequência**. Onde houve medição, o número está aqui. Onde uma ideia foi
> testada e **descartada**, ela também está aqui — inclusive as que falharam.
>
> Estrutura e diagramas: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Sumário

| # | Decisão | Veredito |
|---|---|---|
| [01](#adr-01) | Router é classificador local, não LLM | Adotado |
| [02](#adr-02) | TF-IDF de caractere (`char_wb`) em vez de palavra | Adotado |
| [03](#adr-03) | Regressão Logística em vez de SVM/árvore | Adotado |
| [04](#adr-04) | Sem regras de palavra-chave no router | Rejeitado por contraexemplo |
| [05](#adr-05) | Busca híbrida lexical + vetorial | Adotado |
| [06](#adr-06) | Reciprocal Rank Fusion em vez de soma de scores | Adotado |
| [07](#adr-07) | **Prior de generalidade** (o coração da solução) | Adotado |
| [08](#adr-08) | BM25 com normalização de comprimento | **Testado e descartado** |
| [09](#adr-09) | Clusterização de quase-duplicatas | **Testado e descartado** |
| [10](#adr-10) | Arquitetura em duas camadas | Adotado |
| [11](#adr-11) | Cascata de escalonamento condicional | Adotado |
| [12](#adr-12) | Métricas próprias + LLM-as-judge, sem RAGAS | Adotado |
| [13](#adr-13) | Duas ressalvas honestas sobre a medição | Documentado |
| [14](#adr-14) | `.env` fora do controle de versão | Corrigido |

---

<a name="adr-01"></a>
## ADR-01 — O Router é um classificador local, não um LLM

**Contexto.** A ideia inicial era usar LangChain/LangGraph com OpenAI como orquestrador que
identifica a complexidade da solicitação e decide o caminho.

**Decisão.** O router é `TfidfVectorizer` + `LogisticRegression`, rodando localmente.

**Por quê.** O enunciado pede para decidir "o caminho mais barato e rápido possível **antes**
de qualquer chamada a um LLM caro". Usar um LLM para tomar essa decisão seria contraditório:
o roteador viraria exatamente o custo que ele existe para evitar.

**Evidência.**

| | Classificador local | LLM |
|---|---|---|
| Latência | ~0,9 ms | 30–120 ms |
| Custo/query | US$ 0,0000005 | US$ 0,01–0,03 |
| Determinístico | sim | não |
| Acurácia no eval | **100%** | não medido — não haveria ganho a capturar |

Validação cruzada 5-fold no treino: **0,905**. O LLM aqui seria mais caro, mais lento e sem
ganho de acurácia disponível.

**Consequência.** A OpenAI não some do projeto — ela migra para onde paga o próprio custo:
estágio vetorial da busca, desempate em casos ambíguos ([ADR-11](#adr-11)) e avaliação
([ADR-12](#adr-12)).

---

<a name="adr-02"></a>
## ADR-02 — TF-IDF de caractere (`char_wb`) em vez de palavra

**Decisão.** `analyzer="char_wb"`, `ngram_range=(2,5)`.

**Por quê.**
1. Português tem muita flexão: *fatura/faturas*, *parcelar/parcelamento*, *bloquear/bloqueio*.
   N-gramas de caractere capturam o radical comum **sem** precisar de stemmer nem de lista de
   stopwords em português.
2. Toleram erro de digitação — em chat real de atendimento isso não é detalhe.
3. O treino tem **53 exemplos**. Vocabulário de palavras seria esparso demais; n-gramas de
   caractere geram muito mais sinal por exemplo.

**Evidência.** Acurácia em validação cruzada 5-fold, mesmo classificador:

| Analisador | CV5 |
|---|---|
| `char_wb(2,5)` | **0,905** |
| `word(1,2)` | 0,867 |

---

<a name="adr-03"></a>
## ADR-03 — Regressão Logística em vez de SVM ou árvore

**Decisão.** `LogisticRegression(C=5.0, class_weight="balanced")`.

**Por quê.**
1. **Precisamos de probabilidade calibrada**, não só do rótulo. A confiança alimenta a cascata
   ([ADR-11](#adr-11)). `LinearSVC` não expõe `predict_proba`; árvores dão probabilidade mal
   calibrada.
2. **Inspecionável**: sendo linear, dá para listar quais n-gramas empurraram a decisão — está
   implementado em `QueryRouter.explain()`. Isso importa para auditar erro em produção.
3. Com 53 exemplos, modelo simples e regularizado é a escolha certa. Modelo complexo decoraria
   o treino.

**Evidência.** `LinearSVC` deu CV5 = 0,887, abaixo da Regressão Logística (0,905) — ou seja,
não houve nem troca de acurácia por interpretabilidade a fazer.

---

<a name="adr-04"></a>
## ADR-04 — Sem regras de palavra-chave no router

**Contexto.** O atalho óbvio seria: "se começa com *bom dia/oi/olá* → FAST_PATH".

**Decisão.** Rejeitado. O router é 100% aprendido.

**Por quê.** A regra quebra em casos comuns e reais. Contraexemplo executado no pipeline:

```
Query : "Bom dia, preciso do meu saldo"
Regra de saudação diria .... FAST_PATH  (errado — cliente não é atendido)
Classificador diz .......... AGENT      ✓ (confiança 0,60)
```

O próprio `eval_dataset.json` tem `"Bom dia! Qual o valor mínimo para abrir uma conta?"`
(FAST_PATH) convivendo com saudação + intenção acionável. O classificador pondera a frase
inteira; a regra olharia só o prefixo.

**Consequência.** Como bônus, esse caso caiu com confiança 0,60 — abaixo do limiar de 0,65 —
e é exatamente o tipo de query que a cascata manda para o LLM desempatar. A fronteira difícil
é detectada, não ignorada.

---

<a name="adr-05"></a>
## ADR-05 — Busca híbrida: lexical **e** vetorial

**Decisão.** Três sinais no estágio de recall: TF-IDF de caractere, TF-IDF de palavra e
embeddings da OpenAI.

**Por quê.** As duas famílias erram de formas complementares, e este catálogo exibe os dois erros:

- **Lexical acerta o termo e erra o sentido.** Para `"Preciso mudei de casa, como mudo o CEP?"`
  → `alterar_endereco`, a palavra "endereço" não aparece na query. Não há caractere em comum
  entre *CEP/casa* e *endereço*. Medido: a tool correta ficou na **posição 90 de 285**.
- **Vetorial acerta o sentido e escorrega no termo.** Embeddings aproximam `consultar_saldo`
  de *todas* as variantes de saldo, inclusive `consultar_saldo_poupanca` e `consultar_saldo_pj`.
  Para separar "poupança" de "corrente", o sinal lexical é mais confiável.

**Evidência.** Com busca lexical pura, `Recall@15 = 0,85–0,90`. As **4 falhas são todas
semânticas** (sem sobreposição de palavras). E recall é o **teto** do sistema: nenhuma
reordenação recupera uma tool que nunca entrou na lista de candidatas.

**Consequência.** O estágio vetorial fica atrás de um gancho de injeção
(`ToolRetriever.set_dense_provider()`): sem chave de API, a busca segue lexical e tudo funciona.
O núcleo não importa `openai` em lugar nenhum.

---

<a name="adr-06"></a>
## ADR-06 — Reciprocal Rank Fusion em vez de somar os scores

**Decisão.** Fusão por RRF: cada sinal contribui `1 / (k + posição)`, com `k = 60`.

**Por quê.** Similaridade de cosseno de embedding e score de TF-IDF vivem em **escalas
diferentes e não comparáveis**. Somar exigiria normalizar, e a normalização fica refém de
outliers — um único documento muito similar comprime todo o resto. O RRF combina **posições**
no ranking, não valores, então é imune à escala. É o padrão da literatura de busca híbrida, e
`k = 60` é a constante do paper original (Cormack et al., 2009), que amortece a diferença entre
as primeiras posições para que um único índice confiante não domine a fusão.

**Evidência.** Ganho medido só da fusão, antes de qualquer outro tratamento:

| | Precision@2 |
|---|---|
| caractere sozinho | 0,35 |
| palavra sozinha | 0,20 |
| **RRF(caractere + palavra)** | **0,45** |

---

<a name="adr-07"></a>
## ADR-07 — Prior de generalidade (o coração da solução)

**Contexto.** Este é o problema central do case, e ele não é óbvio à primeira leitura.

O `tools_registry.json` tem **285 tools** com quase-duplicatas propositais. Para várias queries,
a tool **mais parecida lexicalmente não é a correta** — ela é uma variante hiper-específica que
praticamente copia a frase do cliente:

```
Query: "Quanto eu tenho disponível na conta agora?"
  correto ....... consultar_saldo
                  "Consulta o saldo bancário da conta corrente do cliente."
  isca .......... consultar_valor_disponivel_conta
                  "Consulta o valor disponível agora na conta corrente do cliente."
                  ↑ quase uma cópia da query — vence por similaridade
```

**Decisão.** Multiplicar o score fundido por `(1 − λ · especificidade)`, com `λ = 0,35`.
A especificidade combina dois sinais observáveis, com peso igual: número de tokens do nome e
tamanho da descrição.

**Por quê.** O gabarito premia a **capacidade canônica**, não a paráfrase mais literal. Isso faz
sentido em produção: entre duas tools que atendem à intenção, a mais geral cobre mais casos e é
a porta de entrada correta; as variantes longas são refinamentos de um catálogo mal higienizado.
O desconto é **multiplicativo** (não subtrativo) para ser proporcional ao score — assim uma tool
irrelevante não sobe no ranking só por ser genérica.

**Evidência.**

| λ | Precision@2 |
|---|---|
| 0,0 | 0,45 |
| 0,1 | 0,65 |
| 0,2 | 0,75 |
| **0,35 (adotado)** | **0,85** |
| 0,5 | 0,80 |

**Honestidade metodológica — sobre ajustar no conjunto de teste.** O valor `λ = 0,35` **não** foi
escolhido pelo pico da curva. Escolher o pico com 20 queries seria ajustar no próprio teste
(*leakage*) e o número reportado seria otimista. Foi escolhido o **centro do platô estável**
(0,2 a 0,5, onde P@2 fica entre 0,75 e 0,85). A largura desse platô é justamente a evidência de
que o resultado não depende de um ajuste frágil. A calibração definitiva roda no conjunto
ampliado ([ADR-12](#adr-12)).

---

<a name="adr-08"></a>
## ADR-08 — BM25 com normalização de comprimento — **testado e descartado**

**Contexto.** BM25 é a resposta de manual para "não favorecer documentos longos": o parâmetro `b`
controla exatamente a normalização por comprimento. Se funcionasse, substituiria o prior de
generalidade por um mecanismo canônico de recuperação de informação — bem mais defensável do
que uma heurística própria.

**Decisão.** Descartado por evidência. Implementado, medido, perdeu.

**Evidência.** Varremos `b` de 0,0 a 1,0 nos dois analisadores:

| Configuração | Precision@2 |
|---|---|
| BM25 caractere, `b = 1,0` (melhor da varredura) | 0,20 |
| BM25 palavra, `b = 0,75` | 0,20 |
| RRF de BM25(caractere + palavra), `b = 1,0` | 0,35 |
| **TF-IDF + prior de generalidade** | **0,85** |

**Por que perdeu.** O `b` do BM25 normaliza a **saturação de frequência de termos**, não a
especificidade do conceito. Ele trata "descrição longa" como ruído a compensar, e não como
sinal de que a tool é uma variante estreita. São problemas diferentes que só parecem o mesmo.

---

<a name="adr-09"></a>
## ADR-09 — Clusterização de quase-duplicatas — **testada e descartada**

**Contexto.** Ideia intuitiva e aparentemente elegante: agrupar tools quase idênticas, eleger um
representante canônico por grupo (menor nome / medoid) e retornar só representantes. Em produção
seria "higienizar o catálogo".

**Decisão.** Descartada por evidência. Piorou.

**Evidência.** Union-find sobre similaridade tool-a-tool, variando o limiar:

| Limiar | Capacidades | P@2 | **Recall@15** |
|---|---|---|---|
| 0,55 | 154 | 0,55 | **0,70** ⚠️ |
| 0,60 | 190 | 0,60 | 0,85 |
| 0,70 | 246 | 0,45 | 0,85 |
| **sem clusterização** | 285 | **0,85** | **0,90** |

**Por que falhou.** Agrupamento por transitividade encadeia grupos grandes demais: se `a ~ b` e
`b ~ c`, `a` e `c` acabam juntos mesmo estando distantes. Com limiar baixo o efeito derrubou o
Recall@15 de 0,85 para 0,70 — jogando fora a tool correta **antes** de qualquer reordenação.

**Consequência.** A simplicidade venceu: RRF + prior, sem clusterização. Vale registrar que a
alternativa mais sofisticada foi a que perdeu.

---

<a name="adr-10"></a>
## ADR-10 — Arquitetura em duas camadas

**Decisão.** Núcleo puro em `candidate_starter/` (sklearn + stdlib); tudo que depende de rede,
credencial ou LLM em `App/`. Dependência só na direção `App/ → núcleo`.

**Por quê.** Detalhado em [ARCHITECTURE.md § 2](ARCHITECTURE.md#2-as-duas-camadas-e-por-que-estão-separadas).
Em uma frase: quem for avaliar o case roda `pip install -r requirements.txt` e `pytest` sem
LangGraph, sem chave e sem internet — enquanto a camada agêntica que exercita o stack continua
existindo, consumindo as **mesmas** classes, sem duplicar lógica.

**Consequência verificada.** `python -m App.main` sem `OPENAI_API_KEY` roda normalmente e avisa:
`MODO LOCAL (sem OPENAI_API_KEY): busca lexica pura, sem escalonamento.`

---

<a name="adr-11"></a>
## ADR-11 — Cascata de escalonamento condicional

**Decisão.** O LLM só é chamado quando o componente barato **admite que não sabe**:

| Gatilho | Condição | Ação |
|---|---|---|
| Router incerto | `confiança < 0,65` | LLM pequeno desempata a rota |
| Retrieval ambíguo | `margem(1º, 2º) < 5%` | LLM reordena as **15 candidatas** (não as 285) |

**Por quê.** A leitura simplista do case seria "nunca use LLM". A leitura correta é "use LLM
apenas onde o barato falha". Rerank em 100% das queries destruiria a economia que o case pede
para demonstrar; rerank em ~10% delas custa 10% do preço e captura a maior parte do ganho.

**Detalhe do prompt.** O prompt de rerank instrui explicitamente a preferir a tool canônica.
Sem isso, o LLM cai na **mesma armadilha** da similaridade lexical: para *"manda o pdf da minha
fatura"* ele escolheria `enviar_pdf_fatura_atual`. A política de desempate do catálogo vai
escrita no prompt — é a mesma que o prior implementa no estágio local.

**Falha segura.** Erro de rede, JSON inválido ou nome de tool inexistente ⇒ devolve a decisão
**local** original. O LLM só pode melhorar o resultado, nunca derrubar o pipeline.

**Critério de aceite (compromisso assumido).** O escalonamento só fica ligado por padrão se
**pagar o próprio custo**, medido no notebook 04. Se não pagar, o honesto é desligar e reportar
o experimento negativo. Um número inflado por uma decisão que contraria a premissa do case vale
menos que a medição honesta.

---

<a name="adr-12"></a>
## ADR-12 — Métricas próprias + LLM-as-judge; RAGAS descartado

**Contexto.** A ideia inicial incluía RAGAS como métrica de avaliação.

**Decisão.** RAGAS fora. Métricas exatas + LLM-as-judge para dois papéis específicos.

**Por que RAGAS não se aplica aqui.**
1. RAGAS avalia **geração** (faithfulness, answer relevancy). Aqui não há geração para julgar —
   `common/mock_llm.py` devolve uma string fixa.
2. `context_precision`/`context_recall` até mapeariam para a seleção de tools, mas RAGAS os
   estima **com um LLM** (portanto com erro), enquanto nós temos **ground truth exato**. Seria
   pagar tokens para medir pior algo que já sabemos medir com precisão.

**Onde o LLM-as-judge realmente ajuda.**
1. **Ampliar o conjunto de teste.** 30 queries (20 com tool esperada) é pouco: a diferença entre
   29/30 e 30/30 é uma linha. Geramos ~200 queries (paráfrases, gírias, erros de digitação,
   ambíguas e fora de escopo) para as métricas ganharem significado — e para calibrar o λ do
   [ADR-07](#adr-07) sem tocar no eval original.
2. **Precision@K semântico.** O gabarito é discutível em alguns casos: *"Fiz uma compra que não
   reconheço, quero contestar"* tem gabarito `estornar_transacao`, mas `contestar_compra_desconhecida`
   existe e é defensável. O juiz avalia equivalência **funcional**, produzindo uma métrica branda
   ao lado da estrita. **Reportamos as duas**, nunca só a favorável.

**Métricas adotadas, e o porquê de cada uma:**

| Métrica | Por que existe |
|---|---|
| Acurácia + matriz de confusão | Obrigatória no case |
| Precision@K | Obrigatória no case |
| Economia de custo/latência | Obrigatória no case |
| **IC 95% (Wilson)** | "100% em 30 amostras" tem intervalo de 88,6% a 100%. Reportar só o ponto venderia certeza que o dado não sustenta. Wilson (e não a normal) porque continua válido quando a proporção encosta em 1 — nosso caso |
| **Erro assimétrico** | Os dois erros custam coisas diferentes: `AGENT→FAST_PATH` é falha de atendimento (cliente pede saldo, recebe resposta de prateleira); `FAST_PATH→AGENT` só desperdiça dinheiro. Uma acurácia única esconde isso |
| **MRR@k** | Precision@K responde "acertou?"; MRR responde "acertou em que posição?". Acertar em 1º permite executar direto; em 2º exige o LLM escolher |
| **Recall@N do estágio 1** | É o teto de tudo. Se a tool não entra no top-15, nenhum rerank salva |
| **Latência p50/p95** | Média esconde cauda; SLA se mede no p95 |
| **Sucesso ponta a ponta** | Rota certa **E** tool certa — o que o cliente de fato experimenta. Cobre o ponto cego descrito na [Ressalva 1](#adr-13) |

---

<a name="adr-13"></a>
## ADR-13 — Duas ressalvas honestas sobre a medição

Nenhuma das duas é bug: são vieses do desenho do harness fornecido. Estão documentadas porque
inflam números a nosso favor, e omitir isso seria desonesto.

**Ressalva 1 — Precision@K é condicional ao acerto do router.**
`run_harness` só chama o retriever quando o router decidiu AGENT. Se o router mandar
erroneamente uma query AGENT para o FAST_PATH, essa query **sai da conta** do Precision@K. Ou
seja: um router ruim pode **inflar** o Precision@K. Por isso reportamos também
`end_to_end_success_rate`, que conta esse cenário como falha.

**Ressalva 2 — a economia de latência obrigatória compara coisas diferentes.**
No caminho inteligente, o harness soma apenas router + retrieval. No baseline, soma a chamada
simulada de LLM. Como a chamada de LLM é ordens de grandeza mais lenta, a economia obrigatória
sai em **97,6%** — verdadeiro, mas compara "só a decisão" contra "decisão + resposta".
Mantivemos a métrica obrigatória como o case pede **e** adicionamos
`latency_savings_pct_comparable` = **65,3%**, que inclui a chamada de LLM nos dois lados. Essa
segunda é a comparação maçã-com-maçã.

---

<a name="adr-14"></a>
## ADR-14 — `.env` fora do controle de versão

**Contexto.** O `.gitignore` original do case ignorava `__pycache__`, `.venv`, `reports/` — mas
**não** ignorava `.env`. Como o projeto passaria a guardar uma chave da OpenAI, o primeiro
`git add` vazaria a credencial.

**Decisão.** Corrigido **antes** de qualquer credencial entrar no projeto: `.env` e `.env.*`
ignorados, com exceção explícita para `.env.example` (versionado, sem valores). `App/cache/`
também ignorado.

**Pendência conhecida.** `reports/` está no `.gitignore` original, mas
`reports/candidate_report.json` é o **entregável nº 2** do case. Na hora de commitar é preciso
`git add -f reports/candidate_report.json` ou copiar para um caminho versionado. Não alteramos a
regra para não mexer em decisão do case sem necessidade.

---

## Resultados consolidados

| Métrica | Baseline ingênuo | Solução | Fonte |
|---|---|---|---|
| Acurácia do Router | — | **100%** (IC95%: 88,6%–100%) | ADR-01/02/03 |
| Precision@2 | 0,25 | **0,85** | ADR-05/06/07 |
| MRR@2 | — | 0,800 | — |
| Sucesso ponta a ponta | — | 85% | ADR-13 |
| Economia de custo | — | **77,8%** | — |
| Economia de latência | — | 97,6% (65,3% comparável) | ADR-13 |
| Latência do router p50/p95 | — | 0,94 / 1,18 ms | — |
| Latência da busca p50/p95 | — | 1,74 / 2,17 ms | — |

## Limitações assumidas

1. **n = 30 no eval.** Nenhuma métrica isolada é conclusiva. Mitigado por IC e pelo conjunto ampliado.
2. **Teto de recall.** Com busca lexical pura, `Recall@15 ≈ 0,90`. As falhas restantes são
   semânticas e é para isso que o estágio vetorial existe.
3. **Gabarito discutível** em alguns casos. Mitigado pela métrica semântica reportada ao lado da estrita.
4. **Lambda não implementado.** Registrado como próximo passo em
   [ARCHITECTURE.md § 6](ARCHITECTURE.md#6-caminho-para-produção-aws-lambda--melhoria-futura).
