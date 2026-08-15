# Registro de Decisões — V1.0.1

> Continuação de [DECISIONS.md](DECISIONS.md), que vai do ADR-01 ao ADR-14 (a V1).
> Mesmo formato: **contexto → decisão → alternativas → evidência → consequência**.
> Onde houve medição, o número está aqui. O que foi testado e descartado também.

---

## Sumário

| # | Decisão | Veredito |
|---|---|---|
| [15](#adr-15) | Uma chamada de LLM, não duas | Adotado |
| [16](#adr-16) | O LLM escolhe entre 20 candidatas, não 285 | Adotado |
| [17](#adr-17) | O prompt carrega as armadilhas medidas na V1 | Adotado |
| [18](#adr-18) | **`reasoning_effort: none`** — 23 tokens invisíveis medidos | Adotado |
| [19](#adr-19) | **Escolha do modelo por comparativo, não por tabela de preço** | Adotado |
| [20](#adr-20) | Plano B contabilizado, nunca silencioso | Adotado |
| [21](#adr-21) | Conjunto de teste compartilhado entre as versões | Adotado |
| [22](#adr-22) | 5 personas com perfis fechados | Adotado |
| [23](#adr-23) | 2 avaliadores com lentes complementares | Adotado |
| [24](#adr-24) | Avaliador ≠ avaliado (modelos diferentes) | Adotado |
| [25](#adr-25) | Google ADK | **Descartado a pedido** |
| [26](#adr-26) | Rótulos "prata" reportados separados dos "ouro" | Adotado |

---

<a name="adr-15"></a>
## ADR-15 — Uma chamada de LLM, não duas

**Contexto.** A V1.0.1 precisa decidir a rota **e** escolher a ferramenta. O caminho óbvio
seria uma chamada para cada.

**Decisão.** Uma única chamada resolve as duas.

**Por quê.** O custo de um LLM tem dois componentes com comportamentos diferentes:

- **Tokens** → dominam o preço em dólares
- **Ida e volta de rede** → domina a latência (~200–800 ms, mesmo para 5 tokens de resposta)

Duas chamadas pagariam **dois** tempos de rede. Como as duas decisões dependem da mesma leitura
da mensagem, separá-las gastaria o dobro sem ganhar informação.

**Como viabilizamos.** A busca local custa ~2 ms e zero dólar. Rodamos ela **sempre**, antes de
saber a rota, e enviamos as candidatas no mesmo prompt. Se a resposta for FAST_PATH,
descartamos as candidatas — desperdiçamos 2 ms de CPU local, o que é irrelevante.

**Consequência.** A latência da V1.0.1 é de **uma** chamada. Medido: p50 ≈ 1,2 s contra ≈ 2,4 s
que o desenho de duas chamadas custaria.

---

<a name="adr-16"></a>
## ADR-16 — O LLM escolhe entre 20 candidatas, não 285

**Decisão.** Manter o estágio de recuperação local, que reduz 285 → 20 antes do LLM entrar.

**Por quê.** Mandar as 285 ferramentas no prompt é exatamente o que o case pede para evitar:
estoura contexto, confunde o modelo, multiplica custo e latência. A tese central do projeto
continua valendo na V1.0.1 — o componente barato faz a triagem, o caro faz o julgamento fino.

Usamos 20 (e não as 15 da V1) porque aqui a lista vai para um LLM que lê as descrições:
algumas candidatas a mais custam poucos tokens e sobem o teto de acerto.

**Limitação que isso impõe, e que é preciso declarar.** Se a ferramenta correta não estiver
entre as 20 candidatas, **nenhuma inteligência do LLM a recupera**. O `Recall@20` do estágio
local é o teto da V1.0.1 também. A V1.0.1 não conserta um problema de recuperação — ela
conserta um problema de **julgamento**.

---

<a name="adr-17"></a>
## ADR-17 — O prompt carrega as armadilhas medidas na V1

**Contexto.** Um LLM genérico cairia nas mesmas armadilhas que a V1, porque elas não são
problemas de "burrice" do modelo — são propriedades do catálogo.

**Decisão.** Quatro regras explícitas no prompt de sistema, cada uma derivada de um bug
**medido**, não de precaução genérica:

| Regra | Bug da V1 que ela corrige |
|---|---|
| **Negação** — "bloquear" ≠ "desbloquear" | `"quero bloquear meu cartão"` → `desbloquear_cartao` |
| **Canônica** — prefira a ferramenta geral | `"pdf da fatura"` → `enviar_pdf_fatura_atual` em vez de `consultar_fatura` |
| **Saudação** — "bom dia" + pedido = AGENT | `"oi, preciso de ajuda com o meu saldo"` → FAST_PATH |
| **Registro** — formalidade e typo não mudam a intenção | `"solicito o informe de rendimentos"` → FAST_PATH |

**Evidência.** No comparativo, os 5 casos críticos (os que a V1 erra):

| Versão | Precision@2 nos casos críticos |
|---|---|
| V1 | **0%** |
| V1.0.1 / gpt-4o-mini | **100%** |
| V1.0.1 / gpt-5.4-nano | **100%** |
| V1.0.1 / gpt-5.6-luna | **100%** |

Sem a regra de negação, o LLM escolheria `desbloquear_cartao` pela mesma razão que a V1: o nome
é mais parecido. A regra é o que faz a diferença.

---

<a name="adr-18"></a>
## ADR-18 — `reasoning_effort: none` — 23 tokens invisíveis medidos

**Contexto.** A pesquisa de modelos alertou: a família GPT-5.6 vem com `reasoning.effort` =
**`medium`** por padrão. Modelos de raciocínio geram tokens de "pensamento" que **não aparecem
na resposta mas são cobrados como saída** — o preço mais caro.

**Decisão.** Enviar `reasoning_effort: "none"` explicitamente em todo modelo de raciocínio
usado como orquestrador.

**Evidência — medimos em vez de confiar na documentação:**

| Configuração | Tokens de saída | Dos quais, raciocínio |
|---|---|---|
| `gpt-5.6-luna` (default de fábrica) | 41 | **23** |
| `gpt-5.6-luna` com `effort: none` | 12 | **0** |
| `gpt-4o-mini` (não é modelo de raciocínio) | 6 | 0 |

**56% dos tokens de saída no default eram pensamento invisível.** Numa tarefa de classificação
binária, esses tokens não compram acerto — o modelo não precisa deliberar para decidir se
"quero meu saldo" é AGENT.

**Consequência operacional.** Adicionamos um alerta no rastro de execução: se
`reasoning_tokens > 0` aparecer no orquestrador, isso é sinalizado. É uma regressão de custo que,
de outra forma, seria **silenciosa** — a fatura sobe e nada no comportamento denuncia.

⚠️ **Armadilha de migração:** `gpt-5.4-nano` tem default `none`; `gpt-5.6-luna` tem default
`medium`. Quem migrar de um para o outro sem revisar o parâmetro leva um susto na conta.

---

<a name="adr-19"></a>
## ADR-19 — Escolha do modelo por comparativo, não por tabela de preço

**Contexto.** A pesquisa de custos deu uma recomendação sólida no papel (`gpt-5.6-luna`). Mas
preço de tabela e benchmark público não dizem qual modelo acerta **na nossa tarefa**, que é
classificar português brasileiro e escolher entre ferramentas quase-duplicadas.

**Decisão.** Rodar o mesmo pipeline, com o mesmo prompt e as mesmas mensagens, trocando apenas o
modelo ([`App/eval/model_bakeoff.py`](eval/model_bakeoff.py)).

**Evidência:**

| Versão / modelo | rota | P@2 oficial | P@2 críticos | US$/msg | p50 |
|---|---|---|---|---|---|
| V1 (ML clássico) | 100% | 75% | **0%** | 0,006668 | 377 ms |
| V1.0.1 / `gpt-4o-mini` | 100% | **65%** ↓ | 100% | 0,006870 | 1425 ms |
| V1.0.1 / `gpt-5.4-nano` | 100% | 75% | 100% | 0,006974 | 897 ms |
| **V1.0.1 / `gpt-5.6-luna`** | 100% | **80%** | 100% | 0,006950 | 1236 ms |

**Três leituras que só o comparativo revela:**

1. **"Usar LLM" não é garantia.** O `gpt-4o-mini` corrige 100% dos casos críticos mas
   **regride** no dataset oficial (65% contra os 75% da V1). Trocar ML por LLM sem medir teria
   piorado o sistema num conjunto e melhorado noutro, sem ninguém perceber.
2. **`gpt-5.6-luna` é o único que ganha nos dois** — confirma a recomendação da pesquisa,
   agora com dado nosso.
3. **A diferença de custo é ~4%, não ordens de grandeza.** Porque a chamada do LLM que
   *executa* a ferramenta (US$ 0,01) domina a conta nas duas versões. **O trade-off real é
   latência, não dinheiro** — e isso inverte a intuição de quem esperava "LLM é caro".

**Modelos descartados por desligamento anunciado:** `gpt-4.1-nano` (23/10/2026) e `gpt-5-nano`
(11/12/2026) são os mais baratos da conta, mas têm data de validade. O código mantém um registro
`MODELOS_DESCONTINUADOS` que avisa quando um deles é escolhido.

---

<a name="adr-20"></a>
## ADR-20 — Plano B contabilizado, nunca silencioso

**Contexto.** A V1.0.1 depende de um provedor externo. Se a rede cair, o atendimento não pode
parar. O plano B natural é usar a decisão local — ou seja, a própria V1.

**Decisão.** Implementar o plano B **e contar cada acionamento**.

**Por quê o contador importa mais do que parece.** Se a V1.0.1 silenciosamente usasse respostas
da V1, estaríamos comparando a V1 **com ela mesma** e chamando isso de melhoria. O contador
`fallbacks` aparece no relatório com uma observação explícita: *"a V1.0.1 respondeu como a V1 em
N mensagens, o que reduz a diferença medida entre as duas"*.

**Camadas de proteção contra alucinação:**
1. A rota precisa ser um dos dois valores válidos.
2. Ao menos uma ferramenta citada precisa existir entre as candidatas.
3. Na aplicação da resposta, cada nome é filtrado de novo — uma lista pode conter um nome
   inventado no meio de nomes válidos.

Executar uma ferramenta inventada seria um erro em produção, não uma imprecisão.

---

<a name="adr-21"></a>
## ADR-21 — Conjunto de teste compartilhado entre as versões

**Contexto.** O pedido original era 5 personas + 2 avaliadores para **cada** versão — 14
execuções de subagente.

**Decisão.** As personas geram o conjunto **uma vez**; as mesmas mensagens rodam nas duas
versões; os avaliadores julgam a comparação. Total: 7 subagentes.

**Por quê — e o motivo principal não é custo.** Se cada versão fosse testada com mensagens
diferentes, uma diferença de acerto poderia vir do **conjunto de mensagens** em vez do sistema.
Usando exatamente as mesmas entradas, a única variável que resta é a versão. Em experimento,
isso se chama **controle**, e sem ele a comparação não conclui nada.

O custo menor (metade das execuções) é consequência, não a razão.

**Consequência.** Ganhamos uma métrica que o desenho original não permitiria: a **divergência
por caso** — quais mensagens uma versão acerta e a outra erra. É a única informação que fala
diretamente sobre a diferença entre as duas.

---

<a name="adr-22"></a>
## ADR-22 — 5 personas com perfis fechados

**Contexto.** Precisamos de mensagens de teste que se pareçam com clientes reais.

**Decisão.** Cinco agentes, cada um preso a um perfil de escrita, em vez de um agente gerando
tudo.

**Por quê.** Um único agente com "gere 300 mensagens de clientes de banco" produz 300 variações
do **mesmo registro** — provavelmente o registro neutro e bem escrito que domina seus dados de
treino. Foi exatamente esse o ponto cego que derrubou a V1: ela ia bem com quem escrevia parecido
com os 53 exemplos, e mal com todo o resto.

**Os perfis, e o desempenho da V1 em cada um:**

| Persona | Desempenho na V1 |
|---|---|
| `jovem_girias` | 54% ← melhor (parecido com o treino) |
| `leigo_idoso` | 39% |
| `dedos_gordos` | 32% |
| `especialista_bancario` | **25%** ← pior |
| `caotico` | robustez, não acurácia |

As duas **pontas** do espectro falharam mais. Ambas se afastam do registro médio do treino, em
direções opostas.

**Detalhe de implementação que importa:** cada persona recebe **exemplos concretos** do estilo,
não só a descrição. Descrever um estilo em palavras é ambíguo; mostrar três frases não é.

---

<a name="adr-23"></a>
## ADR-23 — 2 avaliadores com lentes complementares

**Decisão.** Dois avaliadores com prompts **diferentes**:

- **`avaliador_metrico`** — *"os números sustentam a conclusão?"* Rigor estatístico, viés de
  medição, vazamento, tamanho de amostra, se a comparação é justa.
- **`avaliador_producao`** — *"isso aguenta o mundo real?"* Latência, custo em escala, modos de
  falha, segurança, dependência de terceiro, operação.

**Por quê.** Dois avaliadores com o **mesmo** prompt dariam praticamente a mesma resposta duas
vezes — o dobro do custo, zero cobertura adicional. Redundância só vira valor quando as
perspectivas são distintas.

As duas lentes cobrem as duas formas de uma decisão de arquitetura dar errado: **(a) os números
mentem**, ou **(b) os números estão certos mas o sistema quebra em produção**.

**Contra o viés de concordância.** O caminho de menor esforço de um LLM é concordar com a
narrativa que recebe. Os prompts mandam **procurar ativamente motivos para a V1.0.1 não valer a
pena**, e as versões chegam identificadas só por nome — nada de "a versão nova e melhorada".

---

<a name="adr-24"></a>
## ADR-24 — Avaliador ≠ avaliado

**Decisão.** Os subagentes usam `OPENAI_MODEL_AGENTS`, **nunca** `OPENAI_MODEL_ORCHESTRATOR`.

**Por quê.** LLMs tendem a preferir as próprias saídas — *viés de auto-preferência*. Se o mesmo
modelo escolhesse a ferramenta e depois julgasse se a escolha foi boa, a nota estaria viciada a
favor.

**Limitação que permanece, e que declaramos.** Orquestrador e avaliadores são ambos da OpenAI,
de famílias próximas (`gpt-5.6-luna` e `gpt-5.6-sol`). Isso reduz o viés, mas não o elimina —
modelos da mesma família compartilham dados de treino e tendências. Um controle mais forte usaria
um provedor diferente. Fica registrado como limitação conhecida, não resolvida.

---

<a name="adr-25"></a>
## ADR-25 — Google ADK: **descartado a pedido**

**Contexto.** O plano inicial era construir os subagentes com o Agent Development Kit do Google.

**O que aconteceu.** Verificamos: o SDK **não estava instalado** em nenhum Python da máquina, nem
via npm. Instalamos (`google-adk` 2.7.0) e mapeamos a API — `LlmAgent`, `ParallelAgent`,
`SequentialAgent`, `InMemoryRunner`, e `LiteLlm` para apontar a modelos não-Google. O
`ParallelAgent` seria adequado para rodar as 5 personas.

**Decisão.** Descartado, por mudança de plano do time: usar o SDK da OpenAI.

**Consequência positiva.** Some a segunda credencial e o projeto fica com **um provedor só** —
menos superfície de configuração, menos segredo para gerenciar. A configuração morta
(`GOOGLE_API_KEY`, `ADK_MODEL`) foi **removida** de `config.py`, `.env` e `.env.example`;
configuração morta é armadilha de documentação.

**Débito assumido.** Os pacotes `google-adk` e `litellm` continuam instalados no ambiente virtual,
sem uso. São inofensivos, e mantê-los deixa o caminho aberto caso a decisão mude. Se a V2 fechar
sem eles, devem sair do ambiente.

**Nota de método.** Não usamos framework de agentes nesta camada, nem da OpenAI. Frameworks
agregam valor quando há ferramentas, memória entre turnos e decisão de próximo passo. Aqui cada
agente faz **uma** chamada e devolve um JSON. Um framework adicionaria dependência e indireção
sem resolver problema nenhum que exista.

---

<a name="adr-26"></a>
## ADR-26 — Rótulos "prata" reportados separados dos "ouro"

**Contexto.** As personas rotulam cada mensagem com a rota e a ferramenta esperadas. Esses
rótulos vêm de um LLM.

**Decisão.** Distinguir e **nunca somar**:

- **Ouro** — escrito por humano. É o caso do `eval_dataset.json` oficial (30 mensagens).
- **Prata** — gerado por modelo. Útil em escala, mas pode errar.

O relatório traz os dois blocos **lado a lado**, sempre marcados.

**O risco específico que isso mitiga.** As personas usam um modelo da OpenAI, e a V1.0.1 também.
Se os dois compartilharem o mesmo viés sobre qual ferramenta é "a certa", a V1.0.1 leva vantagem
indevida no conjunto das personas. Por isso o dataset oficial — cujos rótulos nenhum modelo
escreveu — permanece como a referência principal, apesar de menor.

**Proteção adicional no código:** rótulo de ferramenta inexistente (alucinação da persona) é
**descartado**. Preferimos perder o rótulo a carregar um gabarito errado — uma ferramenta
inventada faria as duas versões errarem por igual e poluiria a comparação com ruído.

---

## Limitações assumidas na V1.0.1

1. **A latência subiu de milissegundos para ~1,2 s.** É o custo real desta versão, e é o que
   precisa de aprovação — não o financeiro, que ficou em ~4%.
2. **O teto de recuperação não mudou.** A V1.0.1 conserta julgamento, não recuperação. Se a
   ferramenta certa não entra nas 20 candidatas, nada a salva.
3. **Dependência de provedor externo** no caminho da requisição, com todos os modos de falha que
   isso implica.
4. **Viés de família compartilhada** entre orquestrador e avaliadores ([ADR-24](#adr-24)).
5. **O vazamento treino/eval da V1 continua valendo** no dataset oficial e favorece a V1 nesse
   conjunto — o que torna a vitória da V1.0.1 nele mais significativa, não menos.
