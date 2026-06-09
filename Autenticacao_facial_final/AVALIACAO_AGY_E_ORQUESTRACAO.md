# Avaliação honesta: AGY (Antigravity CLI), o modelo executor e a orquestração de agentes

> Relatório pedido pelo usuário **sem clubismo**. Baseado em (1) ter de fato
> orquestrado o AGY para gerar o projeto `Autenticacao_facial_final/` e
> (2) leitura de documentação.
> Data: 2026-06-09.
>
> ⚠️ **Correção importante (atualizado):** as §1–§7 abaixo foram escritas com a
> doc do **Gemini CLI** (`geminicli.com/docs`), passada por engano. Com a doc
> **certa do Antigravity** (`antigravity.google/docs`), **a §8 (Adendo) corrige e
> revisa** as conclusões — em especial sobre **tokens e observabilidade**, que
> mudam de figura por causa do **Antigravity SDK**. Leia a §8 como a palavra final.

---

## 0. TL;DR

- O par **"Claude planeja/valida + Gemini executa"** funciona, mas o ganho **não
  é economia de token** — é **especialização** e **deslocamento de custo** (a
  geração pesada roda no tier gratuito do Gemini em vez de no Claude).
- **A parte difícil foi minha**, não do Gemini: entreguei o código da Tasks API
  do MediaPipe pronto no prompt. Com um prompt vago, o Flash provavelmente teria
  feito algo mais simples (como o próprio notebook original do Google, que usou
  sorriso em vez de piscada).
- O **modelo executor (Gemini 3.5 Flash High) foi bom e confiável** no que foi
  bem especificado, e até se auto-corrigiu de um erro. Rápido.
- O **AGY instalado é imaturo como ferramenta headless**: stdout vazio, sem
  `--output-format`/estatísticas de token, trava sem redirecionar stdin. Para um
  orquestrador, isso é **falta de observabilidade**.
- A **documentação do Gemini CLI promete bem mais** do que o AGY instalado
  entrega hoje (JSON com tokens, subagents, hooks). Há divergência real entre o
  `gemini` documentado e o `agy` da sua máquina.

---

## 1. O que é o AGY, afinal (e a divergência com a doc)

"AGY" é o **Antigravity CLI** (`agy.exe`). A própria documentação do Gemini CLI
avisa: *"Gemini CLI will be replaced by Antigravity CLI on June 18th"*. Ou seja,
**o AGY é o sucessor do Gemini CLI** — mas o build instalado na sua máquina expõe
um conjunto de flags **menor e diferente** do que a doc descreve:

| Recurso | Gemini CLI (doc) | AGY instalado (`agy --help`) |
|---|---|---|
| Prompt headless | `-p/--prompt` (+ lê stdin) | `-p/--print` (lê o prompt pelo **stdin**) |
| Saída estruturada | `--output-format text\|json\|stream-json` (com **stats de token**) | ❌ **não existe** (testei: retorna 0 bytes) |
| Auto-aprovar tools | `--approval-mode=yolo` / `-y` | `--dangerously-skip-permissions` |
| Incluir pastas | `--include-directories` | `--add-dir` |
| Retomar sessão | `--resume` / `--list-sessions` | `--continue` / `--conversation` |
| Subcomandos | `update, extensions, mcp, skills` | `update, plugin, models, install, changelog` |

**Conclusão prática:** ao usar o AGY, vale o `agy --help` da máquina, **não** a
doc ao pé da letra. Muita coisa da doc (subagents, hooks, JSON) é conceitualmente
válida mas **não está acessível** nesse build via CLI.

---

## 2. O que aprendi sobre acionar o AGY (engenharia real)

1. **Prompt entra pelo STDIN, não pelo argumento.** Em `--print`, sem redirecionar
   o stdin de um arquivo, o processo **trava esperando EOF** (travou ~1h, 0 bytes,
   log vazio). A doc confirma a semântica: o `-p` é *"appended to stdin input"*.
   - ✅ Forma correta: `agy --print --dangerously-skip-permissions --add-dir <proj> < prompt.txt`
2. **O stdout redirecionado vem vazio.** O AGY só renderiza a resposta no
   terminal (TTY). Para "ler" o que o Gemini fez, é preciso abrir o transcript em
   `~/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/transcript.jsonl`
   (e `transcript_full.jsonl`, que inclui até o **raciocínio/thinking** do modelo).
3. **Sem visibilidade de token.** Como você está autenticado por **OAuth** (conta
   Google), o *token caching* e o `/stats` da doc **não se aplicam** (a doc diz
   explicitamente que OAuth/Code Assist não suporta cache). Não há número de
   tokens em lugar nenhum nesse fluxo headless.
4. **Modelo padrão** fica em `~/.gemini/antigravity-cli/settings.json`
   (`"model": "Gemini 3.5 Flash (High)"`). Dá para sobrescrever com `--model`.
5. **`agy models` não imprime nada** quando redirecionado (só TTY).

---

## 3. Avaliação do MODELO executor (Gemini 3.5 Flash – High)

**Pontos fortes (honestos):**
- **Confiável sob spec detalhada.** Seguiu o plano célula a célula, usou
  `nbformat` como pedido, e **rodou o build sozinho** (`python _build_nb.py`)
  conferindo o sucesso.
- **Auto-correção.** Emitiu uma chamada de ferramenta malformada
  (`CodeContent is a required parameter`) e **se recuperou sozinho** no passo
  seguinte, sem intervenção.
- **Qualidade dos textos.** README e relatório próprio (`_AGY_REPORT.md`) ficaram
  claros e corretos.
- **Velocidade.** O build inteiro (9 arquivos/ações) levou ~45s de relógio.

**Pontos fracos / ressalvas (sem clubismo):**
- **O mérito técnico do núcleo é meu, não dele.** Entreguei no prompt o código
  pronto da Tasks API (criação do `FaceLandmarker`, cálculo do EAR, leitura de
  blendshapes). O Flash **montou**, não **inventou**. Evidência: o notebook que o
  Gemini fez *sem essa orientação* (o `autenticacao_facial_google` original) usou
  a abordagem mais simples (sorriso + `DeepFace.find` no detector legado). Ou
  seja, **livre, o Flash tende ao simples**; a robustez veio da minha
  especificação.
- **Não validou em runtime.** Ele confirmou que o `_build_nb.py` rodou, mas não
  executou as células do notebook nem testou o pipeline. **Quem validou** (import,
  download do modelo, init do landmarker, EAR numa foto real, `DeepFace.find`
  ponta a ponta) **fui eu**.
- É um modelo **Flash** (rápido/barato), adequado para execução guiada; eu **não**
  confiaria nele para decisões de arquitetura abertas sem revisão.

---

## 4. Avaliação da FERRAMENTA AGY como alvo de orquestração

| Critério | Nota honesta | Comentário |
|---|---|---|
| Capacidade do agente (tools, planejar, rodar comando) | **Boa** | Cria arquivos, roda terminal, planeja, tem thinking. |
| Ergonomia headless | **Fraca** | Trava sem stdin redirecionado; stdout vazio; sem JSON. |
| Observabilidade (o que ele fez/gastou) | **Fraca** | Só via transcript no disco; zero métrica de token no OAuth. |
| Estabilidade | **Boa** | Exit codes corretos, build reproduzível. |
| Aderência à documentação | **Média** | Flags divergem; recursos da doc ausentes no build. |
| Custo (no seu setup OAuth) | **Ótimo** | Gratuito; é o grande atrativo do deslocamento de carga. |

Para orquestração **séria** (um programa dirigindo o AGY), a falta de
`--output-format json` e de stdout capturável é o maior incômodo: você fica
dependente de fazer *scraping* dos transcripts. Quando o build com
`--output-format` (prometido na doc) chegar a esse binário, a história muda.

---

## 5. Custo de token: e se fosse só o Claude fazendo tudo?

**Não dá para comparar com número exato do Gemini** — o AGY/OAuth não expõe
tokens (ver §2.3). Então segue uma **estimativa fundamentada do MEU lado**.

Artefatos gerados (tamanho real):

| Arquivo | Bytes |
|---|---|
| `autenticacao_facial.ipynb` | 26.711 |
| `_build_nb.py` | 22.270 |
| `README.md` | 4.079 |
| `_AGY_REPORT.md` | 3.160 |
| `requirements.txt` | 101 |

**Se eu (Claude) tivesse feito tudo sozinho**, o trabalho útil seria gerar **um**
veículo (o `_build_nb.py` *ou* o `.ipynb` direto) + README + requirements, mais
leitura dos 2 notebooks-fonte e 1–2 iterações de correção:

- Saída gerada: ~**8–10k tokens** (o `_build_nb.py` de 22 KB ≈ 6–7k; README ≈
  1,2k; resto pouco), +1–2 ciclos de ajuste ≈ **12–18k tokens de saída**.
- Entrada (ler os 2 notebooks + iterar): ~**10–20k tokens**.
- **Total atribuível à tarefa de build: ~30–60k tokens** (fora o overhead fixo de
  contexto por turno, que existe nas duas abordagens).

**O ponto honesto e contra-intuitivo:** a orquestração **não me poupou tokens**.
Nesta rodada eu gastei do **meu** lado com: analisar os 2 projetos, escrever um
prompt mestre de ~1.500 palavras (~2–3k de saída), **validar pesado** (rodar
código, ler transcripts) e escrever os MDs (~4k de saída). Isso é **comparável ou
maior** do que eu escrever o notebook direto. O que mudou foi **onde** a geração
pesada aconteceu: **no Gemini (grátis), não no Claude**.

> **Veredito de custo:** para **um** notebook pequeno, orquestrar custou *mais*
> trabalho/token **somando os dois sistemas** do que eu fazer direto. A
> orquestração compensa quando (a) você quer **deslocar custo** para um executor
> barato/gratuito, (b) há **paralelismo/escala** (muitas tarefas), ou (c) o
> executor tem **especialização** que você não tem. Não compensa como atalho de
> token para tarefas únicas e pequenas.

---

## 6. Quando vale (e quando não vale) esse modelo de orquestração

**Vale a pena:**
- Tarefas **bem especificáveis** e repetitivas (gerar N módulos, boilerplate,
  scaffolding, conversões em lote).
- Quando o custo de billing importa e o executor é **gratuito/barato** (seu caso OAuth).
- Quando dá para rodar **em paralelo/background** enquanto o orquestrador cuida de
  outra coisa.

**Não vale a pena:**
- Tarefa **única e pequena** (o overhead de prompt+validação supera o ganho).
- Decisões de **arquitetura aberta** que exigem julgamento — o executor Flash
  precisa ser guiado, e guiá-lo bem custa quase tanto quanto fazer.
- Quando você precisa de **observabilidade fina** (tokens, resposta estruturada) —
  o AGV headless atual não entrega.

---

## 7. Receita recomendada para projetos futuros (Claude × AGY)

1. **Claude** analisa, decide com você e escreve um **prompt mestre prescritivo**
   (entregando os trechos difíceis prontos).
2. Invocar: `agy --print --dangerously-skip-permissions --add-dir <proj> < prompt.txt`
   (sempre stdin via arquivo; rodar no diretório do projeto).
3. **Claude lê o transcript** (`brain/<id>/.../transcript.jsonl`) para saber o que
   o Gemini fez/pensou.
4. **Claude valida de verdade** (compila, executa o que dá sem hardware, testa
   ponta a ponta) — **não** confiar no "deu certo" do executor.
5. Iterar com prompts de correção curtos, referenciando arquivos por caminho.
6. Se/quando o binário ganhar `--output-format json`, **migrar para JSON** para
   capturar resposta + tokens automaticamente.

---

## 8. ADENDO — lendo a doc CERTA (Antigravity) e corrigindo o relatório

Fonte: `https://antigravity.google/docs` (SPA em JS — extraí o conteúdo via busca
indexada, pois o fetch direto só retorna o título). O que muda:

### 8.1 O que eu errei / suavizo

- **"Não há visibilidade de token / saída estruturada" → parcialmente FALSO.**
  Isso vale para o **print-mode do CLI** (que usei), mas **não** para a
  plataforma. Existe o **Antigravity SDK** (Python, `pip install google-antigravity`),
  que é o caminho programático de verdade e expõe:
  - **Saída estruturada** por schema (JSON/dict/**Pydantic**) via `response.structured_output()`.
  - **Streaming**: `async for chunk in response`.
  - **Tokens**: `usage_metadata` com uso **por turno e acumulado** — *prompt,
    candidate, cached e thinking tokens*. ⇒ **dá, sim, para medir token** — só
    não pelo CLI `--print` que usei.
  - Registrar qualquer callable Python como **tool** do agente.
- **Orquestração é mais rica do que a abordagem "vários processos `agy`" que usei.**
  O modelo nativo é o **agente primário spawnar subagentes**:
  - **Subagentes** com **isolamento de workspace** (não poluem o contexto do
    agente principal; paralelizam tarefas pesadas).
  - **Tarefas assíncronas em background**: o handler principal invoca o subagente
    e **devolve o controle na hora**; um *message client* interceptador faz
    *stream* das respostas do subagente de volta ao log do principal.
  - **Hooks** (scripts shell locais) em pontos do ciclo: *antes/depois de uma
    tool*, *antes/depois da chamada do modelo*, e em *condições de parada*. Global
    e por workspace (JSON). ⇒ eu poderia ter logado tokens/ações com um hook
    "after model call".
  - **Scheduled Tasks** (`/schedule`): crons de prompts recorrentes.
  - Slash commands: `/agents`, `/config`, `/browser`, `/schedule`.

### 8.2 O que se confirma (continua valendo)

- O **binário `agy` instalado** realmente é uma superfície **enxuta**: `--print`
  lê stdin, **sem `--output-format`**, stdout vazio, `models` mudo. Para
  observabilidade fina **via CLI**, segue ruim — a saída é o SDK ou hooks.
- **Config** em `~/.gemini/antigravity-cli/settings.json` (modelo, etc.).
- O modelo **Flash livre tende ao simples**; a robustez veio do meu prompt.
- A **divisão de mérito** e a conclusão de **custo** (orquestrar não poupou MEU
  token; deslocou a geração pesada pro Gemini) seguem de pé.

### 8.3 Recomendação revisada para projetos futuros

| Necessidade | Ferramenta certa |
|---|---|
| Rodada rápida, guiada, "faça este arquivo" | **CLI `agy --print`** via stdin (o que fizemos) |
| Orquestração **com métricas de token** e saída tipada | **Antigravity SDK** (Python) — `usage_metadata` + `structured_output()` |
| Paralelizar muitas subtarefas | **subagentes**/background do próprio agente, não N processos `agy` |
| Logar/auditar cada passo e tokens | **hooks** "after model call" / "after tool" |
| Rotina recorrente | **Scheduled Tasks** (`/schedule`) |

> **Conclusão revisada:** minha crítica de "falta observabilidade" era válida
> **para o CLI print-mode**, mas **injusta como veredito da plataforma**. Para
> orquestração séria — inclusive para **responder com precisão "quantos tokens o
> Gemini gastou"** — o caminho é o **Antigravity SDK** (`usage_metadata`), não o
> `agy --print`. Se você topar, num próximo projeto eu monto a orquestração pelo
> SDK e aí entrego os **números reais de token** dos dois lados, não estimativa.

### Fontes (Antigravity)
- Blog CLI: `antigravity.google/blog/introducing-google-antigravity-cli`
- Blog SDK: `antigravity.google/blog/introducing-google-antigravity-sdk`
- Deep-dive de features: `antigravity.google/blog/google-io-2026-feature-deep-dive`
- Docs: `antigravity.google/docs/cli-features`, `/docs/sdk-overview`, `/docs/agent-manager`, `/docs/skills`
