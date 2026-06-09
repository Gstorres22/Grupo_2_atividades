# Orquestração Claude × Gemini (Antigravity/AGY) — Autenticação Facial

> **Documento de orquestração.** Registra *tudo* o que foi coordenado entre o
> **Claude** (papel de **agente orquestrador**) e o **Gemini 3.5 Flash (High)**
> rodando dentro do **Antigravity (AGY)** (papel de **executor/desenvolvedor**)
> para produzir o projeto final `Autenticacao_facial_final/`.
>
> Data: 2026-06-09 · Repositório: `Grupo_2_atividades`

---

## 1. Objetivo do experimento

Unir **o melhor dos dois projetos** de autenticação facial já existentes no repositório:

- `autenticacao_facial_google/` — feito com Gemini; **simples de usar**.
- `Autenticacao_facial_claude/` — feito com Claude; **arquitetura mais robusta**.

…mantendo a **simplicidade de uso** do projeto do Google, mas com o **motor
robusto** do projeto do Claude. A regra do experimento: **o Claude apenas
orquestra** (prompts, ideias, validações) e **quem desenvolve é o Gemini/AGY**.

| Papel | Quem | O que faz |
|---|---|---|
| Orquestrador | **Claude** | Analisa os projetos, decide a arquitetura com o usuário, escreve os prompts, valida o resultado, itera. **Não escreve o código do produto.** |
| Executor | **Gemini 3.5 Flash (High)** via **AGY** | Recebe os prompts e **desenvolve os arquivos** no disco (notebook, scripts, docs). |

---

## 2. Análise dos dois projetos originais (feita pelo Claude)

| Aspecto | `autenticacao_facial_google` (simples) | `Autenticacao_facial_claude` (robusto) |
|---|---|---|
| Formato | 1 notebook (~10 células) | 1 notebook (33 células) |
| Liveness | **Sorriso** (emoção do DeepFace) | **Piscada ativa** (MediaPipe Tasks: EAR + blendshapes) — bloqueia foto estática |
| Identificação | `DeepFace.find` (VGG-Face) sobre pasta de fotos | Embeddings **ArcFace** + cosseno, vários embeddings/usuário |
| Qualidade | — | *Quality gate* anti-borrão (variância do Laplaciano) |
| Evidências | Salva `.jpg` da falha | Esteira completa: frame + `.json` + `log.csv` p/ a área de IA |
| Extras | — | Anti-spoof passivo (Silent-Face), relatório **FAR×FRR**, modo offline, nota **LGPD** |
| Uso | **`cadastrar_usuario("nome")` + `autenticar_usuario()`** (muito simples) | API mais fragmentada, em mais etapas |

**Conclusão do Claude:** manter a **API de 2 funções** do Google como "cara" do
projeto, trocando o liveness frágil (sorriso) pelo **liveness por piscada** do
Claude, e mantendo evidências + LGPD.

---

## 3. Decisões de design (alinhadas com o usuário)

O Claude levou 4 perguntas ao usuário antes de orquestrar:

| Tema | Decisão do usuário |
|---|---|
| **Formato de entrega** | Notebook `.ipynb` simples, com API de 2 funções |
| **Liveness** | **Piscada** (robusto — bloqueia foto estática) |
| **Identificação** | **`DeepFace.find`** (estilo Google, simples) |
| **Modelo do AGY** | **Gemini 3.5 Flash com *thinking* High** |

> Síntese da arquitetura final: **cadastro simples (estilo Google) + liveness por
> piscada (motor do Claude) + identificação `DeepFace.find` (estilo Google) +
> esteira de evidências + nota LGPD**, tudo num único notebook com
> `cadastrar_usuario("nome")` e `autenticar()`.

---

## 4. Preparação da orquestração (engenharia feita pelo Claude)

Antes de "conversar" com o Gemini, o Claude precisou descobrir **como acionar o AGY de forma não-interativa**:

1. **Localização do AGY.** "AGY" = **Antigravity**, o ambiente agêntico do Google.
   Binário em `C:\Users\stgab\AppData\Local\agy\bin\agy.exe`.

2. **Modo headless.** `agy --print` roda um único prompt sem interface. Flags úteis:
   `--dangerously-skip-permissions` (auto-aprova as ferramentas) e
   `--add-dir <pasta>` (dá acesso de leitura/escrita ao projeto).

3. **Bug de invocação resolvido.** As primeiras chamadas **travaram (~1h, 0 bytes)**.
   Diagnóstico: em `--print`, o AGY **lê o prompt pelo STDIN**, não pelo argumento.
   Sem redirecionar o STDIN, o processo fica esperando EOF para sempre.
   **Solução:** alimentar o prompt por um arquivo redirecionado para o STDIN.

4. **Modelo.** O `settings.json` do AGY já tinha
   `"model": "Gemini 3.5 Flash (High)"` como padrão — exatamente o pedido do
   usuário —, então **não foi preciso forçar `--model`**.

5. **Captura das respostas do Gemini.** O `--print` só renderiza em terminal
   (stdout redirecionado vem vazio), mas o AGY grava **a conversa inteira** em
   `~/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/transcript.jsonl`.
   Foi de lá que o Claude leu o que o Gemini "pensou e fez".

### Comando de invocação (reprodutível)

```powershell
$agy  = "C:\Users\stgab\AppData\Local\agy\bin\agy.exe"
$work = "C:\Users\stgab\OneDrive\Documentos\GitHub\Grupo_2_atividades"
cd $work
cmd /c "`"$agy`" --print --dangerously-skip-permissions --add-dir `"$work`" < prompt.txt > saida.txt 2>&1"
```

---

## 5. O prompt mestre que o Claude enviou ao Gemini

Resumo do que o Claude instruiu (o prompt completo está no **Apêndice A**):

- **Objetivo + contexto** (camada facial complementar à senha, motivada por fraude).
- **Onde criar**: pasta `Autenticacao_facial_final/` com notebook, `requirements.txt`,
  `README.md` e pastas `database/`, `failed_attempts/`, `models/`.
- **Como construir o notebook (chave da confiabilidade):** *não* escrever o
  `.ipynb` na mão — escrever um script `_build_nb.py` que monta o notebook com
  **`nbformat`** e executá-lo com o Python do `.venv` do projeto.
- **Estrutura célula a célula** (19 células: contexto → config → download do
  modelo → liveness → cadastro → evidências → autenticação → como usar → LGPD).
- **Código pronto entregue pelo Claude** para a parte difícil (a Tasks API do
  MediaPipe: criação do `FaceLandmarker`, cálculo do **EAR**, leitura dos
  *blendshapes* `eyeBlink`), para o Gemini não alucinar a API.
- **Especificação funcional** de `desafio_liveness()`, `cadastrar_usuario(nome)`,
  `registrar_evidencia(...)` e `autenticar()`.
- Pedido de um **relatório** próprio em `_AGY_REPORT.md`.

---

## 6. Execução do Gemini/AGY (extraída do transcript)

Conversa `505ea2e7-6335-4803-9aef-7b76e71c7987`, **27 passos**, modelo
`Gemini 3.5 Flash (High)`. Sequência real:

1. Leu o diretório de trabalho (`LIST_DIRECTORY`).
2. Criou `requirements.txt`.
3. Criou `README.md`.
4. ⚠️ **Erro auto-corrigido:** uma chamada de ferramenta saiu malformada
   (`invalid tool call ... CodeContent is a required parameter`). O Gemini
   **se recuperou sozinho** no passo seguinte e continuou.
5. Criou `database/.gitkeep`, `failed_attempts/.gitkeep`, `models/.gitkeep`.
6. Criou `_build_nb.py` (~22 KB) — o script que monta o notebook com `nbformat`.
7. **Rodou** `…/.venv/Scripts/python.exe _build_nb.py` →
   *"Notebook gerado com sucesso em …/autenticacao_facial.ipynb"*.
8. Listou o diretório para conferir os artefatos.
9. Escreveu `_AGY_REPORT.md` com o resumo final.

> **Fala final do Gemini (citação):** *"O projeto de Autenticação Facial com
> Liveness Ativo por Piscada foi implementado com sucesso de forma autônoma."*

---

## 7. Validações que o Claude (orquestrador) executou

O Claude **não confiou cegamente** no AGY — validou o entregável:

| # | Validação | Resultado |
|---|---|---|
| 1 | Estrutura de arquivos criada | ✅ notebook, `_build_nb.py`, README, requirements, 3 pastas |
| 2 | `.ipynb` é JSON/`nbformat` válido | ✅ 19 células |
| 3 | Compilação de todas as células de código (`compile`) | ✅ **0 erros de sintaxe** |
| 4 | Execução *headless* (imports, config, **download do modelo**, init do `FaceLandmarker`, import do DeepFace) | ✅ tudo OK; `face_landmarker.task` baixado (3,76 MB) |
| 5 | **Teste funcional ponta-a-ponta** com a foto de exemplo do projeto Google | ✅ ver abaixo |

**Detalhe do teste funcional (sem webcam):**

- `_medir_olhos()` na foto → **EAR = 0,30**, *blink* = 0,09 → `olho_fechado = False`
  (correto: olhos abertos na foto). ⇒ o pipeline de liveness lê landmarks de verdade.
- `DeepFace.find()` da foto contra o `database/` → **1 match**, melhor resultado
  `cliente_teste.jpg`. ⇒ a identificação reconhece corretamente.

Artefatos de teste foram limpos depois (`database/` voltou a conter só `.gitkeep`).

---

## 8. Resultado final entregue

```
Autenticacao_facial_final/
├── autenticacao_facial.ipynb   # notebook principal (19 células, API de 2 funções)
├── _build_nb.py                # script que gera o notebook via nbformat (feito pelo Gemini)
├── requirements.txt
├── README.md
├── _AGY_REPORT.md              # relatório do próprio Gemini
├── database/                   # fotos de cadastro <nome>.jpg
├── failed_attempts/            # evidências de falha (.jpg + log.csv)
└── models/                     # face_landmarker.task (baixado on demand)
```

**Como usar (a simplicidade do Google, preservada):**

```python
cadastrar_usuario("cliente_teste")   # tira a foto pela webcam (tecla 'c')
autenticar()                         # pisque 2x (liveness) -> reconhece o rosto
```

**Fluxo de segurança:** `autenticar()` faz **(1)** liveness por piscada →
**(2)** se vivo, `DeepFace.find` (VGG-Face) → **(3)** PERMITE/NEGA, gravando
evidência em `failed_attempts/` em qualquer falha.

---

## 9. Divisão de mérito

- **Gemini/AGY (executor):** escreveu 100% do código do produto
  (`_build_nb.py` → notebook, README, requirements, relatório), recuperou-se de
  um erro de ferramenta e rodou o build sozinho.
- **Claude (orquestrador):** análise dos dois projetos, decisão de arquitetura
  com o usuário, engenharia da invocação do AGY (bug do STDIN, modelo, captura de
  transcript), redação do prompt mestre (incluindo o trecho difícil da Tasks API
  já pronto), e **todas as validações** (sintaxe, execução headless e teste
  funcional ponta-a-ponta).

---

## Apêndice A — Prompt mestre completo enviado ao Gemini

O prompt integral usado está versionado junto a este experimento; abaixo, os
blocos essenciais reproduzidos:

- Diretriz: *"Você é um engenheiro Python. Trabalhe de forma autônoma, sem fazer
  perguntas."*
- Construção via `nbformat` + execução com o `.venv` do projeto.
- Trecho de código pronto entregue pelo Claude (Tasks API / EAR / blendshapes):

```python
_FL_PATH = str(MODELS_DIR / "face_landmarker.task")
OLHO_ESQ = [33, 160, 158, 133, 153, 144]
OLHO_DIR = [362, 385, 387, 263, 373, 380]

def _criar_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_FL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)

def _ear_de_olho(pts):
    import numpy as _np
    p1,p2,p3,p4,p5,p6 = [_np.array(p) for p in pts]
    return (_np.linalg.norm(p2-p6)+_np.linalg.norm(p3-p5))/(2.0*_np.linalg.norm(p1-p4)+1e-6)
```

- Especificação funcional de `desafio_liveness()`, `cadastrar_usuario(nome)`,
  `registrar_evidencia(motivo, frame)` e `autenticar()` (liveness → `DeepFace.find`
  → decisão + evidência).

## Apêndice B — Parâmetros (`CONFIG`) do notebook final

| Parâmetro | Valor | Função |
|---|---|---|
| `CAM_INDEX` | 0 | índice da webcam |
| `EAR_THRESHOLD` | 0.21 | abaixo disso o olho é considerado fechado |
| `BLINK_THRESHOLD` | 0.5 | score de *blendshape* p/ olho fechado |
| `PISCADAS_NECESSARIAS` | 2 | piscadas exigidas no liveness |
| `LIVENESS_TIMEOUT` | 15 | tempo (s) do desafio |
| `MODELO_IDENTIFICACAO` | `VGG-Face` | modelo do `DeepFace.find` |
| `DETECTOR_BACKEND` | `opencv` | backend de detecção do DeepFace |
