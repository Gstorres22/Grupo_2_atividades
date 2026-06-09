# Plano de Ação — Sistema de Autenticação Facial

## Contexto

O setor de fraudes identificou contratações indevidas (ex.: crédito pessoal) em que a **senha foi validada corretamente**, mas o cliente nega ter contratado o serviço. A senha sozinha não prova *presença* nem *identidade física* de quem opera. A proposta é adicionar uma **camada de autenticação facial complementar**, acionada conforme o serviço, com três capacidades:

1. **Detecção de faces** — localizar rosto(s) no quadro.
2. **Identificação de faces** — confirmar que o rosto é do cliente cadastrado.
3. **Liveness (vivacidade)** — garantir que é uma pessoa real ao vivo, e não uma foto estática/print.

Quando o cliente **não é autenticado**, o fluxo o encaminha para uma **esteira dedicada** e grava **evidências** (frame, scores, limiares, motivo) para a **área de IA** ajustar parâmetros e limiares do modelo.

**Decisões aprovadas pelo usuário:**
- Captura: **webcam ao vivo** (OpenCV).
- Identificação: **DeepFace** (embeddings ArcFace; sem dlib; instalação tranquila no Windows).
- Liveness: **ativo por piscada/desafio** (MediaPipe FaceMesh + Eye Aspect Ratio). Foto estática não pisca → bloqueia o fraudador.

Linguagem **Python**, sem interface, **tudo desenvolvido em um único Jupyter Notebook**, executado por código.

---

## Stack tecnológica

| Função | Biblioteca |
|---|---|
| Captura / processamento de imagem | `opencv-python` |
| Detecção de face + landmarks (liveness) | `mediapipe` (Face Detection + Face Mesh, 468 pontos) |
| Identificação (embeddings ArcFace + anti-spoof) | `deepface` (TensorFlow por baixo) |
| Vetores, similaridade, métricas | `numpy`, `scikit-learn` |
| Visualização no notebook | `matplotlib` |

Python 3.10+. Stack escolhida evita `dlib` (que é problemático de compilar no Windows).

---

## Estrutura de diretórios a criar

```
Autenticacao_facial_claude/
├── autenticacao_facial.ipynb     # NOTEBOOK PRINCIPAL (todo o fluxo)
├── requirements.txt
├── README.md
├── PLANO_DE_ACAO.md              # cópia deste plano
├── data/
│   ├── enrolled/<user_id>/       # fotos de cadastro por usuário
│   └── embeddings/               # embeddings salvos (.pkl)
└── evidences/                    # evidências de não-identificação p/ área de IA
```

---

## Estrutura do Notebook (seções/células)

1. **Setup & Imports** — célula de instalação (`pip install -r requirements.txt`) + imports.
2. **Configuração central** — `CONFIG` com paths, `THRESHOLD_SIMILARIDADE`, `EAR_THRESHOLD`, `PISCADAS_NECESSARIAS`, `N_FOTOS_CADASTRO`, `MODELO = "ArcFace"`. Tudo ajustável num só lugar.
3. **Utilitários** — carregar/salvar imagem, *quality gate* (variância do Laplaciano p/ detectar borrão), desenho de bounding box e landmarks.
4. **Detector de Faces** — função sobre MediaPipe Face Detection: recebe frame → retorna faces + caixa; valida "exatamente 1 rosto".
5. **Liveness ativo (piscada)** — MediaPipe FaceMesh → cálculo do **EAR (Eye Aspect Ratio)** → contador de piscadas com janela de tempo. Função `desafio_liveness(timeout)` que só aprova após N piscadas reais.
6. **Identificação** — `extrair_embedding()` (DeepFace/ArcFace), `cadastrar_usuario()` (salva embeddings) e `verificar()` (similaridade de cosseno contra o `user_id` alegado, decisão por limiar).
7. **Fluxo de Cadastro (Enrollment)** — captura N frames pela webcam → detecta → *quality gate* → extrai embeddings → salva conjunto/média por `user_id` em `data/embeddings/`.
8. **Fluxo de Autenticação** — informa `user_id` alegado → captura ao vivo → **detecção** → **desafio de liveness (piscar)** → **embedding + comparação** → **decisão**. Só identifica se passou no liveness.
9. **Esteira dedicada & Evidências** — em qualquer falha (liveness reprovado, rosto não bate, baixa qualidade): salva frame + `evidence.json` (timestamp, user_id, score, limiar, motivo) em `evidences/` e registra linha em `evidences/log.csv`.
10. **Relatório & Ajuste de limiares** — demonstra impacto do limiar (conceito FAR/FRR), visualiza scores de tentativas e como a área de IA usaria as evidências para recalibrar.

> Cada seção terá uma célula explicativa em Markdown (texto didático para a faculdade) seguida do código.

---

## Lógica de decisão

- **Liveness:** EAR cai abaixo de `EAR_THRESHOLD` (~0.21) por alguns frames = 1 piscada. Exige `PISCADAS_NECESSARIAS` (ex.: 2) dentro do `timeout`. Reprovou → esteira dedicada.
- **Identificação:** similaridade de cosseno entre embedding ao vivo e o cadastrado. `>= THRESHOLD_SIMILARIDADE` → autenticado; abaixo → esteira dedicada + evidência.
- **Qualidade:** frames borrados/sem rosto/múltiplos rostos são rejeitados antes da decisão.

---

## Melhorias/sugestões incluídas

- **Múltiplos embeddings por usuário** (várias fotos no cadastro) → compara pelo melhor/média, mais robusto.
- **Quality gate** (anti-borrão) para reduzir falsos negativos.
- **Limiares configuráveis** numa única célula `CONFIG`.
- **Anti-spoof passivo opcional** — DeepFace traz Silent-Face "de graça"; deixo uma célula opcional combinando com o liveness ativo, gerando mais sinal para a área de IA.
- **Modo offline opcional** — célula que roda o fluxo a partir de imagens/vídeo gravados, garantindo reprodutibilidade na apresentação caso a webcam falhe.
- **Privacidade/LGPD** — recomendação de armazenar *embeddings* em vez de fotos cruas (nota no README), pertinente ao contexto bancário/fraude.

---

## Verificação (como testar de ponta a ponta)

1. Rodar a célula de instalação e imports (sem erros).
2. **Cadastro:** executar o fluxo de enrollment olhando para a webcam → confirmar arquivos em `data/embeddings/<user_id>.pkl`.
3. **Autenticação OK:** rodar o fluxo informando o próprio `user_id`, **piscar** quando solicitado → resultado "AUTENTICADO".
4. **Liveness:** apontar uma **foto estática** do rosto → não pisca → "REPROVADO (liveness)" + evidência gravada em `evidences/`.
5. **Identificação negativa:** autenticar com `user_id` de outra pessoa → "REPROVADO" + evidência.
6. Conferir `evidences/log.csv` e os `evidence.json` com scores e motivos.

---

## Entregáveis

- `autenticacao_facial.ipynb` completo e comentado (todo o fluxo).
- `requirements.txt`, `README.md` (como rodar + nota LGPD), `PLANO_DE_ACAO.md`.
- Estrutura de pastas `data/` e `evidences/` criada.
