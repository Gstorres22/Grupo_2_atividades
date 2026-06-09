# Autenticação Facial — Camada complementar à senha

Sistema de autenticação facial para serviços sensíveis (ex.: crédito pessoal),
construído como camada **complementar à senha**. Resolve o problema apontado pelo
setor de fraudes: contratações em que a senha foi validada corretamente, mas o
cliente nega ter contratado o serviço.

O sistema entrega três capacidades:

1. **Detecção de faces** — localiza o rosto no quadro (MediaPipe Face Detection).
2. **Identificação de faces** — confirma que o rosto é do cliente cadastrado
   (DeepFace / ArcFace + similaridade de cosseno).
3. **Liveness (vivacidade)** — detecção **ativa por piscada** (MediaPipe FaceMesh +
   Eye Aspect Ratio) para impedir o uso de **foto estática** por um fraudador.

Quando o cliente **não é autenticado**, ele é encaminhado para uma **esteira
dedicada** e as **evidências** (frame, scores, limiares e motivo) são gravadas em
`evidences/` para a **área de IA** recalibrar parâmetros e limiares.

## Como rodar

```bash
# 1. (opcional) criar e ativar um ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. instalar dependencias
pip install -r requirements.txt

# 3. abrir o notebook
jupyter notebook autenticacao_facial.ipynb
```

Execute as células em ordem. O notebook está dividido em seções (Setup,
Configuração, Detector, Liveness, Identificação, Cadastro, Autenticação,
Evidências, Relatório).

### Fluxo de uso
1. **Cadastro:** rode a seção de *Cadastro* olhando para a webcam → gera os
   embeddings em `data/embeddings/<user_id>.pkl`.
2. **Autenticação:** rode a seção de *Autenticação* informando o `user_id`,
   **pisque** quando solicitado (desafio de liveness) → resultado
   `AUTENTICADO` ou `REPROVADO`.

> Se a webcam não estiver disponível, use o **Modo offline** (última seção) com
> imagens/vídeo gravados.

## Estrutura

```
Autenticacao_facial_claude/
├── autenticacao_facial.ipynb     # notebook principal (todo o fluxo)
├── requirements.txt
├── README.md
├── PLANO_DE_ACAO.md
├── data/
│   ├── enrolled/<user_id>/       # fotos de cadastro
│   └── embeddings/               # embeddings salvos (.pkl)
└── evidences/                    # evidencias de nao-identificacao p/ area de IA
```

## Nota de privacidade (LGPD)

Dados biométricos são **dados pessoais sensíveis**. Por padrão o sistema
armazena **embeddings** (vetores numéricos), e não as fotos cruas — isso reduz a
superfície de risco. As fotos de cadastro em `data/enrolled/` são opcionais e
servem apenas para auditoria/re-treino; em produção devem ser criptografadas,
ter prazo de retenção definido e consentimento explícito do titular.

## Limiares (ajustáveis na célula `CONFIG`)

| Parâmetro | Função |
|---|---|
| `THRESHOLD_SIMILARIDADE` | corte da similaridade de cosseno p/ aceitar a identidade |
| `EAR_THRESHOLD` | abaixo deste valor o olho é considerado fechado (piscada) |
| `PISCADAS_NECESSARIAS` | nº de piscadas exigidas no desafio de liveness |
| `N_FOTOS_CADASTRO` | nº de frames capturados no cadastro |
| `MODELO` | modelo de embedding do DeepFace (padrão `ArcFace`) |
