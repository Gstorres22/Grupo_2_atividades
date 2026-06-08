# QuantumFinance · Credit Score · POC Streamlit + MLOps

Solução completa para o desafio de score de crédito da QuantumFinance:

1. **Streamlit app** com login, formulário e visualizações.
2. **Treinamento** com rastreio de experimentos via **MLflow** e promoção
   automática para Production.
3. **Inferência** que carrega o modelo da versão mais recente em Production.

A API REST (autenticação + throttling) é entregue separadamente, mas o
projeto já está preparado para consumi-la — e tem **fallback automático**
para o modelo local quando a API estiver fora.

---

## ✨ Funcionalidades

### App Streamlit
- 🔐 **Tela de login** que chama `POST /login` da API e guarda o JWT na sessão.
- 🎯 **Predição individual** — formulário em abas por grupo de features.
- 📦 **Predição em lote** — upload de CSV + template para download.
- 🕘 **Histórico de sessão** — export para CSV.
- 📊 **Gauge de score** e **barras de probabilidade** por classe.
- 📡 **Health-check** e **versão do modelo em produção**.
- 🤖 **Modo híbrido**: API HTTP, Modelo local (MLflow) ou Demo simulado — selecionável na sidebar, com fallback automático.

### Treinamento (`training/train.py`)
- Pipeline `ColumnTransformer` + algoritmo (RF / GB / LogReg).
- Métricas rastreadas: `accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`.
- Artefatos rastreados: modelo serializado, label encoder, classification report, matriz de confusão (PNG), amostra de validação.
- Registro automático no **MLflow Model Registry** como `credit_score_clf`.
- **Critério de promoção**: novo modelo só vira Production se `f1_macro` ≥ atual de Production.
- **Promoção**: `MlflowClient.transition_model_version_stage` move o novo para Production e arquiva o anterior automaticamente.

### Inferência (`inference/predictor.py`)
- Carrega o modelo da última versão promovida (`models:/credit_score_clf/Production`).
- API Python (`Predictor.predict_one`, `predict_batch`) + CLI.
- Aplica exatamente o mesmo pipeline de limpeza usado no treino.

---

## 🚀 Como executar

### Pré-requisitos
- Python 3.10+
- Linux/Mac/Windows
- ~500 MB livres (MLflow + dependências)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 1. Treinar o modelo (cria o `mlruns/` local)

```bash
# Usa data/train.csv por padrão. Para testes rápidos use uma amostra:
python -m training.train --data data/train_sample.csv --algo random_forest

# Outros algoritmos:
python -m training.train --algo gradient_boosting --learning-rate 0.05
python -m training.train --algo logistic_regression --C 0.5
```

Saída esperada:
```
[train] nova versão registrada: v1
[train] v1 -> Staging
[train] não há modelo em Production. Promovendo v1.
[train] decisão: promoted (first model)
```

### 2. Visualizar experimentos (opcional)

```bash
mlflow ui --backend-store-uri ./mlruns
# abrir http://localhost:5000
```

### 3. Rodar a inferência via CLI

```bash
# Um único cliente:
python -m inference.predictor --single '{"Age": 30, "Annual_Income": 50000, ...}'

# Em lote:
python -m inference.predictor --input data/test.csv --output predictions.csv
```

### 4. Subir o app Streamlit

```bash
streamlit run app.py
```

Acesse <http://localhost:8501>.

**Credenciais para modo demonstração** (ative o toggle na tela de login):
- `admin` / `admin123`
- `analista` / `analista123`
- `parceiro` / `parceiro123`

---

## 📁 Estrutura do projeto

```
quantumfinance_streamlit/
├── app.py                       # Streamlit entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   ├── config.toml              # tema escuro custom
│   └── secrets.toml.example
├── app/
│   ├── components/
│   │   ├── charts.py            # gauge + barras
│   │   └── form_builder.py      # form dinâmico
│   ├── pages/
│   │   ├── 1_predict.py
│   │   ├── 2_batch.py
│   │   ├── 3_history.py
│   │   ├── 4_status.py
│   │   └── 5_about.py
│   └── utils/
│       ├── api_client.py        # cliente HTTP + fallback local
│       ├── auth.py              # tela de login + JWT/demo
│       ├── schema.py            # 23 features do dataset
│       └── styles.py
├── training/
│   ├── data_prep.py             # limpeza/transformação do dataset bruto
│   └── train.py                 # treino + tracking + promoção MLflow
├── inference/
│   └── predictor.py             # carregamento da Production + predição
├── data/
│   ├── train.csv                # dataset Kaggle completo (100k)
│   ├── test.csv
│   └── train_sample.csv         # amostra de 5k linhas para testes rápidos
├── mlruns/                      # backend local do MLflow (após treinar)
├── docs/
│   └── integration_notes.md     # notas para a equipe da API
└── tests/
    └── test_smoke.py
```

---

## 🔌 Contratos esperados da API REST

O app já está configurado para chamar estes endpoints (ajustar em
`app/utils/api_client.py` se necessário).

### `POST /login`

```json
// request
{ "username": "admin", "password": "admin123" }

// response
{ "access_token": "<JWT>", "token_type": "Bearer", "expires_in": 3600 }
```

### `POST /predict`

```json
// request
{ "input": { "Age": 30, "Annual_Income": 50000, "...": "..." } }

// response
{
  "credit_score": "Good",
  "probabilities": { "Good": 0.72, "Standard": 0.22, "Poor": 0.06 },
  "model_version": "3",
  "request_id": "uuid",
  "latency_ms": 23.4
}
```

### `POST /predict/batch`

```json
{ "inputs": [ { "Age": 30, "...": "..." }, { "...": "..." } ] }
```

### `GET /health` · `GET /model/info`

Headers exigidos em todas as rotas autenticadas:

```
Authorization: Bearer <JWT>
Content-Type:  application/json
```

**Sugestão para quem montar a API:** use `mlflow.sklearn.load_model("models:/credit_score_clf/Production")` no startup — exatamente o que o `inference/predictor.py` faz. Assim a API consome o mesmo modelo promovido pelo script de treinamento.

---

## 🧪 Testes

```bash
pytest -q
```

Cobre: imports, schema, predição em modo demo/local, login demo e
limpeza do dataset.

---

## 📈 Métricas do modelo atual (referência)

Treinado com 5000 amostras / RandomForest n=50, depth=10:

| Métrica          | Valor   |
| ---------------- | ------- |
| accuracy         | 0.726   |
| f1_macro         | 0.715   |
| f1_weighted      | 0.729   |
| precision_macro  | 0.703   |
| recall_macro     | 0.740   |

Para o treino final, usar `data/train.csv` (100k linhas) com `n_estimators=200`.

---

## 👥 Equipe

Trabalho em grupo — disciplina de MLOps / Engenharia de ML.
