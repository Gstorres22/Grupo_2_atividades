# Notas de integração — para o restante da equipe

Este documento descreve as suposições feitas no front-end Streamlit e o
que precisa ser ajustado/validado pelos colegas que estão construindo
a API e o modelo. O objetivo é facilitar a integração final.

## 1. Endpoints assumidos

| Método | Path             | Uso                                     |
| ------ | ---------------- | --------------------------------------- |
| POST   | `/predict`       | Predição de um único cliente            |
| POST   | `/predict/batch` | Predição em lote                        |
| GET    | `/health`        | Health-check                            |
| GET    | `/model/info`    | Versão / estágio do modelo em produção |

Se a API usar outros paths (ex.: `/api/v1/score`), basta ajustar as
chamadas em `app/utils/api_client.py`.

## 2. Schema do request `/predict`

```json
{ "input": { "<Feature>": <valor>, ... } }
```

A lista completa de features e seus tipos está em `app/utils/schema.py`.
Pontos de atenção:

- `Type_of_Loan` é enviado como **lista de strings**.
- `Credit_History_Age_Months` é um **inteiro em meses** — se a API esperar
  o formato original "X Years and Y Months" (string), ajustar tanto a
  conversão quanto o `schema.py`.
- Os campos `ID`, `Customer_ID`, `Name`, `SSN` do dataset original
  **não são enviados** (são identificadores).

## 3. Schema do response

```json
{
  "credit_score": "Good" | "Standard" | "Poor",
  "probabilities": { "Good": 0.7, "Standard": 0.2, "Poor": 0.1 },
  "model_version": "3",
  "request_id": "uuid",
  "latency_ms": 23.4
}
```

- `probabilities` é opcional, mas se ausente a UI não conseguirá
  desenhar o gauge nem o gráfico de barras.
- `model_version` deve ser a versão registrada no MLflow Model Registry.

## 4. Autenticação

O cliente envia ambos os headers para máxima compatibilidade:

```
Authorization: Bearer <API_KEY>
X-API-Key:     <API_KEY>
```

A API pode optar por qualquer um. Se for diferente, ajustar
`CreditScoreAPIClient._headers`.

## 5. Erros tratados

| HTTP | Comportamento na UI                                    |
| ---- | ------------------------------------------------------ |
| 401  | "API Key inválida ou ausente."                        |
| 403  | "Acesso negado."                                       |
| 429  | "Limite de requisições atingido — aguarde."           |
| 5xx  | "Erro interno da API."                                 |
| 4xx  | Mensagem genérica + payload original em expander      |

## 6. CORS / domínio

Se o app Streamlit e a API forem hospedados em domínios diferentes,
a API precisa habilitar CORS para o origin do Streamlit. Como o
Streamlit faz as chamadas **a partir do servidor Python** (não do
browser), CORS normalmente não é necessário — mas vale validar.

## 7. Modo demonstração

Há um toggle "Modo demonstração" na sidebar que faz o app gerar
predições simuladas (regra simples sobre as features). Útil para
testar a UI antes da API estar pronta. Deve ser desativado em
ambiente de demonstração com a API real.
