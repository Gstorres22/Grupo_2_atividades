# Documentação da API de Score de Crédito — QuantumFinance

Esta documentação descreve a API REST desenvolvida para o serviço de **Credit Score (Classificação de Score de Crédito)** da QuantumFinance. Esta API permite autenticar usuários, realizar predições individuais ou em lote (batch) de score de crédito utilizando o modelo Machine Learning em produção (`credit_score_clf`), e monitorar o status do serviço.

O frontend desenvolvido em **Streamlit** está totalmente configurado para se integrar a este contrato de API. Em cenários onde a API não está acessível, o frontend possui uma política de contingência (*fallback*) automática para execução do modelo via MLflow local.

---

## 1. Visão Geral e Arquitetura

A API serve como a camada de interface entre o frontend (ou outros sistemas corporativos) e o pipeline de inferência do modelo treinado. 

```
                                  +------------------------------------+
                                  |         App Streamlit / CLI        |
                                  +------------------------------------+
                                      |                            |
                             (API no ar)                     (API fora)
                                      v                            v
                      +-------------------------------+    +-----------------------+
                      |         API REST HTTP         |    |   Fallback Local      |
                      |   (FastAPI/Flask/Functions)   |    |    (MLflow Run)       |
                      +-------------------------------+    +-----------------------+
                                      |                            |
                                      v                            v
                                  +------------------------------------+
                                  |    Modelo credit_score_clf (vX)    |
                                  |       Stage: "Production"          |
                                  +------------------------------------+
```

### Endereço Base (Base URL)
Por padrão, o app Streamlit tenta se conectar à URL configurada em `app.py` ou especificada na tela de login.
* URL Padrão de Desenvolvimento/Local: `http://localhost:8000` (ou conforme especificado no startup da sua API).

---

## 2. Autenticação e Cabeçalhos (Headers)

A API utiliza autenticação baseada em **Bearer Token (JWT)** para proteger as rotas de inferência e de metadados do modelo.

### Cabeçalhos Requeridos para Rotas Protegidas
Todas as chamadas para os endpoints protegidos (`/predict`, `/predict/batch` e `/model/info`) devem incluir os seguintes headers HTTP:

```http
Authorization: Bearer <SEU_JWT_TOKEN>
Content-Type: application/json
Accept: application/json
```

> [!NOTE]
> Para fins de compatibilidade e facilidade de testes, o cliente HTTP do Streamlit também envia a chave através do cabeçalho `X-API-Key: <SEU_JWT_TOKEN>`. A API pode optar por validar qualquer um dos formatos.

---

## 3. Catálogo de Endpoints

### A. Login e Obtenção de Token
* **Endpoint:** `/login`
* **Método:** `POST`
* **Autenticação:** Nenhuma (Rota Pública)
* **Descrição:** Autentica o usuário com credenciais do sistema e retorna o JSON Web Token (JWT) necessário para chamadas subsequentes.

#### Payload da Requisição (Body JSON)
| Campo | Tipo | Obrigatório | Descrição | Exemplo |
| :--- | :--- | :--- | :--- | :--- |
| `username` | `string` | Sim | Nome de usuário cadastrado | `"admin"` |
| `password` | `string` | Sim | Senha do usuário | `"admin123"` |

*Exemplo de Request:*
```json
{
  "username": "admin",
  "password": "admin123"
}
```

#### Respostas
* **`200 OK`**: Login efetuado com sucesso. Devolve o token de acesso.
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

* **`401 Unauthorized`**: Credenciais incorretas ou usuário inexistente.
```json
{
  "detail": "Usuário ou senha inválidos."
}
```

* **`429 Too Many Requests`**: Limite de tentativas de login excedido.
```json
{
  "detail": "Limite de tentativas atingido. Aguarde antes de tentar de novo."
}
```

---

### B. Predição Individual de Score de Crédito
* **Endpoint:** `/predict`
* **Método:** `POST`
* **Autenticação:** Sim (`Bearer Token` ou `X-API-Key`)
* **Descrição:** Recebe as features financeiras e cadastrais de um único cliente e retorna a classificação de score de crédito ("Good", "Standard" ou "Poor"), acompanhada das probabilidades de cada classe.

#### Payload da Requisição (Body JSON)
A requisição deve conter um objeto `"input"` contendo as 23 features listadas abaixo:

```json
{
  "input": {
    "Age": 30,
    "Occupation": "Engineer",
    "Month": "May",
    "Annual_Income": 50000.0,
    ...
  }
}
```

#### Tabela de Features Aceitas no Objeto `"input"`
A tabela abaixo define os tipos e restrições de cada uma das 23 features tratadas pelo pipeline:

| Feature (Nome JSON) | Tipo de Dado | Valores Válidos / Limites | Valor Padrão | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `Age` | `integer` | `18` a `100` | `30` | Idade do cliente em anos. |
| `Occupation` | `string` | Ver [Lista de Profissões](#lista-de-profissões) | `"Engineer"` | Profissão informada pelo cliente. |
| `Month` | `string` | Ver [Lista de Meses](#lista-de-meses) | `"May"` | Mês de referência da observação financeira. |
| `Annual_Income` | `float` | `0.0` a `1000000.0` | `50000.0` | Renda bruta anual do cliente (em USD). |
| `Monthly_Inhand_Salary`| `float` | `0.0` a `100000.0` | `3500.0` | Salário líquido mensal recebido em mãos. |
| `Monthly_Balance` | `float` | `-5000.0` a `50000.0` | `300.0` | Saldo médio disponível após despesas mensais. |
| `Amount_invested_monthly`| `float` | `0.0` a `20000.0` | `100.0` | Valor investido em média a cada mês. |
| `Num_Bank_Accounts` | `integer` | `0` a `30` | `3` | Quantidade de contas bancárias ativas. |
| `Num_Credit_Card` | `integer` | `0` a `30` | `4` | Quantidade de cartões de crédito do cliente. |
| `Interest_Rate` | `float` | `0.0` a `50.0` | `12.0` | Taxa de juros média cobrada nos cartões/contas. |
| `Num_of_Loan` | `integer` | `0` a `20` | `2` | Quantidade de empréstimos ativos atualmente. |
| `Type_of_Loan` | `array[str]`| Ver [Lista de Empréstimos](#tipos-de-empréstimos)| `["Personal Loan"]` | Lista dos tipos de empréstimo ativos. |
| `Outstanding_Debt` | `float` | `0.0` a `200000.0` | `1000.0` | Montante total da dívida pendente (em USD). |
| `Total_EMI_per_month` | `float` | `0.0` a `20000.0` | `100.0` | Valor da parcela mensal de empréstimos (EMI). |
| `Delay_from_due_date` | `integer` | `-10` a `90` | `5` | Média de dias de atraso após a data de vencimento. |
| `Num_of_Delayed_Payment`| `integer` | `0` a `50` | `2` | Quantidade de pagamentos mensais atrasados. |
| `Payment_of_Min_Amount`| `string` | `"Yes"`, `"No"`, `"NM"` | `"No"` | Pagou o valor mínimo do cartão? (NM: Não Mencionado)|
| `Payment_Behaviour` | `string` | Ver [Lista de Comportamentos](#comportamentos-de-pagamento)| `"Low_spent_Small_value_payments"` | Comportamento de gastos e pagamentos do cliente. |
| `Changed_Credit_Limit` | `float` | `-100.0` a `100.0` | `5.0` | Percentual de alteração do limite de crédito. |
| `Num_Credit_Inquiries` | `integer` | `0` a `50` | `3` | Número de consultas de crédito realizadas (Bureau).|
| `Credit_Mix` | `string` | `"Good"`, `"Standard"`, `"Bad"`| `"Standard"` | Mix/Qualidade dos créditos contratados. |
| `Credit_Utilization_Ratio`| `float` | `0.0` a `100.0` | `30.0` | Percentual utilizado do limite total de crédito. |
| `Credit_History_Age_Months`| `integer`| `0` a `600` | `120` | Idade do histórico de crédito em meses. |

*Exemplo de Request:*
```json
{
  "input": {
    "Age": 30,
    "Occupation": "Engineer",
    "Month": "May",
    "Annual_Income": 50000.0,
    "Monthly_Inhand_Salary": 3500.0,
    "Monthly_Balance": 300.0,
    "Amount_invested_monthly": 100.0,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 4,
    "Interest_Rate": 12.0,
    "Num_of_Loan": 2,
    "Type_of_Loan": ["Personal Loan"],
    "Outstanding_Debt": 1000.0,
    "Total_EMI_per_month": 100.0,
    "Delay_from_due_date": 5,
    "Num_of_Delayed_Payment": 2,
    "Payment_of_Min_Amount": "No",
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Changed_Credit_Limit": 5.0,
    "Num_Credit_Inquiries": 3,
    "Credit_Mix": "Standard",
    "Credit_Utilization_Ratio": 30.0,
    "Credit_History_Age_Months": 120
  }
}
```

#### Respostas
* **`200 OK`**: Predição efetuada com sucesso. Retorna o score final, probabilidades associadas, versão do modelo ativo no MLflow e metadados.
```json
{
  "credit_score": "Good",
  "probabilities": {
    "Good": 0.72,
    "Standard": 0.22,
    "Poor": 0.06
  },
  "model_version": "3",
  "request_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
  "latency_ms": 23.4
}
```

* **`400 Bad Request`**: Parâmetros com valores inválidos ou campos faltantes.
```json
{
  "detail": "Falha na validação dos campos. O campo 'Age' deve ser maior ou igual a 18."
}
```

---

### C. Predição em Lote (Batch)
* **Endpoint:** `/predict/batch`
* **Método:** `POST`
* **Autenticação:** Sim (`Bearer Token` ou `X-API-Key`)
* **Descrição:** Processa a predição para múltiplos clientes simultaneamente de forma otimizada. Muito útil para o processamento de planilhas CSV enviadas pelo frontend.

#### Payload da Requisição (Body JSON)
A requisição deve conter um objeto `"inputs"` que contém uma lista com vários dicionários contendo as features explicadas anteriormente.

```json
{
  "inputs": [
    {
      "Age": 30,
      "Occupation": "Engineer",
      "Month": "May",
      "Annual_Income": 50000.0,
      "Monthly_Inhand_Salary": 3500.0,
      "Monthly_Balance": 300.0,
      "Amount_invested_monthly": 100.0,
      "Num_Bank_Accounts": 3,
      "Num_Credit_Card": 4,
      "Interest_Rate": 12.0,
      "Num_of_Loan": 2,
      "Type_of_Loan": ["Personal Loan"],
      "Outstanding_Debt": 1000.0,
      "Total_EMI_per_month": 100.0,
      "Delay_from_due_date": 5,
      "Num_of_Delayed_Payment": 2,
      "Payment_of_Min_Amount": "No",
      "Payment_Behaviour": "Low_spent_Small_value_payments",
      "Changed_Credit_Limit": 5.0,
      "Num_Credit_Inquiries": 3,
      "Credit_Mix": "Standard",
      "Credit_Utilization_Ratio": 30.0,
      "Credit_History_Age_Months": 120
    },
    {
      "Age": 45,
      "Occupation": "Doctor",
      "Month": "May",
      "Annual_Income": 120000.0,
      "Monthly_Inhand_Salary": 8500.0,
      "Monthly_Balance": 1200.0,
      "Amount_invested_monthly": 500.0,
      "Num_Bank_Accounts": 2,
      "Num_Credit_Card": 2,
      "Interest_Rate": 6.5,
      "Num_of_Loan": 1,
      "Type_of_Loan": ["Mortgage Loan"],
      "Outstanding_Debt": 50000.0,
      "Total_EMI_per_month": 450.0,
      "Delay_from_due_date": 1,
      "Num_of_Delayed_Payment": 0,
      "Payment_of_Min_Amount": "Yes",
      "Payment_Behaviour": "High_spent_Large_value_payments",
      "Changed_Credit_Limit": 2.0,
      "Num_Credit_Inquiries": 1,
      "Credit_Mix": "Good",
      "Credit_Utilization_Ratio": 15.0,
      "Credit_History_Age_Months": 240
    }
  ]
}
```

#### Respostas
* **`200 OK`**: Retorna uma lista de objetos de predição correspondente e na mesma ordem do envio (ou sob a chave `"predictions"`).
```json
[
  {
    "credit_score": "Good",
    "probabilities": { "Good": 0.72, "Standard": 0.22, "Poor": 0.06 },
    "model_version": "3"
  },
  {
    "credit_score": "Good",
    "probabilities": { "Good": 0.91, "Standard": 0.07, "Poor": 0.02 },
    "model_version": "3"
  }
]
```

---

### D. Health Check (Verificação de Saúde)
* **Endpoint:** `/health`
* **Método:** `GET`
* **Autenticação:** Nenhuma (Rota Pública)
* **Descrição:** Indica se o servidor da API REST está operacional e respondendo requisições.

#### Respostas
* **`200 OK`**: API operando normalmente.
```json
{
  "status": "ok",
  "mode": "production",
  "uptime_s": 86400
}
```

---

### E. Informações do Modelo em Produção
* **Endpoint:** `/model/info`
* **Método:** `GET`
* **Autenticação:** Sim (`Bearer Token` ou `X-API-Key`)
* **Descrição:** Retorna os metadados do modelo de Machine Learning que está ativamente promovido no estágio de **Production** do MLflow Model Registry.

#### Respostas
* **`200 OK`**: Metadados do modelo retornados com sucesso.
```json
{
  "name": "credit_score_clf",
  "version": "3",
  "stage": "Production",
  "algorithm": "RandomForestClassifier",
  "f1_macro": 0.715,
  "accuracy": 0.726,
  "trained_at": "2026-05-25T18:30:00Z"
}
```

---

## 4. Tabelas de Domínio de Parâmetros

### Lista de Profissões (`Occupation`)
O campo `Occupation` aceita apenas uma das seguintes strings:
* `"Scientist"`, `"Teacher"`, `"Engineer"`, `"Entrepreneur"`, `"Developer"`, `"Lawyer"`, `"Media_Manager"`, `"Doctor"`, `"Journalist"`, `"Manager"`, `"Accountant"`, `"Musician"`, `"Mechanic"`, `"Writer"`, `"Architect"`

### Lista de Meses (`Month`)
O campo `Month` aceita apenas:
* `"January"`, `"February"`, `"March"`, `"April"`, `"May"`, `"June"`, `"July"`, `"August"`

### Tipos de Empréstimos (`Type_of_Loan`)
Os elementos da lista `Type_of_Loan` devem pertencer a:
* `"Not Specified"`, `"Credit-Builder Loan"`, `"Personal Loan"`, `"Debt Consolidation Loan"`, `"Student Loan"`, `"Payday Loan"`, `"Mortgage Loan"`, `"Auto Loan"`, `"Home Equity Loan"`

### Comportamentos de Pagamento (`Payment_Behaviour`)
O campo `Payment_Behaviour` aceita apenas:
* `"High_spent_Small_value_payments"`
* `"Low_spent_Large_value_payments"`
* `"Low_spent_Medium_value_payments"`
* `"Low_spent_Small_value_payments"`
* `"High_spent_Medium_value_payments"`
* `"High_spent_Large_value_payments"`

---

## 5. Respostas de Erro Globais e Tratamento

A tabela a seguir apresenta os códigos HTTP de erro implementados ou tratados pela API:

| Código HTTP | Causa | Formato da Resposta (JSON) | Ação Recomendada |
| :--- | :--- | :--- | :--- |
| **`400`** | Erro de validação dos dados de entrada (schema incorreto ou fora dos limites). | `{"detail": "Mensagem detalhada do erro"}` | Validar os tipos de dados e os limites estabelecidos na tabela de features. |
| **`401`** | Token JWT/API Key não fornecido, malformado ou expirado. | `{"detail": "Token inválido ou expirado."}` | Executar o fluxo de `/login` novamente para obter um novo token ou verificar o header. |
| **`403`** | Acesso proibido. O usuário autenticado não possui permissão para o recurso. | `{"detail": "Acesso negado."}` | Verificar se o usuário possui os privilégios corretos no provedor de identidade. |
| **`429`** | Limite de requisições excedido (Rate Limiting). | `{"detail": "Limite de requisições atingido. Tente novamente mais tarde."}` | Aguardar alguns instantes e reimplementar políticas de Backoff Exponencial no cliente. |
| **`500`** | Falha interna no servidor (ex: erro ao carregar o modelo Sklearn, falha de persistência). | `{"detail": "Erro interno no servidor."}` | Verificar os logs do servidor da API para entender a falha na inferência. |
| **`503`** | O serviço de MLflow Registry ou a infraestrutura do modelo está indisponível. | `{"detail": "Serviço temporariamente indisponível."}` | Aguardar a inicialização dos microsserviços. O app Streamlit entrará em fallback local se ativo. |

---

## 6. Troubleshooting e FAQ (Resolução de Problemas Comuns)

### P1: Recebo erro "Não foi possível conectar à API" no frontend Streamlit.
* **Possível Causa 1:** O servidor da API REST não foi iniciado ou está escutando em uma porta/host diferente.
  * **Solução:** Certifique-se de que a API REST está ativa executando algo similar a `uvicorn main:app --port 8000`. Verifique se a URL no input do login coincide com a URL e porta reais da API.
* **Possível Causa 2:** Problemas de conectividade de rede ou firewall bloqueando a porta.
  * **Solução:** Teste o acesso ao endpoint `/health` via terminal ou navegador (ex: `curl http://localhost:8000/health`).

### P2: O app do Streamlit indica que está em "Fallback Local (Modo Local)" ou "Modo Demonstração".
* **Causa:** O cliente da API tenta acessar os endpoints protegidos e, ao falhar por indisponibilidade de conexão (Erro de Rede) ou respostas de erro `5xx`, ele carrega o modelo local em fallback automático (quando a opção `fallback_local` está ativada).
  * **Solução:** Verifique o status da API REST. Se a API estiver offline, o fallback local é o comportamento esperado para garantir a disponibilidade do sistema. Para forçar a API e desativar o fallback local, ajuste o toggle correspondente na interface do usuário (sidebar do Streamlit) or configure `fallback_local=False` nas configurações do cliente API.

### P3: Recebo erro HTTP 401 Unauthorized após algum tempo de uso.
* **Causa:** O token JWT retornado pelo endpoint `/login` possui tempo de expiração padrão (ex: 3600 segundos/1 hora). Uma vez expirado, as chamadas subsequentes retornam `401`.
  * **Solução:** O app Streamlit realiza a limpeza automática de sessão expirada. Basta preencher as credenciais novamente na tela de login para gerar um novo token JWT ativo.

### P4: Erro 400 Bad Request ao tentar realizar predições.
* **Possível Causa 1:** Tipos de dados inadequados no JSON. Por exemplo, passar `'Age': '30'` (string) em vez de `30` (integer).
  * **Solução:** Revise o payload enviado e compare-o estritamente com os tipos definidos na [Tabela de Features](#tabela-de-features-aceitas-no-objeto-input).
* **Possível Causa 2:** O campo `Type_of_Loan` foi enviado como uma string única.
  * **Solução:** Ele deve ser sempre enviado como uma lista de strings (array JSON, ex: `["Personal Loan", "Auto Loan"]`), mesmo se contiver apenas um tipo de empréstimo.

### P5: O console do servidor da API REST acusa "RuntimeError: Nenhum modelo encontrado em estágio 'Production'".
* **Causa:** O código que serve a API REST tenta carregar o modelo usando `Predictor.from_registry()` ou `mlflow.sklearn.load_model("models:/credit_score_clf/Production")`, mas nenhum modelo foi promovido a este estágio ainda no MLflow local.
  * **Solução:** Execute o script de treinamento primeiro para treinar o modelo e registrá-lo automaticamente no MLflow Registry:
    ```bash
    python -m training.train --data data/train_sample.csv --algo random_forest
    ```
    O script moverá automaticamente o modelo inicial treinado para a fase de `Production`.

### P6: Problemas com CORS (Cross-Origin Resource Sharing) no navegador do usuário.
* **Causa:** Embora o app Streamlit faça requisições para a API **a partir do seu próprio servidor Python** (o que elimina a necessidade de configuração CORS no browser, agindo como um proxy), integrações diretas do front-end via Javascript/Client-side no browser à API REST podem disparar bloqueios de CORS se estiverem em domínios ou portas distintas.
  * **Solução:** Caso a API seja exposta para chamadas diretas client-side, configure o Middleware do FastAPI/Flask para aceitar requisições de outras origens:
    ```python
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Ou o domínio específico do seu Streamlit
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ```
