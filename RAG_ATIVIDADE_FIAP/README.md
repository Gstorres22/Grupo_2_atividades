# RAG - Regras do Futebol (FIFA Laws of the Game)

Este projeto consiste em uma aplicação de **Retrieval-Augmented Generation (RAG)** que utiliza os serviços da Azure (Azure OpenAI e Azure AI Search) para responder perguntas sobre as regras oficiais de futebol.

# Link de execução do projeto no youtube

Link: https://youtu.be/Vlw_o1tJ6KI


## Visão Geral e Arquitetura

O sistema é dividido em duas partes principais:
1. **Ingestão:** Um script Python (`ingest.py`) lê o PDF com as regras, faz o particionamento em partes menores (chunking), transforma em vetores numéricos via embeddings e os armazena no Azure AI Search.
2. **API (Azure Functions):** Uma função serverless que expõe o endpoint `/api/query`. Ela recebe perguntas do usuário, busca as partes mais relevantes no banco de dados vetorial e envia o contexto ao `GPT-4o` para gerar uma resposta precisa baseada **somente** nas regras.

### O Problema e a Solução
A complexidade das regras oficiais da FIFA dificulta a consulta rápida e precisa durante dúvidas em campo ou estudos. A solução proposta é uma API baseada em RAG que recebe perguntas em linguagem natural, extrai exatamente os trechos relevantes do documento PDF oficial utilizando busca vetorial no Azure AI Search, e gera uma resposta clara e fundamentada usando o modelo GPT-4o, garantindo que não haja "alucinações" fora das regras oficiais.

## 🛠️ Como Executar Localmente

### Pré-requisitos
- Python 3.10+
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local)
- Conta no [Portal do Azure](https://portal.azure.com/) com serviço AI Search criado.
- Chave de API da OpenAI (padrão `sk-...`).

### 1. Preparação do Ambiente e Variáveis
Clone o repositório, ative o ambiente virtual e crie os arquivos `.env` e `local.settings.json`:

**Exemplo do `.env` e `local.settings.json` (Variáveis necessárias):**
```env
API_KEY_OPEN_AI="sk-proj-sua-chave-aqui"
AZURE_SEARCH_ENDPOINT="https://seu-servico.search.windows.net"
AZURE_SEARCH_KEY="sua-chave-admin"
AZURE_SEARCH_INDEX_NAME="regras-futebol-index"
```

### 2. Instalando as Dependências
```bash
pip install -r requirements.txt
```

### 3. Ingestão de Dados (Rode apenas 1 vez)
O script abaixo vai ler o PDF, particionar o texto, criar os embeddings e salvar na Azure.
```bash
python ingest.py
```

### 4. Executando a API Localmente
```bash
func start
```
A API ficará disponível no endpoint `http://localhost:7071/api/query`.

## ☁️ Como Fazer o Deploy (Nuvem)

A aplicação foi projetada para rodar em **Azure Functions (Consumption Plan)**.

**Passo a passo pelo VSCode (Recomendado):**
1. Instale a extensão oficial **Azure Functions** no VSCode.
2. Faça login com sua conta da Azure.
3. No Portal do Azure, crie um **Function App** (Linux, Python).
4. No Portal, vá na sua Function App > **Variáveis de ambiente**, e adicione:
   - `API_KEY_OPEN_AI`
   - `AZURE_SEARCH_ENDPOINT`
   - `AZURE_SEARCH_KEY`
   - `SCM_DO_BUILD_DURING_DEPLOYMENT=true` (Garante a instalação do requirements.txt na nuvem).
5. Volte no VSCode, clique no ícone da nuvem com a seta para cima (Deploy) e selecione o seu App.

Alternativamente, via CLI:
```bash
func azure functionapp publish <NOME_DA_SUA_APP> --build remote
```

## 💡 Exemplos de Uso

Abaixo estão 3 exemplos práticos de como testar a API enviando perguntas para a rota `/api/query` (via cURL).

### Exemplo 1: Conduta Violenta
**Requisição:**
```bash
curl -X POST https://<sua-url>.azurewebsites.net/api/query \
-H "Content-Type: application/json" \
-d "{\"question\": \"O que acontece se o jogador cuspir no arbitro?\"}"
```
**Resposta Esperada (Resumo):**
```json
{
  "answer": "De acordo com as regras fornecidas, se um jogador cuspir em alguém, incluindo o árbitro, ele deverá ser expulso por conduta violenta.",
  "sources": [{"chunk": "...comportar-se de maneira agressiva... incluindo ao cuspir...", "page": 119}]
}
```

### Exemplo 2: Regra do Impedimento
**Requisição:**
```bash
curl -X POST https://<sua-url>.azurewebsites.net/api/query \
-H "Content-Type: application/json" \
-d "{\"question\": \"O jogador pode estar impedido em um arremesso manual (lateral)?\"}"
```
**Resposta Esperada (Resumo):**
```json
{
  "answer": "Não. De acordo com as Regras do Jogo, não há infração de impedimento quando o jogador recebe a bola diretamente de um arremesso manual.",
  "sources": [{"chunk": "Não há infração de impedimento se o jogador receber a bola diretamente de: • um tiro de meta; • um arremesso manual...", "page": 100}]
}
```

### Exemplo 3: Equipamento dos Jogadores
**Requisição:**
```bash
curl -X POST https://<sua-url>.azurewebsites.net/api/query \
-H "Content-Type: application/json" \
-d "{\"question\": \"Um jogador pode atuar usando colares ou brincos?\"}"
```
**Resposta Esperada (Resumo):**
```json
{
  "answer": "Não. Todos os itens de joalheria (como colares, anéis, pulseiras, brincos, etc.) são estritamente proibidos e devem ser removidos. Não é permitido cobri-los com fita adesiva.",
  "sources": [{"chunk": "Todos os itens de joalheria (colares, anéis, pulseiras, brincos, etc.) são estritamente proibidos...", "page": 43}]
}
```

## Decisões Técnicas
1. **Estratégia de Chunking:** Foi utilizado `RecursiveCharacterTextSplitter` pois as regras de futebol possuem tópicos hierárquicos e seções longas, essa estratégia preserva melhor a quebra entre parágrafos sem cortar frases no meio.

2. **Banco Vetorial:** O **Azure AI Search** foi escolhido para integração nativa no ecossistema da Azure. Utilizamos a API Python (`azure-search-documents` v11.6) para aplicar buscas de `VectorizedQuery` que fornece alta performance baseada em HNSW.

3. **Modelos:** `text-embedding-3-small` foi escolhido por possuir um custo/benefício fantástico para embeddings eficientes. `GPT-4o` foi o modelo geracional selecionado pela sua capacidade formidável de sumarização e formatação baseada unicamente no contexto injetado, evitando alucinações.

4. **Substituição do Azure OpenAI para OpenAI Padrão:** Durante a implementação, identificou-se que a assinatura de Azure do gerava chaves do modelo "Azure AI Foundry Unified Keys", causando conflitos de autenticação 401 Unauthorized nas classes originais do `langchain_openai`. Para manter a robustez e seguir os padrões de mercado sem aumentar a complexidade com Autenticação do Entra ID, eu utilizei as classes Padrão da OpenAI (`OpenAIEmbeddings` e `ChatOpenAI`) consumindo as chaves clássicas da OpenAI (`sk-...`). Isso garantiu resiliência, contornou o problema das chaves unificadas e manteve todos os requisitos arquiteturais intactos.
