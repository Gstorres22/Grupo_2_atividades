# Relatório de Erros e Correções (Consultor Financeiro AI)

Este documento lista os erros encontrados durante o desenvolvimento e execução inicial do fluxo LangGraph integrado ao MCP da B3 e como foram solucionados.

## 1. `ModuleNotFoundError: No module named 'pywintypes'`
**Erro:** O Python não conseguia encontrar o módulo `pywintypes` utilizado na conexão MCP com stdio no Windows, mesmo com a biblioteca `pywin32` instalada.
**Causa:** Este é um erro comum em ambientes virtuais (`.venv`) no Windows onde as DLLs do `pywin32` não são carregadas corretamente sem o script de pós-instalação ou quando a instalação falha em registrar os caminhos globais.
**Correção:** Forçamos a reinstalação e atualização dos pacotes no ambiente virtual e ajustamos os imports necessários para garantir que o MCP Client pudesse rodar corretamente sob o Windows.

## 2. Erro de BadRequest da OpenAI (HTTP 400)
**Erro:** `openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.", 'type': 'invalid_request_error', 'param': 'messages', 'code': None}}`
**Causa:** No LangGraph, quando o modelo de linguagem principal (`lead_advisor`) aciona ferramentas paralelamente (ou delega para subagentes), o histórico de mensagens devolvido à OpenAI deve **exatamente** corresponder na ordem: uma `AIMessage` contendo `tool_calls` seguida obrigatoriamente por `ToolMessage`s para cada `tool_call_id`. Se a estrutura do nó falhasse em alinhar esses IDs ou substituísse parte do histórico, a API da OpenAI rejeitava a requisição.
**Correção:**
1. Modificamos o objeto `State` (AgentState) para utilizar o `reducer` `add_messages` (`Annotated[list[AnyMessage], add_messages]`). Isso assegurou que as mensagens não fossem simplesmente sobrescritas ao transitar entre nós, mas sim adicionadas ao histórico.
2. Refatoramos os nós de subagentes paralelos em um único nó chamado `subagents_executor_node`. Este nó extrai as ferramentas solicitadas pelo `lead_advisor`, executa os subagentes sequencialmente para garantir a integridade dos resultados, e retorna as `ToolMessage` correspondentes a cada solicitação efetuada, mantendo o histórico consistente.

## 3. Bibliotecas Ausentes (`duckduckgo-search` e referências)
**Erro:** Falha de importação no Market Analyst ao tentar realizar pesquisas web.
**Causa:** As dependências do `duckduckgo-search` (necessárias pela LangChain para pesquisa) não estavam presentes.
**Correção:** Instalação das bibliotecas `duckduckgo-search` e `langchain-community` no ambiente virtual (`.venv`).

## 4. `NameError: name 'HumanMessage' is not defined`
**Erro:** Variáveis de mensagem do LangChain não reconhecidas.
**Causa:** Esquecimento de importar explicitamente os objetos de mensagem usados para inicializar os testes do LangGraph.
**Correção:** Adição dos imports: `from langchain_core.messages import HumanMessage, AIMessage, ToolMessage`.

## Estado Atual
O fluxo foi refatorado para utilizar uma estrutura assíncrona/robusta (no arquivo `test_flow.py`) que orquestra a comunicação do `lead_advisor` com o `market_analyst` e o `agente_b3` (acessando a ferramenta MCP). Estamos aguardando os testes finais rodarem para validar o processo fim-a-fim.
