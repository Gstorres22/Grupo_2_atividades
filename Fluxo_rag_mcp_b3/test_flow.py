import os
import asyncio
import nest_asyncio
from typing import Annotated, List, Dict, Any, TypedDict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Set API Keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["BOLSAI_API_KEY"] = "sk_d91c47af31660d0e794ca8a497fccdf5d722346785f37009"

nest_asyncio.apply()

class BolsaiMCPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.server_params = StdioServerParameters(
            command="uvx",
            args=["bolsai-mcp"],
            env={"BOLSAI_API_KEY": self.api_key}
        )
        self.read_stream = None
        self.write_stream = None
        self.session = None
        self._client_context = None

    async def connect(self):
        import sys
        errlog = sys.stderr
        if hasattr(sys, "__stderr__") and sys.__stderr__ is not None:
            try:
                sys.__stderr__.fileno()
                errlog = sys.__stderr__
            except Exception:
                pass
        self._client_context = stdio_client(self.server_params, errlog=errlog)
        self.read_stream, self.write_stream = await self._client_context.__aenter__()
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        print("Conectado com sucesso ao Bolsai MCP Server!")

    async def disconnect(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
        except Exception:
            pass
        print("Desconectado do Bolsai MCP Server.")

    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.session:
            raise RuntimeError("Cliente MCP não conectado.")
        result = await self.session.call_tool(tool_name, arguments)
        if not result.content:
            return "Nenhum dado retornado."
        return result.content[0].text

# Instantiate client
bolsai_client = BolsaiMCPClient(api_key=os.environ["BOLSAI_API_KEY"])

# Setup tools
web_search = DuckDuckGoSearchRun()

@tool
def search_financial_concept(query: str) -> str:
    """Busca na internet informações conceituais sobre produtos e regulamentos financeiros."""
    try:
        print(f"-> Chamando DuckDuckGo para query: '{query}'...", flush=True)
        res = web_search.run(query)
        print(f"-> DuckDuckGo retornou {len(res)} caracteres.", flush=True)
        return res
    except Exception as e:
        print(f"-> Erro no DuckDuckGo: {str(e)}", flush=True)
        return f"Erro na busca: {str(e)}"

@tool
def get_stock_quote(ticker: str) -> str:
    """Busca a cotação em tempo real de uma ação ou FII na B3."""
    print(f"-> [get_stock_quote] Requisitando para {ticker}...", flush=True)
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(bolsai_client.call_tool("get_stock_quote", {"ticker": ticker.upper()}))
    print(f"-> [get_stock_quote] Finalizado para {ticker}.", flush=True)
    return res

@tool
def get_fundamentals(ticker: str) -> str:
    """Busca múltiplos fundamentalistas de uma ação da B3."""
    print(f"-> [get_fundamentals] Requisitando para {ticker}...", flush=True)
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(bolsai_client.call_tool("get_fundamentals", {"ticker": ticker.upper()}))
    print(f"-> [get_fundamentals] Finalizado para {ticker}.", flush=True)
    return res

@tool
def get_dividends(ticker: str) -> str:
    """Busca informações de dividendos de um ativo da B3."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(bolsai_client.call_tool("get_dividends", {"ticker": ticker.upper()}))

@tool
def get_macro_data(indicator: str) -> str:
    """Busca o valor atual e histórico recente de indicadores macroeconômicos brasileiros."""
    print(f"-> [get_macro_data] Requisitando {indicator}...", flush=True)
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(bolsai_client.call_tool("get_macro_data", {"indicator": indicator.lower()}))
    print(f"-> [get_macro_data] Finalizado para {indicator}.", flush=True)
    return res

@tool
def compare_stocks(tickers: list) -> str:
    """Compara múltiplos fundamentalistas de até 5 tickers de ações da B3 lado a lado."""
    loop = asyncio.get_event_loop()
    tickers_upper = [t.upper() for t in tickers]
    return loop.run_until_complete(bolsai_client.call_tool("compare_stocks", {"tickers": tickers_upper}))

@tool
def get_fii_data(ticker: str) -> str:
    """Busca indicadores específicos de Fundos Imobiliários (FIIs)."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(bolsai_client.call_tool("get_fii_data", {"ticker": ticker.upper()}))

@tool
def screen_stocks(roe_min: float = None, pl_max: float = None, sector: str = None) -> str:
    """Filtra e retorna uma lista de ações da B3 que atendem a múltiplos financeiros específicos."""
    loop = asyncio.get_event_loop()
    args = {}
    if roe_min is not None: args["roe_min"] = roe_min
    if pl_max is not None: args["pl_max"] = pl_max
    if sector is not None: args["sector"] = sector
    return loop.run_until_complete(bolsai_client.call_tool("screen_stocks", args))

from langgraph.graph.message import add_messages

# LangGraph agent definitions
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Dict[str, Any]

@tool
def consult_market_analyst(query: str) -> str:
    """Delega uma tarefa ao Agente Pesquisador (Market Analyst)."""
    pass

@tool
def consult_b3_specialist(task: str) -> str:
    """Delega uma tarefa ao Especialista B3."""
    pass

# Initialize models
advisor_model = ChatOpenAI(model="gpt-4o", temperature=0.2).bind_tools([
    consult_market_analyst,
    consult_b3_specialist
])

ADVISOR_SYSTEM_PROMPT = (
    "Você é o Lead Advisor (Estrategista Financeiro) da Quantum Finance.\n"
    "Seu papel é receber as dúvidas do cliente, entender o perfil dele, planejar a busca de dados reais "
    "e delegar tarefas para seus subagentes usando as ferramentas:\n"
    "- 'consult_market_analyst' para dúvidas teóricas ou conceitos.\n"
    "- 'consult_b3_specialist' para dados de mercado, cotações e múltiplos da B3.\n\n"
    "Perfil do Cliente Atual:\n"
    "{user_profile}\n\n"
    "Diretrizes:\n"
    "1. Nunca tente chutar ou alucinar cotações e múltiplos. Sempre chame o consult_b3_specialist.\n"
    "2. Quando tiver todos os dados necessários reunidos, elabore um relatório final com a recomendação de alocação personalizada, "
    "justificando cada indicação de acordo com o perfil do cliente e os múltiplos da B3 de 2026 obtidos."
)

def lead_advisor_node(state: AgentState):
    print("LEAD ADVISOR MESSAGES RECEIVED:")
    for i, m in enumerate(state["messages"]):
        print(f"  [{i}] Type: {type(m).__name__}, ID: {getattr(m, 'id', None)}, ToolCalls: {getattr(m, 'tool_calls', None)}, ToolCallID: {getattr(m, 'tool_call_id', None)}")
    profile_str = "\n".join([f"- {k}: {v}" for k, v in state["user_profile"].items()])
    sys_msg = SystemMessage(content=ADVISOR_SYSTEM_PROMPT.format(user_profile=profile_str))
    response = advisor_model.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def subagents_executor_node(state: AgentState):
    last_message = state["messages"][-1]
    new_messages = []
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "consult_market_analyst":
            query = tool_call["args"]["query"]
            print(f"\n[Nó Ativo: subagents_executor -> Executando Market Analyst para a busca: '{query}']")
            
            analyst_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([search_financial_concept])
            analyst_system = (
                "Você é o Agente Pesquisador (Market Analyst). Sua tarefa é explicar de forma didática conceitos financeiros.\n"
                "Use a busca na internet para responder."
            )
            analyst_messages = [SystemMessage(content=analyst_system), HumanMessage(content=query)]
            response = analyst_model.invoke(analyst_messages)
            
            while response.tool_calls:
                analyst_messages.append(response)
                for tc in response.tool_calls:
                    result = search_financial_concept.invoke(tc["args"])
                    analyst_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                response = analyst_model.invoke(analyst_messages)
                
            tool_msg = ToolMessage(content=response.content, tool_call_id=tool_call["id"], name=tool_call["name"])
            new_messages.append(tool_msg)
            print(f"-> Resposta do Market Analyst obtida.")
            
        elif tool_call["name"] == "consult_b3_specialist":
            task = tool_call["args"]["task"]
            print(f"\n[Nó Ativo: subagents_executor -> Executando Agente B3 para a tarefa: '{task}']")
            
            b3_tools = [
                get_stock_quote,
                get_fundamentals,
                get_dividends,
                get_macro_data,
                compare_stocks,
                get_fii_data,
                screen_stocks
            ]
            b3_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(b3_tools)
            b3_system = (
                "Você é o Agente de Dados B3. Sua tarefa é extrair múltiplos e cotações da B3."
            )
            b3_messages = [SystemMessage(content=b3_system), HumanMessage(content=task)]
            print(f"-> Chamando b3_model.invoke...", flush=True)
            response = b3_model.invoke(b3_messages)
            print(f"-> b3_model.invoke retornou {len(response.tool_calls)} tool calls.", flush=True)
            
            while response.tool_calls:
                b3_messages.append(response)
                for tc in response.tool_calls:
                    target_tool = next(t for t in b3_tools if t.name == tc["name"])
                    print(f"-> Invocando tool {tc['name']}...", flush=True)
                    result = target_tool.invoke(tc["args"])
                    b3_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                print(f"-> Chamando b3_model.invoke novamente...", flush=True)
                response = b3_model.invoke(b3_messages)
                
            tool_msg = ToolMessage(content=response.content, tool_call_id=tool_call["id"], name=tool_call["name"])
            new_messages.append(tool_msg)
            print(f"-> Resposta do Agente B3 obtida.")
            
    return {"messages": new_messages}

def route_after_advisor(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "subagents_executor"

from langgraph.checkpoint.memory import MemorySaver

# Graph
memory = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("lead_advisor", lead_advisor_node)
workflow.add_node("subagents_executor", subagents_executor_node)
workflow.add_edge(START, "lead_advisor")
workflow.add_conditional_edges(
    "lead_advisor",
    route_after_advisor,
    {
        "subagents_executor": "subagents_executor",
        END: END
    }
)
workflow.add_edge("subagents_executor", "lead_advisor")
app = workflow.compile(checkpointer=memory)

def run_test():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bolsai_client.connect())
    
    perfil_arrojado = {
        "Nome": "Gabriel",
        "Perfil de Risco": "Arrojado",
        "Objetivo": "Maximizar ganhos em renda variável no longo prazo",
        "Horizonte de Tempo": "10 anos"
    }
    
    config = {"configurable": {"thread_id": "gabriel_thread"}, "recursion_limit": 15}
    
    # --- TURNO 1 ---
    pergunta1 = "Vale a pena investir em ações da WEGE3 ou ITUB4 hoje comparado com a Selic atual? Me explique brevemente o risco e a comparação de múltiplos."
    print("\n--- INICIANDO FLUXO DE TESTES (TURNO 1) ---")
    print(f"Pergunta: {pergunta1}")
    
    for event in app.stream({"messages": [HumanMessage(content=pergunta1)], "user_profile": perfil_arrojado}, config):
        for node_name, state_update in event.items():
            print(f"\n[Nó Ativo: {node_name}]")
            messages = state_update.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        print(f"-> Delegando ferramenta: {msg.tool_calls[0]['name']}")
                    else:
                        print(f"-> Resposta final do Lead Advisor:\n{msg.content}")
                elif isinstance(msg, ToolMessage):
                    print(f"-> Resposta de subagente recebida.")
                    
    # --- TURNO 2 ---
    pergunta2 = "E se meu perfil mudasse para Conservador, o que recomendaria?"
    perfil_conservador = {
        "Nome": "Gabriel",
        "Perfil de Risco": "Conservador",
        "Objetivo": "Preservar capital com retornos previsíveis",
        "Horizonte de Tempo": "10 anos"
    }
    
    print("\n--- CONTINUANDO FLUXO DE TESTES (TURNO 2 - CONVERSACIONAL COM MEMÓRIA) ---")
    print(f"Pergunta: {pergunta2}")
    
    for event in app.stream({"messages": [HumanMessage(content=pergunta2)], "user_profile": perfil_conservador}, config):
        for node_name, state_update in event.items():
            print(f"\n[Nó Ativo: {node_name}]")
            messages = state_update.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        print(f"-> Delegando ferramenta: {msg.tool_calls[0]['name']}")
                    else:
                        print(f"-> Resposta final do Lead Advisor:\n{msg.content}")
                elif isinstance(msg, ToolMessage):
                    print(f"-> Resposta de subagente recebida.")
                    
    loop.run_until_complete(bolsai_client.disconnect())

if __name__ == "__main__":
    run_test()
