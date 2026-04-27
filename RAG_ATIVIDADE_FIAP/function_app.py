import os
import json
import logging
import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

app = func.FunctionApp()

# Configurações da OpenAI
OPENAI_API_KEY = os.getenv("API_KEY_OPEN_AI")

# Configurações Azure AI Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://seu-servico-search.search.windows.net")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "regras-futebol-index")

@app.route(route="query", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def query_rag(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Recebida uma requisição de RAG no endpoint /api/query.')

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Corpo da requisição inválido. Esperado JSON.",
            status_code=400
        )

    question = req_body.get('question')
    if not question:
        return func.HttpResponse(
            json.dumps({"error": "O campo 'question' é obrigatório."}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        # 1. Gerar o embedding da pergunta
        embeddings_model = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        question_vector = embeddings_model.embed_query(question)

        # 2. Fazer a busca vetorial no Azure AI Search
        search_client = SearchClient(
            endpoint=SEARCH_ENDPOINT, 
            index_name=INDEX_NAME, 
            credential=AzureKeyCredential(SEARCH_KEY)
        )

        vector_query = VectorizedQuery(
            vector=question_vector, 
            k_nearest_neighbors=5, 
            fields="content_vector"
        )
        
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["content", "source", "page"],
            top=5
        )

        # Formatar resultados
        sources = []
        context_texts = []
        for result in results:
            context_texts.append(f"Trecho (Pág {result['page']}): {result['content']}")
            sources.append({
                "chunk": result['content'],
                "source": result['source'],
                "page": result['page'],
                "score": result['@search.score']
            })

        context = "\n\n---\n\n".join(context_texts)

        # 3. Chamar o GPT-4o para gerar a resposta com base no contexto
        chat_model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )

        system_prompt = (
            "Você é um assistente especialista nas regras do futebol. "
            "Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE nos trechos de contexto fornecidos abaixo.\n"
            "Se o contexto não contiver a resposta, diga que não encontrou informações nas regras oficiais.\n\n"
            f"Contexto:\n{context}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        ai_msg = chat_model.invoke(messages)
        answer = ai_msg.content

        # 4. Retornar resposta
        response_body = {
            "answer": answer,
            "sources": sources
        }

        return func.HttpResponse(
            json.dumps(response_body, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Erro ao processar a consulta: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
