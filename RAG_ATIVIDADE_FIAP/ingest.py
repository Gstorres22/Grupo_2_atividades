import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)

# Carrega variáveis de ambiente
load_dotenv()

# Configurações Azure OpenAI
OPENAI_API_KEY = os.getenv("API_KEY_OPEN_AI")

# Configurações Azure AI Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://seu-servico-search.search.windows.net")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "sua_chave_admin_do_search")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "regras-futebol-index")

def create_index_if_not_exists(index_client: SearchIndexClient):
    try:
        index_client.get_index(INDEX_NAME)
        print(f"Índice '{INDEX_NAME}' já existe.")
    except Exception:
        print(f"Criando o índice '{INDEX_NAME}'...")
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, 
                vector_search_dimensions=1536, 
                vector_search_profile_name="myHnswProfile"
            ),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True)
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw")
            ],
            profiles=[
                VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")
            ]
        )

        index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
        index_client.create_index(index)
        print(f"Índice '{INDEX_NAME}' criado com sucesso.")

def main():
    print("Iniciando pipeline de ingestão...")

    pdf_path = os.path.join("regras_futebol", "REGRAS-DO-JOGO-24-25.pdf")
    if not os.path.exists(pdf_path):
        print(f"ERRO: Arquivo não encontrado em {pdf_path}")
        return

    # 1. Carregamento do PDF
    print("Carregando documento PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Chunking Recursivo
    print("Realizando chunking recursivo...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Gerados {len(chunks)} chunks.")

    # 3. Geração de Embeddings
    print("Inicializando gerador de embeddings...")
    embeddings_model = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    # 4. Configuração Azure AI Search
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
    create_index_if_not_exists(index_client)

    search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

    # 5. Inserção no Banco Vetorial
    print("Processando embeddings e inserindo no Azure AI Search...")
    batch = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        # Tratamento de metadados padrão do PyPDFLoader
        source = chunk.metadata.get("source", "REGRAS-DO-JOGO-24-25.pdf")
        page = chunk.metadata.get("page", 0)

        # Gera o vetor
        vector = embeddings_model.embed_query(content)

        doc = {
            "id": str(uuid.uuid4()),
            "content": content,
            "content_vector": vector,
            "source": os.path.basename(source),
            "page": page
        }
        batch.append(doc)

        # Envia em lotes de 100
        if len(batch) >= 100:
            search_client.upload_documents(documents=batch)
            print(f"Enviados {i + 1} / {len(chunks)} chunks...")
            batch = []

    # Envia os restantes
    if batch:
        search_client.upload_documents(documents=batch)
        print(f"Enviados {len(chunks)} / {len(chunks)} chunks.")

    print("Ingestão concluída com sucesso!")

if __name__ == "__main__":
    main()
