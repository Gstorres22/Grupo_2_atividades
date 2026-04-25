import os
import tempfile
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

# Configuracoes
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16 MB
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_profile_with_ai(text_content):
    # A chave da API esta configurada no ambiente como API_KEY_OPEN_AI
    api_key = os.environ.get("API_KEY_OPEN_AI")
    if not api_key:
        raise ValueError("A variavel de ambiente API_KEY_OPEN_AI nao foi encontrada ou esta vazia.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=api_key)
    
    # System Prompt extremamente robusto e focado
    system_prompt = (
        "Voce e um Tech Recruiter Senior e Especialista em Personal Branding e Otimizacao de LinkedIn. "
        "Sua tarefa e realizar uma auditoria completa e implacavel em um perfil do LinkedIn (fornecido como texto extraido de um PDF) "
        "para otimizar a atratividade do usuario para recrutadores de tecnologia de alto nivel.\n\n"
        "REGRAS ESTRITAS DE FORMATACAO:\n"
        "1. NAO utilize NENHUM emoji em sua resposta. Isso e absolutamente proibido e inegociavel.\n"
        "2. Utilize Markdown limpo para formatar sua resposta (titulos com #, listas com -, negrito com **).\n"
        "3. Sua analise deve focar na 'narrativa de carreira' do usuario, nao apenas em palavras-chave soltas.\n\n"
        "ESTRUTURA OBRIGATORIA DA SUA RESPOSTA:\n"
        "A. MAPEAMENTO DE ACERTOS: Identifique o que o perfil atual faz bem.\n"
        "B. PONTOS FRACOS E ERROS: Aponte detalhadamente os erros (ex: falta de palavras-chave, descricoes vagas, ausencia de metricas de impacto).\n"
        "C. SUGESTOES PRATICAS E REESCRITAS: Forneca reescritas reais e diretas de como o usuario pode melhorar:\n"
        "   - O Titulo (Headline)\n"
        "   - O Resumo (About)\n"
        "   - A secao de Experiencias (focando em metricas de impacto e clareza)\n\n"
        "Seja profissional, direto e forneca muito valor tecnico. Lembre-se: proibido usar emojis."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("Aqui esta o texto extraido do PDF do perfil do LinkedIn do usuario:\n\n{profile_text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"profile_text": text_content})
    return response

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado.'}), 400
        
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado.'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extrair o texto do PDF usando o PyPDFLoader do LangChain
            loader = PyPDFLoader(filepath)
            pages = loader.load_and_split()
            full_text = " ".join([page.page_content for page in pages])
            
            if not full_text.strip():
                return jsonify({'error': 'Nao foi possivel extrair texto deste PDF. Ele pode estar vazio ou ser apenas imagens.'}), 400
            
            # Enviar para a IA analisar
            analysis_result = analyze_profile_with_ai(full_text)
            
            return jsonify({'result': analysis_result})
            
        except Exception as e:
            return jsonify({'error': f'Erro ao processar o PDF ou na analise da IA: {str(e)}'}), 500
            
        finally:
            # Limpar o arquivo temporario
            if os.path.exists(filepath):
                os.remove(filepath)
                
    else:
        return jsonify({'error': 'Apenas arquivos PDF sao permitidos.'}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
