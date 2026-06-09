# Projeto de Autenticação Facial com Liveness Ativo por Piscada 👤✨

Este projeto foi desenvolvido como um trabalho acadêmico para demonstrar uma solução robusta e simples de **Autenticação Facial**, unindo a facilidade de identificação do DeepFace com a detecção de vivacidade (liveness) ativa baseada em piscadas utilizando o **MediaPipe Face Landmarker**.

## 🚀 Funcionalidades

O sistema expõe uma API extremamente simples de duas funções principais:
1. **`cadastrar_usuario(nome)`**: Captura a face do usuário via webcam, valida a presença de um rosto e salva a imagem cadastrada na pasta `database/`.
2. **`autenticar()`**: Inicia o fluxo completo de autenticação:
   - **Liveness Detecção**: Exige que o usuário pisque os olhos pelo menos 2 vezes dentro do limite de tempo para garantir que se trata de uma pessoa real (bloqueando ataques de foto estática ou vídeo pausado).
   - **Reconhecimento Facial**: Identifica se o rosto da pessoa liveness-aprovada corresponde a algum dos usuários cadastrados no banco de dados local.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **OpenCV**: Captura de vídeo e processamento de imagem em tempo real.
- **MediaPipe Tasks (Face Landmarker)**: Extração de marcos faciais (landmarks) e blendshapes para cálculo do EAR (Eye Aspect Ratio) e detecção de piscadas.
- **DeepFace**: Biblioteca de análise e reconhecimento facial (usa modelo VGG-Face por padrão).
- **Matplotlib / Numpy / Pandas / CSV**: Visualização e manipulação de logs/dados de evidência.

---

## 📋 Pré-requisitos e Como Rodar

1. **Clone o repositório** e navegue até a pasta do projeto:
   ```bash
   cd C:\Users\stgab\OneDrive\Documentos\GitHub\Grupo_2_atividades\Autenticacao_facial_final\
   ```

2. **Crie e ative um ambiente virtual** (Venv):
   ```bash
   # No Windows (PowerShell/CMD):
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Abra o Jupyter Notebook**:
   ```bash
   jupyter notebook autenticacao_facial.ipynb
   ```
   *Ou execute diretamente pelo VS Code / ambiente de desenvolvimento preferido.*

---

## 👁️ Estratégia de Liveness (Detecção de Piscada)

Para evitar fraudes por fotos estáticas ou telas de celular, implementamos um desafio de **Liveness Ativo**:
- É calculado o **EAR (Eye Aspect Ratio)** de ambos os olhos usando as coordenadas tridimensionais dos marcos faciais do MediaPipe.
- Adicionalmente, são analisados os **Blendshapes** da Tasks API do MediaPipe (`eyeBlinkLeft` e `eyeBlinkRight`).
- O usuário deve piscar um número configurado de vezes (`CONFIG["PISCADAS_NECESSARIAS"] = 2`) dentro de um limite de tempo (`CONFIG["LIVENESS_TIMEOUT"] = 15` segundos).
- Se o olho for detectado como fechado e em seguida aberto, conta-se uma piscada. Apenas após passar no liveness o rosto é enviado ao classificador facial.

---

## 📂 Estrutura de Diretórios

- `database/`: Armazena as fotos cadastradas dos usuários no formato `<nome>.jpg`.
- `failed_attempts/`: Guarda fotos de evidência de tentativas que falharam no liveness ou no reconhecimento, além de um log estruturado em CSV (`log.csv`).
- `models/`: Pasta onde o modelo do MediaPipe (`face_landmarker.task`) é baixado automaticamente.

---

## 🔒 Nota de Privacidade (LGPD)

> [!IMPORTANT]
> Em conformidade com a **Lei Geral de Proteção de Dados (LGPD)**, dados biométricos (como imagens faciais e coordenadas geométricas de rostos) são considerados **dados pessoais sensíveis**. 
> Em um ambiente de produção real, é mandatório:
> 1. Obter o consentimento explícito e destacado do titular dos dados para a finalidade específica de autenticação biométrica.
> 2. Criptografar as imagens e representações vetoriais em trânsito e em repouso.
> 3. Definir políticas rígidas de retenção e descarte seguro dos dados.
> 4. Garantir que as evidências salvas de tentativas malsucedidas sejam acessadas apenas por pessoal autorizado e com fins de auditoria de fraudes.
