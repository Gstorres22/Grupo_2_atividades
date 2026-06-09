# Relatório de Desenvolvimento - Autenticação Facial com Liveness Ativo 🛡️👤

Este relatório descreve o desenvolvimento do projeto de **Autenticação Facial** contendo detecção de vivacidade (liveness) ativa por piscada de olhos e identificação de identidade.

---

## 📂 Arquivos Criados

Os seguintes arquivos foram gerados na estrutura de pastas recomendada:

1. `autenticacao_facial.ipynb`: Notebook Jupyter principal contendo todo o fluxo didático, códigos documentados e explicações.
2. `requirements.txt`: Arquivo com a especificação das dependências necessárias para a execução do projeto (OpenCV, MediaPipe, DeepFace, etc.).
3. `README.md`: Documentação descrevendo os pré-requisitos, passo a passo para execução, funcionamento do liveness e nota de privacidade de acordo com a LGPD.
4. `_build_nb.py`: Script Python responsável pela montagem automatizada do notebook célula a célula via `nbformat`.
5. `database/.gitkeep`: Diretório reservado para armazenar as fotos cadastradas dos usuários (`<nome>.jpg`).
6. `failed_attempts/.gitkeep`: Diretório reservado para logs de evidências e imagens de tentativas malsucedidas de acesso.
7. `models/.gitkeep`: Diretório reservado para conter os modelos de aprendizado de máquina (como o `face_landmarker.task` do MediaPipe).

---

## 🛠️ Montagem do Notebook e Decisões de Projeto

A geração do notebook Jupyter seguiu rigorosamente os passos indicados:
- O arquivo `_build_nb.py` foi programado para criar as células e o formato de saída do notebook seguindo a especificação `nbformat`.
- **Liveness por Piscada**: Unimos o cálculo matemático do EAR (Eye Aspect Ratio) com a avaliação de Blendshapes do MediaPipe Face Landmarker (`eyeBlinkLeft`/`eyeBlinkRight`), criando uma detecção extremamente robusta e imune a ataques de fotos estáticas.
- **Cadastro**: Captura do frame do usuário pela webcam e validação da presença de face com a biblioteca DeepFace antes de salvar a foto de forma permanente. Em seguida, limpa-se o cache do DeepFace para forçar a re-indexação imediata.
- **Evidências**: Para qualquer tentativa de fraude, o frame é salvo em `failed_attempts/` junto com um registro unificado no CSV `log.csv`.
- **Autenticação**: O fluxo executa primeiramente a validação de liveness ativo e, caso aprovado, submete o rosto ao classificador facial `DeepFace.find` utilizando o modelo `VGG-Face`.
- **Execução**: O script `_build_nb.py` foi executado com o interpretador do ambiente virtual da aplicação (`.venv\Scripts\python.exe`) e gerou o notebook `autenticacao_facial.ipynb` com sucesso e sem qualquer erro sintático ou de execução.

---

## ⚙️ Como Executar o Projeto

1. **Ative o ambiente virtual**:
   ```bash
   .venv\Scripts\activate
   ```
2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Abra e execute o Notebook**:
   ```bash
   jupyter notebook autenticacao_facial.ipynb
   ```
   No notebook, a célula correspondente ao modelo do MediaPipe fará o download do arquivo `face_landmarker.task` automaticamente se ele ainda não estiver na pasta `models/`.
