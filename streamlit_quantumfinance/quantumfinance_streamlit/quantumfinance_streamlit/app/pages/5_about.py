"""Página: Sobre a POC."""

import streamlit as st


def render(client) -> None:
    st.subheader("Sobre esta POC")

    st.markdown(
        """
        Esta aplicação é uma **prova de conceito** da QuantumFinance que
        demonstra, para empresas parceiras, como integrar e consumir a
        API de **Credit Score**.

        ### O que esta interface faz
        - Coleta os dados do cliente em um formulário organizado por seções.
        - Envia os dados para a API de predição via `POST /predict`.
        - Exibe a classe prevista (Bom / Padrão / Ruim), a probabilidade
          de cada classe e a versão do modelo usada na inferência.
        - Permite predição em lote via upload de CSV.
        - Mantém um histórico de consultas durante a sessão.
        - Permite monitorar a saúde da API e a versão do modelo em produção.

        ### Dataset utilizado
        O modelo é treinado sobre o dataset
        [Credit Score Classification (Kaggle)](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data),
        contendo informações financeiras de clientes (renda, dívidas,
        histórico de pagamentos, etc.).

        ### Arquitetura
        - **Front-end:** Streamlit + Plotly.
        - **Back-end:** API REST desenvolvida pela equipe, com autenticação
          via API Key (Bearer Token) e throttling.
        - **Modelagem:** rastreamento e versionamento via MLflow Model Registry.

        ### Equipe
        Trabalho em grupo — disciplina de MLOps / Engenharia de ML.
        """
    )
