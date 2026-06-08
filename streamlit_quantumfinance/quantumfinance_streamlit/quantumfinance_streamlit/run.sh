#!/usr/bin/env bash
# Script de conveniência para subir o app Streamlit localmente.
set -euo pipefail

# Cria venv se não existir
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Ativa venv e instala dependências
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Sobe o app
streamlit run app.py "$@"
