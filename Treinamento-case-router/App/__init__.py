"""Camada de aplicacao do case: orquestracao, cascata de LLM e avaliacao.

Depende do entregavel (`candidate_starter/` e `common/`). O contrario NUNCA
acontece — essa direcao unica de dependencia e o que mantem o nucleo do case
rodando offline, sem chave de API e sem dependencia alem do requirements.txt
original.

--------------------------------------------------------------------------
POR QUE ESTE ARQUIVO MEXE COM sys.path
--------------------------------------------------------------------------
Esta camada vive FORA da pasta do entregavel (ela e material de apoio). Mas
importa `candidate_starter` e `common`, que sao pacotes de topo la dentro. Sem
ajuda, `import candidate_starter` so funcionaria com o interpretador iniciado
de dentro do entregavel.

Em vez de exigir um cwd especifico ou um PYTHONPATH manual, o pacote localiza a
pasta do entregavel sozinho e a coloca no `sys.path`. Isso roda uma unica vez,
no import de `App`, antes de qualquer submodulo — por isso mora aqui e nao em
cada arquivo.

A busca e por CONTEUDO, nunca por nome de pasta: procura quem contem
`candidate_starter/` E `common/`. Ja renomeamos essas pastas uma vez
(`case-agents` -> `entregavel-case-router`); amarrar a um nome fixo significaria
quebrar de novo na proxima. O criterio de conteudo sobrevive a renomeacao.

    cd Treinamento-case-router
    python -m App.main
    python -m App.eval.run_batch --query "quero bloquear meu cartao"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Quantos niveis subir antes de desistir. 4 cobre com folga o layout real
# (App/ -> Treinamento-case-router/ -> raiz do repositorio); o limite existe
# para nao varrer C:\ inteiro se algo estiver fora do lugar.
_NIVEIS_MAX = 4


def _e_o_entregavel(caminho: Path) -> bool:
    """A pasta contem os dois pacotes que precisamos importar?"""
    return (caminho / "candidate_starter").is_dir() and (caminho / "common").is_dir()


def _localizar_entregavel() -> Path:
    """Devolve a pasta do entregavel do case.

    Sobe a arvore a partir daqui e, em cada nivel, testa o proprio nivel e
    depois cada subpasta dele — e assim que a pasta irma e encontrada. Levanta
    erro explicito se nao achar: falhar aqui com mensagem clara e melhor do que
    estourar um `ModuleNotFoundError` opaco la na frente.
    """
    aqui = Path(__file__).resolve().parent
    for base in (aqui, *list(aqui.parents)[:_NIVEIS_MAX]):
        if _e_o_entregavel(base):
            return base
        try:
            filhos = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            continue  # sem permissao de leitura nesse nivel; segue subindo
        for filho in filhos:
            if _e_o_entregavel(filho):
                return filho
    raise RuntimeError(
        "Nao encontrei a pasta do entregavel (a que contem candidate_starter/ e "
        f"common/) subindo ate {_NIVEIS_MAX} niveis a partir de {aqui}. "
        "Esta camada precisa dela para importar o nucleo do case."
    )


CASE_DIR = _localizar_entregavel()
"""Raiz do entregavel do case. Unica fonte de verdade para caminhos do nucleo."""

APP_DIR = Path(__file__).resolve().parent
"""Raiz desta camada de aplicacao (onde ficam .env, cache/ e reports/)."""

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
