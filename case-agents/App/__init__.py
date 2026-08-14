"""Camada de aplicacao do case: orquestracao, cascata de LLM e avaliacao.

Depende de `candidate_starter/` e `common/`. O contrario NUNCA acontece — essa
direcao unica de dependencia e o que mantem o nucleo do case rodando offline,
sem chave de API e sem dependencia alem do requirements.txt original.
"""
