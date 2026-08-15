"""Subagentes de TESTE — personas de usuario e avaliadores de engenharia.

Estes agentes NAO fazem parte do produto. Eles nao atendem cliente nenhum:
existem apenas para gerar casos de teste e julgar resultados, sempre offline.

    personas.py    -> 5 agentes que escrevem mensagens como clientes reais
    evaluators.py  -> 2 agentes que avaliam a comparacao V1 x V1.0.1
    run_suite.py   -> orquestra tudo e salva os relatorios

Construidos com o SDK da OpenAI. Uma restricao de metodo atravessa todo o
modulo: os subagentes usam um modelo DIFERENTE do orquestrador que eles
avaliam, porque LLMs tendem a preferir as proprias saidas.
"""
