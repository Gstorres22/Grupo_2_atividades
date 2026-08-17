"""As duas versoes do cerebro de roteamento, sob um contrato comum.

    v1_classic      -> V1     : classificador de Machine Learning classico (scikit-learn)
    v1_0_1_orchestrator -> V1.0.1: agente orquestrador por LLM

Ambas implementam `BasePipeline` (ver base.py). Isso permite que o MESMO
conjunto de mensagens de teste rode nas duas e que os resultados sejam
comparaveis linha a linha.
"""
