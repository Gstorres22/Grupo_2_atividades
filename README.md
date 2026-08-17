# Arquivos

Em [`entregavel-case-router/`](entregavel-case-router/) está o
entregável do case.

Já em [`Treinamento-case-router/`](Treinamento-case-router/) eu deixei os notebooks, os testes com
personas, os relatórios e tudo que usei para desenvolver.

# Obejtivo Case

| Pilar | Entregavel | Oque eu coloquei |
|---|---|---|
| 1 — Router | `candidate_starter/router.py` | Classificador local: TF-IDF de caractere + regressão logística |
| 2 — Seleção de tools | `candidate_starter/retrieval.py` | Busca híbrida em dois estágios, 100% local |
| 3 — Harness | `candidate_starter/harness.py` | As três métricas pedidas |

## Decisão

Pensei inicialmente em ter um "orquestrador" logo após a solicitação do usuário, onde ele decidiria
entre chamar a query simples ou a complexa. No entanto, como o objetivo era evitar custo, usar LLM
não era a melhor abordagem no início. Então preferi uma abordagem híbrida, usando ML clássico para
criar um classificador local: termos de frequência e regressão logística, treinando com os 53
exemplos que eu tinha.

Como eu tinha poucos exemplos para treinar, só os 53, ai eeu fui com n-gramas de caracteres ao invés de
palavras. Isso resolve o problema de palavras como (parcelar/parcelamento) e não afeta entradas com
erro de digitação, que eu forcei nos testes.

Para a busca das tools, só medir a similaridade não teve um bom resultado porque das 285 tools tem
algumas que são muito parecidas e para várias mensagens a mais parecida não é a certa:

> "Manda o pdf da minha fatura atual" → a busca devolve `enviar_pdf_fatura_atual`,
> mas o esperado é `consultar_fatura`.

Para corrigir isso eu testei algumas abordagens de busca híbrida e o que funcionou foi fundir duas buscas (caractere e palavra) 
e depois penalizar as ferramentas de nome muito específico, preferindo a mais genérica.

## Resultados

| Métrica pedida | Baseline ingênuo | Meu resultado |
|---|---|---|
| **Acurácia do router** | 66,7% (chutar sempre a classe majoritária) | **100%** · IC 95%: [88,6%, 100%] |
| **Matriz de confusão** | — | 10/10 FAST_PATH · 20/20 AGENT · zero erros |
| **Precision@2 do retriever** | 25% (similaridade pura) · 0,7% (aleatório) | **85%** |
| **Economia de custo** | 0% (mandar tudo pro LLM caro) | **77,8%** |
| **Economia de latência** | 0% | **97,7%** · 65,3% comparando escopo igual |
