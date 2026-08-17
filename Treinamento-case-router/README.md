# Treinamento-case-router — material de apoio

**Nada aqui é entregável.** O case está em
[`../entregavel-case-router/`](../entregavel-case-router/), que roda sozinho, offline, com o
`requirements.txt` original — e não importa uma linha desta pasta.

Aqui fica o que foi construído **em volta** do case: a camada de aplicação que serviu para
testá-lo com clientes simulados, as versões alternativas que foram comparadas, e a documentação
do raciocínio.

## Estrutura

```
Treinamento-case-router/
├── Case_Storytelling_Revisao.docx   a jornada do projeto, em narrativa
├── docs/                            documentação atual
│   └── historico/                   versões superadas, mantidas como registro
└── App/                             a camada de aplicação (código)
    ├── core/ versions/ agents/ eval/
    ├── notebooks/                   desenvolvimento passo a passo
    └── reports/                     JSON bruto dos experimentos
        └── historico/
```

## Por onde começar

| Quero… | Abrir |
|---|---|
| Entender a jornada do projeto, em narrativa | [`Case_Storytelling_Revisao.docx`](Case_Storytelling_Revisao.docx) |
| Saber **por que** cada escolha técnica foi feita | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Ver os bugs achados testando com personas de usuário | [`docs/V1_DESCOBERTAS.md`](docs/V1_DESCOBERTAS.md) |
| Ver os diagramas e a separação núcleo × aplicação | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Entender a comparação entre as três versões | [`docs/V1_0_2_RESULTADO.md`](docs/V1_0_2_RESULTADO.md) |

Os documentos em [`docs/historico/`](docs/historico/) foram medidos antes da correção do prior de
generalidade e estão marcados como tal. Ficaram porque o **método** e as ressalvas continuam
válidos — reescrever os números de uma execução que não aconteceu falsearia o histórico.

## O que tem em `App/`

Uma camada de orquestração (LangGraph + SDK da OpenAI) construída sobre o núcleo do case, usada
para responder três perguntas que o case não pede mas a conversa técnica provavelmente pede:

1. **Um LLM decidindo a rota seria melhor?** → `versions/v1_0_1_orchestrator.py`
2. **Dá para ter a qualidade do LLM sem pagar por toda mensagem?** → `versions/v1_0_2_hybrid.py`
3. **O resultado do case se sustenta fora do dataset dele?** → `agents/personas.py`, 5 personas de
   usuário que geraram 150 mensagens fora do conjunto oficial

Os JSON em [`App/reports/`](App/reports/) são o registro bruto desses experimentos, versionados de
propósito: sem eles, as conclusões da documentação não teriam como ser conferidas por outra pessoa.

## Como rodar

Precisa de `OPENAI_API_KEY` em `App/.env` (use [`App/.env.example`](App/.env.example) como modelo)
e das dependências extras. **A partir desta pasta:**

```bash
pip install -r App/requirements-app.txt
```

```bash
python -m App.main
```

O `App/__init__.py` localiza a pasta do entregável sozinho — procurando por **conteúdo**, não por
nome — e a coloca no `sys.path`. Não é preciso configurar `PYTHONPATH`. Sem chave de API, tudo
continua rodando em modo local (léxico puro) e avisa no console.
