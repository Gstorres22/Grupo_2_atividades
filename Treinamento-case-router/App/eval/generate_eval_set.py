"""Amplia o conjunto de avaliacao usando um LLM como gerador de dados.

===============================================================================
POR QUE AMPLIAR
===============================================================================

O `eval_dataset.json` tem 30 queries (20 com tool esperada). Nesse tamanho, a
diferenca entre 29/30 e 30/30 e UMA linha, e o intervalo de confianca de 95%
para "100% de acerto" vai de 88,6% a 100%. Qualquer decisao de engenharia
tomada com base em uma diferenca de 5 pontos percentuais nesse conjunto esta,
na pratica, sendo tomada no ruido.

Ampliar para ~200 queries reduz a largura do intervalo pela metade e permite:
  * calibrar o `generality_lambda` do retriever SEM tocar no eval original
    (que fica reservado como conjunto de teste limpo);
  * medir robustez a erro de digitacao e a giria, que o eval original nao cobre;
  * medir o comportamento com queries FORA DE ESCOPO — o catalogo tem uma
    categoria `utilidades` (piada, horoscopo, receita) que existe justamente
    como ruido, e o eval oficial nao testa isso.

===============================================================================
CUIDADOS (dado sintetico tem armadilhas)
===============================================================================

1. NAO gerar a partir do gabarito para depois avaliar contra o gabarito.
   Geramos PARAFRASES das queries existentes preservando o rotulo original — o
   rotulo vem do humano que montou o eval, nao do LLM.
2. Para queries novas (fora de escopo, ambiguas), o rotulo do LLM e SILVER
   (prata), nao gold. O arquivo marca cada linha com `label_source`, e as
   metricas devem ser reportadas separadas por fonte de rotulo.
3. Revisao humana por amostragem e obrigatoria antes de usar em decisao.
   O script imprime uma amostra aleatoria ao final justamente para isso.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from App.core.config import APP_DIR, get_settings
from common.data_loader import load_eval_dataset

PARAPHRASE_PROMPT = """Voce gera dados de teste para um roteador de atendimento bancario.

Receba uma mensagem de cliente e produza {n} VARIACOES dela que preservem
EXATAMENTE a mesma intencao. Distribua as variacoes entre estes estilos:

- "leigo": informal, frases curtas, sem termos tecnicos ("quanto tenho de grana?")
- "typo": com erros de digitacao reais de quem escreve rapido no celular
- "formal": educado e completo, como um e-mail
- "especialista": usa o jargao bancario correto ("saldo em conta corrente")
- "prolixo": da contexto demais antes de pedir o que quer

REGRAS:
- A intencao NAO pode mudar. Se o original pede o saldo, todas pedem o saldo.
- Nao invente dados sensiveis (CPF, numero de conta, valores reais).
- Escreva em portugues do Brasil.

Responda SOMENTE com JSON:
{{"variacoes": [{{"query": "...", "estilo": "leigo"}}]}}"""

NEW_QUERIES_PROMPT = """Voce gera dados de teste para um roteador de atendimento de um banco digital.

Gere {n} mensagens de cliente da categoria: {categoria}

Definicao da categoria:
{descricao}

REGRAS:
- Portugues do Brasil, linguagem de chat real.
- Variedade: nao repita a mesma estrutura de frase.
- Nao invente dados sensiveis.

Responda SOMENTE com JSON:
{{"queries": [{{"query": "...", "expected_route": "FAST_PATH" | "AGENT"}}]}}"""

CATEGORIES: Dict[str, str] = {
    "fora_de_escopo": (
        "Pedidos que NAO tem nada a ver com banco (piada, previsao do tempo, receita, "
        "horoscopo, musica). O catalogo tem tools de 'utilidades' que atendem isso, "
        "entao a rota esperada e AGENT — o objetivo e medir se o sistema nao se perde."
    ),
    "ambigua": (
        "Mensagens curtas e vagas onde nem um humano decidiria com certeza "
        "('preciso de ajuda com meu cartao', 'tenho um problema'). "
        "Rota esperada: AGENT, por dependerem da conta do cliente."
    ),
    "faq_generica": (
        "Perguntas sobre a instituicao que NAO dependem dos dados da conta "
        "(horario, taxas de tabela, documentos, como funciona um produto). "
        "Rota esperada: FAST_PATH."
    ),
    "acao_conta": (
        "Pedidos que dependem dos dados ou de uma acao na conta daquele cliente "
        "(saldo, fatura, bloqueio, cadastro, investimento). Rota esperada: AGENT."
    ),
}


def _chat_json(client, model: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.9,  # alta de proposito: queremos DIVERSIDADE, nao consistencia
    )
    return json.loads(response.choices[0].message.content)


def generate(n_variations: int = 4, n_per_category: int = 15, seed: int = 42) -> List[dict]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit(
            "Este script precisa de OPENAI_API_KEY em App/.env.\n"
            "O restante do projeto roda sem chave — so a ampliacao do eval depende dela."
        )
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    random.seed(seed)
    generated: List[dict] = []

    # 1) Parafrases: herdam o rotulo HUMANO do eval original -> gold.
    for item in load_eval_dataset():
        payload = _chat_json(
            client,
            settings.model_judge,
            PARAPHRASE_PROMPT.format(n=n_variations) + f"\n\nMensagem original: {item['query']!r}",
        )
        for variation in (payload or {}).get("variacoes", []):
            if not variation.get("query"):
                continue
            generated.append({
                "query": variation["query"],
                "expected_route": item["expected_route"],   # rotulo humano preservado
                "expected_tool": item.get("expected_tool"),
                "persona": variation.get("estilo", "parafrase"),
                "label_source": "gold_derivado",
                "origem": item["query"],
            })

    # 2) Queries novas: rotulo do LLM -> silver, marcado como tal.
    for categoria, descricao in CATEGORIES.items():
        payload = _chat_json(
            client,
            settings.model_judge,
            NEW_QUERIES_PROMPT.format(n=n_per_category, categoria=categoria, descricao=descricao),
        )
        for entry in (payload or {}).get("queries", []):
            if not entry.get("query"):
                continue
            generated.append({
                "query": entry["query"],
                "expected_route": entry.get("expected_route"),
                "expected_tool": None,       # sem gabarito de tool: nao inventamos
                "persona": categoria,
                "label_source": "silver_llm",
            })

    random.shuffle(generated)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Amplia o conjunto de avaliacao via LLM.")
    parser.add_argument("--variations", type=int, default=4)
    parser.add_argument("--per-category", type=int, default=15)
    parser.add_argument("--out", default=str(APP_DIR / "eval" / "eval_ampliado.json"))
    args = parser.parse_args()

    dataset = generate(args.variations, args.per_category)
    Path(args.out).write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    gold = sum(1 for d in dataset if d["label_source"] == "gold_derivado")
    print(f"Geradas {len(dataset)} queries ({gold} gold-derivadas, {len(dataset)-gold} silver).")
    print(f"Salvo em: {args.out}\n")
    print("AMOSTRA PARA REVISAO HUMANA (obrigatoria antes de usar em decisao):")
    for item in random.sample(dataset, min(12, len(dataset))):
        print(f"  [{item['label_source']:14s}][{item['persona']:14s}] "
              f"{item['expected_route'] or '?':9s} | {item['query']}")


if __name__ == "__main__":
    main()
