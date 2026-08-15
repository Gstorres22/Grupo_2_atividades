"""Bateria completa de testes: personas geram, as duas versoes rodam, avaliadores julgam.

===============================================================================
O DESENHO DO EXPERIMENTO, E POR QUE ELE E ASSIM
===============================================================================

    [1] 5 personas geram mensagens ................. UMA vez
                    |
                    v
    [2] MESMAS mensagens rodam na V1 e na V1.0.1 ... comparacao controlada
                    |
                    v
    [3] Metricas + deteccao de divergencias ........ codigo, nao LLM
                    |
                    v
    [4] 2 avaliadores julgam o resultado ........... lentes diferentes

**Por que as personas geram UMA vez, e nao uma vez por versao.** Se cada versao
fosse testada com mensagens diferentes, uma diferenca de acerto poderia vir do
conjunto de mensagens em vez do sistema. Usando exatamente as mesmas entradas,
a unica variavel que resta e a versao — que e o que queremos medir. Em
experimento, isso se chama controle.

**Por que a etapa 3 e codigo e nao LLM.** Contar acertos e uma operacao exata.
Pedir a um LLM que calcule metricas introduz erro onde nao precisa haver
nenhum. LLM entra so onde ha julgamento subjetivo — na etapa 4.

USO
---
    python -m App.agents.run_suite                      # bateria completa
    python -m App.agents.run_suite --n-por-persona 10   # rodada rapida
    python -m App.agents.run_suite --pular-geracao      # reaproveita o dataset ja gerado
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from App.agents.base import resumir_custo
from App.agents.evaluators import avaliar
from App.agents.personas import gerar_dataset
from App.core.config import get_settings
from App.versions.base import BasePipeline, PipelineResult
from App.versions.v1_0_1_orchestrator import V101OrchestratorPipeline
from App.versions.v1_classic import V1ClassicPipeline
from common.data_loader import load_eval_dataset

DIR_RELATORIOS = Path(__file__).resolve().parent.parent / "reports"


# ---------------------------------------------------------------- metricas
def calcular_metricas(linhas: List[PipelineResult], casos: List[dict]) -> Dict:
    """Agrega as metricas de uma versao sobre um conjunto de casos.

    `linhas[i]` corresponde a `casos[i]` — a ordem e preservada de proposito,
    para conseguirmos cruzar resultado com gabarito sem precisar de chave.
    """
    acertos_rota = 0
    acertos_tool = 0
    total_tool = 0

    for resultado, caso in zip(linhas, casos):
        if caso.get("expected_route") and resultado.route == caso["expected_route"]:
            acertos_rota += 1
        if caso.get("expected_tool"):
            total_tool += 1
            # Conta como acerto SO se a ferramenta certa apareceu no top-k.
            # Se o roteador mandou para FAST_PATH, `tools` esta vazia e isso
            # conta como erro — de proposito. E o ponto cego que encontramos
            # na V1: calcular P@K apenas sobre o que o roteador acertou remove
            # da conta justamente os casos dificeis e infla a metrica.
            if caso["expected_tool"] in resultado.tools:
                acertos_tool += 1

    com_gabarito = sum(1 for c in casos if c.get("expected_route"))
    latencias = sorted(r.latency_ms for r in linhas)

    def _p(q: float) -> float:
        if not latencias:
            return 0.0
        return latencias[min(len(latencias) - 1, int(round(q * (len(latencias) - 1))))]

    return {
        "n": len(linhas),
        "n_com_gabarito_rota": com_gabarito,
        "n_com_gabarito_tool": total_tool,
        "acuracia_rota": acertos_rota / com_gabarito if com_gabarito else 0.0,
        "precision_at_k": acertos_tool / total_tool if total_tool else None,
        "custo_total_usd": sum(r.cost_usd for r in linhas),
        "custo_por_mensagem_usd": sum(r.cost_usd for r in linhas) / len(linhas) if linhas else 0.0,
        "latencia_p50_ms": statistics.median(latencias) if latencias else 0.0,
        "latencia_p95_ms": _p(0.95),
        "tokens_entrada": sum(r.prompt_tokens for r in linhas),
        "tokens_saida": sum(r.completion_tokens for r in linhas),
        "tokens_raciocinio": sum(r.reasoning_tokens for r in linhas),
        "chamadas_llm": sum(r.llm_calls for r in linhas),
        "erros": sum(1 for r in linhas if r.error),
    }


def metricas_por_persona(linhas: List[PipelineResult], casos: List[dict]) -> Dict:
    """Recorte por persona — mostra se o sistema quebra com um perfil especifico.

    A media geral esconde isso: um sistema pode ir bem no agregado e falhar
    sistematicamente com idosos. Foi exatamente o que a V1 fazia.
    """
    grupos: Dict[str, List[int]] = defaultdict(list)
    for indice, caso in enumerate(casos):
        grupos[caso.get("persona", "sem_persona")].append(indice)
    return {
        persona: calcular_metricas([linhas[i] for i in indices], [casos[i] for i in indices])
        for persona, indices in sorted(grupos.items())
    }


def detectar_divergencias(
    linhas_v1: List[PipelineResult], linhas_v101: List[PipelineResult], casos: List[dict]
) -> List[dict]:
    """Encontra os casos em que as duas versoes discordam.

    Sao os unicos que carregam informacao sobre a DIFERENCA entre elas. Onde as
    duas acertam ou as duas erram, nao ha nada a aprender sobre qual e melhor.
    Ordenamos colocando primeiro os casos em que uma acertou e a outra errou.
    """
    divergencias = []
    for a, b, caso in zip(linhas_v1, linhas_v101, casos):
        if a.route == b.route and a.tools == b.tools:
            continue  # concordaram: nao informa nada
        esperada = caso.get("expected_tool")
        v1_ok = a.route == caso.get("expected_route") and (not esperada or esperada in a.tools)
        v101_ok = b.route == caso.get("expected_route") and (not esperada or esperada in b.tools)
        divergencias.append({
            "query": caso["query"],
            "persona": caso.get("persona"),
            "expected_route": caso.get("expected_route"),
            "expected_tool": esperada,
            "v1_route": a.route, "v1_tools": a.tools, "v1_ok": v1_ok,
            "v101_route": b.route, "v101_tools": b.tools, "v101_ok": v101_ok,
            # 1 = so a V1.0.1 acertou; -1 = so a V1 acertou; 0 = ambas ou nenhuma
            "vantagem": (1 if v101_ok and not v1_ok else -1 if v1_ok and not v101_ok else 0),
        })
    # Primeiro os casos decisivos (vantagem != 0), depois os demais.
    divergencias.sort(key=lambda d: abs(d["vantagem"]), reverse=True)
    return divergencias


# ------------------------------------------------------------------ execucao
def rodar_versao(pipeline: BasePipeline, casos: List[dict], k: int = 2) -> List[PipelineResult]:
    """Roda um pipeline sobre a lista de casos, preservando a ordem."""
    resultados = []
    for i, caso in enumerate(casos, 1):
        if i % 25 == 0:
            print(f"    {i}/{len(casos)}...")
        resultados.append(pipeline.process(caso["query"], k=k))
    return resultados


def executar_bateria(
    n_por_persona: int = 30,
    pular_geracao: bool = False,
    k: int = 2,
) -> Dict:
    """Roda o experimento completo e devolve o dossie."""
    settings = get_settings()
    if not settings.agents_enabled:
        raise SystemExit("A bateria precisa de OPENAI_API_KEY em App/.env.")

    caminho_dataset = DIR_RELATORIOS / "dataset_personas.json"
    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------- [1] personas geram
    if pular_geracao and caminho_dataset.exists():
        print("[1/4] Reaproveitando o dataset ja gerado...")
        bruto = json.loads(caminho_dataset.read_text(encoding="utf-8"))
        casos_personas = bruto["queries"]
        custo_personas = bruto.get("custo", {})
    else:
        print(f"[1/4] 5 personas gerando ~{n_por_persona} mensagens cada (em paralelo)...")
        inicio = time.perf_counter()
        gerado = gerar_dataset(n_por_persona=n_por_persona, settings=settings)
        casos_personas = gerado["queries"]
        custo_personas = {
            "n_execucoes": len(gerado["execucoes"]),
            "n_sucesso": sum(1 for e in gerado["execucoes"] if e["ok"]),
            "custo_total_usd": sum(e["cost_usd"] or 0 for e in gerado["execucoes"]),
            "tokens_entrada": sum(e["prompt_tokens"] for e in gerado["execucoes"]),
            "tokens_saida": sum(e["completion_tokens"] for e in gerado["execucoes"]),
            "tempo_s": time.perf_counter() - inicio,
        }
        caminho_dataset.write_text(
            json.dumps({"queries": casos_personas, "custo": custo_personas,
                        "execucoes": gerado["execucoes"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"      {len(casos_personas)} mensagens geradas "
              f"(US$ {custo_personas['custo_total_usd']:.4f}, "
              f"{custo_personas['tempo_s']:.0f}s)")

    casos_oficiais = load_eval_dataset()

    # ------------------------------------------- [2] as duas versoes rodam
    print("\n[2/4] Preparando as duas versoes (mesma configuracao)...")
    v1 = V1ClassicPipeline(settings).setup()
    v101 = V101OrchestratorPipeline(settings).setup()
    print(f"      V1     : {v1.description}")
    print(f"      V1.0.1 : orquestrador = {settings.orchestrator_model} "
          f"(effort={settings.orchestrator_reasoning_effort})")

    comparacao: Dict = {"observacoes": [], "config": {
        "orchestrator_model": settings.orchestrator_model,
        "reasoning_effort": settings.orchestrator_reasoning_effort,
        "embedding_model": settings.embedding_model,
        "agents_model": settings.agents_model,
        "k": k,
    }}

    for chave, titulo, casos in [
        ("oficial", "dataset oficial (rotulos humanos)", casos_oficiais),
        ("personas", "conjunto das personas (rotulos de LLM)", casos_personas),
    ]:
        if not casos:
            continue
        print(f"\n      Rodando {titulo}: {len(casos)} mensagens")
        print("      V1...")
        linhas_v1 = rodar_versao(v1, casos, k)
        print("      V1.0.1...")
        linhas_v101 = rodar_versao(v101, casos, k)

        bloco = {
            "n_casos": len(casos),
            "metricas": {
                "V1": calcular_metricas(linhas_v1, casos),
                "V1.0.1": calcular_metricas(linhas_v101, casos),
            },
            "linhas": {
                "V1": [r.as_dict() for r in linhas_v1],
                "V1.0.1": [r.as_dict() for r in linhas_v101],
            },
        }
        if chave == "personas":
            bloco["por_persona"] = {
                persona: {
                    "V1": calcular_metricas(
                        [linhas_v1[i] for i, c in enumerate(casos) if c.get("persona") == persona],
                        [c for c in casos if c.get("persona") == persona]),
                    "V1.0.1": calcular_metricas(
                        [linhas_v101[i] for i, c in enumerate(casos) if c.get("persona") == persona],
                        [c for c in casos if c.get("persona") == persona]),
                }
                for persona in sorted({c.get("persona", "sem_persona") for c in casos})
            }
        comparacao[chave] = bloco
        comparacao.setdefault("divergencias", []).extend(
            detectar_divergencias(linhas_v1, linhas_v101, casos)
        )

    comparacao["fallbacks_v101"] = v101.fallbacks
    if v101.fallbacks:
        comparacao["observacoes"].append(
            f"A V1.0.1 caiu no plano B (decisao local da V1) em {v101.fallbacks} mensagens. "
            "Nessas, ela respondeu como a V1 — o que reduz a diferenca medida entre as duas."
        )
    if v101.custo_desconhecido:
        comparacao["observacoes"].append(
            f"Preco do modelo {settings.orchestrator_model} nao esta na tabela: "
            "o custo da V1.0.1 esta SUBESTIMADO no relatorio."
        )
    comparacao["observacoes"].append(
        "40% do dataset oficial tem sobreposicao alta com os 53 exemplos de treino "
        "da V1 (3 sao copias literais), o que favorece a V1 nesse conjunto."
    )
    comparacao["custo_geracao_personas"] = custo_personas

    # ------------------------------------------------ [3]+[4] avaliadores
    print("\n[3/4] Metricas calculadas. Divergencias encontradas: "
          f"{len(comparacao.get('divergencias', []))}")
    print("\n[4/4] 2 avaliadores especialistas analisando (em paralelo)...")
    execucoes = avaliar(comparacao, settings=settings)
    comparacao["avaliacoes"] = [e.as_dict() for e in execucoes]
    comparacao["custo_avaliadores"] = resumir_custo(execucoes)

    return comparacao


def imprimir_resumo(c: Dict) -> None:
    print("\n" + "=" * 96)
    print("BATERIA DE TESTES — V1 (ML classico)  x  V1.0.1 (orquestrador por LLM)")
    print("=" * 96)
    for chave, titulo in [("oficial", "DATASET OFICIAL (rotulos humanos)"),
                          ("personas", "CONJUNTO DAS PERSONAS (rotulos de LLM)")]:
        bloco = c.get(chave)
        if not bloco:
            continue
        print(f"\n{titulo} — {bloco['n_casos']} mensagens")
        print(f"  {'versao':10s} {'rota':>7s} {'P@2':>7s} {'$/msg':>11s} {'p50':>8s} {'p95':>8s}")
        for versao, m in bloco["metricas"].items():
            p2 = m["precision_at_k"]
            print(f"  {versao:10s} {m['acuracia_rota']:7.1%} "
                  f"{(p2 if p2 is not None else 0):7.1%} "
                  f"{m['custo_por_mensagem_usd']:11.6f} "
                  f"{m['latencia_p50_ms']:8.0f} {m['latencia_p95_ms']:8.0f}")

    por_persona = c.get("personas", {}).get("por_persona")
    if por_persona:
        print(f"\n  Por persona (rota / P@2):")
        for persona, versoes in por_persona.items():
            a, b = versoes["V1"], versoes["V1.0.1"]
            print(f"    {persona:24s} V1: {a['acuracia_rota']:.0%}/"
                  f"{(a['precision_at_k'] or 0):.0%}   "
                  f"V1.0.1: {b['acuracia_rota']:.0%}/{(b['precision_at_k'] or 0):.0%}")

    div = c.get("divergencias", [])
    print(f"\n  Divergencias: {len(div)} | "
          f"so V1.0.1 acertou: {sum(1 for d in div if d['vantagem'] == 1)} | "
          f"so V1 acertou: {sum(1 for d in div if d['vantagem'] == -1)}")

    print("\n  PARECER DOS AVALIADORES:")
    for a in c.get("avaliacoes", []):
        if not a["ok"]:
            print(f"    {a['agent_name']}: FALHOU ({a['error']})")
            continue
        p = a["payload"]
        print(f"    {a['agent_name']:22s} -> recomenda {p.get('recomendacao_final','?'):12s} "
              f"(confianca {p.get('confianca', 0):.0%})")
    for o in c.get("observacoes", []):
        print(f"\n  [!] {o}")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bateria de testes com subagentes.")
    parser.add_argument("--n-por-persona", type=int, default=30)
    parser.add_argument("--pular-geracao", action="store_true")
    parser.add_argument("-k", type=int, default=2)
    parser.add_argument("--out", default=str(DIR_RELATORIOS / "bateria_v1_x_v101.json"))
    args = parser.parse_args()

    dossie = executar_bateria(args.n_por_persona, args.pular_geracao, args.k)
    imprimir_resumo(dossie)

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dossie, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatorio completo: {destino}")


if __name__ == "__main__":
    main()
