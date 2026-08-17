"""Bateria de testes: N versoes, as MESMAS mensagens, 2 avaliadores.

===============================================================================
O DESENHO DO EXPERIMENTO, E POR QUE ELE E ASSIM
===============================================================================

    [1] 5 personas geram mensagens ................. UMA vez, e reaproveitadas
                    |
                    v
    [2] As MESMAS mensagens rodam em TODAS as versoes ... comparacao controlada
                    |
                    v
    [3] Metricas + divergencias par a par .......... codigo, nao LLM
                    |
                    v
    [4] 2 avaliadores julgam o resultado ........... lentes diferentes

**Por que as personas geram UMA vez.** Se cada versao fosse testada com mensagens
diferentes, uma diferenca de acerto poderia vir do conjunto de mensagens em vez
do sistema. Com as mesmas entradas, a unica variavel e a versao. Em experimento,
isso se chama controle.

Isso vale ainda mais forte a partir da V1.0.2: o arquivo `dataset_personas.json`
gerado na rodada anterior e REAPROVEITADO, entao a V1.0.2 e comparada com a
V1.0.1 exatamente sobre as mesmas 150 mensagens que a V1.0.1 ja enfrentou.
Gerar mensagens novas invalidaria a comparacao com o resultado ja publicado.

**Por que a etapa 3 e codigo e nao LLM.** Contar acerto e operacao exata. Pedir a
um LLM que calcule metricas introduz erro onde nao precisa haver nenhum. LLM
entra so onde ha julgamento subjetivo — na etapa 4.

USO
---
    python -m App.agents.run_suite                          # V1, V1.0.1 e V1.0.2
    python -m App.agents.run_suite --versoes V1.0.1 V1.0.2  # so as duas novas
    python -m App.agents.run_suite --gerar-dataset          # forca gerar mensagens novas
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from App.agents.base import resumir_custo
from App.agents.evaluators import avaliar
from App.agents.personas import gerar_dataset
from App.core.config import get_settings
from App.versions.base import BasePipeline, PipelineResult
from App.versions.v1_0_1_orchestrator import V101OrchestratorPipeline
from App.versions.v1_0_2_hybrid import V102HybridPipeline
from App.versions.v1_classic import V1ClassicPipeline
from common.data_loader import load_eval_dataset

DIR_RELATORIOS = Path(__file__).resolve().parent.parent / "reports"

#: Fabricas das versoes. Sao funcoes (e nao instancias) para que cada rodada
#: construa a sua — os pipelines guardam contadores internos que nao podem
#: vazar de uma execucao para outra.
FABRICAS = {
    "V1": lambda cfg: V1ClassicPipeline(cfg),
    "V1.0.1": lambda cfg: V101OrchestratorPipeline(cfg),
    "V1.0.2": lambda cfg: V102HybridPipeline(cfg),
}


# ---------------------------------------------------------------- metricas
def calcular_metricas(linhas: List[PipelineResult], casos: List[dict]) -> Dict:
    """Agrega as metricas de uma versao sobre um conjunto de casos.

    `linhas[i]` corresponde a `casos[i]` — a ordem e preservada de proposito,
    para cruzar resultado com gabarito sem precisar de chave.
    """
    acertos_rota = acertos_tool = total_tool = 0

    for resultado, caso in zip(linhas, casos):
        if caso.get("expected_route") and resultado.route == caso["expected_route"]:
            acertos_rota += 1
        if caso.get("expected_tool"):
            total_tool += 1
            # Conta acerto SO se a ferramenta certa apareceu no top-k. Se o
            # roteador mandou para FAST_PATH, `tools` esta vazia e isso conta
            # como erro — DE PROPOSITO. Calcular Precision@K apenas sobre o que
            # o roteador acertou removeria da conta justamente os casos
            # dificeis e inflaria a metrica. Foi o ponto cego que encontramos
            # no harness da V1.
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
        "latencia_media_ms": statistics.fmean(latencias) if latencias else 0.0,
        "tokens_entrada": sum(r.prompt_tokens for r in linhas),
        "tokens_saida": sum(r.completion_tokens for r in linhas),
        "tokens_raciocinio": sum(r.reasoning_tokens for r in linhas),
        "chamadas_llm": sum(r.llm_calls for r in linhas),
        "mensagens_sem_llm": sum(1 for r in linhas if r.llm_calls == 0),
        "erros": sum(1 for r in linhas if r.error),
    }


def detectar_divergencias(
    linhas_a: List[PipelineResult],
    linhas_b: List[PipelineResult],
    casos: List[dict],
    nome_a: str,
    nome_b: str,
) -> List[dict]:
    """Casos em que duas versoes discordam.

    Sao os unicos que carregam informacao sobre a DIFERENCA entre elas: onde as
    duas acertam ou as duas erram, nao ha o que comparar.
    """
    divergencias = []
    for a, b, caso in zip(linhas_a, linhas_b, casos):
        if a.route == b.route and a.tools == b.tools:
            continue
        esperada = caso.get("expected_tool")
        a_ok = a.route == caso.get("expected_route") and (not esperada or esperada in a.tools)
        b_ok = b.route == caso.get("expected_route") and (not esperada or esperada in b.tools)
        divergencias.append({
            "par": f"{nome_a} x {nome_b}",
            "query": caso["query"],
            "persona": caso.get("persona"),
            "expected_route": caso.get("expected_route"),
            "expected_tool": esperada,
            f"{nome_a}_route": a.route, f"{nome_a}_tools": a.tools, f"{nome_a}_ok": a_ok,
            f"{nome_b}_route": b.route, f"{nome_b}_tools": b.tools, f"{nome_b}_ok": b_ok,
            # 1 = so B acertou; -1 = so A acertou; 0 = ambas ou nenhuma
            "vantagem_b": (1 if b_ok and not a_ok else -1 if a_ok and not b_ok else 0),
        })
    divergencias.sort(key=lambda d: abs(d["vantagem_b"]), reverse=True)
    return divergencias


# ------------------------------------------------------------------ execucao
def rodar_versao(pipeline: BasePipeline, casos: List[dict], k: int = 2) -> List[PipelineResult]:
    resultados = []
    for i, caso in enumerate(casos, 1):
        if i % 50 == 0:
            print(f"        {i}/{len(casos)}...")
        resultados.append(pipeline.process(caso["query"], k=k))
    return resultados


def obter_dataset_personas(
    n_por_persona: int, forcar_geracao: bool, settings
) -> Tuple[List[dict], Dict]:
    """Carrega o conjunto das personas, gerando so se ainda nao existir.

    O padrao e REAPROVEITAR. Gerar mensagens novas a cada rodada quebraria a
    comparacao com os resultados ja publicados das versoes anteriores.
    """
    caminho = DIR_RELATORIOS / "dataset_personas.json"
    if caminho.exists() and not forcar_geracao:
        bruto = json.loads(caminho.read_text(encoding="utf-8"))
        print(f"[1/4] Reaproveitando {len(bruto['queries'])} mensagens ja geradas")
        print("      (mesmo conjunto das rodadas anteriores — comparacao controlada)")
        return bruto["queries"], bruto.get("custo", {})

    print(f"[1/4] 5 personas gerando ~{n_por_persona} mensagens cada (em paralelo)...")
    inicio = time.perf_counter()
    gerado = gerar_dataset(n_por_persona=n_por_persona, settings=settings)
    custo = {
        "n_execucoes": len(gerado["execucoes"]),
        "n_sucesso": sum(1 for e in gerado["execucoes"] if e["ok"]),
        "custo_total_usd": sum(e["cost_usd"] or 0 for e in gerado["execucoes"]),
        "tempo_s": time.perf_counter() - inicio,
    }
    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps({"queries": gerado["queries"], "custo": custo,
                    "execucoes": gerado["execucoes"]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"      {len(gerado['queries'])} mensagens (US$ {custo['custo_total_usd']:.4f})")
    return gerado["queries"], custo


def executar_bateria(
    versoes: List[str],
    n_por_persona: int = 30,
    forcar_geracao: bool = False,
    k: int = 2,
) -> Dict:
    settings = get_settings()
    if not settings.agents_enabled:
        raise SystemExit("A bateria precisa de OPENAI_API_KEY em App/.env.")

    casos_personas, custo_personas = obter_dataset_personas(n_por_persona, forcar_geracao, settings)
    casos_oficiais = load_eval_dataset()

    print(f"\n[2/4] Preparando {len(versoes)} versoes (mesma configuracao)...")
    pipelines: Dict[str, BasePipeline] = {}
    for nome in versoes:
        if nome not in FABRICAS:
            raise SystemExit(f"Versao desconhecida: {nome}. Opcoes: {list(FABRICAS)}")
        pipelines[nome] = FABRICAS[nome](settings).setup()
        print(f"      {nome:8s} {pipelines[nome].description}")

    comparacao: Dict = {
        "versoes": versoes,
        "observacoes": [],
        "config": {
            "orchestrator_model": settings.orchestrator_model,
            "reasoning_effort": settings.orchestrator_reasoning_effort,
            "embedding_model": settings.embedding_model,
            "agents_model": settings.agents_model,
            "k": k,
        },
    }

    for chave, titulo, casos in [
        ("oficial", "dataset oficial (rotulos humanos)", casos_oficiais),
        ("personas", "conjunto das personas (rotulos de LLM)", casos_personas),
    ]:
        if not casos:
            continue
        print(f"\n      Rodando {titulo}: {len(casos)} mensagens")
        linhas: Dict[str, List[PipelineResult]] = {}
        for nome, pipeline in pipelines.items():
            print(f"      {nome}...")
            linhas[nome] = rodar_versao(pipeline, casos, k)

        bloco = {
            "n_casos": len(casos),
            "metricas": {n: calcular_metricas(l, casos) for n, l in linhas.items()},
            "linhas": {n: [r.as_dict() for r in l] for n, l in linhas.items()},
        }
        if chave == "personas":
            personas_unicas = sorted({c.get("persona", "sem_persona") for c in casos})
            bloco["por_persona"] = {
                persona: {
                    nome: calcular_metricas(
                        [l[i] for i, c in enumerate(casos) if c.get("persona") == persona],
                        [c for c in casos if c.get("persona") == persona])
                    for nome, l in linhas.items()
                }
                for persona in personas_unicas
            }
        comparacao[chave] = bloco

        # Divergencias par a par entre todas as versoes.
        for a, b in combinations(versoes, 2):
            comparacao.setdefault("divergencias", []).extend(
                detectar_divergencias(linhas[a], linhas[b], casos, a, b)
            )

    # Telemetria especifica de cada versao.
    telemetria: Dict[str, Dict] = {}
    for nome, pipeline in pipelines.items():
        if hasattr(pipeline, "telemetria"):
            telemetria[nome] = pipeline.telemetria()
        elif hasattr(pipeline, "fallbacks"):
            telemetria[nome] = {"fallbacks": pipeline.fallbacks}
    comparacao["telemetria"] = telemetria

    for nome, t in telemetria.items():
        if t.get("fallbacks") or t.get("fallbacks_do_orquestrador"):
            n = t.get("fallbacks") or t.get("fallbacks_do_orquestrador")
            comparacao["observacoes"].append(
                f"{nome} caiu no plano B (decisao local) em {n} mensagens. Nessas ela "
                "respondeu como a V1, o que reduz a diferenca medida."
            )
        if "taxa_desvio" in t:
            comparacao["observacoes"].append(
                f"{nome} desviou {t['taxa_desvio']:.0%} das mensagens sem chamar LLM. "
                "Essa taxa e proporcional a fracao de FAST_PATH no trafego — o conjunto "
                "das personas e pesado em AGENT, entao SUBESTIMA o ganho em producao."
            )
    comparacao["observacoes"].append(
        "40% do dataset oficial tem sobreposicao alta com os 53 exemplos de treino "
        "da V1 (3 sao copias literais), o que favorece a V1 nesse conjunto."
    )
    comparacao["custo_geracao_personas"] = custo_personas

    print(f"\n[3/4] Metricas calculadas. Divergencias: {len(comparacao.get('divergencias', []))}")
    print("\n[4/4] 2 avaliadores especialistas analisando (em paralelo)...")
    execucoes = avaliar(comparacao, settings=settings)
    comparacao["avaliacoes"] = [e.as_dict() for e in execucoes]
    comparacao["custo_avaliadores"] = resumir_custo(execucoes)
    return comparacao


def imprimir_resumo(c: Dict) -> None:
    versoes = c["versoes"]
    print("\n" + "=" * 100)
    print("BATERIA DE TESTES — " + "  x  ".join(versoes))
    print("=" * 100)

    for chave, titulo in [("oficial", "DATASET OFICIAL (rotulos humanos)"),
                          ("personas", "CONJUNTO DAS PERSONAS (rotulos de LLM)")]:
        bloco = c.get(chave)
        if not bloco:
            continue
        print(f"\n{titulo} — {bloco['n_casos']} mensagens")
        print(f"  {'versao':9s} {'rota':>7s} {'P@2':>7s} {'$/msg':>11s} "
              f"{'p50':>8s} {'p95':>8s} {'media':>8s} {'s/ LLM':>8s}")
        for v in versoes:
            m = bloco["metricas"][v]
            p2 = m["precision_at_k"] or 0
            print(f"  {v:9s} {m['acuracia_rota']:7.1%} {p2:7.1%} "
                  f"{m['custo_por_mensagem_usd']:11.6f} {m['latencia_p50_ms']:8.0f} "
                  f"{m['latencia_p95_ms']:8.0f} {m['latencia_media_ms']:8.0f} "
                  f"{m['mensagens_sem_llm']:8d}")

    por_persona = c.get("personas", {}).get("por_persona")
    if por_persona:
        print(f"\n  Por persona (rota / P@2):")
        for persona, dados in por_persona.items():
            partes = [f"{v}: {dados[v]['acuracia_rota']:.0%}/"
                      f"{(dados[v]['precision_at_k'] or 0):.0%}" for v in versoes]
            print(f"    {persona:24s} " + "   ".join(partes))

    div = c.get("divergencias", [])
    if div:
        print("\n  Divergencias par a par:")
        pares = sorted({d["par"] for d in div})
        for par in pares:
            do_par = [d for d in div if d["par"] == par]
            a, b = par.split(" x ")
            print(f"    {par:22s} {len(do_par):3d} divergencias | "
                  f"so {b} acertou: {sum(1 for d in do_par if d['vantagem_b'] == 1):3d} | "
                  f"so {a} acertou: {sum(1 for d in do_par if d['vantagem_b'] == -1):3d}")

    tel = c.get("telemetria", {})
    if tel:
        print("\n  Telemetria:")
        for v, t in tel.items():
            if "taxa_desvio" in t:
                print(f"    {v:9s} desvios={t['desvios']} ({t['taxa_desvio']:.0%}) | "
                      f"chamadas LLM={t['chamadas_llm']} | motivos={t['motivos_de_nao_desvio']}")
            elif t.get("fallbacks"):
                print(f"    {v:9s} fallbacks={t['fallbacks']}")

    print("\n  PARECER DOS AVALIADORES:")
    for a in c.get("avaliacoes", []):
        if not a["ok"]:
            print(f"    {a['agent_name']}: FALHOU ({a['error']})")
            continue
        p = a["payload"]
        print(f"    {a['agent_name']:22s} -> recomenda {str(p.get('recomendacao_final','?')):12s} "
              f"(confianca {p.get('confianca', 0):.0%})")
    for o in c.get("observacoes", []):
        print(f"\n  [!] {o}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bateria de testes com subagentes.")
    parser.add_argument("--versoes", nargs="*", default=["V1", "V1.0.1", "V1.0.2"])
    parser.add_argument("--n-por-persona", type=int, default=30)
    parser.add_argument("--gerar-dataset", action="store_true",
                        help="forca gerar mensagens novas (quebra a comparacao com rodadas anteriores)")
    parser.add_argument("-k", type=int, default=2)
    parser.add_argument("--out", default=str(DIR_RELATORIOS / "bateria_3_versoes.json"))
    args = parser.parse_args()

    dossie = executar_bateria(args.versoes, args.n_por_persona, args.gerar_dataset, args.k)
    imprimir_resumo(dossie)

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dossie, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatorio completo: {destino}")


if __name__ == "__main__":
    main()
