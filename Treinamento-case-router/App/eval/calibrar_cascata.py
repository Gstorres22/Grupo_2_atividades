"""Calibra o limiar de desvio da cascata (V1.0.2) SEM tocar nos conjuntos de teste.

===============================================================================
O PROBLEMA QUE ESTE ARQUIVO RESOLVE
===============================================================================

A V1.0.2 pula a chamada de LLM quando o classificador local esta seguro. "Seguro"
precisa de um numero: qual e o limiar?

A tentacao e olhar o conjunto de teste, achar o limiar que da o melhor resultado
e usar esse. Isso e ajustar no teste (leakage): o numero reportado depois fica
otimista, porque o limiar foi escolhido justamente para funcionar naquelas
mensagens.

Aqui o limiar sai APENAS dos 53 exemplos de treino, por validacao cruzada. Os
conjuntos de teste (oficial e personas) so aparecem depois, para VALIDAR — nunca
para escolher.

===============================================================================
POR QUE DOIS SINAIS, E NAO SO A CONFIANCA DO MODELO
===============================================================================

O diagnostico mostrou que a probabilidade da regressao logistica e mal calibrada
neste problema. Medido nas 150 mensagens de persona:

    confianca [0,60-0,70) -> 92% de precisao
    confianca [0,70-0,80) -> 71% de precisao   <- MAIOR confianca, MENOR precisao
    confianca [0,80-0,90) -> 100% de precisao

Nao e monotonica. Com 53 exemplos de treino, a probabilidade que o modelo
reporta nao ordena bem os casos.

Por isso a cascata usa DOIS sinais que precisam concordar:

  1. CONFIANCA  - o quanto o modelo acredita na propria resposta
  2. FAMILIARIDADE - o quanto a mensagem se parece com algo do treino

O segundo e uma forma simples de deteccao de "fora da distribuicao": se a
mensagem nao se parece com nada que o modelo ja viu, ele nao tem base para
opinar, por mais confiante que soe. Foi exatamente assim que a V1 falhou com
linguagem formal — um registro ausente dos 53 exemplos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from common.data_loader import load_router_training_data

DIR_RELATORIOS = Path(__file__).resolve().parent.parent / "reports"


@dataclass
class Calibracao:
    """O resultado da calibracao: os dois limiares e como se chegou neles."""

    limiar_confianca: float
    limiar_familiaridade: float
    cobertura_esperada: float
    """Fracao das mensagens FAST_PATH que serao desviadas (nao chamam LLM)."""
    precisao_esperada: float
    """Das desviadas, quantas estao corretas. Queremos 100%: um erro aqui
    significa cliente sem atendimento, e o LLM nem foi consultado."""
    n_treino: int
    percentil_familiaridade: float
    familiaridade_interna: Dict[str, float]
    """Estatisticas da similaridade dos exemplos de treino ENTRE SI. E a regua
    contra a qual julgamos se uma mensagem nova e realmente familiar."""
    curva: List[Dict]
    """A varredura completa, para o relatorio mostrar como a escolha foi feita."""

    def as_dict(self) -> Dict:
        return {
            "limiar_confianca": self.limiar_confianca,
            "limiar_familiaridade": self.limiar_familiaridade,
            "cobertura_esperada": self.cobertura_esperada,
            "precisao_esperada": self.precisao_esperada,
            "n_treino": self.n_treino,
            "percentil_familiaridade": self.percentil_familiaridade,
            "familiaridade_interna_do_treino": self.familiaridade_interna,
            "curva": self.curva,
            "metodo": (
                "CONFIANCA: validacao cruzada 5-fold estratificada sobre os 53 exemplos "
                "de treino. FAMILIARIDADE: percentil "
                f"{self.percentil_familiaridade:.0f} da similaridade dos exemplos de "
                "treino entre si — criterio relativo, que exige que a mensagem nova "
                "se pareca com o treino mais do que o treino se parece consigo mesmo. "
                "Nenhum conjunto de teste foi consultado."
            ),
        }


def _pipeline() -> Pipeline:
    """O MESMO classificador da V1 — precisa ser identico, senao calibramos
    um modelo e usamos outro."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  sublinear_tf=True, lowercase=True, min_df=1)),
        ("clf", LogisticRegression(C=5.0, max_iter=1000,
                                   class_weight="balanced", random_state=42)),
    ])


def _familiaridade(textos_alvo: List[str], textos_referencia: List[str]) -> np.ndarray:
    """Maior similaridade de cada texto-alvo com QUALQUER texto de referencia.

    Usa os mesmos n-gramas de caractere do classificador, por coerencia: se o
    modelo enxerga o texto assim, a nocao de "parecido" deve ser a mesma.
    """
    if not textos_referencia:
        return np.zeros(len(textos_alvo))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    ref = normalize(vec.fit_transform(textos_referencia))
    alvo = normalize(vec.transform(textos_alvo))
    return (alvo @ ref.T).toarray().max(axis=1)


def _familiaridade_interna(textos: List[str]) -> np.ndarray:
    """Para cada exemplo de treino, sua similaridade com o exemplo MAIS PARECIDO
    entre os outros.

    Essa distribuicao responde: "quao parecidos os exemplos de treino sao entre
    si?". Ela e a regua contra a qual julgamos se uma mensagem nova e realmente
    familiar. A diagonal e zerada para um exemplo nao se comparar consigo mesmo.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    matriz = normalize(vec.fit_transform(textos))
    sim = (matriz @ matriz.T).toarray()
    np.fill_diagonal(sim, 0.0)
    return sim.max(axis=1)


def calibrar(
    n_folds: int = 5,
    precisao_minima: float = 1.0,
    percentil_familiaridade: float = 90.0,
) -> Calibracao:
    """Encontra os limiares usando somente os dados de treino.

    Como a validacao cruzada funciona aqui: os 53 exemplos sao divididos em 5
    partes. Para cada parte, treinamos nas outras 4 e prevemos nela. Assim cada
    exemplo recebe uma previsao feita por um modelo que NUNCA o viu — que e o
    que simula o comportamento com mensagem nova.

    A familiaridade tambem e calculada fora da dobra: comparamos o exemplo
    apenas com os exemplos de TREINO daquela dobra. Se comparassemos com todos,
    cada exemplo teria similaridade 1,0 consigo mesmo e o sinal seria inutil.
    """
    textos, rotulos = load_router_training_data()
    textos = list(textos)
    rotulos = np.array(rotulos)

    confiancas = np.zeros(len(textos))
    previsoes = np.empty(len(textos), dtype=object)
    familiaridades = np.zeros(len(textos))

    cv = StratifiedKFold(n_folds, shuffle=True, random_state=42)
    for idx_treino, idx_teste in cv.split(textos, rotulos):
        modelo = _pipeline().fit([textos[i] for i in idx_treino], rotulos[idx_treino])
        probs = modelo.predict_proba([textos[i] for i in idx_teste])
        classes = list(modelo.classes_)
        for pos, i in enumerate(idx_teste):
            melhor = int(probs[pos].argmax())
            previsoes[i] = classes[melhor]
            confiancas[i] = probs[pos][melhor]
        familiaridades[idx_teste] = _familiaridade(
            [textos[i] for i in idx_teste], [textos[i] for i in idx_treino]
        )

    # -----------------------------------------------------------------
    # O limiar de FAMILIARIDADE nao pode sair da validacao cruzada.
    #
    # Motivo, medido: n-grama de caractere mede sobreposicao de LETRAS, nao de
    # registro linguistico. Duas frases em portugues compartilham muitos
    # trigramas mesmo com vocabulario totalmente diferente. Resultado pratico:
    #
    #     "solicito a emissao do informe de rendimentos" -> familiaridade 0,347
    #     mediana da familiaridade INTERNA do treino     -> 0,336
    #
    # Ou seja, uma frase de registro formal — exatamente o que derruba a V1 —
    # parece MAIS familiar que metade dos proprios exemplos de treino. Um limiar
    # calibrado por validacao cruzada aceitaria essa mensagem, e o LLM nunca
    # seria consultado para corrigir.
    #
    # O criterio correto e RELATIVO a estrutura interna do treino:
    #
    #     "a mensagem precisa se parecer com algum exemplo de treino MAIS do que
    #      os exemplos de treino se parecem entre si"
    #
    # Implementado como um percentil alto da distribuicao de familiaridade
    # interna. Sai 100% dos dados de treino, sem consultar teste nenhum, e
    # sobrevive a mudanca de registro — que a validacao cruzada, por construcao,
    # nao consegue enxergar (as dobras vem todas da mesma distribuicao estreita).
    interna = _familiaridade_interna(textos)
    limiar_familiaridade = float(np.percentile(interna, percentil_familiaridade))

    # Varredura: para cada par de limiares, quantas mensagens FAST_PATH seriam
    # desviadas e qual a precisao dessas.
    eh_fast = np.array([p == "FAST_PATH" for p in previsoes])
    correto = np.array([p == r for p, r in zip(previsoes, rotulos)])

    # Com a familiaridade fixada pelo criterio relativo, varremos apenas a
    # confianca. Um parametro a menos para ajustar e um risco a menos de
    # sobreajuste.
    curva: List[Dict] = []
    for lc in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        desviadas = eh_fast & (confiancas >= lc) & (familiaridades >= limiar_familiaridade)
        n = int(desviadas.sum())
        acertos = int((desviadas & correto).sum()) if n else 0
        curva.append({
            "limiar_confianca": lc,
            "limiar_familiaridade": limiar_familiaridade,
            "n_desviadas": n,
            "cobertura_fast_path": n / max(1, int(eh_fast.sum())),
            "precisao": (acertos / n) if n else 1.0,
        })

    # Escolha: entre os limiares que atingem a precisao minima, o de MAIOR
    # cobertura. A ordem importa — precisao primeiro, cobertura depois. Desviar
    # uma mensagem errada deixa o cliente sem atendimento, e o LLM nem e
    # consultado para corrigir. Ja perder cobertura so gasta uma chamada a mais.
    candidatos = [c for c in curva if c["n_desviadas"] > 0]
    seguros = [c for c in candidatos if c["precisao"] >= precisao_minima]
    escolhido = (
        min(seguros, key=lambda c: c["limiar_confianca"]) if seguros
        else max(candidatos, key=lambda c: (c["precisao"], c["cobertura_fast_path"]))
        if candidatos else curva[0]
    )

    return Calibracao(
        limiar_confianca=escolhido["limiar_confianca"],
        limiar_familiaridade=limiar_familiaridade,
        cobertura_esperada=escolhido["cobertura_fast_path"],
        precisao_esperada=escolhido["precisao"],
        n_treino=len(textos),
        percentil_familiaridade=percentil_familiaridade,
        familiaridade_interna={
            "p25": float(np.percentile(interna, 25)),
            "p50": float(np.percentile(interna, 50)),
            "p75": float(np.percentile(interna, 75)),
            "p90": float(np.percentile(interna, 90)),
            "max": float(interna.max()),
        },
        curva=curva,
    )


def main() -> None:
    cal = calibrar()
    print("=" * 84)
    print("CALIBRACAO DA CASCATA (V1.0.2) — somente dados de treino")
    print("=" * 84)
    print(f"Exemplos de treino: {cal.n_treino}\n")
    print("SIMILARIDADE DOS EXEMPLOS DE TREINO ENTRE SI (a regua):")
    for chave, valor in cal.familiaridade_interna.items():
        print(f"   {chave:4s} {valor:.3f}")
    print(f"\n=> limiar de familiaridade = percentil {cal.percentil_familiaridade:.0f} "
          f"= {cal.limiar_familiaridade:.3f}")
    print("   Criterio: a mensagem nova precisa se parecer com algum exemplo de treino")
    print("   MAIS do que os exemplos de treino se parecem entre si.\n")

    print("VARREDURA DA CONFIANCA (com a familiaridade ja fixada):")
    print(f"{'conf>=':>8s} {'desviadas':>10s} {'cobertura':>10s} {'precisao':>9s}")
    for c in cal.curva:
        marca = "  <== escolhido" if c["limiar_confianca"] == cal.limiar_confianca else ""
        print(f"{c['limiar_confianca']:8.2f} {c['n_desviadas']:10d} "
              f"{c['cobertura_fast_path']:10.0%} {c['precisao']:9.0%}{marca}")

    print(f"\nLIMIARES ESCOLHIDOS: confianca >= {cal.limiar_confianca:.2f} "
          f"E familiaridade >= {cal.limiar_familiaridade:.3f}")
    print(f"  cobertura esperada: {cal.cobertura_esperada:.0%} das mensagens FAST_PATH")
    print(f"  precisao esperada : {cal.precisao_esperada:.0%}")
    print("\nPrecisao vem antes de cobertura: desviar uma mensagem errada deixa o")
    print("cliente sem atendimento, e o LLM nem e consultado para corrigir.")

    destino = DIR_RELATORIOS / "calibracao_cascata.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(cal.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {destino}")


if __name__ == "__main__":
    main()
