"""Estagio VETORIAL da busca hibrida (embeddings da OpenAI, com cache em disco).

===============================================================================
POR QUE BUSCA HIBRIDA (VETORIAL + LEXICAL) E NAO SO UMA DAS DUAS
===============================================================================

As duas familias erram de formas complementares, e o catalogo deste case exibe
os dois tipos de erro:

  LEXICAL (TF-IDF) acerta o termo exato e erra o sentido.
      "Preciso mudei de casa, como mudo o CEP?"  ->  alterar_endereco
      A palavra "endereco" nao aparece na query. Nenhum peso de TF-IDF
      aproxima "CEP"/"casa" de "endereco": nao ha caractere em comum.
      Medido: a tool correta ficou na posicao 90 de 285.

  VETORIAL (embeddings) acerta o sentido e escorrega no termo exato.
      Embeddings aproximam "consultar_saldo" de TODAS as variantes de saldo,
      inclusive `consultar_saldo_poupanca` e `consultar_saldo_pj`. Para
      separar "poupanca" de "corrente" o sinal lexical e mais confiavel.

Medicao que motivou incluir o vetorial: com busca lexica pura, o Recall@15 fica
em 0,85-0,90. As 4 falhas sao TODAS semanticas — casos sem sobreposicao de
palavras. E recall e o TETO do sistema: nenhuma reordenacao recupera uma tool
que nunca entrou na lista de candidatas.

COMO FUNDIMOS: os tres sinais (caractere, palavra, vetorial) entram no mesmo
Reciprocal Rank Fusion em `candidate_starter/retrieval.py`. O RRF combina
POSICOES no ranking, o que resolve o problema classico da busca hibrida — que
similaridade de cosseno de embedding e score de TF-IDF vivem em escalas
diferentes e nao comparaveis. Sem RRF seria preciso normalizar os dois, e a
normalizacao fica refem de outliers.

===============================================================================
POR QUE CACHE EM DISCO
===============================================================================

O catalogo tem 285 tools e nunca muda entre execucoes. Sem cache, cada rodada
do harness pagaria 285 embeddings de novo — nao pelo custo (que e irrisorio,
~US$ 0,0002), mas pela LATENCIA: seriam segundos de rede em um pipeline que se
propoe a medir milissegundos. Com cache, o indice sobe instantaneo a partir da
segunda execucao, e a unica chamada de rede por query e o embedding da query.

O cache e enderecado por conteudo: a chave e o hash do texto + o nome do modelo.
Trocar de modelo de embedding invalida as entradas antigas automaticamente, sem
precisar limpar nada na mao.

"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from App.core.config import CACHE_DIR, Settings


class OpenAIEmbeddingProvider:
    """Callable compativel com o gancho `ToolRetriever.set_dense_provider()`.

    Recebe uma lista de textos e devolve a matriz (n_textos, n_dimensoes).
    Toda a integracao com a OpenAI fica aqui — o nucleo do case nao importa
    `openai` em lugar nenhum.
    """

    def __init__(self, settings: Settings, cache_dir: Path = CACHE_DIR, batch_size: int = 128):
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAIEmbeddingProvider exige OPENAI_API_KEY. "
                "Sem chave, use o modo local (nao registre um dense provider)."
            )
        from openai import OpenAI  # import tardio: so quando o modo cascata e usado

        self.settings = settings
        self.model = settings.embedding_model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=settings.openai_api_key)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_dir / f"embeddings_{self.model.replace('/', '_')}.npz"
        self._cache: Dict[str, np.ndarray] = self._load_cache()
        self.api_calls = 0          # telemetria: quantas chamadas de rede foram feitas
        self.cache_hits = 0

    # -- Cache ----------------------------------------------------------------
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()[:32]

    def _load_cache(self) -> Dict[str, np.ndarray]:
        if not self._cache_path.exists():
            return {}
        try:
            with np.load(self._cache_path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        except (OSError, ValueError):
            # Cache corrompido nao pode derrubar o pipeline: recomeca vazio.
            return {}

    def flush(self) -> None:
        """Persiste o cache. Chamar apos indexar o catalogo."""
        if self._cache:
            np.savez_compressed(self._cache_path, **self._cache)

    # -- Geracao --------------------------------------------------------------
    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        missing = [t for t in texts if self._key(t) not in self._cache]
        self.cache_hits += len(texts) - len(missing)

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            self.api_calls += 1
            for text, item in zip(batch, response.data):
                self._cache[self._key(text)] = np.asarray(item.embedding, dtype=np.float32)

        if missing:
            self.flush()
        return np.vstack([self._cache[self._key(t)] for t in texts])

    def stats(self) -> Dict[str, int]:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cached_vectors": len(self._cache),
        }


def build_dense_provider(settings: Settings) -> "OpenAIEmbeddingProvider | None":
    """Cria o provider se houver chave; devolve None para o modo local.

    Este e o unico ponto de decisao entre "hibrido com vetorial" e "so lexico".
    Quem chama nao precisa saber se ha chave — passa o resultado direto para
    `retriever.set_dense_provider(...)`, que aceita None.
    """
    if not settings.llm_enabled:
        return None
    try:
        return OpenAIEmbeddingProvider(settings)
    except Exception as error:  # rede, credencial invalida, pacote ausente
        print(f"[embeddings] Modo vetorial indisponivel ({error}). Seguindo em modo lexico.")
        return None
