"""V1.0.1 — Agente orquestrador por LLM.

===============================================================================
O QUE MUDA EM RELACAO A V1
===============================================================================

    V1     : classificador treinado decide a rota  +  similaridade de texto escolhe a ferramenta
    V1.0.1 : um LLM decide a rota  E  escolhe a ferramenta, na mesma chamada

A aposta e trocar custo por acerto. A V1 e praticamente gratuita mas quebra com
linguagem formal, erro de digitacao e negacao. Um LLM entende linguagem de
verdade — deveria resolver os tres.

===============================================================================
DECISAO DE ARQUITETURA 1 — UMA chamada de LLM, nao duas
===============================================================================

O caminho obvio seria fazer duas chamadas:

    chamada 1: "esta mensagem e FAST_PATH ou AGENT?"
    chamada 2: "entre estas 20 ferramentas, quais 2 servem?"

Rejeitamos esse desenho. Fazemos UMA chamada que responde as duas perguntas.

**Por que.** O custo de uma chamada de LLM tem duas partes muito diferentes:

  * TOKENS  -> dominam o preco em dolares
  * IDA E VOLTA DE REDE -> domina a LATENCIA (tipicamente 200-800 ms, e esse
    tempo existe mesmo que a resposta tenha 5 tokens)

Duas chamadas pagam DOIS tempos de rede. Como a decisao de rota e a escolha da
ferramenta dependem da mesma leitura da mensagem, separa-las gasta o dobro da
latencia sem ganhar informacao.

**Como isso e possivel.** A busca local de candidatas custa ~1,7 ms e zero
dolar. Entao rodamos ela SEMPRE, antes de saber a rota, e mandamos as candidatas
junto no mesmo prompt. Se o LLM responder FAST_PATH, jogamos as candidatas fora
e nao perdemos nada — desperdicamos 1,7 ms de CPU local, o que e irrelevante.

===============================================================================
DECISAO DE ARQUITETURA 2 — o LLM escolhe entre 20 candidatas, nao entre 285
===============================================================================

Mandar as 285 ferramentas no prompt e exatamente o que o case pede para evitar:
estoura o contexto, confunde o modelo, multiplica custo e latencia.

Mantemos o estagio de recuperacao local (o mesmo da V1, com embeddings) para
reduzir 285 -> 20, e so entao o LLM entra. Isso preserva a tese central do
projeto: o componente barato faz a triagem, o componente caro faz o julgamento
fino.

**Consequencia importante e limitante:** se a ferramenta correta nao estiver
entre as 20 candidatas, NENHUMA inteligencia do LLM a recupera. O `Recall@20`
do estagio local e o TETO da V1.0.1 tambem. Medimos isso no relatorio.

Usamos 20 (e nao as 15 da V1) porque aqui a lista vai para um LLM que le as
descricoes — algumas candidatas a mais custam poucos tokens e sobem o teto.

===============================================================================
DECISAO DE ARQUITETURA 3 — o prompt carrega as duas armadilhas conhecidas
===============================================================================

Os testes da V1 revelaram dois modos de falha. O prompt trata os dois
explicitamente, porque um LLM generico cairia nos mesmos:

  1. NEGACAO — "bloquear" esta contido em "desbloquear". A V1 devolvia
     `desbloquear_cartao` para quem pediu bloqueio. E o bug mais grave da V1.
  2. QUASE-DUPLICATAS — o catalogo tem variantes hiper-especificas cujo nome
     quase copia a frase do cliente. Sem instrucao, o LLM escolhe a parafrase
     mais literal em vez da ferramenta canonica, igual a similaridade de texto.

===============================================================================
DECISAO DE ARQUITETURA 4 — falha segura, mas CONTABILIZADA
===============================================================================

Se a rede cair ou o LLM devolver algo invalido, caimos de volta na decisao
local (a propria V1). Isso e obrigatorio em producao: o atendimento nao pode
parar porque a OpenAI teve um soluco.

Mas isso cria um risco de medicao: se a V1.0.1 silenciosamente usasse respostas
da V1, estariamos comparando a V1 com ela mesma. Por isso todo acionamento do
plano B e registrado no `trace` e contado em `fallbacks`. O relatorio mostra
quantas mensagens foram, de fato, respondidas pela V1 dentro da V1.0.1.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from App.core.config import Settings, get_settings
from App.core.embeddings import build_dense_provider
from App.versions.base import BasePipeline, PipelineResult
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data, load_tools
from common.mock_llm import COST_AGENT_LLM_CALL_USD, COST_RETRIEVAL_USD, COST_ROUTER_USD

# =============================================================================
# TABELA DE PRECOS
# =============================================================================
# Precos em dolares por 1 MILHAO de tokens: (entrada, entrada_cacheada, saida).
#
# Fonte: pesquisa em developers.openai.com/api/docs/pricing, verificada em
# 14/08/2026 (ver o relatorio completo em App/reports/pesquisa_modelos_openai.md).
#
# Por que tres valores e nao dois: a "entrada cacheada" e cobrada quando o
# prefixo do prompt se repete entre chamadas. O desconto varia MUITO entre
# modelos — 90% no gpt-5.6-luna contra 50% no gpt-4o-mini — e isso inverte o
# ranking de custo. Ignorar essa coluna levaria a escolher o modelo errado.
#
# ATENCAO: modelo ausente desta tabela devolve custo `None`, e o relatorio
# marca "custo desconhecido" em vez de assumir zero. Nunca inventamos preco:
# um custo zerado por engano faria o modelo parecer gratuito na comparacao.
PRECOS_POR_MILHAO_TOKENS: Dict[str, Tuple[float, float, float]] = {
    # --- Familia 5.6: luna=nano, terra=mini, sol=completo ---
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    # --- Geracao anterior ---
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),   # desligamento em 11/12/2026
    "gpt-5-nano": (0.05, 0.005, 0.40),   # desligamento em 11/12/2026
    "gpt-5": (1.25, 0.125, 10.00),
    # --- GPT-4.x ---
    "gpt-4.1-nano": (0.10, 0.025, 0.40),  # desligamento em 23/10/2026
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    # --- Serie o ---
    "o3": (2.00, 0.50, 8.00),
    "o1": (15.00, 7.50, 60.00),
}

#: Modelos com data de desligamento anunciada. Usar em producao e assumir
#: divida tecnica com prazo. O comparador avisa quando um deles e escolhido.
MODELOS_DESCONTINUADOS: Dict[str, str] = {
    "gpt-4.1-nano": "23/10/2026",
    "gpt-5-nano": "11/12/2026",
    "gpt-5-mini": "11/12/2026",
}


def _buscar_precos(modelo: str) -> Optional[Tuple[float, float, float]]:
    """Encontra a linha de precos de um modelo.

    O nome pode vir com data ("gpt-5.6-luna-2026-07-09"), entao tentamos o nome
    exato e depois o prefixo mais LONGO que casar. O prefixo mais longo importa:
    "gpt-5.4-nano" e "gpt-5.4" ambos casam com "gpt-5.4-nano-2026-03-17", e o
    correto e o primeiro.
    """
    if modelo in PRECOS_POR_MILHAO_TOKENS:
        return PRECOS_POR_MILHAO_TOKENS[modelo]
    candidatos = [k for k in PRECOS_POR_MILHAO_TOKENS if modelo.startswith(k)]
    if not candidatos:
        return None
    return PRECOS_POR_MILHAO_TOKENS[max(candidatos, key=len)]


def calcular_custo(
    modelo: str, tokens_entrada: int, tokens_saida: int, tokens_cacheados: int = 0
) -> Optional[float]:
    """Converte tokens em dolares. Devolve None se o preco for desconhecido.

    Args:
        tokens_entrada: total de tokens de entrada reportado pela API.
        tokens_saida: tokens gerados. IMPORTANTE: nos modelos de raciocinio
            isso JA INCLUI os tokens de raciocinio invisiveis, que sao cobrados
            como saida. Por isso nao somamos nada aqui — a API ja soma.
        tokens_cacheados: subconjunto de `tokens_entrada` que veio do cache e
            e cobrado mais barato. Vem de `usage.prompt_tokens_details`.
    """
    precos = _buscar_precos(modelo)
    if precos is None:
        return None
    preco_entrada, preco_cache, preco_saida = precos
    # Os tokens cacheados fazem parte do total de entrada; separamos os dois
    # grupos para nao cobrar duas vezes pelo mesmo token.
    nao_cacheados = max(0, tokens_entrada - tokens_cacheados)
    return (
        nao_cacheados / 1_000_000 * preco_entrada
        + tokens_cacheados / 1_000_000 * preco_cache
        + tokens_saida / 1_000_000 * preco_saida
    )


def modelo_usa_raciocinio(modelo: str) -> bool:
    """O modelo aceita o parametro `reasoning_effort`?

    Precisamos saber porque enviar esse parametro para um modelo que nao o
    suporta (como o gpt-4o-mini) causa erro na API. E NAO enviar para um modelo
    que o suporta deixa o default de fabrica valendo — que na familia 5.6 e
    `medium`, gastando tokens de raciocinio invisiveis numa tarefa que nao
    precisa deles. Medimos: 23 tokens de raciocinio por chamada no default.
    """
    return modelo.startswith("gpt-5") or (
        len(modelo) > 1 and modelo[0] == "o" and modelo[1].isdigit()
    )


# =============================================================================
# O PROMPT
# =============================================================================
# Escrito como constante de modulo (e nao dentro da funcao) por tres motivos:
# ficar visivel para revisao, poder ser testado isoladamente, e deixar claro que
# e FIXO — o que habilita o cache de prompt do provedor, que cobra mais barato
# pela parte repetida entre chamadas.
PROMPT_SISTEMA = """Voce e o cerebro de roteamento do atendimento de um banco digital.

Para cada mensagem do cliente voce decide DUAS coisas de uma vez:

## 1. A ROTA

- "FAST_PATH": saudacao, agradecimento, despedida, ou pergunta GENERICA sobre a
  instituicao que NAO depende dos dados daquele cliente (horario de atendimento,
  taxas de tabela, documentos necessarios, como funciona um produto).

- "AGENT": qualquer pedido que dependa dos DADOS ou de uma ACAO na conta daquele
  cliente especifico (saldo, fatura, bloquear cartao, atualizar cadastro,
  investimentos), OU qualquer pedido que precise executar uma ferramenta.

REGRA DE DESEMPATE: se a mensagem tem saudacao E um pedido acionavel
("Bom dia, preciso do meu saldo"), a rota e AGENT. A saudacao e so cortesia e
NAO deve dominar a decisao.

Cliente escrevendo formalmente ("solicito a emissao do informe de rendimentos")
ou com erros de digitacao ("qero blqouear meu cartaao") continua sendo AGENT se
o pedido for acionavel. Nem formalidade nem erro de digitacao mudam a intencao.

## 2. AS FERRAMENTAS (apenas quando a rota for AGENT)

Voce recebe uma lista de ferramentas candidatas. Escolha as melhores, em ordem.

REGRA A — NEGACAO. Preste atencao maxima a prefixos de negacao. "bloquear" e
"desbloquear" sao acoes OPOSTAS, assim como "ativar"/"desativar",
"cancelar"/"contratar", "autorizar"/"revogar". Nomes parecidos podem significar
o contrario do que o cliente pediu. Leia a descricao, nao so o nome.

REGRA B — PREFIRA A FERRAMENTA CANONICA. O catalogo tem muitas variantes quase
identicas. Escolha a ferramenta mais GERAL e direta que resolve a intencao.
Variantes hiper-especificas, cujo nome praticamente repete a frase do cliente,
costumam ser refinamentos e NAO a porta de entrada correta.
Exemplo: para "manda o pdf da minha fatura", a escolha certa e a ferramenta
geral de consultar fatura, e nao uma especializada em enviar PDF.

REGRA C — So escolha uma variante especifica se o cliente mencionou
EXPLICITAMENTE aquela condicao (poupanca, PJ, consignado, internacional,
pre-pago, corporativo).

REGRA D — Use EXATAMENTE os nomes da lista de candidatas. Nunca invente nome.

## FORMATO DA RESPOSTA

Responda SOMENTE com JSON, sem texto antes ou depois:

{"route": "FAST_PATH" | "AGENT",
 "tools": ["nome_exato_1", "nome_exato_2"],
 "confidence": 0.0 a 1.0,
 "reason": "uma frase curta"}

Se a rota for FAST_PATH, "tools" deve ser uma lista vazia."""


class V101OrchestratorPipeline(BasePipeline):
    """Orquestrador por LLM: uma chamada decide rota e ferramenta."""

    name = "V1.0.1"
    description = "Orquestrador por LLM: 1 chamada decide rota e ferramenta, sobre candidatas do estagio local"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        n_candidatas: int = 20,
        use_embeddings: bool = True,
    ):
        """
        Args:
            n_candidatas: quantas ferramentas o estagio local entrega ao LLM.
                Mais candidatas = teto de acerto maior, porem mais tokens de
                entrada. 20 e o ponto de partida; o efeito real e medido.
            use_embeddings: precisa ser IGUAL ao da V1 na comparacao, senao
                estariamos medindo o embedding em vez do orquestrador.
        """
        self.settings = settings or get_settings()
        self.n_candidatas = n_candidatas
        self.use_embeddings = use_embeddings
        self._retriever: Optional[ToolRetriever] = None
        self._router_fallback: Optional[QueryRouter] = None
        self._catalogo_por_nome: Dict[str, object] = {}
        self._cliente = None

        # Contadores acumulados ao longo do lote inteiro.
        self.fallbacks = 0
        """Quantas vezes o plano B (decisao local) foi acionado. Se este numero
        for alto, a comparacao esta medindo a V1 disfarcada de V1.0.1."""

        self.custo_desconhecido = False
        """Vira True se o modelo nao estiver na tabela de precos."""

    # ------------------------------------------------------------------ setup
    def setup(self) -> "V101OrchestratorPipeline":
        """Prepara o estagio de recuperacao, o plano B e o cliente da API."""
        ferramentas = load_tools()

        # 1) O MESMO estagio de recuperacao da V1. Reaproveitar (em vez de
        #    reimplementar) e o que garante que a unica diferenca entre as duas
        #    versoes e quem toma a decisao final.
        self._retriever = ToolRetriever()
        if self.use_embeddings:
            self._retriever.set_dense_provider(build_dense_provider(self.settings))
        self._retriever.fit(ferramentas)

        # 2) Indice nome -> ferramenta, para montar o prompt com as descricoes.
        #    Dicionario porque a busca por nome acontece 20 vezes por mensagem;
        #    percorrer a lista de 285 a cada vez seria desperdicio.
        self._catalogo_por_nome = {t.name: t for t in ferramentas}

        # 3) O plano B: a propria V1. So e usado se o LLM falhar.
        textos, rotulos = load_router_training_data()
        self._router_fallback = QueryRouter().fit(textos, rotulos)

        # 4) Cliente da OpenAI. Import tardio para que este modulo possa ser
        #    importado (e inspecionado) em uma maquina sem o pacote instalado.
        if self.settings.openai_api_key:
            from openai import OpenAI

            self._cliente = OpenAI(api_key=self.settings.openai_api_key, timeout=30.0)
        return self

    # -------------------------------------------------------- prompt do usuario
    def _montar_prompt_usuario(self, query: str, candidatas: List[str]) -> str:
        """Monta a parte variavel do prompt: a mensagem + as candidatas.

        Enviamos `nome: descricao` porque o nome sozinho e ambiguo justamente
        nos casos dificeis. Para separar `bloquear_cartao` de
        `desbloquear_cartao`, a descricao e o que resolve.
        """
        linhas = [
            f"- {nome}: {self._catalogo_por_nome[nome].description}"
            for nome in candidatas
            if nome in self._catalogo_por_nome
        ]
        return (
            f"Mensagem do cliente: {query!r}\n\n"
            f"Ferramentas candidatas ({len(linhas)} de 285 do catalogo):\n"
            + "\n".join(linhas)
        )

    # ---------------------------------------------------------------- process
    def process(self, query: str, k: int = 2) -> PipelineResult:
        """Processa uma mensagem: recuperacao local -> 1 chamada de LLM."""
        if self._retriever is None:
            raise RuntimeError("Chame setup() antes de process().")

        resultado = PipelineResult(query=query, route="FAST_PATH")
        inicio = time.perf_counter()

        # ETAPA 1 — Recuperacao local. Sempre roda, mesmo sem saber a rota
        #           ainda, porque custa ~1,7 ms e zero dolar (ver Decisao 1).
        candidatas: List[str] = []
        try:
            busca = self._retriever.search(query, k=self.n_candidatas)
            candidatas = [m.name for m in busca.matches]
            resultado.cost_usd += COST_RETRIEVAL_USD
            resultado.trace.append(f"recuperacao_local -> {len(candidatas)} candidatas")
        except Exception as erro:
            # Entrada degenerada (vazia, nula) pode quebrar a vetorizacao.
            # Seguimos com lista vazia: o LLM ainda consegue decidir a rota.
            resultado.trace.append(f"recuperacao_local falhou: {type(erro).__name__}")

        # ETAPA 2 — A unica chamada de LLM.
        if self._cliente is None:
            resultado.trace.append("sem OPENAI_API_KEY -> plano B (decisao local)")
            self._aplicar_plano_b(query, candidatas, k, resultado)
        else:
            payload, uso = self._chamar_llm(query, candidatas)

            if uso:  # houve resposta da API: contabiliza tokens e custo
                resultado.llm_calls = 1
                resultado.prompt_tokens = uso.get("prompt_tokens", 0)
                resultado.completion_tokens = uso.get("completion_tokens", 0)
                resultado.reasoning_tokens = uso.get("reasoning_tokens", 0)
                resultado.cached_tokens = uso.get("cached_tokens", 0)
                if resultado.reasoning_tokens:
                    # Sinaliza no rastro: numa tarefa de classificacao isso e
                    # dinheiro gasto em pensamento que nao vira resposta.
                    resultado.trace.append(
                        f"ATENCAO: {resultado.reasoning_tokens} tokens de raciocinio cobrados"
                    )
                custo = calcular_custo(
                    self.settings.orchestrator_model,
                    resultado.prompt_tokens,
                    resultado.completion_tokens,
                    resultado.cached_tokens,
                )
                if custo is None:
                    self.custo_desconhecido = True
                    resultado.trace.append(
                        f"custo desconhecido para o modelo {self.settings.orchestrator_model}"
                    )
                else:
                    resultado.cost_usd += custo

            if payload and self._resposta_valida(payload, candidatas):
                self._aplicar_resposta_llm(payload, candidatas, k, resultado)
            else:
                # Resposta ausente ou malformada -> plano B, registrado.
                self.fallbacks += 1
                resultado.trace.append("resposta do LLM invalida -> plano B (decisao local)")
                self._aplicar_plano_b(query, candidatas, k, resultado)

        # O custo do LLM "caro" que executa a ferramenta continua existindo nas
        # duas versoes; some so no FAST_PATH. Mantemos a mesma contabilidade da
        # V1 para os totais serem comparaveis.
        resultado.cost_usd += COST_ROUTER_USD
        if resultado.route == "AGENT":
            resultado.cost_usd += COST_AGENT_LLM_CALL_USD

        resultado.latency_ms = (time.perf_counter() - inicio) * 1000
        return resultado

    # ------------------------------------------------------- auxiliares do LLM
    def _chamar_llm(self, query: str, candidatas: List[str]):
        """Faz a chamada e devolve (payload_json, uso_de_tokens).

        Devolve (None, uso) se a resposta nao for JSON valido, e (None, None) se
        a chamada falhar por completo. Separar os dois casos importa: no
        primeiro ja gastamos tokens e precisamos contabiliza-los.
        """
        modelo = self.settings.orchestrator_model

        # Parametros que variam conforme a familia do modelo.
        extras: Dict[str, object] = {}
        if modelo_usa_raciocinio(modelo):
            # OBRIGATORIO na familia 5.6: o default de fabrica e "medium", que
            # gasta tokens de raciocinio invisiveis. Medimos 23 tokens por
            # chamada no default contra 0 com "none" — numa tarefa de
            # classificacao esses tokens nao compram acerto nenhum.
            extras["reasoning_effort"] = self.settings.orchestrator_reasoning_effort
        else:
            # Modelos sem raciocinio aceitam temperatura. Zero para a decisao
            # ser reprodutivel: a mesma mensagem deve dar a mesma rota.
            # Em roteamento, criatividade e defeito.
            # (Modelos de raciocinio recusam esse parametro, por isso o else.)
            extras["temperature"] = 0

        try:
            resposta = self._cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": self._montar_prompt_usuario(query, candidatas)},
                ],
                # Forca a saida a ser JSON valido. Sem isso, o modelo as vezes
                # embrulha em ```json ... ``` e o parse quebra.
                response_format={"type": "json_object"},
                **extras,
            )
            uso = self._extrair_uso(resposta.usage)
            try:
                return json.loads(resposta.choices[0].message.content), uso
            except (json.JSONDecodeError, TypeError):
                return None, uso  # gastamos tokens, mas a resposta nao serviu
        except Exception:
            return None, None  # rede, credencial, limite de taxa

    @staticmethod
    def _extrair_uso(uso_bruto) -> Dict[str, int]:
        """Le o bloco `usage` da resposta, incluindo os campos aninhados.

        Os tokens de raciocinio e os cacheados vivem em sub-objetos opcionais
        que nem todo modelo preenche. Usamos `getattr` com valor padrao em vez
        de acesso direto porque um modelo sem raciocinio simplesmente nao tem
        esse campo — e um `AttributeError` aqui derrubaria a chamada depois de
        ela ja ter sido paga.
        """
        if uso_bruto is None:
            return {"prompt_tokens": 0, "completion_tokens": 0,
                    "reasoning_tokens": 0, "cached_tokens": 0}

        det_saida = getattr(uso_bruto, "completion_tokens_details", None)
        det_entrada = getattr(uso_bruto, "prompt_tokens_details", None)
        return {
            "prompt_tokens": getattr(uso_bruto, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(uso_bruto, "completion_tokens", 0) or 0,
            "reasoning_tokens": (getattr(det_saida, "reasoning_tokens", 0) or 0) if det_saida else 0,
            "cached_tokens": (getattr(det_entrada, "cached_tokens", 0) or 0) if det_entrada else 0,
        }

    @staticmethod
    def _resposta_valida(payload: dict, candidatas: List[str]) -> bool:
        """Valida a resposta do LLM antes de confiar nela.

        Checamos duas coisas:
          1. A rota e um dos dois valores permitidos.
          2. Se for AGENT, ao menos uma ferramenta citada existe entre as
             candidatas — blindagem contra ALUCINACAO, que e o modelo inventar
             um nome de ferramenta que nao existe. Executar um nome inventado
             seria um erro em producao, nao so uma imprecisao.
        """
        if payload.get("route") not in {"FAST_PATH", "AGENT"}:
            return False
        if payload["route"] == "AGENT":
            nomes = payload.get("tools") or []
            if not isinstance(nomes, list) or not any(n in candidatas for n in nomes):
                return False
        return True

    def _aplicar_resposta_llm(
        self, payload: dict, candidatas: List[str], k: int, resultado: PipelineResult
    ) -> None:
        """Transfere a resposta validada do LLM para o resultado."""
        resultado.route = payload["route"]
        resultado.confidence = payload.get("confidence")
        if resultado.route == "AGENT":
            # Filtra novamente contra as candidatas: mesmo tendo passado na
            # validacao, a lista pode conter UM nome inventado no meio de nomes
            # validos. So passam adiante nomes que existem.
            escolhidas = [n for n in payload.get("tools", []) if n in candidatas][:k]
            # Se o LLM devolveu menos de k, completamos com o ranking local.
            for nome in candidatas:
                if len(escolhidas) >= k:
                    break
                if nome not in escolhidas:
                    escolhidas.append(nome)
            resultado.tools = escolhidas
        resultado.trace.append(
            f"llm -> {resultado.route} {resultado.tools} "
            f"(motivo: {str(payload.get('reason', ''))[:60]})"
        )

    def _aplicar_plano_b(
        self, query: str, candidatas: List[str], k: int, resultado: PipelineResult
    ) -> None:
        """Plano B: usa a decisao local da V1.

        Chamado quando nao ha chave, a rede falha, ou o LLM devolve algo
        invalido. Cada acionamento ja foi registrado no `trace` por quem chamou.
        """
        try:
            decisao = self._router_fallback.predict(query)
            resultado.route = decisao.route
            resultado.confidence = decisao.confidence
            if decisao.route == "AGENT":
                resultado.tools = candidatas[:k]
        except Exception as erro:
            resultado.error = f"{type(erro).__name__}: {erro}"
