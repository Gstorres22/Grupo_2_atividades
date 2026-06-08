"""
Cliente HTTP para a API de Credit Score da QuantumFinance.

Este módulo encapsula todas as chamadas à API e foi escrito de forma
defensiva para que, mesmo antes da API estar disponível, o app Streamlit
continue funcionando — exibindo mensagens de erro amigáveis ou usando
um modo "demo" (mock) quando habilitado.

Quando a equipe finalizar a API, basta ajustar os endpoints abaixo se
necessário. Os contratos esperados são:

  POST  /predict           -> { "input": {...features...} }     -> PredictionResponse
  POST  /predict/batch     -> { "inputs": [ {...}, {...} ] }   -> list[PredictionResponse]
  GET   /health            -> { "status": "ok", ... }
  GET   /model/info        -> { "name", "version", "stage", ... }

PredictionResponse esperado:
  {
    "credit_score": "Good" | "Standard" | "Poor",
    "probabilities": { "Good": 0.6, "Standard": 0.3, "Poor": 0.1 },
    "model_version": "3",
    "request_id": "uuid",
    "latency_ms": 23.4
  }
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any

import requests


# ----------------------------------------------------------------------
# Tipos
# ----------------------------------------------------------------------

@dataclass
class APIConfig:
    base_url: str
    api_key: str = ""
    timeout: int = 30
    auth_token: str = ""  # JWT obtido via /login


class APIError(Exception):
    """Erro genérico de chamada à API, com status code e mensagem amigável."""

    def __init__(self, message: str, status_code: int | None = None,
                 details: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


# ----------------------------------------------------------------------
# Cliente
# ----------------------------------------------------------------------

class CreditScoreAPIClient:
    """Wrapper fino sobre `requests` para a API de Credit Score.

    Suporta três modos de funcionamento:

    * **api**:   chama a API HTTP normalmente.
    * **local**: ignora a API e usa o `Predictor` MLflow local para
                 inferir (útil quando a API ainda não subiu).
    * **demo**:  retorna predições simuladas (regra simples).

    Quando `fallback_local=True`, em caso de falha de conexão com a API
    o cliente automaticamente tenta usar o modelo local.
    """

    def __init__(self, config: APIConfig, demo_mode: bool = False,
                 use_local: bool = False, fallback_local: bool = True):
        self.config = config
        self.demo_mode = demo_mode
        self.use_local = use_local
        self.fallback_local = fallback_local
        self._local_predictor = None  # carregamento lazy

    # ------------------------- helpers internos --------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Prioridade: JWT do /login > API key fixa.
        token = self.config.auth_token or self.config.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key
        return headers

    def _url(self, path: str) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}{path}"

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            resp = requests.request(
                method=method,
                url=self._url(path),
                headers=self._headers(),
                timeout=self.config.timeout,
                **kwargs,
            )
        except requests.exceptions.ConnectionError as e:
            raise APIError(
                "Não foi possível conectar à API. Verifique se a URL está "
                "correta e se o serviço está no ar.",
                details=str(e),
            )
        except requests.exceptions.Timeout as e:
            raise APIError(
                f"A API demorou mais que {self.config.timeout}s para responder.",
                details=str(e),
            )
        except requests.exceptions.RequestException as e:
            raise APIError("Erro inesperado ao chamar a API.", details=str(e))

        if resp.status_code == 401:
            raise APIError("API Key inválida ou ausente (401).", 401, resp.text)
        if resp.status_code == 403:
            raise APIError("Acesso negado (403). Verifique permissões da chave.",
                           403, resp.text)
        if resp.status_code == 429:
            raise APIError(
                "Limite de requisições atingido (429). "
                "Aguarde alguns instantes antes de tentar novamente.",
                429, resp.text,
            )
        if resp.status_code >= 500:
            raise APIError(
                f"Erro interno da API ({resp.status_code}). Tente novamente.",
                resp.status_code, resp.text,
            )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text
            raise APIError(
                f"Requisição inválida ({resp.status_code}).",
                resp.status_code, payload,
            )

        try:
            return resp.json()
        except ValueError:
            raise APIError("Resposta da API não é um JSON válido.",
                           resp.status_code, resp.text)

    # ---------------------------- modo local ----------------------------

    def _get_local_predictor(self):
        """Carrega (sob demanda) o Predictor MLflow local."""
        if self._local_predictor is None:
            try:
                # Import tardio para não pagar o custo se não for usado.
                from inference.predictor import Predictor
                self._local_predictor = Predictor.from_registry()
            except Exception as e:
                raise APIError(
                    "Não foi possível carregar o modelo local. "
                    "Verifique se o treinamento já foi executado e se "
                    "existe um modelo em Production no MLflow.",
                    details=str(e),
                )
        return self._local_predictor

    def _maybe_fallback(self, exc: APIError, op: str, *args):
        """Se fallback_local estiver ativo e o erro for de conexão/5xx,
        tenta o modelo local antes de propagar o erro."""
        if not self.fallback_local:
            raise exc
        is_connection_issue = (
            exc.status_code is None
            or (exc.status_code is not None and exc.status_code >= 500)
        )
        if not is_connection_issue:
            raise exc
        try:
            predictor = self._get_local_predictor()
        except APIError:
            raise exc  # falhou também no local -> retorna o erro original
        if op == "predict":
            return predictor.predict_one(args[0])
        if op == "predict_batch":
            return predictor.predict_batch(args[0])
        raise exc

    # ------------------------------- API ---------------------------------

    def health(self) -> dict:
        if self.demo_mode:
            return {"status": "ok", "mode": "demo", "uptime_s": 12345}
        if self.use_local:
            predictor = self._get_local_predictor()
            return {
                "status": "ok", "mode": "local",
                "model_version": predictor.version_info["version"],
            }
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        if self.demo_mode:
            return {
                "name": "credit_score_clf",
                "version": "demo-0",
                "stage": "Production",
                "algorithm": "RandomForestClassifier",
                "trained_at": "2026-05-20T12:00:00Z",
            }
        if self.use_local:
            predictor = self._get_local_predictor()
            return {
                "name": predictor.version_info["name"],
                "version": predictor.version_info["version"],
                "stage": predictor.version_info["stage"],
                "f1_macro": predictor.version_info["metrics"].get("f1_macro"),
                "accuracy": predictor.version_info["metrics"].get("accuracy"),
                "mode": "local",
            }
        return self._request("GET", "/model/info")

    def predict(self, features: dict[str, Any]) -> dict:
        if self.demo_mode:
            return _mock_prediction(features)
        if self.use_local:
            return self._get_local_predictor().predict_one(features)
        try:
            return self._request("POST", "/predict",
                                 json={"input": features})
        except APIError as e:
            return self._maybe_fallback(e, "predict", features)

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[dict]:
        if self.demo_mode:
            return [_mock_prediction(r) for r in rows]
        if self.use_local:
            return self._get_local_predictor().predict_batch(rows)
        try:
            payload = {"inputs": rows}
            result = self._request("POST", "/predict/batch", json=payload)
            if isinstance(result, list):
                return result
            return result.get("predictions", [])
        except APIError as e:
            return self._maybe_fallback(e, "predict_batch", rows)


# ----------------------------------------------------------------------
# Mock para o modo demo (quando a API ainda não está disponível)
# ----------------------------------------------------------------------

def _mock_prediction(features: dict[str, Any]) -> dict:
    """Gera uma predição falsa usando regras simples sobre as features,
    apenas para que o app possa ser demonstrado sem a API real."""
    score = 50
    score += min(int(features.get("Annual_Income", 0)) // 5000, 30)
    score -= int(features.get("Num_of_Delayed_Payment", 0)) * 2
    score -= int(features.get("Delay_from_due_date", 0)) // 2
    score -= int(features.get("Outstanding_Debt", 0)) // 1000

    if score >= 60:
        label = "Good"
        probs = {"Good": 0.72, "Standard": 0.22, "Poor": 0.06}
    elif score >= 30:
        label = "Standard"
        probs = {"Good": 0.20, "Standard": 0.65, "Poor": 0.15}
    else:
        label = "Poor"
        probs = {"Good": 0.05, "Standard": 0.25, "Poor": 0.70}

    return {
        "credit_score": label,
        "probabilities": probs,
        "model_version": "demo-0",
        "request_id": f"demo-{random.randint(1000, 9999)}",
        "latency_ms": round(random.uniform(15, 50), 2),
    }
