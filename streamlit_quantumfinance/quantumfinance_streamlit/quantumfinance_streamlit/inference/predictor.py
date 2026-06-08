"""
QuantumFinance · Credit Score · Inferência
==========================================

Lê o modelo atualmente em **Production** no MLflow Model Registry
(o último promovido pelo script de treinamento) e expõe uma API
Python simples para fazer predições.

Uso direto (CLI):

    python -m inference.predictor --input data/test.csv --output preds.csv

Uso programático:

    from inference.predictor import Predictor
    pred = Predictor.from_registry()   # carrega versão Production
    pred.predict_one({"Age": 30, "Annual_Income": 50000, ...})

Esta mesma classe é usada como **fallback local** pelo app Streamlit
quando a API HTTP está indisponível, e pode ser usada também pela
equipe que está construindo a API REST (basta envolver com FastAPI).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from training.data_prep import clean_dataframe


# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------

REGISTERED_MODEL_NAME = "credit_score_clf"
PRODUCTION_STAGE = "Production"


def default_tracking_uri() -> str:
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"file://{Path('mlruns').resolve()}",
    )


# ----------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------

class Predictor:
    """Carrega o modelo em Production e faz predições."""

    def __init__(self, model, label_encoder, version_info: dict):
        self.model = model
        self.label_encoder = label_encoder
        self.version_info = version_info

    # --------- Factories ---------

    @classmethod
    def from_registry(cls, tracking_uri: str | None = None,
                      stage: str = PRODUCTION_STAGE,
                      model_name: str = REGISTERED_MODEL_NAME) -> "Predictor":
        """Carrega a versão `stage` (default: Production) do Registry."""
        tracking_uri = tracking_uri or default_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise RuntimeError(
                f"Nenhum modelo encontrado em estágio '{stage}'. "
                f"Rode `python -m training.train` antes."
            )
        v = versions[0]
        print(f"[inference] carregando {model_name} v{v.version} "
              f"(stage={stage}, run_id={v.run_id})")

        # Carrega o modelo como flavor nativo do sklearn para acessar
        # predict_proba diretamente e evitar a validação estrita de schema
        # do pyfunc (nossa limpeza já garante os tipos corretos).
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)

        # Carrega o label encoder salvo como artefato no run
        artifact_path = client.download_artifacts(
            v.run_id, "artifacts/label_encoder.joblib"
        )
        le = joblib.load(artifact_path)

        run = client.get_run(v.run_id)
        version_info = {
            "name": model_name,
            "version": v.version,
            "stage": stage,
            "run_id": v.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": {**v.tags},
        }
        return cls(model, le, version_info)

    # --------- Predição ---------

    def _prepare(self, payload: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Aceita dict (1 cliente) ou DataFrame (várias linhas)."""
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        else:
            df = pd.DataFrame(payload)
        return clean_dataframe(df, training=False)

    def predict_one(self, payload: dict[str, Any]) -> dict:
        """Predição para um único cliente."""
        df = self._prepare(payload)
        return self._predict_df(df)[0]

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[dict]:
        """Predição para uma lista de clientes."""
        if not rows:
            return []
        df = self._prepare(pd.DataFrame(rows))
        return self._predict_df(df)

    def _predict_df(self, df: pd.DataFrame) -> list[dict]:
        raw_preds = self.model.predict(df)
        proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(df)
            except Exception:
                proba = None

        results = []
        labels = list(self.label_encoder.classes_)
        for i, pred in enumerate(np.asarray(raw_preds).ravel()):
            label = self.label_encoder.inverse_transform([int(pred)])[0]
            item = {
                "credit_score": str(label),
                "model_version": str(self.version_info["version"]),
                "model_stage": self.version_info["stage"],
            }
            if proba is not None:
                item["probabilities"] = {
                    lab: float(proba[i, j]) for j, lab in enumerate(labels)
                }
            results.append(item)
        return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência de Credit Score")
    p.add_argument("--input", required=False,
                   help="CSV com clientes a serem scorados.")
    p.add_argument("--output", default="predictions.csv",
                   help="CSV de saída com as predições.")
    p.add_argument("--single", default=None,
                   help="JSON inline com 1 cliente (ex.: --single '{\"Age\": 30, ...}')")
    p.add_argument("--stage", default=PRODUCTION_STAGE,
                   help="Estágio do modelo a carregar (default: Production)")
    p.add_argument("--tracking-uri", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor.from_registry(
        tracking_uri=args.tracking_uri, stage=args.stage,
    )

    # Informações do modelo carregado
    print("[inference] modelo carregado:")
    print(json.dumps({
        "name": predictor.version_info["name"],
        "version": predictor.version_info["version"],
        "stage": predictor.version_info["stage"],
        "f1_macro": predictor.version_info["metrics"].get("f1_macro"),
        "accuracy": predictor.version_info["metrics"].get("accuracy"),
    }, indent=2))

    if args.single:
        payload = json.loads(args.single)
        result = predictor.predict_one(payload)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.input:
        raise SystemExit("Informe --input <csv> ou --single '<json>'.")

    df = pd.read_csv(args.input, low_memory=False)
    print(f"[inference] {len(df)} linhas para predizer.")
    rows = df.to_dict(orient="records")
    preds = predictor.predict_batch(rows)

    out = df.copy()
    out["credit_score"] = [p["credit_score"] for p in preds]
    if "probabilities" in preds[0]:
        for cls in preds[0]["probabilities"]:
            out[f"prob_{cls}"] = [p["probabilities"][cls] for p in preds]
    out["model_version"] = [p["model_version"] for p in preds]

    out.to_csv(args.output, index=False)
    print(f"[inference] ✅ {len(out)} predições salvas em {args.output}")


if __name__ == "__main__":
    main()
