"""
QuantumFinance · Credit Score · Script de treinamento
=====================================================

Treina um modelo de classificação de credit score (Good/Standard/Poor)
sobre o dataset Kaggle e:

  1. Cria um experimento no MLflow.
  2. Para cada execução, registra **parâmetros, métricas e artefatos**
     (modelo serializado, label encoder, classification report, matriz
     de confusão como imagem, e o próprio dataset de validação).
  3. Registra o modelo no **MLflow Model Registry** com versão automática.
  4. Avalia o critério de promoção: se a métrica `f1_macro` for melhor
     do que a do modelo atualmente em Produção, promove o novo modelo
     para o estágio "Production" e arquiva o anterior.

Como rodar:

    python -m training.train --data data/train.csv --algo random_forest

Critério de avaliação adotado:
    F1-score macro >= F1-score macro do modelo atual em Production.
    Isso evita degradar a performance e respeita o desbalanceamento
    natural das 3 classes.

Promoção:
    Realizada via MlflowClient.transition_model_version_stage().
    O modelo anterior em "Production" é arquivado automaticamente.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # backend sem GUI (para rodar em servidores)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Imports internos do projeto
from training.data_prep import (
    CATEGORICAL_COLS, NUMERIC_COLS, TARGET,
    clean_dataframe, split_features_target,
)


# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------

EXPERIMENT_NAME = "quantumfinance-credit-score"
REGISTERED_MODEL_NAME = "credit_score_clf"
PRODUCTION_STAGE = "Production"
STAGING_STAGE = "Staging"
ARCHIVED_STAGE = "Archived"

# Métrica usada como gate de promoção (deve estar entre as métricas logadas)
PROMOTION_METRIC = "f1_macro"


# ----------------------------------------------------------------------
# Modelos disponíveis (CLI --algo)
# ----------------------------------------------------------------------

def build_estimator(algo: str, **params):
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 20),
            min_samples_split=params.get("min_samples_split", 5),
            class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
    if algo == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 5),
            random_state=42,
        )
    if algo == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0), max_iter=1000,
            class_weight="balanced", n_jobs=-1,
        )
    raise ValueError(f"Algoritmo desconhecido: {algo}")


def build_pipeline(estimator) -> Pipeline:
    """Pré-processamento + estimador em um único Pipeline serializável."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])
    return Pipeline([("preprocessor", preprocessor), ("clf", estimator)])


# ----------------------------------------------------------------------
# Avaliação
# ----------------------------------------------------------------------

def evaluate(pipeline, X_val, y_val, label_encoder) -> dict:
    """Avalia o pipeline. `y_val` deve estar codificado (inteiros)."""
    y_pred = pipeline.predict(X_val)
    metrics = {
        "accuracy":        accuracy_score(y_val, y_pred),
        "f1_macro":        f1_score(y_val, y_pred, average="macro"),
        "f1_weighted":     f1_score(y_val, y_pred, average="weighted"),
        "precision_macro": precision_score(y_val, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro":    recall_score(y_val, y_pred, average="macro",
                                        zero_division=0),
    }
    labels_str = list(label_encoder.classes_)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(labels_str))))
    report = classification_report(
        y_val, y_pred, labels=list(range(len(labels_str))),
        target_names=labels_str, zero_division=0,
    )
    return {"metrics": metrics, "confusion_matrix": cm,
            "labels": labels_str, "report": report}


def save_confusion_matrix_png(cm, labels, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------
# Lógica de promoção (gate)
# ----------------------------------------------------------------------

def current_production_metric(client: MlflowClient,
                              model_name: str, metric: str) -> float | None:
    """Retorna a métrica registrada na versão atualmente em Production."""
    try:
        versions = client.get_latest_versions(model_name, stages=[PRODUCTION_STAGE])
    except mlflow.exceptions.RestException:
        return None
    if not versions:
        return None
    run_id = versions[0].run_id
    run = client.get_run(run_id)
    return run.data.metrics.get(metric)


def promote_to_production(client: MlflowClient,
                          model_name: str, version: str) -> None:
    """Promove a versão `version` para Production e arquiva as anteriores.

    `archive_existing_versions=True` faz com que qualquer outra versão
    que esteja em Production seja movida para Archived automaticamente.
    """
    client.transition_model_version_stage(
        name=model_name, version=version,
        stage=PRODUCTION_STAGE, archive_existing_versions=True,
    )


# ----------------------------------------------------------------------
# Treino + tracking
# ----------------------------------------------------------------------

def train(args) -> None:
    # --- MLflow tracking URI ---
    tracking_uri = args.tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", f"file://{Path('mlruns').resolve()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"[train] tracking URI: {tracking_uri}")
    print(f"[train] dataset: {args.data}")

    # --- Carrega e limpa dados ---
    df_raw = pd.read_csv(args.data, low_memory=False)
    print(f"[train] dataset bruto: {df_raw.shape}")
    df = clean_dataframe(df_raw, training=True).dropna(subset=[TARGET])
    print(f"[train] dataset após limpeza: {df.shape}")

    X, y_raw = split_features_target(df)
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # --- Treina ---
    params = {
        "n_estimators":   args.n_estimators,
        "max_depth":      args.max_depth,
        "learning_rate":  args.learning_rate,
        "C":              args.C,
    }
    estimator = build_estimator(args.algo, **params)
    pipeline = build_pipeline(estimator)

    run_name = f"{args.algo}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[train] run_id: {run_id}")

        # --- LOG: parâmetros ---
        mlflow.log_param("algo", args.algo)
        mlflow.log_param("test_size", args.test_size)
        for k, v in params.items():
            if v is not None:
                mlflow.log_param(k, v)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("classes", list(le.classes_))

        # --- Treina ---
        pipeline.fit(X_train, y_train)

        # --- Avalia ---
        eval_out = evaluate(pipeline, X_val, y_val, le)
        metrics = eval_out["metrics"]
        print("[train] métricas:", json.dumps(metrics, indent=2))

        # --- LOG: métricas ---
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # --- LOG: artefatos ---
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # classification_report
            (tmp / "classification_report.txt").write_text(eval_out["report"])
            # confusion matrix png
            save_confusion_matrix_png(
                eval_out["confusion_matrix"], eval_out["labels"],
                str(tmp / "confusion_matrix.png"),
            )
            # label encoder (precisamos dele na inferência para decodificar)
            import joblib
            joblib.dump(le, tmp / "label_encoder.joblib")
            # amostra do val
            sample = X_val.head(20).copy()
            sample["__y_true__"] = le.inverse_transform(y_val[:20])
            sample.to_csv(tmp / "validation_sample.csv", index=False)

            mlflow.log_artifacts(str(tmp), artifact_path="artifacts")

        # --- LOG: modelo (vai para o Registry) ---
        # Assinatura para que a API saiba que tipos espera/retorna
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_val.head(5),
                                     pipeline.predict(X_val.head(5)))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=X_val.head(2),
        )

    # --- Recupera a versão recém-criada e decide a promoção ---
    latest = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
    if not latest:
        print("[train] não foi possível localizar a versão recém-registrada.")
        return
    new_version = latest[0].version
    print(f"[train] nova versão registrada: v{new_version}")

    # Move imediatamente para Staging (boa prática)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=new_version, stage=STAGING_STAGE,
    )
    print(f"[train] v{new_version} -> Staging")

    # --- GATE de promoção ---
    new_score = metrics[PROMOTION_METRIC]
    prod_score = current_production_metric(
        client, REGISTERED_MODEL_NAME, PROMOTION_METRIC
    )

    if prod_score is None:
        print(f"[train] não há modelo em Production. Promovendo v{new_version}.")
        promote_to_production(client, REGISTERED_MODEL_NAME, new_version)
        decision = "promoted (first model)"
    elif new_score >= prod_score:
        print(f"[train] {PROMOTION_METRIC} novo={new_score:.4f} "
              f">= prod={prod_score:.4f} → promovendo")
        promote_to_production(client, REGISTERED_MODEL_NAME, new_version)
        decision = "promoted"
    else:
        print(f"[train] {PROMOTION_METRIC} novo={new_score:.4f} "
              f"< prod={prod_score:.4f} → mantendo em Staging")
        decision = "kept_in_staging"

    # --- Tag para histórico ---
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, new_version, "decision", decision
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, new_version, "f1_macro", f"{new_score:.4f}"
    )
    print(f"[train] decisão: {decision}")
    print("[train] ✅ Concluído.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treinamento de Credit Score")
    p.add_argument("--data", default="data/train.csv",
                   help="Caminho do CSV de treino (default: data/train.csv)")
    p.add_argument("--algo", default="random_forest",
                   choices=["random_forest", "gradient_boosting",
                            "logistic_regression"])
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI (default: file:./mlruns)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
