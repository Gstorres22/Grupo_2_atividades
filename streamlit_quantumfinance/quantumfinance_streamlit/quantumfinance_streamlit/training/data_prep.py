"""
Limpeza e preparação do dataset Credit Score Classification (Kaggle).

O dataset original contém muito ruído nas colunas:
  - números armazenados como string com underscores ("19114.12_"),
  - placeholders ("_______", "!@9#%8", "NM"),
  - listas como string ("Auto Loan, Credit-Builder Loan, and ...").

Este módulo cuida disso e devolve um DataFrame pronto para treinar.
Também é importado pela inferência para aplicar exatamente as
mesmas transformações no payload recebido pela API.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Colunas
# ----------------------------------------------------------------------

# Colunas que NÃO vão para o modelo (identificadores ou alvo)
DROP_COLS = ["ID", "Customer_ID", "Name", "SSN"]
TARGET = "Credit_Score"

# Colunas numéricas que vêm contaminadas com '_' e outros chars
NUMERIC_STRING_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    "Monthly_Balance",
]

# Colunas categóricas finais que o modelo vai usar
CATEGORICAL_COLS = [
    "Month", "Occupation", "Credit_Mix",
    "Payment_of_Min_Amount", "Payment_Behaviour",
]

# Colunas numéricas finais
NUMERIC_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Credit_History_Age_Months", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
    "Num_Type_of_Loan",
]

# Faixas razoáveis para clipar outliers absurdos
CLIP_RANGES = {
    "Age": (0, 100),
    "Num_Bank_Accounts": (0, 30),
    "Num_Credit_Card": (0, 30),
    "Interest_Rate": (0, 50),
    "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 50),
    "Num_Credit_Inquiries": (0, 50),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    """Converte uma coluna 'suja' para numérica, tratando lixo."""
    s = series.astype(str).str.replace("_", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NA": np.nan, "!@9#%8": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _parse_credit_history(value) -> float:
    """Converte 'X Years and Y Months' em meses (float). NaN se inválido."""
    if pd.isna(value):
        return np.nan
    m = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(value))
    if not m:
        return np.nan
    return int(m.group(1)) * 12 + int(m.group(2))


def _count_loan_types(value) -> int:
    """Conta quantos tipos de empréstimo o cliente tem."""
    if pd.isna(value) or str(value).strip() in {"", "Not Specified"}:
        return 0
    raw = str(value).replace(" and ", ", ")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return len(parts)


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Aplica todo o pipeline de limpeza.

    Args:
        df: DataFrame bruto (formato do CSV original do Kaggle ou payload da API).
        training: Se True, mantém a coluna target. Se False, ignora.
    """
    df = df.copy()

    # Remove colunas de identificação (existirem)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                 errors="ignore")

    # Trata Credit_History_Age (pode vir como string "X Years and Y Months"
    # ou já convertida em meses pelo payload da API).
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(
            _parse_credit_history
        )
        df = df.drop(columns=["Credit_History_Age"])
    elif "Credit_History_Age_Months" in df.columns:
        df["Credit_History_Age_Months"] = pd.to_numeric(
            df["Credit_History_Age_Months"], errors="coerce"
        )
    else:
        df["Credit_History_Age_Months"] = np.nan

    # Type_of_Loan -> número de tipos
    if "Type_of_Loan" in df.columns:
        df["Num_Type_of_Loan"] = df["Type_of_Loan"].apply(_count_loan_types)
        df = df.drop(columns=["Type_of_Loan"])
    else:
        df["Num_Type_of_Loan"] = 0

    # Limpeza das colunas numéricas que vêm como string
    for col in NUMERIC_STRING_COLS:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # Garantir tipo numérico nos demais campos numéricos
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categóricas: padroniza placeholders como NaN
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip()
                       .replace({"_______": np.nan, "_": np.nan,
                                 "!@9#%8": np.nan, "nan": np.nan}))

    # Clipa valores absurdos (idade negativa, etc.)
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Seleção final das colunas
    keep = NUMERIC_COLS + CATEGORICAL_COLS
    if training and TARGET in df.columns:
        keep = keep + [TARGET]
    df = df[[c for c in keep if c in df.columns]]

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y. Espera o DataFrame já limpo."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def feature_columns() -> dict[str, list[str]]:
    """Retorna as colunas finais separadas por tipo (uso do pipeline)."""
    return {"numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS}
