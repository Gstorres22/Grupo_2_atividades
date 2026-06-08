"""
Schema do dataset de Credit Score Classification.

Define as features esperadas pela API, seus tipos, valores válidos e
valores default para o formulário. Centralizar isso facilita a manutenção
quando os outros integrantes do time finalizarem o modelo.

Dataset de referência:
https://www.kaggle.com/datasets/parisrohan/credit-score-classification
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------------
# Listas de valores categóricos válidos (baseadas no dataset Kaggle)
# ----------------------------------------------------------------------

OCCUPATIONS = [
    "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
    "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
    "Accountant", "Musician", "Mechanic", "Writer", "Architect",
]

CREDIT_MIX = ["Good", "Standard", "Bad"]

PAYMENT_OF_MIN_AMOUNT = ["Yes", "No", "NM"]

PAYMENT_BEHAVIOUR = [
    "High_spent_Small_value_payments",
    "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August"]

TYPE_OF_LOAN = [
    "Not Specified", "Credit-Builder Loan", "Personal Loan", "Debt Consolidation Loan",
    "Student Loan", "Payday Loan", "Mortgage Loan", "Auto Loan", "Home Equity Loan",
]


# ----------------------------------------------------------------------
# Definição estruturada das features
# ----------------------------------------------------------------------

@dataclass
class Feature:
    name: str
    label: str
    kind: str            # "number", "int", "select", "multiselect", "text"
    default: Any = None
    options: list | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | int | None = None
    help: str = ""
    group: str = "Geral"


FEATURES: list[Feature] = [
    # ---------- Dados pessoais ----------
    Feature("Age", "Idade", "int", default=30, min_value=18, max_value=100,
            step=1, help="Idade do cliente em anos.", group="Dados Pessoais"),
    Feature("Occupation", "Ocupação", "select", default="Engineer",
            options=OCCUPATIONS, help="Profissão informada pelo cliente.",
            group="Dados Pessoais"),
    Feature("Month", "Mês de referência", "select", default="May",
            options=MONTHS, help="Mês da observação financeira.",
            group="Dados Pessoais"),

    # ---------- Renda ----------
    Feature("Annual_Income", "Renda anual (US$)", "number", default=50000.0,
            min_value=0.0, max_value=1_000_000.0, step=100.0,
            help="Renda bruta anual do cliente.", group="Renda"),
    Feature("Monthly_Inhand_Salary", "Salário líquido mensal (US$)", "number",
            default=3500.0, min_value=0.0, max_value=100_000.0, step=50.0,
            help="Salário recebido em mãos por mês.", group="Renda"),
    Feature("Monthly_Balance", "Saldo mensal disponível (US$)", "number",
            default=300.0, min_value=-5000.0, max_value=50_000.0, step=10.0,
            help="Saldo médio mensal disponível após despesas.", group="Renda"),
    Feature("Amount_invested_monthly", "Valor investido mensalmente (US$)",
            "number", default=100.0, min_value=0.0, max_value=20_000.0,
            step=10.0, help="Valor investido em média a cada mês.",
            group="Renda"),

    # ---------- Contas e cartões ----------
    Feature("Num_Bank_Accounts", "Nº de contas bancárias", "int", default=3,
            min_value=0, max_value=30, step=1, group="Contas e Cartões"),
    Feature("Num_Credit_Card", "Nº de cartões de crédito", "int", default=4,
            min_value=0, max_value=30, step=1, group="Contas e Cartões"),
    Feature("Interest_Rate", "Taxa de juros média (%)", "number",
            default=12.0, min_value=0.0, max_value=50.0, step=0.5,
            group="Contas e Cartões"),

    # ---------- Empréstimos ----------
    Feature("Num_of_Loan", "Nº de empréstimos ativos", "int", default=2,
            min_value=0, max_value=20, step=1, group="Empréstimos"),
    Feature("Type_of_Loan", "Tipos de empréstimo", "multiselect",
            default=["Personal Loan"], options=TYPE_OF_LOAN,
            help="Selecione todos os tipos aplicáveis.",
            group="Empréstimos"),
    Feature("Outstanding_Debt", "Dívida pendente (US$)", "number",
            default=1000.0, min_value=0.0, max_value=200_000.0, step=50.0,
            group="Empréstimos"),
    Feature("Total_EMI_per_month", "EMI total mensal (US$)", "number",
            default=100.0, min_value=0.0, max_value=20_000.0, step=10.0,
            help="Parcela total mensal de empréstimos.",
            group="Empréstimos"),

    # ---------- Histórico de pagamentos ----------
    Feature("Delay_from_due_date", "Atraso médio (dias)", "int", default=5,
            min_value=-10, max_value=90, step=1,
            help="Dias médios de atraso a partir da data de vencimento.",
            group="Histórico de Pagamentos"),
    Feature("Num_of_Delayed_Payment", "Nº de pagamentos atrasados", "int",
            default=2, min_value=0, max_value=50, step=1,
            group="Histórico de Pagamentos"),
    Feature("Payment_of_Min_Amount", "Pagou valor mínimo?", "select",
            default="No", options=PAYMENT_OF_MIN_AMOUNT,
            help="Yes / No / NM (Not Mentioned).",
            group="Histórico de Pagamentos"),
    Feature("Payment_Behaviour", "Comportamento de pagamento", "select",
            default="Low_spent_Small_value_payments",
            options=PAYMENT_BEHAVIOUR,
            group="Histórico de Pagamentos"),

    # ---------- Crédito ----------
    Feature("Changed_Credit_Limit", "Mudança no limite de crédito (%)",
            "number", default=5.0, min_value=-100.0, max_value=100.0,
            step=0.1, group="Crédito"),
    Feature("Num_Credit_Inquiries", "Nº de consultas de crédito", "int",
            default=3, min_value=0, max_value=50, step=1, group="Crédito"),
    Feature("Credit_Mix", "Mix de crédito", "select", default="Standard",
            options=CREDIT_MIX, group="Crédito"),
    Feature("Credit_Utilization_Ratio", "Taxa de utilização de crédito (%)",
            "number", default=30.0, min_value=0.0, max_value=100.0,
            step=0.5, group="Crédito"),
    Feature("Credit_History_Age_Months", "Idade do histórico (meses)", "int",
            default=120, min_value=0, max_value=600, step=1,
            help="Tempo total de histórico de crédito convertido em meses.",
            group="Crédito"),
]


# Mapeamento de classes do modelo para visualização
CREDIT_SCORE_CLASSES = {
    "Good":     {"label": "Bom",       "color": "#22C55E", "emoji": "🟢"},
    "Standard": {"label": "Padrão",    "color": "#F59E0B", "emoji": "🟡"},
    "Poor":     {"label": "Ruim",      "color": "#EF4444", "emoji": "🔴"},
}


def get_feature_groups() -> dict[str, list[Feature]]:
    """Agrupa as features pelo atributo `group` preservando a ordem."""
    groups: dict[str, list[Feature]] = {}
    for f in FEATURES:
        groups.setdefault(f.group, []).append(f)
    return groups


def default_payload() -> dict[str, Any]:
    """Retorna um dicionário com os valores default de todas as features."""
    return {f.name: f.default for f in FEATURES}


def required_columns() -> list[str]:
    """Lista de colunas obrigatórias para o upload em lote (CSV)."""
    return [f.name for f in FEATURES]
