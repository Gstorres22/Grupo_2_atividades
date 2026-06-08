"""Smoke tests básicos do projeto."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------- App Streamlit ---------

def test_imports_app():
    from app.utils import schema, api_client, styles, auth
    from app.components import charts, form_builder
    assert schema.FEATURES
    assert hasattr(api_client, "CreditScoreAPIClient")
    assert hasattr(auth, "login_demo")
    assert hasattr(auth, "render_login_screen")
    assert hasattr(charts, "score_gauge")
    assert hasattr(form_builder, "render_form")


def test_default_payload_contains_all_features():
    from app.utils.schema import FEATURES, default_payload
    payload = default_payload()
    assert set(payload.keys()) == {f.name for f in FEATURES}


def test_demo_mode_prediction():
    from app.utils.api_client import APIConfig, CreditScoreAPIClient
    from app.utils.schema import default_payload

    client = CreditScoreAPIClient(APIConfig(base_url="http://x"),
                                  demo_mode=True)
    result = client.predict(default_payload())
    assert result["credit_score"] in {"Good", "Standard", "Poor"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_demo_batch():
    from app.utils.api_client import APIConfig, CreditScoreAPIClient
    from app.utils.schema import default_payload

    client = CreditScoreAPIClient(APIConfig(base_url="http://x"),
                                  demo_mode=True)
    out = client.predict_batch([default_payload() for _ in range(3)])
    assert len(out) == 3


# --------- Autenticação ---------

def test_login_demo_ok():
    from app.utils.auth import login_demo
    s = login_demo("admin", "admin123")
    assert s.username == "admin"
    assert s.token
    assert s.is_demo
    assert s.is_valid()


def test_login_demo_fail():
    import pytest
    from app.utils.auth import login_demo
    with pytest.raises(RuntimeError):
        login_demo("admin", "errada")


# --------- Treinamento / Data prep ---------

def test_clean_dataframe_roundtrip():
    import pandas as pd
    from training.data_prep import clean_dataframe, NUMERIC_COLS, TARGET

    raw = pd.DataFrame([{
        "ID": "0x1", "Customer_ID": "CUS_0xa", "Name": "X", "SSN": "1",
        "Month": "May", "Age": "30_", "Occupation": "Engineer",
        "Annual_Income": "50000.0", "Monthly_Inhand_Salary": 3500,
        "Num_Bank_Accounts": 3, "Num_Credit_Card": 4, "Interest_Rate": 12,
        "Num_of_Loan": "2", "Type_of_Loan": "Personal Loan, and Auto Loan",
        "Delay_from_due_date": 5, "Num_of_Delayed_Payment": "2",
        "Changed_Credit_Limit": "5.0", "Num_Credit_Inquiries": 3,
        "Credit_Mix": "Standard", "Outstanding_Debt": "1000.0",
        "Credit_Utilization_Ratio": 30.0, "Credit_History_Age": "10 Years and 0 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 100.0,
        "Amount_invested_monthly": "100.0", "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 300.0, "Credit_Score": "Good",
    }])
    df = clean_dataframe(raw, training=True)
    assert "ID" not in df.columns
    assert "Credit_History_Age_Months" in df.columns
    assert df["Credit_History_Age_Months"].iloc[0] == 120
    assert "Num_Type_of_Loan" in df.columns
    assert df["Num_Type_of_Loan"].iloc[0] == 2
    assert TARGET in df.columns
