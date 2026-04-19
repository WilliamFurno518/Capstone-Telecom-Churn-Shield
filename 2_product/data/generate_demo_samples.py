"""
ChurnShield - Demo Sample Generator (Phase 2)
=============================================

Generates two synthetic CSVs used as demo fixtures for the product:

  - sample_telecom_demo.csv   Telco subscribers with ~15% misaligned plans
  - sample_banking_demo.csv   Banking clients with ~12% misaligned products

These files are *only* used to show the product off on a fresh machine.
The real research signal lives in Phase 1, on a real dataset. The product
itself is data-agnostic: you can drop any CSV with a plan column in and
get the same workflow.

Run from anywhere:

    python generate_demo_samples.py

The script writes its output into the same directory as itself
(`2_product/data/`), regardless of the current working directory.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_telecom_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Synthesise a telecom dataset with a plausible plan-usage relationship.

    ~15% of customers are injected as misaligned on purpose - either heavy
    users on a light plan (upsell candidates) or light users on a heavy
    plan (downsell candidates).
    """
    rng = np.random.default_rng(seed)

    plans = ["Basic", "Standard", "Premium", "Unlimited"]
    plan_prices = {"Basic": 19.99, "Standard": 34.99, "Premium": 59.99, "Unlimited": 89.99}
    plan_data_caps = {"Basic": 5, "Standard": 15, "Premium": 50, "Unlimited": 999}
    plan_call_caps = {"Basic": 100, "Standard": 300, "Premium": 600, "Unlimited": 9999}

    rows = []
    for i in range(n):
        current_plan = rng.choice(plans, p=[0.30, 0.35, 0.25, 0.10])
        misaligned = rng.random() < 0.15

        if misaligned and current_plan in ("Basic", "Standard"):
            # Heavy user on a light plan
            data_usage = rng.uniform(plan_data_caps[current_plan] * 1.5,
                                     plan_data_caps[current_plan] * 4.0)
            calls = int(rng.uniform(plan_call_caps[current_plan] * 1.3,
                                    plan_call_caps[current_plan] * 3.0))
            sms = int(rng.uniform(100, 400))
            complaints = int(rng.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15]))
        elif misaligned:
            # Light user on a heavy plan
            data_usage = rng.uniform(1, plan_data_caps[current_plan] * 0.2)
            calls = int(rng.uniform(10, plan_call_caps[current_plan] * 0.15))
            sms = int(rng.uniform(5, 50))
            complaints = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15]))
        else:
            # Aligned user
            data_usage = rng.uniform(plan_data_caps[current_plan] * 0.3,
                                     plan_data_caps[current_plan] * 0.9)
            calls = int(rng.uniform(plan_call_caps[current_plan] * 0.2,
                                    plan_call_caps[current_plan] * 0.8))
            sms = int(rng.uniform(20, 200))
            complaints = int(rng.choice([0, 0, 0, 1, 1, 2]))

        tenure = int(rng.exponential(18) + 1)
        intl_calls = int(rng.exponential(5))
        monthly_bill = plan_prices[current_plan] + rng.uniform(-5, 15)
        phone = f"+44 7{rng.integers(100, 999)} {rng.integers(100000, 999999)}"

        rows.append({
            "customer_id": f"USR-{i + 1:05d}",
            "phone": phone,
            "current_plan": current_plan,
            "monthly_bill": round(monthly_bill, 2),
            "data_usage_gb": round(data_usage, 1),
            "call_count": calls,
            "sms_count": sms,
            "complaints": complaints,
            "tenure_months": min(tenure, 60),
            "intl_calls": intl_calls,
        })

    return pd.DataFrame(rows)


def generate_banking_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Synthesise a banking dataset with products and balances that feel
    plausible. Misalignment here means: huge balance on a basic checking
    account (upsell to Private Banking), or tiny balance + low activity
    on Private Banking (downsell).
    """
    rng = np.random.default_rng(seed)

    products = ["Checking", "Savings Plus", "Premium Account", "Private Banking"]
    product_fees = {"Checking": 0.0, "Savings Plus": 5.0, "Premium Account": 18.0, "Private Banking": 45.0}

    rows = []
    for i in range(n):
        current = rng.choice(products, p=[0.35, 0.30, 0.25, 0.10])
        misaligned = rng.random() < 0.12

        if current == "Checking":
            balance = rng.uniform(30000, 100000) if misaligned else rng.uniform(500, 15000)
            tx = int(rng.uniform(60, 150)) if misaligned else int(rng.uniform(10, 50))
        elif current == "Private Banking":
            balance = rng.uniform(1000, 10000) if misaligned else rng.uniform(100000, 500000)
            tx = int(rng.uniform(5, 15)) if misaligned else int(rng.uniform(20, 80))
        elif current == "Savings Plus":
            balance = rng.uniform(5000, 50000)
            tx = int(rng.uniform(15, 70))
        else:  # Premium Account
            balance = rng.uniform(20000, 120000)
            tx = int(rng.uniform(25, 90))

        fee = product_fees[current] + rng.uniform(-2, 5)
        phone = f"+33 6{rng.integers(10, 99)} {rng.integers(100000, 999999)}"

        rows.append({
            "client_id": f"CLT-{i + 1:05d}",
            "contact_number": phone,
            "current_product": current,
            "monthly_fee": round(max(fee, 0), 2),
            "avg_balance": round(balance, 2),
            "monthly_transactions": tx,
            "digital_logins": int(rng.uniform(5, 60)),
            "support_tickets": int(rng.choice([0, 0, 0, 1, 1, 2, 3])),
            "account_age_months": int(rng.exponential(24) + 1),
            "international_transfers": int(rng.exponential(2)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    telecom = generate_telecom_dataset()
    telecom_path = os.path.join(HERE, "sample_telecom_demo.csv")
    telecom.to_csv(telecom_path, index=False)
    print(f"Telecom demo:  {len(telecom)} rows, {len(telecom.columns)} columns -> {telecom_path}")

    banking = generate_banking_dataset()
    banking_path = os.path.join(HERE, "sample_banking_demo.csv")
    banking.to_csv(banking_path, index=False)
    print(f"Banking demo:  {len(banking)} rows, {len(banking.columns)} columns -> {banking_path}")

    print("\nTelecom columns:", list(telecom.columns))
    print("Banking columns:", list(banking.columns))


if __name__ == "__main__":
    main()
