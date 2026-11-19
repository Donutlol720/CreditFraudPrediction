"""Do exact repeated values (within train) carry fraud signal that trees can't see?
Usage: python diagnose2.py      (run in the folder with train.csv/test.csv)"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T = "is_fraud"
train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
feats = [c for c in test.columns if c != "id"]
y = train[T].astype(int).values
prior = y.mean()
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))
both = pd.concat([train[feats], test[feats]])


def auc(p):
    a = roc_auc_score(y, p)
    return max(a, 1 - a)


def oof_te(key, smooth=20):
    """Fraud rate of each exact value, computed only from the other folds (no leakage)."""
    out = np.zeros(len(train))
    for tr, va in folds:
        g = pd.DataFrame({"k": key.iloc[tr].values, "y": y[tr]}).groupby("k")["y"].agg(["sum", "count"])
        rate = (g["sum"] + prior * smooth) / (g["count"] + smooth)
        out[va] = key.iloc[va].map(rate).fillna(prior).values
    return out


rows = []
for c in feats:
    key = train[c].astype(str)
    cnt = train[c].map(both[c].value_counts()).values
    raw = auc(train[c].astype(float)) if pd.api.types.is_numeric_dtype(train[c]) or train[c].dtype == bool else np.nan
    rows.append({"column": c, "n_unique": train[c].nunique(),
                 "rows_per_value": round(len(train) / train[c].nunique(), 1),
                 "raw_auc": round(raw, 4) if raw == raw else None,
                 "count_auc": round(auc(cnt), 4),
                 "oof_TE_auc": round(auc(oof_te(key)), 4)})
    print(f"  done {c}", flush=True)
print("\n[A] single columns (sorted by oof_TE_auc - raw_auc)")
df = pd.DataFrame(rows)
df["TE_minus_raw"] = (df["oof_TE_auc"] - df["raw_auc"].fillna(df["oof_TE_auc"])).round(4)
print(df.sort_values("TE_minus_raw", ascending=False).to_string(index=False))

# A few combinations that could identify the same customer/card
print("\n[B] column combinations (could identify a repeat customer/card)")
combos = [["customer_age", "card_age_months"],
          ["customer_age", "card_age_months", "card_type"],
          ["account_balance_usd", "customer_age"],
          ["card_type", "card_age_months", "customer_age", "device_type"],
          ["merchant_category", "amount_usd"]]
for cols in combos:
    key = train[cols].astype(str).agg("|".join, axis=1)
    n = key.nunique()
    print(f"  {' + '.join(cols):60s} unique {n:>8,}  rows/value {len(train) / n:7.1f}  oof_TE_auc {auc(oof_te(key)):.4f}")

# How strong is a simple stacked check: does adding TE of all columns beat raw features?
print("\nFor reference, your best model CV AUC is 0.947.")
