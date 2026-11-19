"""Look for hidden links between the competition data and the original dataset.
Usage: python diagnose.py credit_card_fraud_2026.csv      (run in the folder with train.csv/test.csv)"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = "is_fraud"
train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
orig = pd.read_csv(sys.argv[1])
feats = [c for c in test.columns if c != "id"]
y = train[T].astype(int).values
print(f"train {len(train):,}  test {len(test):,}  original {len(orig):,}")

# 1. Does row order / id carry signal?
a = roc_auc_score(y, train["id"])
print(f"\n[1] AUC of id alone: {max(a, 1 - a):.4f}   (0.5 = no signal)")


def key(df):
    return df[feats].astype(str).agg("|".join, axis=1)


# 2. Exact copies of whole rows
ok, tk, sk = key(orig), key(train), key(test)
olab = pd.Series(orig[T].astype(int).values, index=ok).groupby(level=0).mean()
m = tk.isin(olab.index)
print(f"\n[2] train rows identical to an original row: {m.sum():,} ({m.mean():.2%})")
print(f"    test rows identical to an original row:  {sk.isin(olab.index).sum():,} ({sk.isin(olab.index).mean():.2%})")
if m.sum():
    agree = ((tk[m].map(olab) > 0.5).astype(int).values == y[m.values]).mean()
    print(f"    label agreement on those train rows: {agree:.2%}")
print(f"    duplicate rows within train: {tk.duplicated().sum():,}   train rows also in test: {tk.isin(set(sk)).sum():,}")

# 3. Per column: do exact values come from the original, and does the original's fraud
#    rate for that exact value predict fraud in train?
print("\n[3] per column (sorted by orig_TE_auc)")
rows = []
for c in feats:
    g = orig.groupby(c)[T].agg(["mean", "count"])
    hit = train[c].isin(g.index)
    te = train[c].map(g["mean"]).fillna(orig[T].mean()).values
    raw = roc_auc_score(y, train[c].astype(float)) if pd.api.types.is_numeric_dtype(train[c]) or train[c].dtype == bool else np.nan
    te_auc = roc_auc_score(y, te)
    rows.append({"column": c, "n_unique_train": train[c].nunique(), "n_unique_orig": orig[c].nunique(),
                 "pct_train_values_in_orig": round(hit.mean() * 100, 1),
                 "raw_auc": round(max(raw, 1 - raw), 4) if raw == raw else None,
                 "orig_TE_auc": round(max(te_auc, 1 - te_auc), 4)})
print(pd.DataFrame(rows).sort_values("orig_TE_auc", ascending=False).to_string(index=False))

# 4. Rows whose amount matches an original FRAUD row
fraud_amts = set(orig.loc[orig[T] == 1, "amount_usd"])
legit_amts = set(orig.loc[orig[T] == 0, "amount_usd"])
for label, s in [("only fraud rows", fraud_amts - legit_amts), ("only legit rows", legit_amts - fraud_amts)]:
    mm = train["amount_usd"].isin(s)
    print(f"\n[4] train rows whose amount_usd appears in original {label}: {mm.sum():,}, fraud rate {y[mm.values].mean() if mm.sum() else float('nan'):.4f}")
print(f"    overall train fraud rate {y.mean():.4f}")
