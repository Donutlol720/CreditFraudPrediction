"""
Print an overview of the Kaggle competition files (train.csv, test.csv, sample_submission.csv).

Usage:
    python inspect_data.py                # looks for the CSVs in the current folder
    python inspect_data.py path/to/data   # or in the folder you give it

Also saves everything it prints to data_overview.txt in that folder, so you can paste it back.
"""
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out_lines = []


def show(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    out_lines.append(text)


def section(title):
    show("\n" + "=" * 80)
    show(title)
    show("=" * 80)


def load(name):
    path = data_dir / name
    if not path.exists():
        show(f"[missing] {path}")
        return None
    return pd.read_csv(path)


files = sorted(p.name for p in data_dir.glob("*.csv"))
section(f"CSV files in {data_dir.resolve()}")
show(files if files else "none found")

train = load("train.csv")
test = load("test.csv")
sub = load("sample_submission.csv")

for name, df in [("train.csv", train), ("test.csv", test), ("sample_submission.csv", sub)]:
    if df is None:
        continue
    section(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    show("\nColumns, types, missing values, unique values:")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "example": df.iloc[0],
    })
    show(info.to_string())

    show("\nFirst 5 rows:")
    show(df.head().to_string())

    num = df.select_dtypes("number")
    if not num.empty:
        show("\nNumeric summary:")
        show(num.describe().T.to_string())

    cat = df.select_dtypes(exclude="number")
    for col in cat.columns:
        show(f"\nTop values of '{col}' ({df[col].nunique()} unique):")
        show(df[col].value_counts(dropna=False).head(15).to_string())

# Compare train vs test to find the target column
if train is not None and test is not None:
    section("train vs test")
    only_train = [c for c in train.columns if c not in test.columns]
    only_test = [c for c in test.columns if c not in train.columns]
    show("Columns only in train (likely the target):", only_train)
    show("Columns only in test:", only_test)

    for target in only_train:
        section(f"Target '{target}' distribution in train")
        vc = train[target].value_counts(dropna=False)
        show(pd.DataFrame({"count": vc, "pct": (vc / len(train) * 100).round(3)}).to_string())

        # Quick look at which features relate to the target
        if train[target].nunique() == 2:
            y = (train[target] == train[target].max()).astype(int)
            num_cols = [c for c in train.select_dtypes("number").columns if c not in (target, "id")]
            if num_cols:
                show(f"\nCorrelation of numeric features with '{target}':")
                corr = train[num_cols].corrwith(y).sort_values(key=abs, ascending=False)
                show(corr.round(4).to_string())
            cat_cols = [c for c in train.select_dtypes(exclude="number").columns if c != target]
            for col in cat_cols:
                if train[col].nunique() <= 30:
                    show(f"\nFraud rate by '{col}':")
                    rate = y.groupby(train[col]).agg(["mean", "count"])
                    show(rate.sort_values("mean", ascending=False).round(4).to_string())

    shared_cat = [c for c in test.select_dtypes(exclude="number").columns if c in train.columns]
    for col in shared_cat:
        unseen = set(test[col].dropna()) - set(train[col].dropna())
        if unseen:
            show(f"\n'{col}': {len(unseen)} values in test that never appear in train")

out_path = data_dir / "data_overview.txt"
out_path.write_text("\n".join(out_lines))
print(f"\nSaved this output to {out_path.resolve()}")
