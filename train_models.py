"""
Fraud model: LightGBM, XGBoost and CatBoost with 5-fold stratified CV, then a blend.

Usage (run from the folder with train.csv / test.csv / sample_submission.csv):
    python train_models.py                         # all three models + blend
    python train_models.py --models lgb            # just LightGBM (fastest, good first run)
    python train_models.py --original original.csv # also train on the original dataset
    python train_models.py --data path/to/data

Install once:
    pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy

Outputs (in the data folder):
    submission_<model>.csv   one submission per model
    submission_blend.csv     the ensemble (upload this one if its CV score is best)
    oof_<model>.npy / test_<model>.npy   saved predictions, so later blends don't need retraining

Note: assumes the metric is ROC AUC. Log loss is also printed. If the competition uses a
different metric, change `score()` below.
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "is_fraud"
SEED = 42
N_FOLDS = 5

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=".", help="folder with the competition CSVs")
parser.add_argument("--models", default="lgb,xgb,cat", help="comma list from: lgb,xgb,cat")
parser.add_argument("--original", default=None,
                    help="optional CSV of the original dataset; must have an is_fraud column "
                         "and the same feature names (only shared columns are used)")
args = parser.parse_args()
data_dir = Path(args.data)
models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]


def score(y, p):
    return roc_auc_score(y, p)


# ---------------------------------------------------------------- data
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sub = pd.read_csv(data_dir / "sample_submission.csv")

original = None
if args.original:
    original = pd.read_csv(args.original)
    if TARGET not in original.columns:
        raise SystemExit(f"--original file has no '{TARGET}' column; rename its label column first")
    missing = [c for c in test.columns if c != "id" and c not in original.columns]
    if missing:
        print(f"[original] these features are missing and will be NaN: {missing}")
    original = original.reindex(columns=[c for c in train.columns if c != "id"])
    original[TARGET] = original[TARGET].astype(float)
    print(f"[original] {len(original):,} rows, fraud rate {original[TARGET].mean():.4f}")


def add_features(df):
    df = df.copy()
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["log_amount"] = np.log1p(df["amount_usd"])
    flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    df["n_risk_flags"] = df[[f for f in flags if f in df.columns]].astype(float).sum(axis=1)
    df["is_night"] = df["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    return df


FEATURES = [c for c in test.columns if c != "id"]
CAT_COLS = [c for c in FEATURES if train[c].dtype == object or str(train[c].dtype) in ("str", "string")]
BOOL_COLS = [c for c in FEATURES if train[c].dtype == bool]


def prepare(df):
    df = add_features(df)
    for c in BOOL_COLS:
        df[c] = df[c].astype(float)  # float so NaN from the original dataset is allowed
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c], categories=sorted(train[c].dropna().unique()))
    return df


X = prepare(train[FEATURES])
X_test = prepare(test[FEATURES])
y = train[TARGET].astype(int).values
FEATURES = list(X.columns)
if original is not None:
    X_orig = prepare(original[[c for c in original.columns if c != TARGET]])[FEATURES]
    y_orig = original[TARGET].astype(int).values

print(f"train {X.shape}, test {X_test.shape}, fraud rate {y.mean():.4f}")
print(f"categorical: {CAT_COLS}")

# ---------------------------------------------------------------- models
pos_weight = 1.0  # AUC doesn't need class weights; keeps probabilities calibrated for log loss


def fit_lgb(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        scale_pos_weight=pos_weight, random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m, lambda d: m.predict_proba(d)[:, 1], m.best_iteration_


def fit_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=5000, learning_rate=0.03, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
        early_stopping_rounds=200, scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, lambda d: m.predict_proba(d)[:, 1], m.best_iteration


def fit_cat(Xtr, ytr, Xva, yva):
    from catboost import CatBoostClassifier

    def to_cat(d):
        d = d.copy()
        for c in CAT_COLS:
            d[c] = d[c].astype(str)
        return d

    m = CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="AUC",
        od_type="Iter", od_wait=200, random_seed=SEED, verbose=False, thread_count=-1)
    m.fit(to_cat(Xtr), ytr, cat_features=CAT_COLS, eval_set=(to_cat(Xva), yva), use_best_model=True)
    return m, lambda d: m.predict_proba(to_cat(d))[:, 1], m.get_best_iteration()


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}

# ---------------------------------------------------------------- CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))
oof, test_pred = {}, {}

for name in models_to_run:
    print(f"\n===== {name} =====")
    oof[name] = np.zeros(len(X))
    test_pred[name] = np.zeros(len(X_test))
    t0 = time.time()
    for k, (tr, va) in enumerate(folds):
        Xtr, ytr = X.iloc[tr], y[tr]
        if original is not None:  # original data only ever goes into training, never validation
            Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
            ytr = np.concatenate([ytr, y_orig])
        model, predict, best_iter = FITTERS[name](Xtr, ytr, X.iloc[va], y[va])
        oof[name][va] = predict(X.iloc[va])
        test_pred[name] += predict(X_test) / N_FOLDS
        print(f"  fold {k + 1}: AUC {score(y[va], oof[name][va]):.5f}  (trees {best_iter})")
    print(f"  {name} CV AUC {score(y, oof[name]):.5f}  logloss {log_loss(y, oof[name]):.5f}"
          f"  [{time.time() - t0:.0f}s]")

    np.save(data_dir / f"oof_{name}.npy", oof[name])
    np.save(data_dir / f"test_{name}.npy", test_pred[name])
    sub[TARGET] = test_pred[name]
    sub.to_csv(data_dir / f"submission_{name}.csv", index=False)

    if name == "lgb":
        imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
        print("\n  LightGBM feature importance (gain, last fold, top 15):")
        print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())

# ---------------------------------------------------------------- blend
if len(oof) > 1:
    print("\n===== blend =====")
    names = list(oof)
    print("  correlation of OOF predictions (lower = more useful to blend):")
    print(pd.DataFrame(oof).corr().round(4).to_string())

    # Greedy hill-climbing (Caruana ensemble selection) on OOF AUC, with replacement.
    counts = {n: 0 for n in names}
    best_name = max(names, key=lambda n: score(y, oof[n]))
    counts[best_name] = 1
    current = oof[best_name].copy()
    best = score(y, current)
    for _ in range(50):
        total = sum(counts.values())
        trials = {n: score(y, (current * total + oof[n]) / (total + 1)) for n in names}
        n_best = max(trials, key=trials.get)
        if trials[n_best] <= best + 1e-6:
            break
        counts[n_best] += 1
        best = trials[n_best]
        current = (current * total + oof[n_best]) / (total + 1)
    weights = {n: c / sum(counts.values()) for n, c in counts.items()}

    equal = np.mean([oof[n] for n in names], axis=0)
    print(f"\n  equal-weight average  CV AUC {score(y, equal):.5f}")
    print(f"  hill-climbed weights  CV AUC {best:.5f}  weights {({k: round(v, 3) for k, v in weights.items()})}")

    sub[TARGET] = sum(weights[n] * test_pred[n] for n in names)
    sub.to_csv(data_dir / "submission_blend.csv", index=False)

print("\nSummary:")
for n in oof:
    print(f"  {n:6s} CV AUC {score(y, oof[n]):.5f}")
print(f"Submissions written to {data_dir.resolve()}")
