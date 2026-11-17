"""
Fraud model: LightGBM, XGBoost, CatBoost and a neural net with 5-fold stratified CV, then a blend.

Usage (run from the folder with train.csv / test.csv / sample_submission.csv):
    python train_models.py --models lgb                  # one model
    python train_models.py --models lgb,xgb,cat,nn       # several (default: lgb,xgb,cat)
    python train_models.py --models lgb --catnum         # treat small-integer columns as categories
    python train_models.py --models lgb --seeds 3        # average 3 random seeds (slower, steadier)
    python train_models.py --models lgb --tag v2         # save as oof_lgb_v2.npy etc. (keeps old runs)
    python train_models.py --original original.csv       # also train on the original dataset
    python train_models.py --blend-only                  # blend every saved oof_*.npy, no training

Every run saves its out-of-fold and test predictions. All runs use the same 5 folds, so
--blend-only can combine models from different runs (e.g. lgb from one run, nn from another).

Install:
    conda install -c conda-forge pandas numpy scipy scikit-learn lightgbm xgboost catboost
    conda install -c conda-forge pytorch          # only needed for --models nn

Assumes the metric is ROC AUC (log loss is printed too). Change `score()` if it's something else.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

TARGET = "is_fraud"
FOLD_SEED = 42  # never change: keeps folds identical across runs so blends are valid
N_FOLDS = 5
CATNUM_COLS = ["cvv_retry_count", "txn_count_last_24h", "prior_disputes",
               "day_of_week", "time_of_day_hour"]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=".", help="folder with the competition CSVs")
parser.add_argument("--models", default="lgb,xgb,cat", help="comma list from: lgb,xgb,cat,nn")
parser.add_argument("--seeds", type=int, default=1, help="random seeds to average per model")
parser.add_argument("--catnum", action="store_true",
                    help=f"also treat {CATNUM_COLS} as categorical (tree models)")
parser.add_argument("--tag", default="", help="suffix for saved files, e.g. v2")
parser.add_argument("--original", default=None,
                    help="optional CSV of the original dataset with an is_fraud column")
parser.add_argument("--blend-only", action="store_true", help="skip training, blend saved runs")
args = parser.parse_args()
data_dir = Path(args.data)
suffix = f"_{args.tag}" if args.tag else ""


def score(y, p):
    return roc_auc_score(y, p)


# ---------------------------------------------------------------- data
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sub = pd.read_csv(data_dir / "sample_submission.csv")
y = train[TARGET].astype(int).values
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train, y))


def blend(oof, test_pred, out_name):
    names = list(oof)
    print("\n===== blend =====")
    for n in names:
        print(f"  {n:16s} CV AUC {score(y, oof[n]):.5f}")
    print("\n  correlation of OOF predictions (lower = more useful to blend):")
    print(pd.DataFrame(oof).corr().round(4).to_string())

    # Rank-normalise so models with different probability scales blend fairly (AUC only
    # cares about order). Then greedy hill-climbing (Caruana ensemble selection).
    def rank(a):
        return pd.Series(a).rank(pct=True).values

    r_oof = {n: rank(oof[n]) for n in names}
    r_test = {n: rank(test_pred[n]) for n in names}
    counts = {n: 0 for n in names}
    first = max(names, key=lambda n: score(y, oof[n]))
    counts[first] = 1
    current, best = r_oof[first].copy(), score(y, oof[first])
    for _ in range(100):
        total = sum(counts.values())
        trials = {n: score(y, (current * total + r_oof[n]) / (total + 1)) for n in names}
        nb = max(trials, key=trials.get)
        if trials[nb] <= best + 1e-6:
            break
        counts[nb] += 1
        best = trials[nb]
        current = (current * total + r_oof[nb]) / (total + 1)
    w = {n: c / sum(counts.values()) for n, c in counts.items()}

    print(f"\n  equal-weight rank average  CV AUC {score(y, np.mean(list(r_oof.values()), axis=0)):.5f}")
    print(f"  hill-climbed blend         CV AUC {best:.5f}")
    print("  weights:", {k: round(v, 3) for k, v in w.items() if v > 0})
    # Rank blend is right for AUC. NOTE: these are ranks in [0,1], not calibrated
    # probabilities, so don't use this file if the metric turns out to be log loss.
    sub[TARGET] = sum(w[n] * r_test[n] for n in names)
    sub.to_csv(data_dir / out_name, index=False)
    print(f"  wrote {out_name}")


if args.blend_only:
    oof, test_pred = {}, {}
    for f in sorted(data_dir.glob("oof_*.npy")):
        name = f.stem[len("oof_"):]
        tf = data_dir / f"test_{name}.npy"
        if tf.exists():
            oof[name], test_pred[name] = np.load(f), np.load(tf)
    if len(oof) < 2:
        raise SystemExit(f"need at least 2 saved runs to blend, found: {list(oof)}")
    blend(oof, test_pred, "submission_blend.csv")
    raise SystemExit

original = None
if args.original:
    original = pd.read_csv(args.original)
    if TARGET not in original.columns:
        raise SystemExit(f"--original file has no '{TARGET}' column; rename its label column first")
    missing = [c for c in test.columns if c != "id" and c not in original.columns]
    if missing:
        print(f"[original] these features are missing and will be NaN: {missing}")
    original = original.reindex(columns=[c for c in train.columns if c != "id"])
    print(f"[original] {len(original):,} rows, fraud rate {original[TARGET].astype(float).mean():.4f}")


def add_features(df):
    df = df.copy()
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["log_amount"] = np.log1p(df["amount_usd"])
    flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    df["n_risk_flags"] = df[[f for f in flags if f in df.columns]].astype(float).sum(axis=1)
    df["is_night"] = df["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    return df


BASE = [c for c in test.columns if c != "id"]
CAT_COLS = [c for c in BASE if not pd.api.types.is_numeric_dtype(train[c]) and train[c].dtype != bool]
BOOL_COLS = [c for c in BASE if train[c].dtype == bool]
TREE_CATS = CAT_COLS + (CATNUM_COLS if args.catnum else [])


def prepare(df):
    df = add_features(df)
    for c in BOOL_COLS:
        df[c] = df[c].astype(float)  # float so NaN from the original dataset is allowed
    if args.catnum:  # keep the numeric version too, add a categorical copy
        for c in CATNUM_COLS:
            df[c + "_cat"] = df[c]
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c], categories=sorted(train[c].dropna().unique()))
    if args.catnum:
        for c in CATNUM_COLS:
            vals = sorted(pd.concat([train[c], test[c]]).unique())
            df[c + "_cat"] = pd.Categorical(df[c + "_cat"], categories=vals)
    return df


X = prepare(train[BASE])
X_test = prepare(test[BASE])
FEATURES = list(X.columns)
CAT_FEATS = [c for c in FEATURES if isinstance(X[c].dtype, pd.CategoricalDtype)]
if original is not None:
    X_orig = prepare(original[BASE])[FEATURES]
    y_orig = original[TARGET].astype(int).values

print(f"train {X.shape}, test {X_test.shape}, fraud rate {y.mean():.4f}")
print(f"categorical: {CAT_FEATS}")


# ---------------------------------------------------------------- models
# Tuned toward simpler, more regularised trees: the first run stopped at ~400 trees,
# a sign the patterns are simple and deeper trees start fitting noise.
def fit_lgb(Xtr, ytr, Xva, yva, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=10000, learning_rate=0.02, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5,
        cat_smooth=20, min_data_per_group=200, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return m, lambda d: m.predict_proba(d)[:, 1], m.best_iteration_


def fit_xgb(Xtr, ytr, Xva, yva, seed):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=10000, learning_rate=0.02, max_depth=5, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
        early_stopping_rounds=300, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, lambda d: m.predict_proba(d)[:, 1], m.best_iteration


def fit_cat(Xtr, ytr, Xva, yva, seed):
    from catboost import CatBoostClassifier

    def to_cat(d):
        d = d.copy()
        for c in CAT_FEATS:
            d[c] = d[c].astype(str)
        return d

    m = CatBoostClassifier(
        iterations=10000, learning_rate=0.04, depth=6, l2_leaf_reg=5, eval_metric="AUC",
        od_type="Iter", od_wait=300, random_seed=seed, verbose=False, thread_count=-1)
    m.fit(to_cat(Xtr), ytr, cat_features=CAT_FEATS, eval_set=(to_cat(Xva), yva),
          use_best_model=True)
    return m, lambda d: m.predict_proba(to_cat(d))[:, 1], m.get_best_iteration()


# --- neural net: embeddings for categories + MLP on quantile-normalised numbers
NN_CACHE = {}


def nn_inputs():
    """Numeric matrix (quantile -> normal) and integer category codes for train, test, original."""
    if NN_CACHE:
        return NN_CACHE
    from sklearn.preprocessing import QuantileTransformer
    num_cols = [c for c in FEATURES if c not in CAT_FEATS]
    frames = {"train": X, "test": X_test}
    if original is not None:
        frames["orig"] = X_orig
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200_000, random_state=0)
    qt.fit(pd.concat([X[num_cols], X_test[num_cols]]).astype(float))
    for k, d in frames.items():
        num = qt.transform(d[num_cols].astype(float))
        num = np.nan_to_num(num, nan=0.0).astype(np.float32)
        cats = np.stack([d[c].cat.codes.values + 1 for c in CAT_FEATS], axis=1).astype(np.int64)  # 0 = unknown/NaN
        NN_CACHE[k] = (num, cats)
    NN_CACHE["cards"] = [len(X[c].cat.categories) + 1 for c in CAT_FEATS]
    return NN_CACHE


def fit_nn(tr_idx, va_idx, seed):
    import torch
    import torch.nn as tnn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cpu"
    data = nn_inputs()
    num, cats = data["train"]
    Xn_tr, Xc_tr, y_tr = num[tr_idx], cats[tr_idx], y[tr_idx]
    if original is not None:
        on, oc = data["orig"]
        Xn_tr, Xc_tr = np.vstack([Xn_tr, on]), np.vstack([Xc_tr, oc])
        y_tr = np.concatenate([y_tr, y_orig])

    class Net(tnn.Module):
        def __init__(self, n_num, cards):
            super().__init__()
            self.embs = tnn.ModuleList([tnn.Embedding(c, min(8, (c + 1) // 2)) for c in cards])
            d_in = n_num + sum(e.embedding_dim for e in self.embs)
            self.mlp = tnn.Sequential(
                tnn.Linear(d_in, 256), tnn.BatchNorm1d(256), tnn.SiLU(), tnn.Dropout(0.2),
                tnn.Linear(256, 128), tnn.BatchNorm1d(128), tnn.SiLU(), tnn.Dropout(0.2),
                tnn.Linear(128, 64), tnn.BatchNorm1d(64), tnn.SiLU(), tnn.Dropout(0.1),
                tnn.Linear(64, 1))

        def forward(self, xn, xc):
            e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
            return self.mlp(torch.cat([xn] + e, dim=1)).squeeze(1)

    net = Net(num.shape[1], data["cards"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=1)
    loss_fn = tnn.BCEWithLogitsLoss()

    def predict(n_arr, c_arr):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(n_arr), 16384):
                out.append(torch.sigmoid(net(torch.from_numpy(n_arr[i:i + 16384]),
                                             torch.from_numpy(c_arr[i:i + 16384]))).numpy())
        return np.concatenate(out)

    tn, tc, ty = torch.from_numpy(Xn_tr), torch.from_numpy(Xc_tr), torch.from_numpy(y_tr.astype(np.float32))
    best_auc, best_state, bad, epoch = -1, None, 0, 0
    for epoch in range(1, 31):
        net.train()
        perm = torch.randperm(len(ty))
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            opt.zero_grad()
            loss_fn(net(tn[b], tc[b]), ty[b]).backward()
            opt.step()
        auc = score(y[va_idx], predict(num[va_idx], cats[va_idx]))
        sched.step(auc)
        if auc > best_auc + 1e-5:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    tnum, tcat = data["test"]
    return (lambda which: predict(*((num[va_idx], cats[va_idx]) if which == "val" else (tnum, tcat)))), epoch


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}

# ---------------------------------------------------------------- CV
models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
oof, test_pred = {}, {}
for name in models_to_run:
    key = name + suffix
    print(f"\n===== {key}  (seeds={args.seeds}{', catnum' if args.catnum else ''}) =====")
    oof[key] = np.zeros(len(X))
    test_pred[key] = np.zeros(len(X_test))
    t0 = time.time()
    for k, (tr, va) in enumerate(folds):
        fold_va = np.zeros(len(va))
        for s in range(args.seeds):
            seed = 1000 + s
            if name == "nn":
                predict, n_iter = fit_nn(tr, va, seed)
                fold_va += predict("val") / args.seeds
                test_pred[key] += predict("test") / (args.seeds * N_FOLDS)
                continue
            Xtr, ytr = X.iloc[tr], y[tr]
            if original is not None:  # original data only ever goes into training, never validation
                Xtr = pd.concat([Xtr, X_orig], ignore_index=True)
                ytr = np.concatenate([ytr, y_orig])
            model, predict, n_iter = FITTERS[name](Xtr, ytr, X.iloc[va], y[va], seed)
            fold_va += predict(X.iloc[va]) / args.seeds
            test_pred[key] += predict(X_test) / (args.seeds * N_FOLDS)
        oof[key][va] = fold_va
        unit = "epochs" if name == "nn" else "trees"
        print(f"  fold {k + 1}: AUC {score(y[va], fold_va):.5f}  ({unit} {n_iter}, last seed)")
    print(f"  {key} CV AUC {score(y, oof[key]):.5f}  logloss {log_loss(y, np.clip(oof[key], 1e-7, 1 - 1e-7)):.5f}"
          f"  [{time.time() - t0:.0f}s]")

    np.save(data_dir / f"oof_{key}.npy", oof[key])
    np.save(data_dir / f"test_{key}.npy", test_pred[key])
    sub[TARGET] = test_pred[key]
    sub.to_csv(data_dir / f"submission_{key}.csv", index=False)

    if name == "lgb":
        imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
        print("\n  LightGBM feature importance (gain, last fold, top 15):")
        print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())

if len(oof) > 1:
    blend(oof, test_pred, f"submission_blend{suffix}.csv")

print(f"\nSubmissions written to {data_dir.resolve()}")
print("Tip: python train_models.py --blend-only   blends every saved run together.")
