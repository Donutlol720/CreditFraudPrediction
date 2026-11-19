"""
Fraud models with 5-fold stratified CV, saved out-of-fold predictions, and blending.

Usage (run from the folder with train.csv / test.csv / sample_submission.csv):
    python train_models.py --models lgb --tag v2                 # one model
    python train_models.py --models lgb4,xgb2,glm --dig --tag dig  # shallow models on digit features
    python train_models.py --models lgb --dig --cnt --te --tag dte # + value counts + exact-value encoding
    python train_models.py --blend-only                          # blend every saved run, no training

Models:  lgb (31 leaves)   lgb4 (4 leaves)   xgb (depth 5)   xgb2 (depth 2)
         cat (CatBoost)    nn (neural net)   glm (logistic regression on one-hot features)
         ebm (Explainable Boosting Machine; needs: pip install interpret-core)
         xgbglm (depth-2 XGBoost that starts from the logistic regression's predictions)
Feature options:
    --dig      split numeric columns into digits (leading digits, digits by position, length)
    --cnt      how often each exact value appears (train+test), for many-value columns
    --te       fraud rate of each exact value, computed only from other folds (tree models only)
    --catnum   treat small-integer columns as categories (did not help in v2)
Other:
    --seeds N  average N random seeds      --tag NAME  suffix for saved files
    --original FILE  add the original dataset's rows to training

Every run saves oof_<model>_<tag>.npy and test_<model>_<tag>.npy. All runs use the same 5 folds,
so --blend-only can combine models from different runs.

Assumes the metric is ROC AUC. The blend writes rank scores, which is fine for AUC only.
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
DIGIT_COLS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
              "velocity_score", "merchant_risk_score", "customer_age", "card_age_months"]
CNT_COLS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
            "velocity_score", "merchant_risk_score"]
TE_KEYS = [["account_balance_usd"], ["hours_since_last_txn"], ["distance_from_home_km"],
           ["amount_usd"], ["customer_age"], ["merchant_category", "amount_usd"]]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=".", help="folder with the competition CSVs")
parser.add_argument("--models", default="lgb,xgb,cat",
                    help="comma list from: lgb,lgb4,xgb,xgb2,cat,nn,glm")
parser.add_argument("--seeds", type=int, default=1, help="random seeds to average per model")
parser.add_argument("--dig", action="store_true", help="add digit features")
parser.add_argument("--cnt", action="store_true", help="add exact-value count features")
parser.add_argument("--te", action="store_true", help="add fold-safe exact-value fraud rates (tree models)")
parser.add_argument("--catnum", action="store_true", help=f"also treat {CATNUM_COLS} as categorical")
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
        print(f"  {n:20s} CV AUC {score(y, oof[n]):.5f}")
    print("\n  correlation of OOF predictions (lower = more useful to blend):")
    print(pd.DataFrame(oof).corr().round(3).to_string())

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

BASE = [c for c in test.columns if c != "id"]
CAT_COLS = [c for c in BASE if not pd.api.types.is_numeric_dtype(train[c]) and train[c].dtype != bool]
BOOL_COLS = [c for c in BASE if train[c].dtype == bool]
ALL_RAW = pd.concat([train[BASE], test[BASE]], ignore_index=True)


def n_decimals(s):
    """Smallest number of decimals (0-4) that represents every value exactly."""
    v = s.dropna().values
    for d in range(5):
        if np.allclose(v * 10 ** d, np.round(v * 10 ** d), atol=1e-6):
            return d
    return 4


DECIMALS = {c: n_decimals(ALL_RAW[c]) for c in DIGIT_COLS if c in BASE}
DIGIT_CATS = list(range(-1, 21))


def digit_features(df):
    """Write each number as its digit string (decimal point removed, e.g. 61.16 -> '6116')
    and take: length, 1st and 2nd leading digit, digits at positions 0-3 from the right,
    and number of trailing zeros. All as small categories (-1 = position doesn't exist)."""
    out = {}
    for c, d in DECIMALS.items():
        v = np.round(df[c].astype(float).fillna(-1).values * 10 ** d).astype(np.int64)
        v = np.where(v < 0, 0, v)
        length = np.where(v > 0, np.floor(np.log10(np.maximum(v, 1))).astype(int) + 1, 1)
        out[f"{c}_len"] = length
        out[f"{c}_lead1"] = v // 10 ** (length - 1)
        out[f"{c}_lead2"] = np.where(length >= 2, (v // 10 ** np.maximum(length - 2, 0)) % 10, -1)
        for p in range(4):
            out[f"{c}_p{p}"] = np.where(length > p, (v // 10 ** p) % 10, -1)
        tz = np.zeros(len(v), dtype=int)
        rem = v.copy()
        for _ in range(6):
            z = (rem % 10 == 0) & (rem > 0)
            tz += z
            rem = np.where(z, rem // 10, rem)
        out[f"{c}_tz"] = tz
    return pd.DataFrame(out, index=df.index)


def prepare(df):
    df = df.copy()
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["log_amount"] = np.log1p(df["amount_usd"])
    flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    df["n_risk_flags"] = df[[f for f in flags if f in df.columns]].astype(float).sum(axis=1)
    df["is_night"] = df["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    for c in BOOL_COLS:
        df[c] = df[c].astype(float)  # float so NaN from the original dataset is allowed
    if args.cnt:
        for c in CNT_COLS:
            df[c + "_cnt"] = df[c].map(ALL_RAW[c].value_counts()).fillna(0).values
    new_cats = []
    if args.dig:
        dg = digit_features(df)
        df = pd.concat([df, dg], axis=1)
        new_cats += list(dg.columns)
    if args.catnum:
        for c in CATNUM_COLS:
            df[c + "_cat"] = df[c]
            new_cats.append(c + "_cat")
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c], categories=sorted(train[c].dropna().unique()))
    for c in new_cats:
        cats = DIGIT_CATS if args.dig and c not in [k + "_cat" for k in CATNUM_COLS] else \
            sorted(ALL_RAW[c[:-4]].unique())
        df[c] = pd.Categorical(df[c], categories=cats)
    return df


X = prepare(train[BASE])
X_test = prepare(test[BASE])
FEATURES = list(X.columns)
CAT_FEATS = [c for c in FEATURES if isinstance(X[c].dtype, pd.CategoricalDtype)]
if original is not None:
    X_orig = prepare(original[BASE])[FEATURES]
    y_orig = original[TARGET].astype(int).values

print(f"train {X.shape}, test {X_test.shape}, fraud rate {y.mean():.4f}")
print(f"options: dig={args.dig} cnt={args.cnt} te={args.te} catnum={args.catnum} seeds={args.seeds}")
print(f"{len(FEATURES)} features, {len(CAT_FEATS)} categorical")


# ---------------------------------------------------------------- fold-safe exact-value fraud rates
def te_keys(df):
    return [df[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)
            for cols in TE_KEYS]


if args.te:
    KEYS_TR, KEYS_TE = te_keys(train), te_keys(test)
    KEYS_OR = te_keys(original) if original is not None else None


def te_rate(keys_fit, y_fit, keys_apply, prior, smooth=20):
    g = pd.DataFrame({"k": keys_fit.values, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    rate = (g["sum"] + prior * smooth) / (g["count"] + smooth)
    return keys_apply.map(rate).fillna(prior).values


def add_te(tr, va, Xtr, Xva, Xte, Xor):
    """Train rows get rates from inner folds of the training part; validation/test/original rows
    get rates from the whole training part. The validation fold's labels are never used."""
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    Xor = Xor.copy() if Xor is not None else None
    ytr = y[tr]
    prior = ytr.mean()
    inner = list(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, ytr))
    for cols, ktr_all, kte in zip(TE_KEYS, KEYS_TR, KEYS_TE):
        name = "te_" + "_".join(cols)
        ktr = ktr_all.iloc[tr]
        inner_vals = np.zeros(len(tr))
        for a, b in inner:
            inner_vals[b] = te_rate(ktr.iloc[a], ytr[a], ktr.iloc[b], prior)
        Xtr[name] = inner_vals
        Xva[name] = te_rate(ktr, ytr, ktr_all.iloc[va], prior)
        Xte[name] = te_rate(ktr, ytr, kte, prior)
        if Xor is not None:
            kor = KEYS_OR[TE_KEYS.index(cols)]
            Xor[name] = te_rate(ktr, ytr, kor, prior)
    return Xtr, Xva, Xte, Xor


# ---------------------------------------------------------------- tree models
def fit_lgb(Xtr, ytr, Xva, yva, seed, leaves=31):
    import lightgbm as lgb
    shallow = leaves <= 8
    m = lgb.LGBMClassifier(
        n_estimators=30000, learning_rate=0.05 if shallow else 0.02, num_leaves=leaves,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        reg_lambda=5.0, reg_alpha=0.5, cat_smooth=20, min_data_per_group=200, max_cat_to_onehot=12,
        random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return m, lambda d: m.predict_proba(d)[:, 1], m.best_iteration_


def fit_xgb(Xtr, ytr, Xva, yva, seed, depth=5, margins=None):
    """margins=(train, val) logits to start boosting from (used by xgbglm)."""
    import xgboost as xgb
    shallow = depth <= 3
    m = xgb.XGBClassifier(
        n_estimators=30000, learning_rate=0.05 if shallow else 0.02, max_depth=depth,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5,
        tree_method="hist", enable_categorical=True, max_cat_to_onehot=12, eval_metric="auc",
        early_stopping_rounds=300, random_state=seed, n_jobs=-1)
    if margins is None:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, lambda d: m.predict_proba(d)[:, 1], m.best_iteration
    m.fit(Xtr, ytr, base_margin=margins[0], eval_set=[(Xva, yva)],
          base_margin_eval_set=[margins[1]], verbose=False)
    return m, lambda d, mg: m.predict_proba(d, base_margin=mg)[:, 1], m.best_iteration


def fit_ebm(Xtr, ytr, Xva, yva, seed):
    """Explainable Boosting Machine: one small boosted model per feature, added together,
    plus 10 automatically chosen feature pairs. Uses its own internal validation split."""
    from interpret.glassbox import ExplainableBoostingClassifier
    cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]

    def to_ebm(d):
        d = d.copy()
        for c in cats:
            d[c] = d[c].astype(str)
        return d

    types = ["nominal" if c in cats else "continuous" for c in Xtr.columns]
    m = ExplainableBoostingClassifier(feature_types=types, interactions=10, outer_bags=4,
                                      max_bins=256, random_state=seed, n_jobs=-1)
    m.fit(to_ebm(Xtr), ytr)
    return m, lambda d: m.predict_proba(to_ebm(d))[:, 1], "-"


def fit_cat(Xtr, ytr, Xva, yva, seed):
    from catboost import CatBoostClassifier
    cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]

    def to_cat(d):
        d = d.copy()
        for c in cats:
            d[c] = d[c].astype(str)
        return d

    m = CatBoostClassifier(
        iterations=10000, learning_rate=0.04, depth=6, l2_leaf_reg=5, eval_metric="AUC",
        od_type="Iter", od_wait=300, random_seed=seed, verbose=False, thread_count=-1)
    m.fit(to_cat(Xtr), ytr, cat_features=cats, eval_set=(to_cat(Xva), yva), use_best_model=True)
    return m, lambda d: m.predict_proba(to_cat(d))[:, 1], m.get_best_iteration()


TREE_FITTERS = {"lgb": fit_lgb, "lgb4": lambda *a: fit_lgb(*a, leaves=4),
                "xgb": fit_xgb, "xgb2": lambda *a: fit_xgb(*a, depth=2), "cat": fit_cat,
                "ebm": fit_ebm, "xgbglm": None}  # xgbglm is handled in the CV loop

# ---------------------------------------------------------------- GLM (logistic regression)
GLM_CACHE = {}


def glm_inputs():
    """One-hot of every feature: categories as-is, numbers with <=100 values as-is,
    other numbers cut into 50 quantile bins. Fitted on train+test (no labels used)."""
    if GLM_CACHE:
        return GLM_CACHE
    from sklearn.preprocessing import OneHotEncoder
    frames = {"train": X, "test": X_test}
    if original is not None:
        frames["orig"] = X_orig
    both = pd.concat([X, X_test], ignore_index=True)
    edges = {}
    for c in FEATURES:
        if c not in CAT_FEATS and both[c].nunique() > 100:
            edges[c] = np.unique(np.quantile(both[c].astype(float).dropna(), np.linspace(0, 1, 51)[1:-1]))

    def to_str(d):
        out = {}
        for c in FEATURES:
            if c in edges:
                out[c] = np.searchsorted(edges[c], d[c].astype(float).values).astype(str)
            else:
                out[c] = d[c].astype(str).values
        return pd.DataFrame(out)

    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, dtype=np.float32)
    enc.fit(to_str(both))
    for k, d in frames.items():
        GLM_CACHE[k] = enc.transform(to_str(d)).tocsr()
    print(f"  glm: {GLM_CACHE['train'].shape[1]} one-hot columns")
    return GLM_CACHE


def fit_glm(tr_idx, va_idx, seed):
    from scipy.sparse import vstack
    from sklearn.linear_model import LogisticRegression
    data = glm_inputs()
    Xtr, ytr = data["train"][tr_idx], y[tr_idx]
    if original is not None:
        Xtr, ytr = vstack([Xtr, data["orig"]]).tocsr(), np.concatenate([ytr, y_orig])
    m = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000)
    m.fit(Xtr, ytr)
    return (lambda which: m.predict_proba(data["train"][va_idx] if which == "val" else data["test"])[:, 1]), m.n_iter_[0]


def glm_margins(tr, va):
    """Logistic-regression logits for xgbglm. Training rows get them from inner folds of the
    training part (so XGBoost sees honest starting points); validation/test/original rows
    from a regression fitted on the whole training part."""
    from sklearn.linear_model import LogisticRegression
    data = glm_inputs()
    A, ytr = data["train"][tr], y[tr]

    def fit(Xm, ym):
        return LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000).fit(Xm, ym)

    m_tr = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(tr, ytr):
        m_tr[b] = fit(A[a], ytr[a]).decision_function(A[b])
    full = fit(A, ytr)
    m_or = full.decision_function(data["orig"]) if original is not None else None
    return m_tr, full.decision_function(data["train"][va]), full.decision_function(data["test"]), m_or


# ---------------------------------------------------------------- neural net
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

    net = Net(num.shape[1], data["cards"])
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


INDEX_FITTERS = {"nn": fit_nn, "glm": fit_glm}  # these take row indices, not DataFrames

# ---------------------------------------------------------------- CV
models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
unknown = [m for m in models_to_run if m not in TREE_FITTERS and m not in INDEX_FITTERS]
if unknown:
    raise SystemExit(f"unknown models {unknown}; choose from {list(TREE_FITTERS) + list(INDEX_FITTERS)}")
if args.te and any(m in INDEX_FITTERS for m in models_to_run):
    print("note: --te is used by tree models only (nn and glm ignore it)")

oof, test_pred = {}, {}
for name in models_to_run:
    key = name + suffix
    print(f"\n===== {key} =====")
    oof[key] = np.zeros(len(X))
    test_pred[key] = np.zeros(len(X_test))
    t0 = time.time()
    model = None
    for k, (tr, va) in enumerate(folds):
        fold_va = np.zeros(len(va))
        if name in TREE_FITTERS:
            Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X_test
            Xor = X_orig if original is not None else None
            if args.te:
                Xtr, Xva, Xte, Xor = add_te(tr, va, Xtr, Xva, Xte, Xor)
            ytr = y[tr]
            if original is not None:  # original data only ever goes into training, never validation
                Xtr = pd.concat([Xtr, Xor], ignore_index=True)
                ytr = np.concatenate([ytr, y_orig])
        for s in range(args.seeds):
            seed = 1000 + s
            if name in INDEX_FITTERS:
                predict, n_iter = INDEX_FITTERS[name](tr, va, seed)
                fold_va += predict("val") / args.seeds
                test_pred[key] += predict("test") / (args.seeds * N_FOLDS)
            elif name == "xgbglm":
                if s == 0:
                    m_tr, m_va, m_te, m_or = glm_margins(tr, va)
                    if original is not None:
                        m_tr = np.concatenate([m_tr, m_or])
                model, predict, n_iter = fit_xgb(Xtr, ytr, Xva, y[va], seed, depth=2,
                                                 margins=(m_tr, m_va))
                fold_va += predict(Xva, m_va) / args.seeds
                test_pred[key] += predict(Xte, m_te) / (args.seeds * N_FOLDS)
            else:
                model, predict, n_iter = TREE_FITTERS[name](Xtr, ytr, Xva, y[va], seed)
                fold_va += predict(Xva) / args.seeds
                test_pred[key] += predict(Xte) / (args.seeds * N_FOLDS)
        oof[key][va] = fold_va
        unit = {"nn": "epochs", "glm": "iterations", "ebm": "rounds"}.get(name, "trees")
        print(f"  fold {k + 1}: AUC {score(y[va], fold_va):.5f}  ({unit} {n_iter})  [{time.time() - t0:.0f}s]")
    print(f"  {key} CV AUC {score(y, oof[key]):.5f}  logloss {log_loss(y, np.clip(oof[key], 1e-7, 1 - 1e-7)):.5f}"
          f"  [{time.time() - t0:.0f}s]")

    np.save(data_dir / f"oof_{key}.npy", oof[key])
    np.save(data_dir / f"test_{key}.npy", test_pred[key])
    sub[TARGET] = test_pred[key]
    sub.to_csv(data_dir / f"submission_{key}.csv", index=False)

    if name.startswith("lgb") and model is not None:
        imp = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
        print("\n  LightGBM feature importance (gain, last fold, top 20):")
        print((imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())

if len(oof) > 1:
    blend(oof, test_pred, f"submission_blend{suffix}.csv")

print(f"\nSubmissions written to {data_dir.resolve()}")
print("Tip: python train_models.py --blend-only   blends every saved run together.")
