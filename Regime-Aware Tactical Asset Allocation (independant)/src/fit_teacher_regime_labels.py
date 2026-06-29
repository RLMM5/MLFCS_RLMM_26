import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from numba import njit

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 900)
pd.set_option("display.width", 760)
pd.set_option("display.max_rows", 4000)
pd.set_option("display.float_format", lambda x: f"{x:0.6f}")

EPS = 1e-12
RANDOM_SEED = 123
DAILY_INITIAL_YEARS = 12
MONTHLY_INITIAL_MONTHS = 144
STANDARDIZE_CLIP_SIGMA = 3.0
MAX_CD_ITER = 100
N_INIT = 5

EQUITY_LAMBDA_GRID = [2, 4, 8, 16, 32, 64, 128]
BOND_LAMBDA_GRID = [0, 1e-5, 1e-4, 0.001, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
MODEL_FAMILIES = ["standard_l2_jump_model", "robust_l1_medoids_jump_model"]
MODEL_LABELS = {"standard_l2_jump_model": "L2", "robust_l1_medoids_jump_model": "L1-medoid"}
MODEL_SHORT_NAME = {"standard_l2_jump_model": "l2", "robust_l1_medoids_jump_model": "l1med"}
MAJORITY_SCOPES = {
    "all_models": None,
    "l2_only": ["standard_l2_jump_model"],
    "l1_medoids_only": ["robust_l1_medoids_jump_model"],
}

MODEL_VARIANT = "independent_teacher_feature_group_majority"
BOND_FEATURE_GROUP = "return_downside_sharpe_vol"
MODEL_OUTPUT_DIR = TEACHER_LABEL_OUTPUT_DIR
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_LABEL_PATH_PLOTS = True
RUN_LAMBDA_DIAGNOSTIC_PLOTS = True
RUN_MAJORITY_VOTE_PLOTS = True

STATE_COLORS = {1: "#b7e4bd", 0: "#f5b5b5"}

CRISIS_WINDOWS = [
    ("1990 recession", "1990-07-31", "1991-03-31"),
    ("1994 bond shock", "1994-01-31", "1994-12-31"),
    ("1998 LTCM-like", "1998-07-31", "1998-10-31"),
    ("Dot-com bust", "2000-03-31", "2002-10-31"),
    ("GFC", "2007-10-31", "2009-03-31"),
    ("Euro sovereign 2011", "2011-05-31", "2011-12-31"),
    ("Covid crash", "2020-02-29", "2020-04-30"),
    ("Inflation 2022", "2022-01-31", "2022-12-31"),
    ("Recent 2024-2026", "2024-01-31", "2026-04-30"),
]

NON_FEATURE_COLS = {
    "date", "month_end", "asset", "teacher_panel", "source_frequency",
    "feature_source", "label_source", "feature_group", "excess_return",
    "excess_return_source", "n_daily_obs", "asset_excess", "asset_excess_from_features",
    "feature_panel_excess", "eq_excess", "bd_excess", "level", "state", "prob",
    "vote_prob", "vote_strength", "n_teachers", "majority_scope", "majority_state",
    "vote_margin", "unanimous", "tie", "mean_teacher_prob", "min_teacher_state",
    "max_teacher_state", "variant", "model_variant", "path_id", "model_family", "lambda",
    "fit_end_date", "n_train_obs", "n_obs_in_month", "raw_state_mapped_to_good",
    "mapping_metric",
}


def month_end(s):
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def safe_first_date(s):
    s = pd.to_datetime(s, errors="coerce").dropna()
    return pd.NaT if s.empty else s.min()


def safe_last_date(s):
    s = pd.to_datetime(s, errors="coerce").dropna()
    return pd.NaT if s.empty else s.max()


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def sanitize_float_for_col(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "none"
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def sanitize_name(x):
    return (
        str(x)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("|", "_")
        .replace(".", "p")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(":", "_")
        .lower()
    )


def path_id(asset, feature_group, model_family, lam):
    prefix = {"equity": "eq", "bond": "bd"}.get(asset, asset)
    model_short = MODEL_SHORT_NAME.get(model_family, model_family)
    return "_".join([
        prefix,
        sanitize_name(feature_group),
        sanitize_name(model_short),
        f"lam{sanitize_float_for_col(lam)}",
    ])


def annualization_factor(freq):
    return 252.0 if freq == "daily" else 12.0


def compound_return(ret):
    r = pd.Series(ret).dropna().astype(float)
    return float(np.prod(1.0 + r) - 1.0) if len(r) else np.nan


def count_switches(state):
    s = pd.Series(state).dropna().astype(int).reset_index(drop=True)
    return 0 if len(s) <= 1 else int((s != s.shift(1)).sum() - 1)


def spell_lengths(state):
    s = pd.Series(state).dropna().astype(int).reset_index(drop=True)
    if len(s) == 0:
        return []
    run_id = (s != s.shift(1)).cumsum()
    return [(int(g.iloc[0]), int(len(g))) for _, g in s.groupby(run_id)]


def mean_spell_length(state):
    spells = spell_lengths(state)
    return float(np.mean([n for _, n in spells])) if spells else np.nan


def mean_spell_length_for_state(state, target_state):
    vals = [n for st, n in spell_lengths(state) if st == int(target_state)]
    return float(np.mean(vals)) if vals else np.nan


def current_spell(state):
    s = pd.Series(state).dropna().astype(int).reset_index(drop=True)
    if len(s) == 0:
        return np.nan, np.nan
    last_state = int(s.iloc[-1])
    n = 1
    for i in range(len(s) - 2, -1, -1):
        if int(s.iloc[i]) == last_state:
            n += 1
        else:
            break
    return last_state, n


def transition_stats(state):
    s = pd.Series(state).dropna().astype(int).reset_index(drop=True)
    if len(s) < 2:
        return {
            "p_bad_bad": np.nan,
            "p_bad_good": np.nan,
            "p_good_bad": np.nan,
            "p_good_good": np.nan,
            "expected_duration_bad": np.nan,
            "expected_duration_good": np.nan,
        }
    prev = s.shift(1).iloc[1:].astype(int)
    curr = s.iloc[1:].astype(int)

    def p(i, j):
        den = int((prev == i).sum())
        return np.nan if den == 0 else float(((prev == i) & (curr == j)).sum() / den)

    p00, p01, p10, p11 = p(0, 0), p(0, 1), p(1, 0), p(1, 1)
    return {
        "p_bad_bad": p00,
        "p_bad_good": p01,
        "p_good_bad": p10,
        "p_good_good": p11,
        "expected_duration_bad": 1.0 / max(1.0 - p00, EPS) if np.isfinite(p00) else np.nan,
        "expected_duration_good": 1.0 / max(1.0 - p11, EPS) if np.isfinite(p11) else np.nan,
    }


def ann_mean(r, ann_factor=12.0):
    x = pd.Series(r).dropna().astype(float)
    return float(ann_factor * x.mean()) if len(x) else np.nan


def ann_vol(r, ann_factor=12.0):
    x = pd.Series(r).dropna().astype(float)
    return float(np.sqrt(ann_factor) * x.std(ddof=1)) if len(x) >= 3 else np.nan


def ann_sharpe(r, ann_factor=12.0):
    x = pd.Series(r).dropna().astype(float)
    if len(x) < 3:
        return np.nan
    sd = x.std(ddof=1)
    return np.nan if not np.isfinite(sd) or sd < EPS else float(np.sqrt(ann_factor) * x.mean() / sd)


def downside_ann(r, ann_factor=12.0):
    x = pd.Series(r).dropna().astype(float)
    return float(np.sqrt(ann_factor) * np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))) if len(x) >= 3 else np.nan


def cvar(r, q=0.01):
    x = pd.Series(r).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    cutoff = x.quantile(q)
    tail = x.loc[x <= cutoff]
    return float(tail.mean()) if len(tail) else np.nan


def max_drawdown(r):
    x = pd.Series(r).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    wealth = (1.0 + x.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def feature_block(feature):
    f = str(feature).lower()
    if "log_downside" in f or "downside" in f or "tail" in f:
        return "downside_risk"
    if "sharpe" in f or "sortino" in f:
        return "risk_adjusted_return"
    if "realized_vol" in f or "vol_" in f or f.startswith("vol") or "_vol" in f:
        return "volatility"
    if "ewm_return" in f or "return" in f or "mean" in f:
        return "return_trend"
    return "other"


def date_span_edges(dates):
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    p = d.dt.to_period("M")
    return p.dt.start_time, p.dt.end_time


def add_state_background_runs(ax, df, date_col="date", state_col="state"):
    x = df[[date_col, state_col]].dropna().copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[state_col] = x[state_col].astype(int)
    x = x.sort_values(date_col).reset_index(drop=True)
    left, right = date_span_edges(x[date_col])
    x["left"] = left
    x["right"] = right
    x["run_id"] = (x[state_col] != x[state_col].shift(1)).cumsum()
    ymin, ymax = ax.get_ylim()
    for _, g in x.groupby("run_id", sort=True):
        st = int(g[state_col].iloc[0])
        ax.axvspan(
            g["left"].iloc[0],
            g["right"].iloc[-1],
            facecolor=STATE_COLORS[st],
            alpha=0.60,
            edgecolor="none",
            linewidth=0,
            zorder=0,
        )
    ax.set_ylim(ymin, ymax)


@njit(cache=False)
def dp_path_numba(loss, jump_lambda):
    n = loss.shape[0]
    V = np.empty((n, 2))
    back = np.empty((n, 2), dtype=np.int64)
    V[0, 0] = loss[0, 0]
    V[0, 1] = loss[0, 1]
    back[0, 0] = 0
    back[0, 1] = 1
    for t in range(1, n):
        for s in range(2):
            stay = V[t - 1, s]
            switch = V[t - 1, 1 - s] + jump_lambda
            if stay <= switch:
                V[t, s] = loss[t, s] + stay
                back[t, s] = s
            else:
                V[t, s] = loss[t, s] + switch
                back[t, s] = 1 - s
    path = np.empty(n, dtype=np.int64)
    path[n - 1] = 0 if V[n - 1, 0] <= V[n - 1, 1] else 1
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, V[n - 1, path[n - 1]]


def dp_path(loss, jump_lambda):
    return dp_path_numba(np.asarray(loss, dtype=np.float64), float(jump_lambda))


def clip_and_standardize_train(X):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(~np.isfinite(sd) | (sd < EPS), 1.0, sd)
    lo = mu - STANDARDIZE_CLIP_SIGMA * sd
    hi = mu + STANDARDIZE_CLIP_SIGMA * sd
    Xc = np.minimum(np.maximum(X, lo), hi)
    Xs = (Xc - mu) / sd
    return np.where(np.isfinite(Xs), Xs, 0.0), mu, sd, lo, hi


def valid_complete_panel(panel, feats):
    x = panel.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = (
        x.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
    )
    return x.loc[x[feats + ["excess_return"]].notna().all(axis=1)].reset_index(drop=True).copy()


def refit_end_positions(df, freq):
    if freq == "daily":
        t = df.copy()
        t["month_end"] = month_end(t["date"])
        return sorted(t.groupby("month_end", as_index=False).tail(1).index.to_numpy(dtype=int).tolist())
    return list(range(len(df)))


def training_allowed(end_date, first_date, freq):
    end_date = pd.Timestamp(end_date)
    first_date = pd.Timestamp(first_date)
    if freq == "daily":
        return end_date >= first_date + pd.DateOffset(years=DAILY_INITIAL_YEARS)
    return (end_date.to_period("M") - first_date.to_period("M")).n >= MONTHLY_INITIAL_MONTHS - 1


def min_train_obs(freq):
    return int(252 * DAILY_INITIAL_YEARS) if freq == "daily" else int(MONTHLY_INITIAL_MONTHS)


def init_labels(X, rng, asset_ret=None, method="kmeans"):
    if method == "kmeans":
        lab = KMeans(n_clusters=2, n_init=10, random_state=int(rng.integers(1, 10_000_000))).fit_predict(X)
        if len(np.unique(lab)) == 2:
            return lab
    if method == "quantile_ret" and asset_ret is not None:
        r = pd.Series(asset_ret).astype(float).fillna(0.0).to_numpy()
        lab = (r > np.nanmedian(r)).astype(int)
        if len(np.unique(lab)) == 2:
            return lab
    lab = rng.integers(0, 2, size=X.shape[0])
    if len(np.unique(lab)) == 2:
        return lab
    lab[: len(lab) // 2] = 0
    lab[len(lab) // 2 :] = 1
    return lab


def mean_centers(X, labels):
    glob = np.nanmean(X, axis=0)
    centers = np.zeros((2, X.shape[1]))
    for k in [0, 1]:
        centers[k] = glob if not (labels == k).any() else np.nanmean(X[labels == k], axis=0)
    return np.where(np.isfinite(centers), centers, 0.0)


def nearest_to_l1_median_centers(X, labels):
    glob = np.nanmedian(X, axis=0)
    centers = np.zeros((2, X.shape[1]))
    for k in [0, 1]:
        Xk = X[labels == k]
        if len(Xk) == 0:
            centers[k] = glob
        else:
            med = np.nanmedian(Xk, axis=0)
            centers[k] = Xk[int(np.argmin(np.sum(np.abs(Xk - med), axis=1)))]
    return np.where(np.isfinite(centers), centers, 0.0)


def l2_loss(X, centers):
    return np.column_stack([
        0.5 * np.sum((X - centers[0]) ** 2, axis=1),
        0.5 * np.sum((X - centers[1]) ** 2, axis=1),
    ])


def l1_loss(X, centers):
    return np.column_stack([
        np.sum(np.abs(X - centers[0]), axis=1),
        np.sum(np.abs(X - centers[1]), axis=1),
    ])


def center_distance(centers):
    return np.nan if centers is None else float(np.linalg.norm(centers[1] - centers[0]))


def fit_jump_model(X, jump_lambda, asset_ret, seed, kind):
    rng = np.random.default_rng(seed)
    best = None
    methods = ["kmeans", "quantile_ret", "random"]
    for i in range(N_INIT):
        labels = init_labels(X, rng, asset_ret, methods[i % len(methods)])
        centers = mean_centers(X, labels) if kind == "l2" else nearest_to_l1_median_centers(X, labels)
        last = None
        for _ in range(MAX_CD_ITER):
            loss = l2_loss(X, centers) if kind == "l2" else l1_loss(X, centers)
            path, _ = dp_path(loss, jump_lambda)
            if last is not None and np.array_equal(path, last):
                break
            last = path.copy()
            centers = mean_centers(X, path) if kind == "l2" else nearest_to_l1_median_centers(X, path)
        loss = l2_loss(X, centers) if kind == "l2" else l1_loss(X, centers)
        path, _ = dp_path(loss, jump_lambda)
        data_loss = float(np.sum(loss[np.arange(len(path)), path]))
        jump_loss = float(jump_lambda * np.sum(path[1:] != path[:-1]))
        cand = {
            "labels": path,
            "centers": centers,
            "obj": data_loss + jump_loss,
            "data_loss": data_loss,
            "jump_loss": jump_loss,
            "center_dist": center_distance(centers),
        }
        if best is None or cand["obj"] < best["obj"]:
            best = cand
    return best


def bcss_from_labels(X, labels):
    glob = np.nanmean(X, axis=0)
    b = np.zeros(X.shape[1])
    labels = np.asarray(labels, dtype=int)
    for k in [0, 1]:
        m = labels == k
        if m.any():
            b += int(m.sum()) * (np.nanmean(X[m], axis=0) - glob) ** 2
    return np.where(np.isfinite(b), b, 0.0)


def map_raw_state_to_good(labels, ret, freq):
    labels = np.asarray(labels, dtype=int)
    ret = np.asarray(ret, dtype=float)
    ann = annualization_factor(freq)
    stats = {}
    for k in [0, 1]:
        rk = pd.Series(ret[labels == k]).dropna().astype(float)
        stats[f"raw_state{k}_n"] = int(len(rk))
        stats[f"raw_state{k}_ann_mean"] = ann_mean(rk, ann)
        stats[f"raw_state{k}_ann_vol"] = ann_vol(rk, ann)
        stats[f"raw_state{k}_ann_sharpe"] = ann_sharpe(rk, ann)
        stats[f"raw_state{k}_ann_downside"] = downside_ann(rk, ann)
    s0, s1 = stats["raw_state0_ann_sharpe"], stats["raw_state1_ann_sharpe"]
    if np.isfinite(s0) and np.isfinite(s1) and abs(s0 - s1) > 1e-10:
        good = 0 if s0 >= s1 else 1
        metric = "higher_realized_sharpe"
    else:
        m0, m1 = stats["raw_state0_ann_mean"], stats["raw_state1_ann_mean"]
        good = 0 if np.isfinite(m0) and np.isfinite(m1) and m0 >= m1 else 1
        metric = "higher_realized_mean_fallback"
    stats.update({"raw_state_mapped_to_good": int(good), "mapping_metric": metric})
    return int(good), stats


def centroid_diagnostics(X, labels, feats, raw_good, meta):
    labels = np.asarray(labels, dtype=int)
    raw_bad = 1 - int(raw_good)
    glob = np.nanmean(X, axis=0)
    bcss = bcss_from_labels(X, labels)
    rows = []
    for j, f in enumerate(feats):
        xb = X[labels == raw_bad, j]
        xg = X[labels == raw_good, j]
        cb = float(np.nanmean(xb)) if len(xb) else np.nan
        cg = float(np.nanmean(xg)) if len(xg) else np.nan
        sb = float(np.nanstd(xb, ddof=0)) if len(xb) else np.nan
        sg = float(np.nanstd(xg, ddof=0)) if len(xg) else np.nan
        gap = cg - cb
        sep = abs(gap) / (np.sqrt(0.5 * ((sb if np.isfinite(sb) else 0) ** 2 + (sg if np.isfinite(sg) else 0) ** 2)) + EPS)
        rows.append({
            **meta,
            "feature": f,
            "centroid_bad": cb,
            "centroid_good": cg,
            "gap_good_minus_bad": gap,
            "abs_gap": abs(gap),
            "within_bad_std": sb,
            "within_good_std": sg,
            "centroid_separation_ratio": sep,
            "BCSS": float(bcss[j]),
            "global_mean_scaled": float(glob[j]),
        })
    return rows


def fit_one_teacher_path(panel, asset, feature_group, model_family, lam):
    panel = panel.copy()
    panel["asset"] = asset
    panel["feature_group"] = feature_group
    freq = str(panel["source_frequency"].dropna().iloc[0])
    label_source = f"{asset}_{feature_group}_{freq}_sjm"
    panel["label_source"] = label_source
    feats = feature_cols(panel)
    df = valid_complete_panel(panel, feats)
    first = pd.Timestamp(df["date"].min())
    minobs = min_train_obs(freq)
    positions = refit_end_positions(df, freq)
    pid = path_id(asset, feature_group, model_family, lam)
    monthly_rows = []
    log_rows = []
    centroid_rows = []
    last_fit = None
    t0 = time.time()

    for end_pos in positions:
        end = pd.Timestamp(df.loc[end_pos, "date"])
        if not training_allowed(end, first, freq):
            continue
        tr = df.iloc[: end_pos + 1].copy()
        if len(tr) < minobs:
            continue

        X, mu, sd, lo, hi = clip_and_standardize_train(tr[feats].to_numpy(float))
        ret = tr["excess_return"].to_numpy(float)
        seed = RANDOM_SEED + int(abs(hash((asset, feature_group, model_family, float(lam), str(end)))) % 1_000_000)
        fit = fit_jump_model(X, float(lam), ret, seed, "l2" if model_family == "standard_l2_jump_model" else "l1")
        labels = np.asarray(fit["labels"], dtype=int)
        raw_good, stats = map_raw_state_to_good(labels, ret, freq)
        good = (labels == raw_good).astype(float)
        current = end.to_period("M").to_timestamp("M")

        if freq == "daily":
            mm = month_end(tr["date"]).eq(current).to_numpy()
            prob = float(np.nanmean(good[mm]))
            state = float(prob >= 0.5)
            asset_excess = compound_return(tr.loc[mm, "excess_return"])
            nobs = int(mm.sum())
            feature_source = "daily"
            n_daily_obs = np.nan
        else:
            prob = float(good[-1])
            state = prob
            asset_excess = float(tr["excess_return"].iloc[-1])
            nobs = 1
            feature_source = str(tr["feature_source"].iloc[-1]) if "feature_source" in tr.columns else "monthly"
            n_daily_obs = float(tr["n_daily_obs"].iloc[-1]) if "n_daily_obs" in tr.columns and pd.notna(tr["n_daily_obs"].iloc[-1]) else np.nan

        monthly_rows.append({
            "date": current,
            "asset": asset,
            "feature_group": feature_group,
            "model_family": model_family,
            "lambda": float(lam),
            "label_source": label_source,
            "source_frequency": freq,
            "path_id": pid,
            "state": state,
            "prob": prob,
            "asset_excess": asset_excess,
            "n_obs_in_month": nobs,
            "n_train_obs": len(tr),
            "fit_end_date": end,
            "raw_state_mapped_to_good": int(raw_good),
            "mapping_metric": stats["mapping_metric"],
            "feature_source": feature_source,
            "n_daily_obs": n_daily_obs,
        })
        last_fit = fit, X, tr, labels, raw_good, stats

    monthly_df = (
        pd.DataFrame(monthly_rows)
        .sort_values("date")
        .drop_duplicates(["date", "path_id"], keep="last")
        .reset_index(drop=True)
        if monthly_rows
        else pd.DataFrame()
    )

    meta = {
        "path_id": pid,
        "asset": asset,
        "feature_group": feature_group,
        "model_family": model_family,
        "lambda": float(lam),
        "label_source": label_source,
        "source_frequency": freq,
    }

    if last_fit is not None:
        fit, X, tr, labels, raw_good, stats = last_fit
        centroid_rows = centroid_diagnostics(X, labels, feats, raw_good, meta)
        log_rows.append({
            **meta,
            "n_feature_cols": len(feats),
            "n_monthly_labels": len(monthly_df),
            "first_label_date": safe_first_date(monthly_df["date"]) if len(monthly_df) else pd.NaT,
            "last_label_date": safe_last_date(monthly_df["date"]) if len(monthly_df) else pd.NaT,
            "last_n_train_obs": len(tr),
            "last_obj": fit.get("obj", np.nan),
            "last_data_loss": fit.get("data_loss", np.nan),
            "last_jump_loss": fit.get("jump_loss", np.nan),
            "last_center_dist": fit.get("center_dist", np.nan),
            "raw_state_mapped_to_good_last": int(raw_good),
            "mapping_metric_last": stats["mapping_metric"],
            "elapsed_seconds": time.time() - t0,
        })

    return monthly_df, pd.DataFrame(log_rows), pd.DataFrame(centroid_rows)


def prepare_teacher_panel(df, asset, feature_group, source_frequency):
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["asset"] = asset
    x["feature_group"] = feature_group
    x["source_frequency"] = source_frequency
    x["label_source"] = f"{asset}_{feature_group}_{source_frequency}_sjm"
    return x.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


equity_teacher = prepare_teacher_panel(
    equity_teacher,
    asset="equity",
    feature_group="return_sortino",
    source_frequency="daily",
)

bond_teacher = prepare_teacher_panel(
    bond_teacher,
    asset="bond",
    feature_group=BOND_FEATURE_GROUP,
    source_frequency="monthly",
)

teacher_feature_sets = {
    ("equity", "return_sortino"): EQUITY_RETURN_SORTINO_FEATURES,
    ("bond", BOND_FEATURE_GROUP): BOND_MONTHLY_TEACHER_FEATURES,
}

teacher_panel_registry = {
    ("equity", "return_sortino"): equity_teacher.copy(),
    ("bond", BOND_FEATURE_GROUP): bond_teacher.copy(),
}

for key, feats in teacher_feature_sets.items():
    panel = teacher_panel_registry[key].copy()
    for c in feats + ["excess_return"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    teacher_panel_registry[key] = panel

equity_teacher = teacher_panel_registry[("equity", "return_sortino")].copy()
bond_teacher = teacher_panel_registry[("bond", BOND_FEATURE_GROUP)].copy()

teacher_daily_panels = {
    ("equity", "return_sortino", "daily_sjm"): equity_teacher.copy(),
}

teacher_monthly_panels = {
    ("bond", BOND_FEATURE_GROUP, "monthly_sjm"): bond_teacher.copy(),
}

model_specs = []
for model_family in MODEL_FAMILIES:
    for (asset, feature_group), panel in teacher_panel_registry.items():
        lambda_grid = EQUITY_LAMBDA_GRID if asset == "equity" else BOND_LAMBDA_GRID
        for lam in lambda_grid:
            model_specs.append((panel.copy(), asset, feature_group, model_family, float(lam)))

label_parts = []
log_parts = []
centroid_parts = []


for panel, asset, feature_group, model_family, lam in model_specs:
    monthly_df, fit_log_df, centroid_df = fit_one_teacher_path(
        panel=panel,
        asset=asset,
        feature_group=feature_group,
        model_family=model_family,
        lam=lam,
    )
    if len(monthly_df):
        monthly_df = monthly_df.copy()
        monthly_df["model_variant"] = MODEL_VARIANT
        label_parts.append(monthly_df)
    if len(fit_log_df):
        fit_log_df = fit_log_df.copy()
        fit_log_df["model_variant"] = MODEL_VARIANT
        log_parts.append(fit_log_df)
    if len(centroid_df):
        centroid_df = centroid_df.copy()
        centroid_df["model_variant"] = MODEL_VARIANT
        centroid_parts.append(centroid_df)

teacher_monthly_labels_long = pd.concat(label_parts, ignore_index=True, sort=False)
teacher_fit_log = pd.concat(log_parts, ignore_index=True, sort=False)
teacher_centroid_feature_diagnostics = pd.concat(centroid_parts, ignore_index=True, sort=False)

teacher_monthly_labels_long["date"] = pd.to_datetime(teacher_monthly_labels_long["date"], errors="coerce")
teacher_monthly_labels_long["lambda"] = pd.to_numeric(teacher_monthly_labels_long["lambda"], errors="coerce")
teacher_monthly_labels_long["state"] = pd.to_numeric(teacher_monthly_labels_long["state"], errors="coerce")
teacher_monthly_labels_long["prob"] = pd.to_numeric(teacher_monthly_labels_long["prob"], errors="coerce")
teacher_monthly_labels_long["asset_excess"] = pd.to_numeric(teacher_monthly_labels_long["asset_excess"], errors="coerce")
teacher_monthly_labels_long = (
    teacher_monthly_labels_long
    .sort_values(["asset", "feature_group", "model_family", "lambda", "date"])
    .reset_index(drop=True)
)


diag_group_cols = [
    "path_id",
    "asset",
    "feature_group",
    "model_family",
    "lambda",
    "label_source",
    "source_frequency",
]


def persistence_row(g):
    g = g.sort_values("date").dropna(subset=["state"]).copy()
    s = g["state"].astype(int)
    n_months = int(len(g))
    n_switches = count_switches(s)
    return pd.Series({
        "n_months": n_months,
        "first_date": g["date"].min() if n_months else pd.NaT,
        "last_date": g["date"].max() if n_months else pd.NaT,
        "good_share": float(s.mean()) if n_months else np.nan,
        "bad_share": float(1.0 - s.mean()) if n_months else np.nan,
        "n_switches": n_switches,
        "mean_spell_length": mean_spell_length(s),
        "mean_good_spell_length": mean_spell_length_for_state(s, 1),
        "mean_bad_spell_length": mean_spell_length_for_state(s, 0),
        "monthly_turnover": float(n_switches / max(n_months - 1, 1)) if n_months else np.nan,
    })


def state_return_row(g):
    g = g.sort_values("date").dropna(subset=["state", "asset_excess"]).copy()
    s = g["state"].astype(int)
    good = g.loc[s.eq(1), "asset_excess"]
    bad = g.loc[s.eq(0), "asset_excess"]
    return pd.Series({
        "n_months": int(len(g)),
        "n_good": int(len(good)),
        "n_bad": int(len(bad)),
        "state_good_ann_return": ann_mean(good, 12.0),
        "state_bad_ann_return": ann_mean(bad, 12.0),
        "state_good_ann_vol": ann_vol(good, 12.0),
        "state_bad_ann_vol": ann_vol(bad, 12.0),
        "state_good_downside": downside_ann(good, 12.0),
        "state_bad_downside": downside_ann(bad, 12.0),
        "state_good_sharpe": ann_sharpe(good, 12.0),
        "state_bad_sharpe": ann_sharpe(bad, 12.0),
        "good_minus_bad_ann_return": ann_mean(good, 12.0) - ann_mean(bad, 12.0),
        "good_minus_bad_ann_vol": ann_vol(good, 12.0) - ann_vol(bad, 12.0),
        "good_minus_bad_downside": downside_ann(good, 12.0) - downside_ann(bad, 12.0),
        "good_minus_bad_sharpe": ann_sharpe(good, 12.0) - ann_sharpe(bad, 12.0),
        "good_max_drawdown": max_drawdown(good),
        "bad_max_drawdown": max_drawdown(bad),
        "good_cvar_1pct": cvar(good, 0.01),
        "bad_cvar_1pct": cvar(bad, 0.01),
    })


teacher_persistence_diagnostics = (
    teacher_monthly_labels_long
    .groupby(diag_group_cols, dropna=False)
    .apply(persistence_row)
    .reset_index()
    .sort_values(["asset", "feature_group", "model_family", "lambda"])
    .reset_index(drop=True)
)

teacher_state_return_diagnostics = (
    teacher_monthly_labels_long
    .groupby(diag_group_cols, dropna=False)
    .apply(state_return_row)
    .reset_index()
    .sort_values(["asset", "feature_group", "model_family", "lambda"])
    .reset_index(drop=True)
)

teacher_feature_separation_by_path = teacher_centroid_feature_diagnostics.copy()
teacher_feature_separation_by_path["centroid_separation_ratio"] = pd.to_numeric(teacher_feature_separation_by_path["centroid_separation_ratio"], errors="coerce")
teacher_feature_separation_by_path["abs_gap"] = pd.to_numeric(teacher_feature_separation_by_path["abs_gap"], errors="coerce")
teacher_feature_separation_by_path["gap_good_minus_bad"] = pd.to_numeric(teacher_feature_separation_by_path["gap_good_minus_bad"], errors="coerce")
teacher_feature_separation_by_path["dominant_direction"] = np.where(
    teacher_feature_separation_by_path["gap_good_minus_bad"] >= 0,
    "higher_in_good",
    "higher_in_bad",
)
teacher_feature_separation_by_path = (
    teacher_feature_separation_by_path
    .sort_values(["asset", "feature_group", "model_family", "lambda", "centroid_separation_ratio"], ascending=[True, True, True, True, False])
    .reset_index(drop=True)
)
teacher_feature_separation_by_path["rank_within_path"] = (
    teacher_feature_separation_by_path
    .groupby("path_id", dropna=False)["centroid_separation_ratio"]
    .rank(method="first", ascending=False)
    .astype(int)
)

teacher_feature_separation_summary = (
    teacher_feature_separation_by_path
    .groupby(["asset", "feature_group", "model_family", "feature"], dropna=False)
    .agg(
        n_paths=("path_id", "nunique"),
        median_rank=("rank_within_path", "median"),
        mean_rank=("rank_within_path", "mean"),
        best_rank=("rank_within_path", "min"),
        worst_rank=("rank_within_path", "max"),
        median_sep=("centroid_separation_ratio", "median"),
        mean_sep=("centroid_separation_ratio", "mean"),
        min_sep=("centroid_separation_ratio", "min"),
        max_sep=("centroid_separation_ratio", "max"),
        median_abs_gap=("abs_gap", "median"),
        median_bcss=("BCSS", "median"),
        positive_gap_share=("gap_good_minus_bad", lambda z: float((pd.to_numeric(z, errors="coerce") > 0).mean())),
        negative_gap_share=("gap_good_minus_bad", lambda z: float((pd.to_numeric(z, errors="coerce") < 0).mean())),
    )
    .reset_index()
)
teacher_feature_separation_summary["dominant_direction"] = np.where(
    teacher_feature_separation_summary["positive_gap_share"] >= teacher_feature_separation_summary["negative_gap_share"],
    "higher_in_good",
    "higher_in_bad",
)
teacher_feature_separation_summary["direction_consistency"] = teacher_feature_separation_summary[["positive_gap_share", "negative_gap_share"]].max(axis=1)
teacher_feature_separation_summary = teacher_feature_separation_summary.sort_values(
    ["asset", "feature_group", "model_family", "median_sep"],
    ascending=[True, True, True, False],
).reset_index(drop=True)

path_key = diag_group_cols

top_feature_base = (
    teacher_feature_separation_by_path
    .sort_values(path_key + ["centroid_separation_ratio"], ascending=[True] * len(path_key) + [False], na_position="last")
    .groupby(path_key, dropna=False)
    .head(5)
    .copy()
)
top_feature_base["feature_desc"] = (
    top_feature_base["feature"].astype(str)
    + ":"
    + top_feature_base["centroid_separation_ratio"].apply(lambda v: "nan" if pd.isna(v) else f"{float(v):0.2f}")
)
top_features_by_path = (
    top_feature_base
    .groupby(path_key, dropna=False)["feature_desc"]
    .agg(lambda z: ", ".join([str(v) for v in z if pd.notna(v)]))
    .reset_index()
    .rename(columns={"feature_desc": "top5_centroid_features"})
)
centroid_by_path = (
    teacher_feature_separation_by_path
    .groupby(path_key, dropna=False)
    .agg(
        n_features=("feature", "count"),
        median_centroid_separation=("centroid_separation_ratio", "median"),
        mean_centroid_separation=("centroid_separation_ratio", "mean"),
        max_centroid_separation=("centroid_separation_ratio", "max"),
        min_centroid_separation=("centroid_separation_ratio", "min"),
        n_features_sep_ge_0p5=("centroid_separation_ratio", lambda z: int((pd.to_numeric(z, errors="coerce") >= 0.5).sum())),
        n_features_sep_ge_1p0=("centroid_separation_ratio", lambda z: int((pd.to_numeric(z, errors="coerce") >= 1.0).sum())),
        n_features_sep_ge_1p5=("centroid_separation_ratio", lambda z: int((pd.to_numeric(z, errors="coerce") >= 1.5).sum())),
        median_abs_gap=("abs_gap", "median"),
        max_abs_gap=("abs_gap", "max"),
        total_bcss=("BCSS", "sum"),
    )
    .reset_index()
    .merge(top_features_by_path, on=path_key, how="left")
)

teacher_path_diagnostics = (
    teacher_persistence_diagnostics
    .merge(teacher_state_return_diagnostics, on=path_key, how="left", suffixes=("", "_ret"))
    .merge(centroid_by_path, on=path_key, how="left")
)

for c in [
    "good_share", "n_switches", "mean_spell_length", "mean_good_spell_length",
    "mean_bad_spell_length", "monthly_turnover", "good_minus_bad_ann_return",
    "good_minus_bad_ann_vol", "good_minus_bad_downside", "good_minus_bad_sharpe",
    "median_centroid_separation", "max_centroid_separation", "n_features_sep_ge_1p0",
]:
    if c in teacher_path_diagnostics.columns:
        teacher_path_diagnostics[c] = pd.to_numeric(teacher_path_diagnostics[c], errors="coerce")

teacher_path_diagnostics["state_balance_ok"] = teacher_path_diagnostics["good_share"].between(0.10, 0.90)
teacher_path_diagnostics["not_constant"] = teacher_path_diagnostics["n_switches"].ge(2)
teacher_path_diagnostics["not_monthly_flipping"] = teacher_path_diagnostics["mean_spell_length"].ge(3.0)
teacher_path_diagnostics["economic_order_ok"] = (
    (teacher_path_diagnostics["good_minus_bad_ann_return"] > 0)
    & (teacher_path_diagnostics["good_minus_bad_sharpe"] > 0)
)
teacher_path_diagnostics["centroid_ok"] = (
    (teacher_path_diagnostics["median_centroid_separation"] >= 0.5)
    & (teacher_path_diagnostics["n_features_sep_ge_1p0"] >= 3)
)
teacher_path_diagnostics = teacher_path_diagnostics.sort_values(["asset", "feature_group", "model_family", "lambda"]).reset_index(drop=True)


def make_wide_labels(long_df):
    x = long_df[["date", "asset", "feature_group", "path_id", "state", "prob", "asset_excess"]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    pieces = []
    for pid, g in x.groupby("path_id", sort=True):
        p = g[["date", "state", "prob"]].sort_values("date").copy()
        p = p.rename(columns={"state": f"{pid}_state", "prob": f"{pid}_prob"})
        pieces.append(p)
    wide = pieces[0]
    for p in pieces[1:]:
        wide = wide.merge(p, on="date", how="outer")

    ret = (
        x.groupby(["date", "asset"], as_index=False)["asset_excess"]
        .mean()
        .pivot(index="date", columns="asset", values="asset_excess")
        .reset_index()
    )
    ret.columns.name = None
    ret = ret.rename(columns={"equity": "eq_excess", "bond": "bd_excess"})
    return wide.merge(ret, on="date", how="left").sort_values("date").reset_index(drop=True)


teacher_monthly_labels_wide = make_wide_labels(teacher_monthly_labels_long)
teacher_prediction_targets_h1_all_paths = teacher_monthly_labels_wide.copy()
non_date_cols = [c for c in teacher_prediction_targets_h1_all_paths.columns if c != "date"]
teacher_prediction_targets_h1_all_paths[non_date_cols] = teacher_prediction_targets_h1_all_paths[non_date_cols].shift(-1)

majority_parts = []
for (asset, feature_group), asset_fg_df in teacher_monthly_labels_long.groupby(["asset", "feature_group"], sort=True):
    for majority_scope, model_filter in MAJORITY_SCOPES.items():
        h = asset_fg_df.copy() if model_filter is None else asset_fg_df.loc[asset_fg_df["model_family"].isin(model_filter)].copy()
        if h.empty:
            continue
        h = h.sort_values(["path_id", "date"]).drop_duplicates(["path_id", "date"], keep="last").copy()
        m = (
            h.groupby("date", as_index=False)
            .agg(
                asset=("asset", "first"),
                feature_group=("feature_group", "first"),
                vote_prob=("state", "mean"),
                n_teachers=("path_id", "nunique"),
                asset_excess=("asset_excess", "first"),
                mean_teacher_prob=("prob", "mean"),
                min_teacher_state=("state", "min"),
                max_teacher_state=("state", "max"),
            )
        )
        m["majority_scope"] = majority_scope
        m["majority_state"] = (m["vote_prob"] >= 0.5).astype(int)
        m["vote_margin"] = (m["vote_prob"] - 0.5).abs()
        m["vote_strength"] = np.maximum(m["vote_prob"], 1.0 - m["vote_prob"])
        m["unanimous"] = m["vote_strength"].eq(1.0)
        m["tie"] = np.isclose(m["vote_prob"], 0.5)
        m["variant"] = MODEL_VARIANT
        majority_parts.append(m)

majority_vote_labels_long = (
    pd.concat(majority_parts, axis=0, ignore_index=True, sort=False)
    .sort_values(["asset", "feature_group", "majority_scope", "date"])
    .reset_index(drop=True)
)


def majority_metrics_row(g):
    g = g.sort_values("date").dropna(subset=["majority_state", "asset_excess"]).copy()
    s = g["majority_state"].astype(int)
    good = g.loc[s.eq(1), "asset_excess"]
    bad = g.loc[s.eq(0), "asset_excess"]
    trans = transition_stats(s)
    n_switches = count_switches(s)
    return pd.Series({
        "n_months": int(len(g)),
        "first_date": g["date"].min() if len(g) else pd.NaT,
        "last_date": g["date"].max() if len(g) else pd.NaT,
        "good_share": float(s.mean()) if len(g) else np.nan,
        "bad_share": float(1.0 - s.mean()) if len(g) else np.nan,
        "n_switches": n_switches,
        "mean_spell_length": mean_spell_length(s),
        "mean_good_spell_length": mean_spell_length_for_state(s, 1),
        "mean_bad_spell_length": mean_spell_length_for_state(s, 0),
        "monthly_turnover": float(n_switches / max(len(g) - 1, 1)) if len(g) else np.nan,
        "avg_n_teachers": float(g["n_teachers"].mean()) if "n_teachers" in g.columns and len(g) else np.nan,
        "min_n_teachers": int(g["n_teachers"].min()) if "n_teachers" in g.columns and len(g) else np.nan,
        "avg_vote_prob": float(g["vote_prob"].mean()) if "vote_prob" in g.columns and len(g) else np.nan,
        "avg_vote_strength": float(g["vote_strength"].mean()) if "vote_strength" in g.columns and len(g) else np.nan,
        "weak_vote_share_50_60": float((g["vote_strength"] < 0.60).mean()) if "vote_strength" in g.columns and len(g) else np.nan,
        "strong_vote_share_80_100": float((g["vote_strength"] >= 0.80).mean()) if "vote_strength" in g.columns and len(g) else np.nan,
        "state_good_ann_return": ann_mean(good, 12.0),
        "state_bad_ann_return": ann_mean(bad, 12.0),
        "state_good_ann_vol": ann_vol(good, 12.0),
        "state_bad_ann_vol": ann_vol(bad, 12.0),
        "state_good_downside": downside_ann(good, 12.0),
        "state_bad_downside": downside_ann(bad, 12.0),
        "state_good_sharpe": ann_sharpe(good, 12.0),
        "state_bad_sharpe": ann_sharpe(bad, 12.0),
        "state_good_cvar_1pct": cvar(good, 0.01),
        "state_bad_cvar_1pct": cvar(bad, 0.01),
        "good_minus_bad_ann_return": ann_mean(good, 12.0) - ann_mean(bad, 12.0),
        "good_minus_bad_ann_vol": ann_vol(good, 12.0) - ann_vol(bad, 12.0),
        "good_minus_bad_downside": downside_ann(good, 12.0) - downside_ann(bad, 12.0),
        "good_minus_bad_sharpe": ann_sharpe(good, 12.0) - ann_sharpe(bad, 12.0),
        **trans,
    })


majority_vote_metrics = (
    majority_vote_labels_long
    .groupby(["asset", "feature_group", "majority_scope"], dropna=False)
    .apply(majority_metrics_row)
    .reset_index()
    .sort_values(["asset", "feature_group", "majority_scope"])
    .reset_index(drop=True)
)

majority_wide_parts = []
for (asset, feature_group, scope), g in majority_vote_labels_long.groupby(["asset", "feature_group", "majority_scope"], sort=True):
    prefix = f"{asset}_{feature_group}_{scope}"
    p = g[["date", "majority_state", "vote_prob", "vote_strength", "n_teachers", "asset_excess"]].copy()
    p = p.rename(columns={
        "majority_state": f"{prefix}_state",
        "vote_prob": f"{prefix}_vote_prob",
        "vote_strength": f"{prefix}_vote_strength",
        "n_teachers": f"{prefix}_n_teachers",
        "asset_excess": f"{prefix}_excess",
    })
    majority_wide_parts.append(p)

majority_vote_labels_wide = majority_wide_parts[0]
for p in majority_wide_parts[1:]:
    majority_vote_labels_wide = majority_vote_labels_wide.merge(p, on="date", how="outer")
majority_vote_labels_wide = majority_vote_labels_wide.sort_values("date").reset_index(drop=True)

majority_vote_prediction_targets_h1 = majority_vote_labels_wide.copy()
non_date_cols = [c for c in majority_vote_prediction_targets_h1.columns if c != "date"]
majority_vote_prediction_targets_h1[non_date_cols] = majority_vote_prediction_targets_h1[non_date_cols].shift(-1)

majority_vote_path_summary = (
    teacher_monthly_labels_long
    .groupby(["asset", "feature_group", "model_family"], dropna=False)
    .agg(
        n_paths=("path_id", "nunique"),
        lambdas=("lambda", lambda z: ", ".join([f"{float(v):g}" for v in sorted(pd.Series(z).dropna().unique())])),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_rows=("state", "count"),
    )
    .reset_index()
    .sort_values(["asset", "feature_group", "model_family"])
    .reset_index(drop=True)
)


def path_df(pid):
    return teacher_monthly_labels_long.loc[teacher_monthly_labels_long["path_id"].eq(pid)].sort_values("date").reset_index(drop=True).copy()


def find_path_id(asset, feature_group, model_family, lam, long_df=None):
    long_df = teacher_monthly_labels_long if long_df is None else long_df
    meta = long_df[["path_id", "asset", "feature_group", "model_family", "lambda"]].drop_duplicates().copy()
    m = (
        meta["asset"].eq(asset)
        & meta["feature_group"].eq(feature_group)
        & meta["model_family"].eq(model_family)
        & np.isclose(pd.to_numeric(meta["lambda"], errors="coerce"), float(lam), atol=1e-10, rtol=0.0)
    )
    vals = meta.loc[m, "path_id"].dropna().unique().tolist()
    return vals[0] if vals else None


def plot_all_lambda_label_grid(asset, feature_group, model_family):
    meta = (
        teacher_monthly_labels_long.loc[
            teacher_monthly_labels_long["asset"].eq(asset)
            & teacher_monthly_labels_long["feature_group"].eq(feature_group)
            & teacher_monthly_labels_long["model_family"].eq(model_family),
            ["path_id", "asset", "feature_group", "model_family", "lambda"],
        ]
        .drop_duplicates()
        .sort_values("lambda")
        .reset_index(drop=True)
    )
    if meta.empty:
        return

    n = len(meta)
    ncols = 3 if n <= 9 else 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.9 * nrows), squeeze=False)

    for ax, (_, row) in zip(axes.flatten(), meta.iterrows()):
        pid = row["path_id"]
        lam = float(row["lambda"])
        g = path_df(pid).dropna(subset=["state", "asset_excess"]).copy()
        g["state"] = g["state"].astype(int)
        g["level"] = 100.0 * (1.0 + g["asset_excess"].fillna(0.0)).cumprod()
        ax.plot(g["date"], g["level"], color="black", linewidth=1.05, zorder=3)
        add_state_background_runs(ax, g, "date", "state")

        r = teacher_path_diagnostics.loc[teacher_path_diagnostics["path_id"].eq(pid)]
        if len(r):
            rr = r.iloc[0]
            plot_title = (
                f"lambda={lam:g} | good={100 * rr['good_share']:0.0f}% | "
                f"sw={int(rr['n_switches'])} | spell={rr['mean_spell_length']:0.1f}m\n"
                f"mean G/B {100 * rr['state_good_ann_return']:0.1f}/{100 * rr['state_bad_ann_return']:0.1f}% | "
                f"Sharpe {rr['state_good_sharpe']:0.2f}/{rr['state_bad_sharpe']:0.2f}"
            )
        else:
            plot_title = f"lambda={lam:g}"
        ax.set_title(plot_title, fontsize=9, loc="left")
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.flatten()[len(meta):]:
        ax.axis("off")

    axes.flatten()[0].legend(
        handles=[
            Patch(facecolor=STATE_COLORS[1], edgecolor="none", alpha=0.60, label="Favourable"),
            Patch(facecolor=STATE_COLORS[0], edgecolor="none", alpha=0.60, label="Unfavourable"),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.suptitle(f"{asset.title()} {feature_group} label paths - {MODEL_LABELS.get(model_family, model_family)}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


for asset, feature_group in teacher_panel_registry.keys():
    for model in MODEL_FAMILIES:
        if RUN_LABEL_PATH_PLOTS:
            plot_all_lambda_label_grid(asset, feature_group, model)


def plot_lambda_diagnostics(asset, feature_group):
    h = teacher_path_diagnostics.loc[
        teacher_path_diagnostics["asset"].eq(asset)
        & teacher_path_diagnostics["feature_group"].eq(feature_group)
    ].copy()
    if h.empty:
        return
    metrics = [
        ("good_share", "Favourable-state share"),
        ("n_switches", "Number of switches"),
        ("mean_good_spell_length", "Mean favourable spell"),
        ("mean_bad_spell_length", "Mean unfavourable spell"),
        ("good_minus_bad_ann_return", "Favourable - unfavourable annual return"),
        ("good_minus_bad_sharpe", "Favourable - unfavourable Sharpe"),
        ("state_bad_ann_vol", "Unfavourable-state annual volatility"),
        ("median_centroid_separation", "Median centroid separation"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(18, 15))
    axes = axes.flatten()
    for ax, (col, plot_title) in zip(axes, metrics):
        for model in MODEL_FAMILIES:
            g = h.loc[h["model_family"].eq(model)].sort_values("lambda").copy()
            if g.empty or col not in g.columns:
                continue
            x = np.arange(len(g))
            ax.plot(x, g[col], marker="o", linewidth=1.3, label=MODEL_LABELS.get(model, model))
            ax.set_xticks(x)
            ax.set_xticklabels([f"{float(v):g}" for v in g["lambda"]], rotation=45, ha="right", fontsize=8)
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"{asset.title()} {feature_group} lambda diagnostics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


for asset, feature_group in teacher_panel_registry.keys():
    if RUN_LAMBDA_DIAGNOSTIC_PLOTS:
        plot_lambda_diagnostics(asset, feature_group)


duration_rows = []
for pid, g in teacher_monthly_labels_long.groupby("path_id", sort=True):
    g = g.sort_values("date").dropna(subset=["state"]).copy()
    s = g["state"].astype(int)
    last_state, current_spell_len = current_spell(s)
    trans = transition_stats(s)
    switch_1m = (s.shift(-1) != s).astype(float)
    switch_1m.iloc[-1] = np.nan
    st = s.reset_index(drop=True)
    switch_3m = []
    for i in range(len(st)):
        future = st.iloc[i + 1 : i + 4]
        switch_3m.append(np.nan if len(future) == 0 else float((future != st.iloc[i]).any()))
    meta = g.iloc[0]
    duration_rows.append({
        "path_id": pid,
        "asset": meta["asset"],
        "feature_group": meta["feature_group"],
        "model_family": meta["model_family"],
        "lambda": float(meta["lambda"]),
        "last_state": int(last_state),
        "current_spell_length": int(current_spell_len),
        "mean_spell_length": mean_spell_length(s),
        "mean_good_spell_length": mean_spell_length_for_state(s, 1),
        "mean_bad_spell_length": mean_spell_length_for_state(s, 0),
        "switch_next_1m_rate": float(np.nanmean(switch_1m)),
        "switch_next_3m_rate": float(np.nanmean(switch_3m)),
        **trans,
    })

teacher_duration_diagnostics = pd.DataFrame(duration_rows).sort_values(["asset", "feature_group", "model_family", "lambda"]).reset_index(drop=True)

crisis_rows = []
for pid, g in teacher_monthly_labels_long.groupby("path_id", sort=True):
    g = g.sort_values("date").copy()
    meta = g.iloc[0]
    for window, start, end in CRISIS_WINDOWS:
        h = g.loc[g["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if h.empty:
            continue
        r = h["asset_excess"]
        crisis_rows.append({
            "window": window,
            "path_id": pid,
            "asset": meta["asset"],
            "feature_group": meta["feature_group"],
            "model_family": meta["model_family"],
            "lambda": float(meta["lambda"]),
            "n_months": int(len(h)),
            "good_share": float(h["state"].mean()),
            "bad_share": float(1.0 - h["state"].mean()),
            "ann_return": ann_mean(r, 12.0),
            "ann_vol": ann_vol(r, 12.0),
            "sharpe": ann_sharpe(r, 12.0),
            "worst_month": float(r.min()) if len(r.dropna()) else np.nan,
            "cvar_10pct": cvar(r, 0.10),
            "max_drawdown": max_drawdown(r),
        })

teacher_crisis_diagnostics = pd.DataFrame(crisis_rows).sort_values(["asset", "feature_group", "model_family", "lambda", "window"]).reset_index(drop=True)
teacher_crisis_window_summary = teacher_crisis_diagnostics.loc[
    teacher_crisis_diagnostics["window"].isin(["1994 bond shock", "Dot-com bust", "GFC", "Covid crash", "Inflation 2022", "Recent 2024-2026"])
].copy()


def plot_majority_asset_feature_group(asset, feature_group):
    h = majority_vote_labels_long.loc[
        majority_vote_labels_long["asset"].eq(asset)
        & majority_vote_labels_long["feature_group"].eq(feature_group)
    ].copy()
    if h.empty:
        return
    scopes = [s for s in ["l2_only", "l1_medoids_only", "all_models"] if s in h["majority_scope"].unique()]
    fig, axes = plt.subplots(len(scopes), 1, figsize=(17.5, 4.0 * len(scopes)), squeeze=False)
    for i, scope in enumerate(scopes):
        ax = axes[i, 0]
        g = h.loc[h["majority_scope"].eq(scope)].dropna(subset=["majority_state", "asset_excess"]).sort_values("date").copy()
        g["level"] = 100.0 * (1.0 + g["asset_excess"].fillna(0.0)).cumprod()
        ax.plot(g["date"], g["level"], color="black", linewidth=1.15, zorder=3)
        add_state_background_runs(ax, g, "date", "majority_state")
        m = majority_vote_metrics.loc[
            majority_vote_metrics["asset"].eq(asset)
            & majority_vote_metrics["feature_group"].eq(feature_group)
            & majority_vote_metrics["majority_scope"].eq(scope)
        ]
        if len(m):
            r = m.iloc[0]
            stats = (
                f"fav={100*r['good_share']:0.1f}% | sw={int(r['n_switches'])} | "
                f"spell={r['mean_spell_length']:0.1f} | F/U spell={r['mean_good_spell_length']:0.1f}/{r['mean_bad_spell_length']:0.1f} | "
                f"mean F/U={100*r['state_good_ann_return']:0.1f}/{100*r['state_bad_ann_return']:0.1f}% | "
                f"Sharpe F/U={r['state_good_sharpe']:0.2f}/{r['state_bad_sharpe']:0.2f} | "
                f"vote strength={100*r['avg_vote_strength']:0.1f}%"
            )
        else:
            stats = ""
        ax.set_title(f"{asset.title()} {feature_group} majority vote - {scope}\n{stats}", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(g["date"], g["vote_prob"], linewidth=0.9, alpha=0.75)
        ax2.axhline(0.5, linestyle="--", linewidth=0.8, alpha=0.60)
        ax2.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(
        handles=[
            Patch(facecolor=STATE_COLORS[1], edgecolor="none", alpha=0.60, label="Favourable majority"),
            Patch(facecolor=STATE_COLORS[0], edgecolor="none", alpha=0.60, label="Unfavourable majority"),
        ],
        loc="upper left",
        fontsize=9,
    )
    fig.suptitle(f"{asset.title()} {feature_group} feature-group majority labels", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    plt.show()


if RUN_MAJORITY_VOTE_PLOTS:
    for asset, fg in majority_vote_labels_long[["asset", "feature_group"]].drop_duplicates().itertuples(index=False, name=None):
        plot_majority_asset_feature_group(asset, fg)



raw_feature_rows = []

for (asset, feature_group), df in teacher_panel_registry.items():
    feats = teacher_feature_sets[(asset, feature_group)]
    for f in feats:
        v = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw_feature_rows.append({
            "asset": asset,
            "feature_group": feature_group,
            "feature": f,
            "n_obs": int(v.notna().sum()),
            "missing": int(v.isna().sum()),
            "missing_pct": float(v.isna().mean()),
            "mean": float(v.mean()) if v.notna().any() else np.nan,
            "std": float(v.std(ddof=1)) if v.notna().sum() >= 2 else np.nan,
            "min": float(v.min()) if v.notna().any() else np.nan,
            "p01": float(v.quantile(0.01)) if v.notna().any() else np.nan,
            "p05": float(v.quantile(0.05)) if v.notna().any() else np.nan,
            "p50": float(v.quantile(0.50)) if v.notna().any() else np.nan,
            "p95": float(v.quantile(0.95)) if v.notna().any() else np.nan,
            "p99": float(v.quantile(0.99)) if v.notna().any() else np.nan,
            "max": float(v.max()) if v.notna().any() else np.nan,
            "max_abs": float(v.abs().max()) if v.notna().any() else np.nan,
            "max_abs_over_p99_abs": float(v.abs().max() / max(v.abs().quantile(0.99), EPS)) if v.notna().any() else np.nan,
        })

teacher_majority_vote_labels_long = majority_vote_labels_long
teacher_majority_vote_labels_wide = majority_vote_labels_wide
teacher_majority_vote_prediction_targets_h1 = majority_vote_prediction_targets_h1
majority_wide = majority_vote_labels_wide

teacher_monthly_labels_long.to_pickle(MODEL_OUTPUT_DIR / "teacher_monthly_labels_long.pkl")
teacher_monthly_labels_long.to_csv(MODEL_OUTPUT_DIR / "teacher_Monthly Teacher Labels Long.csv", index=False)

teacher_monthly_labels_wide.to_pickle(MODEL_OUTPUT_DIR / "teacher_monthly_labels_wide.pkl")
teacher_monthly_labels_wide.to_csv(MODEL_OUTPUT_DIR / "teacher_Monthly Teacher Labels Wide.csv", index=False)

teacher_prediction_targets_h1_all_paths.to_pickle(MODEL_OUTPUT_DIR / "teacher_prediction_targets_h1_all_paths.pkl")
teacher_prediction_targets_h1_all_paths.to_csv(MODEL_OUTPUT_DIR / "teacher_One Month Ahead Prediction Targets All Paths.csv", index=False)

teacher_fit_log.to_pickle(MODEL_OUTPUT_DIR / "teacher_fit_log.pkl")
teacher_fit_log.to_csv(MODEL_OUTPUT_DIR / "teacher_Teacher Fit Log.csv", index=False)

teacher_path_diagnostics.to_pickle(MODEL_OUTPUT_DIR / "teacher_path_diagnostics.pkl")
teacher_path_diagnostics.to_csv(MODEL_OUTPUT_DIR / "teacher_Teacher Path Diagnostics.csv", index=False)

teacher_feature_separation_by_path.to_pickle(MODEL_OUTPUT_DIR / "teacher_feature_separation_by_path.pkl")
teacher_feature_separation_by_path.to_csv(MODEL_OUTPUT_DIR / "teacher_Teacher Feature Separation by Path.csv", index=False)

teacher_feature_separation_summary.to_pickle(MODEL_OUTPUT_DIR / "teacher_feature_separation_summary.pkl")
teacher_feature_separation_summary.to_csv(MODEL_OUTPUT_DIR / "teacher_Teacher Feature Separation Summary.csv", index=False)

teacher_duration_diagnostics.to_pickle(MODEL_OUTPUT_DIR / "teacher_duration_diagnostics.pkl")
teacher_duration_diagnostics.to_csv(MODEL_OUTPUT_DIR / "teacher_Teacher Duration Diagnostics.csv", index=False)

teacher_crisis_diagnostics.to_pickle(MODEL_OUTPUT_DIR / "teacher_crisis_diagnostics.pkl")
teacher_crisis_diagnostics.to_csv(MODEL_OUTPUT_DIR / "teacher_crisis_diagnostics.csv", index=False)

teacher_crisis_window_summary.to_pickle(MODEL_OUTPUT_DIR / "teacher_crisis_window_summary.pkl")
teacher_crisis_window_summary.to_csv(MODEL_OUTPUT_DIR / "teacher_Teacher Crisis Window Summary.csv", index=False)

majority_vote_labels_long.to_pickle(MAJORITY_VOTE_OUTPUT_DIR / "majority_vote_labels_long.pkl")
majority_vote_labels_long.to_csv(MAJORITY_VOTE_OUTPUT_DIR / "Majority Vote Labels Long.csv", index=False)

majority_vote_labels_wide.to_pickle(MAJORITY_VOTE_OUTPUT_DIR / "majority_vote_labels_wide.pkl")
majority_vote_labels_wide.to_csv(MAJORITY_VOTE_OUTPUT_DIR / "Majority Vote Labels Wide.csv", index=False)

majority_vote_prediction_targets_h1.to_pickle(MAJORITY_VOTE_OUTPUT_DIR / "majority_vote_prediction_targets_h1.pkl")
majority_vote_prediction_targets_h1.to_csv(MAJORITY_VOTE_OUTPUT_DIR / "Majority Vote One Month Ahead Prediction Targets.csv", index=False)

majority_vote_metrics.to_pickle(MODEL_OUTPUT_DIR / "majority_vote_metrics.pkl")
majority_vote_metrics.to_csv(MODEL_OUTPUT_DIR / "Majority Vote Label Metrics.csv", index=False)

majority_vote_path_summary.to_pickle(MODEL_OUTPUT_DIR / "majority_vote_path_summary.pkl")
majority_vote_path_summary.to_csv(MODEL_OUTPUT_DIR / "Majority Vote Path Summary.csv", index=False)
