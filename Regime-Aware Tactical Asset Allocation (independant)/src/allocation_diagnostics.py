import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        brier_score_loss,
        log_loss,
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ==================================================================================================
# 0. CONFIG
# ==================================================================================================

FEATURE_IMPORTANCE_CUM_SHARE_THRESHOLD = 0.80
FEATURE_IMPORTANCE_MIN_FEATURES = 8
FEATURE_IMPORTANCE_MAX_FEATURES = 18
FEATURE_IMPORTANCE_PLOT = True
PREDICTION_DIAGNOSTIC_PLOT = True

FEATURE_DESIGNS_TO_USE = ["z_smooth_only", "z_plus_zsmooth"]
FAMILIES_TO_USE = ["LOGIT", "RF"]
TARGETS_TO_USE = [
    "bd_bond_canonical_state_h1",
    "eq_return_sortino_state_h1",
    "eq_vol_downside_state_h1",
]

ALLOCATION_PAIRS_TO_USE = ["return_sortino_vs_bond"]
BUDGET_MODES_TO_USE = ["cash_allowed"]

GAMMA_GRID = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
TCOST_GRID = [0.0005, 0.0010, 0.0025, 0.0050]
LAMBDA_TRADE_GRID = [0.0, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
PROB_SMOOTH_HALFLIFE_GRID = [0, 3, 6]

WEIGHT_GRID_STEP = 0.02
MIN_STATE_OBS = 24
MOMENT_SHRINK_N = 120
MOMENT_RIDGE = 1e-6

# Table/plot limits
PRINT_TOP_N_FEATURES_PER_COMBO = 8
PRINT_TOP_N_STRATEGIES = 25
MAX_ALLOCATION_PLOTS = 14
TURNOVER_FRONTIER_CAPS = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

MIN_ANN_VOL = 0.045
MAX_AVG_CASH = 0.85
MIN_AVG_EQ = 0.10
MIN_BOTH_BAD_CASH = 0.25

# ==================================================================================================
# 1. SMALL UTILITIES
# ==================================================================================================

def _first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _canonical_target(x):
    s = str(x).lower()
    if "return_sortino" in s:
        return "eq_return_sortino_state_h1"
    if "vol_downside" in s:
        return "eq_vol_downside_state_h1"
    if "bond_canonical" in s or "bd_bond" in s or "bond_bond" in s:
        return "bd_bond_canonical_state_h1"
    return str(x)


def _target_short(x):
    x = _canonical_target(x)
    return {
        "eq_return_sortino_state_h1": "eq_return_sortino",
        "eq_vol_downside_state_h1": "eq_vol_downside",
        "bd_bond_canonical_state_h1": "bd_bond_canonical",
    }.get(x, str(x))


def _fmt_param(x):
    if pd.isna(x):
        return "nan"
    x = float(x)
    s = f"{x:g}".replace("-", "m").replace(".", "p")
    return s


def _safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def print_table(title, df, cols=None, sort_by=None, ascending=False, n=20, floatfmt=4):
    print("\n" + "=" * 160)
    print(title)
    print("=" * 160)
    if df is None or len(df) == 0:
        print("EMPTY")
        return
    out = df.copy()
    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)
    if cols is not None:
        cols = [c for c in cols if c in out.columns]
        out = out[cols]
    if n is not None:
        out = out.head(n)
    float_cols = out.select_dtypes(include=[np.number]).columns
    with pd.option_context("display.max_columns", 200, "display.width", 240):
        for c in float_cols:
            out[c] = out[c].astype(float).round(floatfmt)
        print(out.to_string(index=False))


def _wealth_index(r, start=100.0):
    r = pd.Series(r).fillna(0.0).astype(float)
    return start * (1.0 + r).cumprod()


def _drawdown_from_wealth(w):
    w = pd.Series(w).astype(float)
    return w / w.cummax() - 1.0


def _cvar_left_tail(r, q=0.01):
    r = pd.Series(r).dropna().astype(float)
    if len(r) == 0:
        return np.nan
    cutoff = r.quantile(q)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def _perf_stats(r):
    r = pd.Series(r).dropna().astype(float)
    if len(r) == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "cvar_1pct_monthly": np.nan,
            "final_wealth": np.nan,
        }
    wealth = _wealth_index(r)
    ann_return = float((wealth.iloc[-1] / wealth.iloc[0]) ** (12.0 / max(len(r), 1)) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(12.0)) if len(r) > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(_drawdown_from_wealth(wealth).min()),
        "cvar_1pct_monthly": _cvar_left_tail(r, 0.01),
        "final_wealth": float(wealth.iloc[-1]),
    }


def _normalize_feature_design_family(df):
    df = df.copy()
    if "feature_design" not in df.columns:
        c = _first_existing_col(df, ["feature_set", "design", "feature_design_name"])
        if c is not None:
            df = df.rename(columns={c: "feature_design"})
    if "family" not in df.columns:
        c = _first_existing_col(df, ["model", "model_family", "estimator", "learner"])
        if c is not None:
            df = df.rename(columns={c: "family"})
    if "target" not in df.columns:
        c = _first_existing_col(df, ["label", "target_name", "y_name"])
        if c is not None:
            df = df.rename(columns={c: "target"})
    if "feature_design" in df.columns:
        df["feature_design"] = df["feature_design"].astype(str)
    if "family" in df.columns:
        df["family"] = df["family"].astype(str).str.upper()
    if "target" in df.columns:
        df["target"] = df["target"].map(_canonical_target)
    return df

# ==================================================================================================
# 2. MODEL-SPECIFIC FEATURE IMPORTANCE - RF + LOGIT, ALL LABELS
# ==================================================================================================

def prepare_feature_importance():
    if "student_feature_importance_summary" in globals():
        fi = student_feature_importance_summary.copy()
    elif "student_feature_importance_long" in globals():
        raw = student_feature_importance_long.copy()
        raw = _normalize_feature_design_family(raw)
        if "feature" not in raw.columns:
            c = _first_existing_col(raw, ["variable", "feature_name", "x"])
            if c is not None:
                raw = raw.rename(columns={c: "feature"})
        abs_col = _first_existing_col(raw, ["abs_importance", "importance_abs", "abs_coefficient", "abs_coef", "mean_abs_importance"])
        signed_col = _first_existing_col(raw, ["signed_importance", "importance", "coefficient", "coef", "mean_signed_importance"])
        if abs_col is None and signed_col is not None:
            raw["_abs_importance"] = pd.to_numeric(raw[signed_col], errors="coerce").abs()
            abs_col = "_abs_importance"
        if signed_col is None and abs_col is not None:
            raw["_signed_importance"] = pd.to_numeric(raw[abs_col], errors="coerce")
            signed_col = "_signed_importance"
        group_cols = [c for c in ["feature_design", "family", "target", "feature", "feature_block", "feature_transform"] if c in raw.columns]
        if abs_col is None:
            raise RuntimeError("Could not find feature-importance column in student_feature_importance_long.")
        agg_map = {
            abs_col: "mean",
        }
        if signed_col is not None:
            agg_map[signed_col] = "mean"
        fi = raw.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
        fi = fi.rename(columns={abs_col: "mean_abs_importance"})
        if signed_col is not None:
            fi = fi.rename(columns={signed_col: "mean_signed_importance"})
    else:
        raise RuntimeError("Need student_feature_importance_summary or student_feature_importance_long from student prediction stage.")

    fi = _normalize_feature_design_family(fi)

    if "feature" not in fi.columns:
        c = _first_existing_col(fi, ["variable", "feature_name", "x"])
        if c is not None:
            fi = fi.rename(columns={c: "feature"})

    if "mean_abs_importance" not in fi.columns:
        abs_col = _first_existing_col(fi, ["abs_importance", "importance_abs", "abs_coefficient", "abs_coef"])
        signed_col = _first_existing_col(fi, ["mean_signed_importance", "signed_importance", "importance", "coefficient", "coef"])
        if abs_col is not None:
            fi["mean_abs_importance"] = pd.to_numeric(fi[abs_col], errors="coerce")
        elif signed_col is not None:
            fi["mean_abs_importance"] = pd.to_numeric(fi[signed_col], errors="coerce").abs()
        else:
            raise RuntimeError("Could not construct mean_abs_importance.")

    if "mean_signed_importance" not in fi.columns:
        signed_col = _first_existing_col(fi, ["signed_importance", "importance", "coefficient", "coef"])
        if signed_col is not None:
            fi["mean_signed_importance"] = pd.to_numeric(fi[signed_col], errors="coerce")
        else:
            fi["mean_signed_importance"] = np.nan

    if "feature_block" not in fi.columns:
        fi["feature_block"] = "unknown"
    if "feature_transform" not in fi.columns:
        fi["feature_transform"] = np.where(fi["feature"].astype(str).str.contains("z_smooth"), "z_smooth", "z")

    fi = fi[
        fi["feature_design"].isin(FEATURE_DESIGNS_TO_USE)
        & fi["family"].isin(FAMILIES_TO_USE)
        & fi["target"].isin(TARGETS_TO_USE)
    ].copy()

    fi["mean_abs_importance"] = pd.to_numeric(fi["mean_abs_importance"], errors="coerce").fillna(0.0)
    fi["mean_signed_importance"] = pd.to_numeric(fi["mean_signed_importance"], errors="coerce")

    group_cols = ["feature_design", "family", "target"]
    fi["total_abs_importance"] = fi.groupby(group_cols)["mean_abs_importance"].transform("sum")
    fi["importance_share"] = np.where(
        fi["total_abs_importance"] > 0,
        fi["mean_abs_importance"] / fi["total_abs_importance"],
        np.nan,
    )
    fi = fi.sort_values(group_cols + ["mean_abs_importance"], ascending=[True, True, True, False])
    fi["importance_rank"] = fi.groupby(group_cols)["mean_abs_importance"].rank(method="first", ascending=False).astype(int)
    fi["cum_importance_share"] = fi.groupby(group_cols)["importance_share"].cumsum()

    selected_rank = []
    for _, g in fi.groupby(group_cols, dropna=False):
        g = g.sort_values("importance_rank")
        hit = g[g["cum_importance_share"] >= FEATURE_IMPORTANCE_CUM_SHARE_THRESHOLD]
        k = int(hit["importance_rank"].iloc[0]) if len(hit) else min(len(g), FEATURE_IMPORTANCE_MAX_FEATURES)
        k = max(k, FEATURE_IMPORTANCE_MIN_FEATURES)
        k = min(k, FEATURE_IMPORTANCE_MAX_FEATURES, len(g))
        selected_rank.extend(list(g.index[g["importance_rank"] <= k]))
    fi["selected_for_plot"] = fi.index.isin(selected_rank)

    return fi


def plot_feature_importance_one(fi, feature_design, family, target):
    g = fi[
        fi["feature_design"].eq(feature_design)
        & fi["family"].eq(family)
        & fi["target"].eq(target)
    ].copy()
    if len(g) == 0:
        return

    g = g.sort_values("importance_rank").head(FEATURE_IMPORTANCE_MAX_FEATURES).copy()
    k_selected = int(g["selected_for_plot"].sum())
    g_plot = g.iloc[::-1].copy()

    colors = ["#1f77b4" if bool(x) else "#bdbdbd" for x in g_plot["selected_for_plot"]]
    fig_h = max(5.0, 0.33 * len(g_plot) + 2.2)
    fig, ax = plt.subplots(figsize=(15, fig_h))

    y = np.arange(len(g_plot))
    ax.barh(y, g_plot["importance_share"].values, color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(g_plot["feature"].astype(str).values, fontsize=9)
    ax.set_xlabel("Relative model-specific importance within this model-target combination")
    ax.set_title(
        f"FEATURE IMPORTANCE | {feature_design} | {family} | {_target_short(target)}\n"
        f"Selected features = minimum set reaching {FEATURE_IMPORTANCE_CUM_SHARE_THRESHOLD:.0%} cumulative importance "
        f"with min={FEATURE_IMPORTANCE_MIN_FEATURES}, max={FEATURE_IMPORTANCE_MAX_FEATURES}"
    )

    for yy, share, cum, rank in zip(
        y,
        g_plot["importance_share"].values,
        g_plot["cum_importance_share"].values,
        g_plot["importance_rank"].values,
    ):
        ax.text(share + 0.001, yy, f"rank {rank} | cum {cum:.0%}", va="center", fontsize=8)

    # separator between selected and not selected, when both appear in plotted sample
    if 0 < k_selected < len(g):
        sep_y = len(g_plot) - k_selected - 0.5
        ax.axhline(sep_y, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(ax.get_xlim()[1] * 0.97, sep_y + 0.15, "selection threshold", ha="right", va="bottom", fontsize=9)

    if family.upper() == "LOGIT":
        formula = (
            r"Model-specific importance: mean absolute standardized LOGIT coefficient, "
            r"$I_j = |\mathcal{T}|^{-1}\sum_t |\hat\beta_{j,t}|$; share $s_j=I_j/\sum_k I_k$. "
            r"Descriptive, not statistical significance; correlated predictors can split importance."
        )
    else:
        formula = (
            r"Model-specific importance: mean RF impurity-decrease importance (MDI), "
            r"$I_j = |\mathcal{T}|^{-1}\sum_t \widehat{MDI}_{j,t}$; share $s_j=I_j/\sum_k I_k$. "
            r"Standard descriptive measure; correlated predictors can split importance."
        )
    fig.text(0.5, 0.015, formula, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    plt.show()

# ==================================================================================================
# 3. PREDICTION PANEL + ACCURACY / SEED-VARIATION DIAGNOSTICS
# ==================================================================================================

def _find_prediction_probability_col(df):
    return _first_existing_col(
        df,
        [
            "p_good", "p_good_state", "predicted_probability", "pred_prob", "prob", "probability",
            "y_pred_proba", "prediction", "p_hat", "mean_prob", "prob_mean",
        ],
    )


def _find_truth_col(df):
    return _first_existing_col(
        df,
        [
            "y_true", "actual", "actual_state", "realized_state", "label_value", "target_value",
            "state", "y", "true_label",
        ],
    )


def _get_truth_wide_from_existing():
    candidates = []
    if "final_h1_target_wide_for_student" in globals():
        candidates.append(final_h1_target_wide_for_student.copy())
    if "final_h1_historical_label_return_panel_for_allocation" in globals():
        candidates.append(final_h1_historical_label_return_panel_for_allocation.copy())
    if "historical_targets_h1" in globals():
        candidates.append(historical_targets_h1.copy())

    for x in candidates:
        if "date" not in x.columns:
            continue
        x = x.copy()
        x["date"] = pd.to_datetime(x["date"])
        rename = {}
        for c in x.columns:
            cc = _canonical_target(c)
            if cc in TARGETS_TO_USE and c != cc:
                rename[c] = cc
        x = x.rename(columns=rename)
        keep = ["date"] + [c for c in TARGETS_TO_USE if c in x.columns]
        if len(keep) > 1:
            return x[keep].drop_duplicates("date")
    return None



def _seed_probability_dispersion_from_diagnostics(pred_agg):
    """
    Recover genuine per-seed test probabilities from student prediction stage diagnostics.

    student prediction stage persisted one-observation Brier scores for every seed/date but stored
    only the seed-averaged probability in the prediction panel. For binary y and
    one observation, Brier=(p-y)^2, so p is recovered exactly from Brier and y.
    If a future student prediction stage stores per-seed probabilities directly, those are used.
    """
    diag = None
    for name in ["student_seed_diagnostics", "student_seed_diagnostics"]:
        candidate = globals().get(name)
        if isinstance(candidate, pd.DataFrame) and len(candidate):
            diag = candidate.copy()
            break
    if diag is None:
        return pd.DataFrame()

    diag = _normalize_feature_design_family(diag)
    date_col = _first_existing_col(diag, ["date", "test_date", "prediction_date"])
    seed_col = _first_existing_col(diag, ["seed", "random_seed", "model_seed"])
    if date_col is None or seed_col is None:
        return pd.DataFrame()

    diag["date"] = pd.to_datetime(diag[date_col], errors="coerce")
    if "split" in diag.columns:
        selected = diag["split"].astype(str).eq("test_selected_model")
        if selected.any():
            diag = diag.loc[selected].copy()

    group_cols = ["feature_design", "family", "target", "date"]
    truth = pred_agg[group_cols + ["y_true"]].drop_duplicates(group_cols)
    diag = diag.merge(truth, on=group_cols, how="inner")

    direct_p_col = _first_existing_col(
        diag,
        ["p_seed", "seed_probability", "predicted_probability", "p_hat", "probability"],
    )
    if direct_p_col is not None:
        diag["p_seed_recovered"] = pd.to_numeric(diag[direct_p_col], errors="coerce")
    else:
        brier_col = _first_existing_col(diag, ["brier", "brier_score"])
        n_col = _first_existing_col(diag, ["n", "n_obs"])
        if brier_col is None:
            return pd.DataFrame()
        if n_col is not None:
            diag = diag.loc[pd.to_numeric(diag[n_col], errors="coerce").eq(1)].copy()
        root_brier = np.sqrt(np.clip(pd.to_numeric(diag[brier_col], errors="coerce"), 0.0, 1.0))
        y = pd.to_numeric(diag["y_true"], errors="coerce")
        diag["p_seed_recovered"] = np.where(y.eq(1), 1.0 - root_brier, root_brier)

    diag["p_seed_recovered"] = pd.to_numeric(diag["p_seed_recovered"], errors="coerce").clip(0.0, 1.0)
    diag = diag.dropna(subset=["p_seed_recovered"])
    if diag.empty:
        return pd.DataFrame()

    return (
        diag.groupby(group_cols, dropna=False)
        .agg(
            p_seed_std=("p_seed_recovered", "std"),
            p_seed_min=("p_seed_recovered", "min"),
            p_seed_max=("p_seed_recovered", "max"),
            n_seed=("p_seed_recovered", "count"),
        )
        .reset_index()
        .assign(p_seed_std=lambda x: x["p_seed_std"].fillna(0.0))
    )


def prepare_prediction_panel():
    if "student_oos_prediction_panel_long" in globals():
        pred = student_oos_prediction_panel_long.copy()
    elif "student_rf_prediction_panel_long" in globals():
        pred = student_rf_prediction_panel_long.copy()
    else:
        raise RuntimeError("Need student_oos_prediction_panel_long from student prediction stage.")

    pred = _normalize_feature_design_family(pred)

    if "date" not in pred.columns:
        c = _first_existing_col(pred, ["month", "prediction_date", "test_date"])
        if c is not None:
            pred = pred.rename(columns={c: "date"})
        else:
            raise RuntimeError("Prediction panel has no date column.")

    p_col = _find_prediction_probability_col(pred)
    if p_col is None:
        raise RuntimeError("Could not find predicted probability column in student_oos_prediction_panel_long.")

    y_col = _find_truth_col(pred)
    seed_col = _first_existing_col(pred, ["seed", "random_seed", "model_seed"])

    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["p_raw"] = pd.to_numeric(pred[p_col], errors="coerce")

    if y_col is not None:
        pred["y_raw"] = pd.to_numeric(pred[y_col], errors="coerce")
    else:
        pred["y_raw"] = np.nan

    pred = pred[
        pred["feature_design"].isin(FEATURE_DESIGNS_TO_USE)
        & pred["family"].isin(FAMILIES_TO_USE)
        & pred["target"].isin(TARGETS_TO_USE)
    ].copy()

    group_cols = ["feature_design", "family", "target", "date"]

    if seed_col is not None and seed_col in pred.columns:
        agg = pred.groupby(group_cols, dropna=False).agg(
            p_mean=("p_raw", "mean"),
            p_seed_std=("p_raw", "std"),
            p_seed_min=("p_raw", "min"),
            p_seed_max=("p_raw", "max"),
            n_seed=("p_raw", "count"),
            y_true=("y_raw", "first"),
        ).reset_index()
        agg["p_seed_std"] = agg["p_seed_std"].fillna(0.0)
    else:
        agg = pred.groupby(group_cols, dropna=False).agg(
            p_mean=("p_raw", "mean"),
            p_seed_std=("p_raw", "std"),
            p_seed_min=("p_raw", "min"),
            p_seed_max=("p_raw", "max"),
            n_seed=("p_raw", "count"),
            y_true=("y_raw", "first"),
        ).reset_index()
        agg["p_seed_std"] = agg["p_seed_std"].fillna(0.0)

    truth_wide = _get_truth_wide_from_existing()
    if truth_wide is not None:
        long_truth = truth_wide.melt("date", var_name="target", value_name="y_from_truth_wide")
        long_truth["target"] = long_truth["target"].map(_canonical_target)
        agg = agg.merge(long_truth, on=["date", "target"], how="left")
        agg["y_true"] = np.where(
            agg["y_true"].notna(),
            agg["y_true"],
            pd.to_numeric(agg["y_from_truth_wide"], errors="coerce"),
        )
        agg = agg.drop(columns=["y_from_truth_wide"], errors="ignore")

    agg["y_true"] = pd.to_numeric(agg["y_true"], errors="coerce")

    seed_dispersion = _seed_probability_dispersion_from_diagnostics(agg)
    if len(seed_dispersion):
        seed_cols = ["p_seed_std", "p_seed_min", "p_seed_max", "n_seed"]
        agg = agg.drop(columns=seed_cols, errors="ignore").merge(
            seed_dispersion,
            on=group_cols,
            how="left",
        )
        agg["p_seed_std"] = pd.to_numeric(agg["p_seed_std"], errors="coerce").fillna(0.0)

    agg["y_pred"] = (agg["p_mean"] >= 0.5).astype(int)

    return agg


def _classification_metrics(y, p):
    d = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(d) == 0:
        return {}
    y = d["y"].astype(int).values
    p = np.clip(d["p"].astype(float).values, 1e-6, 1 - 1e-6)
    yhat = (p >= 0.5).astype(int)

    if _HAS_SKLEARN:
        out = {
            "n_obs": len(y),
            "actual_good_share": float(np.mean(y)),
            "pred_good_share": float(np.mean(yhat)),
            "accuracy": float(accuracy_score(y, yhat)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yhat)) if len(np.unique(y)) == 2 else np.nan,
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
        return out

    tp = np.sum((y == 1) & (yhat == 1))
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n_obs": len(y),
        "actual_good_share": float(np.mean(y)),
        "pred_good_share": float(np.mean(yhat)),
        "accuracy": float(np.mean(y == yhat)),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": np.nan,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }


def build_prediction_metrics(pred_agg):
    rows = []
    for keys, g in pred_agg.groupby(["feature_design", "family", "target"], dropna=False):
        feature_design, family, target = keys
        met = _classification_metrics(g["y_true"], g["p_mean"])
        if not met:
            continue
        met.update({
            "feature_design": feature_design,
            "family": family,
            "target": target,
            "avg_seed_prob_std": float(pd.to_numeric(g["p_seed_std"], errors="coerce").mean()),
            "max_seed_prob_std": float(pd.to_numeric(g["p_seed_std"], errors="coerce").max()),
        })
        rows.append(met)
    out = pd.DataFrame(rows)
    if len(out):
        ordered = ["feature_design", "family", "target"] + [c for c in out.columns if c not in ["feature_design", "family", "target"]]
        out = out[ordered]
    return out


def _shade_real_regimes(ax, d, y_col="y_true"):
    d = d.sort_values("date").copy()
    dates = pd.to_datetime(d["date"]).values
    y = pd.to_numeric(d[y_col], errors="coerce").values
    if len(dates) == 0:
        return
    for i in range(len(dates)):
        start = pd.Timestamp(dates[i])
        if i < len(dates) - 1:
            end = pd.Timestamp(dates[i + 1])
        else:
            end = start + pd.offsets.MonthEnd(1)
        color = "#2ca02c" if y[i] == 1 else "#d62728"
        ax.axvspan(start, end, color=color, alpha=0.075, lw=0)


def plot_prediction_one(pred_agg, metrics_df, feature_design, family, target):
    d = pred_agg[
        pred_agg["feature_design"].eq(feature_design)
        & pred_agg["family"].eq(family)
        & pred_agg["target"].eq(target)
    ].copy()
    if len(d) == 0:
        return
    d = d.sort_values("date")

    m = metrics_df[
        metrics_df["feature_design"].eq(feature_design)
        & metrics_df["family"].eq(family)
        & metrics_df["target"].eq(target)
    ]
    m = m.iloc[0].to_dict() if len(m) else {}

    fig = plt.figure(figsize=(19, 5.6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[4.8, 4.8, 0.25, 2.25])
    ax = fig.add_subplot(gs[0, :3])
    ax_box = fig.add_subplot(gs[0, 3])
    ax_box.axis("off")

    _shade_real_regimes(ax, d, "y_true")
    ax.plot(d["date"], d["p_mean"], color="#0057b8", lw=1.7, label="Predicted good-state probability")
    if pd.to_numeric(d["p_seed_std"], errors="coerce").max() > 0:
        lo = np.clip(d["p_mean"] - d["p_seed_std"], 0, 1)
        hi = np.clip(d["p_mean"] + d["p_seed_std"], 0, 1)
        ax.fill_between(d["date"], lo, hi, color="#0057b8", alpha=0.18, label="+/-1 seed std")
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.65)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probability")
    ax.set_title(f"PREDICTED PROBABILITY VS REAL REGIME | {feature_design} | {family} | {_target_short(target)}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    text = (
        "PREDICTION METRICS\n\n"
        f"N obs                  {m.get('n_obs', np.nan):.0f}\n"
        f"Accuracy               {m.get('accuracy', np.nan):.3f}\n"
        f"Balanced accuracy      {m.get('balanced_accuracy', np.nan):.3f}\n"
        f"Precision              {m.get('precision', np.nan):.3f}\n"
        f"Recall                 {m.get('recall', np.nan):.3f}\n"
        f"F1                     {m.get('f1', np.nan):.3f}\n"
        f"ROC AUC                {m.get('roc_auc', np.nan):.3f}\n"
        f"Brier                  {m.get('brier', np.nan):.3f}\n"
        f"Log loss               {m.get('log_loss', np.nan):.3f}\n\n"
        "PROBABILITY DIAGNOSTICS\n"
        f"Actual good share      {m.get('actual_good_share', np.nan):.3f}\n"
        f"Predicted good share   {m.get('pred_good_share', np.nan):.3f}\n"
        f"Avg seed prob std      {m.get('avg_seed_prob_std', np.nan):.4f}\n"
        f"Max seed prob std      {m.get('max_seed_prob_std', np.nan):.4f}\n\n"
        "BACKGROUND\n"
        "green = realized good regime\n"
        "red   = realized bad regime"
    )
    ax_box.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.95),
    )
    fig.tight_layout()
    plt.show()


# ==================================================================================================
# 4. ALLOCATION - L1 TRADE-PENALIZED GRID USING THE STUDENT PREDICTION ALLOCATION ENGINE
# ==================================================================================================

# Clean objective. There is NO tau and NO lambda + tau*c double counting.
#
#   max_w  mu_t'w - 0.5 * gamma * w' Sigma_t w - lambda_trade * ||w_t - w_{t-1}||_1
#
# Realized net return is evaluated separately:
#
#   r_net,t = w_eq,t r_eq,t+1 + w_bd,t r_bd,t+1 - tcost_one_way * ||w_t - w_{t-1}||_1
#
# Interpretation:
#   gamma          = risk aversion
#   lambda_trade   = ex-ante turnover aversion inside the optimizer
#   tcost_one_way  = realized transaction cost used only for net performance
#   phl            = probability smoothing halflife

# --------------------------------------------------------------------------------------------------
# 4.0 Allocation settings
# --------------------------------------------------------------------------------------------------

L1_ALLOCATION_PAIR_ID = "return_sortino_vs_bond"
L1_FEATURE_DESIGNS = ["z_smooth_only", "z_plus_zsmooth"]
L1_FAMILIES = ["LOGIT", "RF"]
L1_BUDGET_MODE = "cash_allowed"

L1_GAMMA_GRID = [0.25, 0.50, 1.00, 2.00, 3.00, 5.00, 8.00, 12.00]
L1_TCOST_GRID = [0.0005, 0.0010, 0.0025, 0.0050]
L1_PROB_SMOOTH_HALFLIFE_GRID = [0.0, 3.0, 6.0]
L1_LAMBDA_TRADE_GRID = [
    0.0,
    0.00025,
    0.00050,
    0.00100,
    0.00150,
    0.00200,
    0.00250,
    0.00300,
    0.00400,
    0.00500,
    0.00750,
    0.01000,
    0.01500,
    0.02000,
    0.03000,
    0.05000,
]

L1_MIN_AVG_EQ_WEIGHT = 0.25
L1_MAX_AVG_EQ_WEIGHT = 0.90
L1_MAX_AVG_BD_WEIGHT = 0.80
L1_MAX_AVG_CASH_WEIGHT = 0.45
L1_MIN_ANN_VOL = 0.055
L1_MIN_ANN_RETURN = 0.040
L1_MIN_BOTH_BAD_CASH = 0.30
L1_MIN_EQUITY_GOOD_EQ = 0.55
L1_MIN_BOND_GOOD_BD = 0.50

L1_TURNOVER_CAPS_TO_REPORT = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
L1_MAX_PLOTS = 14
L1_PRINT_TOP_N = 25

L1_KNOWN_GOOD_POINTS = pd.DataFrame([
    {"feature_design": "z_smooth_only",  "family": "LOGIT", "gamma": 0.25, "tcost_one_way": 0.0005, "prob_smooth_halflife": 0.0, "lambda_trade": 0.0000},
    {"feature_design": "z_smooth_only",  "family": "LOGIT", "gamma": 0.50, "tcost_one_way": 0.0005, "prob_smooth_halflife": 0.0, "lambda_trade": 0.0000},
    {"feature_design": "z_smooth_only",  "family": "LOGIT", "gamma": 1.00, "tcost_one_way": 0.0005, "prob_smooth_halflife": 0.0, "lambda_trade": 0.0000},
    {"feature_design": "z_plus_zsmooth", "family": "LOGIT", "gamma": 3.00, "tcost_one_way": 0.0005, "prob_smooth_halflife": 0.0, "lambda_trade": 0.0010},
    {"feature_design": "z_smooth_only",  "family": "RF",    "gamma": 2.00, "tcost_one_way": 0.0005, "prob_smooth_halflife": 0.0, "lambda_trade": 0.0010},
    {"feature_design": "z_plus_zsmooth", "family": "RF",    "gamma": 2.00, "tcost_one_way": 0.0005, "prob_smooth_halflife": 0.0, "lambda_trade": 0.0005},
])

# --------------------------------------------------------------------------------------------------
# 4.1 Table printer and small helpers
# --------------------------------------------------------------------------------------------------

def to_print_table(title, df, cols=None, sort_by=None, ascending=False, n=20, float_digits=4):
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)
    if df is None or len(df) == 0:
        print("EMPTY")
        return
    x = df.copy()
    if sort_by is not None and sort_by in x.columns:
        x = x.sort_values(sort_by, ascending=ascending)
    if cols is not None:
        cols = [c for c in cols if c in x.columns]
        x = x[cols]
    if n is not None:
        x = x.head(n)
    with pd.option_context(
        "display.max_rows", n if n is not None else 500,
        "display.max_columns", 140,
        "display.width", 280,
        "display.float_format", lambda v: f"{v:.{float_digits}f}",
    ):
        print(x.to_string(index=False))


def _safe_colname_l1(x):
    if "safe_colname" in globals():
        return safe_colname(x)
    return str(x).replace("-", "m").replace(".", "p")


def _require_global_l1(name):
    if name not in globals():
        raise RuntimeError(f"Required object `{name}` is missing. Run student prediction stage first.")
    return globals()[name]


def _pick_prediction_long_for_l1():
    for nm in [
        "student_oos_prediction_panel_long",
        "student_prediction_panel_long",
        "student_rf_prediction_panel_long",
        "student_prediction_panel_long",
    ]:
        obj = globals().get(nm, None)
        if isinstance(obj, pd.DataFrame) and len(obj):
            out = obj.copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            if "year" not in out.columns:
                out["year"] = out["date"].dt.year
            print(f"Using prediction panel for allocation: {nm}")
            return out
    raise RuntimeError("Could not find the Cell-3 long prediction panel. Expected `student_oos_prediction_panel_long`.")


def _ensure_date_column_l1(df, object_name):
    """Return a copy with a real `date` column. No assumptions about index/name."""
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out

    for c in ["month", "month_end", "prediction_date", "eom", "time"]:
        if c in out.columns:
            out = out.rename(columns={c: "date"})
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            return out

    # Some notebook objects are indexed by date.
    idx_name = out.index.name
    if idx_name is not None and str(idx_name).lower() in ["date", "month", "month_end", "prediction_date"]:
        out = out.reset_index().rename(columns={idx_name: "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out

    # Last attempt: if the index itself looks like dates.
    parsed_idx = pd.to_datetime(out.index, errors="coerce")
    if pd.Series(parsed_idx).notna().mean() > 0.80:
        out = out.copy()
        out.insert(0, "date", parsed_idx)
        return out

    raise RuntimeError(f"Object `{object_name}` has no date column and no date-like index. Columns: {list(out.columns)[:30]}")


def _pick_hist_panel_for_l1():
    """
    Return:
      hist_panel: dated DataFrame with historical labels/returns
      historical_targets: dict in the exact format expected by Cell-3 functions:
          {pair_id: {"eq_y": <equity label column>, "bd_y": <bond label column>}}

    Important: Cell-3 precompute_allocation_moments expects `historical_targets` to be
    a dictionary, not a DataFrame. This avoids KeyError(pair_id).
    """
    hp_name = None
    hp = None
    for nm in [
        "historical_label_return_panel_h1",
        "final_h1_historical_label_return_panel_for_allocation",
        "final_h1_target_wide_for_student",
        "majority_vote_prediction_targets_h1",
    ]:
        obj = globals().get(nm, None)
        if isinstance(obj, pd.DataFrame) and len(obj):
            hp_name = nm
            hp = obj
            break

    if hp is None:
        raise RuntimeError(
            "Missing historical H=1 label/return panel. Expected one of: "
            "historical_label_return_panel_h1, final_h1_historical_label_return_panel_for_allocation, "
            "final_h1_target_wide_for_student, majority_vote_prediction_targets_h1."
        )

    out_hp = _ensure_date_column_l1(hp, hp_name)

    pair = _find_allocation_pair_l1(L1_ALLOCATION_PAIR_ID)
    eq_col, bd_col = _get_l1_pair_target_cols(pair)

    # Robust aliases in case the pair dict uses a slightly different naming convention.
    eq_aliases = [
        eq_col,
        "eq_return_sortino_state_h1",
        "equity_return_sortino_state_h1",
        "equity_return_sortino_all_models_state_h1",
        "equity_return_sortino_all_models_state",
        "eq_selected_state_h1",
    ]
    bd_aliases = [
        bd_col,
        "bd_bond_canonical_state_h1",
        "bond_bond_canonical_state_h1",
        "bond_bond_canonical_all_models_state_h1",
        "bond_bond_canonical_all_models_state",
        "bd_selected_state_h1",
    ]

    eq_found = next((c for c in eq_aliases if c in out_hp.columns), None)
    bd_found = next((c for c in bd_aliases if c in out_hp.columns), None)

    if eq_found is None or bd_found is None:
        raise RuntimeError(
            "Could not find required historical target label columns for allocation.\n"
            f"Pair id: {L1_ALLOCATION_PAIR_ID}\n"
            f"Expected equity aliases: {eq_aliases}\n"
            f"Expected bond aliases: {bd_aliases}\n"
            f"Available columns sample: {list(out_hp.columns)[:80]}"
        )

    # Standardize names when aliases were used.
    if eq_found != eq_col:
        out_hp[eq_col] = out_hp[eq_found]
    if bd_found != bd_col:
        out_hp[bd_col] = out_hp[bd_found]

    # Ensure return columns exist. Prefer already-standard columns.
    if "equity_ret_h1" not in out_hp.columns:
        for c in ["equity_ret", "eq_ret_h1", "eq_ret", "equity_return_h1", "equity_return"]:
            if c in out_hp.columns:
                out_hp["equity_ret_h1"] = out_hp[c]
                break
    if "bond_ret_h1" not in out_hp.columns:
        for c in ["bond_ret", "bd_ret_h1", "bd_ret", "bond_return_h1", "bond_return"]:
            if c in out_hp.columns:
                out_hp["bond_ret_h1"] = out_hp[c]
                break

    missing_returns = [c for c in ["equity_ret_h1", "bond_ret_h1"] if c not in out_hp.columns]
    if missing_returns:
        raise RuntimeError(
            f"Historical panel is missing required return columns: {missing_returns}. "
            f"Available columns sample: {list(out_hp.columns)[:80]}"
        )

    out_hp["date"] = pd.to_datetime(out_hp["date"], errors="coerce")
    out_hp = out_hp.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    historical_targets_dict = {
        L1_ALLOCATION_PAIR_ID: {
            "eq_y": eq_col,
            "bd_y": bd_col,
        }
    }

    return out_hp, historical_targets_dict

def _find_allocation_pair_l1(pair_id):
    pairs = _require_global_l1("ALLOCATION_PAIRS")
    for p in pairs:
        if p.get("pair_id") == pair_id:
            return p
    raise RuntimeError(f"Could not find allocation pair `{pair_id}` in ALLOCATION_PAIRS.")


def _get_l1_pair_target_cols(pair):
    eq_col = pair.get("equity_target") or pair.get("eq_target")
    bd_col = pair.get("bond_target") or pair.get("bd_target")
    if eq_col is None:
        if pair.get("pair_id") == "return_sortino_vs_bond":
            eq_col = "eq_return_sortino_state_h1"
        elif pair.get("pair_id") == "vol_downside_vs_bond":
            eq_col = "eq_vol_downside_state_h1"
    if bd_col is None:
        bd_col = "bd_bond_canonical_state_h1"
    return eq_col, bd_col


def _get_l1_weight_grid():
    if "get_weight_grid" in globals():
        return get_weight_grid(L1_BUDGET_MODE)
    # fallback cash-allowed grid
    vals = np.round(np.arange(0.0, 1.0 + 1e-12, WEIGHT_GRID_STEP), 10)
    rows = []
    for we in vals:
        for wb in vals:
            if we + wb <= 1.0 + 1e-12:
                rows.append((we, wb))
    return np.asarray(rows, dtype=float)


def _l1_strategy_summary_from_path(g):
    ret = pd.Series(g["strategy_net_ret"]).dropna().astype(float)
    wealth = MV_TC_STARTING_WEALTH * (1.0 + ret).cumprod() if "MV_TC_STARTING_WEALTH" in globals() else 100.0 * (1.0 + ret).cumprod()
    ann_return = float((wealth.iloc[-1] / wealth.iloc[0]) ** (12.0 / max(len(ret), 1)) - 1.0) if len(ret) else np.nan
    ann_vol = float(ret.std(ddof=1) * np.sqrt(12.0)) if len(ret) > 1 else np.nan
    sharpe = float(ann_return / ann_vol) if np.isfinite(ann_vol) and ann_vol > 1e-12 else np.nan
    dd = wealth / wealth.cummax() - 1.0 if len(wealth) else pd.Series(dtype=float)
    q = ret.quantile(0.01) if len(ret) else np.nan
    tail = ret[ret <= q] if len(ret) else pd.Series(dtype=float)

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "cvar_1pct_monthly": float(tail.mean()) if len(tail) else np.nan,
        "final_wealth": float(wealth.iloc[-1]) if len(wealth) else np.nan,
        "ann_turnover": float(12.0 * g["turnover"].mean()),
        "avg_monthly_turnover": float(g["turnover"].mean()),
        "max_monthly_turnover": float(g["turnover"].max()),
        "ann_tcost_drag": float(12.0 * g["tcost"].mean()),
        "avg_tcost_paid_monthly": float(g["tcost"].mean()),
        "avg_w_eq": float(g["w_eq"].mean()),
        "avg_w_bd": float(g["w_bd"].mean()),
        "avg_w_cash": float(g["w_cash"].mean()),
    }


def _make_full_l1_param_grid():
    rows = []
    for gamma in L1_GAMMA_GRID:
        for c in L1_TCOST_GRID:
            for phl in L1_PROB_SMOOTH_HALFLIFE_GRID:
                for lam in L1_LAMBDA_TRADE_GRID:
                    rows.append({
                        "budget_mode": L1_BUDGET_MODE,
                        "gamma": float(gamma),
                        "tcost_one_way": float(c),
                        "prob_smooth_halflife": float(phl),
                        "lambda_trade": float(lam),
                    })
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _simulate_l1_param_block(monthly_inputs, scenario_id, family_name, allocation_pair, params):
    pair_id = allocation_pair["pair_id"]
    m = monthly_inputs.copy().sort_values("date").reset_index(drop=True)
    params = params.copy().reset_index(drop=True)
    if m.empty or params.empty:
        return pd.DataFrame()

    W = _get_l1_weight_grid()
    W_eq = W[:, 0]
    W_bd = W[:, 1]

    gamma = params["gamma"].to_numpy(dtype=float)
    tcost_one_way = params["tcost_one_way"].to_numpy(dtype=float)
    lambda_trade = params["lambda_trade"].to_numpy(dtype=float)

    n_param = len(params)
    prev_w_eq = np.zeros(n_param, dtype=float)
    prev_w_bd = np.zeros(n_param, dtype=float)
    out_parts = []

    for _, row in m.iterrows():
        mu_eq = float(row["mu_eq"])
        mu_bd = float(row["mu_bd"])
        cov00 = float(row["cov00"])
        cov01 = float(row["cov01"])
        cov11 = float(row["cov11"])

        ret_term = W_eq * mu_eq + W_bd * mu_bd
        risk_term = cov00 * W_eq ** 2 + 2.0 * cov01 * W_eq * W_bd + cov11 * W_bd ** 2
        turnover_grid = np.abs(W_eq[None, :] - prev_w_eq[:, None]) + np.abs(W_bd[None, :] - prev_w_bd[:, None])

        obj = ret_term[None, :] - 0.5 * gamma[:, None] * risk_term[None, :] - lambda_trade[:, None] * turnover_grid
        idx = np.nanargmax(obj, axis=1)

        w_eq = W_eq[idx]
        w_bd = W_bd[idx]
        w_cash = 1.0 - w_eq - w_bd
        selected_turnover = np.abs(w_eq - prev_w_eq) + np.abs(w_bd - prev_w_bd)
        selected_tcost = tcost_one_way * selected_turnover
        gross_ret = w_eq * float(row["equity_ret_h1"]) + w_bd * float(row["bond_ret_h1"])
        net_ret = gross_ret - selected_tcost
        selected_obj = obj[np.arange(n_param), idx]

        part = params.copy()
        part["scenario_id"] = scenario_id
        part["feature_design"] = scenario_id
        part["date"] = row["date"]
        part["family"] = family_name
        part["allocation_pair"] = pair_id
        part["allocation_pair_name"] = allocation_pair.get("pretty_name", pair_id)
        part["p_eq"] = float(row["p_eq"])
        part["p_bd"] = float(row["p_bd"])
        part["mu_eq"] = mu_eq
        part["mu_bd"] = mu_bd
        part["sd_eq"] = float(row.get("sd_eq", np.nan))
        part["sd_bd"] = float(row.get("sd_bd", np.nan))
        part["rho"] = float(row.get("rho", np.nan))
        part["w_eq"] = w_eq
        part["w_bd"] = w_bd
        part["w_cash"] = w_cash
        part["turnover"] = selected_turnover
        part["tcost"] = selected_tcost
        part["tcost_paid"] = selected_tcost
        part["equity_ret_h1"] = float(row["equity_ret_h1"])
        part["bond_ret_h1"] = float(row["bond_ret_h1"])
        part["strategy_gross_ret"] = gross_ret
        part["strategy_net_ret"] = net_ret
        part["objective"] = selected_obj

        for c in [
            "eq_n_state0", "eq_n_state1", "bd_n_state0", "bd_n_state1",
            "eq_weight_state0", "eq_weight_state1", "bd_weight_state0", "bd_weight_state1",
            "rho_source", "rho_n_obs", "rho_weight", "rho_empirical",
        ]:
            if c in row.index:
                part[c] = row[c]

        part["strategy_name"] = (
            scenario_id
            + "__" + family_name + "_" + pair_id
            + "__" + part["budget_mode"].astype(str)
            + "__g" + part["gamma"].map(_safe_colname_l1)
            + "__lam" + part["lambda_trade"].map(_safe_colname_l1)
            + "__c" + part["tcost_one_way"].map(_safe_colname_l1)
            + "__phl" + part["prob_smooth_halflife"].map(_safe_colname_l1)
        )

        out_parts.append(part)
        prev_w_eq = w_eq
        prev_w_bd = w_bd

    return pd.concat(out_parts, axis=0, ignore_index=True, sort=False) if out_parts else pd.DataFrame()


def _attach_realized_state_labels_l1(long, allocation_pair, hist_panel):
    """
    Attach realized teacher state labels to the L1 allocation path.

    `precompute_allocation_moments` needs historical_targets as a dict, but this
    function needs the dated historical panel. Therefore pass `hist_panel` here,
    not the historical_targets dict.
    """
    out = long.copy()
    eq_col, bd_col = _get_l1_pair_target_cols(allocation_pair)

    if hist_panel is None or not isinstance(hist_panel, pd.DataFrame) or "date" not in hist_panel.columns:
        out["eq_realized_good"] = np.nan
        out["bd_realized_good"] = np.nan
        return out

    h = hist_panel.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"]).copy()

    # robust aliases in case the panel has canonical or older names
    eq_aliases = [
        eq_col,
        "eq_return_sortino_state_h1",
        "equity_return_sortino_state_h1",
        "equity_return_sortino_all_models_state_h1",
        "equity_return_sortino_all_models_state",
        "eq_selected_state_h1",
    ]
    bd_aliases = [
        bd_col,
        "bd_bond_canonical_state_h1",
        "bond_bond_canonical_state_h1",
        "bond_bond_canonical_all_models_state_h1",
        "bond_bond_canonical_all_models_state",
        "bd_selected_state_h1",
    ]

    eq_found = next((c for c in eq_aliases if c in h.columns), None)
    bd_found = next((c for c in bd_aliases if c in h.columns), None)

    keep = ["date"]
    rename = {}

    if eq_found is not None:
        keep.append(eq_found)
        rename[eq_found] = "eq_realized_good"
    else:
        out["eq_realized_good"] = np.nan

    if bd_found is not None:
        keep.append(bd_found)
        rename[bd_found] = "bd_realized_good"
    else:
        out["bd_realized_good"] = np.nan

    if len(keep) > 1:
        h = h[keep].rename(columns=rename).drop_duplicates("date")
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.merge(h, on="date", how="left")

    if "eq_realized_good" not in out.columns:
        out["eq_realized_good"] = np.nan
    if "bd_realized_good" not in out.columns:
        out["bd_realized_good"] = np.nan

    out["eq_realized_good"] = pd.to_numeric(out["eq_realized_good"], errors="coerce")
    out["bd_realized_good"] = pd.to_numeric(out["bd_realized_good"], errors="coerce")
    return out

def _summarize_l1_results(allocation_results):
    if allocation_results is None or allocation_results.empty:
        return pd.DataFrame()

    rows = []
    for strategy_name, g in allocation_results.groupby("strategy_name", sort=False):
        g = g.copy().sort_values("date").reset_index(drop=True)
        d = _l1_strategy_summary_from_path(g)
        d["strategy"] = strategy_name
        d["strategy_name"] = strategy_name
        d["scenario_id"] = str(g["scenario_id"].iloc[0])
        d["feature_design"] = str(g["feature_design"].iloc[0])
        d["family"] = str(g["family"].iloc[0])
        d["allocation_pair"] = str(g["allocation_pair"].iloc[0])
        d["budget_mode"] = str(g["budget_mode"].iloc[0])
        d["gamma"] = float(g["gamma"].iloc[0])
        d["tcost_one_way"] = float(g["tcost_one_way"].iloc[0])
        d["prob_smooth_halflife"] = float(g["prob_smooth_halflife"].iloc[0])
        d["lambda_trade"] = float(g["lambda_trade"].iloc[0])

        if g["eq_realized_good"].notna().any() and g["bd_realized_good"].notna().any():
            eq_good = g["eq_realized_good"].eq(1)
            bd_good = g["bd_realized_good"].eq(1)
        else:
            eq_good = g["p_eq"].ge(0.5)
            bd_good = g["p_bd"].ge(0.5)

        both_good = eq_good & bd_good
        eq_good_only = eq_good & ~bd_good
        bd_good_only = ~eq_good & bd_good
        both_bad = ~eq_good & ~bd_good

        def _state_mean(mask, col):
            return float(g.loc[mask, col].mean()) if int(mask.sum()) > 0 else np.nan

        d["both_good_n"] = int(both_good.sum())
        d["equity_good_only_n"] = int(eq_good_only.sum())
        d["bond_good_only_n"] = int(bd_good_only.sum())
        d["both_bad_n"] = int(both_bad.sum())
        d["both_good_avg_eq"] = _state_mean(both_good, "w_eq")
        d["both_good_avg_bd"] = _state_mean(both_good, "w_bd")
        d["both_good_avg_cash"] = _state_mean(both_good, "w_cash")
        d["equity_good_only_avg_eq"] = _state_mean(eq_good_only, "w_eq")
        d["equity_good_only_avg_bd"] = _state_mean(eq_good_only, "w_bd")
        d["equity_good_only_avg_cash"] = _state_mean(eq_good_only, "w_cash")
        d["bond_good_only_avg_eq"] = _state_mean(bd_good_only, "w_eq")
        d["bond_good_only_avg_bd"] = _state_mean(bd_good_only, "w_bd")
        d["bond_good_only_avg_cash"] = _state_mean(bd_good_only, "w_cash")
        d["both_bad_avg_eq"] = _state_mean(both_bad, "w_eq")
        d["both_bad_avg_bd"] = _state_mean(both_bad, "w_bd")
        d["both_bad_avg_cash"] = _state_mean(both_bad, "w_cash")
        rows.append(d)

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["sharpe", "ann_return", "ann_turnover"], ascending=[False, False, True]).reset_index(drop=True)


def _build_l1_6040_benchmark(common_dates, returns_local):
    if "build_true_6040_benchmark" in globals():
        try:
            bench = build_true_6040_benchmark(common_dates).copy()
            bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
            return bench
        except Exception:
            pass
    r = returns_local.copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r = r.loc[r["date"].isin(pd.to_datetime(common_dates))].sort_values("date")
    eq_ret = pd.to_numeric(r["equity_ret_h1"], errors="coerce")
    bd_ret = pd.to_numeric(r["bond_ret_h1"], errors="coerce")
    r["ret_60_40"] = 0.60 * eq_ret + 0.40 * bd_ret
    r["turnover_60_40"] = 0.0
    for i in range(1, len(r)):
        growth = 0.60 * (1.0 + float(eq_ret.iloc[i - 1])) + 0.40 * (1.0 + float(bd_ret.iloc[i - 1]))
        if np.isfinite(growth) and abs(growth) > 1e-12:
            drift_eq = 0.60 * (1.0 + float(eq_ret.iloc[i - 1])) / growth
            drift_bd = 0.40 * (1.0 + float(bd_ret.iloc[i - 1])) / growth
            r.iloc[i, r.columns.get_loc("turnover_60_40")] = abs(0.60 - drift_eq) + abs(0.40 - drift_bd)
    start = MV_TC_STARTING_WEALTH if "MV_TC_STARTING_WEALTH" in globals() else 100.0
    r["wealth_60_40"] = start * (1.0 + r["ret_60_40"].fillna(0.0)).cumprod()
    return r[["date", "ret_60_40", "wealth_60_40", "turnover_60_40"]].copy()


def _run_l1_allocation_grid(prediction_long_source):
    returns_local = _require_global_l1("returns").copy()
    returns_local["date"] = pd.to_datetime(returns_local["date"], errors="coerce")
    allocation_pair_l1 = _find_allocation_pair_l1(L1_ALLOCATION_PAIR_ID)
    hist_panel, hist_targets = _pick_hist_panel_for_l1()

    params_full = _make_full_l1_param_grid()
    n_scenarios = len(L1_FEATURE_DESIGNS) * len(L1_FAMILIES) * len(params_full)
    print("\n" + "=" * 150)
    print("RUNNING L1 TRADE-PENALIZED ALLOCATION GRID")
    print("=" * 150)
    print(f"Feature designs: {L1_FEATURE_DESIGNS}")
    print(f"Model families:  {L1_FAMILIES}")
    print(f"Allocation pair: {L1_ALLOCATION_PAIR_ID}")
    print(f"Budget mode:     {L1_BUDGET_MODE}")
    print(f"Scenarios:       {n_scenarios}")
    print("Objective:        mu'w - 0.5*gamma*w'Sigma*w - lambda_trade*||w_t - w_(t-1)||_1")
    print("Realized return:  gross portfolio return - tcost_one_way*turnover")

    result_parts = []
    panel_map = {}

    for feature_design in L1_FEATURE_DESIGNS:
        for family in L1_FAMILIES:
            print(f"\nL1 BLOCK | feature_design={feature_design} | family={family}", flush=True)
            pred_block = prediction_long_source.copy()
            if "feature_design" in pred_block.columns:
                pred_block = pred_block.loc[pred_block["feature_design"].eq(feature_design)].copy()
            elif "scenario_id" in pred_block.columns:
                pred_block = pred_block.loc[pred_block["scenario_id"].eq(feature_design)].copy()
            if "family" in pred_block.columns:
                pred_block = pred_block.loc[pred_block["family"].astype(str).str.upper().eq(str(family).upper())].copy()
            if pred_block.empty:
                print(f"  SKIP: empty prediction block for {feature_design} | {family}", flush=True)
                continue

            pair_wide = build_allocation_pair_wide_panel(
                prediction_long=pred_block,
                pair=allocation_pair_l1,
                scenario_id=feature_design,
                feature_design=feature_design,
                family=family,
            )
            if pair_wide.empty:
                print(f"  SKIP: empty pair-wide prediction panel for {feature_design} | {family}", flush=True)
                continue

            panel = pair_wide.copy()
            panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
            panel = panel.merge(returns_local[["date", "equity_ret_h1", "bond_ret_h1"]].copy(), on="date", how="left")
            first_prediction_date = panel["date"].dropna().min()
            initial_moments = initial_moments_from_pre_oos(hist_panel, first_prediction_date)

            moment_lookup = precompute_allocation_moments(
                panel=panel,
                allocation_pair=allocation_pair_l1,
                historical_label_return_panel=hist_panel,
                historical_targets=hist_targets,
                initial_moments=initial_moments,
            )
            panel_map[(feature_design, family, L1_ALLOCATION_PAIR_ID)] = panel.copy()

            for phl in L1_PROB_SMOOTH_HALFLIFE_GRID:
                monthly_inputs = build_monthly_allocation_inputs(
                    panel=panel,
                    allocation_pair=allocation_pair_l1,
                    moment_lookup=moment_lookup,
                    prob_smooth_halflife=float(phl),
                )
                params_phl = params_full.loc[np.isclose(params_full["prob_smooth_halflife"], float(phl))].copy()
                if params_phl.empty:
                    continue
                res = _simulate_l1_param_block(
                    monthly_inputs=monthly_inputs,
                    scenario_id=feature_design,
                    family_name=family,
                    allocation_pair=allocation_pair_l1,
                    params=params_phl,
                )
                if len(res):
                    result_parts.append(res)

    long = pd.concat(result_parts, axis=0, ignore_index=True, sort=False) if result_parts else pd.DataFrame()
    if long.empty:
        raise RuntimeError("No L1 allocation results were produced.")

    long = _attach_realized_state_labels_l1(long, allocation_pair_l1, hist_panel)
    common_dates = sorted(long["date"].dropna().unique().tolist())
    bench = _build_l1_6040_benchmark(common_dates, returns_local)
    long = long[long["date"].isin(bench["date"])].copy()
    start = MV_TC_STARTING_WEALTH if "MV_TC_STARTING_WEALTH" in globals() else 100.0
    long["wealth"] = (
        long.sort_values(["strategy_name", "date"])
            .groupby("strategy_name")["strategy_net_ret"]
            .transform(lambda r: start * (1.0 + r).cumprod())
    )
    summary = _summarize_l1_results(long)
    return long, summary, bench, panel_map


def _economic_l1_universe(summary):
    x = summary.copy().replace([np.inf, -np.inf], np.nan)
    needed = [
        "sharpe", "ann_return", "ann_vol", "ann_turnover", "avg_w_eq", "avg_w_bd", "avg_w_cash",
        "both_bad_avg_cash", "equity_good_only_avg_eq", "bond_good_only_avg_bd",
    ]
    for c in needed:
        if c not in x.columns:
            x[c] = np.nan
    x = x.loc[
        x["budget_mode"].eq(L1_BUDGET_MODE)
        & x["allocation_pair"].eq(L1_ALLOCATION_PAIR_ID)
        & x["sharpe"].notna()
        & x["ann_return"].ge(L1_MIN_ANN_RETURN)
        & x["ann_vol"].ge(L1_MIN_ANN_VOL)
        & x["avg_w_eq"].between(L1_MIN_AVG_EQ_WEIGHT, L1_MAX_AVG_EQ_WEIGHT)
        & x["avg_w_bd"].le(L1_MAX_AVG_BD_WEIGHT)
        & x["avg_w_cash"].le(L1_MAX_AVG_CASH_WEIGHT)
        & x["both_bad_avg_cash"].ge(L1_MIN_BOTH_BAD_CASH)
        & x["equity_good_only_avg_eq"].ge(L1_MIN_EQUITY_GOOD_EQ)
        & x["bond_good_only_avg_bd"].ge(L1_MIN_BOND_GOOD_BD)
    ].copy()
    return x.sort_values(["sharpe", "ann_return", "ann_turnover"], ascending=[False, False, True]).reset_index(drop=True)


def _sensitivity_table_l1(df, group_cols):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(group_cols))
    x = df.copy().replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["sharpe"])
    if len(x) == 0:
        return pd.DataFrame(columns=list(group_cols))
    group_cols = list(group_cols)
    rows = []
    for key, g in x.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        best = g.sort_values(["sharpe", "ann_return", "ann_turnover"], ascending=[False, False, True]).iloc[0]
        row = {c: v for c, v in zip(group_cols, key)}
        row.update({
            "n_strategies": int(len(g)),
            "best_sharpe": float(g["sharpe"].max()),
            "median_sharpe": float(g["sharpe"].median()),
            "best_ann_return": float(g["ann_return"].max()),
            "median_ann_return": float(g["ann_return"].median()),
            "median_ann_turnover": float(g["ann_turnover"].median()),
            "min_ann_turnover": float(g["ann_turnover"].min()),
            "best_ann_turnover": float(best["ann_turnover"]),
            "median_avg_cash": float(g["avg_w_cash"].median()),
            "best_strategy": best.get("strategy", best.get("strategy_name", "")),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def _turnover_frontier_l1(df):
    rows = []
    for cap in L1_TURNOVER_CAPS_TO_REPORT:
        z = df.loc[df["ann_turnover"].le(cap)].copy()
        if len(z):
            r = z.sort_values(["sharpe", "ann_return", "ann_turnover"], ascending=[False, False, True]).iloc[0].to_dict()
            r["turnover_cap"] = cap
            rows.append(r)
    out = pd.DataFrame(rows)
    return out.drop_duplicates("strategy", keep="first").reset_index(drop=True) if len(out) else out


def _exact_strategy_match_l1(summary, spec):
    x = summary.copy()
    x = x.loc[
        x["feature_design"].astype(str).eq(str(spec["feature_design"]))
        & x["family"].astype(str).str.upper().eq(str(spec["family"]).upper())
        & x["allocation_pair"].eq(L1_ALLOCATION_PAIR_ID)
        & x["budget_mode"].eq(L1_BUDGET_MODE)
    ].copy()
    for c in ["gamma", "tcost_one_way", "prob_smooth_halflife", "lambda_trade"]:
        x = x.loc[np.isclose(pd.to_numeric(x[c], errors="coerce"), float(spec[c]), atol=1e-12)].copy()
    if len(x) == 0:
        return None
    return x.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0]


def _select_l1_plot_strategies(summary, economic):
    plot_rows = []
    seen = set()

    def add(reason, row):
        if row is None:
            return
        r = dict(row)
        strat = r.get("strategy", r.get("strategy_name", None))
        if strat is None or strat in seen:
            return
        r["selection_reason"] = reason
        plot_rows.append(r)
        seen.add(strat)

    econ = economic.copy()
    if len(econ) == 0:
        econ = summary.loc[summary["sharpe"].notna()].copy()

    # 1) Previous economically good reference points.
    for _, spec in L1_KNOWN_GOOD_POINTS.iterrows():
        row = _exact_strategy_match_l1(summary, spec)
        add(
            f"reference: {spec['feature_design']} {spec['family']} gamma={spec['gamma']:g}, lambda={spec['lambda_trade']:g}, cost={spec['tcost_one_way']:g}, phl={spec['prob_smooth_halflife']:g}",
            row,
        )

    # 2) Main decision rows from the economic universe.
    if len(econ):
        add("best Sharpe in economic universe", econ.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0])
        for cap in [0.75, 1.00, 1.25, 1.50, 2.00]:
            z = econ.loc[econ["ann_turnover"].le(cap)].copy()
            if len(z):
                add(f"best Sharpe with annual turnover <= {cap:g}", z.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0])
        z = econ.loc[econ["ann_turnover"].le(1.50)].copy()
        if len(z):
            add("highest return with annual turnover <= 1.5", z.sort_values(["ann_return", "sharpe"], ascending=[False, False]).iloc[0])
        best_sh = float(econ["sharpe"].max())
        z = econ.loc[econ["sharpe"].ge(best_sh - 0.05)].copy()
        if len(z):
            add("lowest turnover within 0.05 Sharpe of best", z.sort_values(["ann_turnover", "sharpe"], ascending=[True, False]).iloc[0])

    # 3) Parameter sweeps around the best <= 1 turnover strategy, otherwise best economic strategy.
    if len(econ):
        base_pool = econ.loc[econ["ann_turnover"].le(1.0)].copy()
        if len(base_pool):
            base = base_pool.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0]
        else:
            base = econ.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0]

        same_model = summary.loc[
            summary["feature_design"].astype(str).eq(str(base["feature_design"]))
            & summary["family"].astype(str).str.upper().eq(str(base["family"]).upper())
            & summary["allocation_pair"].eq(str(base["allocation_pair"]))
            & summary["budget_mode"].eq(str(base["budget_mode"]))
        ].copy()

        def sweep(param, values):
            sub = same_model.copy()
            for fixed in ["gamma", "tcost_one_way", "prob_smooth_halflife", "lambda_trade"]:
                if fixed == param:
                    continue
                sub = sub.loc[np.isclose(pd.to_numeric(sub[fixed], errors="coerce"), float(base[fixed]), atol=1e-12)].copy()
            for val in values:
                cand = sub.loc[np.isclose(pd.to_numeric(sub[param], errors="coerce"), float(val), atol=1e-12)].copy()
                if len(cand):
                    add(f"sweep {param} = {val:g}", cand.sort_values(["sharpe", "ann_return"], ascending=[False, False]).iloc[0])

        sweep("gamma", [min(L1_GAMMA_GRID), 1.0, 3.0, 5.0, max(L1_GAMMA_GRID)])
        sweep("tcost_one_way", [min(L1_TCOST_GRID), 0.0010, 0.0025, max(L1_TCOST_GRID)])
        sweep("lambda_trade", [0.0, 0.0005, 0.0010, 0.0025, 0.0050, 0.0100, 0.0200, 0.0500])
        sweep("prob_smooth_halflife", [0.0, 3.0, 6.0])

    out = pd.DataFrame(plot_rows)
    if len(out) == 0:
        return out
    return out.head(L1_MAX_PLOTS).reset_index(drop=True)

# --------------------------------------------------------------------------------------------------
# 4.2 L1 allocation plot in the same dashboard format
# --------------------------------------------------------------------------------------------------

def _l1_drawdown_from_wealth(w):
    w = pd.Series(w).astype(float)
    return w / w.cummax() - 1.0


def _state_from_realized_or_prob_l1(g):
    if "eq_realized_good" in g.columns and "bd_realized_good" in g.columns and g["eq_realized_good"].notna().any() and g["bd_realized_good"].notna().any():
        eq_good = g["eq_realized_good"].eq(1)
        bd_good = g["bd_realized_good"].eq(1)
    else:
        eq_good = g["p_eq"].ge(0.5)
        bd_good = g["p_bd"].ge(0.5)
    state = pd.Series(index=g.index, dtype="object")
    state.loc[eq_good & bd_good] = "both_good"
    state.loc[eq_good & ~bd_good] = "equity_good_only"
    state.loc[~eq_good & bd_good] = "bond_good_only"
    state.loc[~eq_good & ~bd_good] = "both_bad"
    return state.fillna("missing")


def _add_l1_regime_background(ax, g):
    x = g[["date"]].copy()
    x["state"] = _state_from_realized_or_prob_l1(g).values
    x = x.dropna().sort_values("date")
    if x.empty:
        return
    color_map = {
        "both_good": "#bfe5bf",
        "equity_good_only": "#c7dcef",
        "bond_good_only": "#ffd8a8",
        "both_bad": "#f2b6b6",
        "missing": "#eeeeee",
    }
    states = x["state"].tolist()
    dates = pd.to_datetime(x["date"]).tolist()
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            ax.axvspan(dates[start], dates[i - 1], color=color_map.get(states[start], "#eeeeee"), alpha=0.22, lw=0)
            start = i


def _prepare_l1_benchmark_for_dates(dates):
    bench = l1_turnover_benchmark.copy()
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
    bench = bench.loc[bench["date"].isin(pd.to_datetime(dates))].sort_values("date").copy()

    ret_col = _first_existing_col(
        bench,
        ["ret_60_40_net", "ret_60_40", "benchmark_ret", "ret_6040", "ret_60_40_gross"],
    )
    if ret_col is None:
        raise RuntimeError(f"60/40 benchmark has no recognized return column. Columns: {list(bench.columns)}")
    bench["ret_60_40"] = pd.to_numeric(bench[ret_col], errors="coerce")

    turnover_col = _first_existing_col(bench, ["turnover_60_40", "turnover_6040", "benchmark_turnover"])
    if turnover_col is not None:
        bench["turnover_60_40"] = pd.to_numeric(bench[turnover_col], errors="coerce").fillna(0.0)
    else:
        bench["turnover_60_40"] = 0.0

    if "wealth_60_40" not in bench.columns or bench["wealth_60_40"].isna().all():
        start = MV_TC_STARTING_WEALTH if "MV_TC_STARTING_WEALTH" in globals() else 100.0
        bench["wealth_60_40"] = start * (1.0 + bench["ret_60_40"].fillna(0.0)).cumprod()

    return bench


def _benchmark_l1_stats(bench):
    stats = _perf_stats(bench["ret_60_40"])
    stats["ann_turnover"] = float(12.0 * pd.to_numeric(bench["turnover_60_40"], errors="coerce").fillna(0.0).mean())
    return stats


def _l1_perf_box_text(srow, bench_stats):
    def f(x, d=3):
        try:
            if pd.isna(x):
                return "nan"
            return f"{float(x):.{d}f}"
        except Exception:
            return "nan"
    return (
        "PERFORMANCE METRICS\n\n"
        "                 Strategy    60/40\n"
        f"Ann. return      {f(srow.get('ann_return'),3):>8}  {f(bench_stats.get('ann_return'),3):>8}\n"
        f"Ann. vol         {f(srow.get('ann_vol'),3):>8}  {f(bench_stats.get('ann_vol'),3):>8}\n"
        f"Sharpe           {f(srow.get('sharpe'),3):>8}  {f(bench_stats.get('sharpe'),3):>8}\n"
        f"Max drawdown     {f(srow.get('max_drawdown'),3):>8}  {f(bench_stats.get('max_drawdown'),3):>8}\n"
        f"Monthly CVaR 1%  {f(srow.get('cvar_1pct_monthly'),3):>8}  {f(bench_stats.get('cvar_1pct_monthly'),3):>8}\n"
        f"Final wealth     {f(srow.get('final_wealth'),1):>8}  {f(bench_stats.get('final_wealth'),1):>8}\n\n"
        "PORTFOLIO DIAGNOSTICS\n"
        f"Ann. turnover    {f(srow.get('ann_turnover'),3):>8}  {f(bench_stats.get('ann_turnover'),3):>8}\n"
        f"Avg weights      EQ {f(srow.get('avg_w_eq'),2)} | BD {f(srow.get('avg_w_bd'),2)} | Cash {f(srow.get('avg_w_cash'),2)}\n\n"
        "PARAMETERS\n"
        f"gamma={srow.get('gamma'):g}\n"
        f"lambda_trade={srow.get('lambda_trade'):g}\n"
        f"cost={srow.get('tcost_one_way'):g}, prob halflife={srow.get('prob_smooth_halflife'):g}"
    )


def plot_l1_strategy_dashboard(strategy_name):
    if l1_turnover_allocation_results_long is None or l1_turnover_allocation_results_long.empty:
        print(f"No L1 allocation results available for plot: {strategy_name}")
        return

    g = l1_turnover_allocation_results_long.loc[l1_turnover_allocation_results_long["strategy_name"].astype(str).eq(str(strategy_name))].copy()
    if g.empty:
        print(f"Strategy not found in L1 long results: {strategy_name}")
        return
    g = g.sort_values("date").reset_index(drop=True)

    s = l1_turnover_allocation_strategy_summary.loc[l1_turnover_allocation_strategy_summary["strategy"].astype(str).eq(str(strategy_name))].copy()
    if s.empty:
        srow = pd.Series({})
    else:
        srow = s.iloc[0]

    bench = _prepare_l1_benchmark_for_dates(g["date"])
    bench_stats = _benchmark_l1_stats(bench)

    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(
        4, 2,
        width_ratios=[4.6, 1.55],
        height_ratios=[2.2, 1.7, 1.2, 1.0],
        hspace=0.25,
        wspace=0.08,
    )

    ax_wealth = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_weights = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_probs = fig.add_subplot(gs[2, 0])
    ax_dd = fig.add_subplot(gs[2, 1])
    ax_turnover = fig.add_subplot(gs[3, :])

    for ax in [ax_wealth, ax_weights, ax_probs]:
        _add_l1_regime_background(ax, g)

    title = (
        f"{srow.get('feature_design', '')} | {srow.get('family', '')} | {srow.get('allocation_pair', '')} | {srow.get('budget_mode', '')} | "
        f"gamma={srow.get('gamma', np.nan):g}, lambda={srow.get('lambda_trade', np.nan):g}, "
        f"cost={srow.get('tcost_one_way', np.nan):g}, phl={srow.get('prob_smooth_halflife', np.nan):g}"
    )

    ax_wealth.plot(g["date"], g["wealth"], color="#0057b8", lw=2.2, label="Strategy wealth")
    ax_wealth.plot(bench["date"], bench["wealth_60_40"], color="black", lw=1.9, ls="--", alpha=0.70, label="60/40 benchmark wealth")
    ax_wealth.set_title("STRATEGY VS BENCHMARK WEALTH INDEX\n" + title, fontsize=12)
    ax_wealth.set_ylabel("Wealth index, start = 100")
    ax_wealth.grid(True, alpha=0.25)
    ax_wealth.legend(loc="upper left", fontsize=9, frameon=True)

    ax_metrics.axis("off")
    ax_metrics.text(
        0.02, 0.98,
        _l1_perf_box_text(srow, bench_stats),
        va="top", ha="left", family="monospace", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="black", alpha=0.96),
    )

    ax_weights.stackplot(
        g["date"],
        g["w_eq"].clip(0, 1),
        g["w_bd"].clip(0, 1),
        g["w_cash"].clip(0, 1),
        labels=["Equity", "Bonds", "Cash"],
        colors=["#3182bd", "#31a354", "#bdbdbd"],
        alpha=0.88,
        linewidth=0.0,
    )
    ax_weights.set_ylim(0, 1.02)
    ax_weights.set_title("FILLED WEIGHT PLANE THROUGH TIME", fontsize=11)
    ax_weights.set_ylabel("Portfolio weight")
    ax_weights.grid(True, alpha=0.25)
    ax_weights.legend(loc="upper left", fontsize=9, ncol=3, frameon=True)

    # Keep the dashboard layout unchanged, but leave the redundant regime-description panel empty.
    ax_legend.axis("off")

    ax_probs.plot(g["date"], g["p_eq"], color="#0057b8", lw=1.4, label="Predicted equity good-state probability")
    ax_probs.plot(g["date"], g["p_bd"], color="#006d2c", lw=1.4, label="Predicted bond good-state probability")
    ax_probs.axhline(0.5, color="black", lw=0.9, ls="--", alpha=0.55)
    ax_probs.set_ylim(-0.02, 1.02)
    ax_probs.set_title("PREDICTED REGIME PROBABILITIES", fontsize=11)
    ax_probs.set_ylabel("Probability")
    ax_probs.grid(True, alpha=0.25)
    ax_probs.legend(loc="upper left", fontsize=8, frameon=True)

    strategy_dd = _l1_drawdown_from_wealth(g["wealth"])
    bench_dd = _l1_drawdown_from_wealth(bench["wealth_60_40"])
    ax_dd.plot(g["date"], strategy_dd, color="#0057b8", lw=1.4, label="Strategy DD")
    ax_dd.plot(bench["date"], bench_dd, color="black", lw=1.3, ls="--", alpha=0.65, label="60/40 DD")
    ax_dd.set_title("DRAWDOWN", fontsize=11)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(True, alpha=0.25)
    ax_dd.legend(loc="lower left", fontsize=8, frameon=True)

    ax_turnover.bar(g["date"], g["turnover"], width=22, color="#8d8bd6", alpha=0.38, label="Monthly turnover")
    ax_turnover.set_title("MONTHLY TURNOVER", fontsize=11)
    ax_turnover.set_ylabel("Turnover")
    ax_turnover.grid(True, alpha=0.25)
    ax_turnover.legend(loc="upper left", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------------------
# 4.3 Representative parameter-sweep dashboards
# --------------------------------------------------------------------------------------------------

L1_SWEEP_FEATURE_DESIGN = "z_plus_zsmooth"
L1_SWEEP_COST = 0.0005
L1_SWEEP_PROB_HALFLIFE = 0.0
L1_SWEEP_FIXED_LAMBDA = 0.0015
L1_SWEEP_FIXED_GAMMA = 5.0
L1_SWEEP_MAX_LAMBDA_TO_PLOT = 0.01


def _select_l1_parameter_sweep_rows(family, sweep_param):
    x = l1_turnover_allocation_strategy_summary.copy()
    x = x.loc[
        x["feature_design"].astype(str).eq(L1_SWEEP_FEATURE_DESIGN)
        & x["family"].astype(str).str.upper().eq(str(family).upper())
        & x["allocation_pair"].astype(str).eq(L1_ALLOCATION_PAIR_ID)
        & x["budget_mode"].astype(str).eq(L1_BUDGET_MODE)
        & np.isclose(pd.to_numeric(x["tcost_one_way"], errors="coerce"), L1_SWEEP_COST)
        & np.isclose(pd.to_numeric(x["prob_smooth_halflife"], errors="coerce"), L1_SWEEP_PROB_HALFLIFE)
    ].copy()

    if sweep_param == "gamma":
        x = x.loc[np.isclose(pd.to_numeric(x["lambda_trade"], errors="coerce"), L1_SWEEP_FIXED_LAMBDA)].copy()
    elif sweep_param == "lambda_trade":
        x = x.loc[np.isclose(pd.to_numeric(x["gamma"], errors="coerce"), L1_SWEEP_FIXED_GAMMA)].copy()
        x = x.loc[pd.to_numeric(x["lambda_trade"], errors="coerce").le(L1_SWEEP_MAX_LAMBDA_TO_PLOT)].copy()
    else:
        raise ValueError("sweep_param must be 'gamma' or 'lambda_trade'.")

    return (
        x.sort_values([sweep_param, "sharpe"], ascending=[True, False])
        .drop_duplicates(sweep_param, keep="first")
        .reset_index(drop=True)
    )


def _parameter_sweep_metrics_table(rows, bench_stats, sweep_param):
    metrics = [
        ("Ann. return", "ann_return", 3),
        ("Ann. vol", "ann_vol", 3),
        ("Sharpe", "sharpe", 3),
        ("Max drawdown", "max_drawdown", 3),
        ("Monthly CVaR 1%", "cvar_1pct_monthly", 3),
        ("Final wealth", "final_wealth", 1),
        ("Ann. turnover", "ann_turnover", 3),
    ]
    table_rows = []
    for _, row in rows.iterrows():
        record = {"Specification": f"{sweep_param}={float(row[sweep_param]):g}"}
        for label, col, digits in metrics:
            record[label] = f"{float(row[col]):.{digits}f}" if pd.notna(row.get(col)) else "nan"
        table_rows.append(record)

    bench_record = {"Specification": "60/40"}
    for label, col, digits in metrics:
        value = bench_stats.get(col, np.nan)
        bench_record[label] = f"{float(value):.{digits}f}" if pd.notna(value) else "nan"
    table_rows.append(bench_record)
    return pd.DataFrame(table_rows)


def plot_l1_parameter_sweep_dashboard(family, sweep_param):
    rows = _select_l1_parameter_sweep_rows(family, sweep_param)
    if rows.empty:
        print(f"No representative {sweep_param} sweep rows found for {family}.")
        return

    strategy_names = rows["strategy"].astype(str).tolist()
    paths = l1_turnover_allocation_results_long.loc[
        l1_turnover_allocation_results_long["strategy_name"].astype(str).isin(strategy_names)
    ].copy()
    if paths.empty:
        print(f"No representative {sweep_param} sweep paths found for {family}.")
        return
    paths["date"] = pd.to_datetime(paths["date"], errors="coerce")
    paths = paths.sort_values(["strategy_name", "date"])

    bench = _prepare_l1_benchmark_for_dates(paths["date"].drop_duplicates())
    bench_stats = _benchmark_l1_stats(bench)
    metrics_table = _parameter_sweep_metrics_table(rows, bench_stats, sweep_param)

    n = len(rows)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(24, 8.5 + 3.0 * nrows))
    outer = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[3.7, 2.3],
        height_ratios=[2.3, max(2.2, 1.45 * nrows)],
        hspace=0.25,
        wspace=0.10,
    )
    ax_wealth = fig.add_subplot(outer[0, 0])
    ax_metrics = fig.add_subplot(outer[0, 1])
    weight_grid = outer[1, :].subgridspec(nrows, ncols, hspace=0.38, wspace=0.16)

    reference_path = paths.loc[paths["strategy_name"].astype(str).eq(strategy_names[0])].copy()
    _add_l1_regime_background(ax_wealth, reference_path)
    colors = plt.cm.tab20(np.linspace(0.0, 0.95, max(n, 2)))

    for color, (_, row) in zip(colors, rows.iterrows()):
        g = paths.loc[paths["strategy_name"].astype(str).eq(str(row["strategy"]))].sort_values("date")
        ax_wealth.plot(
            g["date"],
            g["wealth"],
            color=color,
            lw=1.8,
            label=f"{sweep_param}={float(row[sweep_param]):g}",
        )

    ax_wealth.plot(
        bench["date"],
        bench["wealth_60_40"],
        color="black",
        lw=2.0,
        ls="--",
        alpha=0.75,
        label="60/40",
    )
    fixed_text = (
        f"lambda={L1_SWEEP_FIXED_LAMBDA:g}"
        if sweep_param == "gamma"
        else f"gamma={L1_SWEEP_FIXED_GAMMA:g}"
    )
    ax_wealth.set_title(
        f"REPRESENTATIVE {sweep_param.upper()} SWEEP | {L1_SWEEP_FEATURE_DESIGN} | {family}\n"
        f"{L1_ALLOCATION_PAIR_ID} | {L1_BUDGET_MODE} | {fixed_text} | "
        f"cost={L1_SWEEP_COST:g} | phl={L1_SWEEP_PROB_HALFLIFE:g}",
        fontsize=12,
    )
    ax_wealth.set_ylabel("Wealth index, start = 100")
    ax_wealth.grid(True, alpha=0.25)
    ax_wealth.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)

    ax_metrics.axis("off")
    table = ax_metrics.table(
        cellText=metrics_table.values,
        colLabels=metrics_table.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.15)
    ax_metrics.set_title("PERFORMANCE METRICS ACROSS THE PARAMETER SWEEP", fontsize=10, pad=8)

    for plot_idx, (color, (_, row)) in enumerate(zip(colors, rows.iterrows())):
        ax = fig.add_subplot(weight_grid[plot_idx // ncols, plot_idx % ncols])
        g = paths.loc[paths["strategy_name"].astype(str).eq(str(row["strategy"]))].sort_values("date")
        _add_l1_regime_background(ax, g)
        ax.stackplot(
            g["date"],
            g["w_eq"].clip(0, 1),
            g["w_bd"].clip(0, 1),
            g["w_cash"].clip(0, 1),
            colors=["#3182bd", "#31a354", "#bdbdbd"],
            alpha=0.88,
            linewidth=0.0,
        )
        ax.set_ylim(0, 1.02)
        ax.set_title(
            f"{sweep_param}={float(row[sweep_param]):g} | Sharpe={float(row['sharpe']):.3f} | "
            f"Turnover={float(row['ann_turnover']):.3f}",
            fontsize=8,
            color=color,
        )
        ax.grid(True, alpha=0.20)
        if plot_idx % ncols == 0:
            ax.set_ylabel("Weight")
        if plot_idx // ncols < nrows - 1:
            ax.tick_params(labelbottom=False)
        ax.tick_params(axis="both", labelsize=7)

    for plot_idx in range(n, nrows * ncols):
        ax = fig.add_subplot(weight_grid[plot_idx // ncols, plot_idx % ncols])
        ax.axis("off")

    fig.text(
        0.5,
        0.015,
        "Filled weight planes: blue = equity, green = bonds, grey = cash.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.99])
    plt.show()


# ==================================================================================================
# 5. RUN FEATURE IMPORTANCE + PREDICTION + L1 ALLOCATION
# ==================================================================================================

print("\n" + "#" * 150)
print("STUDENT PREDICTION AND ALLOCATION DIAGNOSTICS START")
print("#" * 150)

# --------------------------------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------------------------------
feature_importance_summary = prepare_feature_importance()
feature_importance_top_table = (
    feature_importance_summary
    .sort_values(["feature_design", "family", "target", "importance_rank"])
    .groupby(["feature_design", "family", "target"], dropna=False)
    .head(PRINT_TOP_N_FEATURES_PER_COMBO)
    .reset_index(drop=True)
)

to_print_table(
    "FEATURE IMPORTANCE - TOP FEATURES PER FEATURE SET x MODEL x TARGET",
    feature_importance_top_table,
    cols=[
        "feature_design", "family", "target", "importance_rank", "feature", "feature_block", "feature_transform",
        "mean_abs_importance", "mean_signed_importance", "importance_share", "cum_importance_share", "selected_for_plot",
    ],
    n=None,
)

if FEATURE_IMPORTANCE_PLOT:
    for fd in FEATURE_DESIGNS_TO_USE:
        for fam in FAMILIES_TO_USE:
            for target in TARGETS_TO_USE:
                plot_feature_importance_one(feature_importance_summary, fd, fam, target)

# --------------------------------------------------------------------------------------------------
# Prediction diagnostics
# --------------------------------------------------------------------------------------------------
prediction_panel = prepare_prediction_panel()
prediction_metrics = build_prediction_metrics(prediction_panel)

to_print_table(
    "PREDICTION PERFORMANCE - FEATURE SET x MODEL x TARGET",
    prediction_metrics,
    cols=[
        "feature_design", "family", "target", "n_obs", "actual_good_share", "pred_good_share",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss",
        "avg_seed_prob_std", "max_seed_prob_std",
    ],
    sort_by="balanced_accuracy",
    ascending=False,
    n=None,
)

if PREDICTION_DIAGNOSTIC_PLOT:
    for fd in FEATURE_DESIGNS_TO_USE:
        for fam in FAMILIES_TO_USE:
            for target in TARGETS_TO_USE:
                plot_prediction_one(prediction_panel, prediction_metrics, fd, fam, target)

# --------------------------------------------------------------------------------------------------
# L1 trade-penalized allocation
# --------------------------------------------------------------------------------------------------
prediction_long_source_l1 = _pick_prediction_long_for_l1()

l1_turnover_allocation_results_long, l1_turnover_allocation_strategy_summary, l1_turnover_benchmark, l1_panel_map = _run_l1_allocation_grid(
    prediction_long_source_l1
)

l1_economic_universe = _economic_l1_universe(l1_turnover_allocation_strategy_summary)

allocation_cols = [
    "feature_design", "family", "allocation_pair", "gamma", "lambda_trade",
    "tcost_one_way", "prob_smooth_halflife",
    "ann_return", "ann_vol", "sharpe", "max_drawdown", "cvar_1pct_monthly", "final_wealth",
    "ann_turnover", "ann_tcost_drag", "avg_w_eq", "avg_w_bd", "avg_w_cash",
    "both_bad_avg_cash", "equity_good_only_avg_eq", "bond_good_only_avg_bd", "strategy",
]

to_print_table(
    "L1 ALLOCATION - TOP ECONOMIC STRATEGIES",
    l1_economic_universe,
    cols=allocation_cols,
    sort_by="sharpe",
    ascending=False,
    n=L1_PRINT_TOP_N,
)

# Parameter sensitivity based on the economic universe, so the table is not dominated by all-cash rows.
l1_lambda_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["lambda_trade"])
l1_gamma_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["gamma"])
l1_tcost_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["tcost_one_way"])
l1_prob_smoothing_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["prob_smooth_halflife"])
l1_model_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["feature_design", "family"])
l1_lambda_gamma_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["lambda_trade", "gamma"])
l1_lambda_tcost_sensitivity = _sensitivity_table_l1(l1_economic_universe, ["lambda_trade", "tcost_one_way"])
l1_turnover_frontier = _turnover_frontier_l1(l1_economic_universe)

sens_cols = [
    "n_strategies", "best_sharpe", "median_sharpe", "best_ann_return", "median_ann_return",
    "median_ann_turnover", "min_ann_turnover", "best_ann_turnover", "median_avg_cash",
]

to_print_table("L1 PARAMETER SENSITIVITY - LAMBDA TRADE", l1_lambda_sensitivity, cols=["lambda_trade"] + sens_cols, n=None)
to_print_table("L1 PARAMETER SENSITIVITY - RISK AVERSION GAMMA", l1_gamma_sensitivity, cols=["gamma"] + sens_cols, n=None)
to_print_table("L1 PARAMETER SENSITIVITY - TRANSACTION COST", l1_tcost_sensitivity, cols=["tcost_one_way"] + sens_cols, n=None)
to_print_table("L1 PARAMETER SENSITIVITY - PROBABILITY SMOOTHING", l1_prob_smoothing_sensitivity, cols=["prob_smooth_halflife"] + sens_cols, n=None)
to_print_table("L1 MODEL SENSITIVITY - FEATURE SET x MODEL", l1_model_sensitivity, cols=["feature_design", "family"] + sens_cols, n=None)
to_print_table("L1 JOINT SENSITIVITY - LAMBDA x GAMMA", l1_lambda_gamma_sensitivity, cols=["lambda_trade", "gamma"] + sens_cols, sort_by="best_sharpe", ascending=False, n=40)
to_print_table("L1 JOINT SENSITIVITY - LAMBDA x TRANSACTION COST", l1_lambda_tcost_sensitivity, cols=["lambda_trade", "tcost_one_way"] + sens_cols, sort_by="best_sharpe", ascending=False, n=40)

to_print_table(
    "L1 TURNOVER FRONTIER - BEST ECONOMIC STRATEGY UNDER EACH TURNOVER CAP",
    l1_turnover_frontier,
    cols=["turnover_cap"] + allocation_cols,
    n=None,
)

l1_selected_plot_strategies = _select_l1_plot_strategies(l1_turnover_allocation_strategy_summary, l1_economic_universe)

to_print_table(
    "L1 STRATEGIES SELECTED FOR FINAL PLOTS",
    l1_selected_plot_strategies,
    cols=["selection_reason"] + allocation_cols,
    n=None,
)

for _, row in l1_selected_plot_strategies.iterrows():
    plot_l1_strategy_dashboard(row["strategy"])

# Representative one-parameter-at-a-time comparisons for both student-model families.
for sweep_family in ["LOGIT", "RF"]:
    plot_l1_parameter_sweep_dashboard(sweep_family, "gamma")
    plot_l1_parameter_sweep_dashboard(sweep_family, "lambda_trade")

