import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 720)
pd.set_option("display.max_rows", 1200)
pd.set_option("display.float_format", lambda x: f"{x:0.6f}")

EPS = 1e-12
FINAL_MAJORITY_SCOPE = "all_models"
BOND_FEATURE_GROUP = "return_downside_sharpe_vol"

STUDENT_OUTPUT_DIR = ALLOCATION_OUTPUT_DIR
FIGURE_DIR = PROJECT_DIR / "paper" / "figs"
STUDENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_FEATURE_DESIGNS = {
    "smoothed_predictors": "student columns ending in _z_smooth",
    "standardized_and_smoothed_predictors": "student columns ending in _z or _z_smooth",
}

STUDENT_MODELS = ["logistic_regression", "random_forest"]
MODEL_RANDOM_SEEDS = [101, 111, 121]
MODEL_MIN_TRAIN_MONTHS = 96
MODEL_VALIDATION_MONTHS = 36
MODEL_EVAL_THRESHOLD = 0.50

TRANSACTION_COST_ONE_WAY = 0.0005
MOMENT_SHRINKAGE_K = 36.0
WEIGHT_STEP = 0.02
STARTING_WEALTH = 100.0
MAX_EQUITY_WEIGHT = 1.0
MAX_BOND_WEIGHT = 1.0
MAX_TOTAL_RISKY_WEIGHT = 1.0

ALLOCATION_BUDGET_MODES = ["fully_invested", "cash_allowed"]
GAMMA_GRID = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
TAU_GRID = [0.0, 1.0, 3.0, 5.0]
TRANSACTION_COST_GRID = [0.0005, 0.0010, 0.0025, 0.0050]
PROBABILITY_SMOOTHING_HALFLIFE_GRID = [0, 3, 6]
TARGET_VOLATILITY_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
SELECTED_ALLOCATION_TARGET_VOLS = [0.10, 0.15, 0.20]

LOGISTIC_GRID = [
    {"model": "logistic_regression", "model_name": "logistic_l2_C0p03_balanced", "C": 0.03, "class_weight": "balanced"},
    {"model": "logistic_regression", "model_name": "logistic_l2_C0p05_balanced", "C": 0.05, "class_weight": "balanced"},
    {"model": "logistic_regression", "model_name": "logistic_l2_C0p10_balanced", "C": 0.10, "class_weight": "balanced"},
    {"model": "logistic_regression", "model_name": "logistic_l2_C0p25_balanced", "C": 0.25, "class_weight": "balanced"},
    {"model": "logistic_regression", "model_name": "logistic_l2_C0p50_balanced", "C": 0.50, "class_weight": "balanced"},
    {"model": "logistic_regression", "model_name": "logistic_l2_C1p00_balanced", "C": 1.00, "class_weight": "balanced"},
]

RANDOM_FOREST_GRID = [
    {"model": "random_forest", "model_name": "random_forest_d2_leaf10", "n_estimators": 500, "max_depth": 2, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    {"model": "random_forest", "model_name": "random_forest_d2_leaf20", "n_estimators": 500, "max_depth": 2, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    {"model": "random_forest", "model_name": "random_forest_d3_leaf10", "n_estimators": 500, "max_depth": 3, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    {"model": "random_forest", "model_name": "random_forest_d3_leaf20", "n_estimators": 500, "max_depth": 3, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    {"model": "random_forest", "model_name": "random_forest_d4_leaf20", "n_estimators": 500, "max_depth": 4, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": "balanced_subsample"},
]

MODEL_GRID = {
    "logistic_regression": LOGISTIC_GRID,
    "random_forest": RANDOM_FOREST_GRID,
}

TARGET_SPECS = [
    {
        "target": "equity_favourable",
        "asset": "equity",
        "feature_group": "return_sortino",
        "state_col": f"equity_return_sortino_{FINAL_MAJORITY_SCOPE}_state",
        "prob_col": f"equity_return_sortino_{FINAL_MAJORITY_SCOPE}_vote_prob",
        "strength_col": f"equity_return_sortino_{FINAL_MAJORITY_SCOPE}_vote_strength",
        "excess_col": f"equity_return_sortino_{FINAL_MAJORITY_SCOPE}_excess",
        "return_col_h1": "equity_ret_h1",
        "label": "Equity favourable state",
    },
    {
        "target": "bond_favourable",
        "asset": "bond",
        "feature_group": BOND_FEATURE_GROUP,
        "state_col": f"bond_{BOND_FEATURE_GROUP}_{FINAL_MAJORITY_SCOPE}_state",
        "prob_col": f"bond_{BOND_FEATURE_GROUP}_{FINAL_MAJORITY_SCOPE}_vote_prob",
        "strength_col": f"bond_{BOND_FEATURE_GROUP}_{FINAL_MAJORITY_SCOPE}_vote_strength",
        "excess_col": f"bond_{BOND_FEATURE_GROUP}_{FINAL_MAJORITY_SCOPE}_excess",
        "return_col_h1": "bond_ret_h1",
        "label": "Bond favourable state",
    },
]

ALLOCATION_PAIR = {
    "pair": "independent_equity_bond",
    "equity_target": "equity_favourable_h1",
    "bond_target": "bond_favourable_h1",
    "label": "Independent equity and bond favourable states",
}

LINE_COLORS = {
    "60/40": "#4d4d4d",
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
}

STATE_COLORS = {
    "both_favourable": "#2ca25f",
    "equity_favourable_only": "#3182bd",
    "bond_favourable_only": "#fdae6b",
    "both_unfavourable": "#de2d26",
}


def month_end(s):
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def ann_return(x):
    x = pd.Series(x).dropna().astype(float)
    return float(12.0 * x.mean()) if len(x) else np.nan


def ann_vol(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(np.sqrt(12.0) * sd) if np.isfinite(sd) else np.nan


def ann_sharpe(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= EPS:
        return np.nan
    return float(np.sqrt(12.0) * x.mean() / sd)


def max_drawdown_from_returns(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    wealth = (1.0 + x).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1.0).min())


def cvar_left_tail(x, alpha=0.01):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    q = x.quantile(alpha)
    tail = x.loc[x <= q]
    return float(tail.mean()) if len(tail) else np.nan


def balanced_accuracy_binary(y_true, y_pred):
    y_true = pd.Series(y_true).astype(int).to_numpy()
    y_pred = pd.Series(y_pred).astype(int).to_numpy()
    if len(y_true) == 0:
        return np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    r0 = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    r1 = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    vals = [v for v in [r0, r1] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def classification_metrics(y_true, p_hat):
    y = pd.Series(y_true).reset_index(drop=True)
    p = pd.Series(p_hat).reset_index(drop=True)
    m = y.notna() & p.notna()
    y = y.loc[m].astype(int).to_numpy()
    p = p.loc[m].astype(float).clip(EPS, 1.0 - EPS).to_numpy()
    out = {"n": int(len(y)), "auc": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "brier": np.nan, "log_loss": np.nan}
    if len(y) == 0:
        return out
    pred = (p >= MODEL_EVAL_THRESHOLD).astype(int)
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = balanced_accuracy_binary(y, pred)
    out["brier"] = float(brier_score_loss(y, p))
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def good_state_signal(p_hat):
    return (np.asarray(p_hat, dtype=float) >= MODEL_EVAL_THRESHOLD).astype(float)


def timing_rule_metrics(asset_ret_h1, p_hat):
    x = pd.DataFrame({"ret": pd.Series(asset_ret_h1).astype(float).to_numpy(), "p_hat": np.asarray(p_hat, dtype=float)})
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    out = {"signal_share": np.nan, "turnover": np.nan, "net_sharpe": np.nan, "buyhold_sharpe": np.nan, "drawdown_improvement": np.nan, "vol_reduction": np.nan}
    if len(x) < 2:
        return out
    signal = pd.Series(good_state_signal(x["p_hat"]), index=x.index).astype(float)
    turnover = signal.diff().abs()
    if len(turnover):
        turnover.iloc[0] = signal.iloc[0]
    turnover = turnover.fillna(0.0)
    net = signal * x["ret"] - TRANSACTION_COST_ONE_WAY * turnover
    buyhold = x["ret"]
    net_sharpe = ann_sharpe(net)
    buyhold_sharpe = ann_sharpe(buyhold)
    net_dd = max_drawdown_from_returns(net)
    buyhold_dd = max_drawdown_from_returns(buyhold)
    net_vol = ann_vol(net)
    buyhold_vol = ann_vol(buyhold)
    out["signal_share"] = float(signal.mean())
    out["turnover"] = float(turnover.mean())
    out["net_sharpe"] = net_sharpe
    out["buyhold_sharpe"] = buyhold_sharpe
    out["drawdown_improvement"] = net_dd - buyhold_dd if np.isfinite(net_dd) and np.isfinite(buyhold_dd) else np.nan
    out["vol_reduction"] = buyhold_vol - net_vol if np.isfinite(net_vol) and np.isfinite(buyhold_vol) else np.nan
    return out


def safe_colname(x):
    s = str(x).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s).strip("_")


def feature_block_from_name(feature):
    s = str(feature).lower()
    if s.startswith("ff_") or "factor" in s or "small" in s or "value" in s or "industry" in s:
        return "equity_cross_section"
    if "tsy" in s or "bond" in s or "curve" in s or "dgs" in s or "rate" in s:
        return "treasury_rates"
    if "credit" in s or "baa" in s or "aaa" in s or "hy" in s or "ig" in s:
        return "credit"
    if "cmdty" in s or "commodity" in s:
        return "commodities"
    if "nfci" in s or "anfci" in s or "financial" in s:
        return "financial_conditions"
    if "vix" in s or "vol" in s or "rv_" in s or "rho" in s or "corr" in s or "lambda" in s:
        return "volatility_dependence"
    if "inflation" in s or "cfnai" in s or "sentiment" in s or "macro" in s:
        return "macro"
    return "other"


def student_feature_cols(df, feature_design):
    cols = []
    for c in df.columns:
        if c == "date":
            continue
        s = str(c)
        if feature_design == "smoothed_predictors" and s.endswith("_z_smooth"):
            cols.append(c)
        if feature_design == "standardized_and_smoothed_predictors" and (s.endswith("_z") or s.endswith("_z_smooth")):
            cols.append(c)
    return cols


def load_monthly_returns():
    candidate_paths = [
        RAW_DATA_DIR / "base_panel_monthly_through_2026_04.parquet",
        RAW_DATA_DIR / "base_panel_monthly_through_2025_12.parquet",
        RAW_DATA_DIR / "Monthly Base Panel through 2024-11.parquet",
        RAW_DATA_DIR / "base_panel_monthly.parquet",
    ]
    for p in candidate_paths:
        if p.exists():
            x = pd.read_parquet(p).copy()
            x["date"] = month_end(x["date"])
            for c in ["sprtrn_sp500", "agg_ret", "rf"]:
                if c not in x.columns:
                    x[c] = np.nan
                x[c] = pd.to_numeric(x[c], errors="coerce")
            out = x[["date", "sprtrn_sp500", "agg_ret", "rf"]].copy()
            out = out.dropna(subset=["date", "sprtrn_sp500", "agg_ret", "rf"]).sort_values("date").drop_duplicates("date", keep="last")
            out["equity_excess"] = out["sprtrn_sp500"] - out["rf"]
            out["bond_excess"] = out["agg_ret"] - out["rf"]
            out["equity_ret_h1"] = out["equity_excess"].shift(-1)
            out["bond_ret_h1"] = out["bond_excess"].shift(-1)
            out["return_source"] = str(p)
            return out.reset_index(drop=True)
    out = majority_vote_labels_wide[["date"]].copy()
    out["equity_excess"] = pd.to_numeric(majority_vote_labels_wide[TARGET_SPECS[0]["excess_col"]], errors="coerce")
    out["bond_excess"] = pd.to_numeric(majority_vote_labels_wide[TARGET_SPECS[1]["excess_col"]], errors="coerce")
    out["equity_ret_h1"] = out["equity_excess"].shift(-1)
    out["bond_ret_h1"] = out["bond_excess"].shift(-1)
    out["return_source"] = "teacher_label_excess_returns"
    return out


def build_prediction_targets_h1():
    h1 = majority_vote_prediction_targets_h1.copy()
    h1["date"] = month_end(h1["date"])
    h1 = h1.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out = h1[["date"]].copy()
    for spec in TARGET_SPECS:
        target = spec["target"]
        out[f"{target}_h1"] = pd.to_numeric(h1[spec["state_col"]], errors="coerce")
        out[f"{target}_vote_prob_h1"] = pd.to_numeric(h1[spec["prob_col"]], errors="coerce")
        out[f"{target}_vote_strength_h1"] = pd.to_numeric(h1[spec["strength_col"]], errors="coerce")
    return out


def build_historical_label_return_panel(targets_h1, returns):
    out = targets_h1[["date", "equity_favourable_h1", "bond_favourable_h1"]].copy()
    out = out.merge(returns[["date", "equity_ret_h1", "bond_ret_h1"]], on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def build_prediction_base(student_df, targets_h1, returns, feature_design):
    student = student_df.copy()
    student["date"] = month_end(student["date"])
    student = student.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    fcols = student_feature_cols(student, feature_design)
    target_cols = [f"{spec['target']}_h1" for spec in TARGET_SPECS]
    x = student.merge(targets_h1, on="date", how="inner")
    x = x.merge(returns[["date", "equity_excess", "bond_excess", "equity_ret_h1", "bond_ret_h1"]], on="date", how="left")
    x = x.dropna(subset=target_cols, how="all")
    x = x.loc[x[fcols].notna().any(axis=1)].copy()
    x["year"] = x["date"].dt.year
    return x.sort_values("date").reset_index(drop=True), fcols, target_cols


def target_meta(target_cols):
    meta = {}
    specs = {s["target"]: s for s in TARGET_SPECS}
    for c in target_cols:
        target = str(c).replace("_h1", "")
        spec = specs[target]
        meta[c] = {
            "target": target,
            "asset": spec["asset"],
            "feature_group": spec["feature_group"],
            "return_col_h1": spec["return_col_h1"],
            "label": spec["label"],
            "target_design": "next_month_majority_vote_label",
        }
    return meta


student_predictor_panel = student_ensemble_feature_panel.copy()
returns = load_monthly_returns()
prediction_targets_h1 = build_prediction_targets_h1()
historical_label_return_panel = build_historical_label_return_panel(prediction_targets_h1, returns)

prediction_base_by_feature_design = {}
feature_cols_by_feature_design = {}
for feature_design in STUDENT_FEATURE_DESIGNS:
    base, fcols, target_cols_h1 = build_prediction_base(student_predictor_panel, prediction_targets_h1, returns, feature_design)
    prediction_base_by_feature_design[feature_design] = base.copy()
    feature_cols_by_feature_design[feature_design] = list(fcols)

target_meta_h1 = target_meta(target_cols_h1)


def build_estimator(params, seed):
    if params["model"] == "logistic_regression":
        clf = LogisticRegression(
            penalty="l2",
            C=params["C"],
            class_weight=params["class_weight"],
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", clf)])
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=seed,
        n_jobs=1,
        bootstrap=True,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)])


def probability_of_favourable_state(pipe, X):
    model = pipe.named_steps["model"]
    probs = pipe.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return probs[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def choose_model(validation_table):
    x = validation_table.copy()
    x["auc_rank"] = x["val_auc"].fillna(-999.0)
    x["balanced_accuracy_rank"] = x["val_balanced_accuracy"].fillna(-999.0)
    x["brier_rank"] = x["val_brier"].fillna(999.0)
    x["net_sharpe_rank"] = x["val_timing_net_sharpe"].fillna(-999.0)
    x["drawdown_rank"] = x["val_timing_drawdown_improvement"].fillna(-999.0)
    x["vol_reduction_rank"] = x["val_timing_vol_reduction"].fillna(-999.0)
    x = x.sort_values(
        ["auc_rank", "balanced_accuracy_rank", "brier_rank", "net_sharpe_rank", "drawdown_rank", "vol_reduction_rank", "model_name"],
        ascending=[False, False, True, False, False, False, True],
    )
    return str(x.iloc[0]["model_name"])


def model_importance(pipe, feature_cols, model_name):
    model = pipe.named_steps["model"]
    if model_name == "logistic_regression" and hasattr(model, "coef_"):
        signed = np.asarray(model.coef_[0], dtype=float)
        importance = np.abs(signed)
    elif model_name == "random_forest" and hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        signed = importance.copy()
    else:
        signed = np.zeros(len(feature_cols), dtype=float)
        importance = np.zeros(len(feature_cols), dtype=float)
    if len(signed) != len(feature_cols):
        signed = np.zeros(len(feature_cols), dtype=float)
    if len(importance) != len(feature_cols):
        importance = np.zeros(len(feature_cols), dtype=float)
    return signed, importance


def run_student_model(feature_design, model_name, df, feature_cols, target_cols, meta):
    grid = MODEL_GRID[model_name]
    result_rows = []
    selected_rows = []
    prediction_parts = []
    importance_parts = []
    seed_rows = []
    unique_dates = df["date"].sort_values().drop_duplicates().reset_index(drop=True)
    first_test_idx = MODEL_MIN_TRAIN_MONTHS + MODEL_VALIDATION_MONTHS

    for target_col in target_cols:
        asset = meta[target_col]["asset"]
        feature_group = meta[target_col]["feature_group"]
        ret_col = meta[target_col]["return_col_h1"]
        valid_dates = df.loc[df[target_col].notna(), "date"].sort_values().drop_duplicates().reset_index(drop=True)
        test_dates = valid_dates.iloc[first_test_idx:].tolist() if len(valid_dates) > first_test_idx else []
        target_prediction_parts = []

        for test_date in test_dates:
            date_index = unique_dates[unique_dates.eq(test_date)]
            if len(date_index) == 0:
                continue
            test_idx = int(date_index.index[0])
            train_end_idx = test_idx - MODEL_VALIDATION_MONTHS
            if train_end_idx < MODEL_MIN_TRAIN_MONTHS:
                continue

            train_dates = unique_dates.iloc[:train_end_idx]
            validation_dates = unique_dates.iloc[train_end_idx:test_idx]
            work_cols = ["date", "year", ret_col, target_col] + list(feature_cols)
            train = df.loc[df["date"].isin(train_dates), work_cols].dropna(subset=[target_col]).copy()
            validation = df.loc[df["date"].isin(validation_dates), work_cols].dropna(subset=[target_col]).copy()
            test = df.loc[df["date"].eq(test_date), work_cols].dropna(subset=[target_col]).copy()

            train = train.loc[train[feature_cols].notna().any(axis=1)].copy()
            validation = validation.loc[validation[feature_cols].notna().any(axis=1)].copy()
            test = test.loc[test[feature_cols].notna().any(axis=1)].copy()

            if len(train) < MODEL_MIN_TRAIN_MONTHS or len(validation) < MODEL_VALIDATION_MONTHS or test.empty:
                continue

            y_train = train[target_col].astype(int)
            y_validation = validation[target_col].astype(int)
            y_test = test[target_col].astype(int)
            if y_train.nunique() < 2:
                continue

            X_train = train[feature_cols]
            X_validation = validation[feature_cols]
            X_test = test[feature_cols]
            validation_rows = []

            for params in grid:
                seed_predictions = []
                for seed in MODEL_RANDOM_SEEDS:
                    pipe = build_estimator(params, seed)
                    pipe.fit(X_train, y_train)
                    p_seed = probability_of_favourable_state(pipe, X_validation)
                    seed_predictions.append(p_seed)
                    m_seed = classification_metrics(y_validation, p_seed)
                    seed_rows.append({
                        "feature_design": feature_design,
                        "model": model_name,
                        "target": target_col,
                        "asset": asset,
                        "feature_group": feature_group,
                        "test_date": test_date,
                        "model_name": params["model_name"],
                        "seed": int(seed),
                        "split": "validation",
                        "n": m_seed["n"],
                        "auc": m_seed["auc"],
                        "accuracy": m_seed["accuracy"],
                        "balanced_accuracy": m_seed["balanced_accuracy"],
                        "brier": m_seed["brier"],
                        "log_loss": m_seed["log_loss"],
                    })

                p_validation = np.mean(np.column_stack(seed_predictions), axis=1)
                m_validation = classification_metrics(y_validation, p_validation)
                timing = timing_rule_metrics(validation[ret_col].to_numpy(dtype=float), p_validation)
                row = {
                    "feature_design": feature_design,
                    "model": model_name,
                    "target": target_col,
                    "asset": asset,
                    "feature_group": feature_group,
                    "test_date": test_date,
                    "model_name": params["model_name"],
                    "threshold": float(MODEL_EVAL_THRESHOLD),
                    "n_train": int(len(train)),
                    "n_validation": int(len(validation)),
                    "n_test": int(len(test)),
                    "train_positive_share": float(y_train.mean()),
                    "validation_positive_share": float(y_validation.mean()),
                    "test_positive_share": float(y_test.mean()),
                    "val_auc": m_validation["auc"],
                    "val_accuracy": m_validation["accuracy"],
                    "val_balanced_accuracy": m_validation["balanced_accuracy"],
                    "val_brier": m_validation["brier"],
                    "val_log_loss": m_validation["log_loss"],
                    "val_timing_signal_share": timing["signal_share"],
                    "val_timing_turnover": timing["turnover"],
                    "val_timing_net_sharpe": timing["net_sharpe"],
                    "val_timing_buyhold_sharpe": timing["buyhold_sharpe"],
                    "val_timing_drawdown_improvement": timing["drawdown_improvement"],
                    "val_timing_vol_reduction": timing["vol_reduction"],
                }
                for k, v in params.items():
                    if k not in row:
                        row[k] = v
                validation_rows.append(row)

            validation_table = pd.DataFrame(validation_rows)
            if validation_table.empty:
                continue

            selected_name = choose_model(validation_table)
            selected_params = next(p for p in grid if p["model_name"] == selected_name)
            train_validation = pd.concat([train, validation], axis=0).sort_values("date").reset_index(drop=True)
            X_train_validation = train_validation[feature_cols]
            y_train_validation = train_validation[target_col].astype(int)
            if y_train_validation.nunique() < 2:
                continue

            test_seed_predictions = []
            importance_rows = []
            for seed in MODEL_RANDOM_SEEDS:
                pipe = build_estimator(selected_params, seed)
                pipe.fit(X_train_validation, y_train_validation)
                p_seed_test = probability_of_favourable_state(pipe, X_test)
                test_seed_predictions.append(p_seed_test)
                m_test_seed = classification_metrics(y_test, p_seed_test)
                seed_rows.append({
                    "feature_design": feature_design,
                    "model": model_name,
                    "target": target_col,
                    "asset": asset,
                    "feature_group": feature_group,
                    "test_date": test_date,
                    "model_name": selected_name,
                    "seed": int(seed),
                    "split": "test_selected_model",
                    "n": m_test_seed["n"],
                    "auc": m_test_seed["auc"],
                    "accuracy": m_test_seed["accuracy"],
                    "balanced_accuracy": m_test_seed["balanced_accuracy"],
                    "brier": m_test_seed["brier"],
                    "log_loss": m_test_seed["log_loss"],
                })
                signed, importance = model_importance(pipe, feature_cols, model_name)
                importance_rows.append(pd.DataFrame({
                    "feature_design": feature_design,
                    "model": model_name,
                    "target": target_col,
                    "asset": asset,
                    "feature_group": feature_group,
                    "test_date": test_date,
                    "selected_model": selected_name,
                    "seed": int(seed),
                    "feature": feature_cols,
                    "signed_importance": signed,
                    "abs_importance": np.abs(signed) if model_name == "logistic_regression" else importance,
                    "importance": importance,
                }))

            p_test = np.mean(np.column_stack(test_seed_predictions), axis=1)
            m_test = classification_metrics(y_test, p_test)
            signal = good_state_signal(p_test)

            for _, r in validation_table.iterrows():
                row = r.to_dict()
                selected = row["model_name"] == selected_name
                row["selected_model"] = bool(selected)
                row["test_auc"] = m_test["auc"] if selected else np.nan
                row["test_accuracy"] = m_test["accuracy"] if selected else np.nan
                row["test_balanced_accuracy"] = m_test["balanced_accuracy"] if selected else np.nan
                row["test_brier"] = m_test["brier"] if selected else np.nan
                row["test_log_loss"] = m_test["log_loss"] if selected else np.nan
                result_rows.append(row)
                if selected:
                    selected_rows.append(row)

            pred = test[["date", "year", target_col, ret_col]].copy().rename(columns={target_col: "y"})
            pred["p_hat"] = p_test
            pred["signal"] = signal
            pred["threshold"] = MODEL_EVAL_THRESHOLD
            pred["feature_design"] = feature_design
            pred["model"] = model_name
            pred["selected_model"] = selected_name
            pred["target"] = target_col
            pred["asset"] = asset
            pred["feature_group"] = feature_group
            pred["n_train_validation"] = int(len(train_validation))
            pred["train_start"] = train_validation["date"].min()
            pred["train_end"] = train_validation["date"].max()
            target_prediction_parts.append(pred)

            if importance_rows:
                importance_parts.append(pd.concat(importance_rows, axis=0, ignore_index=True))

        if target_prediction_parts:
            prediction_parts.append(pd.concat(target_prediction_parts, axis=0, ignore_index=True))

    result_table = pd.DataFrame(result_rows)
    selected_model_table = pd.DataFrame(selected_rows)
    prediction_panel = pd.concat(prediction_parts, axis=0, ignore_index=True).sort_values(["feature_design", "model", "target", "date"]).reset_index(drop=True) if prediction_parts else pd.DataFrame()
    feature_importance = pd.concat(importance_parts, axis=0, ignore_index=True) if importance_parts else pd.DataFrame()
    seed_diagnostics = pd.DataFrame(seed_rows)

    return result_table, selected_model_table, prediction_panel, feature_importance, seed_diagnostics


def summarize_student_predictions(prediction_panel, base_by_feature_design, meta):
    rows = []
    if prediction_panel.empty:
        return pd.DataFrame()
    for (feature_design, model_name, target_col), g in prediction_panel.groupby(["feature_design", "model", "target"], dropna=False):
        ret_col = meta[target_col]["return_col_h1"]
        base = base_by_feature_design[feature_design]
        x = g[["date", "y", "p_hat", "signal"]].merge(base[["date", ret_col]].drop_duplicates("date"), on="date", how="left")
        x = x.dropna(subset=["y", "p_hat"]).copy()
        if x.empty:
            continue
        metrics = classification_metrics(x["y"], x["p_hat"])
        signal = x["signal"].astype(float)
        turnover = signal.diff().abs()
        if len(turnover):
            turnover.iloc[0] = signal.iloc[0]
        turnover = turnover.fillna(0.0)
        gross = signal * x[ret_col].astype(float)
        net = gross - TRANSACTION_COST_ONE_WAY * turnover
        buyhold = x[ret_col].astype(float)
        net_sharpe = ann_sharpe(net)
        buyhold_sharpe = ann_sharpe(buyhold)
        net_dd = max_drawdown_from_returns(net)
        buyhold_dd = max_drawdown_from_returns(buyhold)
        net_vol = ann_vol(net)
        buyhold_vol = ann_vol(buyhold)
        rows.append({
            "feature_design": feature_design,
            "model": model_name,
            "target": target_col,
            "asset": meta[target_col]["asset"],
            "feature_group": meta[target_col]["feature_group"],
            "n_oos_months": int(len(x)),
            "first_oos": x["date"].min(),
            "last_oos": x["date"].max(),
            "positive_share": float(x["y"].mean()),
            "prob_mean": float(x["p_hat"].mean()),
            "prob_std": float(x["p_hat"].std(ddof=1)),
            "oos_auc": metrics["auc"],
            "oos_accuracy": metrics["accuracy"],
            "oos_balanced_accuracy": metrics["balanced_accuracy"],
            "oos_brier": metrics["brier"],
            "oos_log_loss": metrics["log_loss"],
            "oos_signal_share": float(signal.mean()),
            "oos_turnover": float(turnover.mean()),
            "oos_net_ann_return": ann_return(net),
            "oos_net_ann_vol": net_vol,
            "oos_net_sharpe": net_sharpe,
            "oos_net_max_drawdown": net_dd,
            "oos_buyhold_ann_return": ann_return(buyhold),
            "oos_buyhold_ann_vol": buyhold_vol,
            "oos_buyhold_sharpe": buyhold_sharpe,
            "oos_buyhold_max_drawdown": buyhold_dd,
            "oos_sharpe_lift_vs_buyhold": net_sharpe - buyhold_sharpe if np.isfinite(net_sharpe) and np.isfinite(buyhold_sharpe) else np.nan,
            "oos_drawdown_improvement_vs_buyhold": net_dd - buyhold_dd if np.isfinite(net_dd) and np.isfinite(buyhold_dd) else np.nan,
            "oos_vol_reduction_vs_buyhold": buyhold_vol - net_vol if np.isfinite(net_vol) and np.isfinite(buyhold_vol) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["feature_design", "target", "model"]).reset_index(drop=True)


def summarize_student_feature_importance(feature_importance):
    if feature_importance.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    x = feature_importance.copy()
    x["feature_block"] = x["feature"].map(feature_block_from_name)
    x["rank_in_fit"] = x.groupby(["feature_design", "model", "target", "test_date", "seed"], dropna=False)["abs_importance"].rank(method="first", ascending=False)
    x["is_top10"] = x["rank_in_fit"].le(10)
    x["is_top20"] = x["rank_in_fit"].le(20)
    x["positive"] = x["signed_importance"].gt(0)
    x["negative"] = x["signed_importance"].lt(0)
    feature_summary = (
        x.groupby(["feature_design", "model", "target", "asset", "feature_group", "feature", "feature_block"], dropna=False)
        .agg(
            n_fits=("abs_importance", "count"),
            mean_signed_importance=("signed_importance", "mean"),
            mean_abs_importance=("abs_importance", "mean"),
            median_abs_importance=("abs_importance", "median"),
            importance_std=("abs_importance", "std"),
            positive_share=("positive", "mean"),
            negative_share=("negative", "mean"),
            top10_frequency=("is_top10", "mean"),
            top20_frequency=("is_top20", "mean"),
        )
        .reset_index()
    )
    feature_summary["sign_consistency"] = feature_summary[["positive_share", "negative_share"]].max(axis=1)
    feature_summary = feature_summary.sort_values(["feature_design", "model", "target", "mean_abs_importance"], ascending=[True, True, True, False]).reset_index(drop=True)
    block_summary = (
        feature_summary.groupby(["feature_design", "model", "target", "asset", "feature_group", "feature_block"], dropna=False)
        .agg(
            n_features=("feature", "nunique"),
            total_mean_abs_importance=("mean_abs_importance", "sum"),
            avg_mean_abs_importance=("mean_abs_importance", "mean"),
            max_mean_abs_importance=("mean_abs_importance", "max"),
            avg_top20_frequency=("top20_frequency", "mean"),
        )
        .reset_index()
    )
    block_summary["block_share_of_importance"] = block_summary.groupby(["feature_design", "model", "target"], dropna=False)["total_mean_abs_importance"].transform(lambda z: z / max(float(z.sum()), EPS))
    block_summary = block_summary.sort_values(["feature_design", "model", "target", "block_share_of_importance"], ascending=[True, True, True, False]).reset_index(drop=True)
    logit_coefficients = x.loc[x["model"].eq("logistic_regression")].copy()
    return feature_summary, block_summary, logit_coefficients


def build_pair_prediction_panel(prediction_panel, feature_design, model_name):
    x = prediction_panel.loc[prediction_panel["feature_design"].eq(feature_design) & prediction_panel["model"].eq(model_name)].copy()
    pair_id = ALLOCATION_PAIR["pair"]
    base = x[["date", "year"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    wide = base.copy()
    mapping = [("equity", ALLOCATION_PAIR["equity_target"]), ("bond", ALLOCATION_PAIR["bond_target"])]
    for short, target_col in mapping:
        g = x.loc[x["target"].eq(target_col)].copy()
        if g.empty:
            continue
        h = g[["date", "y", "p_hat", "signal", "selected_model", "threshold"]].copy()
        h = h.rename(columns={
            "y": f"y_{short}_{pair_id}",
            "p_hat": f"p_{short}_{pair_id}",
            "signal": f"signal_{short}_{pair_id}",
            "selected_model": f"model_{short}_{pair_id}",
            "threshold": f"threshold_{short}_{pair_id}",
        })
        wide = wide.merge(h, on="date", how="left")
    wide["feature_design"] = feature_design
    wide["model"] = model_name
    wide["allocation_pair"] = pair_id
    return wide


def probability_mixture_moments(p1, moments):
    p1 = float(np.clip(p1, 0.0, 1.0))
    p0 = 1.0 - p1
    mu0 = moments[0]["mu"]
    mu1 = moments[1]["mu"]
    var0 = moments[0]["var"]
    var1 = moments[1]["var"]
    mu = p0 * mu0 + p1 * mu1
    second = p0 * (var0 + mu0 ** 2) + p1 * (var1 + mu1 ** 2)
    return float(mu), float(max(second - mu ** 2, 1e-6))


def initial_moments_from_pre_oos(historical_panel, first_prediction_date):
    pre = historical_panel.loc[historical_panel["date"].lt(first_prediction_date)].copy()
    z = pre[["equity_ret_h1", "bond_ret_h1"]].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    mu_eq = float(z["equity_ret_h1"].mean()) if len(z) else 0.0
    mu_bd = float(z["bond_ret_h1"].mean()) if len(z) else 0.0
    var_eq = max(float(z["equity_ret_h1"].var(ddof=1)), 1e-6) if len(z) >= 2 else 1e-6
    var_bd = max(float(z["bond_ret_h1"].var(ddof=1)), 1e-6) if len(z) >= 2 else 1e-6
    rho = 0.0
    if len(z) >= 3:
        rho = float(z["equity_ret_h1"].corr(z["bond_ret_h1"]))
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, -0.95, 0.95))
    return {"eq": {"mu": mu_eq, "var": var_eq}, "bd": {"mu": mu_bd, "var": var_bd}, "rho": rho, "n_obs": int(len(z))}


def moment_dict_from_arrays(y, r, initial_mu, initial_var):
    y = np.asarray(y, dtype=float)
    r = np.asarray(r, dtype=float)
    valid = np.isfinite(y) & np.isfinite(r)
    out = {}
    for state in [0, 1]:
        rs = r[valid & (y == float(state))]
        n = int(len(rs))
        mu_raw = float(np.mean(rs)) if n > 0 else float(initial_mu)
        var_raw = float(np.var(rs, ddof=1)) if n >= 2 else float(initial_var)
        w = n / (n + MOMENT_SHRINKAGE_K) if (n + MOMENT_SHRINKAGE_K) > 0 else 0.0
        mu = w * mu_raw + (1.0 - w) * float(initial_mu)
        var = w * var_raw + (1.0 - w) * float(initial_var)
        out[state] = {"n": n, "weight": float(w), "mu": float(mu), "var": float(max(var, EPS)), "mu_raw": mu_raw, "var_raw": var_raw}
    return out


def corr_from_arrays(eq_ret, bd_ret, initial_rho):
    eq_ret = np.asarray(eq_ret, dtype=float)
    bd_ret = np.asarray(bd_ret, dtype=float)
    valid = np.isfinite(eq_ret) & np.isfinite(bd_ret)
    x = eq_ret[valid]
    y = bd_ret[valid]
    n = int(len(x))
    rho_emp = float(initial_rho)
    if n >= 3:
        rho_emp = float(np.corrcoef(x, y)[0, 1])
    rho_emp = rho_emp if np.isfinite(rho_emp) else float(initial_rho)
    w = n / (n + MOMENT_SHRINKAGE_K) if (n + MOMENT_SHRINKAGE_K) > 0 else 0.0
    rho = w * rho_emp + (1.0 - w) * float(initial_rho)
    return float(np.clip(rho, -0.999, 0.999)), n, float(w), rho_emp


def precompute_allocation_moments(panel, historical_panel, initial_moments):
    dates = pd.to_datetime(panel["date"], errors="coerce").dropna().sort_values().drop_duplicates().tolist()
    hist = historical_panel[["date", "equity_favourable_h1", "bond_favourable_h1", "equity_ret_h1", "bond_ret_h1"]].copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    h_dates = hist["date"].to_numpy(dtype="datetime64[ns]")
    eq_state = pd.to_numeric(hist["equity_favourable_h1"], errors="coerce").to_numpy(dtype=float)
    bd_state = pd.to_numeric(hist["bond_favourable_h1"], errors="coerce").to_numpy(dtype=float)
    eq_ret = pd.to_numeric(hist["equity_ret_h1"], errors="coerce").to_numpy(dtype=float)
    bd_ret = pd.to_numeric(hist["bond_ret_h1"], errors="coerce").to_numpy(dtype=float)
    out = {}
    for dt in dates:
        d64 = np.datetime64(pd.Timestamp(dt).to_datetime64())
        m = h_dates < d64
        eq_mom = moment_dict_from_arrays(eq_state[m], eq_ret[m], initial_moments["eq"]["mu"], initial_moments["eq"]["var"])
        bd_mom = moment_dict_from_arrays(bd_state[m], bd_ret[m], initial_moments["bd"]["mu"], initial_moments["bd"]["var"])
        rho, rho_n, rho_weight, rho_empirical = corr_from_arrays(eq_ret[m], bd_ret[m], initial_moments["rho"])
        out[pd.Timestamp(dt)] = {"eq_mom": eq_mom, "bd_mom": bd_mom, "rho": rho, "rho_n_obs": rho_n, "rho_weight": rho_weight, "rho_empirical": rho_empirical}
    return out


def smooth_probability(s, halflife):
    x = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    if halflife is None or float(halflife) <= 0:
        return x
    return x.ewm(halflife=float(halflife), adjust=False, min_periods=1).mean()


def build_monthly_allocation_inputs(panel, moment_lookup, prob_halflife):
    pair_id = ALLOCATION_PAIR["pair"]
    cols = ["date", "equity_ret_h1", "bond_ret_h1", f"y_equity_{pair_id}", f"p_equity_{pair_id}", f"y_bond_{pair_id}", f"p_bond_{pair_id}"]
    x = panel[cols].copy().replace([np.inf, -np.inf], np.nan).sort_values("date").reset_index(drop=True)
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[f"p_equity_{pair_id}"] = smooth_probability(x[f"p_equity_{pair_id}"], prob_halflife)
    x[f"p_bond_{pair_id}"] = smooth_probability(x[f"p_bond_{pair_id}"], prob_halflife)
    rows = []
    for _, row in x.iterrows():
        date = pd.Timestamp(row["date"])
        if date not in moment_lookup:
            continue
        if not np.isfinite(row[f"p_equity_{pair_id}"]) or not np.isfinite(row[f"p_bond_{pair_id}"]):
            continue
        if not np.isfinite(row["equity_ret_h1"]) or not np.isfinite(row["bond_ret_h1"]):
            continue
        m = moment_lookup[date]
        mu_eq, var_eq = probability_mixture_moments(row[f"p_equity_{pair_id}"], m["eq_mom"])
        mu_bd, var_bd = probability_mixture_moments(row[f"p_bond_{pair_id}"], m["bd_mom"])
        sd_eq = np.sqrt(max(var_eq, 1e-8))
        sd_bd = np.sqrt(max(var_bd, 1e-8))
        rho = float(m["rho"])
        rows.append({
            "date": date,
            "p_eq": float(row[f"p_equity_{pair_id}"]),
            "p_bd": float(row[f"p_bond_{pair_id}"]),
            "mu_eq": float(mu_eq),
            "mu_bd": float(mu_bd),
            "sd_eq": float(sd_eq),
            "sd_bd": float(sd_bd),
            "rho": rho,
            "cov00": float(sd_eq ** 2),
            "cov01": float(rho * sd_eq * sd_bd),
            "cov11": float(sd_bd ** 2),
            "equity_ret_h1": float(row["equity_ret_h1"]),
            "bond_ret_h1": float(row["bond_ret_h1"]),
            "eq_n_state0": int(m["eq_mom"][0]["n"]),
            "eq_n_state1": int(m["eq_mom"][1]["n"]),
            "bd_n_state0": int(m["bd_mom"][0]["n"]),
            "bd_n_state1": int(m["bd_mom"][1]["n"]),
            "eq_weight_state0": float(m["eq_mom"][0]["weight"]),
            "eq_weight_state1": float(m["eq_mom"][1]["weight"]),
            "bd_weight_state0": float(m["bd_mom"][0]["weight"]),
            "bd_weight_state1": float(m["bd_mom"][1]["weight"]),
            "rho_n_obs": int(m["rho_n_obs"]),
            "rho_weight": float(m["rho_weight"]),
            "rho_empirical": float(m["rho_empirical"]),
        })
    return pd.DataFrame(rows)


def make_weight_grid(budget_mode):
    grid = np.arange(0.0, 1.0 + 0.5 * WEIGHT_STEP, WEIGHT_STEP)
    rows = []
    for w_eq in grid:
        for w_bd in grid:
            total = w_eq + w_bd
            keep = False
            if budget_mode == "fully_invested":
                keep = abs(total - 1.0) <= 0.5 * WEIGHT_STEP + 1e-10
            if budget_mode == "cash_allowed":
                keep = total <= MAX_TOTAL_RISKY_WEIGHT + EPS
            if keep and w_eq <= MAX_EQUITY_WEIGHT + EPS and w_bd <= MAX_BOND_WEIGHT + EPS:
                rows.append((float(w_eq), float(w_bd)))
    return np.asarray(rows, dtype=float)


WEIGHT_GRIDS = {mode: make_weight_grid(mode) for mode in ALLOCATION_BUDGET_MODES}


def allocation_param_frame(budget_mode, prob_halflife):
    rows = []
    for gamma in GAMMA_GRID:
        for tau in TAU_GRID:
            for tcost in TRANSACTION_COST_GRID:
                rows.append({"budget_mode": budget_mode, "gamma": float(gamma), "tau": float(tau), "tcost_one_way": float(tcost), "prob_smooth_halflife": float(prob_halflife)})
    return pd.DataFrame(rows)


def simulate_allocation_param_block(monthly_inputs, feature_design, model_name, budget_mode, prob_halflife):
    pair_id = ALLOCATION_PAIR["pair"]
    m = monthly_inputs.copy().sort_values("date").reset_index(drop=True)
    if m.empty:
        return pd.DataFrame()
    params = allocation_param_frame(budget_mode, prob_halflife)
    n_param = len(params)
    W = WEIGHT_GRIDS[budget_mode]
    W_eq = W[:, 0]
    W_bd = W[:, 1]
    gamma = params["gamma"].to_numpy(dtype=float)
    tau = params["tau"].to_numpy(dtype=float)
    tcost_one_way = params["tcost_one_way"].to_numpy(dtype=float)
    prev_w_eq = np.full(n_param, 0.60 if budget_mode == "fully_invested" else 0.0, dtype=float)
    prev_w_bd = np.full(n_param, 0.40 if budget_mode == "fully_invested" else 0.0, dtype=float)
    out_parts = []

    for _, row in m.iterrows():
        mu_eq = float(row["mu_eq"])
        mu_bd = float(row["mu_bd"])
        cov00 = float(row["cov00"])
        cov01 = float(row["cov01"])
        cov11 = float(row["cov11"])
        ret_term = W_eq * mu_eq + W_bd * mu_bd
        risk_term = cov00 * W_eq ** 2 + 2.0 * cov01 * W_eq * W_bd + cov11 * W_bd ** 2
        turnover = np.abs(W_eq[None, :] - prev_w_eq[:, None]) + np.abs(W_bd[None, :] - prev_w_bd[:, None])
        obj = ret_term[None, :] - gamma[:, None] * risk_term[None, :] - tau[:, None] * tcost_one_way[:, None] * turnover
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
        part["feature_design"] = feature_design
        part["date"] = row["date"]
        part["model"] = model_name
        part["allocation_pair"] = pair_id
        part["p_eq"] = float(row["p_eq"])
        part["p_bd"] = float(row["p_bd"])
        part["mu_eq"] = mu_eq
        part["mu_bd"] = mu_bd
        part["sd_eq"] = float(row["sd_eq"])
        part["sd_bd"] = float(row["sd_bd"])
        part["rho"] = float(row["rho"])
        part["w_eq"] = w_eq
        part["w_bd"] = w_bd
        part["w_cash"] = w_cash
        part["turnover"] = selected_turnover
        part["tcost"] = selected_tcost
        part["equity_ret_h1"] = float(row["equity_ret_h1"])
        part["bond_ret_h1"] = float(row["bond_ret_h1"])
        part["strategy_gross_ret"] = gross_ret
        part["strategy_net_ret"] = net_ret
        part["objective"] = selected_obj
        part["shrink_k"] = float(MOMENT_SHRINKAGE_K)
        part["eq_n_state0"] = int(row["eq_n_state0"])
        part["eq_n_state1"] = int(row["eq_n_state1"])
        part["bd_n_state0"] = int(row["bd_n_state0"])
        part["bd_n_state1"] = int(row["bd_n_state1"])
        part["eq_weight_state0"] = float(row["eq_weight_state0"])
        part["eq_weight_state1"] = float(row["eq_weight_state1"])
        part["bd_weight_state0"] = float(row["bd_weight_state0"])
        part["bd_weight_state1"] = float(row["bd_weight_state1"])
        part["rho_n_obs"] = int(row["rho_n_obs"])
        part["rho_weight"] = float(row["rho_weight"])
        part["rho_empirical"] = float(row["rho_empirical"])
        part["strategy"] = (
            model_name
            + "__" + feature_design
            + "__" + pair_id
            + "__" + part["budget_mode"].astype(str)
            + "__g" + part["gamma"].map(safe_colname)
            + "__tau" + part["tau"].map(safe_colname)
            + "__c" + part["tcost_one_way"].map(safe_colname)
            + "__phl" + part["prob_smooth_halflife"].map(safe_colname)
        )
        out_parts.append(part)
        prev_w_eq = w_eq
        prev_w_bd = w_bd

    return pd.concat(out_parts, axis=0, ignore_index=True, sort=False) if out_parts else pd.DataFrame()


def build_6040_benchmark(date_index):
    bench = returns[["date", "equity_ret_h1", "bond_ret_h1"]].copy().dropna()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench[bench["date"].isin(date_index)].copy().sort_values("date").reset_index(drop=True)
    bench["ret_60_40_gross"] = 0.60 * bench["equity_ret_h1"] + 0.40 * bench["bond_ret_h1"]
    turnover = np.zeros(len(bench), dtype=float)
    if len(bench) > 0:
        turnover[0] = 1.0
    for i in range(1, len(bench)):
        prev_eq_ret = float(bench.loc[i - 1, "equity_ret_h1"])
        prev_bd_ret = float(bench.loc[i - 1, "bond_ret_h1"])
        growth = 0.60 * (1.0 + prev_eq_ret) + 0.40 * (1.0 + prev_bd_ret)
        if np.isfinite(growth) and abs(growth) > EPS:
            drift_eq = 0.60 * (1.0 + prev_eq_ret) / growth
            drift_bd = 0.40 * (1.0 + prev_bd_ret) / growth
            turnover[i] = abs(0.60 - drift_eq) + abs(0.40 - drift_bd)
    bench["turnover_60_40"] = turnover
    bench["tcost_60_40"] = TRANSACTION_COST_ONE_WAY * bench["turnover_60_40"]
    bench["ret_60_40_net"] = bench["ret_60_40_gross"] - bench["tcost_60_40"]
    bench["wealth_60_40"] = STARTING_WEALTH * (1.0 + bench["ret_60_40_net"]).cumprod()
    return bench


def strategy_summary(ret, turnover, tcost_paid, w_eq, w_bd, w_cash, strategy, model_name, allocation_pair, feature_design, budget_mode, gamma, tau, tcost_one_way, prob_halflife):
    r = pd.Series(ret).dropna().astype(float)
    turnover = pd.Series(turnover).reindex(r.index).fillna(0.0).astype(float)
    tcost_paid = pd.Series(tcost_paid).reindex(r.index).fillna(0.0).astype(float)
    w_eq = pd.Series(w_eq).reindex(r.index).astype(float)
    w_bd = pd.Series(w_bd).reindex(r.index).astype(float)
    w_cash = pd.Series(w_cash).reindex(r.index).astype(float)
    wealth = STARTING_WEALTH * (1.0 + r).cumprod() if len(r) else pd.Series(dtype=float)
    return {
        "feature_design": feature_design,
        "strategy": strategy,
        "model": model_name,
        "allocation_pair": allocation_pair,
        "budget_mode": budget_mode,
        "n_months": int(len(r)),
        "first_date": r.index.min() if isinstance(r.index, pd.DatetimeIndex) and len(r) else pd.NaT,
        "last_date": r.index.max() if isinstance(r.index, pd.DatetimeIndex) and len(r) else pd.NaT,
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": ann_sharpe(r),
        "max_drawdown": max_drawdown_from_returns(r),
        "cvar_1pct_monthly": cvar_left_tail(r),
        "final_wealth": float(wealth.iloc[-1]) if len(wealth) else np.nan,
        "avg_monthly_turnover": float(turnover.mean()) if len(turnover) else np.nan,
        "ann_turnover": float(12.0 * turnover.mean()) if len(turnover) else np.nan,
        "avg_tcost_paid_monthly": float(tcost_paid.mean()) if len(tcost_paid) else np.nan,
        "ann_tcost_drag": float(12.0 * tcost_paid.mean()) if len(tcost_paid) else np.nan,
        "avg_w_eq": float(w_eq.mean()) if len(w_eq) else np.nan,
        "avg_w_bd": float(w_bd.mean()) if len(w_bd) else np.nan,
        "avg_w_cash": float(w_cash.mean()) if len(w_cash) else np.nan,
        "gamma": float(gamma) if np.isfinite(gamma) else np.nan,
        "tau": float(tau) if np.isfinite(tau) else np.nan,
        "tcost_one_way": float(tcost_one_way) if np.isfinite(tcost_one_way) else np.nan,
        "prob_smooth_halflife": float(prob_halflife) if np.isfinite(prob_halflife) else np.nan,
        "shrink_k": float(MOMENT_SHRINKAGE_K),
    }


def summarize_allocation(results, benchmark, feature_design):
    rows = []
    if len(benchmark):
        b = benchmark.copy().set_index("date")
        rows.append(strategy_summary(
            ret=b["ret_60_40_net"],
            turnover=b["turnover_60_40"],
            tcost_paid=b["tcost_60_40"],
            w_eq=pd.Series(0.60, index=b.index),
            w_bd=pd.Series(0.40, index=b.index),
            w_cash=pd.Series(0.00, index=b.index),
            strategy="monthly_rebalanced_60_40",
            model_name="60/40",
            allocation_pair="benchmark",
            feature_design=feature_design,
            budget_mode="fully_invested",
            gamma=np.nan,
            tau=np.nan,
            tcost_one_way=np.nan,
            prob_halflife=np.nan,
        ))
    for strategy, g in results.groupby("strategy"):
        h = g.copy().sort_values("date").set_index("date")
        rows.append(strategy_summary(
            ret=h["strategy_net_ret"],
            turnover=h["turnover"],
            tcost_paid=h["tcost"],
            w_eq=h["w_eq"],
            w_bd=h["w_bd"],
            w_cash=h["w_cash"],
            strategy=strategy,
            model_name=str(h["model"].iloc[0]),
            allocation_pair=str(h["allocation_pair"].iloc[0]),
            feature_design=str(h["feature_design"].iloc[0]),
            budget_mode=str(h["budget_mode"].iloc[0]),
            gamma=float(h["gamma"].iloc[0]),
            tau=float(h["tau"].iloc[0]),
            tcost_one_way=float(h["tcost_one_way"].iloc[0]),
            prob_halflife=float(h["prob_smooth_halflife"].iloc[0]),
        ))
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def allocation_shrinkage_summary(results):
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["feature_design", "model", "allocation_pair", "budget_mode", "gamma", "tau", "tcost_one_way", "prob_smooth_halflife"], as_index=False)
        .agg(
            n_months=("date", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            shrink_k=("shrink_k", "first"),
            min_eq_n_state0=("eq_n_state0", "min"),
            min_eq_n_state1=("eq_n_state1", "min"),
            min_bd_n_state0=("bd_n_state0", "min"),
            min_bd_n_state1=("bd_n_state1", "min"),
            avg_eq_weight_state0=("eq_weight_state0", "mean"),
            avg_eq_weight_state1=("eq_weight_state1", "mean"),
            avg_bd_weight_state0=("bd_weight_state0", "mean"),
            avg_bd_weight_state1=("bd_weight_state1", "mean"),
            avg_rho_weight=("rho_weight", "mean"),
            avg_monthly_turnover=("turnover", "mean"),
            ann_turnover=("turnover", lambda x: 12.0 * float(pd.Series(x).mean())),
        )
        .sort_values(["feature_design", "allocation_pair", "budget_mode", "model", "gamma", "tau", "tcost_one_way", "prob_smooth_halflife"])
        .reset_index(drop=True)
    )


def select_by_target_vol(summary):
    x = summary.loc[summary["model"].isin(STUDENT_MODELS)].copy()
    rows = []
    for keys, g in x.groupby(["feature_design", "model", "allocation_pair", "budget_mode"], dropna=False):
        for target_vol in TARGET_VOLATILITY_GRID:
            h = g.copy()
            h["target_vol"] = float(target_vol)
            h["vol_gap"] = (h["ann_vol"] - float(target_vol)).abs()
            h = h.sort_values(["vol_gap", "sharpe", "ann_return"], ascending=[True, False, False])
            if len(h):
                rows.append(h.iloc[0].to_dict())
    return pd.DataFrame(rows).sort_values(["feature_design", "model", "allocation_pair", "budget_mode", "target_vol"]).reset_index(drop=True) if rows else pd.DataFrame()


def gamma_vol_map(summary):
    x = summary.loc[summary["model"].isin(STUDENT_MODELS)].copy()
    if x.empty:
        return pd.DataFrame()
    return (
        x.groupby(["feature_design", "model", "allocation_pair", "budget_mode", "gamma"], dropna=False)
        .agg(
            median_ann_vol=("ann_vol", "median"),
            min_ann_vol=("ann_vol", "min"),
            max_ann_vol=("ann_vol", "max"),
            median_avg_w_eq=("avg_w_eq", "median"),
            median_ann_turnover=("ann_turnover", "median"),
            best_sharpe=("sharpe", "max"),
        )
        .reset_index()
        .sort_values(["feature_design", "allocation_pair", "budget_mode", "model", "gamma"])
    )


def turnover_cost_summary(summary):
    x = summary.loc[summary["model"].isin(STUDENT_MODELS)].copy()
    if x.empty:
        return pd.DataFrame()
    return (
        x.groupby(["feature_design", "model", "allocation_pair", "budget_mode", "tau", "tcost_one_way", "prob_smooth_halflife"], dropna=False)
        .agg(
            n_strategies=("strategy", "count"),
            median_ann_turnover=("ann_turnover", "median"),
            median_tcost_drag=("ann_tcost_drag", "median"),
            median_ann_vol=("ann_vol", "median"),
            median_avg_w_eq=("avg_w_eq", "median"),
            best_sharpe=("sharpe", "max"),
        )
        .reset_index()
        .sort_values(["feature_design", "allocation_pair", "budget_mode", "model", "tau", "tcost_one_way", "prob_smooth_halflife"])
    )


def run_allocation_for_model(feature_design, model_name, pair_panel):
    panel = pair_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(returns[["date", "equity_ret_h1", "bond_ret_h1"]], on="date", how="left")
    first_prediction_date = panel["date"].dropna().min()
    initial_moments = initial_moments_from_pre_oos(historical_label_return_panel, first_prediction_date)
    moment_lookup = precompute_allocation_moments(panel, historical_label_return_panel, initial_moments)
    result_parts = []
    for prob_halflife in PROBABILITY_SMOOTHING_HALFLIFE_GRID:
        monthly_inputs = build_monthly_allocation_inputs(panel, moment_lookup, prob_halflife)
        for budget_mode in ALLOCATION_BUDGET_MODES:
            result_parts.append(simulate_allocation_param_block(monthly_inputs, feature_design, model_name, budget_mode, prob_halflife))
    results = pd.concat([x for x in result_parts if len(x)], axis=0, ignore_index=True, sort=False) if result_parts else pd.DataFrame()
    common_dates = sorted(set(results["date"].dropna().tolist())) if len(results) else []
    benchmark = build_6040_benchmark(common_dates)
    if len(results) and len(benchmark):
        results = results[results["date"].isin(benchmark["date"])].copy()
        results["wealth"] = results.groupby("strategy")["strategy_net_ret"].transform(lambda r: STARTING_WEALTH * (1.0 + r).cumprod())
    summary = summarize_allocation(results, benchmark, feature_design)
    shrinkage = allocation_shrinkage_summary(results)
    return results, summary, shrinkage, benchmark, initial_moments


student_result_parts = []
student_selected_model_parts = []
student_prediction_parts = []
student_importance_parts = []
student_seed_parts = []

for feature_design in STUDENT_FEATURE_DESIGNS:
    prediction_base = prediction_base_by_feature_design[feature_design]
    feature_cols = feature_cols_by_feature_design[feature_design]
    for model_name in STUDENT_MODELS:
        result_table_part, selected_model_part, prediction_part, importance_part, seed_part = run_student_model(
            feature_design=feature_design,
            model_name=model_name,
            df=prediction_base,
            feature_cols=feature_cols,
            target_cols=target_cols_h1,
            meta=target_meta_h1,
        )
        student_result_parts.append(result_table_part)
        student_selected_model_parts.append(selected_model_part)
        student_prediction_parts.append(prediction_part)
        student_importance_parts.append(importance_part)
        student_seed_parts.append(seed_part)

student_result_table = pd.concat(student_result_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "model", "target", "test_date", "model_name"]).reset_index(drop=True)
student_selected_model_table = pd.concat(student_selected_model_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "model", "target", "test_date"]).reset_index(drop=True)
student_oos_prediction_panel_long = pd.concat(student_prediction_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "model", "target", "date"]).reset_index(drop=True)
student_feature_importance_long = pd.concat(student_importance_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "model", "target", "test_date", "seed", "feature"]).reset_index(drop=True)
student_seed_diagnostics = pd.concat(student_seed_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "model", "target", "test_date", "model_name", "seed", "split"]).reset_index(drop=True)
student_forecast_summary = summarize_student_predictions(student_oos_prediction_panel_long, prediction_base_by_feature_design, target_meta_h1)
student_feature_importance_summary, student_feature_block_importance_summary, student_logit_coefficients_long = summarize_student_feature_importance(student_feature_importance_long)
student_feature_importance_top = (
    student_feature_importance_summary
    .sort_values(["feature_design", "model", "target", "mean_abs_importance"], ascending=[True, True, True, False])
    .groupby(["feature_design", "model", "target"], as_index=False)
    .head(30)
    .reset_index(drop=True)
    if len(student_feature_importance_summary) else pd.DataFrame()
)

student_prediction_panels_by_model_pair = {}
allocation_result_parts = []
allocation_summary_parts = []
allocation_shrinkage_parts = []
allocation_benchmarks = {}
allocation_initial_moments = {}

for feature_design in STUDENT_FEATURE_DESIGNS:
    for model_name in STUDENT_MODELS:
        pair_panel = build_pair_prediction_panel(student_oos_prediction_panel_long, feature_design, model_name)
        student_prediction_panels_by_model_pair[(feature_design, model_name, ALLOCATION_PAIR["pair"])] = pair_panel.copy()
        results, summary, shrinkage, benchmark, initial_moments = run_allocation_for_model(feature_design, model_name, pair_panel)
        allocation_result_parts.append(results)
        allocation_summary_parts.append(summary)
        allocation_shrinkage_parts.append(shrinkage)
        allocation_benchmarks[(feature_design, model_name, ALLOCATION_PAIR["pair"])] = benchmark
        allocation_initial_moments[(feature_design, model_name, ALLOCATION_PAIR["pair"])] = initial_moments

allocation_results_long = pd.concat(allocation_result_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "model", "allocation_pair", "budget_mode", "gamma", "tau", "tcost_one_way", "prob_smooth_halflife", "date"]).reset_index(drop=True)
allocation_strategy_summary = pd.concat(allocation_summary_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "allocation_pair", "budget_mode", "sharpe"], ascending=[True, True, True, False]).reset_index(drop=True)
allocation_moment_shrinkage_summary = pd.concat(allocation_shrinkage_parts, axis=0, ignore_index=True, sort=False).sort_values(["feature_design", "allocation_pair", "budget_mode", "model", "gamma", "tau", "tcost_one_way", "prob_smooth_halflife"]).reset_index(drop=True)
allocation_target_vol_selection = select_by_target_vol(allocation_strategy_summary)
allocation_gamma_vol_map = gamma_vol_map(allocation_strategy_summary)
allocation_turnover_cost_summary = turnover_cost_summary(allocation_strategy_summary)


def contiguous_spans(mask_series):
    mask_series = mask_series.fillna(False).astype(bool)
    spans = []
    start = None
    prev_date = None
    for dt, val in zip(mask_series.index, mask_series.values):
        if val and start is None:
            start = dt
        if (not val) and start is not None:
            spans.append((start, prev_date))
            start = None
        prev_date = dt
    if start is not None:
        spans.append((start, prev_date))
    return spans


def add_binary_label_shading(ax, data):
    g = data.copy().sort_values("date")
    g["date"] = pd.to_datetime(g["date"])
    for state, color, label in [(1.0, "#b7e4bd", "favourable"), (0.0, "#f5b5b5", "unfavourable")]:
        spans = contiguous_spans(pd.Series(g["y"].eq(state).values, index=g["date"]))
        for start, end in spans:
            ax.axvspan(start, end + pd.offsets.MonthEnd(1), color=color, alpha=0.35, lw=0, zorder=0)
    return [Patch(facecolor="#b7e4bd", edgecolor="none", alpha=0.35, label="favourable"), Patch(facecolor="#f5b5b5", edgecolor="none", alpha=0.35, label="unfavourable")]


def best_probability_path(target_col):
    rows = student_forecast_summary.loc[student_forecast_summary["target"].eq(target_col)].copy()
    rows = rows.sort_values(["oos_auc", "oos_balanced_accuracy", "oos_brier"], ascending=[False, False, True])
    if rows.empty:
        return pd.DataFrame(), {}
    row = rows.iloc[0].to_dict()
    g = student_oos_prediction_panel_long.loc[
        student_oos_prediction_panel_long["target"].eq(target_col)
        & student_oos_prediction_panel_long["feature_design"].eq(row["feature_design"])
        & student_oos_prediction_panel_long["model"].eq(row["model"])
    ].copy()
    return g.sort_values("date").reset_index(drop=True), row


def plot_probability_path(target_col, filename, title):
    data, selected = best_probability_path(target_col)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    shade_handles = add_binary_label_shading(ax, data)
    ax.plot(data["date"], data["p_hat"], color="black", lw=1.8, label="Predicted favourable-state probability", zorder=3)
    ax.axhline(MODEL_EVAL_THRESHOLD, color="black", linestyle="--", lw=1.0, alpha=0.75)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{title}: {selected.get('model', '')}, {selected.get('feature_design', '')}", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + shade_handles, labels + [h.get_label() for h in shade_handles], loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_feature_block_importance():
    x = student_feature_block_importance_summary.copy()
    x = x.loc[x["feature_design"].eq("standardized_and_smoothed_predictors")].copy()
    x = x.sort_values(["target", "model", "block_share_of_importance"], ascending=[True, True, False])
    if x.empty:
        return None
    top = x.groupby(["target", "model"], as_index=False).head(10).reset_index(drop=True)
    top["panel"] = top["target"].str.replace("_h1", "") + " | " + top["model"]
    panels = list(top["panel"].drop_duplicates())
    fig, axes = plt.subplots(len(panels), 1, figsize=(12.5, max(4.0, 2.8 * len(panels))))
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        h = top.loc[top["panel"].eq(panel)].sort_values("block_share_of_importance", ascending=True)
        ax.barh(h["feature_block"], h["block_share_of_importance"])
        ax.set_title(panel, loc="left")
        ax.set_xlabel("Share of mean absolute importance")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_block_importance.jpg", dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def state_background(date_index):
    bg = historical_label_return_panel[["date", "equity_favourable_h1", "bond_favourable_h1"]].copy()
    bg["date"] = pd.to_datetime(bg["date"])
    bg = bg[bg["date"].isin(date_index)].sort_values("date").reset_index(drop=True)
    bg["state_background"] = "missing"
    bg.loc[bg["equity_favourable_h1"].eq(1.0) & bg["bond_favourable_h1"].eq(1.0), "state_background"] = "both_favourable"
    bg.loc[bg["equity_favourable_h1"].eq(1.0) & bg["bond_favourable_h1"].eq(0.0), "state_background"] = "equity_favourable_only"
    bg.loc[bg["equity_favourable_h1"].eq(0.0) & bg["bond_favourable_h1"].eq(1.0), "state_background"] = "bond_favourable_only"
    bg.loc[bg["equity_favourable_h1"].eq(0.0) & bg["bond_favourable_h1"].eq(0.0), "state_background"] = "both_unfavourable"
    return bg


def add_stock_bond_state_spans(ax, bg):
    specs = [
        ("both_favourable", "equity favourable / bond favourable"),
        ("equity_favourable_only", "equity favourable / bond unfavourable"),
        ("bond_favourable_only", "equity unfavourable / bond favourable"),
        ("both_unfavourable", "equity unfavourable / bond unfavourable"),
    ]
    handles = []
    for state, label in specs:
        spans = contiguous_spans(pd.Series(bg["state_background"].eq(state).values, index=bg["date"]))
        for start, end in spans:
            ax.axvspan(start, end + pd.offsets.MonthEnd(1), color=STATE_COLORS[state], alpha=0.13, lw=0, zorder=0)
        handles.append(Patch(facecolor=STATE_COLORS[state], edgecolor="none", alpha=0.22, label=label))
    return handles


def plot_selected_allocation(row, filename):
    strategy = row["strategy"]
    feature_design = row["feature_design"]
    model_name = row["model"]
    g = allocation_results_long.loc[allocation_results_long["strategy"].eq(strategy)].copy().sort_values("date")
    benchmark = allocation_benchmarks[(feature_design, model_name, ALLOCATION_PAIR["pair"])]
    if g.empty or benchmark.empty:
        return None
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    ax.plot(benchmark["date"], benchmark["wealth_60_40"], color=LINE_COLORS["60/40"], linestyle="--", lw=2.2, label="60/40", zorder=3)
    ax.plot(g["date"], g["wealth"], color=LINE_COLORS.get(model_name, None), lw=2.1, label=model_name.replace("_", " "), zorder=4)
    bg = state_background(benchmark["date"])
    state_handles = add_stock_bond_state_spans(ax, bg)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + state_handles, labels + [h.get_label() for h in state_handles], loc="upper left", ncol=2, fontsize=8, frameon=True)
    ax.set_title(f"Regime-aware allocation: {model_name.replace('_', ' ')}, {feature_design}", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


plot_probability_path("equity_favourable_h1", "equity_prediction_probabilities.jpg", "Equity favourable-regime probability")
plot_probability_path("bond_favourable_h1", "bond_prediction_probabilities.jpg", "Bond favourable-regime probability")
plot_feature_block_importance()

selected_allocation_plots = allocation_target_vol_selection.loc[
    allocation_target_vol_selection["budget_mode"].eq("fully_invested")
    & allocation_target_vol_selection["target_vol"].isin(SELECTED_ALLOCATION_TARGET_VOLS)
].copy()
selected_allocation_plots = selected_allocation_plots.sort_values(["feature_design", "model", "target_vol"]).head(12)
for _, row in selected_allocation_plots.iterrows():
    file_name = (
        "allocation_"
        + safe_colname(row["feature_design"])
        + "_"
        + safe_colname(row["model"])
        + "_target_vol_"
        + safe_colname(row["target_vol"])
        + ".jpg"
    )
    plot_selected_allocation(row, file_name)


def save_frame(df, name):
    df.to_pickle(STUDENT_OUTPUT_DIR / f"{name}.pkl")
    df.to_csv(STUDENT_OUTPUT_DIR / f"{name}.csv", index=False)


save_frame(student_result_table, "student_model_grid_results")
save_frame(student_selected_model_table, "student_selected_models")
save_frame(student_oos_prediction_panel_long, "student_oos_prediction_panel")
save_frame(student_forecast_summary, "student_forecast_summary")
save_frame(student_feature_importance_long, "student_feature_importance_long")
save_frame(student_feature_importance_summary, "student_feature_importance_summary")
save_frame(student_feature_block_importance_summary, "student_feature_block_importance_summary")
save_frame(student_logit_coefficients_long, "student_logit_coefficients_long")
save_frame(student_seed_diagnostics, "student_seed_diagnostics")
save_frame(allocation_results_long, "allocation_results_long")
save_frame(allocation_strategy_summary, "allocation_strategy_summary")
save_frame(allocation_target_vol_selection, "allocation_target_vol_selection")
save_frame(allocation_gamma_vol_map, "allocation_gamma_vol_map")
save_frame(allocation_turnover_cost_summary, "allocation_turnover_cost_summary")
save_frame(allocation_moment_shrinkage_summary, "allocation_moment_shrinkage_summary")
save_frame(prediction_targets_h1, "prediction_targets_h1")
save_frame(historical_label_return_panel, "historical_label_return_panel")

student_allocation_results = {
    "student_result_table": student_result_table,
    "student_selected_model_table": student_selected_model_table,
    "student_oos_prediction_panel_long": student_oos_prediction_panel_long,
    "student_forecast_summary": student_forecast_summary,
    "student_feature_importance_long": student_feature_importance_long,
    "student_feature_importance_summary": student_feature_importance_summary,
    "student_feature_block_importance_summary": student_feature_block_importance_summary,
    "student_logit_coefficients_long": student_logit_coefficients_long,
    "student_seed_diagnostics": student_seed_diagnostics,
    "allocation_results_long": allocation_results_long,
    "allocation_strategy_summary": allocation_strategy_summary,
    "allocation_target_vol_selection": allocation_target_vol_selection,
    "allocation_gamma_vol_map": allocation_gamma_vol_map,
    "allocation_turnover_cost_summary": allocation_turnover_cost_summary,
    "allocation_moment_shrinkage_summary": allocation_moment_shrinkage_summary,
    "prediction_base_by_feature_design": prediction_base_by_feature_design,
    "feature_cols_by_feature_design": feature_cols_by_feature_design,
    "prediction_targets_h1": prediction_targets_h1,
    "historical_label_return_panel": historical_label_return_panel,
}
pd.to_pickle(student_allocation_results, STUDENT_OUTPUT_DIR / "student_allocation_results.pkl")

forecast_display_cols = [
    "feature_design", "model", "target", "asset", "n_oos_months", "first_oos", "last_oos",
    "positive_share", "prob_mean", "prob_std", "oos_auc", "oos_balanced_accuracy",
    "oos_brier", "oos_log_loss", "oos_net_sharpe", "oos_buyhold_sharpe", "oos_turnover",
]
allocation_display_cols = [
    "feature_design", "model", "allocation_pair", "budget_mode", "target_vol", "vol_gap",
    "ann_return", "ann_vol", "sharpe", "max_drawdown", "cvar_1pct_monthly", "final_wealth",
    "ann_turnover", "avg_w_eq", "avg_w_bd", "avg_w_cash", "tcost_one_way", "gamma", "tau",
    "prob_smooth_halflife", "strategy",
]
best_strategy_cols = [
    "feature_design", "strategy", "model", "allocation_pair", "budget_mode", "n_months", "ann_return",
    "ann_vol", "sharpe", "max_drawdown", "cvar_1pct_monthly", "final_wealth", "ann_turnover",
    "avg_w_eq", "avg_w_bd", "avg_w_cash", "tcost_one_way", "gamma", "tau", "prob_smooth_halflife",
]

display(student_forecast_summary[[c for c in forecast_display_cols if c in student_forecast_summary.columns]])
display(allocation_target_vol_selection[[c for c in allocation_display_cols if c in allocation_target_vol_selection.columns]])
display(
    allocation_strategy_summary.loc[allocation_strategy_summary["model"].isin(STUDENT_MODELS)]
    .sort_values(["feature_design", "allocation_pair", "budget_mode", "sharpe"], ascending=[True, True, True, False])
    .groupby(["feature_design", "allocation_pair", "budget_mode"], as_index=False)
    .head(1)
    .reset_index(drop=True)[[c for c in best_strategy_cols if c in allocation_strategy_summary.columns]]
)
