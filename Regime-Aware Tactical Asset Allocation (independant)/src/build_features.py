import time
import math
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path
import matplotlib.pyplot as plt


# Teacher and student feature construction

PROJECT_DIR = Path.cwd()
if PROJECT_DIR.name == "notebooks":
    PROJECT_DIR = PROJECT_DIR.parent

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TEACHER_FEATURE_OUTPUT_DIR = OUTPUT_DIR / "teacher_features"
TEACHER_LABEL_OUTPUT_DIR = OUTPUT_DIR / "teacher_labels"
MAJORITY_VOTE_OUTPUT_DIR = OUTPUT_DIR / "majority_vote"
STUDENT_PREDICTION_OUTPUT_DIR = OUTPUT_DIR / "student_prediction"
ALLOCATION_OUTPUT_DIR = OUTPUT_DIR / "allocation"
FIGURE_DIR = PROJECT_DIR / "paper" / "figs"

for directory in [
    PROCESSED_DATA_DIR,
    TEACHER_FEATURE_OUTPUT_DIR,
    TEACHER_LABEL_OUTPUT_DIR,
    MAJORITY_VOTE_OUTPUT_DIR,
    STUDENT_PREDICTION_OUTPUT_DIR,
    ALLOCATION_OUTPUT_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

BASE_DIR = PROJECT_DIR

MKT_PATH = RAW_DATA_DIR / "Raw Market Structure Panel.csv"
CMDTY_PATH = RAW_DATA_DIR / "Monthly Commodity Index Levels.xlsx"
MODEL_PANEL_PATH = RAW_DATA_DIR / "Monthly Model Panel from 1970-01.parquet"

OUT_WITH_CREDIT = PROCESSED_DATA_DIR / "Market Structure Panel with Credit.csv"
OUT_WO_CREDIT = PROCESSED_DATA_DIR / "Market Structure Panel without Credit.csv"

STUDENT_WITH_CREDIT_PATH = PROCESSED_DATA_DIR / "Student Feature Panel with Credit.csv"
STUDENT_WO_CREDIT_PATH = PROCESSED_DATA_DIR / "Student Feature Panel without Credit.csv"

DAILY_DATA_PATH = RAW_DATA_DIR / "Daily Bloomberg Total Return Indices.xlsx"

TEACHER_DAILY_FEATURES_PATH = PROCESSED_DATA_DIR / "Daily Teacher Features.csv"
TEACHER_MONTHLY_FEATURES_PATH = PROCESSED_DATA_DIR / "Monthly Teacher Features.csv"
TEACHER_FEATURE_INVENTORY_PATH = PROCESSED_DATA_DIR / "Teacher Feature Inventory.csv"
TEACHER_DATA_AVAILABILITY_PATH = PROCESSED_DATA_DIR / "Teacher Data Availability.csv"

TEACHER_EQUITY_DAILY_FEATURES_PKL = TEACHER_FEATURE_OUTPUT_DIR / "teacher_Equity Daily Return and Sortino Features.pkl"
TEACHER_BOND_MONTHLY_FEATURES_PKL = TEACHER_FEATURE_OUTPUT_DIR / "teacher_Bond Monthly Return Downside Sharpe and Volatility Features.pkl"
TEACHER_EQUITY_MONTHLY_EXCESS_RETURNS_PKL = TEACHER_FEATURE_OUTPUT_DIR / "teacher_equity_monthly_excess_returns.pkl"
TEACHER_DAILY_PANELS_PKL = TEACHER_FEATURE_OUTPUT_DIR / "teacher_daily_panels.pkl"
TEACHER_MONTHLY_PANELS_PKL = TEACHER_FEATURE_OUTPUT_DIR / "teacher_monthly_panels.pkl"
TEACHER_FEATURE_OBJECTS_MANIFEST_PATH = TEACHER_FEATURE_OUTPUT_DIR / "teacher_feature_objects_manifest.csv"


FRED_API_KEY = os.getenv("FRED_API_KEY", "")
BASE_OBS = "https://api.stlouisfed.org/fred/series/observations"

EPS = 1e-12

WINSOR_WINDOW = 120
WINSOR_MIN_PERIODS = 60
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99

EXPECTED_FINAL_FEATURE_COUNT = 134

Z_WINDOW = 60
Z_MIN_PERIODS = 36
Z_EPS = 1e-12
Z_SMOOTH_HALFLIFE = 12

RUN_GRID_PLOTS = True

PLOT_START_DATE = None
PLOT_END_DATE = None

N_COLS = 4
N_ROWS = 5
FIGSIZE = (24, 16)

USE_TWIN_AXIS = True

RAW_COLOR = "black"
Z_COLOR = "tab:blue"
SMOOTH_COLOR = "tab:red"

RAW_LINEWIDTH = 1.0
Z_LINEWIDTH = 0.9
SMOOTH_LINEWIDTH = 1.8

RAW_ALPHA = 0.90
Z_ALPHA = 0.75
SMOOTH_ALPHA = 0.95

SHOW_LEGEND = True

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 500)


TEACHER_FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Helper functions

def force_month_end(x):
    return pd.to_datetime(x, errors="coerce").dt.to_period("M").dt.to_timestamp("M")



def fred_obs(series_id, clean_name):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "1900-01-01",
    }

    r = requests.get(BASE_OBS, params=params, timeout=30)
    out = pd.DataFrame(r.json().get("observations", []))

    out = out[["date", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[clean_name] = pd.to_numeric(out["value"].replace(".", pd.NA), errors="coerce")

    return out[["date", clean_name]].dropna(subset=["date"])


def to_month_end_last_available(df, value_col):
    tmp = df.copy()
    tmp["month_end"] = force_month_end(tmp["date"])

    return (
        tmp.sort_values(["month_end", "date"])
           .groupby("month_end", as_index=False)
           .tail(1)
           [["month_end", value_col]]
           .sort_values("month_end")
           .reset_index(drop=True)
    )


def yahoo_download_robust(ticker, start="1970-01-01", max_tries=3):
    errors = []

    for i in range(max_tries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if raw is not None and not raw.empty:
                return raw

            errors.append(f"download start={start} returned empty, try={i + 1}")
            time.sleep(2)

        except Exception as e:
            errors.append(f"download start={start} failed, try={i + 1}: {repr(e)}")
            time.sleep(2)

    for i in range(max_tries):
        try:
            raw = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if raw is not None and not raw.empty:
                raw = raw.loc[pd.to_datetime(raw.index) >= pd.to_datetime(start)].copy()
                if not raw.empty:
                    return raw

            errors.append(f"download period=max returned empty after start filter, try={i + 1}")
            time.sleep(2)

        except Exception as e:
            errors.append(f"download period=max failed, try={i + 1}: {repr(e)}")
            time.sleep(2)

    try:
        raw = yf.Ticker(ticker).history(
            start=start,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

        if raw is not None and not raw.empty:
            return raw

        errors.append("Ticker.history returned empty")

    except Exception as e:
        errors.append(f"Ticker.history failed: {repr(e)}")

    return raw if "raw" in locals() else pd.DataFrame()


def yahoo_monthly_adjusted_return(ticker, ret_col, start="1970-01-01"):
    raw = yahoo_download_robust(ticker=ticker, start=start)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    d = raw[["Adj Close"]].copy()
    d = d.rename(columns={"Adj Close": "adj_close"})
    d = d.reset_index()

    date_col = "Date" if "Date" in d.columns else d.columns[0]
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["month_end"] = force_month_end(d[date_col])

    monthly_level = (
        d.dropna(subset=["month_end", "adj_close"])
         .sort_values(date_col)
         .groupby("month_end", as_index=False)
         .last()[["month_end", "adj_close"]]
         .sort_values("month_end")
         .reset_index(drop=True)
    )

    monthly_level[ret_col] = monthly_level["adj_close"].pct_change()

    return monthly_level[["month_end", ret_col]]


def _row_dispersion(df, cols):
    return df[cols].std(axis=1, ddof=0)


def _row_best_minus_worst(df, cols):
    return df[cols].max(axis=1) - df[cols].min(axis=1)


def _roll_sum(s, w):
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=w).sum()


def _roll_mean(s, w):
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=w).mean()


def _roll_std(s, w):
    return pd.to_numeric(s, errors="coerce").rolling(w, min_periods=w).std(ddof=0)


def _diff1(s):
    return pd.to_numeric(s, errors="coerce").diff()


def _log_diff1(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0)).diff()


def _safe_fisher_corr(rho):
    rho = pd.to_numeric(rho, errors="coerce").clip(-0.999, 0.999)
    return 0.5 * np.log((1.0 + rho) / (1.0 - rho))


def _rolling_corr_fisher_diff(df, x_col, y_col, window=12):
    rho = df[x_col].rolling(window, min_periods=window).corr(df[y_col])
    fisher = _safe_fisher_corr(rho)
    return fisher.diff()


def _rolling_pairwise_corr_stats(df, cols, window=12, prefix=None):
    avg_corr = pd.Series(np.nan, index=df.index)
    absavg_corr = pd.Series(np.nan, index=df.index)
    pca1_share = pd.Series(np.nan, index=df.index)
    rotation_speed = pd.Series(np.nan, index=df.index)

    prev_corr = None
    X = df[cols].copy()

    for i in range(len(df)):
        if i + 1 < window:
            continue

        block = X.iloc[i + 1 - window:i + 1].dropna()

        if len(block) < window:
            prev_corr = None
            continue

        corr = block.corr().to_numpy(dtype=float)

        if not np.isfinite(corr).all():
            prev_corr = None
            continue

        n = corr.shape[0]

        if n < 2:
            prev_corr = None
            continue

        upper = corr[np.triu_indices(n, k=1)]

        avg_corr.iloc[i] = np.nanmean(upper)
        absavg_corr.iloc[i] = np.nanmean(np.abs(upper))

        try:
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = np.maximum(eigvals, 0.0)
            denom = eigvals.sum()
            pca1_share.iloc[i] = eigvals[-1] / denom if denom > EPS else np.nan
        except Exception:
            pca1_share.iloc[i] = np.nan

        if prev_corr is not None and prev_corr.shape == corr.shape:
            rotation_speed.iloc[i] = np.linalg.norm(corr - prev_corr, ord="fro")

        prev_corr = corr.copy()

    return {
        f"{prefix}_corr_avg_12m": avg_corr,
        f"{prefix}_corr_absavg_12m": absavg_corr,
        f"{prefix}_pca1_share_12m": pca1_share,
        f"{prefix}_pca1_share_diff": pca1_share.diff(),
        f"{prefix}_rotation_speed_12m": rotation_speed,
    }


def _lambda_f_series(df, cols, window=12, prefix=None, include_corr=True):
    lam_cov = pd.Series(np.nan, index=df.index)
    lam_corr = pd.Series(np.nan, index=df.index)

    X = df[cols].copy()

    prev_cov = None
    prev_corr = None

    for i in range(len(df)):
        if i + 1 < window:
            continue

        block = X.iloc[i + 1 - window:i + 1].dropna()

        if len(block) < window:
            prev_cov = None
            prev_corr = None
            continue

        cov = block.cov().to_numpy(dtype=float)
        corr = block.corr().to_numpy(dtype=float)

        if not np.isfinite(cov).all():
            prev_cov = None
            prev_corr = None
            continue

        if include_corr and not np.isfinite(corr).all():
            prev_cov = None
            prev_corr = None
            continue

        if prev_cov is not None and prev_cov.shape == cov.shape:
            dF = cov - prev_cov
            comm = cov @ dF - dF @ cov
            denom = np.linalg.norm(cov, ord="fro") * np.linalg.norm(dF, ord="fro")

            if denom > EPS:
                lam_cov.iloc[i] = np.log1p(
                    100.0 * np.linalg.norm(comm, ord="fro") / denom
                )

        if include_corr:
            if prev_corr is not None and prev_corr.shape == corr.shape:
                dF = corr - prev_corr
                comm = corr @ dF - dF @ corr
                denom = np.linalg.norm(corr, ord="fro") * np.linalg.norm(dF, ord="fro")

                if denom > EPS:
                    lam_corr.iloc[i] = np.log1p(
                        100.0 * np.linalg.norm(comm, ord="fro") / denom
                    )

        prev_cov = cov.copy()
        prev_corr = corr.copy() if include_corr else None

    out = {
        f"{prefix}_lambda_cov_12m": lam_cov,
        f"{prefix}_lambda_cov_diff": lam_cov.diff(),
    }

    if include_corr:
        out[f"{prefix}_lambda_corr_12m"] = lam_corr
        out[f"{prefix}_lambda_corr_diff"] = lam_corr.diff()

    return out


def _require_cols(df, cols, block_name):
    return None


def rolling_winsorize_series(
    s,
    window=120,
    min_periods=60,
    q_low=0.01,
    q_high=0.99,
):
    x = pd.to_numeric(s, errors="coerce")

    lo = (
        x.shift(1)
         .rolling(window=window, min_periods=min_periods)
         .quantile(q_low)
    )

    hi = (
        x.shift(1)
         .rolling(window=window, min_periods=min_periods)
         .quantile(q_high)
    )

    out = x.copy()
    valid_band = lo.notna() & hi.notna() & x.notna()

    out.loc[valid_band] = np.minimum(
        np.maximum(x.loc[valid_band], lo.loc[valid_band]),
        hi.loc[valid_band],
    )

    return out


def causal_rolling_zscore(
    s,
    window=60,
    min_periods=36,
    eps=1e-12,
):
    x = pd.to_numeric(s, errors="coerce")

    mu = (
        x.shift(1)
         .rolling(window=window, min_periods=min_periods)
         .mean()
    )

    sig = (
        x.shift(1)
         .rolling(window=window, min_periods=min_periods)
         .std(ddof=0)
    )

    return (x - mu) / (sig + eps)


def add_z_and_smooth(panel, output_date_col="date"):
    out = panel.copy()

    out["month_end"] = pd.to_datetime(out["month_end"]) + pd.offsets.MonthEnd(0)

    out = (
        out.sort_values("month_end")
           .drop_duplicates("month_end", keep="last")
           .reset_index(drop=True)
    )

    raw_feature_cols = [c for c in out.columns if c != "month_end"]

    for c in raw_feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    z_cols = []

    for c in raw_feature_cols:
        z_col = f"{c}_z"
        out[z_col] = causal_rolling_zscore(
            out[c],
            window=Z_WINDOW,
            min_periods=Z_MIN_PERIODS,
            eps=Z_EPS,
        )
        z_cols.append(z_col)

    smooth_cols = []

    for z_col in z_cols:
        smooth_col = f"{z_col}_smooth"
        out[smooth_col] = (
            out[z_col]
            .ewm(
                halflife=Z_SMOOTH_HALFLIFE,
                adjust=False,
                min_periods=1,
            )
            .mean()
        )
        smooth_cols.append(smooth_col)

    final_cols = ["month_end"] + raw_feature_cols + z_cols + smooth_cols
    out = out[final_cols].copy()

    complete_cols = raw_feature_cols + z_cols + smooth_cols
    complete_mask = out[complete_cols].notna().all(axis=1)

    out = (
        out.loc[complete_mask]
           .rename(columns={"month_end": output_date_col})
           .reset_index(drop=True)
    )

    return out, raw_feature_cols, z_cols, smooth_cols



def plot_raw_z_smooth_grid(
    input_df,
    panel_name,
    plot_start_date=None,
    plot_end_date=None,
    n_cols=4,
    n_rows=5,
    figsize=(24, 16),
    use_twin_axis=True,
):
    plot_df = input_df.copy()

    plot_df["date"] = pd.to_datetime(plot_df["date"]) + pd.offsets.MonthEnd(0)

    plot_df = (
        plot_df.sort_values("date")
               .drop_duplicates("date", keep="last")
               .reset_index(drop=True)
    )

    if plot_start_date is not None:
        plot_df = plot_df.loc[plot_df["date"] >= pd.Timestamp(plot_start_date)].copy()

    if plot_end_date is not None:
        plot_df = plot_df.loc[plot_df["date"] <= pd.Timestamp(plot_end_date)].copy()

    all_cols = [c for c in plot_df.columns if c != "date"]

    raw_feature_cols = [
        c for c in all_cols
        if not c.endswith("_z")
        and not c.endswith("_z_smooth")
        and f"{c}_z" in plot_df.columns
        and f"{c}_z_smooth" in plot_df.columns
    ]

    plots_per_fig = n_cols * n_rows
    n_figs = math.ceil(len(raw_feature_cols) / plots_per_fig)


    for fig_id in range(n_figs):
        chunk = raw_feature_cols[
            fig_id * plots_per_fig : (fig_id + 1) * plots_per_fig
        ]

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=False,
        )

        axes = np.asarray(axes).reshape(-1)

        for ax, base_col in zip(axes, chunk):
            z_col = f"{base_col}_z"
            smooth_col = f"{base_col}_z_smooth"

            temp = plot_df[["date", base_col, z_col, smooth_col]].copy()

            temp[base_col] = pd.to_numeric(temp[base_col], errors="coerce")
            temp[z_col] = pd.to_numeric(temp[z_col], errors="coerce")
            temp[smooth_col] = pd.to_numeric(temp[smooth_col], errors="coerce")

            raw_temp = temp[["date", base_col]].dropna()
            z_temp = temp[["date", z_col]].dropna()
            smooth_temp = temp[["date", smooth_col]].dropna()

            if raw_temp.empty and z_temp.empty and smooth_temp.empty:
                ax.set_title(f"{base_col}\nno data", fontsize=8.5, fontweight="bold")
                ax.axis("off")
                continue

            if use_twin_axis:
                ax_z = ax.twinx()

                if not raw_temp.empty:
                    ax.plot(
                        raw_temp["date"],
                        raw_temp[base_col],
                        color=RAW_COLOR,
                        lw=RAW_LINEWIDTH,
                        alpha=RAW_ALPHA,
                        label="raw",
                    )

                if not z_temp.empty:
                    ax_z.plot(
                        z_temp["date"],
                        z_temp[z_col],
                        color=Z_COLOR,
                        lw=Z_LINEWIDTH,
                        alpha=Z_ALPHA,
                        label="z",
                    )

                if not smooth_temp.empty:
                    ax_z.plot(
                        smooth_temp["date"],
                        smooth_temp[smooth_col],
                        color=SMOOTH_COLOR,
                        lw=SMOOTH_LINEWIDTH,
                        alpha=SMOOTH_ALPHA,
                        label="z_smooth",
                    )

                ax.axhline(0.0, color="gray", lw=0.7, ls="--", alpha=0.45)
                ax_z.axhline(0.0, color="gray", lw=0.7, ls=":", alpha=0.35)

                ax.tick_params(axis="y", labelsize=7, colors=RAW_COLOR)
                ax_z.tick_params(axis="y", labelsize=7, colors=Z_COLOR)

                ax.set_ylabel("raw", fontsize=7, color=RAW_COLOR)
                ax_z.set_ylabel("z / smooth", fontsize=7, color=Z_COLOR)

                if SHOW_LEGEND:
                    lines_1, labels_1 = ax.get_legend_handles_labels()
                    lines_2, labels_2 = ax_z.get_legend_handles_labels()

                    ax.legend(
                        lines_1 + lines_2,
                        labels_1 + labels_2,
                        fontsize=7,
                        loc="best",
                        frameon=True,
                    )

            else:
                if not raw_temp.empty:
                    ax.plot(
                        raw_temp["date"],
                        raw_temp[base_col],
                        color=RAW_COLOR,
                        lw=RAW_LINEWIDTH,
                        alpha=RAW_ALPHA,
                        label="raw",
                    )

                if not z_temp.empty:
                    ax.plot(
                        z_temp["date"],
                        z_temp[z_col],
                        color=Z_COLOR,
                        lw=Z_LINEWIDTH,
                        alpha=Z_ALPHA,
                        label="z",
                    )

                if not smooth_temp.empty:
                    ax.plot(
                        smooth_temp["date"],
                        smooth_temp[smooth_col],
                        color=SMOOTH_COLOR,
                        lw=SMOOTH_LINEWIDTH,
                        alpha=SMOOTH_ALPHA,
                        label="z_smooth",
                    )

                ax.axhline(0.0, color="gray", lw=0.7, ls="--", alpha=0.45)

                if SHOW_LEGEND:
                    ax.legend(fontsize=7, loc="best", frameon=True)

            valid_any = temp[[base_col, z_col, smooth_col]].notna().any(axis=1)

            if valid_any.any():
                first_valid = temp.loc[valid_any, "date"].min().date()
                last_valid = temp.loc[valid_any, "date"].max().date()
                subtitle = f"{first_valid} to {last_valid}"
            else:
                subtitle = "no data"

            ax.set_title(
                f"{base_col}\n{subtitle}",
                fontsize=8.5,
                fontweight="bold",
            )

            ax.grid(True, alpha=0.25)

        for ax in axes[len(chunk):]:
            ax.axis("off")

        fig.suptitle(
            f"{panel_name} | raw, z-score, and smoothed z-score | Figure {fig_id + 1}/{n_figs}",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


# Teacher feature construction

TEACHER_EPS = 1e-12
TEACHER_STD_DDOF = 0

CURRENT_DATE = pd.Timestamp("2026-05-19")
FINALIZED_MONTH_END = (
    CURRENT_DATE.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
)

EQUITY_START_MODE = "bond_start"

EQUITY_RETURN_SORTINO_FEATURES = [
    "sortino_21d",
    "sortino_63d",
    "sortino_126d",
    "sortino_252d",
    "ewm_return_21d",
    "ewm_return_63d",
    "ewm_return_126d",
    "ewm_return_252d",
]


EQUITY_ALL_TEACHER_FEATURES = EQUITY_RETURN_SORTINO_FEATURES.copy()

BOND_MONTHLY_TEACHER_FEATURES = [
    "ewm_return_3m",
    "ewm_return_6m",
    "ewm_return_12m",
    "log_downside_1m",
    "log_downside_3m",
    "log_downside_6m",
    "log_downside_12m",
    "sharpe_3m",
    "sharpe_6m",
    "sharpe_12m",
    "realized_vol_6m",
]

TEACHER_NON_FEATURE_COLS = {
    "date",
    "asset",
    "teacher_panel",
    "feature_group",
    "source_frequency",
    "label_source",
    "feature_source",
    "excess_return",
    "excess_return_source",
    "n_daily_obs",
}

def safe_min_date(s):
    s = pd.to_datetime(s, errors="coerce").dropna()
    return pd.NaT if s.empty else s.min()


def safe_max_date(s):
    s = pd.to_datetime(s, errors="coerce").dropna()
    return pd.NaT if s.empty else s.max()


def date_or_nat(x):
    if pd.isna(x):
        return pd.NaT
    return pd.Timestamp(x).date()


def infer_dense_daily_start(dates, calendar_window_days=31, min_obs_in_window=10):
    dt = (
        pd.Series(pd.to_datetime(dates, errors="coerce"))
        .dropna()
        .sort_values()
        .reset_index(drop=True)
    )

    if dt.empty:
        return pd.NaT

    values = dt.to_numpy(dtype="datetime64[ns]")
    flags = []

    for current in values:
        left = np.searchsorted(
            values,
            current - np.timedelta64(calendar_window_days, "D"),
            side="left",
        )
        right = np.searchsorted(
            values,
            current + np.timedelta64(calendar_window_days, "D"),
            side="right",
        )
        flags.append((right - left) >= min_obs_in_window)

    flags = pd.Series(flags)

    if not flags.any():
        return pd.NaT

    return pd.Timestamp(values[int(flags.idxmax())])


def parse_bloomberg_index_sheet(path, sheet_name, asset, level_col=None):
    raw = pd.read_excel(path, sheet_name=sheet_name)

    date_col = raw.columns[0]
    level_col = raw.columns[1] if level_col is None else level_col

    out = raw[[date_col, level_col]].copy()
    out = out.rename(columns={date_col: "date", level_col: "index_level"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["index_level"] = pd.to_numeric(out["index_level"], errors="coerce")

    out = (
        out.dropna(subset=["date", "index_level"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    dense_start = infer_dense_daily_start(out["date"])

    out["asset"] = asset
    out["month_end"] = force_month_end(out["date"])
    out["calendar_gap_days"] = out["date"].diff().dt.days
    out["raw_return"] = out["index_level"].pct_change()
    out.loc[out["calendar_gap_days"].gt(7), "raw_return"] = np.nan

    if pd.notna(dense_start):
        out["is_dense_daily_region"] = out["date"].ge(dense_start)
    else:
        out["is_dense_daily_region"] = False

    return out, dense_start


def load_monthly_rf_for_teacher(model_panel_path):
    mp = pd.read_parquet(model_panel_path)

    rf = mp[["date", "rf"]].copy()
    rf["month_end"] = force_month_end(rf["date"])
    rf["rf"] = pd.to_numeric(rf["rf"], errors="coerce")

    return (
        rf[["month_end", "rf"]]
        .dropna(subset=["month_end"])
        .sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .reset_index(drop=True)
    )


def attach_daily_rf(asset_daily, rf_monthly):
    out = asset_daily.copy()
    out["month_end"] = force_month_end(out["date"])
    out = out.merge(rf_monthly, on="month_end", how="left")

    n_days = (
        out.groupby("month_end")["date"]
        .transform("count")
        .astype(float)
        .replace(0.0, np.nan)
    )

    out["rf_available"] = out["rf"].notna()

    out["rf_daily"] = np.where(
        out["rf"].notna(),
        np.power(1.0 + out["rf"].astype(float), 1.0 / n_days) - 1.0,
        0.0,
    )

    out["excess_return"] = out["raw_return"] - out["rf_daily"]

    out["excess_return_source"] = np.where(
        out["rf_available"],
        "monthly_rf_compounded_to_daily",
        "raw_return_rf_missing_zero",
    )

    return out


def build_monthly_return_panel_from_levels(asset_levels, rf_monthly):
    monthly = (
        asset_levels.dropna(subset=["month_end", "index_level"])
        .sort_values(["month_end", "date"])
        .groupby("month_end", as_index=False)
        .tail(1)
        [["month_end", "asset", "index_level"]]
        .sort_values("month_end")
        .reset_index(drop=True)
    )

    monthly["raw_return"] = monthly["index_level"].pct_change()

    monthly = monthly.merge(rf_monthly, on="month_end", how="left")
    monthly["rf_available"] = monthly["rf"].notna()
    monthly["rf_monthly"] = monthly["rf"].fillna(0.0)
    monthly["excess_return"] = monthly["raw_return"] - monthly["rf_monthly"]

    monthly["excess_return_source"] = np.where(
        monthly["rf_available"],
        "monthly_rf",
        "raw_return_rf_missing_zero",
    )

    monthly = monthly.rename(columns={"month_end": "date"})

    return monthly.sort_values("date").reset_index(drop=True)


def ewm_halflife(s, h, min_periods=None):
    if min_periods is None:
        min_periods = int(h)

    alpha = 1.0 - np.exp(np.log(0.5) / float(h))
    return (
        pd.to_numeric(s, errors="coerce")
        .ewm(alpha=alpha, adjust=False, min_periods=int(min_periods))
        .mean()
    )


def compound_return(ret):
    r = pd.Series(ret).dropna().astype(float)
    return float(np.prod(1.0 + r) - 1.0) if len(r) else np.nan


def build_equity_daily_teacher_features(daily_df):
    base = daily_df[["date", "asset", "excess_return", "excess_return_source"]].copy()
    base["asset"] = "equity"
    base = base.sort_values("date").reset_index(drop=True)

    r = pd.to_numeric(base["excess_return"], errors="coerce")
    neg = r.clip(upper=0.0)
    feat = pd.DataFrame(index=base.index)

    for h in [21, 63, 126, 252]:
        avg = ewm_halflife(r, h, min_periods=h)
        downside = np.sqrt(ewm_halflife(neg.pow(2), h, min_periods=h))
        feat[f"sortino_{h}d"] = avg / (downside + TEACHER_EPS)
        feat[f"ewm_return_{h}d"] = avg

    out = pd.concat(
        [
            base.reset_index(drop=True),
            feat[EQUITY_RETURN_SORTINO_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )

    out["teacher_panel"] = "equity_daily_return_sortino"
    out["feature_group"] = "return_sortino"
    out["source_frequency"] = "daily"
    out["label_source"] = "daily_sjm"
    out["feature_source"] = "daily"

    cols = [
        "date",
        "asset",
        "teacher_panel",
        "feature_group",
        "source_frequency",
        "label_source",
        "feature_source",
        "excess_return",
        "excess_return_source",
    ] + EQUITY_RETURN_SORTINO_FEATURES

    return out[cols].sort_values("date").reset_index(drop=True)

def build_bond_monthly_teacher_features(monthly_df):
    base = monthly_df[["date", "asset", "excess_return", "excess_return_source"]].copy()
    base["asset"] = "bond"
    base = base.sort_values("date").reset_index(drop=True)

    r = pd.to_numeric(base["excess_return"], errors="coerce")
    neg = r.clip(upper=0.0)

    feat = pd.DataFrame(index=base.index)

    for h in [1, 3, 6, 12]:
        avg_ewm = ewm_halflife(r, h)
        downside = np.sqrt(ewm_halflife(neg.pow(2), h))
        feat[f"log_downside_{h}m"] = np.log(downside + TEACHER_EPS)

        if h in [3, 6, 12]:
            feat[f"ewm_return_{h}m"] = avg_ewm

            mean_h = r.rolling(h, min_periods=h).mean()
            vol_h = r.rolling(h, min_periods=h).std(ddof=TEACHER_STD_DDOF)
            feat[f"sharpe_{h}m"] = mean_h / (vol_h + TEACHER_EPS)

    feat["realized_vol_6m"] = (
        r.rolling(6, min_periods=6).std(ddof=TEACHER_STD_DDOF)
    )

    out = pd.concat(
        [
            base.reset_index(drop=True),
            feat[BOND_MONTHLY_TEACHER_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )

    out["teacher_panel"] = "bond_monthly_return_downside_sharpe_vol"
    out["feature_group"] = "return_downside_sharpe_vol"
    out["source_frequency"] = "monthly"
    out["label_source"] = "monthly_sjm"
    out["feature_source"] = "monthly_only"
    out["n_daily_obs"] = np.nan

    cols = [
        "date",
        "asset",
        "teacher_panel",
        "feature_group",
        "source_frequency",
        "label_source",
        "feature_source",
        "excess_return",
        "excess_return_source",
        "n_daily_obs",
    ] + BOND_MONTHLY_TEACHER_FEATURES

    return out[cols].sort_values("date").reset_index(drop=True)


def build_equity_monthly_excess_returns_from_daily(equity_daily_features):
    out = (
        equity_daily_features.assign(month_end=force_month_end(equity_daily_features["date"]))
        .groupby("month_end", as_index=False)
        .agg(
            eq_excess=("excess_return", compound_return),
            n_daily_obs=("date", "count"),
        )
        .rename(columns={"month_end": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return out


def clean_teacher_panel(df, feature_cols):
    out = df.copy()

    for c in feature_cols + ["excess_return"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out



# Teacher features

rf_monthly_teacher = load_monthly_rf_for_teacher(MODEL_PANEL_PATH)

spx_levels, spx_dense_start = parse_bloomberg_index_sheet(
    DAILY_DATA_PATH,
    sheet_name="Sheet1",
    asset="equity",
)

agg_levels, agg_dense_start = parse_bloomberg_index_sheet(
    DAILY_DATA_PATH,
    sheet_name="LBUSTRUU",
    asset="bond",
)

TEACHER_PROJECT_START = pd.Timestamp(agg_levels["date"].min())

spx_levels_full_history = spx_levels.sort_values("date").reset_index(drop=True).copy()

if EQUITY_START_MODE == "bond_start":
    spx_levels = (
        spx_levels_full_history.loc[spx_levels_full_history["date"] >= TEACHER_PROJECT_START]
        .sort_values("date")
        .reset_index(drop=True)
    )

    spx_levels["month_end"] = force_month_end(spx_levels["date"])
    spx_levels["calendar_gap_days"] = spx_levels["date"].diff().dt.days
    spx_levels["raw_return"] = spx_levels["index_level"].pct_change()
    spx_levels.loc[spx_levels["calendar_gap_days"].gt(7), "raw_return"] = np.nan
    spx_levels["is_dense_daily_region"] = True

else:
    spx_levels = spx_levels_full_history.copy()

spx_daily_all_full_history = attach_daily_rf(spx_levels_full_history.copy(), rf_monthly_teacher)
spx_daily_full_history = (
    spx_daily_all_full_history.loc[spx_daily_all_full_history["is_dense_daily_region"]]
    .dropna(subset=["raw_return"])
    .sort_values("date")
    .reset_index(drop=True)
)

spx_daily_all = attach_daily_rf(spx_levels.copy(), rf_monthly_teacher)
agg_daily_all = attach_daily_rf(agg_levels.copy(), rf_monthly_teacher)

spx_daily = (
    spx_daily_all.loc[spx_daily_all["is_dense_daily_region"]]
    .dropna(subset=["raw_return"])
    .sort_values("date")
    .reset_index(drop=True)
)

agg_daily = (
    agg_daily_all.loc[agg_daily_all["is_dense_daily_region"]]
    .dropna(subset=["raw_return"])
    .sort_values("date")
    .reset_index(drop=True)
)

spx_monthly = build_monthly_return_panel_from_levels(spx_levels, rf_monthly_teacher)
agg_monthly = build_monthly_return_panel_from_levels(agg_levels, rf_monthly_teacher)

teacher_data_availability = pd.DataFrame([
    {
        "asset": name,
        "level_obs": int(len(levels)),
        "level_first_date": date_or_nat(levels["date"].min()),
        "level_last_date": date_or_nat(levels["date"].max()),
        "dense_daily_start": date_or_nat(dense_start),
        "daily_obs_used": int(len(daily)),
        "daily_first_date": date_or_nat(safe_min_date(daily["date"])),
        "daily_last_date": date_or_nat(safe_max_date(daily["date"])),
        "monthly_obs": int(len(monthly)),
        "monthly_first_date": date_or_nat(safe_min_date(monthly["date"])),
        "monthly_last_date": date_or_nat(safe_max_date(monthly["date"])),
        "rf_available_daily_pct": float(daily["rf_available"].mean()) if len(daily) else np.nan,
        "rf_available_monthly_pct": float(monthly["rf_available"].mean()) if len(monthly) else np.nan,
    }
    for name, levels, dense_start, daily, monthly in [
        ("equity", spx_levels, spx_dense_start, spx_daily, spx_monthly),
        ("bond", agg_levels, agg_dense_start, agg_daily, agg_monthly),
    ]
])

equity_teacher = build_equity_daily_teacher_features(spx_daily_full_history)

equity_teacher = (
    equity_teacher
    .loc[
        (equity_teacher["date"] >= TEACHER_PROJECT_START)
        & (equity_teacher["date"] <= FINALIZED_MONTH_END)
    ]
    .reset_index(drop=True)
)

equity_teacher = clean_teacher_panel(equity_teacher, EQUITY_RETURN_SORTINO_FEATURES)

equity_monthly_returns = build_equity_monthly_excess_returns_from_daily(equity_teacher)

bond_teacher = build_bond_monthly_teacher_features(agg_monthly)

bond_teacher = (
    bond_teacher
    .loc[bond_teacher["date"] <= FINALIZED_MONTH_END]
    .sort_values("date")
    .drop_duplicates("date", keep="last")
    .reset_index(drop=True)
)

bond_feature_cols = [c for c in BOND_MONTHLY_TEACHER_FEATURES if c in bond_teacher.columns]

bond_teacher[bond_feature_cols] = (
    bond_teacher[bond_feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)

for c in bond_feature_cols:
    valid_idx = bond_teacher.index[bond_teacher[c].notna()]
    if len(valid_idx) > 0:
        bond_teacher.loc[valid_idx[0], c] = np.nan

log_eps_value = float(np.log(TEACHER_EPS))
log_downside_cols = [c for c in bond_feature_cols if c.startswith("log_downside_")]

for c in log_downside_cols:
    for idx in bond_teacher.index:
        x = bond_teacher.loc[idx, c]
        if pd.isna(x):
            continue
        if np.isclose(float(x), log_eps_value, atol=1e-10, rtol=0.0):
            bond_teacher.loc[idx, c] = np.nan
            continue
        break

bond_teacher = clean_teacher_panel(bond_teacher, BOND_MONTHLY_TEACHER_FEATURES)

teacher_daily_panels = {
    ("equity", "return_sortino", "daily_sjm"): equity_teacher.copy(),
}

teacher_monthly_panels = {
    ("bond", "return_downside_sharpe_vol", "monthly_sjm"): bond_teacher.copy(),
}

daily_teacher_features = (
    pd.concat(teacher_daily_panels.values(), axis=0, ignore_index=True, sort=False)
    .sort_values(["asset", "feature_group", "label_source", "date"])
    .reset_index(drop=True)
)

monthly_teacher_features = (
    pd.concat(teacher_monthly_panels.values(), axis=0, ignore_index=True, sort=False)
    .sort_values(["asset", "feature_group", "label_source", "date"])
    .reset_index(drop=True)
)

teacher_feature_inventory = pd.DataFrame([
    {
        "object": "equity_teacher",
        "asset": "equity",
        "teacher_panel": "equity_daily_return_sortino",
        "feature_group": "return_sortino",
        "source_frequency": "daily",
        "label_source": "daily_sjm",
        "feature_source": "daily",
        "n_rows": int(len(equity_teacher)),
        "n_features": int(len(EQUITY_RETURN_SORTINO_FEATURES)),
        "complete_rows": int(equity_teacher[EQUITY_RETURN_SORTINO_FEATURES + ["excess_return"]].notna().all(axis=1).sum()),
        "first_date": date_or_nat(safe_min_date(equity_teacher["date"])),
        "last_date": date_or_nat(safe_max_date(equity_teacher["date"])),
        "avg_feature_missing_pct": float(equity_teacher[EQUITY_RETURN_SORTINO_FEATURES].isna().mean().mean()),
    },
    {
        "object": "bond_teacher",
        "asset": "bond",
        "teacher_panel": "bond_monthly_return_downside_sharpe_vol",
        "feature_group": "return_downside_sharpe_vol",
        "source_frequency": "monthly",
        "label_source": "monthly_sjm",
        "feature_source": "monthly_only",
        "n_rows": int(len(bond_teacher)),
        "n_features": int(len(BOND_MONTHLY_TEACHER_FEATURES)),
        "complete_rows": int(bond_teacher[BOND_MONTHLY_TEACHER_FEATURES + ["excess_return"]].notna().all(axis=1).sum()),
        "first_date": date_or_nat(safe_min_date(bond_teacher["date"])),
        "last_date": date_or_nat(safe_max_date(bond_teacher["date"])),
        "avg_feature_missing_pct": float(bond_teacher[BOND_MONTHLY_TEACHER_FEATURES].isna().mean().mean()),
    },
    {
        "object": "equity_monthly_returns",
        "asset": "equity",
        "teacher_panel": "equity_monthly_return_alignment",
        "feature_group": "monthly_return_alignment",
        "source_frequency": "monthly",
        "label_source": "return_alignment",
        "feature_source": "daily_compounded",
        "n_rows": int(len(equity_monthly_returns)),
        "n_features": 0,
        "complete_rows": int(equity_monthly_returns[["eq_excess"]].notna().all(axis=1).sum()),
        "first_date": date_or_nat(safe_min_date(equity_monthly_returns["date"])),
        "last_date": date_or_nat(safe_max_date(equity_monthly_returns["date"])),
        "avg_feature_missing_pct": np.nan,
    },
])


bond_feature_source_inventory = (
    bond_teacher
    .groupby("feature_source", dropna=False)
    .agg(
        n_months=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        complete_rows=(
            "date",
            lambda z: int(
                bond_teacher.loc[z.index, BOND_MONTHLY_TEACHER_FEATURES + ["excess_return"]]
                .notna()
                .all(axis=1)
                .sum()
            ),
        ),
    )
    .reset_index()
)

daily_teacher_features.to_csv(TEACHER_DAILY_FEATURES_PATH, index=False)
monthly_teacher_features.to_csv(TEACHER_MONTHLY_FEATURES_PATH, index=False)
teacher_feature_inventory.to_csv(TEACHER_FEATURE_INVENTORY_PATH, index=False)
teacher_data_availability.to_csv(TEACHER_DATA_AVAILABILITY_PATH, index=False)

equity_teacher.to_pickle(TEACHER_EQUITY_DAILY_FEATURES_PKL)
bond_teacher.to_pickle(TEACHER_BOND_MONTHLY_FEATURES_PKL)
equity_monthly_returns.to_pickle(TEACHER_EQUITY_MONTHLY_EXCESS_RETURNS_PKL)
pd.to_pickle(teacher_daily_panels, TEACHER_DAILY_PANELS_PKL)
pd.to_pickle(teacher_monthly_panels, TEACHER_MONTHLY_PANELS_PKL)

teacher_feature_objects_manifest = pd.DataFrame([
    {
        "object": "equity_teacher",
        "path": str(TEACHER_EQUITY_DAILY_FEATURES_PKL),
        "shape": str(equity_teacher.shape),
    },
    {
        "object": "bond_teacher",
        "path": str(TEACHER_BOND_MONTHLY_FEATURES_PKL),
        "shape": str(bond_teacher.shape),
    },
    {
        "object": "equity_monthly_returns",
        "path": str(TEACHER_EQUITY_MONTHLY_EXCESS_RETURNS_PKL),
        "shape": str(equity_monthly_returns.shape),
    },
    {
        "object": "teacher_daily_panels",
        "path": str(TEACHER_DAILY_PANELS_PKL),
        "shape": str(len(teacher_daily_panels)),
    },
    {
        "object": "teacher_monthly_panels",
        "path": str(TEACHER_MONTHLY_PANELS_PKL),
        "shape": str(len(teacher_monthly_panels)),
    },
    {
        "object": "daily_teacher_features",
        "path": str(TEACHER_DAILY_FEATURES_PATH),
        "shape": str(daily_teacher_features.shape),
    },
    {
        "object": "monthly_teacher_features",
        "path": str(TEACHER_MONTHLY_FEATURES_PATH),
        "shape": str(monthly_teacher_features.shape),
    },
    {
        "object": "teacher_feature_inventory",
        "path": str(TEACHER_FEATURE_INVENTORY_PATH),
        "shape": str(teacher_feature_inventory.shape),
    },
    {
        "object": "teacher_data_availability",
        "path": str(TEACHER_DATA_AVAILABILITY_PATH),
        "shape": str(teacher_data_availability.shape),
    },
])

teacher_feature_objects_manifest.to_csv(TEACHER_FEATURE_OBJECTS_MANIFEST_PATH, index=False)



# Raw input panel

mkt = pd.read_csv(MKT_PATH)
mkt["month_end"] = force_month_end(mkt["month_end"])

mkt_cols = [
    "ff_mktrf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_mom",
    "smlo_vwret", "smme_vwret", "smhi_vwret",
    "bilo_vwret", "bime_vwret", "bihi_vwret",
    "crsp_ind12_buseq", "crsp_ind12_chems", "crsp_ind12_durbl",
    "crsp_ind12_enrgy", "crsp_ind12_hlth", "crsp_ind12_manuf",
    "crsp_ind12_money", "crsp_ind12_nodur", "crsp_ind12_other",
    "crsp_ind12_shops", "crsp_ind12_telcm", "crsp_ind12_utils",
    "crsp_tsy_1_3y", "crsp_tsy_3_7y", "crsp_tsy_7_10y",
    "crsp_tsy_10_20y", "crsp_tsy_20y_plus",
]

_require_cols(mkt, mkt_cols, "mkt_struct_panel.csv")

mkt = mkt[["month_end"] + mkt_cols].copy()

for c in mkt_cols:
    mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

mkt = (
    mkt.sort_values("month_end")
       .drop_duplicates("month_end", keep="last")
       .reset_index(drop=True)
)

cmdty = pd.read_excel(
    CMDTY_PATH,
    sheet_name="Commodities for the Long Run",
    header=10,
)

cmdty = cmdty.dropna(axis=0, how="all").dropna(axis=1, how="all")
cmdty = cmdty.rename(columns={cmdty.columns[0]: "month_end"})
cmdty = cmdty.rename(columns={
    "Excess return of equal-weight commodities portfolio": "cmdty_broad_aqr_ret",
})

cmdty["month_end"] = force_month_end(cmdty["month_end"])
cmdty = cmdty[cmdty["month_end"].notna()].copy()
cmdty["cmdty_broad_aqr_ret"] = pd.to_numeric(cmdty["cmdty_broad_aqr_ret"], errors="coerce")

cmdty = (
    cmdty[["month_end", "cmdty_broad_aqr_ret"]]
    .sort_values("month_end")
    .drop_duplicates("month_end", keep="last")
    .reset_index(drop=True)
)

x = cmdty["cmdty_broad_aqr_ret"]

cmdty["cmdty_ret_1m"] = x
cmdty["cmdty_ret_6m"] = x.rolling(6, min_periods=6).sum()
cmdty["cmdty_ret_12m"] = x.rolling(12, min_periods=12).sum()
cmdty["cmdty_vol_12m"] = x.rolling(12, min_periods=12).std() * np.sqrt(12)

cmdty["cmdty_sharpe_12m"] = (
    x.rolling(12, min_periods=12).mean()
    / (x.rolling(12, min_periods=12).std() + EPS)
    * np.sqrt(12)
)

cmdty_cols = [
    "cmdty_broad_aqr_ret",
    "cmdty_ret_1m",
    "cmdty_ret_6m",
    "cmdty_ret_12m",
    "cmdty_vol_12m",
    "cmdty_sharpe_12m",
]

cmdty = cmdty[["month_end"] + cmdty_cols].copy()

raw_aligned = (
    mkt.merge(cmdty, on="month_end", how="outer")
       .sort_values("month_end")
       .reset_index(drop=True)
)

FRED_SERIES = {
    "fred_nfci": "NFCI",
    "fred_anfci": "ANFCI",
}

fred_monthly = None

for clean_name, series_id in FRED_SERIES.items():
    raw = fred_obs(series_id, clean_name)
    monthly = to_month_end_last_available(raw, clean_name)

    if fred_monthly is None:
        fred_monthly = monthly
    else:
        fred_monthly = fred_monthly.merge(monthly, on="month_end", how="outer")

fred_monthly = (
    fred_monthly.sort_values("month_end")
                .drop_duplicates("month_end", keep="last")
                .reset_index(drop=True)
)

CREDIT_TICKERS = {
    "credit_ig_proxy_ret": "VWESX",
    "credit_hy_proxy_ret": "VWEHX",
}

credit_parts = []

for ret_col, ticker in CREDIT_TICKERS.items():
    credit_parts.append(
        yahoo_monthly_adjusted_return(
            ticker=ticker,
            ret_col=ret_col,
            start="1970-01-01",
        )
    )

credit_monthly = credit_parts[0]

for part in credit_parts[1:]:
    credit_monthly = credit_monthly.merge(part, on="month_end", how="outer")

credit_monthly = (
    credit_monthly.sort_values("month_end")
                  .drop_duplicates("month_end", keep="last")
                  .reset_index(drop=True)
)

model_panel = pd.read_parquet(MODEL_PANEL_PATH)

model_panel = model_panel.copy()
model_panel["month_end"] = force_month_end(model_panel["date"])

model_raw_cols = [
    "raw_DGS1",
    "derived_slope_10y_2y_wrds",
    "derived_belly_slope",
    "credit_spread_baa_aaa",
    "CORESTICKM159SFRBATL",
    "VIXCLSx",
    "rv_60d",
    "vwretd",
    "sprtrn_sp500",
    "rf",
    "agg_ret",
    "CFNAI",
    "UMCSENTx",
    "bond_10y_ret",
    "bond_10y_mom_12m",
]

_require_cols(model_panel, model_raw_cols, "model_panel_start_1970_01_31.parquet")

model_raw = model_panel[["month_end"] + model_raw_cols].copy()

for c in model_raw_cols:
    model_raw[c] = pd.to_numeric(model_raw[c], errors="coerce")

model_raw = (
    model_raw.sort_values("month_end")
             .drop_duplicates("month_end", keep="last")
             .reset_index(drop=True)
)

external_cols = list(CREDIT_TICKERS.keys()) + list(FRED_SERIES.keys())

raw_aligned_final = (
    raw_aligned
    .drop(columns=[c for c in external_cols + model_raw_cols if c in raw_aligned.columns], errors="ignore")
    .merge(credit_monthly, on="month_end", how="left")
    .merge(fred_monthly, on="month_end", how="left")
    .merge(model_raw, on="month_end", how="left")
    .sort_values("month_end")
    .drop_duplicates("month_end", keep="last")
    .reset_index(drop=True)
)

for c in raw_aligned_final.columns:
    if c != "month_end":
        raw_aligned_final[c] = pd.to_numeric(raw_aligned_final[c], errors="coerce")


# Market-structure features

mkt_raw = raw_aligned_final.copy()

factor_cols = [
    "ff_mktrf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_mom",
]

size_value_cols = [
    "smlo_vwret", "smme_vwret", "smhi_vwret",
    "bilo_vwret", "bime_vwret", "bihi_vwret",
]

industry_cols = sorted([c for c in mkt_raw.columns if c.startswith("crsp_ind12_")])

treasury_cols = [
    "crsp_tsy_1_3y",
    "crsp_tsy_3_7y",
    "crsp_tsy_7_10y",
    "crsp_tsy_10_20y",
    "crsp_tsy_20y_plus",
]

credit_cols = [
    "credit_ig_proxy_ret",
    "credit_hy_proxy_ret",
]

commodity_cols = [
    "cmdty_broad_aqr_ret",
]

financial_conditions_cols = [
    "fred_nfci",
    "fred_anfci",
]

legacy_source_cols = [
    "sprtrn_sp500",
    "agg_ret",
    "rf",
    "raw_DGS1",
    "derived_slope_10y_2y_wrds",
    "derived_belly_slope",
    "credit_spread_baa_aaa",
    "CORESTICKM159SFRBATL",
    "VIXCLSx",
    "rv_60d",
    "vwretd",
    "CFNAI",
    "UMCSENTx",
    "bond_10y_ret",
    "bond_10y_mom_12m",
]

for block_name, cols in [
    ("factor_cols", factor_cols),
    ("size_value_cols", size_value_cols),
    ("industry_cols", industry_cols),
    ("treasury_cols", treasury_cols),
    ("credit_cols", credit_cols),
    ("commodity_cols", commodity_cols),
    ("financial_conditions_cols", financial_conditions_cols),
    ("legacy_source_cols", legacy_source_cols),
]:
    _require_cols(mkt_raw, cols, block_name)

d = {}

d["factor_ret_dispersion_1m"] = _row_dispersion(mkt_raw, factor_cols)
d["factor_ret_dispersion_6m"] = _roll_mean(d["factor_ret_dispersion_1m"], 6)
d["ff_mom_minus_hml"] = mkt_raw["ff_mom"] - mkt_raw["ff_hml"]
d["ff_rmw_minus_cma"] = mkt_raw["ff_rmw"] - mkt_raw["ff_cma"]
d.update(_rolling_pairwise_corr_stats(mkt_raw, factor_cols, window=12, prefix="factor"))
d.update(_lambda_f_series(mkt_raw, factor_cols, window=12, prefix="factor", include_corr=True))

d["small_avg_ret"] = mkt_raw[["smlo_vwret", "smme_vwret", "smhi_vwret"]].mean(axis=1)
d["big_avg_ret"] = mkt_raw[["bilo_vwret", "bime_vwret", "bihi_vwret"]].mean(axis=1)
d["growth_avg_ret"] = mkt_raw[["smlo_vwret", "bilo_vwret"]].mean(axis=1)
d["value_avg_ret"] = mkt_raw[["smhi_vwret", "bihi_vwret"]].mean(axis=1)
d["small_minus_big"] = d["small_avg_ret"] - d["big_avg_ret"]
d["value_minus_growth"] = d["value_avg_ret"] - d["growth_avg_ret"]
d["small_value_minus_big_growth"] = mkt_raw["smhi_vwret"] - mkt_raw["bilo_vwret"]
d["size_value_ret_dispersion_1m"] = _row_dispersion(mkt_raw, size_value_cols)
d["size_value_ret_dispersion_6m"] = _roll_mean(d["size_value_ret_dispersion_1m"], 6)
d.update(_rolling_pairwise_corr_stats(mkt_raw, size_value_cols, window=12, prefix="size_value"))
d.update(_lambda_f_series(mkt_raw, size_value_cols, window=12, prefix="size_value", include_corr=True))

d["industry_ret_dispersion_1m"] = _row_dispersion(mkt_raw, industry_cols)
d["industry_ret_dispersion_6m"] = _roll_mean(d["industry_ret_dispersion_1m"], 6)
d["industry_best_minus_worst_1m"] = _row_best_minus_worst(mkt_raw, industry_cols)
d["industry_best_minus_worst_6m"] = _roll_mean(d["industry_best_minus_worst_1m"], 6)
d.update(_rolling_pairwise_corr_stats(mkt_raw, industry_cols, window=12, prefix="industry"))
d.update(_lambda_f_series(mkt_raw, industry_cols, window=12, prefix="industry", include_corr=True))

d["tsy_short_ret"] = mkt_raw["crsp_tsy_1_3y"]
d["tsy_intermediate_ret"] = mkt_raw[["crsp_tsy_3_7y", "crsp_tsy_7_10y"]].mean(axis=1)
d["tsy_long_ret"] = mkt_raw[["crsp_tsy_10_20y", "crsp_tsy_20y_plus"]].mean(axis=1)
d["tsy_long_minus_short"] = d["tsy_long_ret"] - d["tsy_short_ret"]
d["tsy_20y_minus_1_3y"] = mkt_raw["crsp_tsy_20y_plus"] - mkt_raw["crsp_tsy_1_3y"]
d["tsy_belly_minus_barbell"] = (
    mkt_raw[["crsp_tsy_3_7y", "crsp_tsy_7_10y"]].mean(axis=1)
    - mkt_raw[["crsp_tsy_1_3y", "crsp_tsy_20y_plus"]].mean(axis=1)
)
d["tsy_ret_dispersion_1m"] = _row_dispersion(mkt_raw, treasury_cols)
d["tsy_ret_dispersion_6m"] = _roll_mean(d["tsy_ret_dispersion_1m"], 6)
d.update(_rolling_pairwise_corr_stats(mkt_raw, treasury_cols, window=12, prefix="tsy"))
d.update(_lambda_f_series(mkt_raw, treasury_cols, window=12, prefix="tsy", include_corr=True))

d["credit_hy_minus_ig"] = mkt_raw["credit_hy_proxy_ret"] - mkt_raw["credit_ig_proxy_ret"]
d["credit_ig_ret_6m"] = _roll_sum(mkt_raw["credit_ig_proxy_ret"], 6)
d["credit_hy_ret_6m"] = _roll_sum(mkt_raw["credit_hy_proxy_ret"], 6)
d["credit_hy_ig_corr_12m_fisher"] = _safe_fisher_corr(
    mkt_raw["credit_hy_proxy_ret"].rolling(12, min_periods=12).corr(mkt_raw["credit_ig_proxy_ret"])
)
d["credit_hy_ig_corr_12m_fisher_diff"] = d["credit_hy_ig_corr_12m_fisher"].diff()
d["credit_ret_dispersion_1m"] = _row_dispersion(mkt_raw, credit_cols)
d["credit_ret_dispersion_6m"] = _roll_mean(d["credit_ret_dispersion_1m"], 6)
d.update(_lambda_f_series(mkt_raw, credit_cols, window=12, prefix="credit", include_corr=False))

d["cmdty_ret_1m"] = mkt_raw["cmdty_broad_aqr_ret"]
d["cmdty_ret_6m"] = _roll_sum(mkt_raw["cmdty_broad_aqr_ret"], 6)
d["cmdty_ret_12m"] = _roll_sum(mkt_raw["cmdty_broad_aqr_ret"], 12)
d["cmdty_vol_12m"] = _roll_std(mkt_raw["cmdty_broad_aqr_ret"], 12) * np.sqrt(12.0)

cmdty_roll_mean_12 = _roll_mean(mkt_raw["cmdty_broad_aqr_ret"], 12)
cmdty_roll_std_12 = _roll_std(mkt_raw["cmdty_broad_aqr_ret"], 12)

d["cmdty_sharpe_12m"] = (
    cmdty_roll_mean_12
    / (cmdty_roll_std_12 + EPS)
    * np.sqrt(12.0)
)

d["nfci_level"] = mkt_raw["fred_nfci"]
d["nfci_diff"] = mkt_raw["fred_nfci"].diff()
d["anfci_level"] = mkt_raw["fred_anfci"]
d["anfci_diff"] = mkt_raw["fred_anfci"].diff()
d["nfci_minus_anfci"] = mkt_raw["fred_nfci"] - mkt_raw["fred_anfci"]

d["cross_bond_proxy_ret"] = mkt_raw[treasury_cols].mean(axis=1)

tmp_cross = pd.concat(
    [
        mkt_raw,
        pd.DataFrame({"cross_bond_proxy_ret": d["cross_bond_proxy_ret"]}, index=mkt_raw.index),
    ],
    axis=1,
)

d["cross_eq_bond_corr_12m_fisher_diff"] = _rolling_corr_fisher_diff(
    tmp_cross, "ff_mktrf", "cross_bond_proxy_ret", window=12
)
d["cross_eq_tsy20_corr_12m_fisher_diff"] = _rolling_corr_fisher_diff(
    tmp_cross, "ff_mktrf", "crsp_tsy_20y_plus", window=12
)
d["cross_eq_credit_hy_corr_12m_fisher_diff"] = _rolling_corr_fisher_diff(
    tmp_cross, "ff_mktrf", "credit_hy_proxy_ret", window=12
)
d["cross_bond_credit_ig_corr_12m_fisher_diff"] = _rolling_corr_fisher_diff(
    tmp_cross, "cross_bond_proxy_ret", "credit_ig_proxy_ret", window=12
)
d["cross_eq_cmdty_corr_12m_fisher_diff"] = _rolling_corr_fisher_diff(
    tmp_cross, "ff_mktrf", "cmdty_broad_aqr_ret", window=12
)
d["cross_bond_cmdty_corr_12m_fisher_diff"] = _rolling_corr_fisher_diff(
    tmp_cross, "cross_bond_proxy_ret", "cmdty_broad_aqr_ret", window=12
)

equity_excess = mkt_raw["sprtrn_sp500"] - mkt_raw["rf"]
bond_excess = mkt_raw["agg_ret"] - mkt_raw["rf"]

rho_raw = equity_excess.rolling(12, min_periods=12).corr(bond_excess)
rho_fisher = _safe_fisher_corr(rho_raw)

d["policy_rate_diff"] = _diff1(mkt_raw["raw_DGS1"])
d["curve_10y_2y_diff"] = _diff1(mkt_raw["derived_slope_10y_2y_wrds"])
d["curve_belly_diff"] = _diff1(mkt_raw["derived_belly_slope"])
d["credit_spread_diff"] = _diff1(mkt_raw["credit_spread_baa_aaa"])
d["core_inflation_diff"] = _diff1(mkt_raw["CORESTICKM159SFRBATL"])
d["vix_logdiff"] = _log_diff1(mkt_raw["VIXCLSx"])
d["rv_60d_logdiff"] = _log_diff1(mkt_raw["rv_60d"])
d["equity_market_level"] = mkt_raw["vwretd"]
d["rho_12m_eq_fi_fisher_diff"] = rho_fisher.diff()
d["cfnai_level"] = mkt_raw["CFNAI"]
d["sentiment_logdiff"] = _log_diff1(mkt_raw["UMCSENTx"])
d["bond_10y_ret_level"] = mkt_raw["bond_10y_ret"]
d["bond_10y_mom_diff"] = _diff1(mkt_raw["bond_10y_mom_12m"])

derived_df = pd.DataFrame(d, index=mkt_raw.index)

raw_cols_to_drop = [c for c in derived_df.columns if c in mkt_raw.columns and c != "month_end"]

mkt_struct_panel_intermediate = pd.concat(
    [
        mkt_raw.drop(columns=raw_cols_to_drop, errors="ignore"),
        derived_df,
    ],
    axis=1,
).copy()

mkt_struct_panel_intermediate = (
    mkt_struct_panel_intermediate
    .sort_values("month_end")
    .drop_duplicates("month_end", keep="last")
    .reset_index(drop=True)
)


# Final base-feature panels

SOURCE_ONLY_COLS = [
    "CFNAI",
    "CORESTICKM159SFRBATL",
    "UMCSENTx",
    "VIXCLSx",
    "agg_ret",
    "bond_10y_mom_12m",
    "bond_10y_ret",
    "credit_spread_baa_aaa",
    "derived_belly_slope",
    "derived_slope_10y_2y_wrds",
    "raw_DGS1",
    "rf",
    "rv_60d",
    "sprtrn_sp500",
    "vwretd",
]

mkt_struct_panel_pre_winsor = (
    mkt_struct_panel_intermediate
    .drop(columns=SOURCE_ONLY_COLS)
    .sort_values("month_end")
    .drop_duplicates("month_end", keep="last")
    .reset_index(drop=True)
)

feature_cols = [c for c in mkt_struct_panel_pre_winsor.columns if c != "month_end"]

mkt_struct_panel_merged_full_history = mkt_struct_panel_pre_winsor.copy()

winsor_rows = []

for c in feature_cols:
    before = pd.to_numeric(mkt_struct_panel_merged_full_history[c], errors="coerce")

    after = rolling_winsorize_series(
        before,
        window=WINSOR_WINDOW,
        min_periods=WINSOR_MIN_PERIODS,
        q_low=WINSOR_Q_LOW,
        q_high=WINSOR_Q_HIGH,
    )

    changed = before.notna() & after.notna() & (before != after)

    winsor_rows.append({
        "variable": c,
        "n_obs": int(before.notna().sum()),
        "n_winsorized": int(changed.sum()),
        "winsorized_pct": float(changed.mean()),
        "min_before": float(before.min(skipna=True)) if before.notna().any() else np.nan,
        "max_before": float(before.max(skipna=True)) if before.notna().any() else np.nan,
        "min_after": float(after.min(skipna=True)) if after.notna().any() else np.nan,
        "max_after": float(after.max(skipna=True)) if after.notna().any() else np.nan,
    })

    mkt_struct_panel_merged_full_history[c] = after

winsor_report = (
    pd.DataFrame(winsor_rows)
    .sort_values(["n_winsorized", "variable"], ascending=[False, True])
    .reset_index(drop=True)
)

with_credit_feature_cols = [c for c in mkt_struct_panel_merged_full_history.columns if c != "month_end"]
complete_with_credit = mkt_struct_panel_merged_full_history[with_credit_feature_cols].notna().all(axis=1)

first_with_credit = mkt_struct_panel_merged_full_history.loc[complete_with_credit, "month_end"].iloc[0]
last_with_credit = mkt_struct_panel_merged_full_history.loc[complete_with_credit, "month_end"].iloc[-1]

mkt_struct_panel_merged = (
    mkt_struct_panel_merged_full_history
    .loc[
        (mkt_struct_panel_merged_full_history["month_end"] >= first_with_credit)
        & (mkt_struct_panel_merged_full_history["month_end"] <= last_with_credit)
    ]
    .copy()
    .reset_index(drop=True)
)

mkt_struct_panel_merged.to_csv(OUT_WITH_CREDIT, index=False)

CREDIT_FEATURE_COLS_TO_DROP = [
    "credit_ig_proxy_ret",
    "credit_hy_proxy_ret",
    "credit_hy_minus_ig",
    "credit_ig_ret_6m",
    "credit_hy_ret_6m",
    "credit_hy_ig_corr_12m_fisher",
    "credit_hy_ig_corr_12m_fisher_diff",
    "credit_ret_dispersion_1m",
    "credit_ret_dispersion_6m",
    "credit_lambda_cov_12m",
    "credit_lambda_cov_diff",
    "cross_eq_credit_hy_corr_12m_fisher_diff",
    "cross_bond_credit_ig_corr_12m_fisher_diff",
]

wo_credit_full_history = (
    mkt_struct_panel_merged_full_history
    .drop(columns=CREDIT_FEATURE_COLS_TO_DROP)
    .copy()
)

wo_credit_feature_cols = [c for c in wo_credit_full_history.columns if c != "month_end"]
complete_wo_credit = wo_credit_full_history[wo_credit_feature_cols].notna().all(axis=1)

first_wo_credit = wo_credit_full_history.loc[complete_wo_credit, "month_end"].iloc[0]
last_wo_credit = wo_credit_full_history.loc[complete_wo_credit, "month_end"].iloc[-1]

mkt_struct_panel_merged_wo_credit = (
    wo_credit_full_history
    .loc[
        (wo_credit_full_history["month_end"] >= first_wo_credit)
        & (wo_credit_full_history["month_end"] <= last_wo_credit)
    ]
    .copy()
    .reset_index(drop=True)
)

mkt_struct_panel_merged_wo_credit.to_csv(OUT_WO_CREDIT, index=False)


# Student panels

student_ensemble_feature_panel, raw_cols_credit, z_cols_credit, smooth_cols_credit = add_z_and_smooth(
    mkt_struct_panel_merged,
    output_date_col="date",
)
student_ensemble_feature_panel.to_csv(STUDENT_WITH_CREDIT_PATH, index=False)

student_ensemble_feature_panel_wo_credit, raw_cols_wo_credit, z_cols_wo_credit, smooth_cols_wo_credit = add_z_and_smooth(
    mkt_struct_panel_merged_wo_credit,
    output_date_col="date",
)
student_ensemble_feature_panel_wo_credit.to_csv(STUDENT_WO_CREDIT_PATH, index=False)

plot_raw_z_smooth_grid(
    input_df=student_ensemble_feature_panel,
    panel_name="Student features with credit",
    plot_start_date=PLOT_START_DATE,
    plot_end_date=PLOT_END_DATE,
    n_cols=N_COLS,
    n_rows=N_ROWS,
    figsize=FIGSIZE,
    use_twin_axis=USE_TWIN_AXIS,
)


# Teacher feature plots against asset levels

EQUITY_RETURN_SORTINO_FEATURES_PLOT = [
    "sortino_21d",
    "sortino_63d",
    "sortino_126d",
    "sortino_252d",
    "ewm_return_21d",
    "ewm_return_63d",
    "ewm_return_126d",
    "ewm_return_252d",
]


BOND_MONTHLY_TEACHER_FEATURES_PLOT = [
    "ewm_return_3m",
    "ewm_return_6m",
    "ewm_return_12m",
    "log_downside_1m",
    "log_downside_3m",
    "log_downside_6m",
    "log_downside_12m",
    "sharpe_3m",
    "sharpe_6m",
    "sharpe_12m",
    "realized_vol_6m",
]


def plot_month_end(s):
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def z_for_plot(s):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    sd = x.std(skipna=True, ddof=0)

    if not np.isfinite(sd) or sd <= 1e-12:
        return x * np.nan

    return (x - x.mean(skipna=True)) / sd


def make_equity_level_for_plot():
    level = spx_levels[["date", "index_level"]].copy()
    level["date"] = pd.to_datetime(level["date"], errors="coerce")
    level["level"] = pd.to_numeric(level["index_level"], errors="coerce")

    return (
        level[["date", "level"]]
        .dropna(subset=["date", "level"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def make_bond_level_for_plot():
    level = agg_levels[["date", "index_level"]].copy()
    level["date"] = plot_month_end(level["date"])
    level["level"] = pd.to_numeric(level["index_level"], errors="coerce")

    return (
        level[["date", "level"]]
        .dropna(subset=["date", "level"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def plot_return_and_level(df, level_df, title_txt, frequency):
    x = df[["date", "excess_return"]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["excess_return"] = pd.to_numeric(x["excess_return"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    level = level_df.copy()
    level["date"] = pd.to_datetime(level["date"], errors="coerce")
    level["level"] = pd.to_numeric(level["level"], errors="coerce")
    level = level.dropna(subset=["date", "level"]).sort_values("date").drop_duplicates("date", keep="last")

    if frequency == "daily":
        plot_df = x.merge(level, on="date", how="left")
    else:
        x["date"] = plot_month_end(x["date"])
        level["date"] = plot_month_end(level["date"])
        plot_df = x.merge(level, on="date", how="left")

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)

    axes[0].plot(plot_df["date"], plot_df["level"], color="black", linewidth=1.1)
    axes[0].set_title(title_txt + " - level", loc="left", fontsize=12)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(plot_df["date"], plot_df["excess_return"], color="tab:blue", linewidth=0.85)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title(title_txt + " - excess return used for teacher", loc="left", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()


def plot_feature_grid_against_level(df, features, level_df, title_txt, frequency):
    x = df[["date"] + features].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.sort_values("date").reset_index(drop=True)

    level = level_df.copy()
    level["date"] = pd.to_datetime(level["date"], errors="coerce")
    level["level"] = pd.to_numeric(level["level"], errors="coerce")
    level = level.dropna(subset=["date", "level"]).sort_values("date").drop_duplicates("date", keep="last")

    if frequency == "daily":
        plot_df = x.merge(level, on="date", how="left")
    else:
        x["date"] = plot_month_end(x["date"])
        level["date"] = plot_month_end(level["date"])
        plot_df = x.merge(level, on="date", how="left")

    ncols = 4
    nrows = int(math.ceil(len(features) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        temp = plot_df[["date", "level", feature]].copy()
        temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
        temp["feature_z"] = z_for_plot(temp[feature])

        ax.plot(temp["date"], temp["level"], color="black", linewidth=1.0, alpha=0.85)
        ax.set_title(feature, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=7, colors="black")

        ax2 = ax.twinx()
        ax2.plot(temp["date"], temp["feature_z"], color="tab:blue", linewidth=1.0, alpha=0.85)
        ax2.axhline(0.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.55)
        ax2.tick_params(axis="y", labelsize=7, colors="tab:blue")

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(title_txt, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


equity_level_plot = make_equity_level_for_plot()
bond_level_plot = make_bond_level_for_plot()

plot_return_and_level(
    equity_teacher,
    equity_level_plot,
    "Equity teacher input",
    frequency="daily",
)

plot_return_and_level(
    bond_teacher,
    bond_level_plot,
    "Bond teacher input",
    frequency="monthly",
)

plot_feature_grid_against_level(
    equity_teacher,
    EQUITY_RETURN_SORTINO_FEATURES_PLOT,
    equity_level_plot,
    "Equity return_sortino teacher features vs equity level",
    frequency="daily",
)


plot_feature_grid_against_level(
    bond_teacher,
    BOND_MONTHLY_TEACHER_FEATURES_PLOT,
    bond_level_plot,
    "Bond monthly return, downside, Sharpe, and volatility features vs bond level",
    frequency="monthly",
)
