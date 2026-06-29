"""Baseline helpers for the two-asset statistical jump-model allocation notebook."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from jumpmodels.jump import JumpModel
except Exception:  # pragma: no cover - used only when the optional package is missing
    JumpModel = None


EPS = 1e-12
HALFLIFE_MAP = {"h1": 1, "h2": 2, "h4": 4}
DEFAULT_LAMBDA_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
DEFAULT_CHOSEN_LAMBDAS = {"equity": 10.0, "bond": 3.0}
DEFAULT_STRATEGIES = ["60_40", "EW", "SJM_RULE", "MINVAR", "MINVAR_SJM"]


@dataclass(frozen=True)
class AllocationConfig:
    one_way_transaction_cost: float = 0.0005
    covariance_halflife_months: float = 12.0
    regime_mean_lookback_years: int = 11
    constant_minvar_mu: float = 0.001
    grid_step: float = 0.0025
    min_history_months: int = 60


def load_panel(data_path, trunc_date="2024-11-30"):
    """Load the monthly panel and create total and excess return columns."""
    required_cols = [
        "date",
        "sprtrn_sp500",
        "agg_ret",
        "rf",
        "sp500_fwd_excess_1m",
        "bond_fwd_excess_1m",
        "sp500_fwd_excess_3m",
        "bond_fwd_excess_3m",
        "sp500_fwd_excess_6m",
        "bond_fwd_excess_6m",
        "derived_y_2Y_wrds_diff_1_wrds",
        "derived_slope_10y_2y_wrds",
        "VIXCLSx",
    ]
    panel_raw = pd.read_parquet(data_path).copy()
    panel_raw["date"] = pd.to_datetime(panel_raw["date"])
    panel_raw = panel_raw.sort_values("date").reset_index(drop=True)

    missing = sorted(set(required_cols).difference(panel_raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    trunc_date = pd.Timestamp(trunc_date)
    panel = panel_raw.loc[panel_raw["date"] <= trunc_date, required_cols].copy()
    for col in panel.columns.difference(["date"]):
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float64")

    panel["equity_total_ret"] = panel["sprtrn_sp500"]
    panel["bond_total_ret"] = panel["agg_ret"]
    panel["rf_total_ret"] = panel["rf"]
    panel["equity_excess"] = panel["equity_total_ret"] - panel["rf_total_ret"]
    panel["bond_excess"] = panel["bond_total_ret"] - panel["rf_total_ret"]
    return panel_raw, panel


def _ewm_mean(series, halflife):
    return series.ewm(halflife=halflife, adjust=False).mean()


def _ewm_std(series, halflife):
    return series.ewm(halflife=halflife, adjust=False).std(bias=False)


def _ewm_downside_dev(series, halflife):
    downside_sq = series.clip(upper=0.0).pow(2)
    return np.sqrt(downside_sq.ewm(halflife=halflife, adjust=False).mean())


def _return_features(df, source_col, prefix, ratio_name):
    out = pd.DataFrame({"date": df["date"]}).copy()
    series = pd.to_numeric(df[source_col], errors="coerce")
    for label, halflife in HALFLIFE_MAP.items():
        mean_col = f"{prefix}_mean_{label}"
        dd_col = f"{prefix}_dd_{label}"
        logdd_col = f"{prefix}_logdd_{label}"
        std_col = f"{prefix}_std_{label}"
        ratio_col = f"{prefix}_{ratio_name}_{label}"

        out[mean_col] = _ewm_mean(series, halflife)
        out[dd_col] = _ewm_downside_dev(series, halflife)
        out[logdd_col] = np.log(out[dd_col] + EPS)
        out[std_col] = _ewm_std(series, halflife)
        denominator = out[dd_col] if ratio_name == "sortino" else out[std_col]
        out[ratio_col] = out[mean_col] / (denominator + EPS)
    return out


def build_feature_panel(panel):
    """Build the baseline return, downside-risk, macro, and cross-asset features."""
    feature_panel = panel[["date", "equity_excess", "bond_excess"]].copy()
    equity_features = _return_features(panel, "equity_excess", "eq", "sortino")
    bond_features = _return_features(panel, "bond_excess", "bd", "sharpe")

    feature_panel = feature_panel.merge(equity_features, on="date", how="left", validate="one_to_one")
    feature_panel = feature_panel.merge(bond_features, on="date", how="left", validate="one_to_one")
    feature_panel["macro_y2_change"] = pd.to_numeric(panel["derived_y_2Y_wrds_diff_1_wrds"], errors="coerce")
    feature_panel["macro_slope_10y_2y"] = pd.to_numeric(panel["derived_slope_10y_2y_wrds"], errors="coerce")
    feature_panel["macro_vix"] = pd.to_numeric(panel["VIXCLSx"], errors="coerce")
    feature_panel["eq_bd_corr_h4"] = panel["equity_excess"].rolling(4, min_periods=4).corr(panel["bond_excess"])
    return feature_panel.replace([np.inf, -np.inf], np.nan)


def _run_lengths(state_bull):
    state = pd.Series(state_bull).astype(int).reset_index(drop=True)
    runs = []
    current = int(state.iloc[0])
    length = 1
    for value in state.iloc[1:]:
        value = int(value)
        if value == current:
            length += 1
        else:
            runs.append({"state_bull": current, "run_length": length})
            current = value
            length = 1
    runs.append({"state_bull": current, "run_length": length})
    return pd.DataFrame(runs)


def _transition_matrix(state_bull):
    state = pd.Series(state_bull).dropna().astype(int).reset_index(drop=True)
    counts = np.ones((2, 2), dtype=float) * EPS
    for prev, nxt in zip(state.iloc[:-1], state.iloc[1:]):
        counts[int(prev), int(nxt)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _fit_one_lambda(X_fit, returns, jump_penalty, random_state):
    if JumpModel is not None:
        model = JumpModel(
            n_components=2,
            jump_penalty=float(jump_penalty),
            cont=False,
            random_state=random_state,
            max_iter=1000,
            tol=1e-8,
            n_init=20,
            verbose=0,
        )
        model.fit(X_fit, ret_ser=returns, sort_by="cumret")
        raw_state = pd.Series(np.asarray(model.labels_).astype(int), index=X_fit.index)
        probabilities = np.asarray(model.proba_, dtype=float)
        return raw_state, probabilities

    model = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    raw_state = pd.Series(model.fit_predict(X_fit).astype(int), index=X_fit.index)
    distances = model.transform(X_fit)
    inverse_distance = 1.0 / (distances + EPS)
    probabilities = inverse_distance / inverse_distance.sum(axis=1, keepdims=True)
    return raw_state, probabilities


def fit_asset_sjm(asset, feature_panel, excess_col, feature_cols, lambda_grid=None, random_state=42):
    """Fit two-state statistical jump models across a jump-penalty grid."""
    lambda_grid = DEFAULT_LAMBDA_GRID if lambda_grid is None else lambda_grid
    fit_cols = ["date", excess_col] + feature_cols
    fit_df = feature_panel.loc[feature_panel[fit_cols].notna().all(axis=1), fit_cols].copy().reset_index(drop=True)
    if fit_df.empty:
        raise ValueError(f"No usable rows for {asset} SJM fit.")

    X_raw = fit_df[feature_cols].copy()
    X_fit = pd.DataFrame(StandardScaler().fit_transform(X_raw), index=fit_df.index, columns=feature_cols)
    diagnostics = []
    label_frames = []

    for jump_penalty in lambda_grid:
        raw_state, probabilities = _fit_one_lambda(X_fit, fit_df[excess_col], jump_penalty, random_state)
        cumret_by_state = fit_df.groupby(raw_state)[excess_col].sum().sort_index()
        bull_state_id = int(cumret_by_state.idxmax())
        bear_state_id = int(cumret_by_state.idxmin())
        state_bull = (raw_state == bull_state_id).astype(int)
        bull_col = bull_state_id
        bear_col = bear_state_id

        runs = _run_lengths(state_bull)
        trans = _transition_matrix(state_bull)
        bull_mask = state_bull == 1
        bear_mask = state_bull == 0

        diagnostics.append(
            {
                "asset": asset,
                "lambda": float(jump_penalty),
                "n_obs": int(len(fit_df)),
                "bull_share": float(bull_mask.mean()),
                "bear_share": float(bear_mask.mean()),
                "n_switches": int(state_bull.ne(state_bull.shift(1)).iloc[1:].sum()),
                "trans_bear_to_bear": float(trans[0, 0]),
                "trans_bear_to_bull": float(trans[0, 1]),
                "trans_bull_to_bear": float(trans[1, 0]),
                "trans_bull_to_bull": float(trans[1, 1]),
                "avg_run_length": float(runs["run_length"].mean()),
                "bull_mean_excess": float(fit_df.loc[bull_mask, excess_col].mean()),
                "bear_mean_excess": float(fit_df.loc[bear_mask, excess_col].mean()),
                "bull_n_months": int(bull_mask.sum()),
                "bear_n_months": int(bear_mask.sum()),
            }
        )
        label_frames.append(
            pd.DataFrame(
                {
                    "date": fit_df["date"],
                    "asset": asset,
                    "lambda": float(jump_penalty),
                    "raw_state": raw_state.to_numpy(dtype=int),
                    "state_bull": state_bull.to_numpy(dtype=int),
                    "state_bear": (1 - state_bull).to_numpy(dtype=int),
                    "p_bull": probabilities[:, bull_col].astype(float),
                    "p_bear": probabilities[:, bear_col].astype(float),
                    "excess_return": fit_df[excess_col].to_numpy(dtype=float),
                }
            )
        )

    return pd.DataFrame(diagnostics), pd.concat(label_frames, axis=0, ignore_index=True)


def fit_regime_labels(feature_panel, lambda_grid=None):
    """Fit the equity and bond SJM label panels used by the baseline allocation."""
    equity_cols = ["eq_logdd_h1", "eq_logdd_h4", "eq_mean_h1", "eq_mean_h2", "eq_mean_h4", "eq_sortino_h1", "eq_sortino_h2", "eq_sortino_h4"]
    bond_cols = ["bd_mean_h1", "bd_mean_h2", "bd_mean_h4", "bd_sharpe_h1", "bd_sharpe_h2", "bd_sharpe_h4"]
    equity_diag, equity_labels = fit_asset_sjm("equity", feature_panel, "equity_excess", equity_cols, lambda_grid)
    bond_diag, bond_labels = fit_asset_sjm("bond", feature_panel, "bond_excess", bond_cols, lambda_grid)
    diagnostics = pd.concat([equity_diag, bond_diag], axis=0, ignore_index=True).sort_values(["asset", "lambda"]).reset_index(drop=True)
    labels = pd.concat([equity_labels, bond_labels], axis=0, ignore_index=True).sort_values(["asset", "lambda", "date"]).reset_index(drop=True)
    return diagnostics, labels


def build_allocation_signals(panel, labels, chosen_lambdas=None, min_history_months=60):
    """Merge selected SJM states with one-month-ahead total returns for backtesting."""
    chosen_lambdas = DEFAULT_CHOSEN_LAMBDAS if chosen_lambdas is None else chosen_lambdas
    selected = labels.loc[labels["asset"].isin(chosen_lambdas)].copy()
    selected["chosen_lambda"] = selected["asset"].map(chosen_lambdas)
    selected = selected.loc[np.isclose(selected["lambda"], selected["chosen_lambda"])].copy()

    signal_wide = selected.pivot(index="date", columns="asset", values=["state_bull", "p_bull"])
    signal_wide.columns = [f"{asset}_{field}" for field, asset in signal_wide.columns]
    signal_wide = signal_wide.reset_index().sort_values("date").reset_index(drop=True)
    signal_wide = signal_wide.rename(
        columns={
            "equity_state_bull": "equity_signal_bull",
            "bond_state_bull": "bond_signal_bull",
            "equity_p_bull": "equity_p_bull",
            "bond_p_bull": "bond_p_bull",
        }
    )

    returns = panel[["date", "equity_total_ret", "bond_total_ret", "rf_total_ret", "equity_excess", "bond_excess"]].copy()
    returns["equity_total_ret_next_1m"] = returns["equity_total_ret"].shift(-1)
    returns["bond_total_ret_next_1m"] = returns["bond_total_ret"].shift(-1)
    returns["rf_total_ret_next_1m"] = returns["rf_total_ret"].shift(-1)

    alloc_panel = signal_wide.merge(returns, on="date", how="inner", validate="one_to_one")
    required = ["equity_signal_bull", "bond_signal_bull", "equity_total_ret_next_1m", "bond_total_ret_next_1m", "rf_total_ret_next_1m"]
    alloc_panel = alloc_panel.loc[alloc_panel[required].notna().all(axis=1)].copy().reset_index(drop=True)

    history_panel = returns[["date", "equity_excess", "bond_excess"]].copy()
    history_panel = history_panel.merge(signal_wide[["date", "equity_signal_bull", "bond_signal_bull"]], on="date", how="left", validate="one_to_one")
    if len(alloc_panel) <= min_history_months:
        raise ValueError("Allocation panel is too short after fitting SJM labels.")
    alloc_panel = alloc_panel.iloc[min_history_months:].reset_index(drop=True)
    return alloc_panel, history_panel


def _ewm_weights(n, halflife):
    decay = np.exp(np.log(0.5) / float(halflife))
    exponents = np.arange(n - 1, -1, -1, dtype=float)
    weights = decay ** exponents
    return weights / weights.sum()


def _ewm_mean_and_cov(two_col_df, halflife, min_obs=24):
    x = two_col_df.dropna().astype(float)
    if len(x) < min_obs:
        return np.array([np.nan, np.nan]), np.full((2, 2), np.nan)
    arr = x.values
    weights = _ewm_weights(len(arr), halflife)
    mu = (weights[:, None] * arr).sum(axis=0)
    centered = arr - mu
    cov = np.einsum("n,ni,nj->ij", weights, centered, centered) + np.eye(2) * 1e-10
    return mu, cov


def _regime_conditioned_means(history_panel, end_date, window_years):
    start_date = pd.Timestamp(end_date) - pd.DateOffset(years=window_years)
    use = history_panel.loc[(history_panel["date"] <= end_date) & (history_panel["date"] > start_date)].copy()
    if len(use) < 24:
        use = history_panel.loc[history_panel["date"] <= end_date].copy()

    out = {}
    for asset in ["equity", "bond"]:
        ret_col = f"{asset}_excess"
        state_col = f"{asset}_signal_bull"
        unconditional = float(use[ret_col].dropna().mean()) if use[ret_col].notna().any() else 0.0
        bull_mean = use.loc[use[state_col] == 1, ret_col].dropna().mean()
        bear_mean = use.loc[use[state_col] == 0, ret_col].dropna().mean()
        out[f"{asset}_bull"] = unconditional if pd.isna(bull_mean) else float(bull_mean)
        out[f"{asset}_bear"] = unconditional if pd.isna(bear_mean) else float(bear_mean)
    return out


def _weight_grid(grid_step):
    grid_vals = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step)
    return np.array([(we, wb) for we in grid_vals for wb in grid_vals if (we + wb) <= 1.0 + 1e-12], dtype=float)


def _optimize_two_asset(mu, cov, pre_risky, gamma_risk, gamma_trade, config):
    mu = np.nan_to_num(np.asarray(mu, dtype=float).reshape(2), nan=0.0)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    if np.isnan(cov).any():
        cov = np.eye(2) * 1e-6
    pre_risky = np.asarray(pre_risky, dtype=float).reshape(2)
    grid = _weight_grid(config.grid_step)
    utility_ret = grid @ mu
    utility_var = np.einsum("ij,jk,ik->i", grid, cov, grid)
    utility_trade = np.abs(grid[:, 0] - pre_risky[0]) + np.abs(grid[:, 1] - pre_risky[1])
    utility = utility_ret - gamma_risk * utility_var - gamma_trade * config.one_way_transaction_cost * utility_trade
    return grid[int(np.argmax(utility))].copy()


def _choose_target_weights(strategy, row, history_panel, pre_risky, config):
    date = pd.Timestamp(row["date"])
    eq_bull = int(row["equity_signal_bull"])
    bond_bull = int(row["bond_signal_bull"])
    eq_p = float(row["equity_p_bull"])
    bond_p = float(row["bond_p_bull"])
    pre_for_opt = np.array([0.0, 0.0]) if pre_risky is None else pre_risky.copy()
    hist_to_t = history_panel.loc[history_panel["date"] <= date].copy()
    _, cov = _ewm_mean_and_cov(hist_to_t[["equity_excess", "bond_excess"]], config.covariance_halflife_months)

    if strategy == "60_40":
        return np.array([0.60, 0.40]), np.array([np.nan, np.nan]), np.nan, np.nan, eq_bull, bond_bull, eq_p, bond_p
    if strategy == "EW":
        return np.array([0.50, 0.50]), np.array([np.nan, np.nan]), np.nan, np.nan, eq_bull, bond_bull, eq_p, bond_p
    if strategy == "SJM_RULE":
        if eq_bull == 1 and bond_bull == 1:
            target = np.array([0.50, 0.50])
        elif eq_bull == 1 and bond_bull == 0:
            target = np.array([1.00, 0.00])
        elif eq_bull == 0 and bond_bull == 1:
            target = np.array([0.00, 1.00])
        else:
            target = np.array([0.00, 0.00])
        return target, np.array([np.nan, np.nan]), np.nan, np.nan, eq_bull, bond_bull, eq_p, bond_p
    if strategy == "MINVAR":
        mu = np.array([config.constant_minvar_mu, config.constant_minvar_mu])
        return _optimize_two_asset(mu, cov, pre_for_opt, 10.0, 0.0, config), mu, 10.0, 0.0, eq_bull, bond_bull, eq_p, bond_p
    if strategy == "MINVAR_SJM":
        means = _regime_conditioned_means(hist_to_t, date, config.regime_mean_lookback_years)
        mu = np.array([
            means["equity_bull"] if eq_bull == 1 else means["equity_bear"],
            means["bond_bull"] if bond_bull == 1 else means["bond_bear"],
        ])
        return _optimize_two_asset(mu, cov, pre_for_opt, 10.0, 1.0, config), mu, 10.0, 1.0, eq_bull, bond_bull, eq_p, bond_p
    raise ValueError(f"Unknown strategy: {strategy}")


def build_allocation_weights(alloc_panel, history_panel, strategies=None, config=None):
    """Create target weights for each baseline strategy."""
    strategies = DEFAULT_STRATEGIES if strategies is None else strategies
    config = AllocationConfig() if config is None else config
    frames = []
    for strategy in strategies:
        pre_risky = None
        rows = []
        for _, row in alloc_panel.sort_values("date").iterrows():
            target, mu_used, gamma_risk, gamma_trade, eq_bull, bond_bull, eq_p, bond_p = _choose_target_weights(strategy, row, history_panel, pre_risky, config)
            target_we = float(target[0])
            target_wb = float(target[1])
            target_wc = float(max(0.0, 1.0 - target_we - target_wb))
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "strategy": strategy,
                    "target_w_equity": target_we,
                    "target_w_bond": target_wb,
                    "target_w_cash": target_wc,
                    "target_leverage": target_we + target_wb,
                    "gamma_risk_used": gamma_risk,
                    "gamma_trade_used": gamma_trade,
                    "mu_equity_used": float(mu_used[0]) if np.ndim(mu_used) == 1 else np.nan,
                    "mu_bond_used": float(mu_used[1]) if np.ndim(mu_used) == 1 else np.nan,
                    "eq_pred_bull_binary": eq_bull,
                    "bond_pred_bull_binary": bond_bull,
                    "eq_p_bull": eq_p,
                    "bond_p_bull": bond_p,
                }
            )
            pre_risky = np.array([target_we, target_wb], dtype=float)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, axis=0, ignore_index=True)


def run_backtest(alloc_panel, weights, config=None):
    """Apply target weights to next-month returns, including one-way transaction costs."""
    config = AllocationConfig() if config is None else config
    backtest_frames = []
    returns = alloc_panel[["date", "equity_total_ret_next_1m", "bond_total_ret_next_1m", "rf_total_ret_next_1m"]].copy()

    for strategy, strat_weights in weights.groupby("strategy"):
        work = strat_weights.merge(returns, on="date", how="left", validate="one_to_one").sort_values("date").reset_index(drop=True)
        pre_risky = None
        wealth = 1.0
        rows = []
        for _, row in work.iterrows():
            target_we = float(row["target_w_equity"])
            target_wb = float(row["target_w_bond"])
            target_wc = float(row["target_w_cash"])
            if pre_risky is None:
                turnover = 0.0
            else:
                turnover = float(abs(target_we - pre_risky[0]) + abs(target_wb - pre_risky[1]))
            trade_cost_rate = config.one_way_transaction_cost * turnover
            eq_ret = float(row["equity_total_ret_next_1m"])
            bond_ret = float(row["bond_total_ret_next_1m"])
            rf_ret = float(row["rf_total_ret_next_1m"])
            gross_return = target_we * eq_ret + target_wb * bond_ret + target_wc * rf_ret
            net_total_return = (1.0 - trade_cost_rate) * (1.0 + gross_return) - 1.0
            net_excess_return = net_total_return - rf_ret
            wealth *= 1.0 + net_total_return

            growth_components = (1.0 - trade_cost_rate) * np.array([
                target_we * (1.0 + eq_ret),
                target_wb * (1.0 + bond_ret),
                target_wc * (1.0 + rf_ret),
            ])
            total_after = float(growth_components.sum())
            pre_risky = np.array([0.0, 0.0]) if total_after <= 0 else growth_components[:2] / total_after

            out = row.to_dict()
            out.update(
                {
                    "turnover": turnover,
                    "trade_cost_rate": trade_cost_rate,
                    "gross_portfolio_return": gross_return,
                    "net_total_return": net_total_return,
                    "net_excess_return": net_excess_return,
                    "wealth": wealth,
                }
            )
            rows.append(out)
        backtest_frames.append(pd.DataFrame(rows))
    return pd.concat(backtest_frames, axis=0, ignore_index=True)


def _max_drawdown(wealth):
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def summarize_backtest(backtest):
    """Summarize annualized return, risk, turnover, drawdown, and average weights."""
    rows = []
    for strategy, group in backtest.groupby("strategy"):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        mean_excess = float(group["net_excess_return"].mean())
        std_excess = float(group["net_excess_return"].std(ddof=0))
        ann_excess_return = mean_excess * 12.0
        ann_excess_vol = std_excess * np.sqrt(12.0)
        sharpe_ann = ann_excess_return / ann_excess_vol if ann_excess_vol > 0 else np.nan
        ann_total_return = float(group["wealth"].iloc[-1] ** (12.0 / n) - 1.0) if n > 0 else np.nan
        max_drawdown = _max_drawdown(group["wealth"]) if n > 0 else np.nan
        rows.append(
            {
                "strategy": strategy,
                "start_date": group["date"].min().date(),
                "end_date": group["date"].max().date(),
                "n_months": n,
                "ann_excess_return": ann_excess_return,
                "ann_excess_vol": ann_excess_vol,
                "sharpe_ann": sharpe_ann,
                "max_drawdown": max_drawdown,
                "ann_total_return": ann_total_return,
                "calmar": ann_total_return / abs(max_drawdown) if pd.notna(max_drawdown) and max_drawdown < 0 else np.nan,
                "ann_turnover": float(group["turnover"].mean() * 12.0),
                "avg_leverage": float(group["target_leverage"].mean()),
                "mean_trade_cost_rate": float(group["trade_cost_rate"].mean()),
                "mean_target_equity_weight": float(group["target_w_equity"].mean()),
                "mean_target_bond_weight": float(group["target_w_bond"].mean()),
                "mean_target_cash_weight": float(group["target_w_cash"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe_ann", ascending=False).reset_index(drop=True)


def _apply_year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _contiguous_spans(mask):
    mask = pd.Series(mask).astype(bool)
    spans = []
    start = None
    last_idx = None
    for idx, value in mask.items():
        if value and start is None:
            start = idx
        if not value and start is not None:
            spans.append((start, last_idx))
            start = None
        last_idx = idx
    if start is not None:
        spans.append((start, last_idx))
    return spans


def _add_regime_shading(ax, rule_df):
    specs = [
        ((rule_df["eq_pred_bull_binary"] == 1) & (rule_df["bond_pred_bull_binary"] == 1), "tab:green", "equity bull / bond bull"),
        ((rule_df["eq_pred_bull_binary"] == 1) & (rule_df["bond_pred_bull_binary"] == 0), "tab:blue", "equity bull / bond bear"),
        ((rule_df["eq_pred_bull_binary"] == 0) & (rule_df["bond_pred_bull_binary"] == 1), "tab:orange", "equity bear / bond bull"),
        ((rule_df["eq_pred_bull_binary"] == 0) & (rule_df["bond_pred_bull_binary"] == 0), "lightgray", "equity bear / bond bear"),
    ]
    handles = []
    date_index = pd.to_datetime(rule_df["date"])
    for mask, color, label in specs:
        for start, end in _contiguous_spans(pd.Series(mask.to_numpy(), index=date_index)):
            ax.axvspan(start, end + pd.offsets.MonthEnd(1), color=color, alpha=0.08, lw=0, zorder=0)
        handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.16, label=label))
    return handles


def plot_backtest(backtest, summary):
    """Plot strategy wealth, target weights for the SJM rule, and performance metrics."""
    curve_wide = backtest.pivot(index="date", columns="strategy", values="wealth").sort_index()
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)

    curve_wide.plot(ax=axes[0], logy=True, lw=1.8)
    shade_df = backtest.loc[backtest["strategy"] == "SJM_RULE", ["date", "eq_pred_bull_binary", "bond_pred_bull_binary"]]
    handles = _add_regime_shading(axes[0], shade_df)
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(line_handles + handles, line_labels + [h.get_label() for h in handles], loc="upper left", ncol=2, fontsize=8)
    axes[0].set_title("Baseline strategy wealth")
    axes[0].set_ylabel("wealth (log scale)")
    axes[0].grid(True, alpha=0.3)
    _apply_year_axis(axes[0])

    weights = backtest.loc[backtest["strategy"] == "SJM_RULE"].set_index("date")[["target_w_equity", "target_w_bond", "target_w_cash"]]
    weights.plot(ax=axes[1], lw=1.8)
    axes[1].set_title("SJM rule target weights")
    axes[1].set_ylabel("target weight")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    _apply_year_axis(axes[1])

    table_cols = ["strategy", "ann_excess_return", "ann_excess_vol", "sharpe_ann", "max_drawdown", "ann_turnover"]
    table_data = summary[table_cols].copy()
    for col in table_cols[1:]:
        table_data[col] = table_data[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    axes[2].axis("off")
    table = axes[2].table(cellText=table_data.values, colLabels=table_data.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    axes[2].set_title("Performance summary")

    fig.tight_layout()
    return fig, axes
