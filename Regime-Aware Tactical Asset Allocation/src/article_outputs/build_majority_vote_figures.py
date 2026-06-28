import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from article_outputs.common import OUTPUT_DIR, save_figure


STATE_COLORS = {1: "#b7e4bd", 0: "#f5b5b5"}


def _date_edges(dates):
    dates = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    if len(dates) == 0:
        return [], []
    if len(dates) == 1:
        return [dates.iloc[0] - pd.offsets.MonthBegin(1)], [dates.iloc[0] + pd.offsets.MonthEnd(1)]
    mids = dates.iloc[:-1] + (dates.iloc[1:].to_numpy() - dates.iloc[:-1].to_numpy()) / 2
    left = [dates.iloc[0] - (mids.iloc[0] - dates.iloc[0])] + list(mids)
    right = list(mids) + [dates.iloc[-1] + (dates.iloc[-1] - mids.iloc[-1])]
    return left, right


def _add_state_background(ax, df, state_col):
    x = df[["date", state_col]].dropna().copy()
    if x.empty:
        return
    x["date"] = pd.to_datetime(x["date"])
    x[state_col] = x[state_col].astype(int)
    x = x.sort_values("date").reset_index(drop=True)
    left, right = _date_edges(x["date"])
    x["left"] = left
    x["right"] = right
    x["run"] = (x[state_col] != x[state_col].shift()).cumsum()
    ymin, ymax = ax.get_ylim()
    for _, g in x.groupby("run", sort=True):
        state = int(g[state_col].iloc[0])
        ax.axvspan(g["left"].iloc[0], g["right"].iloc[-1], color=STATE_COLORS[state], alpha=0.60, lw=0)
    ax.set_ylim(ymin, ymax)


def _wealth(r):
    return 100.0 * (1.0 + r.fillna(0.0)).cumprod()


def build_majority_vote_figures():
    path = OUTPUT_DIR / "majority_vote" / "Majority Vote Labels Wide.csv"
    wide = pd.read_csv(path, parse_dates=["date"]).sort_values("date")

    specs = [
        ("equity_return_sortino", "Equity return-risk regime"),
        ("bond_return_downside_sharpe_vol", "Bond return-risk regime"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.2), sharex=True)

    for ax, (base, title) in zip(axes, specs):
        state_col = f"{base}_all_models_state"
        vote_col = f"{base}_all_models_vote_strength"
        ret_col = f"{base}_all_models_excess"
        required = [state_col, vote_col, ret_col]
        missing = [c for c in required if c not in wide.columns]
        if missing:
            raise KeyError(f"Missing columns for {base}: {missing}")

        df = wide[["date", state_col, vote_col, ret_col]].dropna(subset=[state_col, ret_col]).copy()
        df["level"] = _wealth(df[ret_col])
        ax.plot(df["date"], df["level"], color="black", lw=1.25, zorder=3)
        _add_state_background(ax, df, state_col)
        ax2 = ax.twinx()
        ax2.plot(df["date"], df[vote_col], color="#2f5f9f", lw=0.85, alpha=0.80, zorder=4)
        ax2.set_ylim(0.45, 1.02)
        ax2.set_ylabel("Vote strength")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Excess-return wealth")
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Date")
    axes[0].legend(
        handles=[
            Patch(facecolor=STATE_COLORS[1], alpha=0.60, label="Good state"),
            Patch(facecolor=STATE_COLORS[0], alpha=0.60, label="Bad state"),
        ],
        loc="upper left",
        frameon=True,
    )
    fig.suptitle("Majority-vote SJM teacher labels", y=0.99, fontsize=14)
    save_figure(fig, "teacher_regime_labels")


if __name__ == "__main__":
    build_majority_vote_figures()
