import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from article_outputs.common import OUTPUT_DIR, save_figure


MODEL_LABELS = {
    "LOGIT": "Logistic regression",
    "RF": "Random forest",
}


def _metric_text(metrics):
    if metrics.empty:
        return "No metric row"
    r = metrics.iloc[0]
    parts = [
        f"AUC: {r.get('roc_auc', np.nan):0.3f}",
        f"Accuracy: {r.get('accuracy', np.nan):0.3f}",
        f"Balanced accuracy: {r.get('balanced_accuracy', np.nan):0.3f}",
        f"Brier: {r.get('brier', np.nan):0.3f}",
        f"Actual good share: {r.get('actual_good_share', np.nan):0.3f}",
    ]
    return "\n".join(parts)


def _plot_one(panel, metrics, target, stem, title):
    target_panel = panel.loc[panel["target"].eq(target)].copy()
    if target_panel.empty:
        raise ValueError(f"No prediction rows for target {target}")

    families = [x for x in ["LOGIT", "RF"] if x in set(target_panel["family"])]
    if not families:
        families = sorted(target_panel["family"].dropna().unique())

    fig, axes = plt.subplots(len(families), 1, figsize=(13.5, 3.6 * len(families)), sharex=True)
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        df = target_panel.loc[target_panel["family"].eq(family)].sort_values("date").copy()
        if "p_seed_std" in df.columns:
            lo = (df["p_mean"] - df["p_seed_std"]).clip(0.0, 1.0)
            hi = (df["p_mean"] + df["p_seed_std"]).clip(0.0, 1.0)
            ax.fill_between(df["date"], lo, hi, color="#adcbe3", alpha=0.45, lw=0)
        ax.plot(df["date"], df["p_mean"], color="#1f4e79", lw=1.25, label="Predicted good-state probability")
        if "y_true" in df.columns:
            realized = df[["date", "y_true"]].dropna()
            ax.step(realized["date"], realized["y_true"], where="post", color="black", lw=0.95, alpha=0.75, label="Realized label")
        ax.axhline(0.5, color="#666666", lw=0.8, ls="--")
        metric_row = metrics.loc[(metrics["target"].eq(target)) & (metrics["family"].eq(family))]
        ax.text(
            0.012,
            0.04,
            _metric_text(metric_row),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.90, boxstyle="round,pad=0.35"),
        )
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel("Probability")
        ax.set_title(MODEL_LABELS.get(family, family), loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Date")
    fig.suptitle(title, y=0.99, fontsize=14)
    save_figure(fig, stem)


def build_prediction_figures():
    panel = pd.read_csv(OUTPUT_DIR / "student_prediction" / "Student Prediction Panel.csv", parse_dates=["date"])
    metrics = pd.read_csv(OUTPUT_DIR / "student_prediction" / "Student Prediction Metrics.csv")

    _plot_one(
        panel,
        metrics,
        target="eq_return_sortino_state_h1",
        stem="equity_prediction_probabilities",
        title="Equity student prediction probabilities",
    )
    _plot_one(
        panel,
        metrics,
        target="bd_bond_canonical_state_h1",
        stem="bond_prediction_probabilities",
        title="Bond student prediction probabilities",
    )


if __name__ == "__main__":
    build_prediction_figures()
