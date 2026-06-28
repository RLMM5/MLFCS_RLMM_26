import pandas as pd
import matplotlib.pyplot as plt

from article_outputs.common import OUTPUT_DIR, save_figure


TARGET_LABELS = {
    "eq_return_sortino_state_h1": "Equity",
    "bd_bond_canonical_state_h1": "Bond",
}

FAMILY_LABELS = {
    "LOGIT": "Logistic regression",
    "RF": "Random forest",
}


def _ordered_scenarios(df):
    scenarios = []
    for target in TARGET_LABELS:
        for family in ["LOGIT", "RF"]:
            if ((df["target"] == target) & (df["family"] == family)).any():
                scenarios.append((target, family))
    return scenarios


def build_feature_importance_figures():
    imp = pd.read_csv(OUTPUT_DIR / "student_prediction" / "Student Feature Importance.csv")
    imp = imp.loc[imp["target"].isin(TARGET_LABELS)].copy()
    scenarios = _ordered_scenarios(imp)
    if not scenarios:
        raise ValueError("No current feature-importance scenarios found.")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    axes = axes.ravel()
    for ax, (target, family) in zip(axes, scenarios):
        x = imp.loc[(imp["target"] == target) & (imp["family"] == family)].copy()
        block = (
            x.groupby("feature_block", dropna=False)["mean_abs_importance"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
        )
        total = block.sum()
        if total > 0:
            block = block / total
        ax.barh(block.index.astype(str), block.values, color="#386fa4")
        ax.set_xlim(0, max(block.max() * 1.12, 0.01))
        ax.set_title(f"{TARGET_LABELS[target]} - {FAMILY_LABELS.get(family, family)}", loc="left", fontsize=11)
        ax.set_xlabel("Share of plotted importance")
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    fig.suptitle("Feature-block importance", y=0.99, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "feature_block_importance")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = axes.ravel()
    for ax, (target, family) in zip(axes, scenarios):
        x = imp.loc[(imp["target"] == target) & (imp["family"] == family)].copy()
        top = x.sort_values("mean_abs_importance", ascending=False).head(10).sort_values("mean_abs_importance")
        ax.barh(top["feature"].astype(str), top["mean_abs_importance"], color="#7a9e7e")
        ax.set_title(f"{TARGET_LABELS[target]} - {FAMILY_LABELS.get(family, family)}", loc="left", fontsize=11)
        ax.set_xlabel("Mean absolute importance")
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    fig.suptitle("Top individual predictors", y=0.99, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "individual_predictor_importance")


if __name__ == "__main__":
    build_feature_importance_figures()
