import pandas as pd

from article_outputs.common import OUTPUT_DIR, TABLE_DIR, ensure_article_dirs, write_latex_table


def _strategy_label(row):
    bits = [
        str(row.get("family", "")),
        str(row.get("allocation_pair", "")),
        f"gamma={row.get('gamma', '')}",
        f"tau={row.get('tau', '')}",
    ]
    return ", ".join(x for x in bits if x and x != "nan")


def _format_summary(df):
    keep = [
        "strategy_label",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "ann_turnover",
        "avg_w_eq",
        "avg_w_bd",
        "avg_w_cash",
    ]
    out = df.copy()
    out["strategy_label"] = out.apply(_strategy_label, axis=1)
    return out[[c for c in keep if c in out.columns]]


def build_allocation_tables():
    ensure_article_dirs()

    summary_path = OUTPUT_DIR / "allocation" / "Allocation Strategy Summary.csv"
    turnover_path = OUTPUT_DIR / "allocation" / "Allocation Turnover Strategy Summary.csv"
    frontier_path = OUTPUT_DIR / "allocation" / "Turnover Performance Frontier.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        best = summary.sort_values("sharpe", ascending=False).head(8)
        write_latex_table(
            _format_summary(best),
            TABLE_DIR / "allocation_summary_focused.tex",
            caption="Allocation strategies with the highest Sharpe ratios.",
            label="tab:allocation_summary_focused",
        )

    if turnover_path.exists():
        turnover = pd.read_csv(turnover_path)
        selected = (
            turnover.sort_values(["sharpe", "ann_turnover"], ascending=[False, True])
            .head(10)
            .reset_index(drop=True)
        )
        write_latex_table(
            _format_summary(selected),
            TABLE_DIR / "allocation_selected_strategies_appendix.tex",
            caption="Selected allocation strategies from the turnover-controlled specification.",
            label="tab:allocation_selected_strategies_appendix",
        )

    if frontier_path.exists():
        frontier = pd.read_csv(frontier_path)
        write_latex_table(
            _format_summary(frontier),
            TABLE_DIR / "allocation_sweep_cost_turnover_controlled.tex",
            caption="Turnover-controlled allocation frontier.",
            label="tab:allocation_sweep_cost_turnover_controlled",
        )


if __name__ == "__main__":
    build_allocation_tables()
