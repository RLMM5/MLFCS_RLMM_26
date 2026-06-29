from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figs"
TABLE_DIR = PAPER_DIR / "tabs"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_article_dirs():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem, formats=("jpg", "png", "pdf")):
    ensure_article_dirs()
    for ext in formats:
        path = FIGURE_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"jpg", "jpeg"}:
            kwargs["dpi"] = 220
        fig.savefig(path, **kwargs)
    plt.close(fig)


def latex_escape(x):
    return (
        str(x)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_latex_table(df, path, caption=None, label=None, float_format="{:.3f}"):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else float_format.format(x))
    latex = out.to_latex(index=False, escape=True, caption=caption, label=label)
    Path(path).write_text(latex, encoding="utf-8")
