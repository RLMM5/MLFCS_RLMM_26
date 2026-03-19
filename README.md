# MLFCS_RLMM_26

Quick workspace for manual FRED series analysis in the notebook:
`Data/analyse_series_workbench.ipynb`.

## Setup

Requirements:
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

```bash
# create env
uv venv .venv

# activate env
source .venv/bin/activate      # macOS / Linux
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# install deps
uv pip install -r requirements.txt

# register kernel
python -m ipykernel install --user --name mlfcs-rlmm-26 --display-name "Python (mlfcs-rlmm-26)"
```

Create `.env` in repo root:

```env
FRED_API_KEY=your_key_here
```

Then open the notebook in VS Code, select `Python (mlfcs-rlmm-26)`, and run setup cells once.

## Workflow (Quick)

1. Open a template run cell near the bottom.
2. Set `SERIES_ID` and `CATEGORY`.
3. Run the cell to fetch data, show diagnostics, and review tcode candidates.
4. Optional: run similar-series search.
5. Optional: append/update the markdown overview.

## Outputs

- Category overview markdown: `Data/analyzed/<category>.md`
- Saved plots: `Data/analyzed/images/`

Each save updates (upserts) the row for that `SERIES_ID` in the category markdown file.
