# MLFCS_RLMM_26

`Data/`contains the data analysis pipeline of Roberto. There is a notebooke `analyse_series_workbench.ipynb` that is the main interface for analyzing and curating FRED series. It uses the local FRED-MD dataset as a reference and saves your analyzed series in a structured way for future use.
The local FRED-MD files live in `Data/MD-dataset/`. When you save a series the notebook writes to `Data/analyzed/<series_id>/`.

## Getting started

You need [uv](https://github.com/astral-sh/uv) and Python 3.12+.

```bash
# create and activate the virtual environment
uv venv .venv
source .venv/bin/activate        # macOS / Linux
.\.venv\Scripts\Activate.ps1    # Windows PowerShell

# install dependencies
uv pip install -r requirements.txt

# register the Jupyter kernel so VS Code can find it
python -m ipykernel install --user --name mlfcs-rlmm-26 --display-name "Python (mlfcs-rlmm-26)"
```

Then create a `.env` file in the repo root with your FRED API key (get one free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)):

```env
FRED_API_KEY=your_key_here
```

Open the notebook in VS Code, select the `mlfcs-rlmm-26` kernel, and run all cells once from top to bottom to initialise everything.

## Typical workflow

The main run cell near the bottom of the notebook is self-contained — just set `SERIES_ID` to any FRED ticker and execute it. It will:

1. Fetch the series, its metadata, and vintage dates from the FRED API.
2. Plot the level, first difference, z-score of the first difference, and a variance proxy so you can see trend, volatility bursts, and outliers at a glance.
3. Show a concise metadata table (date coverage, missing share, revision count).
4. Run ADF, KPSS, and ARCH tests on both the level and the first difference.
5. Recommend which FRED-MD transformation code makes the series stationary.
6. Score every FRED-MD series by text similarity and detrended correlation so you can see where it sits in the existing dataset.

If you decide the series is worth keeping, uncomment the save block at the bottom of the run cell, fill in a comment and your `useful` label, and run it again. This writes three files to `Data/analyzed/<series_id>/`:

- `_observations.csv` — raw FRED observations
- `_features.csv` — engineered transforms (diffs, log-diffs, z-scores, etc.)
- `_analysis.json` — a compact summary with the suggested transform, top FRED-MD matches, and your notes

To see everything you've reviewed so far, call `show_include_table()` in any cell. It splits your saved series into an include table (`useful = "useful"`) and an exclude/unsure table.
