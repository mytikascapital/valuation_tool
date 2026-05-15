# Valuation toolkit demo from Nikolas

## App instructions (app.py)

The Streamlit app provides a multiple‑based valuation workflow that converts a chosen trailing‑twelve‑month (TTM) metric into a projected terminal price. You can select the metric that best fits the company’s economics—Revenue, Free Cash Flow, Operating Cash Flow, Operating Income, or Net Income—and model a target exit multiple over a configurable horizon. This enables consistent, comparable valuation outputs across different business models.

Key capabilities of the multiple‑based model:

- **Metric flexibility:** switch among Revenue, FCF, Operating Cash Flow, Operating Income, and Net Income to tailor the analysis to the company’s fundamentals.
- **Share count dynamics:** model dilution or buybacks via a share‑reduction (or negative) rate to reflect capital allocation policy.
- **Owner adjustments:** optionally adjust FCF or OCF by subtracting TTM stock‑based compensation to reflect owner‑oriented cash generation; Net Income can be swapped for owner‑earnings when available.
- **Dividend modeling:** if the company pays dividends, the model can project dividend growth and include cumulative dividends in the terminal value.

After saving an analysis locally, use the **Opportunity cost** tab to compare your saved ideas side‑by‑side and identify the opportunities with the highest expected returns (CAGR).

## Features



## Installation

1. Create and activate a Python environment (recommended)
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

