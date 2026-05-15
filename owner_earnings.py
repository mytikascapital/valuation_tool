import yfinance as yf
import pandas as pd


def get_owner_earnings(ticker_symbol: str) -> pd.DataFrame:
    """
    Computes owner-adjusted financial metrics for a given ticker.

    Owner Earnings  = Net Income + D&A
    Owner OCF       = Operating Cash Flow - SBC
    Owner FCF       = Free Cash Flow - SBC  (FCF = OCF + CapEx, CapEx is negative)

    Parameters
    ----------
    ticker_symbol : str
        Yahoo Finance ticker, e.g. "AAPL", "MSFT", "NVDA"
    verbose : bool
        If True, prints a formatted summary. Default True.

    Returns
    -------
    pd.DataFrame
        Columns = fiscal year dates, rows = raw inputs + owner metrics
    """

    t   = yf.Ticker(ticker_symbol)
    cf  = t.cashflow
    inc = t.income_stmt

    # ------------------------------------------------------------------ #
    # Extract raw line items
    # ------------------------------------------------------------------ #

    def safe_get(df: pd.DataFrame, *candidates) -> pd.Series:
        for name in candidates:
            if name in df.index:
                return df.loc[name]
        return pd.Series(0, index=df.columns)

    net_income   = safe_get(inc, "Net Income")
    operating_cf = safe_get(cf,  "Operating Cash Flow", "Cash Flow From Operations")
    capex        = safe_get(cf,  "Capital Expenditure", "Purchase Of PPE")   # negative
    da           = safe_get(cf,  "Depreciation And Amortization",
                                  "Depreciation Amortization Depletion")      # positive
    sbc          = safe_get(cf,  "Stock Based Compensation",
                                  "Share Based Compensation")                  # positive

    # ------------------------------------------------------------------ #
    # Owner Metrics
    # ------------------------------------------------------------------ #

    # Net Income already deducts SBC (income-statement item), so no SBC
    # adjustment here — only add back D&A to recover the non-cash charge.
    owner_earnings = net_income + da

    # OCF adds SBC back as a non-cash add-back → subtract it to reflect
    # the real dilution cost borne by owners.
    owner_ocf = operating_cf - sbc

    # FCF = OCF + CapEx (CapEx is negative in yfinance, so this subtracts it).
    # Then subtract SBC for the same dilution reason as above.
    standard_fcf = operating_cf + capex
    owner_fcf    = standard_fcf - sbc

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    results = pd.DataFrame({
        "Net Income":            net_income,
        "D&A":                   da,
        "SBC":                   sbc,
        "Operating Cash Flow":   operating_cf,
        "CapEx":                 capex,
        "Standard FCF":          standard_fcf,
        "Owner Earnings":        owner_earnings,
        "Owner OCF":             owner_ocf,
        "Owner FCF":             owner_fcf,
    }).T


    return results


# --------------------------------------------------------------------------- #
# Helper: formatted console output
# --------------------------------------------------------------------------- #

def _fmt(value) -> str:
    try:
        v = float(value)
        if abs(v) >= 1e9:
            return f"${v / 1e9:+.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:+.2f}M"
        return f"${v:+.0f}"
    except (TypeError, ValueError):
        return "N/A"



