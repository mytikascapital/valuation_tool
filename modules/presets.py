from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import re

import pandas as pd
import requests

try:
	import cloudscraper
except Exception:
	cloudscraper = None


ANALYSES_DIR = Path("analyses")
WATCHLIST_CACHE_DIR = ANALYSES_DIR / ".cache" / "watchlists"
WATCHLIST_CACHE_MAX_AGE_DAYS = 7

DATAROMA_PORTFOLIO_URLS: dict[str, str] = {
	"Grand portofolio 1": "https://www.dataroma.com/m/g/portfolio.php?pct=0&o=p",
	"Grand portofolio 2": "https://www.dataroma.com/m/g/portfolio.php?pct=0&o=c",
}


# Keep user/custom watchlists intact.
CUSTOM_WATCHLIST_PRESETS: dict[str, list[str]] = {
	"Durable Compounders": [
		"APH", "AXON", "BMI", "ECL", "GE", "GEV", "HEI", "IDXX", "MSI",
		"RBC", "TDG", "TDY", "VLTO", "WST", "MCO", "MSCI", "SPGI", "WM",
	]
}




def _normalize_ticker(ticker: str) -> str:
	t = str(ticker or "").strip().upper()
	if not t:
		return ""
	t = t.replace(".", "-")
	t = re.sub(r"[^A-Z0-9\-]", "", t)
	return t


def _dedupe_and_sort_tickers(tickers: list[str]) -> list[str]:
	cleaned = [_normalize_ticker(t) for t in tickers]
	return sorted({t for t in cleaned if t and t not in {"NAN", "NONE"}})


def _cache_file_path(index_name: str) -> Path:
	WATCHLIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	safe = re.sub(r"[^A-Za-z0-9_-]+", "_", index_name.strip().lower())
	return WATCHLIST_CACHE_DIR / f"{safe}.csv"


def _load_cached_constituents(index_name: str, allow_stale: bool = False) -> list[str] | None:
	path = _cache_file_path(index_name)
	if not path.exists():
		return None

	max_age = timedelta(days=WATCHLIST_CACHE_MAX_AGE_DAYS)
	modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
	if (not allow_stale) and (datetime.now(timezone.utc) - modified_at > max_age):
		return None

	try:
		df = pd.read_csv(path)
		if "Ticker" not in df.columns:
			return None
		vals = _dedupe_and_sort_tickers(df["Ticker"].astype(str).tolist())
		return vals if vals else None
	except Exception:
		return None


def _save_cached_constituents(index_name: str, tickers: list[str]) -> None:
	path = _cache_file_path(index_name)
	clean = _dedupe_and_sort_tickers(tickers)
	if not clean:
		return
	pd.DataFrame({"Ticker": clean}).to_csv(path, index=False)


def _fetch_sp500_tickers() -> list[str]:
	# Reliable public source: DataHub mirror of S&P 500 constituents.
	# Source dataset: datasets/s-and-p-500-companies
	url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
	df = pd.read_csv(url)
	if "Symbol" not in df.columns:
		return []
	return _dedupe_and_sort_tickers(df["Symbol"].astype(str).tolist())


def _fetch_nasdaq100_tickers() -> list[str]:
	# Reliable public source: official Nasdaq API endpoint.
	url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
	headers = {
		"User-Agent": "Mozilla/5.0",
		"Accept": "application/json, text/plain, */*",
		"Referer": "https://www.nasdaq.com/",
	}
	response = requests.get(url, timeout=20, headers=headers)
	response.raise_for_status()
	payload = response.json() if response.content else {}

	rows = (
		payload.get("data", {})
		.get("data", {})
		.get("rows", [])
	)
	if not isinstance(rows, list):
		return []

	tickers = [str(row.get("symbol", "")).strip() for row in rows if isinstance(row, dict)]
	return _dedupe_and_sort_tickers(tickers)


def _fetch_ishares_holdings_tickers(csv_url: str) -> list[str]:
	# Reliable public source proxy for index constituents: iShares ETF holdings CSV (BlackRock).
	# QUAL ~= MSCI USA Quality Index, VLUE ~= MSCI USA Value Index.
	headers = {
		"User-Agent": "Mozilla/5.0",
		"Accept": "text/csv,text/plain,*/*",
	}
	response = requests.get(csv_url, timeout=20, headers=headers)
	response.raise_for_status()

	text = response.text or ""
	if not text:
		return []

	lines = text.splitlines()
	header_idx = None
	for idx, line in enumerate(lines):
		if "Ticker" in line and "Name" in line:
			header_idx = idx
			break
	if header_idx is None:
		return []

	csv_payload = "\n".join(lines[header_idx:])
	df = pd.read_csv(StringIO(csv_payload))
	if "Ticker" not in df.columns:
		return []

	return _dedupe_and_sort_tickers(df["Ticker"].astype(str).tolist())


def _fetch_msci_usa_quality_index_tickers() -> list[str]:
	url = (
		"https://www.ishares.com/us/products/256101/"
		"ishares-msci-usa-quality-factor-etf/1467271812596.ajax"
		"?fileType=csv&fileName=QUAL_holdings&dataType=fund"
	)
	return _fetch_ishares_holdings_tickers(url)


def _fetch_msci_usa_value_index_tickers() -> list[str]:
	url = (
		"https://www.ishares.com/us/products/239725/"
		"ishares-msci-usa-value-factor-etf/1467271812596.ajax"
		"?fileType=csv&fileName=VLUE_holdings&dataType=fund"
	)
	return _fetch_ishares_holdings_tickers(url)


def _extract_dataroma_tickers_from_html(html: str) -> list[str]:
	if not html:
		return []

	# Dataroma stock links usually look like: /m/stock.php?sym=MSFT
	matches = re.findall(r"/m/stock\.php\?sym=([A-Za-z0-9\.\-]+)", html, flags=re.IGNORECASE)
	if matches:
		return _dedupe_and_sort_tickers(matches)

	# Fallback: parse tables and infer ticker-like values from first column.
	try:
		tables = pd.read_html(StringIO(html))
	except Exception:
		tables = []

	tickers: list[str] = []
	for table in tables:
		if table is None or table.empty:
			continue
		first_col = table.columns[0]
		vals = table[first_col].astype(str).tolist()
		for raw in vals:
			candidate = str(raw).strip().upper()
			if re.match(r"^[A-Z\.\-]{1,8}$", candidate):
				tickers.append(candidate)

	return _dedupe_and_sort_tickers(tickers)


def _fetch_dataroma_grand_portfolio_tickers(url: str) -> list[str]:
	headers = {
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
	}

	# Attempt 1: cloudscraper (when installed) for anti-bot bypass.
	if cloudscraper is not None:
		try:
			scraper = cloudscraper.create_scraper(
				browser={
					"browser": "chrome",
					"platform": "windows",
					"mobile": False,
				}
			)
			resp = scraper.get(url, timeout=20, headers=headers)
			resp.raise_for_status()
			tickers = _extract_dataroma_tickers_from_html(resp.text)
			if tickers:
				return tickers
		except Exception:
			pass

	# Attempt 2: plain requests.
	resp = requests.get(url, timeout=20, headers=headers)
	resp.raise_for_status()
	return _extract_dataroma_tickers_from_html(resp.text)


def _load_or_fetch_index(index_name: str, fetch_fn) -> list[str]:
	cached = _load_cached_constituents(index_name)
	if cached:
		return cached

	try:
		fetched = _dedupe_and_sort_tickers(fetch_fn())
		if fetched:
			_save_cached_constituents(index_name, fetched)
			return fetched
	except Exception:
		pass

	# If fresh fetch fails, fall back to stale local cache.
	stale_cached = _load_cached_constituents(index_name, allow_stale=True)
	if stale_cached:
		return stale_cached

	return []


def get_watchlist_preset() -> dict[str, list[str]]:
	presets: dict[str, list[str]] = {
		"S&P 500": _load_or_fetch_index("S&P 500", _fetch_sp500_tickers),
		"Nasdaq 100": _load_or_fetch_index("Nasdaq 100", _fetch_nasdaq100_tickers),
		"MSCI USA Quality": _load_or_fetch_index(
			"MSCI USA Quality Index",
			_fetch_msci_usa_quality_index_tickers,
		),
		"MSCI USA Value": _load_or_fetch_index(
			"MSCI USA Value Index",
			_fetch_msci_usa_value_index_tickers,
		),
		"Grand portofolio 1": _load_or_fetch_index(
			"Grand portofolio 1",
			lambda: _fetch_dataroma_grand_portfolio_tickers(DATAROMA_PORTFOLIO_URLS["Grand portofolio 1"]),
		),
		"Grand portofolio 2": _load_or_fetch_index(
			"Grand portofolio 2",
			lambda: _fetch_dataroma_grand_portfolio_tickers(DATAROMA_PORTFOLIO_URLS["Grand portofolio 2"]),
		),
	}

	# Keep user-maintained/custom lists untouched.
	presets.update(CUSTOM_WATCHLIST_PRESETS)
	return presets
