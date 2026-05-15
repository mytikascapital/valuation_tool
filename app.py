import math
import time
import json
import os
import io
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
try:
	import plotly.express as px
	import plotly.graph_objects as go
	import plotly.io as pio
	_PLOTLY_IMPORT_ERROR = None
except Exception as exc:
	px = None
	go = None
	pio = None
	_PLOTLY_IMPORT_ERROR = exc
import requests
import streamlit as st
import yfinance as yf
from owner_earnings import get_owner_earnings
from modules.opportunity import render_opportunity_cost_tab
from modules.insiders import render_insider_buying_tracker_tab

# local imports
from modules.utils import fmt_money
# import _load_saved_portfolio_sources from portfolio_viz.py
from modules.portofolio_viz import render_portfolio_visualization_tab, _load_saved_portfolio_sources

# load dict variable
from modules.presets import get_watchlist_preset 

try:
	from yfinance.exceptions import YFRateLimitError
except Exception:
	YFRateLimitError = Exception


st.set_page_config(page_title="Simple Invest", layout="wide")

if _PLOTLY_IMPORT_ERROR is not None:
	st.error(
		"Plotly failed to import. Ensure `plotly` is listed in requirements.txt "
		"and redeploy. Error: "
		+ str(_PLOTLY_IMPORT_ERROR)
	)
	st.stop()


@st.cache_data(show_spinner=False, ttl=21600)
def _get_watchlist_presets() -> dict[str, list[str]]:
	try:
		return get_watchlist_preset()
	except Exception:
		return {}

ANALYSES_DIR = Path("analyses")

DATA_CACHE_DIR = ANALYSES_DIR / ".cache"
DATA_CACHE_MAX_AGE_SECONDS = 6 * 3600
DATA_PROVIDERS = ["Yahoo Finance", "Finnhub"]


METRIC_LABEL_CANDIDATES = {
	"FCF (Free Cash Flow)": [
		"Free Cash Flow",
		"FreeCashFlow",
	],
	"Revenue": [
		"Total Revenue",
		"Revenue",
		"Operating Revenue",
		"TotalRevenue",
	],
	"Operating Cash Flow": [
		"Operating Cash Flow",
		"Total Cash From Operating Activities",
		"Cash Flow From Continuing Operating Activities",
	],
	"Operating Income": ["Operating Income", "EBIT", "OperatingIncome"],
	"Earnings (Net Income)": [
		"Net Income",
		"Net Income Common Stockholders",
		"NetIncome",
	],
}


@dataclass
class StockSnapshot:
	ticker: str
	price: float
	shares_outstanding: float
	metric_ttm: float
	metric_name: str
	current_ttm_multiple: float | None = None


def _analysis_csv_path(ticker: str) -> Path:
	ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
	return ANALYSES_DIR / f"{ticker.strip().upper()}.csv"


def _data_cache_csv_path(ticker: str) -> Path:
	DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	return DATA_CACHE_DIR / f"{ticker.strip().upper()}.csv"


def load_data_cache_df(ticker: str) -> pd.DataFrame | None:
	path = _data_cache_csv_path(ticker)
	if not path.exists():
		return None
	try:
		df = pd.read_csv(path)
		return df if not df.empty else None
	except Exception:
		return None


def load_analysis_df(ticker: str) -> pd.DataFrame | None:
	path = _analysis_csv_path(ticker)
	if not path.exists():
		return None
	try:
		df = pd.read_csv(path)
		return df if not df.empty else None
	except Exception:
		return None


def _analysis_get(df: pd.DataFrame | None, section: str, key: str, metric: str | None = None) -> str | None:
	if df is None or df.empty:
		return None
	mask = (df.get("section") == section) & (df.get("key") == key)
	if metric is not None:
		mask = mask & (df.get("metric") == metric)
	rows = df.loc[mask]
	if rows.empty:
		return None
	return str(rows.iloc[-1]["value"])


def _parse_float(value: str | None, default: float) -> float:
	try:
		if value is None or value == "":
			return default
		return float(value)
	except Exception:
		return default


def _parse_bool(value: str | None, default: bool) -> bool:
	if value is None:
		return default
	return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_iso_datetime(value: str | None) -> datetime | None:
	if not value:
		return None
	try:
		parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
		if parsed.tzinfo is None:
			parsed = parsed.replace(tzinfo=timezone.utc)
		return parsed
	except Exception:
		return None


def _is_data_cache_fresh(
	df: pd.DataFrame | None,
	provider: str,
	max_age_seconds: int = DATA_CACHE_MAX_AGE_SECONDS,
) -> bool:
	if df is None or df.empty:
		return False
	cached_provider = _analysis_get(df, "meta", "provider")
	if not cached_provider or str(cached_provider).strip() != provider:
		return False
	fetched_at = _parse_iso_datetime(_analysis_get(df, "meta", "fetched_at"))
	if fetched_at is None:
		return False
	age_seconds = (datetime.now(timezone.utc) - fetched_at).total_seconds()
	return age_seconds <= float(max_age_seconds)


def _save_data_cache_csv(
	ticker: str,
	provider: str,
	snapshots: dict[str, StockSnapshot],
	company_cache: dict[str, float | None],
) -> Path:
	records: list[dict[str, str]] = []

	def add(section: str, key: str, value, metric: str = "") -> None:
		records.append(
			{
				"section": section,
				"metric": metric,
				"key": key,
				"value": "" if value is None else str(value),
			}
		)

	add("meta", "ticker", ticker)
	add("meta", "provider", provider)
	add("meta", "fetched_at", datetime.now(timezone.utc).isoformat())

	for metric_name, snap in snapshots.items():
		add("snapshot", "ticker", snap.ticker, metric=metric_name)
		add("snapshot", "price", snap.price, metric=metric_name)
		add("snapshot", "shares_outstanding", snap.shares_outstanding, metric=metric_name)
		add("snapshot", "metric_ttm", snap.metric_ttm, metric=metric_name)
		add("snapshot", "current_ttm_multiple", snap.current_ttm_multiple, metric=metric_name)

	for key, value in company_cache.items():
		add("company_data", key, value)

	df = pd.DataFrame.from_records(records)
	path = _data_cache_csv_path(ticker)
	df.to_csv(path, index=False)
	return path


def _provider_load_snapshot(ticker: str, metric_name: str, provider: str) -> StockSnapshot:
	if provider == "Finnhub":
		return load_stock_snapshot_finnhub(ticker=ticker, metric_name=metric_name)
	return load_stock_snapshot(ticker=ticker, metric_name=metric_name)


def _metric_previous_year_cache_key(metric_name: str) -> str:
	key_map = {
		"FCF (Free Cash Flow)": "previous_full_year_fcf",
		"Operating Cash Flow": "previous_full_year_operating_cash_flow",
		"Revenue": "previous_full_year_revenue",
		"Operating Income": "previous_full_year_operating_income",
		"Earnings (Net Income)": "previous_full_year_earnings_net_income",
	}
	return key_map.get(metric_name, "")


def _get_previous_full_year_metric_from_cache(data_cache_df: pd.DataFrame | None, metric_name: str) -> float | None:
	cache_key = _metric_previous_year_cache_key(metric_name)
	if not cache_key:
		return None
	value = _parse_float(_analysis_get(data_cache_df, "company_data", cache_key), np.nan)
	if np.isnan(value):
		return None
	return float(value)


@st.cache_data(show_spinner=False, ttl=1800)
def load_previous_full_year_metric_finnhub(ticker: str, metric_name: str) -> float | None:
	ticker = ticker.strip().upper()
	annual_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "annual"})

	cf_lines = _finnhub_extract_statement_lines(annual_reported, "cf")
	ic_lines = _finnhub_extract_statement_lines(annual_reported, "ic")
	latest_cf = cf_lines[0] if cf_lines else []
	latest_ic = ic_lines[0] if ic_lines else []

	if metric_name == "FCF (Free Cash Flow)":
		direct_fcf = _finnhub_find_line_value(latest_cf, ["free cash flow", "freecashflow"])
		if direct_fcf is not None:
			return float(direct_fcf)

		op_cf = _finnhub_find_line_value(
			latest_cf,
			[
				"operating cash flow",
				"net cash from operations",
				"net cash provided by operating activities",
				"net cash provided by used in operating activities",
			],
		)
		capex = _finnhub_find_line_value(
			latest_cf,
			[
				"capital expenditures",
				"capital expenditure",
				"payments to acquire property",
				"payments to acquire property plant and equipment",
			],
		)
		if op_cf is not None and capex is not None:
			return _compute_fcf(op_cf, capex)
		return None

	if metric_name == "Operating Cash Flow":
		value = _finnhub_find_line_value(
			latest_cf,
			[
				"operating cash flow",
				"net cash from operations",
				"net cash provided by operating activities",
				"net cash provided by used in operating activities",
			],
		)
		return None if value is None else float(value)

	if metric_name == "Revenue":
		value = _finnhub_find_line_value(latest_ic, ["revenue", "sales revenue net", "total revenue"])
		return None if value is None else float(value)

	if metric_name == "Operating Income":
		value = _finnhub_find_line_value(latest_ic, ["operating income", "income from operations", "operating income loss"])
		return None if value is None else float(value)

	if metric_name == "Earnings (Net Income)":
		value = _finnhub_find_line_value(latest_ic, ["net income", "profit loss", "net income loss"])
		return None if value is None else float(value)

	return None


@st.cache_data(show_spinner=False, ttl=1800)
def load_company_capital_structure_finnhub(ticker: str) -> tuple[float, float]:
	ticker = ticker.strip().upper()
	metrics_payload = _finnhub_get("stock/metric", {"symbol": ticker, "metric": "all"})
	metric_map = metrics_payload.get("metric") if isinstance(metrics_payload, dict) else {}
	metric_map = metric_map if isinstance(metric_map, dict) else {}

	cash = _safe_float(metric_map.get("totalCash"), np.nan)
	debt = _safe_float(metric_map.get("totalDebt"), np.nan)
	if np.isnan(debt):
		debt = _safe_float(metric_map.get("netDebt"), np.nan)

	annual_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "annual"})
	bs_lines = _finnhub_extract_statement_lines(annual_reported, "bs")
	latest_bs = bs_lines[0] if bs_lines else []

	if np.isnan(cash):
		cash_candidate = _finnhub_find_line_value(
			latest_bs,
			[
				"cash and cash equivalents",
				"cash and short term investments",
				"cash cash equivalents and short term investments",
			],
		)
		if cash_candidate is not None:
			cash = float(cash_candidate)

	if np.isnan(debt):
		debt_candidate = _finnhub_find_line_value(
			latest_bs,
			[
				"total debt",
				"long term debt",
				"longtermdebt",
			],
		)
		if debt_candidate is not None:
			debt = float(debt_candidate)

	return float(0.0 if np.isnan(cash) else cash), float(0.0 if np.isnan(debt) else debt)


@st.cache_data(show_spinner=False, ttl=1800)
def load_fcf_previous_full_year_finnhub(ticker: str) -> float | None:
	ticker = ticker.strip().upper()
	annual_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "annual"})
	cf_lines = _finnhub_extract_statement_lines(annual_reported, "cf")
	if not cf_lines:
		return None

	latest_cf = cf_lines[0]
	direct_fcf = _finnhub_find_line_value(latest_cf, ["free cash flow", "freecashflow"])
	if direct_fcf is not None:
		return float(direct_fcf)

	op_cf = _finnhub_find_line_value(
		latest_cf,
		[
			"operating cash flow",
			"net cash from operations",
			"net cash provided by operating activities",
			"net cash provided by used in operating activities",
		],
	)
	capex = _finnhub_find_line_value(
		latest_cf,
		[
			"capital expenditures",
			"capital expenditure",
			"payments to acquire property",
			"payments to acquire property plant and equipment",
		],
	)
	if op_cf is not None and capex is not None:
		return _compute_fcf(op_cf, capex)

	return None


@st.cache_data(show_spinner=False, ttl=1800)
def load_sbc_ttm_finnhub(ticker: str) -> float | None:
	ticker = ticker.strip().upper()
	quarterly_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "quarterly"})
	annual_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "annual"})

	sbc_ttm = _finnhub_ttm_from_cumulative_reports(
		quarterly_payload=quarterly_reported,
		annual_payload=annual_reported,
		section_key="cf",
		include_terms=[
			"stock based compensation",
			"share based compensation",
			"sharebasedcompensation",
		],
	)
	if sbc_ttm is None:
		return None
	return abs(float(sbc_ttm))


def _build_data_cache_from_provider(
	ticker: str,
	provider: str,
) -> tuple[dict[str, StockSnapshot], dict[str, float | None]]:
	ticker = ticker.strip().upper()
	snapshots: dict[str, StockSnapshot] = {}

	for metric_name in METRIC_LABEL_CANDIDATES.keys():
		try:
			snapshots[metric_name] = _provider_load_snapshot(
				ticker=ticker,
				metric_name=metric_name,
				provider=provider,
			)
		except Exception:
			continue

	if not snapshots:
		raise ValueError(f"Could not load snapshots for {ticker} from {provider}.")

	if provider == "Finnhub":
		cash, total_debt = load_company_capital_structure_finnhub(ticker)
		previous_full_year_fcf = load_fcf_previous_full_year_finnhub(ticker)
		ttm_sbc = load_sbc_ttm_finnhub(ticker)
	else:
		cash, total_debt = load_company_capital_structure(ticker)
		previous_full_year_fcf = load_fcf_previous_full_year(ticker)
		ttm_sbc = load_sbc_ttm(ticker)

	previous_year_metrics: dict[str, float | None] = {}
	for metric_name in METRIC_LABEL_CANDIDATES.keys():
		cache_key = _metric_previous_year_cache_key(metric_name)
		if not cache_key:
			continue
		try:
			if provider == "Finnhub":
				previous_year_metrics[cache_key] = load_previous_full_year_metric_finnhub(ticker, metric_name)
			else:
				previous_year_metrics[cache_key] = load_previous_full_year_metric_yahoo(ticker, metric_name)
		except Exception:
			previous_year_metrics[cache_key] = None

	company_cache = {
		"cash": cash,
		"total_debt": total_debt,
		"previous_full_year_fcf": previous_full_year_fcf,
		"ttm_sbc": ttm_sbc,
	}
	company_cache.update(previous_year_metrics)
	return snapshots, company_cache


def get_or_load_data_cache(
	ticker: str,
	provider: str,
	force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
	ticker = ticker.strip().upper()
	cached_df = load_data_cache_df(ticker)

	if not force_refresh and _is_data_cache_fresh(cached_df, provider=provider):
		return cached_df, "local-cache"

	try:
		snapshots, company_cache = _build_data_cache_from_provider(ticker=ticker, provider=provider)
		_save_data_cache_csv(
			ticker=ticker,
			provider=provider,
			snapshots=snapshots,
			company_cache=company_cache,
		)
		refreshed_df = load_data_cache_df(ticker)
		if refreshed_df is None:
			raise ValueError("Data cache write failed unexpectedly.")
		return refreshed_df, "provider-api"
	except Exception:
		if cached_df is not None and not cached_df.empty:
			return cached_df, "stale-cache"
		raise


def collect_snapshot_cache_from_data_cache(
	data_cache_df: pd.DataFrame | None,
	primary_snapshot: StockSnapshot,
) -> dict[str, StockSnapshot]:
	cache: dict[str, StockSnapshot] = {}
	for metric_name in METRIC_LABEL_CANDIDATES.keys():
		snap = get_saved_snapshot(data_cache_df, metric_name)
		if snap is not None:
			cache[metric_name] = snap

	if primary_snapshot.metric_name not in cache:
		cache[primary_snapshot.metric_name] = primary_snapshot

	return cache


def get_saved_snapshot(df: pd.DataFrame | None, metric_name: str) -> StockSnapshot | None:
	if df is None or df.empty:
		return None
	ticker = _analysis_get(df, "snapshot", "ticker", metric_name)
	price = _parse_float(_analysis_get(df, "snapshot", "price", metric_name), np.nan)
	shares = _parse_float(_analysis_get(df, "snapshot", "shares_outstanding", metric_name), np.nan)
	metric_ttm = _parse_float(_analysis_get(df, "snapshot", "metric_ttm", metric_name), np.nan)
	multiple_raw = _analysis_get(df, "snapshot", "current_ttm_multiple", metric_name)
	multiple = None
	if multiple_raw not in {None, "", "None", "nan"}:
		multiple = _parse_float(multiple_raw, np.nan)
		if np.isnan(multiple):
			multiple = None

	if not ticker or np.isnan(price) or np.isnan(shares) or np.isnan(metric_ttm):
		return None

	return StockSnapshot(
		ticker=ticker,
		price=float(price),
		shares_outstanding=float(shares),
		metric_ttm=float(metric_ttm),
		metric_name=metric_name,
		current_ttm_multiple=multiple,
	)


def hydrate_state_from_saved_analysis(ticker: str, df: pd.DataFrame | None) -> None:
	if st.session_state.get("analysis_loaded_for_ticker") == ticker:
		return

	# Defaults first
	st.session_state["valuation_desired_return_pct"] = _parse_float(
		_analysis_get(df, "valuation", "desired_return_pct"), 15.0
	)
	st.session_state["valuation_metric_name"] = _analysis_get(df, "valuation", "metric_name") or "FCF (Free Cash Flow)"
	st.session_state["valuation_owner_adjustments"] = _parse_bool(
		_analysis_get(df, "valuation", "owner_adjustments"), False
	)
	st.session_state["valuation_years"] = int(_parse_float(_analysis_get(df, "valuation", "years"), 5.0))
	st.session_state["valuation_use_custom_growth"] = _parse_bool(
		_analysis_get(df, "valuation", "use_custom_growth"), False
	)
	st.session_state["valuation_growth_rate_pct"] = _parse_float(
		_analysis_get(df, "valuation", "growth_rate_pct"), 10.0
	)
	st.session_state["valuation_starting_metric_source"] = (
		_analysis_get(df, "valuation", "starting_metric_source") or "TTM"
	)
	st.session_state["valuation_starting_metric_billions"] = _parse_float(
		_analysis_get(df, "valuation", "starting_metric_billions"), 0.0
	)
	st.session_state["valuation_dividend_growth_rate_pct"] = _parse_float(
		_analysis_get(df, "valuation", "dividend_growth_rate_pct"), 0.0
	)
	st.session_state["valuation_buyback_rate_pct"] = _parse_float(
		_analysis_get(df, "valuation", "buyback_rate_pct"), 1.0
	)
	st.session_state["valuation_exit_multiple"] = _parse_float(
		_analysis_get(df, "valuation", "exit_multiple"), 18.0
	)

	growth_rates_json = _analysis_get(df, "valuation", "growth_rates_json")
	if growth_rates_json:
		try:
			rates = json.loads(growth_rates_json)
			if isinstance(rates, list):
				for i, rate in enumerate(rates, start=1):
					st.session_state[f"valuation_custom_growth_{i}"] = float(rate)
		except Exception:
			pass

	st.session_state["reverse_projection_years"] = int(
		_parse_float(_analysis_get(df, "reverse", "projection_years"), 10.0)
	)
	st.session_state["reverse_starting_fcf_source"] = (
		_analysis_get(df, "reverse", "starting_fcf_source") or "TTM"
	)
	st.session_state["reverse_apply_sbc_adjustment"] = _parse_bool(
		_analysis_get(df, "reverse", "apply_sbc_adjustment"), False
	)
	st.session_state["reverse_starting_fcf_billions"] = _parse_float(
		_analysis_get(df, "reverse", "starting_fcf_billions"), 0.0
	)
	st.session_state["reverse_explicit_growth_rate_pct"] = _parse_float(
		_analysis_get(df, "reverse", "explicit_growth_rate_pct"), 8.0
	)
	st.session_state["reverse_terminal_growth_rate_pct"] = _parse_float(
		_analysis_get(df, "reverse", "terminal_growth_rate_pct"), 3.0
	)
	st.session_state["reverse_discount_rate_pct"] = _parse_float(
		_analysis_get(df, "reverse", "discount_rate_pct"), 9.0
	)

	st.session_state["analysis_loaded_for_ticker"] = ticker


def save_analysis_csv(
	ticker: str,
	valuation_params: dict,
	reverse_params: dict | None,
	snapshot_cache: dict[str, StockSnapshot],
	company_cache: dict[str, float | None],
) -> Path:
	records: list[dict[str, str]] = []

	def add(section: str, key: str, value, metric: str = "") -> None:
		records.append(
			{
				"section": section,
				"metric": metric,
				"key": key,
				"value": "" if value is None else str(value),
			}
		)

	add("meta", "ticker", ticker)
	add("meta", "saved_at", pd.Timestamp.utcnow().isoformat())

	add("valuation", "desired_return_pct", st.session_state.get("valuation_desired_return_pct", 15.0))
	add("valuation", "metric_name", st.session_state.get("valuation_metric_name", "FCF (Free Cash Flow)"))
	add("valuation", "owner_adjustments", st.session_state.get("valuation_owner_adjustments", False))
	add("valuation", "years", st.session_state.get("valuation_years", 5))
	add("valuation", "use_custom_growth", st.session_state.get("valuation_use_custom_growth", False))
	add("valuation", "growth_rate_pct", st.session_state.get("valuation_growth_rate_pct", 10.0))
	add("valuation", "starting_metric_source", st.session_state.get("valuation_starting_metric_source", "TTM"))
	add("valuation", "starting_metric_billions", st.session_state.get("valuation_starting_metric_billions", 0.0))
	add("valuation", "dividend_growth_rate_pct", st.session_state.get("valuation_dividend_growth_rate_pct", 0.0))
	add("valuation", "buyback_rate_pct", st.session_state.get("valuation_buyback_rate_pct", 1.0))
	add("valuation", "exit_multiple", valuation_params.get("exit_multiple"))
	growth_rates_pct = [round(float(g) * 100, 6) for g in valuation_params.get("growth_rates", [])]
	add("valuation", "growth_rates_json", json.dumps(growth_rates_pct))

	if reverse_params is not None:
		add("reverse", "projection_years", reverse_params.get("projection_years"))
		add("reverse", "starting_fcf_source", reverse_params.get("starting_fcf_source"))
		add("reverse", "apply_sbc_adjustment", reverse_params.get("apply_sbc_adjustment"))
		add("reverse", "starting_fcf_billions", reverse_params.get("starting_fcf_billions"))
		add("reverse", "explicit_growth_rate_pct", reverse_params.get("explicit_growth_rate_pct"))
		add("reverse", "terminal_growth_rate_pct", reverse_params.get("terminal_growth_rate_pct"))
		add("reverse", "discount_rate_pct", reverse_params.get("discount_rate_pct"))

	for metric_name, snap in snapshot_cache.items():
		add("snapshot", "ticker", snap.ticker, metric=metric_name)
		add("snapshot", "price", snap.price, metric=metric_name)
		add("snapshot", "shares_outstanding", snap.shares_outstanding, metric=metric_name)
		add("snapshot", "metric_ttm", snap.metric_ttm, metric=metric_name)
		add("snapshot", "current_ttm_multiple", snap.current_ttm_multiple, metric=metric_name)

	for key, value in company_cache.items():
		add("company_data", key, value)

	df = pd.DataFrame.from_records(records)
	path = _analysis_csv_path(ticker)
	df.to_csv(path, index=False)
	return path


def collect_snapshot_cache(
	ticker: str,
	primary_snapshot: StockSnapshot,
	saved_df: pd.DataFrame | None,
) -> dict[str, StockSnapshot]:
	cache: dict[str, StockSnapshot] = {primary_snapshot.metric_name: primary_snapshot}

	for metric_name in METRIC_LABEL_CANDIDATES.keys():
		if metric_name in cache:
			continue
		try:
			cache[metric_name] = load_stock_snapshot(ticker=ticker, metric_name=metric_name)
		except Exception:
			saved_snapshot = get_saved_snapshot(saved_df, metric_name)
			if saved_snapshot is not None:
				cache[metric_name] = saved_snapshot

	return cache



@st.cache_data(show_spinner=False, ttl=900)
def _compute_watchlist_indicators(
	tickers: tuple[str, ...],
	sma_period: int,
	rsi_period: int = 14,
) -> pd.DataFrame:
	rows: list[dict[str, float | str]] = []

	def _compute_rsi(close: pd.Series, period: int) -> float | None:
		if close is None or close.empty or len(close) <= period:
			return None

		delta = close.diff()
		gain = delta.clip(lower=0)
		loss = -delta.clip(upper=0)

		avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
		avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

		last_gain = _safe_float(avg_gain.iloc[-1], np.nan)
		last_loss = _safe_float(avg_loss.iloc[-1], np.nan)
		if np.isnan(last_gain) or np.isnan(last_loss):
			return None

		if last_loss == 0 and last_gain == 0:
			return 50.0
		if last_loss == 0:
			return 100.0

		rs = last_gain / last_loss
		rsi = 100 - (100 / (1 + rs))
		return float(min(100.0, max(0.0, rsi)))

	for ticker in tickers:
		try:
			stock = yf.Ticker(ticker)
			history = _with_retries(
				lambda: stock.history(period="18mo", interval="1d", auto_adjust=False)
			)
			if history is None or history.empty:
				continue
			close = pd.to_numeric(history["Close"], errors="coerce").dropna()
			if close.empty or len(close) < max(sma_period, rsi_period + 1):
				continue

			latest_close = float(close.iloc[-1])
			sma_value = float(close.rolling(sma_period).mean().iloc[-1])
			if sma_value <= 0:
				continue
			pct_vs_sma = (latest_close / sma_value - 1.0) * 100.0

			rsi_value = _compute_rsi(close, rsi_period)

			window_52w = close.iloc[-252:] if len(close) >= 252 else close
			low_52w = _safe_float(window_52w.min(), np.nan)
			if np.isnan(low_52w) or low_52w <= 0:
				continue
			pct_from_52w_low = (latest_close / low_52w - 1.0) * 100.0

			rows.append(
				{
					"Ticker": ticker,
					"Close": latest_close,
					"SMA": sma_value,
					"Pct vs SMA": pct_vs_sma,
					"RSI": rsi_value,
					"52W Low": low_52w,
					"Pct from 52W Low": pct_from_52w_low,
				}
			)
		except Exception:
			continue

	if not rows:
		return pd.DataFrame(
			columns=["Ticker", "Close", "SMA", "Pct vs SMA", "RSI", "52W Low", "Pct from 52W Low"]
		)

	return pd.DataFrame(rows).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=900)
def _compute_price_vs_sma(tickers: tuple[str, ...], sma_period: int) -> pd.DataFrame:
	df = _compute_watchlist_indicators(tickers=tickers, sma_period=sma_period, rsi_period=14)
	if df.empty:
		return pd.DataFrame(columns=["Ticker", "Close", "SMA", "Pct vs SMA"])
	return df[["Ticker", "Close", "SMA", "Pct vs SMA"]].sort_values("Pct vs SMA", ascending=True).reset_index(drop=True)


def render_watchlist_tab() -> None:
	st.subheader("Watchlist")

	all_sources = {**_get_watchlist_presets(), **_load_saved_portfolio_sources()}
	source_options = list(all_sources.keys())
	default_source = source_options[0] if source_options else "S&P 500"

	controls = st.columns([1.3, 1.0], gap="medium")
	with controls[0]:
		selected_source = st.selectbox(
			"Watchlist source",
			source_options,
			index=0 if default_source in source_options else None,
			key="watchlist_source",
		)

	with controls[1]:
		sma_label = st.selectbox(
			"SMA Period",
			["50 day SMA", "100 day SMA", "200 day SMA"],
			index=0,
			key="watchlist_sma_label",
		)
		sma_period = int(sma_label.split(" ")[0])

	source_context = selected_source
	if st.session_state.get("watchlist_source_context") != source_context:
		st.session_state["watchlist_tickers"] = list(all_sources.get(selected_source, []))
		st.session_state["watchlist_source_context"] = source_context

	current_tickers = list(st.session_state.get("watchlist_tickers", all_sources.get(selected_source, [])))
	universe_options = sorted(set(current_tickers) | set(all_sources.get(selected_source, [])))

	selected_tickers = st.multiselect(
		"Stocks in watchlist",
		options=universe_options,
		default=current_tickers,
		key="watchlist_multiselect",
	)
	st.session_state["watchlist_tickers"] = selected_tickers

	action_cols = st.columns([1.5, 1.2, 0.8, 1.5, 1.0], gap="small")
	with action_cols[0]:
		add_ticker = st.text_input("Add ticker", value="", key="watchlist_add_ticker").strip().upper()
	with action_cols[1]:
		if st.button("Add", use_container_width=True, key="watchlist_add_btn") and add_ticker:
			if add_ticker not in st.session_state["watchlist_tickers"]:
				st.session_state["watchlist_tickers"] = st.session_state["watchlist_tickers"] + [add_ticker]
			st.rerun()
	with action_cols[3]:
		remove_list = st.multiselect(
			"Remove tickers",
			options=st.session_state.get("watchlist_tickers", []),
			default=[],
			key="watchlist_remove_multiselect",
		)
	with action_cols[4]:
		if st.button("Remove selected", use_container_width=True, key="watchlist_remove_btn") and remove_list:
			st.session_state["watchlist_tickers"] = [
				t for t in st.session_state.get("watchlist_tickers", []) if t not in set(remove_list)
			]
			st.rerun()

	selected_universe = tuple(st.session_state.get("watchlist_tickers", []))
	if not selected_universe:
		st.info("Add at least one ticker to run watchlist ranking analysis.")
		return

	ind_df = _compute_watchlist_indicators(selected_universe, sma_period=sma_period, rsi_period=14)
	if ind_df.empty:
		st.warning("No sufficient price history available for the selected watchlist.")
		return

	max_display = min(20, len(ind_df))

	# 1) SMA ranking plot (left = most below SMA, right = least below / above)
	sma_df = ind_df.sort_values("Pct vs SMA", ascending=True).head(max_display).copy()
	sma_colors = ["#ef4444" if v < 0 else "#22c55e" for v in sma_df["Pct vs SMA"]]
	sma_fig = go.Figure(
		go.Bar(
			x=sma_df["Ticker"],
			y=sma_df["Pct vs SMA"],
			marker=dict(color=sma_colors, line=dict(color="#111827", width=1)),
			text=[f"{v:.1f}%" for v in sma_df["Pct vs SMA"]],
			textposition="outside",
			hovertemplate=(
				"<b>%{x}</b><br>"
				+ "% vs SMA: %{y:.2f}%<br>"
				+ "Close: %{customdata[0]:.2f}<br>"
				+ f"{sma_period}-day SMA: "
				+ "%{customdata[1]:.2f}<extra></extra>"
			),
			customdata=sma_df[["Close", "SMA"]].to_numpy(),
		)
	)
	sma_fig.update_layout(
		title=f"Price vs {sma_period}-day SMA ranking (max {max_display})",
		xaxis_title="Stocks (left: most below SMA, right: least below / above)",
		yaxis_title="% vs SMA",
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		height=520,
		margin=dict(l=20, r=20, t=60, b=30),
	)
	sma_fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
	st.plotly_chart(sma_fig, use_container_width=True)
	st.caption("SMA color guide: red = below SMA (more oversold), green = above SMA (stronger momentum).")

	# 2) RSI ranking plot
	rsi_df = ind_df.dropna(subset=["RSI"]).sort_values("RSI", ascending=True).head(max_display).copy()
	if not rsi_df.empty:
		rsi_colors = [
			"#ef4444" if v < 30 else "#f59e0b" if v <= 70 else "#22c55e"
			for v in rsi_df["RSI"]
		]
		rsi_fig = go.Figure(
			go.Bar(
				x=rsi_df["Ticker"],
				y=rsi_df["RSI"],
				marker=dict(color=rsi_colors, line=dict(color="#111827", width=1)),
				text=[f"{v:.1f}" for v in rsi_df["RSI"]],
				textposition="outside",
				hovertemplate=(
					"<b>%{x}</b><br>"
					+ "RSI (14): %{y:.2f}<br>"
					+ "Close: %{customdata[0]:.2f}<extra></extra>"
				),
				customdata=rsi_df[["Close"]].to_numpy(),
			)
		)
		rsi_fig.update_layout(
			title=f"RSI ranking — most oversold to overbought (max {max_display})",
			xaxis_title="Stocks (left: most oversold, right: less oversold / overbought)",
			yaxis_title="RSI (14)",
			paper_bgcolor="rgba(0,0,0,0)",
			plot_bgcolor="rgba(0,0,0,0)",
			height=520,
			margin=dict(l=20, r=20, t=60, b=30),
		)
		rsi_fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#ef4444")
		rsi_fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="#22c55e")
		st.plotly_chart(rsi_fig, use_container_width=True)
		st.caption(
			"RSI color guide: red < 30 (oversold), amber 30–70 (neutral), green > 70 (overbought)."
		)

	# 3) Distance from 52-week low plot
	low_df = ind_df.sort_values("Pct from 52W Low", ascending=True).head(max_display).copy()
	low_colors = [
		"#ef4444" if v <= 10 else "#f59e0b" if v <= 30 else "#22c55e"
		for v in low_df["Pct from 52W Low"]
	]
	low_fig = go.Figure(
		go.Bar(
			x=low_df["Ticker"],
			y=low_df["Pct from 52W Low"],
			marker=dict(color=low_colors, line=dict(color="#111827", width=1)),
			text=[f"{v:.1f}%" for v in low_df["Pct from 52W Low"]],
			textposition="outside",
			hovertemplate=(
				"<b>%{x}</b><br>"
				+ "% from 52W low: %{y:.2f}%<br>"
				+ "Close: %{customdata[0]:.2f}<br>"
				+ "52W low: %{customdata[1]:.2f}<extra></extra>"
			),
			customdata=low_df[["Close", "52W Low"]].to_numpy(),
		)
	)
	low_fig.update_layout(
		title=f"Distance from 52-week low ranking (max {max_display})",
		xaxis_title="Stocks (left: closest to 52W low, right: furthest)",
		yaxis_title="% above 52W low",
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		height=520,
		margin=dict(l=20, r=20, t=60, b=30),
	)
	st.plotly_chart(low_fig, use_container_width=True)
	st.caption(
		"52W-low color guide: red ≤ 10% above low (near lows), amber 10–30%, green > 30% (closer to highs)."
	)


def _safe_float(value, default=np.nan):
	try:
		if value is None:
			return default
		value = float(value)
		if np.isnan(value) or np.isinf(value):
			return default
		return value
	except Exception:
		return default


def _get_finnhub_api_key() -> str | None:
	# Priority: Streamlit secrets -> environment variable.
	try:
		secret_key = st.secrets.get("FINNHUB_API_KEY")
		if secret_key:
			return str(secret_key).strip()
	except Exception:
		pass

	env_key = os.getenv("FINNHUB_API_KEY")
	if env_key:
		return env_key.strip()
	return None


def _finnhub_get(path: str, params: dict | None = None) -> dict:
	api_key = _get_finnhub_api_key()
	if not api_key:
		raise ValueError("Finnhub API key not configured. Set FINNHUB_API_KEY in secrets or environment.")

	url = f"https://finnhub.io/api/v1/{path.lstrip('/')}"
	query = dict(params or {})
	query["token"] = api_key
	response = requests.get(url, params=query, timeout=10)

	if response.status_code == 429:
		raise ValueError("Finnhub rate limit reached (HTTP 429).")
	if response.status_code in {401, 403}:
		raise ValueError("Finnhub authentication failed. Check FINNHUB_API_KEY.")
	if response.status_code >= 400:
		raise ValueError(f"Finnhub request failed ({response.status_code}) for {path}.")

	data = response.json()
	if isinstance(data, dict) and data.get("error"):
		raise ValueError(f"Finnhub error: {data.get('error')}")
	return data


def _text_match_any(text: str, needles: list[str]) -> bool:
	text_norm = str(text or "").strip().lower()
	text_compact = "".join(ch for ch in text_norm if ch.isalnum())
	for needle in needles:
		needle_norm = str(needle or "").strip().lower()
		needle_compact = "".join(ch for ch in needle_norm if ch.isalnum())
		if (needle_norm and needle_norm in text_norm) or (needle_compact and needle_compact in text_compact):
			return True
	return False


def _finnhub_find_line_value(lines: list[dict], include_terms: list[str]) -> float | None:
	for row in lines:
		label = str(row.get("label") or "")
		concept = str(row.get("concept") or "")
		if _text_match_any(label, include_terms) or _text_match_any(concept, include_terms):
			value = _safe_float(row.get("value"), np.nan)
			if not np.isnan(value):
				return float(value)
	return None


def _finnhub_extract_statement_lines(reported_payload: dict, section_key: str) -> list[list[dict]]:
	entries = reported_payload.get("data") or []
	if not isinstance(entries, list):
		return []

	quarters: list[list[dict]] = []
	for entry in entries:
		report = entry.get("report") if isinstance(entry, dict) else None
		if not isinstance(report, dict):
			continue
		lines = report.get(section_key)
		if isinstance(lines, list) and lines:
			quarters.append(lines)

	return quarters


def _finnhub_extract_cumulative_metric_map(
	reported_payload: dict,
	section_key: str,
	include_terms: list[str],
) -> dict[tuple[int, int], float]:
	entries = reported_payload.get("data") or []
	if not isinstance(entries, list):
		return {}

	values: dict[tuple[int, int], float] = {}
	for entry in entries:
		if not isinstance(entry, dict):
			continue
		try:
			year = int(entry.get("year"))
			quarter = int(entry.get("quarter"))
		except Exception:
			continue
		if quarter < 1 or quarter > 4:
			continue

		report = entry.get("report") or {}
		lines = report.get(section_key)
		if not isinstance(lines, list) or not lines:
			continue

		value = _finnhub_find_line_value(lines, include_terms)
		if value is None:
			continue
		values[(year, quarter)] = float(value)

	return values


def _finnhub_extract_annual_metric_map(
	reported_payload: dict,
	section_key: str,
	include_terms: list[str],
) -> dict[int, float]:
	entries = reported_payload.get("data") or []
	if not isinstance(entries, list):
		return {}

	values: dict[int, float] = {}
	for entry in entries:
		if not isinstance(entry, dict):
			continue
		try:
			year = int(entry.get("year"))
		except Exception:
			continue

		report = entry.get("report") or {}
		lines = report.get(section_key)
		if not isinstance(lines, list) or not lines:
			continue

		value = _finnhub_find_line_value(lines, include_terms)
		if value is None:
			continue
		values[year] = float(value)

	return values


def _finnhub_ttm_from_cumulative_reports(
	quarterly_payload: dict,
	annual_payload: dict,
	section_key: str,
	include_terms: list[str],
) -> float | None:
	quarterly_map = _finnhub_extract_cumulative_metric_map(
		reported_payload=quarterly_payload,
		section_key=section_key,
		include_terms=include_terms,
	)
	annual_map = _finnhub_extract_annual_metric_map(
		reported_payload=annual_payload,
		section_key=section_key,
		include_terms=include_terms,
	)

	if not quarterly_map:
		return None

	latest_year, latest_quarter = max(quarterly_map.keys())
	latest_cum = quarterly_map.get((latest_year, latest_quarter))
	if latest_cum is None:
		return None

	# Finnhub quarterly statements are often fiscal YTD cumulative.
	# For Q1-Q3: TTM = current YTD + prior FY annual - prior FY same-quarter YTD.
	if latest_quarter in {1, 2, 3}:
		prior_same_quarter = quarterly_map.get((latest_year - 1, latest_quarter))
		prior_year_annual = annual_map.get(latest_year - 1)
		if prior_same_quarter is not None and prior_year_annual is not None:
			return float(latest_cum + prior_year_annual - prior_same_quarter)

	# If a Q4 cumulative is available, annual equals TTM.
	if latest_quarter == 4:
		annual_latest = annual_map.get(latest_year)
		if annual_latest is not None:
			return float(annual_latest)

	# Fallback: derive standalone quarters and sum latest 4.
	def previous_quarter(year: int, quarter: int) -> tuple[int, int]:
		return (year - 1, 4) if quarter == 1 else (year, quarter - 1)

	def standalone_quarter(year: int, quarter: int) -> float | None:
		if quarter == 1:
			return quarterly_map.get((year, 1))
		if quarter in {2, 3}:
			curr = quarterly_map.get((year, quarter))
			prev = quarterly_map.get((year, quarter - 1))
			if curr is None or prev is None:
				return None
			return curr - prev
		if quarter == 4:
			annual_total = annual_map.get(year)
			q3_cum = quarterly_map.get((year, 3))
			if annual_total is None or q3_cum is None:
				return None
			return annual_total - q3_cum
		return None

	parts: list[float] = []
	y, q = latest_year, latest_quarter
	for _ in range(4):
		part = standalone_quarter(y, q)
		if part is None:
			return None
		parts.append(float(part))
		y, q = previous_quarter(y, q)

	return float(sum(parts)) if len(parts) == 4 else None


def _finnhub_metric_from_reports(metric_name: str, quarterly_payload: dict, annual_payload: dict) -> float | None:

	if metric_name == "FCF (Free Cash Flow)":
		direct_fcf = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="cf",
			include_terms=["free cash flow", "freecashflow"],
		)
		if direct_fcf is not None:
			return float(direct_fcf)

		ttm_ocf = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="cf",
			include_terms=[
				"operating cash flow",
				"net cash from operations",
				"net cash provided by operating activities",
				"net cash provided by used in operating activities",
			],
		)
		ttm_capex = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="cf",
			include_terms=[
				"capital expenditures",
				"capital expenditure",
				"payments to acquire property",
				"payments to acquire property plant and equipment",
				"purchase of property and equipment",
			],
		)
		if ttm_ocf is not None and ttm_capex is not None:
			return _compute_fcf(ttm_ocf, ttm_capex)

		return None

	if metric_name == "Operating Cash Flow":
		value = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="cf",
			include_terms=[
				"operating cash flow",
				"net cash from operations",
				"net cash provided by operating activities",
				"net cash provided by used in operating activities",
			],
		)
		return None if value is None else float(value)

	if metric_name == "Revenue":
		value = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="ic",
			include_terms=["revenue", "sales revenue net", "total revenue"],
		)
		return None if value is None else float(value)

	if metric_name == "Operating Income":
		value = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="ic",
			include_terms=["operating income", "income from operations", "operating income loss"],
		)
		return None if value is None else float(value)

	if metric_name == "Earnings (Net Income)":
		value = _finnhub_ttm_from_cumulative_reports(
			quarterly_payload=quarterly_payload,
			annual_payload=annual_payload,
			section_key="ic",
			include_terms=["net income", "profit loss", "net income loss"],
		)
		return None if value is None else float(value)

	return None


def _finnhub_shares_outstanding(
	profile_payload: dict,
	metric_payload: dict,
	reported_payload: dict,
	price: float,
) -> float | None:
	metric_map = metric_payload.get("metric") if isinstance(metric_payload, dict) else {}
	metric_map = metric_map if isinstance(metric_map, dict) else {}
	target_market_cap = _safe_float(metric_map.get("marketCapitalization"), np.nan)
	if not np.isnan(target_market_cap):
		target_market_cap = float(target_market_cap) * 1_000_000

	for candidate in [
		metric_map.get("sharesOutstanding"),
		metric_map.get("shareOutstanding"),
		profile_payload.get("shareOutstanding"),
	]:
		val = _safe_float(candidate, np.nan)
		if np.isnan(val) or val <= 0:
			continue

		raw = float(val)
		share_candidates = [raw, raw * 1_000_000]

		if not np.isnan(target_market_cap) and target_market_cap > 0 and price > 0:
			best = min(
				share_candidates,
				key=lambda shares: abs((price * shares) - target_market_cap),
			)
			if best > 0:
				return float(best)

		# Fallback heuristic when no market-cap anchor is available.
		return float(raw * 1_000_000) if raw < 1_000_000 else float(raw)

	ic_quarters = _finnhub_extract_statement_lines(reported_payload, "ic")
	if ic_quarters:
		latest = ic_quarters[0]
		shares = _finnhub_find_line_value(
			latest,
			[
				"weighted average number of shares outstanding basic",
				"weighted average shares outstanding basic",
				"common stock shares outstanding",
			],
		)
		if shares is not None and shares > 0:
			return float(shares)

	return None


@st.cache_data(show_spinner=False, ttl=1800)
def load_stock_snapshot_finnhub(ticker: str, metric_name: str) -> StockSnapshot:
	ticker = ticker.strip().upper()

	quote = _finnhub_get("quote", {"symbol": ticker})
	price = _safe_float(quote.get("c"), np.nan)
	if np.isnan(price) or price <= 0:
		raise ValueError(f"Finnhub quote missing/invalid current price for {ticker}.")

	profile = _finnhub_get("stock/profile2", {"symbol": ticker})
	metrics = _finnhub_get("stock/metric", {"symbol": ticker, "metric": "all"})
	quarterly_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "quarterly"})
	annual_reported = _finnhub_get("stock/financials-reported", {"symbol": ticker, "freq": "annual"})

	shares = _finnhub_shares_outstanding(
		profile_payload=profile,
		metric_payload=metrics,
		reported_payload=quarterly_reported,
		price=float(price),
	)
	if shares is None or shares <= 0:
		raise ValueError(f"Finnhub could not resolve shares outstanding for {ticker}.")

	metric_value = _finnhub_metric_from_reports(
		metric_name=metric_name,
		quarterly_payload=quarterly_reported,
		annual_payload=annual_reported,
	)
	if metric_value is None:
		metric_map = metrics.get("metric") if isinstance(metrics, dict) else {}
		metric_map = metric_map if isinstance(metric_map, dict) else {}
		fallback_fields = {
			"FCF (Free Cash Flow)": ["freeCashFlowTTM", "fcfPerShareTTM"],
			"Operating Cash Flow": ["operatingCashFlowTTM"],
			"Revenue": ["revenueTTM"],
			"Operating Income": ["operatingIncomeTTM"],
			"Earnings (Net Income)": ["netIncomeTTM"],
		}
		for field in fallback_fields.get(metric_name, []):
			candidate = _safe_float(metric_map.get(field), np.nan)
			if not np.isnan(candidate):
				if "PerShare" in field:
					metric_value = candidate * shares
				else:
					metric_value = candidate
				break

	if metric_value is None:
		raise ValueError(f"Finnhub could not resolve TTM value for metric '{metric_name}' on {ticker}.")

	current_ttm_multiple = None
	if metric_value > 0:
		current_ttm_multiple = (price * shares) / metric_value

	return StockSnapshot(
		ticker=ticker,
		price=float(price),
		shares_outstanding=float(shares),
		metric_ttm=float(metric_value),
		metric_name=metric_name,
		current_ttm_multiple=current_ttm_multiple,
	)


def _is_rate_limit_error(exc: Exception) -> bool:
	if isinstance(exc, YFRateLimitError):
		return True
	msg = str(exc).lower()
	return "too many requests" in msg or "rate limit" in msg


def _with_retries(fetch_fn, attempts: int = 3):
	last_exc = None
	for i in range(attempts):
		try:
			return fetch_fn()
		except Exception as exc:
			last_exc = exc
			if i < attempts - 1 and _is_rate_limit_error(exc):
				time.sleep(0.6 * (i + 1))
				continue
			raise
	if last_exc:
		raise last_exc
	return None


def _probe_yahoo_rate_limit(ticker: str) -> bool:
	"""
	Directly probe Yahoo endpoint to detect throttling (HTTP 429).
	Used only as a fallback signal when statements/info are unexpectedly empty.
	"""
	url = (
		f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
		"?modules=financialData,defaultKeyStatistics"
	)
	try:
		response = requests.get(url, timeout=8)
		return response.status_code == 429
	except Exception:
		return False


def _extract_row_latest_value(df: pd.DataFrame, labels: list[str]) -> float | None:
	if df is None or df.empty:
		return None

	normalized_index = {str(idx).strip().lower(): idx for idx in df.index}
	for label in labels:
		key = label.strip().lower()
		if key in normalized_index:
			row = df.loc[normalized_index[key]]
			values = pd.to_numeric(row, errors="coerce").dropna()
			if values.empty:
				return None
			# For TTM statements, first value is typically the latest trailing value.
			return float(values.iloc[0])
	return None


def _extract_row_ttm_from_quarters(df: pd.DataFrame, labels: list[str]) -> float | None:
	if df is None or df.empty:
		return None

	normalized_index = {str(idx).strip().lower(): idx for idx in df.index}
	for label in labels:
		key = label.strip().lower()
		if key in normalized_index:
			row = df.loc[normalized_index[key]]
			values = pd.to_numeric(row, errors="coerce").dropna()
			if values.empty:
				return None
			return float(values.iloc[:4].sum())
	return None


def _extract_value_by_period_mode(df: pd.DataFrame, labels: list[str], period_mode: str) -> float | None:
	if period_mode == "quarterly":
		return _extract_row_ttm_from_quarters(df, labels)
	# ttm/yearly fall back to latest point-in-time value
	return _extract_row_latest_value(df, labels)


def _get_ttm_statement(stock: yf.Ticker, statement_type: str) -> tuple[pd.DataFrame, str, bool]:
	"""
	Load trailing-twelve-month statement in a structured yfinance-first way.
	statement_type: 'cashflow' or 'income'
	Returns: (dataframe, period_mode, rate_limited_detected)
	where period_mode in {'ttm', 'quarterly', 'yearly', 'none'}.
	"""
	rate_limited_detected = False

	if statement_type == "cashflow":
		# Preferred path for newer yfinance versions
		try:
			df = _with_retries(lambda: getattr(stock, "ttm_cashflow"))
			if df is not None and not df.empty:
				return df, "ttm", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# API fallback where 'trailing' is supported
		try:
			df = _with_retries(lambda: stock.get_cash_flow(freq="trailing"))
			if df is not None and not df.empty:
				return df, "ttm", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Older API fallback (quarterly)
		try:
			df = _with_retries(lambda: stock.get_cash_flow(freq="quarterly"))
			if df is not None and not df.empty:
				return df, "quarterly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Older API fallback (yearly)
		try:
			df = _with_retries(lambda: stock.get_cash_flow(freq="yearly"))
			if df is not None and not df.empty:
				return df, "yearly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Legacy fallback
		try:
			df = _with_retries(lambda: stock.quarterly_cashflow)
			if df is not None and not df.empty:
				return df, "quarterly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Additional fallback naming on some yfinance builds
		try:
			df = _with_retries(lambda: stock.cashflow)
			if df is not None and not df.empty:
				return df, "yearly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		return pd.DataFrame(), "none", rate_limited_detected

	if statement_type == "income":
		# Preferred path for newer yfinance versions
		try:
			df = _with_retries(lambda: getattr(stock, "ttm_income_stmt"))
			if df is not None and not df.empty:
				return df, "ttm", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# API fallback where 'trailing' is supported
		try:
			df = _with_retries(lambda: stock.get_income_stmt(freq="trailing"))
			if df is not None and not df.empty:
				return df, "ttm", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Older API fallback (quarterly)
		try:
			df = _with_retries(lambda: stock.get_income_stmt(freq="quarterly"))
			if df is not None and not df.empty:
				return df, "quarterly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Older API fallback (yearly)
		try:
			df = _with_retries(lambda: stock.get_income_stmt(freq="yearly"))
			if df is not None and not df.empty:
				return df, "yearly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Legacy fallback
		try:
			df = _with_retries(lambda: stock.quarterly_income_stmt)
			if df is not None and not df.empty:
				return df, "quarterly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		# Additional fallback naming on some yfinance builds
		try:
			df = _with_retries(lambda: stock.income_stmt)
			if df is not None and not df.empty:
				return df, "yearly", rate_limited_detected
		except Exception as exc:
			if _is_rate_limit_error(exc):
				rate_limited_detected = True
			pass

		return pd.DataFrame(), "none", rate_limited_detected

	raise ValueError(f"Unsupported statement_type: {statement_type}")


def _compute_fcf(op_cf: float, capex: float) -> float:
	# Some datasets store CapEx as negative, others as positive.
	# Normalize so FCF behaves as OCF - |CapEx|.
	return op_cf - abs(capex)


def _get_metric_ttm_value(
	metric_name: str,
	ttm_cashflow: pd.DataFrame,
	cashflow_mode: str,
	ttm_income: pd.DataFrame,
	income_mode: str,
	info: dict | None = None,
) -> float | None:
	info = info or {}

	if metric_name == "FCF (Free Cash Flow)":
		direct_fcf = _extract_value_by_period_mode(
			ttm_cashflow,
			METRIC_LABEL_CANDIDATES[metric_name],
			cashflow_mode,
		)
		if direct_fcf is not None:
			return direct_fcf

		op_cf = _extract_value_by_period_mode(
			ttm_cashflow,
			[
				"Operating Cash Flow",
				"Total Cash From Operating Activities",
				"Cash Flow From Continuing Operating Activities",
			],
			cashflow_mode,
		)
		capex = _extract_value_by_period_mode(
			ttm_cashflow,
			[
				"Capital Expenditures",
				"Capital Expenditure",
				"Purchase Of PPE",
			],
			cashflow_mode,
		)
		if op_cf is not None and capex is not None:
			return _compute_fcf(op_cf, capex)
		return None

	if metric_name == "Operating Cash Flow":
		value = _extract_value_by_period_mode(
			ttm_cashflow,
			METRIC_LABEL_CANDIDATES[metric_name],
			cashflow_mode,
		)
		if value is not None:
			return value

		fallback_ocf = _safe_float(info.get("operatingCashflow"))
		if not np.isnan(fallback_ocf):
			return fallback_ocf
		return None

	if metric_name == "Operating Income":
		return _extract_value_by_period_mode(
			ttm_income,
			METRIC_LABEL_CANDIDATES[metric_name],
			income_mode,
		)

	if metric_name == "Revenue":
		value = _extract_value_by_period_mode(
			ttm_income,
			METRIC_LABEL_CANDIDATES[metric_name],
			income_mode,
		)
		if value is not None:
			return value

		fallback_revenue = _safe_float(info.get("totalRevenue"))
		if not np.isnan(fallback_revenue):
			return fallback_revenue

		fallback_revenue_alt = _safe_float(info.get("revenue"))
		if not np.isnan(fallback_revenue_alt):
			return fallback_revenue_alt
		return None

	if metric_name == "Earnings (Net Income)":
		return _extract_value_by_period_mode(
			ttm_income,
			METRIC_LABEL_CANDIDATES[metric_name],
			income_mode,
		)

	return None


def _statement_rows_preview(df: pd.DataFrame, n: int = 12) -> str:
	if df is None or df.empty:
		return "(none)"
	rows = [str(idx) for idx in df.index[:n]]
	return ", ".join(rows) if rows else "(none)"


def _extract_info_value(info: dict, keys: list[str]) -> float | None:
	for key in keys:
		value = _safe_float(info.get(key))
		if not np.isnan(value):
			return value
	return None


def _get_balance_sheet(stock: yf.Ticker) -> pd.DataFrame:
	try:
		df = _with_retries(lambda: stock.quarterly_balance_sheet)
		if df is not None and not df.empty:
			return df
	except Exception:
		pass

	try:
		df = _with_retries(lambda: stock.balance_sheet)
		if df is not None and not df.empty:
			return df
	except Exception:
		pass

	return pd.DataFrame()


def _get_yearly_cashflow(stock: yf.Ticker) -> pd.DataFrame:
	try:
		df = _with_retries(lambda: stock.get_cash_flow(freq="yearly"))
		if df is not None and not df.empty:
			return df
	except Exception:
		pass

	try:
		df = _with_retries(lambda: stock.cashflow)
		if df is not None and not df.empty:
			return df
	except Exception:
		pass

	return pd.DataFrame()


def _get_yearly_income(stock: yf.Ticker) -> pd.DataFrame:
	try:
		df = _with_retries(lambda: stock.get_income_stmt(freq="yearly"))
		if df is not None and not df.empty:
			return df
	except Exception:
		pass

	try:
		df = _with_retries(lambda: stock.income_stmt)
		if df is not None and not df.empty:
			return df
	except Exception:
		pass

	return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=1800)
def load_previous_full_year_metric_yahoo(ticker: str, metric_name: str) -> float | None:
	stock = yf.Ticker(ticker.strip().upper())
	if metric_name in {"FCF (Free Cash Flow)", "Operating Cash Flow"}:
		yearly_cf = _get_yearly_cashflow(stock)
		if metric_name == "Operating Cash Flow":
			value = _extract_row_latest_value(
				yearly_cf,
				[
					"Operating Cash Flow",
					"Total Cash From Operating Activities",
					"Cash Flow From Continuing Operating Activities",
				],
			)
			return None if value is None else float(value)

		direct_fcf = _extract_row_latest_value(yearly_cf, METRIC_LABEL_CANDIDATES["FCF (Free Cash Flow)"])
		if direct_fcf is not None:
			return float(direct_fcf)

		op_cf = _extract_row_latest_value(
			yearly_cf,
			[
				"Operating Cash Flow",
				"Total Cash From Operating Activities",
				"Cash Flow From Continuing Operating Activities",
			],
		)
		capex = _extract_row_latest_value(
			yearly_cf,
			[
				"Capital Expenditures",
				"Capital Expenditure",
				"Purchase Of PPE",
			],
		)
		if op_cf is not None and capex is not None:
			return _compute_fcf(op_cf, capex)
		return None

	yearly_income = _get_yearly_income(stock)
	if metric_name == "Revenue":
		value = _extract_row_latest_value(yearly_income, METRIC_LABEL_CANDIDATES["Revenue"])
		return None if value is None else float(value)
	if metric_name == "Operating Income":
		value = _extract_row_latest_value(yearly_income, METRIC_LABEL_CANDIDATES["Operating Income"])
		return None if value is None else float(value)
	if metric_name == "Earnings (Net Income)":
		value = _extract_row_latest_value(yearly_income, METRIC_LABEL_CANDIDATES["Earnings (Net Income)"])
		return None if value is None else float(value)

	return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_company_capital_structure(ticker: str) -> tuple[float, float]:
	stock = yf.Ticker(ticker.strip().upper())
	info = {}
	try:
		info = _with_retries(lambda: (stock.info or {}))
	except Exception:
		info = {}

	cash = _extract_info_value(
		info,
		[
			"totalCash",
			"cash",
			"cashAndCashEquivalents",
			"cashAndShortTermInvestments",
		],
	)
	debt = _extract_info_value(
		info,
		[
			"totalDebt",
			"netDebt",
			"longTermDebt",
		],
	)

	bs = _get_balance_sheet(stock)
	if cash is None:
		cash = _extract_row_latest_value(
			bs,
			[
				"Cash And Cash Equivalents",
				"Cash Cash Equivalents And Short Term Investments",
				"Cash And Short Term Investments",
			],
		)

	if debt is None:
		total_debt = _extract_row_latest_value(bs, ["Total Debt"])
		if total_debt is not None:
			debt = total_debt
		else:
			long_term_debt = _extract_row_latest_value(
				bs,
				[
					"Long Term Debt",
					"Long Term Debt And Capital Lease Obligation",
				],
			)
			current_debt = _extract_row_latest_value(
				bs,
				[
					"Current Debt",
					"Current Debt And Capital Lease Obligation",
				],
			)
			debt = (long_term_debt or 0.0) + (current_debt or 0.0)

	return float(cash or 0.0), float(debt or 0.0)


@st.cache_data(show_spinner=False, ttl=3600)
def load_fcf_previous_full_year(ticker: str) -> float | None:
	stock = yf.Ticker(ticker.strip().upper())
	yearly_cf = _get_yearly_cashflow(stock)

	direct_fcf = _extract_row_latest_value(
		yearly_cf,
		METRIC_LABEL_CANDIDATES["FCF (Free Cash Flow)"],
	)
	if direct_fcf is not None:
		return direct_fcf

	op_cf = _extract_row_latest_value(
		yearly_cf,
		[
			"Operating Cash Flow",
			"Total Cash From Operating Activities",
			"Cash Flow From Continuing Operating Activities",
		],
	)
	capex = _extract_row_latest_value(
		yearly_cf,
		[
			"Capital Expenditures",
			"Capital Expenditure",
			"Purchase Of PPE",
		],
	)
	if op_cf is not None and capex is not None:
		return _compute_fcf(op_cf, capex)

	return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_sbc_ttm(ticker: str) -> float | None:
	stock = yf.Ticker(ticker.strip().upper())
	ttm_cashflow, period_mode, _ = _get_ttm_statement(stock, "cashflow")
	sbc = _extract_value_by_period_mode(
		ttm_cashflow,
		[
			"Stock Based Compensation",
			"StockBasedCompensation",
		],
		period_mode,
	)
	if sbc is None:
		return None
	return abs(float(sbc))


@st.cache_data(show_spinner=False, ttl=3600)
def load_owner_adjusted_metric(ticker: str, metric_name: str) -> float | None:
	metric_row_map = {
		"FCF (Free Cash Flow)": "Owner FCF",
		"Operating Cash Flow": "Owner OCF",
		"Earnings (Net Income)": "Owner Earnings",
	}

	owner_row = metric_row_map.get(metric_name)
	if owner_row is None:
		return None

	try:
		owner_df = get_owner_earnings(ticker_symbol=ticker.strip().upper())
	except Exception:
		return None

	if owner_df is None or owner_df.empty or owner_row not in owner_df.index:
		return None

	series = pd.to_numeric(owner_df.loc[owner_row], errors="coerce")
	valid_series = series.dropna()
	if valid_series.empty:
		return None

	# Prefer the latest fiscal period when date-like columns exist.
	parsed_dates = pd.to_datetime(valid_series.index, errors="coerce")
	if parsed_dates.notna().any():
		date_series = pd.Series(parsed_dates, index=valid_series.index).dropna()
		latest_col = date_series.idxmax()
		return float(valid_series.loc[latest_col])

	# Fallback: keep existing ordering behavior if columns are not date-like.
	return float(valid_series.iloc[0])


def load_dividend_profile(ticker: str, current_price: float) -> dict[str, float | bool]:
	ticker = ticker.strip().upper()
	stock = yf.Ticker(ticker)
	info: dict = {}
	fast_info: dict = {}

	try:
		info = _with_retries(lambda: (stock.info or {}))
	except Exception:
		info = {}

	try:
		fast_info = dict(stock.fast_info) if stock.fast_info else {}
	except Exception:
		fast_info = {}

	def _first_positive(data: dict, keys: list[str]) -> float | None:
		for key in keys:
			value = _safe_float(data.get(key), np.nan)
			if not np.isnan(value) and value > 0:
				return float(value)
		return None

	def _normalize_dividend_yield(raw_value: float | None) -> float:
		y = _safe_float(raw_value, np.nan)
		if np.isnan(y) or y <= 0:
			return np.nan
		# Some providers return percent points (e.g., 0.86 for 0.86%).
		# Decimal yields above 20% are rare; treat >= 0.20 as percent points.
		if y >= 0.20:
			return float(y / 100.0)
		return float(y)

	def _normalize_growth_rate(raw_value: float | None) -> float:
		g = _safe_float(raw_value, np.nan)
		if np.isnan(g):
			return np.nan
		# Growth rates can be provided as percent points by some providers.
		if abs(g) > 1:
			g = g / 100.0
		return float(g)

	def _compute_dividend_cagr(annual_series: pd.Series, years: int) -> float:
		if annual_series is None or annual_series.empty or years <= 0:
			return np.nan
		if len(annual_series) < years + 1:
			return np.nan
		end_value = _safe_float(annual_series.iloc[-1], np.nan)
		start_value = _safe_float(annual_series.iloc[-(years + 1)], np.nan)
		if np.isnan(end_value) or np.isnan(start_value) or start_value <= 0 or end_value <= 0:
			return np.nan
		return float((end_value / start_value) ** (1 / years) - 1)

	dividend_yield = _normalize_dividend_yield(
		_first_positive(
			info,
			[
				"dividendYield",
				"trailingAnnualDividendYield",
				"fiveYearAvgDividendYield",
			],
		)
	)
	if np.isnan(dividend_yield):
		dividend_yield = _normalize_dividend_yield(
			_first_positive(
				fast_info,
				[
					"dividendYield",
					"trailingAnnualDividendYield",
				],
			)
		)

	dividend_rate = _safe_float(
		_first_positive(
			info,
			[
				"dividendRate",
				"trailingAnnualDividendRate",
				"lastDividendValue",
			],
		),
		np.nan,
	)
	if np.isnan(dividend_rate):
		dividend_rate = _safe_float(
			_first_positive(
				fast_info,
				[
					"dividendRate",
					"lastDividendValue",
				],
			),
			np.nan,
		)

	dividend_series = pd.Series(dtype=float)
	try:
		dividend_series = pd.to_numeric(stock.dividends, errors="coerce").dropna()
	except Exception:
		dividend_series = pd.Series(dtype=float)

	# Fallbacks: some yfinance builds/providers return empty `dividends`
	# even when corporate actions history contains dividend payments.
	if dividend_series.empty:
		try:
			actions = stock.actions
			if isinstance(actions, pd.DataFrame) and not actions.empty and "Dividends" in actions.columns:
				dividend_series = pd.to_numeric(actions["Dividends"], errors="coerce").dropna()
		except Exception:
			pass

	if dividend_series.empty:
		try:
			hist = _with_retries(
				lambda: stock.history(period="10y", interval="1d", auto_adjust=False, actions=True)
			)
			if isinstance(hist, pd.DataFrame) and not hist.empty and "Dividends" in hist.columns:
				dividend_series = pd.to_numeric(hist["Dividends"], errors="coerce").dropna()
		except Exception:
			pass

	annual_dividends = pd.Series(dtype=float)
	if dividend_series is not None and not dividend_series.empty:
		try:
			dividend_df = pd.DataFrame(
				{"dividend": pd.to_numeric(dividend_series, errors="coerce")},
				index=pd.to_datetime(dividend_series.index, errors="coerce"),
			).dropna()
			annual_dividends = dividend_df.groupby(dividend_df.index.year)["dividend"].sum().sort_index()
			annual_dividends = annual_dividends[annual_dividends > 0]
		except Exception:
			annual_dividends = pd.Series(dtype=float)

	trailing_annual_dividend = float(annual_dividends.iloc[-1]) if not annual_dividends.empty else 0.0
	trailing_annual_dividend = max(
		trailing_annual_dividend,
		float(
			_first_positive(
				info,
				[
					"trailingAnnualDividendRate",
					"lastDividendValue",
					"dividendRate",
				],
			)
			or 0.0
		),
	)
	trailing_annual_dividend = max(
		trailing_annual_dividend,
		float(
			_first_positive(
				fast_info,
				[
					"dividendRate",
					"lastDividendValue",
				],
			)
			or 0.0
		),
	)
	if not np.isnan(dividend_rate) and dividend_rate > 0:
		trailing_annual_dividend = max(trailing_annual_dividend, float(dividend_rate))

	if np.isnan(dividend_yield) and not np.isnan(dividend_rate) and current_price > 0 and dividend_rate > 0:
		dividend_yield = float(dividend_rate) / float(current_price)

	if np.isnan(dividend_yield) and current_price > 0 and trailing_annual_dividend > 0:
		dividend_yield = trailing_annual_dividend / float(current_price)

	# Provider fallback when Yahoo sources are missing/throttled.
	if (
		(np.isnan(dividend_yield) or dividend_yield <= 0)
		and trailing_annual_dividend <= 0
		and (np.isnan(dividend_rate) or dividend_rate <= 0)
	):
		try:
			metrics_payload = _finnhub_get("stock/metric", {"symbol": ticker, "metric": "all"})
			metric_map = metrics_payload.get("metric") if isinstance(metrics_payload, dict) else {}
			metric_map = metric_map if isinstance(metric_map, dict) else {}

			finnhub_yield = _normalize_dividend_yield(
				_first_positive(
					metric_map,
					[
						"dividendYieldIndicatedAnnual",
						"currentDividendYieldTTM",
						"dividendYield",
						"dividendYield5Y",
					],
				)
			)
			finnhub_rate = _safe_float(
				_first_positive(
					metric_map,
					[
						"dividendPerShareAnnual",
						"dividendPerShareTTM",
					],
				),
				np.nan,
			)

			if not np.isnan(finnhub_rate) and finnhub_rate > 0:
				dividend_rate = finnhub_rate
				trailing_annual_dividend = max(trailing_annual_dividend, float(finnhub_rate))

			if np.isnan(dividend_yield) and not np.isnan(finnhub_yield):
				dividend_yield = float(finnhub_yield)

			if np.isnan(dividend_yield) and current_price > 0 and not np.isnan(dividend_rate) and dividend_rate > 0:
				dividend_yield = float(dividend_rate) / float(current_price)
		except Exception:
			pass

	if np.isnan(dividend_yield):
		dividend_yield = 0.0

	# Always prefer CAGR-based defaults from history (not arithmetic mean growth).
	cagr_1y = _compute_dividend_cagr(annual_dividends, years=1)
	cagr_3y = _compute_dividend_cagr(annual_dividends, years=3)
	cagr_5y = _compute_dividend_cagr(annual_dividends, years=5)

	# Priority default for projection: 3Y CAGR -> 5Y CAGR -> 1Y CAGR.
	default_growth = np.nan
	for candidate in [cagr_3y, cagr_5y, cagr_1y]:
		if not np.isnan(candidate):
			default_growth = float(candidate)
			break

	# Fallback when annual history is sparse/unavailable: infer a 1Y-equivalent CAGR from rate fields.
	if np.isnan(default_growth):
		latest_rate = _safe_float(
			_first_positive(info, ["dividendRate", "lastDividendValue"]),
			np.nan,
		)
		trailing_rate = _safe_float(info.get("trailingAnnualDividendRate"), np.nan)
		if np.isnan(latest_rate):
			latest_rate = _safe_float(
				_first_positive(fast_info, ["dividendRate", "lastDividendValue"]),
				np.nan,
			)
		if not np.isnan(latest_rate) and not np.isnan(trailing_rate) and trailing_rate > 0:
			default_growth = float((latest_rate / trailing_rate) - 1.0)

	# Provider fallback when Yahoo growth history is unavailable.
	# Note: provider fields (e.g., dividendGrowthRate5Y) are external precomputed estimates.
	if np.isnan(default_growth):
		try:
			metrics_payload = _finnhub_get("stock/metric", {"symbol": ticker, "metric": "all"})
			metric_map = metrics_payload.get("metric") if isinstance(metrics_payload, dict) else {}
			metric_map = metric_map if isinstance(metric_map, dict) else {}
			provider_cagr_5y = _normalize_growth_rate(metric_map.get("dividendGrowthRate5Y"))
			provider_growth = _normalize_growth_rate(
				_first_positive(
					metric_map,
					[
						"dividendGrowthRate",
					],
				)
			)
			for candidate in [provider_cagr_5y, provider_growth]:
				if not np.isnan(candidate):
					default_growth = float(candidate)
					break
		except Exception:
			pass

	if np.isnan(default_growth):
		default_growth = 0.0

	# Keep defaults in a practical range for UI stability.
	default_growth = float(min(0.5, max(-0.5, float(default_growth))))
	has_dividend_event_signal = bool(
		info.get("dividendDate")
		or info.get("exDividendDate")
		or info.get("lastDividendDate")
		or fast_info.get("lastDividendDate")
		or fast_info.get("last_dividend_date")
	)
	has_dividend = bool(
		(dividend_yield > 0)
		or (trailing_annual_dividend > 0)
		or (not np.isnan(dividend_rate) and dividend_rate > 0)
		or has_dividend_event_signal
	)

	return {
		"has_dividend": has_dividend,
		"dividend_yield": float(max(0.0, dividend_yield)),
		"dividend_rate": float(0.0 if np.isnan(dividend_rate) else max(0.0, float(dividend_rate))),
		"default_dividend_growth_rate": default_growth,
		"dividend_cagr_1y": float(cagr_1y),
		"dividend_cagr_3y": float(cagr_3y),
		"dividend_cagr_5y": float(cagr_5y),
		"trailing_annual_dividend_per_share": float(max(0.0, trailing_annual_dividend)),
		"has_dividend_event_signal": has_dividend_event_signal,
	}


def _get_shares_outstanding(info: dict | None = None, fast_info: dict | None = None) -> float | None:
	info = info or {}
	fast_info = fast_info or {}

	candidates = [
		info.get("sharesOutstanding"),
		info.get("impliedSharesOutstanding"),
		fast_info.get("shares"),
		fast_info.get("shares_outstanding"),
	]

	for candidate in candidates:
		value = _safe_float(candidate)
		if not np.isnan(value) and value > 0:
			return value
	return None


def _get_price(stock: yf.Ticker, info: dict | None = None) -> float | None:
	try:
		history = _with_retries(lambda: stock.history(period="5d", interval="1d", auto_adjust=False))
		if history is not None and not history.empty:
			close = _safe_float(history["Close"].dropna().iloc[-1])
			if not np.isnan(close) and close > 0:
				return close
	except Exception:
		pass

	try:
		info = info or _with_retries(lambda: (stock.info or {}))
		for key in ["currentPrice", "regularMarketPrice", "previousClose"]:
			value = _safe_float(info.get(key))
			if not np.isnan(value) and value > 0:
				return value
	except Exception:
		pass

	return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_stock_snapshot(ticker: str, metric_name: str) -> StockSnapshot:
	ticker = ticker.strip().upper()
	stock = yf.Ticker(ticker)

	info = {}
	fast_info = {}
	info_rate_limited = False
	try:
		info = _with_retries(lambda: (stock.info or {}))
	except Exception as exc:
		if _is_rate_limit_error(exc):
			info_rate_limited = True
		info = {}

	try:
		fast_info = _with_retries(lambda: (dict(stock.fast_info) if stock.fast_info else {}))
	except Exception:
		fast_info = {}

	metric_value = None
	ttm_cashflow = pd.DataFrame()
	ttm_income = pd.DataFrame()
	cashflow_mode = "skipped"
	income_mode = "skipped"
	cashflow_rate_limited = False
	income_rate_limited = False

	if metric_name == "FCF (Free Cash Flow)":
		direct_fcf = _safe_float(info.get("freeCashflow"))
		if not np.isnan(direct_fcf):
			metric_value = direct_fcf
			cashflow_mode = "info.freeCashflow"

	if metric_name == "Operating Cash Flow":
		direct_ocf = _safe_float(info.get("operatingCashflow"))
		if not np.isnan(direct_ocf):
			metric_value = direct_ocf
			cashflow_mode = "info.operatingCashflow"

	if metric_name == "Revenue":
		direct_revenue = _safe_float(info.get("totalRevenue"))
		if not np.isnan(direct_revenue):
			metric_value = direct_revenue
			income_mode = "info.totalRevenue"

	if metric_value is None:
		needs_cashflow = metric_name in {"FCF (Free Cash Flow)", "Operating Cash Flow"}
		needs_income = metric_name in {"Revenue", "Operating Income", "Earnings (Net Income)"}

		if needs_cashflow:
			ttm_cashflow, cashflow_mode, cashflow_rate_limited = _get_ttm_statement(stock, "cashflow")
		if needs_income:
			ttm_income, income_mode, income_rate_limited = _get_ttm_statement(stock, "income")

		metric_value = _get_metric_ttm_value(
			metric_name=metric_name,
			ttm_cashflow=ttm_cashflow,
			cashflow_mode=cashflow_mode,
			ttm_income=ttm_income,
			income_mode=income_mode,
			info=info,
		)

	price = _get_price(stock, info=info)
	shares = _get_shares_outstanding(info=info, fast_info=fast_info)

	if metric_value is None:
		cash_rows = _statement_rows_preview(ttm_cashflow)
		income_rows = _statement_rows_preview(ttm_income)
		rate_limited_flag = cashflow_rate_limited or income_rate_limited or info_rate_limited
		if not rate_limited_flag:
			rate_limited_flag = _probe_yahoo_rate_limit(ticker)
		if metric_name == "Operating Cash Flow":
			has_info_ocf = info.get("operatingCashflow") is not None
			raise ValueError(
				"Couldn't find Operating Cash Flow TTM for "
				f"{ticker}. Direct field info['operatingCashflow'] present: {has_info_ocf}. "
				f"Cashflow mode: {cashflow_mode}, rows: {cash_rows}. "
				f"Rate-limited detected: {rate_limited_flag}. "
				"If True, Yahoo is throttling requests; wait a bit and try again."
			)
		if metric_name == "FCF (Free Cash Flow)":
			has_info_fcf = info.get("freeCashflow") is not None
			raise ValueError(
				"Couldn't find Free Cash Flow TTM for "
				f"{ticker}. Direct field info['freeCashflow'] present: {has_info_fcf}. "
				f"Cashflow mode: {cashflow_mode}, rows: {cash_rows}. "
				f"Rate-limited detected: {rate_limited_flag}. "
				"If True, Yahoo is throttling requests; wait a bit and try again."
			)
		raise ValueError(
			f"Couldn't find TTM data for '{metric_name}' on {ticker}. "
			f"Cashflow mode: {cashflow_mode}, rows: {cash_rows}. "
			f"Income mode: {income_mode}, rows: {income_rows}. "
			f"Rate-limited detected: {rate_limited_flag}."
		)

	if price is None:
		raise ValueError(f"Couldn't load market price for {ticker}.")

	if shares is None:
		raise ValueError(f"Couldn't load shares outstanding for {ticker}.")

	market_cap = price * shares
	current_ttm_multiple = None
	if metric_value > 0:
		current_ttm_multiple = market_cap / metric_value

	return StockSnapshot(
		ticker=ticker,
		price=price,
		shares_outstanding=shares,
		metric_ttm=metric_value,
		metric_name=metric_name,
		current_ttm_multiple=current_ttm_multiple,
	)


def project_terminal_per_share(
	metric_ttm: float,
	shares_outstanding: float,
	years: int,
	growth_rate: float,
	buyback_rate: float,
	exit_multiple: float,
):
	metrics = [metric_ttm]
	shares = [shares_outstanding]

	for _ in range(years):
		metrics.append(metrics[-1] * (1 + growth_rate))
		shares.append(max(shares[-1] * (1 - buyback_rate), shares_outstanding * 0.1))

	terminal_metric = metrics[-1]
	terminal_value = terminal_metric * exit_multiple
	terminal_per_share = terminal_value / shares[-1]

	return terminal_per_share, metrics, shares


def project_terminal_per_share_with_growth_path(
	metric_ttm: float,
	shares_outstanding: float,
	growth_rates: list[float],
	buyback_rate: float,
	exit_multiple: float,
):
	metrics = [metric_ttm]
	shares = [shares_outstanding]

	for growth_rate in growth_rates:
		metrics.append(metrics[-1] * (1 + growth_rate))
		shares.append(max(shares[-1] * (1 - buyback_rate), shares_outstanding * 0.1))

	terminal_metric = metrics[-1]
	terminal_value = terminal_metric * exit_multiple
	terminal_per_share = terminal_value / shares[-1]

	return terminal_per_share, metrics, shares


def implied_growth_rate(
	metric_ttm: float,
	shares_outstanding: float,
	current_price: float,
	desired_return: float,
	years: int,
	buyback_rate: float,
	exit_multiple: float,
) -> float | None:
	target_terminal_price = current_price * (1 + desired_return) ** years

	low, high = -0.30, 0.60
	best = None

	for _ in range(80):
		mid = (low + high) / 2
		terminal_per_share, _, _ = project_terminal_per_share(
			metric_ttm=metric_ttm,
			shares_outstanding=shares_outstanding,
			years=years,
			growth_rate=mid,
			buyback_rate=buyback_rate,
			exit_multiple=exit_multiple,
		)
		diff = terminal_per_share - target_terminal_price
		best = mid
		if abs(diff) < 1e-5:
			break
		if diff < 0:
			low = mid
		else:
			high = mid

	return best


def dcf_enterprise_value(
	fcf_ttm: float,
	years: int,
	explicit_growth_rate: float,
	discount_rate: float,
	terminal_growth_rate: float,
) -> tuple[float, float, float, list[float], list[float]]:
	projected_fcfs = [fcf_ttm * (1 + explicit_growth_rate) ** year for year in range(1, years + 1)]
	discounted_fcfs = [
		fcf / (1 + discount_rate) ** year for year, fcf in enumerate(projected_fcfs, start=1)
	]
	sum_discounted_fcfs = float(sum(discounted_fcfs))

	terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
	terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
	pv_terminal_value = terminal_value / (1 + discount_rate) ** years

	enterprise_value = sum_discounted_fcfs + pv_terminal_value
	return enterprise_value, sum_discounted_fcfs, pv_terminal_value, projected_fcfs, discounted_fcfs


def implied_fcf_growth_rate(
	fcf_ttm: float,
	target_enterprise_value: float,
	years: int,
	discount_rate: float,
	terminal_growth_rate: float,
) -> float:
	low, high = -0.50, 0.60
	best = 0.0

	for _ in range(90):
		mid = (low + high) / 2
		ev, _, _, _, _ = dcf_enterprise_value(
			fcf_ttm=fcf_ttm,
			years=years,
			explicit_growth_rate=mid,
			discount_rate=discount_rate,
			terminal_growth_rate=terminal_growth_rate,
		)
		diff = ev - target_enterprise_value
		best = mid
		if abs(diff) < 1e-5:
			break
		if diff < 0:
			low = mid
		else:
			high = mid

	return best


def fmt_money(value: float) -> str:
	return f"${value:,.2f}"


def fmt_billions(value: float) -> str:
	return f"${value / 1_000_000_000:,.2f}B"


def fmt_big_number(value: float) -> str:
	abs_value = abs(value)
	if abs_value >= 1_000_000_000_000:
		return f"{value / 1_000_000_000_000:.2f}T"
	if abs_value >= 1_000_000_000:
		return f"{value / 1_000_000_000:.2f}B"
	if abs_value >= 1_000_000:
		return f"{value / 1_000_000:.2f}M"
	if abs_value >= 1_000:
		return f"{value / 1_000:.2f}K"
	return f"{value:.2f}"


def _apply_light_mode_styles() -> None:
	pio.templates.default = "plotly_white"
	st.markdown(
		"""
		<style>
		:root, html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
			background-color: #ffffff !important;
			color: #0f172a !important;
		}
		section.main, section.main * {
			color: #0f172a !important;
		}
		label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
			color: #0f172a !important;
		}
		.stButton > button, .stDownloadButton > button {
			background: #f8fafc !important;
			color: #0f172a !important;
			border: 1px solid #e2e8f0 !important;
		}
		.stTextInput input, .stNumberInput input, .stTextArea textarea {
			background: #ffffff !important;
			color: #0f172a !important;
			border: 1px solid #e2e8f0 !important;
		}
		.stSelectbox div[data-baseweb="select"],
		.stRadio div[data-baseweb="radio"],
		.stSelectbox div[data-baseweb="select"] * {
			color: #0f172a !important;
			background-color: #ffffff !important;
		}
		div[data-baseweb="popover"],
		div[data-baseweb="popover"] * {
			background-color: #ffffff !important;
			color: #0f172a !important;
		}
		[data-testid="stToggle"] label, [data-testid="stToggle"] span {
			color: #0f172a !important;
		}
		[data-testid="stToggle"] div[role="switch"] {
			background-color: #e2e8f0 !important;
		}
		[data-testid="stToggle"] div[role="switch"] > div {
			background-color: #0f172a !important;
		}
		[data-testid="stSlider"] * {
			color: #0f172a !important;
		}
		[data-testid="stSidebar"] {
			border-right: 1px solid #e5e7eb !important;
		}
		[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
			color: #0f172a !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _build_valuation_pdf(
	ticker: str,
	valuation_params: dict,
	reverse_params: dict | None,
) -> bytes:
	try:
		from reportlab.lib import colors
		from reportlab.lib.pagesizes import letter
		from reportlab.lib.styles import getSampleStyleSheet
		from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
		from reportlab.graphics.charts.lineplots import LinePlot
		from reportlab.graphics.shapes import Drawing, String
		from reportlab.graphics.widgets.markers import makeMarker
	except Exception as exc:
		raise RuntimeError(
			"PDF export requires 'reportlab'. Install it with: pip install reportlab"
		) from exc

	snapshot: StockSnapshot = valuation_params["snapshot"]
	years = int(valuation_params["years"])
	growth_rates = valuation_params["growth_rates"]
	buyback_rate = float(valuation_params["buyback_rate"])
	exit_multiple = float(valuation_params["exit_multiple"])
	dividend_yield = float(valuation_params["dividend_yield"])
	dividend_growth_rate = float(valuation_params["dividend_growth_rate"])
	desired_return = float(valuation_params["desired_return"])
	metric_name = valuation_params["metric_name"]

	starting_metric_billions = float(
		st.session_state.get("valuation_starting_metric_billions", snapshot.metric_ttm / 1_000_000_000)
	)
	projection_metric_ttm = starting_metric_billions * 1_000_000_000

	terminal_per_share, metrics, shares = project_terminal_per_share_with_growth_path(
		metric_ttm=projection_metric_ttm,
		shares_outstanding=snapshot.shares_outstanding,
		growth_rates=growth_rates,
		buyback_rate=buyback_rate,
		exit_multiple=exit_multiple,
	)
	annual_dividend_per_share = float(snapshot.price) * float(dividend_yield) if dividend_yield > 0 else 0.0
	projected_dividends_per_share = [
		max(0.0, annual_dividend_per_share * ((1 + dividend_growth_rate) ** year_idx))
		for year_idx in range(years)
	]
	cumulative_dividends_per_share = float(sum(projected_dividends_per_share))
	total_terminal_per_share = terminal_per_share + cumulative_dividends_per_share

	expected_return = (
		(total_terminal_per_share / snapshot.price) ** (1 / years) - 1
		if years > 0
		else 0.0
	)
	fair_value_today = (
		total_terminal_per_share / (1 + desired_return) ** years if years > 0 else total_terminal_per_share
	)
	entry_gap = fair_value_today / snapshot.price - 1 if snapshot.price > 0 else 0.0
	entry_price_required = fair_value_today

	years_axis = list(range(0, years + 1))
	price_projection = [
		((metric_value * exit_multiple) / share_value) if share_value > 0 else np.nan
		for metric_value, share_value in zip(metrics, shares)
	]
	chart_points = [
		(x, y)
		for x, y in zip(years_axis, price_projection)
		if y is not None and not np.isnan(y)
	]

	growth_values_pct = [g * 100 for g in growth_rates]
	avg_growth = float(np.mean(growth_values_pct)) if growth_values_pct else 0.0
	growth_range = (
		f"{min(growth_values_pct):.1f}% to {max(growth_values_pct):.1f}%"
		if growth_values_pct
		else "N/A"
	)

	def _kv_table(title: str, rows: list[tuple[str, str]]):
		data = [[title, ""]]
		data += [[k, v] for k, v in rows]
		table = Table(data, colWidths=[250, 290])
		table.setStyle(
			TableStyle(
				[
					("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
					("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
					("SPAN", (0, 0), (-1, 0)),
					("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
					("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
					("FONTSIZE", (0, 0), (-1, -1), 10),
					("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
					("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				]
			)
		)
		return table

	styles = getSampleStyleSheet()
	elements = []
	elements.append(Paragraph("Simple Invest — Valuation Export", styles["Title"]))
	elements.append(Paragraph(f"Ticker: {ticker}", styles["Heading2"]))
	elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
	elements.append(Spacer(1, 12))

	highlight_rows = [
		("Expected CAGR", f"{expected_return * 100:.2f}%"),
		("Entry price for desired return", fmt_money(entry_price_required)),
		("Fair value today", fmt_money(fair_value_today)),
		("Terminal price (incl. dividends)", fmt_money(total_terminal_per_share)),
	]
	highlight_table = Table([[k, v] for k, v in highlight_rows], colWidths=[260, 280])
	expected_color = colors.HexColor("#16a34a") if expected_return > 0.10 else colors.HexColor("#dc2626")
	highlight_table.setStyle(
		TableStyle(
			[
				("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
				("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
				("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
				("FONTSIZE", (0, 0), (-1, -1), 11),
				("TEXTCOLOR", (1, 0), (1, 0), expected_color),
				("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
				("ALIGN", (1, 0), (1, -1), "RIGHT"),
				("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				("TOPPADDING", (0, 0), (-1, -1), 6),
				("BOTTOMPADDING", (0, 0), (-1, -1), 6),
			]
		)
	)
	elements.append(Paragraph("Highlights", styles["Heading3"]))
	elements.append(highlight_table)
	elements.append(Spacer(1, 14))

	if chart_points:
		drawing = Drawing(520, 220)
		line_plot = LinePlot()
		line_plot.x = 40
		line_plot.y = 30
		line_plot.width = 450
		line_plot.height = 160
		line_plot.data = [chart_points]
		line_plot.lines[0].strokeColor = colors.HexColor("#2563EB")
		line_plot.lines[0].strokeWidth = 2
		line_plot.lines[0].symbol = makeMarker("FilledCircle")
		line_plot.lines[0].symbol.size = 4

		x_values = [pt[0] for pt in chart_points]
		y_values = [pt[1] for pt in chart_points]
		line_plot.xValueAxis.valueMin = min(x_values)
		line_plot.xValueAxis.valueMax = max(x_values)
		line_plot.xValueAxis.valueSteps = x_values
		min_y = min(y_values)
		max_y = max(y_values)
		padding = max(1.0, (max_y - min_y) * 0.1)
		line_plot.yValueAxis.valueMin = max(0.0, min_y - padding)
		line_plot.yValueAxis.valueMax = max_y + padding

		drawing.add(String(0, 200, "Projected stock price path", fontName="Helvetica-Bold", fontSize=12))
		drawing.add(line_plot)
		elements.append(drawing)
		elements.append(Spacer(1, 12))

	valuation_rows = [
		("Metric", metric_name),
		("Current price", fmt_money(snapshot.price)),
		(
			"Current TTM multiple",
			f"{snapshot.current_ttm_multiple:,.2f}x" if snapshot.current_ttm_multiple else "N/A",
		),
		("Starting metric (billions)", f"{starting_metric_billions:.2f}"),
		("Model horizon (years)", str(years)),
		("Growth rate (avg)", f"{avg_growth:.2f}%"),
		("Growth rate (range)", growth_range),
		("Buyback rate", f"{buyback_rate * 100:.2f}%"),
		("Exit multiple", f"{exit_multiple:.2f}x"),
		("Dividend yield", f"{dividend_yield * 100:.2f}%"),
		("Dividend growth", f"{dividend_growth_rate * 100:.2f}%"),
		("Expected CAGR", f"{expected_return * 100:.2f}%"),
		("Fair value today", fmt_money(fair_value_today)),
		("Entry price gap", f"{entry_gap * 100:.2f}% vs current"),
		("Terminal price (incl. dividends)", fmt_money(total_terminal_per_share)),
	]
	elements.append(_kv_table("Valuation Summary", valuation_rows))
	elements.append(Spacer(1, 14))

	if reverse_params:
		starting_fcf_billions = float(reverse_params["starting_fcf_billions"])
		effective_starting_fcf = starting_fcf_billions * 1_000_000_000
		projection_years = int(reverse_params["projection_years"])
		explicit_growth_rate = float(reverse_params["explicit_growth_rate_pct"]) / 100
		terminal_growth_rate = float(reverse_params["terminal_growth_rate_pct"]) / 100
		discount_rate = float(reverse_params["discount_rate_pct"]) / 100
		cash = float(reverse_params["cash"])
		total_debt = float(reverse_params["total_debt"])

		target_enterprise_value = snapshot.price * snapshot.shares_outstanding + total_debt - cash
		reverse_growth_rate = implied_fcf_growth_rate(
			fcf_ttm=effective_starting_fcf,
			target_enterprise_value=target_enterprise_value,
			years=projection_years,
			discount_rate=discount_rate,
			terminal_growth_rate=terminal_growth_rate,
		)

		enterprise_value, sum_discounted_fcfs, pv_terminal_value, _, _ = dcf_enterprise_value(
			fcf_ttm=effective_starting_fcf,
			years=projection_years,
			explicit_growth_rate=explicit_growth_rate,
			discount_rate=discount_rate,
			terminal_growth_rate=terminal_growth_rate,
		)
		equity_value = enterprise_value + cash - total_debt
		intrinsic_value_per_share = equity_value / snapshot.shares_outstanding
		relative_gap = intrinsic_value_per_share / snapshot.price - 1 if snapshot.price > 0 else 0.0

		reverse_rows = [
			("Starting FCF (billions)", f"{starting_fcf_billions:.2f}"),
			("Projection years", str(projection_years)),
			("Explicit growth rate", f"{explicit_growth_rate * 100:.2f}%"),
			("Terminal growth rate", f"{terminal_growth_rate * 100:.2f}%"),
			("Discount rate", f"{discount_rate * 100:.2f}%"),
			("Sum discounted FCF", fmt_billions(sum_discounted_fcfs)),
			("PV terminal value", fmt_billions(pv_terminal_value)),
			("Enterprise value", fmt_billions(enterprise_value)),
			("Equity value", fmt_billions(equity_value)),
			("Intrinsic value / share", fmt_money(intrinsic_value_per_share)),
			("Value vs market", f"{relative_gap * 100:.2f}%"),
			("Reverse implied growth", f"{reverse_growth_rate * 100:.2f}%"),
		]
		elements.append(_kv_table("Reverse DCF Summary", reverse_rows))
	else:
		elements.append(Paragraph("Reverse DCF Summary: Not available.", styles["Normal"]))

	buffer = io.BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=letter)
	doc.build(elements)
	return buffer.getvalue()


def render_colored_value_metric(label: str, value_text: str, color: str, subtitle: str | None = None) -> None:
	label_color = "#0f172a" if st.session_state.get("ui_light_mode") else "#ffffff"
	st.markdown(
		f"""
		<div style=\"padding: 0.1rem 0.2rem;\">
			<div style=\"font-size: var(--font-size-sm, 0.875rem); color: {label_color};\">{label}</div>
			<div style=\"font-size: var(--font-size-2xl, 1.5rem); font-weight: 600; color: {color}; line-height: 1.2;\">{value_text}</div>
			{f'<div style=\"font-size: var(--font-size-sm, 0.875rem); color: {label_color};\">{subtitle}</div>' if subtitle else ''}
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_valuation_tab(
	ticker: str,
	saved_df: pd.DataFrame | None = None,
	data_provider: str = "Yahoo Finance",
	data_cache_df: pd.DataFrame | None = None,
	data_cache_source: str = "local-cache",
):
	# st.subheader("Valuation Model with multiples")

	control_col, graph_col = st.columns([0.8, 2.2], gap="large")

	with control_col:
		cached_provider = _analysis_get(data_cache_df, "meta", "provider") or data_provider
		# st.caption(f"Data source: {cached_provider} ({data_cache_source})")

		desired_return = (
			st.number_input(
				"Desired annual return (%)",
				min_value=8.0,
				max_value=30.0,
				value=15.0,
				step=0.5,
				key="valuation_desired_return_pct",
			)
			/ 100
		)
		metric_name = st.selectbox(
			"TTM metric",
			list(METRIC_LABEL_CANDIDATES.keys()),
			index=0,
			key="valuation_metric_name",
		)
		owner_adjustments_enabled = st.toggle(
			"Owner Adjustments",
			help=(
				"Uses owner-adjusted values from get_owner_earnings(): Owner FCF, Owner OCF, or Owner Earnings. "
				"This only affects FCF, Operating Cash Flow, and Earnings (Net Income). "
				"Revenue and Operating Income are unchanged."
			),
			key="valuation_owner_adjustments",
		)

		loaded_from_local_fallback = False
		saved_snapshot_for_metric = get_saved_snapshot(saved_df, metric_name)
		snapshot = get_saved_snapshot(data_cache_df, metric_name)
		if snapshot is None and saved_snapshot_for_metric is not None:
			snapshot = saved_snapshot_for_metric
			loaded_from_local_fallback = True
			st.warning("Metric unavailable in provider cache. Loaded locally saved analysis data for this metric.")

		if snapshot is None:
			st.error(
				f"Could not load '{metric_name}' for {ticker} from provider cache. Try switching provider or force-refresh by changing ticker/provider."
			)
			return None

		if saved_snapshot_for_metric is not None and not loaded_from_local_fallback:
			same_metric = abs(snapshot.metric_ttm - saved_snapshot_for_metric.metric_ttm) <= max(
				1e-6, abs(saved_snapshot_for_metric.metric_ttm) * 1e-6
			)
			if not same_metric:
				st.caption(
					f"Saved local data differs {int(saved_snapshot_for_metric.metric_ttm*1e-6)}M for this metric {int(snapshot.metric_ttm*1e-6)}M"
				)

		standard_metric_ttm = float(snapshot.metric_ttm)
		standard_current_ttm_multiple = snapshot.current_ttm_multiple

		owner_adjustments_applicable = metric_name in {
			"FCF (Free Cash Flow)",
			"Operating Cash Flow",
			"Earnings (Net Income)",
		}
		owner_adjustment_applied = False
		owner_adjustment_message: str | None = None
		sbc_ttm_cached = _parse_float(_analysis_get(data_cache_df, "company_data", "ttm_sbc"), np.nan)

		if owner_adjustments_enabled and owner_adjustments_applicable:
			owner_adjusted_value = None

			if metric_name in {"FCF (Free Cash Flow)", "Operating Cash Flow"}:
				if np.isnan(sbc_ttm_cached):
					st.warning("TTM SBC is unavailable in cache; using standard metric instead.")
					owner_adjustment_message = "Owner adjustment unavailable: TTM SBC missing in cache."
				else:
					owner_adjusted_value = max(0.0, standard_metric_ttm - float(sbc_ttm_cached))
					# owner_adjustment_message = (
					# 	#f"Owner adjustment active ({metric_name}): standard TTM {fmt_billions(standard_metric_ttm)} "
					# 	f"- SBC {fmt_billions(float(sbc_ttm_cached))} = {fmt_billions(owner_adjusted_value)}."
					# )
			elif metric_name == "Earnings (Net Income)":
				owner_adjusted_value = load_owner_adjusted_metric(ticker=ticker, metric_name=metric_name)
				if owner_adjusted_value is None:
					st.warning("Owner-adjusted earnings unavailable right now; using standard metric instead.")
					owner_adjustment_message = "Owner adjustment unavailable for Net Income right now; using standard metric."
				else:
					owner_adjustment_message = "Owner adjustment active (Earnings): using Owner Earnings from owner_earnings model."

			if owner_adjusted_value is not None:
				owner_adjusted_value = min(float(owner_adjusted_value), standard_metric_ttm)
				owner_multiple = None
				if owner_adjusted_value > 0:
					owner_multiple = (snapshot.price * snapshot.shares_outstanding) / owner_adjusted_value

				snapshot = StockSnapshot(
					ticker=snapshot.ticker,
					price=snapshot.price,
					shares_outstanding=snapshot.shares_outstanding,
					metric_ttm=float(owner_adjusted_value),
					metric_name=snapshot.metric_name,
					current_ttm_multiple=owner_multiple,
				)
				owner_adjustment_applied = True
		elif owner_adjustments_enabled and not owner_adjustments_applicable:
			st.caption("Owner Adjustments: no effect for this metric (only FCF, OCF, and Net Income).")

		starting_metric_source = st.radio(
			"Starting metric source",
			["TTM", "Previous full year"],
			horizontal=True,
			key="valuation_starting_metric_source",
		)

		# Use the unadjusted standard base first, then apply owner adjustments once (if enabled).
		base_metric_for_projection = float(standard_metric_ttm)
		if starting_metric_source == "Previous full year":
			prev_year_value = _get_previous_full_year_metric_from_cache(data_cache_df, metric_name)
			if prev_year_value is None:
				st.warning(f"Previous full-year {metric_name} unavailable in cache; using TTM instead.")
			else:
				base_metric_for_projection = float(prev_year_value)

		starting_metric_adjustment_message: str | None = None
		if owner_adjustments_enabled and metric_name in {"FCF (Free Cash Flow)", "Operating Cash Flow"}:
			if np.isnan(sbc_ttm_cached):
				st.warning("Owner-adjusted starting value unavailable: TTM SBC missing in cache.")
				starting_metric_adjustment_message = "Starting metric owner adjustment unavailable: TTM SBC missing in cache."
			else:
				base_metric_for_projection = max(0.0, base_metric_for_projection - float(sbc_ttm_cached))
				starting_metric_adjustment_message = (
					f"Adjusted: -{fmt_billions(float(sbc_ttm_cached))} SBC applied."
				)

		start_metric_context = (
			f"{ticker}:{metric_name}:{starting_metric_source}:{int(owner_adjustments_enabled)}:"
			f"{round(base_metric_for_projection, 2)}"
		)
		default_starting_metric_billions = max(0.0, base_metric_for_projection / 1_000_000_000)
		if st.session_state.get("valuation_starting_metric_context") != start_metric_context:
			st.session_state["valuation_starting_metric_billions"] = float(default_starting_metric_billions)
			st.session_state["valuation_starting_metric_context"] = start_metric_context

		starting_metric_billions = st.number_input(
			"Starting metric ($ billions)",
			min_value=0.0,
			step=0.1,
			key="valuation_starting_metric_billions",
		)
		if owner_adjustments_enabled and owner_adjustment_message:
			st.caption(owner_adjustment_message)
		if starting_metric_adjustment_message:
			st.caption(starting_metric_adjustment_message)

		# Explicit projection base metric used for all downstream calculations (CAGR, chart, exit multiples).
		projection_metric_ttm = float(starting_metric_billions) * 1_000_000_000

		years = int(
			st.number_input(
				"Model horizon",
				min_value=5,
				max_value=20,
				step=1,
				key="valuation_years",
			)
		)

		dividend_profile = load_dividend_profile(ticker=ticker, current_price=float(snapshot.price))
		has_dividend = bool(dividend_profile.get("has_dividend", False))
		dividend_yield = float(dividend_profile.get("dividend_yield", 0.0))
		detected_dividend_yield = float(dividend_yield)
		dividend_rate = float(dividend_profile.get("dividend_rate", 0.0))
		trailing_annual_dividend_per_share = float(
			dividend_profile.get("trailing_annual_dividend_per_share", 0.0)
		)
		dividend_cagr_1y = _safe_float(dividend_profile.get("dividend_cagr_1y"), np.nan)
		dividend_cagr_3y = _safe_float(dividend_profile.get("dividend_cagr_3y"), np.nan)
		dividend_cagr_5y = _safe_float(dividend_profile.get("dividend_cagr_5y"), np.nan)
		has_dividend_event_signal = bool(dividend_profile.get("has_dividend_event_signal", False))
		default_dividend_growth_rate = float(dividend_profile.get("default_dividend_growth_rate", 0.0))

		dividend_growth_rate = 0.0
		if has_dividend:
			dividend_context = (
				f"{ticker}:{round(dividend_yield, 6)}:{round(default_dividend_growth_rate, 6)}"
			)
			if st.session_state.get("valuation_dividend_growth_context") != dividend_context:
				computed_default_pct = float(default_dividend_growth_rate * 100)
				existing_growth_pct = st.session_state.get("valuation_dividend_growth_rate_pct")
				if existing_growth_pct is None or abs(computed_default_pct) > 1e-9:
					st.session_state["valuation_dividend_growth_rate_pct"] = computed_default_pct
				st.session_state["valuation_dividend_growth_context"] = dividend_context

			
			dividend_growth_rate = (
				st.number_input(
					"Dividend growth (%)",
					step=0.25,
					key="valuation_dividend_growth_rate_pct",
				)
				/ 100
			)
			def _fmt_pct_or_na(value: float) -> str:
				return "N/A" if np.isnan(value) else f"{value * 100:.2f}%"

			st.caption(
				f"Default dividend CAGR: {default_dividend_growth_rate * 100:.2f}%")
		else:
			dividend_growth_rate = 0.0
			dividend_yield = 0.0

		use_custom_growth = st.toggle(
			"Custom growth by year",
			help="Enable to input a different FCF/metric growth rate for each projection year.",
			key="valuation_use_custom_growth",
		)

		growth_rates: list[float] = []
		if use_custom_growth:
			st.caption("Custom yearly growth inputs")
			default_growth_pct = float(st.session_state.get("valuation_growth_rate_pct", 10.0))
			for year_idx in range(1, years + 1):
				year_key = f"valuation_custom_growth_{year_idx}"
				if year_key not in st.session_state:
					st.session_state[year_key] = default_growth_pct
				year_growth = (
					st.number_input(
						f"Year {year_idx} growth (%)",
						step=0.5,
						key=year_key,
					)
					/ 100
				)
				growth_rates.append(year_growth)
		else:
			growth_rate = (
				st.slider(
					"Growth rate (%)",
					min_value=-2.0,
					value=8.0,
					max_value=35.0,
					step=0.5,
					key="valuation_growth_rate_pct",
				)
				/ 100
			)
			growth_rates = [growth_rate] * years

		buyback_rate = (
			st.slider(
				"Share reduction (%) or dilution if negative",
				-10.0,
				10.0,
				# value=float(st.session_state.get("valuation_buyback_rate_pct", 1.0)),
				value=0.0,
				step=0.25,
				key="valuation_buyback_rate_pct",
			)
			/ 100
		)
		default_exit_multiple = (
			snapshot.current_ttm_multiple if snapshot.current_ttm_multiple is not None else 18.0
		)
		slider_max = max(40.0, min(120.0, default_exit_multiple * 2.0))
		default_exit_multiple = min(max(default_exit_multiple, 1.0), slider_max)
		exit_context_key = f"{metric_name}:{int(owner_adjustment_applied)}"
		if st.session_state.get("valuation_exit_multiple_context") != exit_context_key:
			st.session_state["valuation_exit_multiple"] = float(default_exit_multiple)
			st.session_state["valuation_exit_multiple_context"] = exit_context_key
		exit_multiple = st.slider(
			"Exit multiple (x)",
			min_value=1.0,
			max_value=float(slider_max),
			value=float(st.session_state.get("valuation_exit_multiple", default_exit_multiple)),
			step=0.5,
			key="valuation_exit_multiple",
		)

	terminal_per_share, metrics, shares = project_terminal_per_share_with_growth_path(
		metric_ttm=projection_metric_ttm,
		shares_outstanding=snapshot.shares_outstanding,
		growth_rates=growth_rates,
		buyback_rate=buyback_rate,
		exit_multiple=exit_multiple,
	)
	annual_dividend_per_share = float(snapshot.price) * float(dividend_yield) if dividend_yield > 0 else 0.0
	projected_dividends_per_share = [
		max(0.0, annual_dividend_per_share * ((1 + dividend_growth_rate) ** year_idx))
		for year_idx in range(years)
	]
	cumulative_dividends_per_share = float(sum(projected_dividends_per_share))
	total_terminal_per_share = terminal_per_share + cumulative_dividends_per_share

	price_projection = [
		((metric_value * exit_multiple) / share_value) if share_value > 0 else np.nan
		for metric_value, share_value in zip(metrics, shares)
	]

	expected_return = (total_terminal_per_share / snapshot.price) ** (1 / years) - 1
	standard_terminal_per_share, _, _ = project_terminal_per_share_with_growth_path(
		metric_ttm=standard_metric_ttm,
		shares_outstanding=snapshot.shares_outstanding,
		growth_rates=growth_rates,
		buyback_rate=buyback_rate,
		exit_multiple=exit_multiple,
	)
	standard_total_terminal_per_share = standard_terminal_per_share + cumulative_dividends_per_share
	standard_expected_return = (standard_total_terminal_per_share / snapshot.price) ** (1 / years) - 1
	fair_value_today = total_terminal_per_share / (1 + desired_return) ** years
	upside_to_fair = fair_value_today / snapshot.price - 1
	entry_price_required = total_terminal_per_share / (1 + desired_return) ** years
	entry_gap = entry_price_required / snapshot.price - 1
	terminal_metric = metrics[-1] if metrics else None
	terminal_shares = shares[-1] if shares else None

	with graph_col:
		overview_stats = st.columns(4)
		overview_stats[0].metric("Company", snapshot.ticker)
		overview_stats[1].metric("Selected metric", snapshot.metric_name)
		ttm_multiple_text = (
			f"{snapshot.current_ttm_multiple:,.2f}x"
			if snapshot.current_ttm_multiple is not None
			else "N/A"
		)
		overview_stats[2].metric("Current TTM multiple", ttm_multiple_text)

		if use_custom_growth:
			growth_values_pct = [g * 100 for g in growth_rates]
			avg_growth = float(np.mean(growth_values_pct)) if growth_values_pct else 0.0
			growth_value = f"{avg_growth:.2f}%"
			growth_delta = f"range {min(growth_values_pct):.1f}% to {max(growth_values_pct):.1f}%"
		else:
			growth_value = f"{growth_rates[0] * 100:.2f}%"
			growth_delta = "constant"
		overview_stats[3].metric("Growth (%)", growth_value, growth_delta)
		# st.caption(f"Base metric used in projection: {fmt_big_number(projection_metric_ttm)}")
		if owner_adjustment_applied:
			base_delta_pct = (projection_metric_ttm / standard_metric_ttm - 1) * 100 if standard_metric_ttm else 0.0
			# st.caption(
			# 	f"Standard base: {fmt_big_number(standard_metric_ttm)} → Owner-adjusted base: {fmt_big_number(projection_metric_ttm)} ({base_delta_pct:+.2f}%)."
			# )

		if owner_adjustment_applied:
			st.caption("Owner Adjustments active: valuation is using owner-adjusted metric values.")
			#st.caption( 				f"Expected CAGR with standard base: {standard_expected_return * 100:.2f}% → owner-adjusted base: {expected_return * 100:.2f}%."
		# if has_dividend and cumulative_dividends_per_share > 0:
		# 	st.caption(
		# 		f"Projected cumulative dividends over {years} years (no reinvestment): {fmt_money(cumulative_dividends_per_share)}."
		# 	)

		years_axis = list(range(0, years + 1))
		growth_df = pd.DataFrame(
			{
				"Year": years_axis,
				"Metric": metrics,
			}
		)
		# fig = px.line(
		# 	growth_df,
		# 	x="Year",
		# 	y="Metric",
		# 	markers=True,
		# 	title=f"{snapshot.metric_name} growth projection",
		# )
		# fig.update_layout(yaxis_title="Projected metric", xaxis_title="Year")
		
		chart_refresh_key = (
			f"valuation_plot:{snapshot.ticker}:{snapshot.metric_name}:{int(owner_adjustment_applied)}:"
			f"{years}:{round(exit_multiple, 4)}:{round(buyback_rate, 4)}:{round(sum(growth_rates), 6)}"
		)
		# st.plotly_chart(fig, use_container_width=True, key=chart_refresh_key)

		price_df = pd.DataFrame(
			{
				"Year": years_axis,
				"Projected Stock Price": price_projection,
			}
		)
		price_fig = px.line(
			price_df,
			x="Year",
			y="Projected Stock Price",
			markers=True,
			title=f"Projected stock price path ({snapshot.ticker}) at {exit_multiple:.1f}x exit multiple",
		)
		price_fig.update_layout(yaxis_title="Projected stock price", xaxis_title="Year")
		st.plotly_chart(
			price_fig,
			use_container_width=True,
			key=f"valuation_price_plot:{chart_refresh_key}",
		)
 
		top_cols = st.columns(4)
		expected_return_color = "#16a34a" if expected_return > 0.10 else "#dc2626"
		with top_cols[0]:
			render_colored_value_metric(
				"Expected return from today's price (CAGR)",
				f"{expected_return * 100:.2f}%",
				expected_return_color,
				"Green above 10% (Market average)",
			)
		top_cols[1].metric("Current price", fmt_money(snapshot.price))
		top_cols[2].metric("Future stock price", fmt_money(terminal_per_share), f"in {years} years")
		entry_price_color = "#16a34a" if entry_price_required > snapshot.price else "#dc2626"
		with top_cols[3]:
			render_colored_value_metric(
				"Entry price for desired return",
				fmt_money(entry_price_required),
				entry_price_color,
				f"{entry_gap * 100:.2f}% vs current",
			)

		exit_mul_cols = st.columns(4)
		for idx, target_return in enumerate([0.10, 0.15, 0.20]):
			if terminal_metric is None or terminal_metric <= 0 or terminal_shares is None or terminal_shares <= 0:
				exit_mul_cols[idx].metric(f"Exit multiple @ {target_return * 100:.0f}%", "N/A")
				continue

			required_terminal_per_share = snapshot.price * (1 + target_return) ** years
			required_exit_multiple = (required_terminal_per_share * terminal_shares) / terminal_metric
			exit_mul_cols[idx].metric(
				f"Exit multiple @ {target_return * 100:.0f}%",
				f"{required_exit_multiple:,.2f}x",
			)
		# add current dividend yield
		if dividend_yield > 0:
			exit_mul_cols[3].metric(
				"Current dividend yield",
				f"{dividend_yield * 100:.2f}%",
			)



	return {
		"snapshot": snapshot,
		"metric_name": metric_name,
		"owner_adjustments_enabled": owner_adjustments_enabled,
		"owner_adjustment_applied": owner_adjustment_applied,
		"dividend_yield": dividend_yield,
		"dividend_growth_rate": dividend_growth_rate,
		"cumulative_dividends_per_share": cumulative_dividends_per_share,
		"growth_rates": growth_rates,
		"years": years,
		"desired_return": desired_return,
		"buyback_rate": buyback_rate,
		"exit_multiple": exit_multiple,
		"loaded_from_local_fallback": loaded_from_local_fallback,
	}


def render_reverse_dcf_tab(
	snapshot: StockSnapshot,
	ticker: str,
	saved_df: pd.DataFrame | None = None,
	data_cache_df: pd.DataFrame | None = None,
):
	st.subheader("Reverse DCF / Embedded Growth")

	left, right = st.columns([1.2, 1.8], gap="large")

	# Reverse DCF must be independent from the first valuation model.
	# Always source its starting FCF from cache/saved data (never from the possibly owner-adjusted valuation snapshot).
	fcf_snapshot = get_saved_snapshot(data_cache_df, "FCF (Free Cash Flow)")
	if fcf_snapshot is None:
		fcf_snapshot = get_saved_snapshot(saved_df, "FCF (Free Cash Flow)")
		if fcf_snapshot is not None:
			st.warning("Using locally saved FCF data for Reverse DCF.")

	if fcf_snapshot is None and snapshot.metric_name == "FCF (Free Cash Flow)":
		fcf_snapshot = snapshot
		st.warning(
			"Reverse DCF is temporarily using the current valuation FCF snapshot because no cached FCF data was found. "
			"Use 'Refresh data now' to restore full independence."
		)

	if fcf_snapshot is None:
		st.error("Reverse DCF requires cached FCF data, but none was found.")
		return None

	fcf_ttm = fcf_snapshot.metric_ttm
	previous_full_year_fcf = _parse_float(_analysis_get(data_cache_df, "company_data", "previous_full_year_fcf"), np.nan)
	ttm_sbc = _parse_float(_analysis_get(data_cache_df, "company_data", "ttm_sbc"), np.nan)
	cash = _parse_float(_analysis_get(data_cache_df, "company_data", "cash"), 0.0)
	total_debt = _parse_float(_analysis_get(data_cache_df, "company_data", "total_debt"), 0.0)

	if np.isnan(previous_full_year_fcf):
		previous_full_year_fcf = None
	if np.isnan(ttm_sbc):
		ttm_sbc = None

	if saved_df is not None:
		if previous_full_year_fcf is None:
			previous_full_year_fcf = _parse_float(_analysis_get(saved_df, "company_data", "previous_full_year_fcf"), np.nan)
			if np.isnan(previous_full_year_fcf):
				previous_full_year_fcf = None
		if ttm_sbc is None:
			ttm_sbc = _parse_float(_analysis_get(saved_df, "company_data", "ttm_sbc"), np.nan)
			if np.isnan(ttm_sbc):
				ttm_sbc = None
		cash = _parse_float(_analysis_get(saved_df, "company_data", "cash"), cash)
		total_debt = _parse_float(_analysis_get(saved_df, "company_data", "total_debt"), total_debt)

	with left:
		if "reverse_projection_years" not in st.session_state:
			st.session_state["reverse_projection_years"] = 10
		if "reverse_starting_fcf_source" not in st.session_state:
			st.session_state["reverse_starting_fcf_source"] = "TTM"
		if "reverse_apply_sbc_adjustment" not in st.session_state:
			st.session_state["reverse_apply_sbc_adjustment"] = False
		if "reverse_explicit_growth_rate_pct" not in st.session_state:
			st.session_state["reverse_explicit_growth_rate_pct"] = 8.0
		if "reverse_terminal_growth_rate_pct" not in st.session_state:
			st.session_state["reverse_terminal_growth_rate_pct"] = 3.0
		if "reverse_discount_rate_pct" not in st.session_state:
			st.session_state["reverse_discount_rate_pct"] = 9.0

		projection_years = int(
			st.number_input(
				"FCF projection years",
				min_value=3,
				max_value=30,
				step=1,
				key="reverse_projection_years",
			)
		)

		starting_fcf_source = st.radio(
			"Starting FCF source",
			["TTM", "Previous full year"],
			horizontal=True,
			index=0,
			key="reverse_starting_fcf_source",
		)

		base_starting_fcf = fcf_ttm
		if starting_fcf_source == "Previous full year":
			if previous_full_year_fcf is None:
				st.warning("Previous full-year FCF unavailable for this ticker right now. Using TTM FCF instead.")
			else:
				base_starting_fcf = previous_full_year_fcf

		apply_sbc_adjustment = st.toggle(
			"Subtract TTM SBC from starting FCF",
			help="If enabled, TTM Stock-Based Compensation is subtracted from starting FCF.",
			key="reverse_apply_sbc_adjustment",
		)

		sbc_adjustment = 0.0
		if apply_sbc_adjustment:
			if ttm_sbc is None:
				st.warning("TTM SBC is unavailable right now, so no SBC adjustment was applied.")
			else:
				sbc_adjustment = ttm_sbc

		default_starting_fcf_billions = max(0.0, (base_starting_fcf - sbc_adjustment) / 1_000_000_000)
		start_fcf_context = (
			f"{ticker}:{starting_fcf_source}:{int(apply_sbc_adjustment)}:"
			f"{round(float(base_starting_fcf), 2)}:{round(float(sbc_adjustment), 2)}"
		)
		if st.session_state.get("reverse_starting_fcf_context") != start_fcf_context:
			st.session_state["reverse_starting_fcf_billions"] = float(default_starting_fcf_billions)
			st.session_state["reverse_starting_fcf_context"] = start_fcf_context

		default_starting_fcf_billions = float(
			st.session_state.get("reverse_starting_fcf_billions", default_starting_fcf_billions)
		)
		starting_fcf_billions = st.number_input(
			"Starting FCF ($ billions)",
			min_value=0.0,
			value=default_starting_fcf_billions,
			step=0.5,
			key="reverse_starting_fcf_billions",
		)
		effective_starting_fcf = starting_fcf_billions * 1_000_000_000

		if ttm_sbc is None:
			st.caption("TTM SBC: N/A")
		else:
			st.caption(f"TTM SBC: {fmt_billions(sbc_adjustment)}")

		explicit_growth_rate = (
			st.slider(
				"FCF growth rate (%)",
				min_value=-10.0,
				max_value=30.0,
				step=0.25,
				key="reverse_explicit_growth_rate_pct",
			)
			/ 100
		)
		terminal_growth_rate = (
			st.slider(
				"Perpetual growth rate / terminal value (%)",
				min_value=0.0,
				max_value=6.0,
				step=0.2,
				key="reverse_terminal_growth_rate_pct",
			)
			/ 100
		)
		discount_rate = (
			st.slider(
				"Discount rate (%)",
				min_value=5.0,
				max_value=15.0,
				step=0.5,
				key="reverse_discount_rate_pct",
			)
			/ 100
		)

	if discount_rate <= terminal_growth_rate:
		st.warning("Discount rate must be higher than perpetual growth rate for a valid DCF terminal value.")
		return None

	target_enterprise_value = snapshot.price * snapshot.shares_outstanding + total_debt - cash
	reverse_growth_rate = implied_fcf_growth_rate(
		fcf_ttm=effective_starting_fcf,
		target_enterprise_value=target_enterprise_value,
		years=projection_years,
		discount_rate=discount_rate,
		terminal_growth_rate=terminal_growth_rate,
	)

	(
		enterprise_value,
		sum_discounted_fcfs,
		pv_terminal_value,
		projected_fcfs,
		discounted_fcfs,
	) = dcf_enterprise_value(
		fcf_ttm=effective_starting_fcf,
		years=projection_years,
		explicit_growth_rate=explicit_growth_rate,
		discount_rate=discount_rate,
		terminal_growth_rate=terminal_growth_rate,
	)

	equity_value = enterprise_value + cash - total_debt
	intrinsic_value_per_share = equity_value / snapshot.shares_outstanding
	relative_gap = intrinsic_value_per_share / snapshot.price - 1

	with right:
		projected_fcfs_with_year0 = [effective_starting_fcf, *projected_fcfs]
		discounted_fcfs_with_year0 = [effective_starting_fcf, *discounted_fcfs]
		chart_df = pd.DataFrame(
			{
				"Year": list(range(0, projection_years + 1)),
				"Projected FCF": projected_fcfs_with_year0,
				"Discounted FCF": discounted_fcfs_with_year0,
			}
		)
		fig = px.bar(
			chart_df,
			x="Year",
			y=["Projected FCF", "Discounted FCF"],
			barmode="group",
			title=f"Reverse DCF cash flow path ({ticker}) - {projection_years} years",
		)
		fig.update_layout(yaxis_title="FCF", xaxis_title="Year")
		reverse_chart_key = (
			f"reverse_plot:{ticker}:{projection_years}:{round(starting_fcf_billions,4)}:"
			f"{round(explicit_growth_rate,6)}:{round(terminal_growth_rate,6)}:{round(discount_rate,6)}:{int(apply_sbc_adjustment)}"
		)
		st.plotly_chart(fig, use_container_width=True, key=reverse_chart_key)

		st.caption(
			f"Starting FCF (Year 0): {fmt_billions(effective_starting_fcf)} | Explicit growth over {projection_years} years: {explicit_growth_rate * 100:.2f}%."
		)
		if apply_sbc_adjustment and sbc_adjustment > 0:
			st.caption(f"SBC adjustment applied: -{fmt_billions(sbc_adjustment)} from TTM SBC.")
		# st.caption(
		# 	f"Reverse FCF growth embedded in market price (holding other assumptions fixed): {reverse_growth_rate * 100:.2f}%"
		# )

		summary_col_1, summary_col_2 = st.columns(2, gap="small")
		with summary_col_1:
			if relative_gap > 0:
				st.metric("Undervalued by", f"{relative_gap * 100:.2f}%")
			elif relative_gap < 0:
				st.metric("Overvalued by", f"{abs(relative_gap) * 100:.2f}%")
			else:
				st.metric("Fairly valued", "0.00%")

			st.metric("Reverse implied growth", f"{reverse_growth_rate * 100:.2f}%")

		with summary_col_2:
			st.metric(
				"Intrinsic value / share",
				fmt_money(intrinsic_value_per_share),
				f"{relative_gap * 100:.2f}% vs market",
				delta_color="normal",
			)
			st.metric("Market price / share", fmt_money(snapshot.price))

	with left:
		metric_col_1, metric_col_2 = st.columns(2, gap="small")
		with metric_col_1:
			st.metric("Sum of discounted FCF", fmt_billions(sum_discounted_fcfs))
			st.metric("Total debt", fmt_billions(total_debt))
			st.metric("Current shares outstanding", fmt_big_number(snapshot.shares_outstanding))
		with metric_col_2:
			st.metric("Cash and cash equivalents", fmt_billions(cash))
			st.metric("Calculated equity value", fmt_billions(equity_value))
			st.metric(
				"Calculated intrinsic value (per share)",
				fmt_money(intrinsic_value_per_share),
				f"vs current {snapshot.price:,.2f}",
			)

	return {
		"projection_years": projection_years,
		"starting_fcf_source": starting_fcf_source,
		"apply_sbc_adjustment": apply_sbc_adjustment,
		"starting_fcf_billions": starting_fcf_billions,
		"explicit_growth_rate_pct": explicit_growth_rate * 100,
		"terminal_growth_rate_pct": terminal_growth_rate * 100,
		"discount_rate_pct": discount_rate * 100,
		"cash": cash,
		"total_debt": total_debt,
		"previous_full_year_fcf": previous_full_year_fcf,
		"ttm_sbc": ttm_sbc,
	}


def main():
	# st.title("Business Valuation")

	if "ui_light_mode" not in st.session_state:
		st.session_state["ui_light_mode"] = False
	if st.session_state.get("ui_light_mode"):
		_apply_light_mode_styles()

	with st.sidebar:
		if "selected_view" not in st.session_state:
			st.session_state["selected_view"] = "Stock Valuation"

		current_view = st.session_state["selected_view"]
		stock_color = "#6EA8FF" if current_view == "Stock Valuation" else "inherit"
		portfolio_color = "#6EA8FF" if current_view == "Portofolio Visualization" else "inherit"
		watchlist_color = "#6EA8FF" if current_view == "Watchlist" else "inherit"
		opportunity_color = "#6EA8FF" if current_view == "Opportunity cost" else "inherit"
		insider_color = "#6EA8FF" if current_view == "Insider Buying Tracker" else "inherit"

		st.markdown(
			f"""
			<style>
			div[data-testid="stSidebar"] .stButton > button {{
				background: transparent !important;
				border: none !important;
				outline: none !important;
				box-shadow: none !important;
				padding: 0.15rem 0 !important;
				min-height: 0 !important;
				font-size: 1.55rem !important;
				font-weight: 700 !important;
				justify-content: flex-start !important;
				border-radius: 0 !important;
			}}
			div[data-testid="stSidebar"] .stButton > button:hover,
			div[data-testid="stSidebar"] .stButton > button:focus,
			div[data-testid="stSidebar"] .stButton > button:active,
			div[data-testid="stSidebar"] .stButton > button:focus-visible {{
				background: transparent !important;
				border: none !important;
				outline: none !important;
				box-shadow: none !important;
			}}
			/* first and second sidebar buttons are the two menu options */
			div[data-testid="stSidebar"] .stButton:nth-of-type(1) > button {{
				color: {stock_color} !important;
			}}
			div[data-testid="stSidebar"] .stButton:nth-of-type(2) > button {{
				color: {portfolio_color} !important;
			}}
			div[data-testid="stSidebar"] .stButton:nth-of-type(3) > button {{
				color: {watchlist_color} !important;
			}}
			div[data-testid="stSidebar"] .stButton:nth-of-type(4) > button {{
				color: {opportunity_color} !important;
			}}
			div[data-testid="stSidebar"] .stButton:nth-of-type(5) > button {{
				color: {insider_color} !important;
			}}
			</style>
			""",
			unsafe_allow_html=True,
		)

		if st.button("Stock Valuation", use_container_width=True, key="nav_stock"):
			st.session_state["selected_view"] = "Stock Valuation"

		if st.button(
			"Portofolio Visualization",
			use_container_width=True,
			key="nav_portfolio",
		):
			st.session_state["selected_view"] = "Portofolio Visualization"

		if st.button(
			"Opportunity cost",
			use_container_width=True,
			key="nav_opportunity",
		):
			st.session_state["selected_view"] = "Opportunity cost"

		st.divider()

	selected_view = st.session_state["selected_view"]

	if selected_view == "Portofolio Visualization":
		render_portfolio_visualization_tab()
		return

	if selected_view == "Watchlist":
		render_watchlist_tab()
		return

	if selected_view == "Opportunity cost":
		render_opportunity_cost_tab(ANALYSES_DIR)
		return

	if selected_view == "Insider Buying Tracker":
		all_sources = {**_get_watchlist_presets(), **_load_saved_portfolio_sources()}
		render_insider_buying_tracker_tab(watchlist_sources=all_sources)
		return

	ticker_col, provider_col, refresh_col, export_col, theme_col, save_col = st.columns(
		[1, 1.2, 0.9, 0.9, 0.9, 1],
		gap="small",
	)
	with ticker_col:
		ticker = st.text_input("Ticker", value="MSFT").strip().upper()
	with provider_col:
		data_provider = st.selectbox(
			"Data provider",
			DATA_PROVIDERS,
			index=1,
			key="valuation_data_provider",
		)
	with refresh_col:
		st.write("")
		refresh_data_now = st.button("Refresh data now", use_container_width=True, key="valuation_refresh_data_now")
	export_placeholder = export_col.empty()
	export_clicked = export_col.button("Export", use_container_width=True, key="valuation_export_pdf")
	with theme_col:
		st.write("")
		st.session_state["ui_light_mode"] = st.toggle(
			"Light mode",
			value=bool(st.session_state.get("ui_light_mode")),
			key="ui_light_mode_toggle",
			help="Switch the entire app to a light theme for printing/export.",
		)
	with save_col:
		save_clicked = st.button("💾 Save", use_container_width=True)

	if not ticker:
		st.info("Enter a ticker to begin.")
		return

	saved_df = load_analysis_df(ticker)
	hydrate_state_from_saved_analysis(ticker=ticker, df=saved_df)
	if saved_df is not None:
		st.caption(f"Loaded saved analysis context for {ticker} from local storage.")

	if refresh_data_now:
		# Explicit user intent: bypass in-memory + local freshness and fetch from provider.
		st.cache_data.clear()

	try:
		data_cache_df, data_cache_source = get_or_load_data_cache(
			ticker=ticker,
			provider=data_provider,
			force_refresh=refresh_data_now,
		)
	except Exception as exc:
		st.error(f"Could not load {data_provider} data for {ticker}: {exc}")
		return

	if data_cache_source == "provider-api":
		st.caption(f"Fetched fresh financial data from {data_provider} and cached locally.")
	elif data_cache_source == "local-cache":
		st.caption(f"Using recent local data cache for {ticker} ({data_provider}).")
	else:
		st.caption("Using stale local cache because live provider fetch failed.")

	st.divider()
	valuation_params = render_valuation_tab(
		ticker=ticker,
		saved_df=saved_df,
		data_provider=data_provider,
		data_cache_df=data_cache_df,
		data_cache_source=data_cache_source,
	)
	if valuation_params is None:
		return

	snapshot = valuation_params["snapshot"]
  
	############################################################
	st.divider()
	reverse_params = render_reverse_dcf_tab(
		snapshot=snapshot,
		ticker=ticker,
		saved_df=saved_df,
		data_cache_df=data_cache_df,
	)

	if export_clicked:
		st.session_state["valuation_export_requested"] = True

	if st.session_state.get("valuation_export_requested"):
		try:
			pdf_bytes = _build_valuation_pdf(
				ticker=ticker,
				valuation_params=valuation_params,
				reverse_params=reverse_params,
			)
			export_placeholder.download_button(
				"Download PDF",
				data=pdf_bytes,
				file_name=f"{ticker}_valuation_report.pdf",
				mime="application/pdf",
				use_container_width=True,
			)
		except Exception as exc:
			export_placeholder.empty()
			st.error(str(exc))

	if save_clicked:
		snapshot_cache = collect_snapshot_cache_from_data_cache(
			data_cache_df=data_cache_df,
			primary_snapshot=snapshot,
		)
		company_cache = {
			"cash": (reverse_params or {}).get("cash", _parse_float(_analysis_get(data_cache_df, "company_data", "cash"), np.nan)),
			"total_debt": (reverse_params or {}).get("total_debt", _parse_float(_analysis_get(data_cache_df, "company_data", "total_debt"), np.nan)),
			"previous_full_year_fcf": (reverse_params or {}).get("previous_full_year_fcf", _parse_float(_analysis_get(data_cache_df, "company_data", "previous_full_year_fcf"), np.nan)),
			"ttm_sbc": (reverse_params or {}).get("ttm_sbc", _parse_float(_analysis_get(data_cache_df, "company_data", "ttm_sbc"), np.nan)),
		}
		path = save_analysis_csv(
			ticker=ticker,
			valuation_params=valuation_params,
			reverse_params=reverse_params,
			snapshot_cache=snapshot_cache,
			company_cache=company_cache,
		)
		st.success(f"Analysis saved to {path}")


if __name__ == "__main__":
	main()
