import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import yfinance as yf
import requests
from pathlib import Path

# local imports from utils.py
from .utils import _parse_numeric_text, fmt_money

ANALYSES_DIR = Path("analyses")
PORTFOLIOS_DIR = ANALYSES_DIR / "portfolios"
DATA_CACHE_DIR = ANALYSES_DIR / ".cache"
_PRICE_MEMORY_CACHE: dict[str, float] = {}


def _to_float_safe(value) -> float:
	try:
		return float(value)
	except Exception:
		text = str(value or "")
		if any(ch.isalpha() for ch in text) or any(sym in text for sym in ["$", "€", "£"]):
			cleaned = re.sub(r"[^0-9,\.\-]", "", text)
			if cleaned.count(",") > 0 and cleaned.count(".") > 0:
				cleaned = cleaned.replace(",", "")
			elif cleaned.count(",") > 0 and cleaned.count(".") == 0:
				cleaned = cleaned.replace(",", ".")
			try:
				return float(cleaned)
			except Exception:
				pass

		parsed = _parse_numeric_text(value)
		try:
			return float(parsed)
		except Exception:
			cleaned = re.sub(r"[^0-9,\.\-]", "", text)
			if cleaned.count(",") > 0 and cleaned.count(".") > 0:
				cleaned = cleaned.replace(",", "")
			elif cleaned.count(",") > 0 and cleaned.count(".") == 0:
				cleaned = cleaned.replace(",", ".")
			try:
				return float(cleaned)
			except Exception:
				return float("nan")


def _detect_currency_code(value) -> str:
	text = str(value or "").upper()
	if "CA$" in text or "CAD" in text:
		return "CAD"
	if "€" in text or "EUR" in text:
		return "EUR"
	if "$" in text or "USD" in text or "US$" in text:
		return "USD"
	return "USD"


@st.cache_data(show_spinner=False, ttl=21600)
def _load_recent_fx_rates_to_usd() -> dict[str, float]:
	# Returns USD per 1 unit of currency code.
	rates: dict[str, float] = {"USD": 1.0}

	try:
		resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
		if resp.ok:
			payload = resp.json() if resp.content else {}
			raw = payload.get("rates") if isinstance(payload, dict) else {}
			if isinstance(raw, dict):
				for code in ["EUR", "CAD"]:
					quote = raw.get(code)
					if quote is not None:
						q = _to_float_safe(quote)
						if np.isfinite(q) and q > 0:
							rates[code] = float(1.0 / q)
	except Exception:
		pass

	missing = [c for c in ["EUR", "CAD"] if c not in rates]
	if missing:
		try:
			fx_tickers = {
				"EUR": "EURUSD=X",
				"CAD": "CADUSD=X",
			}
			for code in missing:
				ticker = fx_tickers[code]
				data = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
				if data is not None and not data.empty and "Close" in data.columns:
					series = data["Close"].dropna()
					if not series.empty:
						v = _to_float_safe(series.iloc[-1])
						if np.isfinite(v) and v > 0:
							rates[code] = float(v)
		except Exception:
			pass

	return rates


def _money_value_to_usd(value, fx_rates: dict[str, float] | None = None) -> float:
	if fx_rates is None:
		fx_rates = _load_recent_fx_rates_to_usd()

	numeric = _to_float_safe(value)
	if not np.isfinite(numeric):
		return np.nan

	currency = _detect_currency_code(value)
	factor = _to_float_safe(fx_rates.get(currency, 1.0))
	if not np.isfinite(factor) or factor <= 0:
		factor = 1.0
	return float(numeric * factor)



def render_portfolio_visualization_tab() -> None:
	st.subheader("Portfolio Visualization")

	saved_portfolio_map = _saved_portfolio_file_map()
	saved_portfolio_names = list(saved_portfolio_map.keys())

	load_col, upload_col = st.columns([1.3, 1.0], gap="small")
	with load_col:
		selected_saved = st.multiselect(
			"Load saved portfolios",
			options=saved_portfolio_names,
			default=st.session_state.get("portfolio_selected_saved", []),
			placeholder="Choose one or more saved portfolios",
			key="portfolio_saved_multiselect",
		)
		st.session_state["portfolio_selected_saved"] = selected_saved
	with upload_col:
		uploaded_csv = st.file_uploader("Upload portfolio from fiscal.ai", type=["csv"], key="portfolio_csv_uploader")

	manual_col = st.columns([1.5, 1, 1])[0]
	with manual_col:
		if st.button("Create New Portfolio", key="portfolio_create_new_btn", use_container_width=True):
			st.session_state["portfolio_builder_open"] = True
			st.session_state["portfolio_active_saved_name"] = None
			st.session_state["portfolio_autosave_signature"] = None
			st.session_state["portfolio_base_df"] = pd.DataFrame()
			st.session_state["portfolio_builder_tickers"] = []
			st.session_state["portfolio_builder_active_tickers"] = []

	current_source_signature = (
		tuple(sorted(selected_saved)),
		getattr(uploaded_csv, "name", ""),
		int(getattr(uploaded_csv, "size", 0) or 0),
	)
	previous_source_signature = st.session_state.get("portfolio_source_signature")

	if current_source_signature != previous_source_signature:
		if selected_saved or uploaded_csv is not None:
			try:
				loaded_df = _load_portfolio_from_selected_sources(
					selected_saved=selected_saved,
					saved_portfolio_map=saved_portfolio_map,
					uploaded_csv=uploaded_csv,
				)
				if loaded_df is not None and not loaded_df.empty:
					# Visualize immediately for uploaded/saved portfolios (including legacy CSV format).
					st.session_state["portfolio_df"] = loaded_df
					_hydrate_builder_from_portfolio_df(loaded_df)
					st.session_state["portfolio_builder_open"] = True
					st.session_state["portfolio_active_saved_name"] = (
						selected_saved[0] if len(selected_saved) == 1 and uploaded_csv is None else None
					)
					st.session_state["portfolio_autosave_signature"] = None
					st.success(f"Loaded {len(loaded_df)} positions from selected source(s).")
			except Exception as exc:
				st.error(f"Could not load selected portfolio source(s): {exc}")
		st.session_state["portfolio_source_signature"] = current_source_signature

	render_manual_portfolio_builder()

	portfolio_df = st.session_state.get("portfolio_df")
	if portfolio_df is None or not isinstance(portfolio_df, pd.DataFrame) or portfolio_df.empty:
		st.info("Click **Build** after entering positions to generate the visualization.")
		return

	portfolio_df = portfolio_df.copy()
	name_map = _resolve_company_names(tuple(portfolio_df["Ticker"].astype(str).tolist()))
	portfolio_df["Company"] = portfolio_df["Ticker"].map(name_map).fillna(portfolio_df["Ticker"])

	total_market_value = float(portfolio_df["Market Value Parsed"].fillna(0.0).sum())
	avg_perf = float(portfolio_df["Ownership Performance Parsed"].dropna().mean()) if portfolio_df[
		"Ownership Performance Parsed"
	].notna().any() else np.nan

	metric_cols = st.columns(2)
	metric_cols[0].metric("Positions", f"{len(portfolio_df)}")
	metric_cols[1].metric("Total market value", fmt_money(total_market_value) if total_market_value > 0 else "N/A")
	# metric_cols[2].metric(
	# 	"Average ownership performance",
	# 	f"{avg_perf:.2f}%" if not np.isnan(avg_perf) else "N/A",
	# )

	chart_df = portfolio_df.sort_values("Weight", ascending=False).copy()
	chart_df["Label"] = chart_df["Company"]
	hover_columns = {
		"Weight Pct": ":.2f",
		"Market Value Parsed": ":,.2f",
		"Ownership Performance Parsed": ":.2f",
		"Shares Parsed": ":,.2f",
		"Average Cost Basis Parsed": ":,.2f",
		"Stock Price Parsed": ":,.2f",
		"Ticker": True,
	}

	bar_df = chart_df.head(15).copy()
	bar_df = bar_df.sort_values("Weight Pct", ascending=False)
	allocation_bar = go.Figure(
		go.Bar(
			x=bar_df["Company"],
			y=bar_df["Weight Pct"],
			marker=dict(
				color=bar_df["Weight Pct"],
				colorscale="Blues",
				line=dict(color="#111827", width=1),
			),
			text=[f"{v:.2f}%" for v in bar_df["Weight Pct"]],
			textposition="outside",
			hovertemplate=(
				"<b>%{x}</b><br>"
				+ "Allocation: %{y:.2f}%<br>"
				+ "Ticker: %{customdata[0]}<extra></extra>"
			),
			customdata=bar_df[["Ticker"]].to_numpy(),
		)
	)
	allocation_bar.update_layout(
		title="Top Allocation (highest to lowest)",
		xaxis_title="Company",
		yaxis_title="Portfolio %",
		xaxis=dict(categoryorder="array", categoryarray=bar_df["Company"].tolist()),
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		height=420,
		margin=dict(l=20, r=20, t=50, b=120),
	)
	st.plotly_chart(allocation_bar, use_container_width=True)

	fig = px.pie(
		chart_df,
		names="Label",
		values="Weight",
		hole=0.42,
		color_discrete_sequence=px.colors.qualitative.Bold,
		hover_data=hover_columns,
	)
	fig.update_traces(
		textposition="outside",
		textinfo="label+percent",
		marker=dict(line=dict(color="#1f1f2e", width=1)),
		pull=[0.0] * len(chart_df),
	)
	fig.update_layout(
		title="Portfolio Allocation",
		showlegend=False,
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		margin=dict(l=20, r=20, t=50, b=20),
	)
	st.plotly_chart(fig, use_container_width=True)

	holdings_preview = chart_df[["Company", "Ticker", "Weight Pct"]].head(8).copy()
	holdings_preview["Weight Pct"] = holdings_preview["Weight Pct"].map(lambda v: f"{v:.2f}%")
	# st.markdown("#### Top holdings")
	# for _, row in holdings_preview.iterrows():
	# 	st.markdown(f"- **{row['Company']}** (`{row['Ticker']}`): {row['Weight Pct']}")

	save_name_col, save_btn_col = st.columns([2, 1], gap="small")
	with save_name_col:
		portfolio_entry_name = st.text_input(
			"Portfolio entry name",
			value=st.session_state.get("portfolio_entry_name", "my_portfolio"),
			key="portfolio_entry_name",
		)
	with save_btn_col:
		save_portfolio = st.button("Save Portfolio", use_container_width=True, key="save_portfolio_btn")

	if save_portfolio:
		path = _portfolio_save_path(portfolio_entry_name)
		template_df = st.session_state.get("portfolio_base_df")
		export_df = _export_portfolio_df_for_storage(chart_df, template_df=template_df)
		export_df.to_csv(path, index=False)
		st.success(f"Portfolio saved locally to {path}")


def render_manual_portfolio_builder() -> None:
	st.markdown("### Portfolio editor")
	st.caption("Enter ticker and press Enter. Then fill shares or percentages. Click Build when ready.")

	if "portfolio_builder_tickers" not in st.session_state:
		st.session_state["portfolio_builder_tickers"] = []
	if "portfolio_builder_price_map" not in st.session_state:
		st.session_state["portfolio_builder_price_map"] = {}
	if "portfolio_builder_mode_pct" not in st.session_state:
		st.session_state["portfolio_builder_mode_pct"] = False
	if "portfolio_builder_active_tickers" not in st.session_state:
		st.session_state["portfolio_builder_active_tickers"] = []
	if "portfolio_base_df" not in st.session_state:
		st.session_state["portfolio_base_df"] = pd.DataFrame()
	if "portfolio_stale_price_tickers" not in st.session_state:
		st.session_state["portfolio_stale_price_tickers"] = []

	st.markdown(
		"""
		<style>
		div[data-testid="stButton"] button[kind="secondary"] {
			min-height: 2.45rem;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	header_cols = st.columns([1.0, 1.2, 1.4], gap="small")
	with header_cols[0]:
		st.text_input(
			"Enter ticker (press Enter to add)",
			value="",
			placeholder="META",
			max_chars=10,
			key="portfolio_builder_ticker_input",
			on_change=_append_builder_ticker_from_input,
		)
	with header_cols[1]:
		st.toggle(
			"Enter percentages instead of shares",
			key="portfolio_builder_mode_pct",
			help="Useful for hypothetical portfolios.",
		)
	with header_cols[2]:
		build_clicked = st.button("Build", key="portfolio_builder_build_btn", use_container_width=True, type="primary")

	base_df = st.session_state.get("portfolio_base_df")
	base_tickers = []
	if isinstance(base_df, pd.DataFrame) and not base_df.empty and "Ticker" in base_df.columns:
		base_tickers = [str(t).strip().upper() for t in base_df["Ticker"].tolist() if str(t).strip()]

	all_tickers = sorted(set(base_tickers) | set(st.session_state.get("portfolio_builder_tickers", [])))
	if not all_tickers:
		st.info("Add your first ticker above and press Enter.")
		return

	current_tickers = [
		t
		for t in st.session_state.get("portfolio_builder_active_tickers", [])
		if t in all_tickers
	]

	selected_tickers = st.multiselect(
		"Stocks in portfolio",
		options=all_tickers,
		default=current_tickers,
		key="portfolio_builder_multiselect",
	)
	st.session_state["portfolio_builder_active_tickers"] = selected_tickers

	if not selected_tickers:
		st.info("Select at least one ticker to edit/build.")
		return

	price_map: dict[str, float] = {
		str(k).strip().upper(): _to_float_safe(v)
		for k, v in st.session_state.get("portfolio_builder_price_map", {}).items()
		if np.isfinite(_to_float_safe(v)) and _to_float_safe(v) > 0
	}
	st.session_state["portfolio_builder_price_map"] = price_map
	name_map = _resolve_company_names(tuple(selected_tickers))

	if selected_tickers:
		refresh_cols = st.columns([1.2, 2.0], gap="small")
		with refresh_cols[0]:
			refresh_missing = st.button(
				"Refresh stock prices",
				key="portfolio_refresh_missing_prices_btn",
				use_container_width=True,
			)
		with refresh_cols[1]:
			st.caption("Updates prices it can find. If not found, existing uploaded/saved price is kept.")
		if refresh_missing:
			fetched = _latest_prices_map(tuple(selected_tickers))
			stale_tickers: list[str] = []
			for ticker, price in fetched.items():
				p = _to_float_safe(price)
				if np.isfinite(p) and p > 0:
					price_map[ticker] = float(p)
			for ticker in selected_tickers:
				fetched_value = fetched.get(ticker)
				fetched_num = _to_float_safe(fetched_value)
				if (fetched_value is None or not np.isfinite(fetched_num) or fetched_num <= 0) and float(price_map.get(ticker, 0.0)) > 0:
					stale_tickers.append(ticker)
			st.session_state["portfolio_stale_price_tickers"] = sorted(set(stale_tickers))
			st.session_state["portfolio_builder_price_map"] = price_map
			st.rerun()

	mode_pct = bool(st.session_state.get("portfolio_builder_mode_pct", False))
	entries: list[dict] = []
	missing_tickers: list[str] = []

	for ticker in selected_tickers:
		row_cols = st.columns([1.6, 1.0, 1.25, 0.34], gap="small")
		with row_cols[0]:
			st.markdown(
				f"**{ticker}**",
				unsafe_allow_html=True,
			)

		resolved_price = float(price_map.get(ticker, 0.0) or 0.0)
		with row_cols[1]:
			if resolved_price > 0:
				st.metric("Latest price", fmt_money(resolved_price))
				if ticker in st.session_state.get("portfolio_stale_price_tickers", []):
					st.caption("Using saved price (latest update unavailable).")
			else:
				st.warning("Price unavailable. Enter price manually.")
				manual_price = float(
					st.number_input(
						f"{ticker} manual price",
						min_value=0.0,
						step=0.01,
						format="%.2f",
						value=float(st.session_state.get(f"portfolio_manual_price_{ticker}", 0.0)),
						key=f"portfolio_manual_price_{ticker}",
					)
				)
				if manual_price > 0:
					resolved_price = manual_price
					price_map[ticker] = manual_price

		with row_cols[2]:
			if mode_pct:
				position_value = float(
					st.number_input(
						f"{ticker} %",
						min_value=0.0,
						step=0.1,
						format="%.2f",
						value=float(st.session_state.get(f"portfolio_pct_{ticker}", 0.0)),
						key=f"portfolio_pct_{ticker}",
					)
				)
				entries.append(
					{
						"ticker": ticker,
						"input_value": position_value,
						"price": resolved_price,
						"mode": "pct",
					}
				)
			else:
				position_value = float(
					st.number_input(
						f"{ticker} shares",
						min_value=0.0,
						step=1.0,
						format="%.2f",
						value=float(st.session_state.get(f"portfolio_shares_{ticker}", 0.0)),
						key=f"portfolio_shares_{ticker}",
					)
				)
				if position_value > 0 and resolved_price <= 0:
					missing_tickers.append(ticker)
				entries.append(
					{
						"ticker": ticker,
						"input_value": position_value,
						"price": resolved_price,
						"mode": "shares",
					}
				)

		with row_cols[3]:
			if st.button("✕", key=f"portfolio_remove_{ticker}", use_container_width=True, type="secondary"):
				st.session_state["portfolio_builder_tickers"] = [
					t for t in st.session_state.get("portfolio_builder_tickers", []) if t != ticker
				]
				st.session_state["portfolio_builder_active_tickers"] = [
					t for t in st.session_state.get("portfolio_builder_active_tickers", []) if t != ticker
				]
				st.session_state.pop(f"portfolio_shares_{ticker}", None)
				st.session_state.pop(f"portfolio_pct_{ticker}", None)
				st.session_state.pop(f"portfolio_manual_price_{ticker}", None)
				st.rerun()

	st.session_state["portfolio_builder_price_map"] = price_map

	if missing_tickers:
		st.warning(
			"Price unavailable for: " + ", ".join(sorted(set(missing_tickers))) + ". Enter manual price to continue."
		)

	stale_tickers_notice = [t for t in st.session_state.get("portfolio_stale_price_tickers", []) if t in selected_tickers]
	if stale_tickers_notice:
		st.info(
			"Latest price update unavailable for: "
			+ ", ".join(stale_tickers_notice)
			+ ". Using loaded/saved price values."
		)

	base_df = st.session_state.get("portfolio_base_df")
	merged_df = _merge_portfolio_base_with_edits(base_df, _build_portfolio_from_builder_entries(entries), mode_pct=mode_pct)

	# Autosave edits when exactly one saved portfolio is active.
	active_saved_name = st.session_state.get("portfolio_active_saved_name")
	if active_saved_name:
		template_df = st.session_state.get("portfolio_base_df")
		autosave_df = _export_portfolio_df_for_storage(merged_df, template_df=template_df)
		autosave_signature = autosave_df.to_json(orient="records") if not autosave_df.empty else "[]"
		prev_signature = st.session_state.get("portfolio_autosave_signature")
		if autosave_signature != prev_signature and not autosave_df.empty:
			try:
				path = _saved_portfolio_file_map().get(active_saved_name)
				if path is not None:
					autosave_df.to_csv(path, index=False)
					st.session_state["portfolio_autosave_signature"] = autosave_signature
			except Exception:
				pass

	if build_clicked:
		if merged_df is None or merged_df.empty:
			st.warning("Nothing to build yet. Enter shares/percentages greater than zero.")
			return
		st.session_state["portfolio_df"] = merged_df
		st.success(f"Built portfolio with {len(merged_df)} positions.")


def _append_builder_ticker_from_input() -> None:
	raw_input = str(st.session_state.get("portfolio_builder_ticker_input", "") or "").strip()
	if not raw_input:
		return

	tokens = [
		t.strip().upper()
		for t in raw_input.replace(";", ",").replace("\n", ",").replace(" ", ",").split(",")
		if t.strip()
	]
	if not tokens:
		st.session_state["portfolio_builder_ticker_input"] = ""
		return

	merged = sorted(set(st.session_state.get("portfolio_builder_tickers", [])) | set(tokens))
	st.session_state["portfolio_builder_tickers"] = merged
	active = sorted(set(st.session_state.get("portfolio_builder_active_tickers", [])) | set(tokens))
	st.session_state["portfolio_builder_active_tickers"] = active
	for token in tokens:
		_seed_editor_fields_for_ticker(token)
	new_prices = _latest_prices_map(tuple(tokens))
	price_map = st.session_state.get("portfolio_builder_price_map", {})
	for ticker in tokens:
		candidate = new_prices.get(ticker)
		candidate_num = _to_float_safe(candidate)
		if candidate is not None and np.isfinite(candidate_num) and candidate_num > 0:
			price_map[ticker] = float(candidate_num)
			if ticker in st.session_state.get("portfolio_stale_price_tickers", []):
				st.session_state["portfolio_stale_price_tickers"] = [
					t for t in st.session_state.get("portfolio_stale_price_tickers", []) if t != ticker
				]
	st.session_state["portfolio_builder_price_map"] = price_map
	st.session_state["portfolio_builder_ticker_input"] = ""


def _hydrate_builder_from_portfolio_df(df: pd.DataFrame) -> None:
	if df is None or df.empty or "Ticker" not in df.columns:
		return

	builder_df = df.copy()
	builder_df["Ticker"] = builder_df["Ticker"].astype(str).str.strip().str.upper()
	tickers = [t for t in builder_df["Ticker"].tolist() if t]
	st.session_state["portfolio_base_df"] = builder_df.copy()
	st.session_state["portfolio_builder_tickers"] = []
	st.session_state["portfolio_builder_active_tickers"] = []

	price_map = st.session_state.get("portfolio_builder_price_map", {})
	for _, row in builder_df.iterrows():
		ticker = str(row.get("Ticker", "")).strip().upper()
		if not ticker:
			continue
		shares = row.get("Shares Parsed", np.nan)
		price = row.get("Stock Price Parsed", np.nan)
		if pd.isna(price):
			price = row.get("Stock Price", np.nan)
		pct = row.get("Weight Pct", np.nan)
		if pd.isna(pct):
			pct = row.get("Portfolio Percentage Parsed", np.nan)
		if pd.isna(pct):
			pct = row.get("Portfolio Percentage", np.nan)
		if pd.notna(shares):
			st.session_state[f"portfolio_shares_{ticker}"] = float(shares)
		price_num = _money_value_to_usd(price)
		if np.isfinite(price_num) and price_num > 0:
			price_map[ticker] = float(price_num)
			st.session_state[f"portfolio_manual_price_{ticker}"] = float(price_num)
		if pd.notna(pct):
			st.session_state[f"portfolio_pct_{ticker}"] = float(pct)

	st.session_state["portfolio_builder_price_map"] = price_map


def _seed_editor_fields_for_ticker(ticker: str) -> None:
	ticker = str(ticker).strip().upper()
	if not ticker:
		return

	base_df = st.session_state.get("portfolio_base_df")
	if base_df is None or not isinstance(base_df, pd.DataFrame) or base_df.empty or "Ticker" not in base_df.columns:
		return

	rows = base_df.loc[base_df["Ticker"].astype(str).str.strip().str.upper() == ticker]
	if rows.empty:
		return

	row = rows.iloc[-1]
	shares = row.get("Shares Parsed", np.nan)
	price = row.get("Stock Price Parsed", np.nan)
	if pd.isna(price):
		price = row.get("Stock Price", np.nan)
	pct = row.get("Weight Pct", np.nan)
	if pd.isna(pct):
		pct = row.get("Portfolio Percentage Parsed", np.nan)
	if pd.isna(pct):
		pct = row.get("Portfolio Percentage", np.nan)

	if pd.notna(shares):
		st.session_state[f"portfolio_shares_{ticker}"] = float(shares)
	if pd.notna(pct):
		st.session_state[f"portfolio_pct_{ticker}"] = float(pct)
	price_num = _money_value_to_usd(price)
	if np.isfinite(price_num) and price_num > 0:
		price_map = st.session_state.get("portfolio_builder_price_map", {})
		price_map[ticker] = float(price_num)
		st.session_state["portfolio_builder_price_map"] = price_map
		st.session_state[f"portfolio_manual_price_{ticker}"] = float(price_num)


def _merge_portfolio_base_with_edits(
	base_df: pd.DataFrame | None,
	edited_df: pd.DataFrame | None,
	mode_pct: bool = False,
) -> pd.DataFrame:
	base = pd.DataFrame() if base_df is None else base_df.copy()
	edits = pd.DataFrame() if edited_df is None else edited_df.copy()

	if base.empty and edits.empty:
		return pd.DataFrame()

	for df in (base, edits):
		if df.empty:
			continue
		if "Ticker" not in df.columns:
			df["Ticker"] = ""
		df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
		if "Weight" not in df.columns:
			df["Weight"] = np.nan
		if "Market Value Parsed" not in df.columns:
			df["Market Value Parsed"] = np.nan
		if "Weight Pct" not in df.columns:
			df["Weight Pct"] = np.nan

	if mode_pct and not base.empty:
		base["Weight"] = pd.to_numeric(base.get("Weight Pct", np.nan), errors="coerce")

	edited_tickers = set(edits["Ticker"].tolist()) if not edits.empty else set()
	if not base.empty and edited_tickers:
		base = base[~base["Ticker"].isin(edited_tickers)]

	combined = pd.concat([base, edits], ignore_index=True)
	combined = combined[(combined["Ticker"] != "") & pd.to_numeric(combined["Weight"], errors="coerce").notna()]
	if combined.empty:
		return pd.DataFrame()

	combined["Weight"] = pd.to_numeric(combined["Weight"], errors="coerce").fillna(0.0)
	combined = combined[combined["Weight"] > 0]
	if combined.empty:
		return pd.DataFrame()

	total_weight = float(combined["Weight"].sum())
	combined["Weight Pct"] = (combined["Weight"] / total_weight) * 100 if total_weight > 0 else 0.0
	return combined.sort_values("Weight", ascending=False).reset_index(drop=True)


def _build_portfolio_from_builder_entries(entries: list[dict]) -> pd.DataFrame:
	rows: list[dict] = []
	for entry in entries:
		ticker = str(entry.get("ticker", "")).strip().upper()
		mode = str(entry.get("mode", "shares")).strip().lower()
		value = float(entry.get("input_value", 0.0) or 0.0)
		price = float(entry.get("price", 0.0) or 0.0)
		if not ticker or value <= 0:
			continue

		if mode == "pct":
			rows.append(
				{
					"Ticker": ticker,
					"Shares Parsed": np.nan,
					"Average Cost Basis Parsed": np.nan,
					"Stock Price Parsed": price if price > 0 else np.nan,
					"Market Value Parsed": np.nan,
					"Ownership Performance Parsed": np.nan,
					"Weight": float(value),
				}
			)
		else:
			if price <= 0:
				continue
			market_value = float(value * price)
			rows.append(
				{
					"Ticker": ticker,
					"Shares Parsed": float(value),
					"Average Cost Basis Parsed": np.nan,
					"Stock Price Parsed": float(price),
					"Market Value Parsed": market_value,
					"Ownership Performance Parsed": np.nan,
					"Weight": market_value,
				}
			)

	portfolio_df = pd.DataFrame(rows)
	if portfolio_df.empty:
		return portfolio_df

	total_weight = float(portfolio_df["Weight"].sum())
	portfolio_df["Weight Pct"] = (portfolio_df["Weight"] / total_weight) * 100 if total_weight > 0 else 0.0
	return portfolio_df.sort_values("Weight", ascending=False).reset_index(drop=True)


def _build_export_df_from_builder_entries(entries: list[dict]) -> pd.DataFrame:
	rows: list[dict] = []
	for entry in entries:
		ticker = str(entry.get("ticker", "")).strip().upper()
		mode = str(entry.get("mode", "shares")).strip().lower()
		value = float(entry.get("input_value", 0.0) or 0.0)
		price = float(entry.get("price", 0.0) or 0.0)
		if not ticker:
			continue
		if mode == "pct":
			rows.append(
				{
					"Ticker": ticker,
					"Portfolio Percentage": float(max(0.0, value)),
					"Stock Price": float(price) if price > 0 else np.nan,
				}
			)
		else:
			market_value = float(value * price) if value > 0 and price > 0 else np.nan
			rows.append(
				{
					"Ticker": ticker,
					"Shares": float(max(0.0, value)),
					"Stock Price": float(price) if price > 0 else np.nan,
					"Market Value": market_value,
				}
			)

	export_df = pd.DataFrame(rows)
	if export_df.empty:
		return export_df

	# Ensure canonical columns exist for reliable reloads.
	if "Market Value" not in export_df.columns:
		export_df["Market Value"] = np.nan
	if "Portfolio Percentage" not in export_df.columns:
		export_df["Portfolio Percentage"] = np.nan

	# For share-based rows, compute missing portfolio percentages from market values.
	market_total = float(export_df["Market Value"].fillna(0.0).sum())
	if market_total > 0:
		missing_pct_mask = export_df["Portfolio Percentage"].isna()
		export_df.loc[missing_pct_mask, "Portfolio Percentage"] = (
			export_df.loc[missing_pct_mask, "Market Value"].fillna(0.0) / market_total
		) * 100

	return export_df


def _export_portfolio_df_for_storage(
	portfolio_df: pd.DataFrame,
	template_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
	if portfolio_df is None or portfolio_df.empty:
		return pd.DataFrame(columns=["Ticker", "Shares", "Stock Price", "Market Value", "Portfolio Percentage"])

	df = portfolio_df.copy()
	out = pd.DataFrame()
	out["Ticker"] = df.get("Ticker", pd.Series(dtype=str)).astype(str).str.strip().str.upper()

	shares = pd.to_numeric(df.get("Shares Parsed", np.nan), errors="coerce")
	price = pd.to_numeric(df.get("Stock Price Parsed", np.nan), errors="coerce")
	market_value = pd.to_numeric(df.get("Market Value Parsed", np.nan), errors="coerce")
	weight_pct = pd.to_numeric(df.get("Weight Pct", np.nan), errors="coerce")

	computed_market_value = shares * price
	market_value = market_value.where(market_value.notna() & (market_value > 0), computed_market_value)

	out["Shares"] = shares
	out["Stock Price"] = price
	out["Market Value"] = market_value

	if weight_pct.notna().any() and (weight_pct > 0).any():
		out["Portfolio Percentage"] = weight_pct
	else:
		total_market = float(out["Market Value"].fillna(0.0).sum())
		out["Portfolio Percentage"] = (
			(out["Market Value"].fillna(0.0) / total_market) * 100 if total_market > 0 else np.nan
		)

	canonical = out.dropna(how="all", subset=["Ticker"]).reset_index(drop=True)

	if template_df is None or not isinstance(template_df, pd.DataFrame) or template_df.empty or "Ticker" not in template_df.columns:
		return canonical

	template = template_df.copy()
	template["Ticker"] = template["Ticker"].astype(str).str.strip().str.upper()

	update_cols = {
		"Shares": "Shares",
		"Stock Price": "Stock Price",
		"Market Value": "Market Value",
		"Portfolio Percentage": "Portfolio Percentage",
		"Shares Parsed": "Shares",
		"Stock Price Parsed": "Stock Price",
		"Market Value Parsed": "Market Value",
		"Portfolio Percentage Parsed": "Portfolio Percentage",
		"Weight": "Market Value",
		"Weight Pct": "Portfolio Percentage",
	}

	for _, row in canonical.iterrows():
		ticker = str(row.get("Ticker", "")).strip().upper()
		if not ticker:
			continue

		matches = template.index[template["Ticker"] == ticker].tolist()
		if matches:
			idx = matches[-1]
		else:
			new_row = {col: np.nan for col in template.columns}
			new_row["Ticker"] = ticker
			template = pd.concat([template, pd.DataFrame([new_row])], ignore_index=True)
			idx = template.index[-1]

		for target_col, source_col in update_cols.items():
			if source_col not in canonical.columns:
				continue
			if target_col not in template.columns:
				template[target_col] = np.nan
			template.at[idx, target_col] = row.get(source_col)

	# Keep uploaded format/columns while ensuring core columns exist.
	for col in ["Ticker", "Shares", "Stock Price", "Market Value", "Portfolio Percentage"]:
		if col not in template.columns:
			template[col] = np.nan

	return template.reset_index(drop=True)


def _load_portfolio_from_selected_sources(
	selected_saved: list[str],
	saved_portfolio_map: dict[str, Path],
	uploaded_csv,
) -> pd.DataFrame | None:
	frames: list[pd.DataFrame] = []
	raw_frames: list[pd.DataFrame] = []
	for name in selected_saved:
		path = saved_portfolio_map.get(name)
		if path is None or not path.exists():
			continue
		raw_df = pd.read_csv(path)
		raw_frames.append(raw_df)
		frames.append(_normalize_portfolio_df(raw_df))

	if uploaded_csv is not None:
		raw_uploaded = pd.read_csv(uploaded_csv)
		raw_frames.append(raw_uploaded)
		frames.append(_normalize_portfolio_df(raw_uploaded))

	if not frames:
		return None

	# Preserve additional columns when a single source is loaded.
	if len(frames) == 1:
		return frames[0].copy()

	return _merge_portfolio_frames(frames)


def _merge_portfolio_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
	combined = pd.concat(frames, ignore_index=True)
	if combined.empty:
		return combined

	def _weighted_avg(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
		values = group[value_col].fillna(0.0)
		weights = group[weight_col].fillna(0.0)
		total = float(weights.sum())
		if total <= 0:
			return np.nan
		return float((values * weights).sum() / total)

	agg_rows: list[dict] = []
	for ticker, grp in combined.groupby("Ticker", dropna=False):
		market_value = float(grp["Market Value Parsed"].fillna(0.0).sum())
		weight_base = market_value if market_value > 0 else float(grp["Weight"].fillna(0.0).sum())
		shares_sum = float(grp["Shares Parsed"].fillna(0.0).sum())
		stock_price = _weighted_avg(grp, "Stock Price Parsed", "Market Value Parsed")
		if np.isnan(stock_price) and shares_sum > 0 and market_value > 0:
			stock_price = market_value / shares_sum
		ownership_perf = _weighted_avg(grp, "Ownership Performance Parsed", "Market Value Parsed")
		avg_cost = _weighted_avg(grp, "Average Cost Basis Parsed", "Shares Parsed")

		agg_rows.append(
			{
				"Ticker": str(ticker).strip().upper(),
				"Shares Parsed": shares_sum if shares_sum > 0 else np.nan,
				"Average Cost Basis Parsed": avg_cost,
				"Stock Price Parsed": stock_price,
				"Market Value Parsed": market_value if market_value > 0 else np.nan,
				"Ownership Performance Parsed": ownership_perf,
				"Weight": max(0.0, weight_base),
			}
		)

	merged = pd.DataFrame(agg_rows)
	merged = merged[(merged["Ticker"] != "") & (merged["Weight"] > 0)]
	if merged.empty:
		return merged

	total_weight = float(merged["Weight"].sum())
	merged["Weight Pct"] = (merged["Weight"] / total_weight) * 100 if total_weight > 0 else 0.0
	return merged.sort_values("Weight", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def _resolve_company_names(tickers: tuple[str, ...]) -> dict[str, str]:
	name_map: dict[str, str] = {}
	for raw_ticker in tickers:
		ticker = str(raw_ticker).strip().upper()
		if not ticker:
			continue
		name = ticker
		try:
			stock = yf.Ticker(ticker)
			info = stock.info if isinstance(stock.info, dict) else {}
			name = (
				str(info.get("longName") or info.get("shortName") or info.get("displayName") or ticker).strip()
			)
		except Exception:
			name = ticker
		name_map[ticker] = name if name else ticker
	return name_map


def _latest_prices_map(tickers: tuple[str, ...]) -> dict[str, float]:
	prices: dict[str, float] = {}
	clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
	if not clean_tickers:
		return prices

	# Preload from memory/local caches first to keep UI stable during provider throttling.
	for ticker in clean_tickers:
		cached = _to_positive_cached_price(ticker)
		if cached is not None:
			prices[ticker] = cached

	def _to_positive_float(value) -> float | None:
		try:
			v = float(value)
			return v if v > 0 else None
		except Exception:
			return None

	def _mapping_get(source, key: str):
		if source is None:
			return None
		if isinstance(source, dict):
			return source.get(key)
		if hasattr(source, "get"):
			try:
				return source.get(key)
			except Exception:
				pass
		try:
			return source[key]
		except Exception:
			return getattr(source, key, None)

	def _extract_close_from_download(data: pd.DataFrame, symbols: list[str]) -> dict[str, float]:
		resolved: dict[str, float] = {}
		if data is None or data.empty:
			return resolved

		if not isinstance(data.columns, pd.MultiIndex):
			# Common single-ticker layout
			if len(symbols) == 1 and "Close" in data.columns:
				series = data["Close"].dropna()
				if not series.empty:
					value = _to_positive_float(series.iloc[-1])
					if value is not None:
						resolved[symbols[0]] = value
			return resolved

		level0_values = {str(v) for v in data.columns.get_level_values(0)}
		level1_values = {str(v) for v in data.columns.get_level_values(1)}

		# Layout A: (PriceField, Ticker) e.g. ('Close', 'CRDO')
		if "Close" in level0_values:
			try:
				close_df = data["Close"]
				for symbol in symbols:
					if symbol in close_df.columns:
						series = close_df[symbol].dropna()
						if not series.empty:
							value = _to_positive_float(series.iloc[-1])
							if value is not None:
								resolved[symbol] = value
			except Exception:
				pass

		# Layout B: (Ticker, PriceField) e.g. ('CRDO', 'Close')
		if "Close" in level1_values:
			for symbol in symbols:
				try:
					if (symbol, "Close") in data.columns:
						series = data[(symbol, "Close")].dropna()
						if not series.empty:
							value = _to_positive_float(series.iloc[-1])
							if value is not None:
								resolved[symbol] = value
				except Exception:
					continue

		return resolved

	# 0) Direct Yahoo quote endpoint (very reliable for single latest price snapshots).
	try:
		symbols = ",".join(clean_tickers)
		response = requests.get(
			"https://query1.finance.yahoo.com/v7/finance/quote",
			params={"symbols": symbols},
			timeout=8,
		)
		if response.ok:
			payload = response.json() if response.content else {}
			results = ((payload or {}).get("quoteResponse") or {}).get("result") or []
			if isinstance(results, list):
				for row in results:
					if not isinstance(row, dict):
						continue
					ticker = str(row.get("symbol") or "").strip().upper()
					if not ticker:
						continue
					for key in [
						"regularMarketPrice",
						"postMarketPrice",
						"preMarketPrice",
						"bid",
						"ask",
						"previousClose",
					]:
						candidate = _to_positive_float(row.get(key))
						if candidate is not None:
							prices[ticker] = candidate
							break
	except Exception:
		pass

	# 1) Try batched download first for better reliability and fewer requests.
	try:
		data = yf.download(
			clean_tickers,
			period="5d",
			interval="1d",
			progress=False,
			auto_adjust=False,
			threads=False,
			group_by="column",
		)
		prices.update(_extract_close_from_download(data, clean_tickers))
	except Exception:
		pass

	for raw_ticker in tickers:
		ticker = str(raw_ticker).strip().upper()
		if not ticker:
			continue
		if ticker in prices:
			continue
		try:
			data = yf.download(
				ticker,
				period="5d",
				interval="1d",
				progress=False,
				auto_adjust=False,
				threads=False,
				group_by="column",
			)
			if data is not None and not data.empty:
				resolved = _extract_close_from_download(data, [ticker])
				if ticker in resolved:
					prices[ticker] = resolved[ticker]
					continue
		except Exception:
			pass

		try:
			stock = yf.Ticker(ticker)
			fi = stock.fast_info
			for key in ["last_price", "regularMarketPrice", "previousClose", "open", "dayHigh"]:
				value = _mapping_get(fi, key)
				candidate = _to_positive_float(value)
				if candidate is not None:
					prices[ticker] = candidate
					break
		except Exception:
			pass

		if ticker in prices:
			continue

		# Additional fallback: yfinance info payload.
		try:
			stock = yf.Ticker(ticker)
			info = stock.info if isinstance(stock.info, dict) else {}
			for key in [
				"currentPrice",
				"regularMarketPrice",
				"previousClose",
				"bid",
				"ask",
			]:
				candidate = _to_positive_float(info.get(key))
				if candidate is not None:
					prices[ticker] = candidate
					break
		except Exception:
			pass

		if ticker in prices:
			continue

		# Final fallback: ticker.history
		try:
			stock = yf.Ticker(ticker)
			hist = stock.history(period="5d", interval="1d", auto_adjust=False)
			if hist is not None and not hist.empty and "Close" in hist.columns:
				series = hist["Close"].dropna()
				if not series.empty:
					price = float(series.iloc[-1])
					if price > 0:
						prices[ticker] = price
		except Exception:
			continue

		if ticker in prices:
			continue

		# Last-resort fallback: local valuation cache generated by main app.
		cached_price = _latest_price_from_local_cache(ticker)
		if cached_price is not None:
			prices[ticker] = cached_price

	# Persist newly resolved values in memory for subsequent reruns.
	for ticker, price in prices.items():
		if price is not None and float(price) > 0:
			_PRICE_MEMORY_CACHE[str(ticker).strip().upper()] = float(price)

	return prices


def _to_positive_cached_price(ticker: str) -> float | None:
	ticker = str(ticker).strip().upper()
	if not ticker:
		return None

	memory_price = _PRICE_MEMORY_CACHE.get(ticker)
	if memory_price is not None and float(memory_price) > 0:
		return float(memory_price)

	for provider in (_latest_price_from_local_cache, _latest_price_from_saved_analysis):
		value = provider(ticker)
		if value is not None and float(value) > 0:
			_PRICE_MEMORY_CACHE[ticker] = float(value)
			return float(value)

	return None


def _latest_price_from_local_cache(ticker: str) -> float | None:
	path = DATA_CACHE_DIR / f"{ticker.strip().upper()}.csv"
	if not path.exists():
		return None
	try:
		df = pd.read_csv(path)
		if df is None or df.empty:
			return None
		mask = (df.get("section") == "snapshot") & (df.get("key") == "price")
		if "value" not in df.columns:
			return None
		rows = df.loc[mask, "value"] if mask is not None else pd.Series(dtype=float)
		if rows.empty:
			return None
		for raw in rows[::-1]:
			try:
				value = float(raw)
				if value > 0:
					return value
			except Exception:
				continue
		return None
	except Exception:
		return None


def _latest_price_from_saved_analysis(ticker: str) -> float | None:
	path = ANALYSES_DIR / f"{ticker.strip().upper()}.csv"
	if not path.exists():
		return None
	try:
		df = pd.read_csv(path)
		if df is None or df.empty or "value" not in df.columns:
			return None

		# Preferred: explicit snapshot price rows
		mask = (df.get("section") == "snapshot") & (df.get("key") == "price")
		rows = df.loc[mask, "value"] if mask is not None else pd.Series(dtype=float)
		for raw in rows[::-1]:
			try:
				value = float(raw)
				if value > 0:
					return value
			except Exception:
				continue

		# Fallback: any numeric price-like key if schema evolves.
		for candidate_key in ["current_price", "stock_price", "close", "last_price"]:
			mask_any = df.get("key") == candidate_key
			rows_any = df.loc[mask_any, "value"] if mask_any is not None else pd.Series(dtype=float)
			for raw in rows_any[::-1]:
				try:
					value = float(raw)
					if value > 0:
						return value
				except Exception:
					continue

		return None
	except Exception:
		return None


def _build_portfolio_from_shares(
	shares_map: dict[str, float],
	prices: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
	valid = {str(t).strip().upper(): float(s) for t, s in shares_map.items() if float(s) > 0}
	if not valid:
		return pd.DataFrame(), []

	price_map = prices or _latest_prices_map(tuple(valid.keys()))
	resolved_tickers = [ticker for ticker in valid.keys() if float(price_map.get(ticker, 0.0)) > 0]
	missing_tickers = sorted(set(valid.keys()) - set(resolved_tickers))
	if not resolved_tickers:
		return pd.DataFrame(), missing_tickers

	rows: list[dict] = []
	for ticker in resolved_tickers:
		shares = valid[ticker]
		price = float(price_map.get(ticker, 0.0))
		market_value = shares * price
		rows.append(
			{
				"Ticker": ticker,
				"Shares Parsed": float(shares),
				"Average Cost Basis Parsed": np.nan,
				"Stock Price Parsed": float(price),
				"Market Value Parsed": float(market_value),
				"Ownership Performance Parsed": np.nan,
				"Weight": float(market_value),
			}
		)

	portfolio_df = pd.DataFrame(rows)
	portfolio_df = portfolio_df[portfolio_df["Weight"] > 0]
	if portfolio_df.empty:
		return pd.DataFrame(), missing_tickers

	total_weight = float(portfolio_df["Weight"].sum())
	portfolio_df["Weight Pct"] = (portfolio_df["Weight"] / total_weight) * 100 if total_weight > 0 else 0.0
	return portfolio_df.sort_values("Weight", ascending=False).reset_index(drop=True), missing_tickers




def _normalize_portfolio_df(raw_df: pd.DataFrame) -> pd.DataFrame:
	if raw_df is None or raw_df.empty:
		raise ValueError("Uploaded file is empty.")

	df = raw_df.copy()
	df.columns = [str(c).strip().strip('"') for c in df.columns]

	column_map = {c.lower(): c for c in df.columns}
	if "ticker" not in column_map:
		raise ValueError("CSV must include a 'Ticker' column.")

	ticker_col = column_map["ticker"]
	df["Ticker"] = df[ticker_col].astype(str).str.strip().str.upper()
	df = df[df["Ticker"] != ""]
	if df.empty:
		raise ValueError("No valid tickers found in CSV.")

	market_value_col = column_map.get("market value") or column_map.get("market value parsed")
	portfolio_pct_col = (
		column_map.get("portfolio percentage")
		or column_map.get("portfolio percentage parsed")
		or column_map.get("weight pct")
	)
	stock_price_col = column_map.get("stock price") or column_map.get("stock price parsed")
	ownership_perf_col = column_map.get("ownership performance") or column_map.get("ownership performance parsed")
	shares_col = column_map.get("shares") or column_map.get("shares parsed")
	cost_basis_col = column_map.get("average cost basis") or column_map.get("average cost basis parsed")
	weight_col = column_map.get("weight")

	fx_rates = _load_recent_fx_rates_to_usd()

	if market_value_col:
		df["Market Value Parsed"] = df[market_value_col].map(lambda v: _money_value_to_usd(v, fx_rates))
	else:
		df["Market Value Parsed"] = np.nan

	if portfolio_pct_col:
		df["Portfolio Percentage Parsed"] = df[portfolio_pct_col].map(_parse_numeric_text)
	else:
		df["Portfolio Percentage Parsed"] = np.nan

	if stock_price_col:
		df["Stock Price Parsed"] = df[stock_price_col].map(lambda v: _money_value_to_usd(v, fx_rates))
	else:
		df["Stock Price Parsed"] = np.nan

	if ownership_perf_col:
		df["Ownership Performance Parsed"] = df[ownership_perf_col].map(_parse_numeric_text)
	else:
		df["Ownership Performance Parsed"] = np.nan

	if shares_col:
		df["Shares Parsed"] = df[shares_col].map(_parse_numeric_text)
	else:
		df["Shares Parsed"] = np.nan

	if cost_basis_col:
		df["Average Cost Basis Parsed"] = df[cost_basis_col].map(lambda v: _money_value_to_usd(v, fx_rates))
	else:
		df["Average Cost Basis Parsed"] = np.nan

	# Derive market value when absent but shares and stock price are present.
	missing_mv_mask = df["Market Value Parsed"].isna() | (df["Market Value Parsed"] <= 0)
	derived_mv = df["Shares Parsed"] * df["Stock Price Parsed"]
	df.loc[missing_mv_mask, "Market Value Parsed"] = derived_mv[missing_mv_mask]

	if df["Market Value Parsed"].notna().any() and (df["Market Value Parsed"] > 0).any():
		df["Weight"] = df["Market Value Parsed"].fillna(0.0)
	elif weight_col:
		df["Weight"] = df[weight_col].map(_parse_numeric_text).fillna(0.0)
	else:
		df["Weight"] = df["Portfolio Percentage Parsed"].fillna(0.0)

	df = df[df["Weight"] > 0]
	if df.empty:
		raise ValueError(
			"CSV needs positive values in either 'Market Value' or 'Portfolio Percentage' to build visualization."
		)

	total_weight = float(df["Weight"].sum())
	df["Weight Pct"] = (df["Weight"] / total_weight) * 100 if total_weight > 0 else 0.0
	return df.reset_index(drop=True)


def _portfolio_save_path(entry_name: str) -> Path:
	PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
	safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", " "} else "_" for ch in entry_name).strip()
	if not safe_name:
		safe_name = "portfolio"
	return PORTFOLIOS_DIR / f"{safe_name}.csv"


def _saved_portfolio_file_map() -> dict[str, Path]:
	PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
	return {csv_file.stem: csv_file for csv_file in sorted(PORTFOLIOS_DIR.glob("*.csv"))}


def _load_saved_portfolio_sources() -> dict[str, list[str]]:
	PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
	sources: dict[str, list[str]] = {}
	for csv_file in sorted(PORTFOLIOS_DIR.glob("*.csv")):
		try:
			df = pd.read_csv(csv_file)
		except Exception:
			continue
		if "Ticker" not in df.columns:
			continue
		tickers = sorted(
			{
				str(t).strip().upper()
				for t in df["Ticker"].tolist()
				if str(t).strip() and str(t).strip().lower() not in {"nan", "none"}
			}
		)
		if tickers:
			sources[f"{csv_file.stem}"] = tickers
	return sources
