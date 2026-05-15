from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import StringIO
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


TIMEFRAME_OPTIONS: dict[str, int] = {
	"Last week": 7,
	"Last month": 30,
	"Last 2 months": 60,
	"Last 4 months": 120,
}

SEC_HEADERS = {
	"User-Agent": "simple-invest/1.0 (public-research; contact: noreply@example.com)",
	"Accept": "application/atom+xml,text/xml,application/xhtml+xml,text/html",
}


def _to_float(value, default=np.nan) -> float:
	try:
		if value is None:
			return float(default)
		text = str(value)
		cleaned = re.sub(r"[^0-9.\-]", "", text)
		if cleaned in {"", "-", ".", "-."}:
			return float(default)
		out = float(cleaned)
		if np.isnan(out) or np.isinf(out):
			return float(default)
		return float(out)
	except Exception:
		return float(default)


def _find_best_openinsider_table(html: str) -> pd.DataFrame:
	try:
		tables = pd.read_html(StringIO(html))
	except Exception:
		return pd.DataFrame()

	if not tables:
		return pd.DataFrame()

	for table in tables:
		cols = {str(c).strip().lower() for c in table.columns}
		has_ticker = any("ticker" in c for c in cols)
		has_trade_date = any("trade date" in c or "trade\xa0date" in c for c in cols)
		has_value = any(c == "value" or "value" in c for c in cols)
		if has_ticker and has_trade_date and has_value:
			return table.copy()

	# fallback: largest table
	return max(tables, key=lambda t: len(t.index)).copy()


@st.cache_data(show_spinner=False, ttl=1800)
def _load_openinsider_ticker_table(ticker: str, timeframe_days: int, max_rows: int = 200) -> pd.DataFrame:
	ticker = str(ticker or "").strip().upper()
	if not ticker:
		return pd.DataFrame()

	url = "https://openinsider.com/screener"
	params = {
		"s": ticker,
		"o": "",
		"pl": "",
		"ph": "",
		"ll": "",
		"lh": "",
		"fd": int(max(1, timeframe_days)),
		"fdr": "",
		"td": 0,
		"tdr": "",
		"xp": 1,
		"vl": 0,
		"vh": "",
		"ocl": "",
		"och": "",
		"sic1": "",
		"sicl": "",
		"sich": "",
		"isofficer": 1,
		"iscorp": 1,
		"isdr": "",
		"cnt": int(max(50, max_rows)),
		"page": 1,
	}
	headers = {
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
		"Accept": "text/html,application/xhtml+xml",
	}

	try:
		response = requests.get(url, params=params, headers=headers, timeout=15)
		if response.status_code >= 400:
			return pd.DataFrame()
		best = _find_best_openinsider_table(response.text)
		if best.empty:
			return pd.DataFrame()

		best.columns = [str(c).replace("\xa0", " ").strip() for c in best.columns]
		if "Ticker" not in best.columns:
			for c in best.columns:
				if "ticker" in str(c).strip().lower():
					best = best.rename(columns={c: "Ticker"})
					break
		if "Trade Date" not in best.columns:
			for c in best.columns:
				if "trade" in str(c).strip().lower() and "date" in str(c).strip().lower():
					best = best.rename(columns={c: "Trade Date"})
					break
		if "Trade Type" not in best.columns:
			for c in best.columns:
				if "trade" in str(c).strip().lower() and "type" in str(c).strip().lower():
					best = best.rename(columns={c: "Trade Type"})
					break
		if "Value" not in best.columns:
			for c in best.columns:
				if str(c).strip().lower() == "value" or "value" in str(c).strip().lower():
					best = best.rename(columns={c: "Value"})
					break

		for needed_col in ["Ticker", "Trade Date", "Trade Type", "Value"]:
			if needed_col not in best.columns:
				best[needed_col] = np.nan

		best["Ticker"] = best["Ticker"].astype(str).str.strip().str.upper()
		best["Trade Type"] = best["Trade Type"].astype(str).str.strip()
		best["Trade Date"] = pd.to_datetime(best["Trade Date"], errors="coerce")
		best["Transaction Value USD"] = best["Value"].map(lambda v: _to_float(v, default=np.nan))
		best["Source URL"] = response.url

		return best
	except Exception:
		return pd.DataFrame()


def _is_purchase(trade_type: str) -> bool:
	t = str(trade_type or "").strip().upper()
	if not t:
		return False
	if "P - PURCHASE" in t:
		return True
	if re.search(r"(^|\s|\()P(\s|\)|-|$)", t):
		return True
	return False


def _sec_strip_ns(tag: str) -> str:
	return tag.split("}")[-1] if "}" in tag else tag


def _sec_find_text(root: ET.Element, path_options: list[tuple[str, ...]]) -> str | None:
	for path in path_options:
		node = root
		ok = True
		for part in path:
			candidates = [child for child in node if _sec_strip_ns(child.tag) == part]
			if not candidates:
				ok = False
				break
			node = candidates[0]
		if ok and node is not None and node.text is not None:
			value = str(node.text).strip()
			if value:
				return value
	return None


def _sec_feed_entries(timeframe_days: int, max_pages: int = 8) -> list[dict[str, str]]:
	entries: list[dict[str, str]] = []
	cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, timeframe_days))).date()

	for page in range(max_pages):
		start = page * 100
		url = (
			"https://www.sec.gov/cgi-bin/browse-edgar"
			f"?action=getcurrent&type=4&owner=only&count=100&start={start}&output=atom"
		)
		try:
			resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
			if resp.status_code >= 400:
				break
			root = ET.fromstring(resp.text)
		except Exception:
			break

		page_entries: list[dict[str, str]] = []
		for entry in root.iter():
			if _sec_strip_ns(entry.tag) != "entry":
				continue

			entry_link = ""
			entry_updated = ""
			for child in entry:
				name = _sec_strip_ns(child.tag)
				if name == "updated" and child.text:
					entry_updated = str(child.text).strip()
				if name == "link":
					href = child.attrib.get("href", "")
					if href:
						entry_link = href

			if not entry_link:
				continue

			try:
				updated_dt = pd.to_datetime(entry_updated, errors="coerce", utc=True)
			except Exception:
				updated_dt = pd.NaT

			if pd.isna(updated_dt):
				continue

			updated_date = updated_dt.date()
			if updated_date < cutoff:
				continue

			page_entries.append(
				{
					"index_url": entry_link,
					"filed_at": updated_dt.isoformat(),
				}
			)

		if not page_entries:
			break

		entries.extend(page_entries)

		oldest_page_date = min(pd.to_datetime(e["filed_at"], utc=True).date() for e in page_entries)
		if oldest_page_date < cutoff:
			break

	return entries


def _sec_index_to_xml_url(index_url: str) -> str | None:
	if not index_url:
		return None
	try:
		resp = requests.get(index_url, headers=SEC_HEADERS, timeout=20)
		if resp.status_code >= 400:
			return None
		html = resp.text
		matches = re.findall(r'href="([^"]+\.xml)"', html, flags=re.IGNORECASE)
		if not matches:
			return None

		preferred = None
		for match in matches:
			m = str(match)
			if "ownership" in m.lower() or "form4" in m.lower():
				preferred = m
				break
		if preferred is None:
			preferred = matches[0]

		return urljoin(index_url, preferred)
	except Exception:
		return None


def _parse_sec_form4_xml(xml_text: str, source_url: str, filed_at: str) -> pd.DataFrame:
	try:
		root = ET.fromstring(xml_text)
	except Exception:
		return pd.DataFrame()

	ticker = _sec_find_text(root, [("issuer", "issuerTradingSymbol")])
	insider_name = _sec_find_text(root, [("reportingOwner", "reportingOwnerId", "rptOwnerName")])
	title = _sec_find_text(root, [("reportingOwner", "reportingOwnerRelationship", "officerTitle")])

	if not ticker:
		return pd.DataFrame()

	records: list[dict] = []

	for child in root.iter():
		if _sec_strip_ns(child.tag) != "nonDerivativeTransaction":
			continue

		transaction_code = _sec_find_text(child, [("transactionCoding", "transactionCode")]) or ""
		acq_disp = _sec_find_text(
			child,
			[("transactionAmounts", "transactionAcquiredDisposedCode", "value")],
		) or ""

		if str(transaction_code).strip().upper() != "P":
			continue
		if str(acq_disp).strip().upper() not in {"A", ""}:
			continue

		shares = _to_float(
			_sec_find_text(child, [("transactionAmounts", "transactionShares", "value")]),
			default=np.nan,
		)
		price = _to_float(
			_sec_find_text(child, [("transactionAmounts", "transactionPricePerShare", "value")]),
			default=np.nan,
		)

		if np.isnan(shares) or shares <= 0:
			continue

		value = shares * price if not np.isnan(price) and price > 0 else np.nan
		trade_date = _sec_find_text(child, [("transactionDate", "value")])
		trade_ts = pd.to_datetime(trade_date, errors="coerce")
		if pd.isna(trade_ts):
			trade_ts = pd.to_datetime(filed_at, errors="coerce", utc=True)
			if not pd.isna(trade_ts):
				trade_ts = trade_ts.tz_localize(None)

		records.append(
			{
				"Trade Date": trade_ts,
				"Ticker": str(ticker).strip().upper(),
				"Trade Type": "P - Purchase",
				"Insider Name": insider_name,
				"Title": title,
				"Price": np.nan if np.isnan(price) else float(price),
				"Qty": float(shares),
				"Owned": np.nan,
				"Delta Own": np.nan,
				"Value": np.nan if np.isnan(value) else f"${value:,.0f}",
				"Transaction Value USD": np.nan if np.isnan(value) else float(value),
				"Source URL": source_url,
			}
		)

	return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner=False, ttl=1200)
def _load_sec_form4_purchase_data(timeframe_days: int, max_filings: int = 400) -> pd.DataFrame:
	entries = _sec_feed_entries(timeframe_days=timeframe_days, max_pages=8)
	if not entries:
		return pd.DataFrame()

	entries = entries[: int(max(50, max_filings))]
	frames: list[pd.DataFrame] = []

	def _fetch_one(entry: dict[str, str]) -> pd.DataFrame:
		index_url = entry.get("index_url", "")
		filed_at = entry.get("filed_at", "")
		xml_url = _sec_index_to_xml_url(index_url)
		if not xml_url:
			return pd.DataFrame()
		try:
			resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=20)
			if resp.status_code >= 400:
				return pd.DataFrame()
			return _parse_sec_form4_xml(resp.text, source_url=index_url, filed_at=filed_at)
		except Exception:
			return pd.DataFrame()

	max_workers = 8
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(_fetch_one, e) for e in entries]
		for f in as_completed(futures):
			try:
				df = f.result()
			except Exception:
				continue
			if df is not None and not df.empty:
				frames.append(df)

	if not frames:
		return pd.DataFrame()

	out = pd.concat(frames, ignore_index=True)
	out = out.dropna(subset=["Trade Date", "Ticker", "Qty"])
	out = out[out["Qty"] > 0]
	return out.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=1200)
def load_insider_buying_data(
	tickers: tuple[str, ...],
	timeframe_days: int,
	min_transaction_value: float,
	max_rows_per_ticker: int = 200,
) -> pd.DataFrame:
	clean_tickers = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
	if not clean_tickers:
		return pd.DataFrame()

	frames: list[pd.DataFrame] = []
	max_workers = min(10, max(2, len(clean_tickers)))

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_map = {
			executor.submit(
				_load_openinsider_ticker_table,
				ticker,
				int(max(1, timeframe_days)),
				int(max(50, max_rows_per_ticker)),
			): ticker
			for ticker in clean_tickers
		}

		for future in as_completed(future_map):
			ticker = future_map[future]
			try:
				df = future.result()
			except Exception:
				continue
			if df is None or df.empty:
				continue
			df = df.copy()
			df["Ticker"] = df.get("Ticker", ticker).astype(str).str.strip().str.upper()
			frames.append(df)

	if not frames:
		out = pd.DataFrame()
	else:
		out = pd.concat(frames, ignore_index=True)

	# Fallback to SEC EDGAR feed when screen-scrape source is empty or partial.
	sec_df = _load_sec_form4_purchase_data(timeframe_days=int(max(1, timeframe_days)), max_filings=400)
	if sec_df is not None and not sec_df.empty:
		sec_df = sec_df[sec_df["Ticker"].isin(clean_tickers)]
		if out.empty:
			out = sec_df.copy()
		else:
			out = pd.concat([out, sec_df], ignore_index=True)

	if out.empty:
		return pd.DataFrame()

	out = out.dropna(subset=["Trade Date", "Transaction Value USD"])
	out = out[out["Ticker"].isin(clean_tickers)]
	out = out[out["Trade Type"].map(_is_purchase)]

	now_utc = datetime.now(timezone.utc)
	cutoff = (now_utc - timedelta(days=int(max(1, timeframe_days)))).replace(tzinfo=None)
	out = out[out["Trade Date"] >= cutoff]
	out = out[out["Transaction Value USD"] >= float(max(0.0, min_transaction_value))]

	if out.empty:
		return pd.DataFrame()

	preferred_cols = [
		"Trade Date",
		"Ticker",
		"Trade Type",
		"Insider Name",
		"Title",
		"Price",
		"Qty",
		"Owned",
		"Delta Own",
		"Value",
		"Transaction Value USD",
		"Source URL",
	]

	for col in preferred_cols:
		if col not in out.columns:
			out[col] = np.nan

	out = out[preferred_cols].copy()
	out = out.drop_duplicates(subset=["Trade Date", "Ticker", "Insider Name", "Qty", "Price", "Transaction Value USD"])
	out = out.sort_values("Transaction Value USD", ascending=False).reset_index(drop=True)
	return out


def _fmt_money(value: float) -> str:
	return f"${value:,.0f}"


def _render_summary_metrics(df: pd.DataFrame) -> None:
	total_value = float(df["Transaction Value USD"].sum()) if not df.empty else 0.0
	avg_value = float(df["Transaction Value USD"].mean()) if not df.empty else 0.0
	unique_tickers = int(df["Ticker"].nunique()) if not df.empty else 0
	latest_date = df["Trade Date"].max() if not df.empty else None

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Qualifying purchases", f"{len(df):,}")
	c2.metric("Total insider buy value", _fmt_money(total_value))
	c3.metric("Average transaction", _fmt_money(avg_value))
	c4.metric(
		"Unique companies",
		f"{unique_tickers}",
		(latest_date.strftime("Latest: %Y-%m-%d") if isinstance(latest_date, pd.Timestamp) else ""),
	)


def _render_charts(df: pd.DataFrame) -> None:
	if df.empty:
		return

	chart_df = df.copy()
	chart_df["Trade Date"] = pd.to_datetime(chart_df["Trade Date"], errors="coerce")
	chart_df = chart_df.dropna(subset=["Trade Date"])
	if chart_df.empty:
		return

	# top transactions
	top_txn = chart_df.head(20).copy()
	top_txn = top_txn.sort_values("Transaction Value USD", ascending=True)
	bar_fig = go.Figure(
		go.Bar(
			x=top_txn["Transaction Value USD"],
			y=top_txn["Ticker"],
			orientation="h",
			marker=dict(
				color=top_txn["Transaction Value USD"],
				colorscale="Viridis",
				line=dict(color="rgba(255,255,255,0.25)", width=1),
			),
			hovertemplate=(
				"<b>%{y}</b><br>"
				+ "Transaction: $%{x:,.0f}<br>"
				+ "Date: %{customdata[0]|%Y-%m-%d}<br>"
				+ "Insider: %{customdata[1]}<extra></extra>"
			),
			customdata=top_txn[["Trade Date", "Insider Name"]],
		)
	)
	bar_fig.update_layout(
		title="Top insider purchases (by transaction value)",
		xaxis_title="Transaction value (USD)",
		yaxis_title="Ticker",
		height=520,
		margin=dict(l=20, r=20, t=60, b=20),
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
	)

	# timeline bubble chart
	bubble_df = chart_df.copy()
	bubble_df["Txn Size (M)"] = bubble_df["Transaction Value USD"] / 1_000_000
	bubble_fig = px.scatter(
		bubble_df,
		x="Trade Date",
		y="Ticker",
		size="Transaction Value USD",
		color="Txn Size (M)",
		color_continuous_scale="Turbo",
		hover_data={
			"Transaction Value USD": ":,.0f",
			"Insider Name": True,
			"Trade Type": True,
			"Txn Size (M)": ":.2f",
		},
		title="Insider buying timeline (bubble size = transaction value)",
	)
	bubble_fig.update_layout(
		height=520,
		margin=dict(l=20, r=20, t=60, b=20),
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
	)
	bubble_fig.update_xaxes(title="Trade date")
	bubble_fig.update_yaxes(title="Ticker")

	left, right = st.columns([1.1, 1.4], gap="medium")
	with left:
		st.plotly_chart(bar_fig, use_container_width=True)
	with right:
		st.plotly_chart(bubble_fig, use_container_width=True)

	# aggregated totals by ticker
	agg = (
		chart_df.groupby("Ticker", as_index=False)["Transaction Value USD"]
		.sum()
		.sort_values("Transaction Value USD", ascending=False)
	)
	if not agg.empty:
		treemap_fig = px.treemap(
			agg,
			path=["Ticker"],
			values="Transaction Value USD",
			color="Transaction Value USD",
			color_continuous_scale="Blues",
			title="Total insider buying by company",
		)
		treemap_fig.update_layout(
			height=420,
			margin=dict(l=10, r=10, t=55, b=10),
			paper_bgcolor="rgba(0,0,0,0)",
			plot_bgcolor="rgba(0,0,0,0)",
		)
		st.plotly_chart(treemap_fig, use_container_width=True)


def render_insider_buying_tracker_tab(watchlist_sources: dict[str, list[str]]) -> None:
	st.subheader("Insider Buying Tracker")
	st.caption(
		"Public data sources: OpenInsider screener and SEC EDGAR Form 4 feeds. "
		"Use as a signal, then verify details on SEC EDGAR before acting."
	)

	if not watchlist_sources:
		st.warning("No watchlist or local portfolio sources were found.")
		return

	source_names = sorted(watchlist_sources.keys())
	default_source = "S&P 500" if "S&P 500" in source_names else source_names[0]
	default_timeframe_index = list(TIMEFRAME_OPTIONS.keys()).index("Last 2 months")

	ctl_a, ctl_b, ctl_c = st.columns([1.5, 1.1, 1.0], gap="small")
	with ctl_a:
		selected_source = st.selectbox(
			"Universe source",
			options=source_names,
			index=source_names.index(default_source),
			key="insider_tracker_source",
		)
	with ctl_b:
		timeframe_label = st.selectbox(
			"Timeframe",
			options=list(TIMEFRAME_OPTIONS.keys()),
			index=default_timeframe_index,
			key="insider_tracker_timeframe",
		)
	with ctl_c:
		min_value_million = st.number_input(
			"Minimum transaction (Million USD)",
			min_value=0.0,
			value=0.1,
			step=0.05,
			format="%.3f",
			key="insider_tracker_min_value",
		)
		min_value = float(min_value_million) * 1_000_000.0

	universe = sorted({str(t).strip().upper() for t in watchlist_sources.get(selected_source, []) if str(t).strip()})
	if not universe:
		st.info("No tickers found in the selected source.")
		return

	with st.expander("Universe controls", expanded=False):
		selected_tickers = st.multiselect(
			"Track tickers",
			options=universe,
			default=universe,
			key="insider_tracker_tickers",
		)
		max_rows = st.slider(
			"Max rows fetched per ticker",
			min_value=50,
			max_value=400,
			value=200,
			step=25,
			key="insider_tracker_max_rows",
		)

	if st.button("Refresh insider data", use_container_width=True, key="insider_tracker_refresh_btn"):
		load_insider_buying_data.clear()
		_load_openinsider_ticker_table.clear()

	if not selected_tickers:
		st.info("Select at least one ticker to analyze insider purchases.")
		return

	timeframe_days = TIMEFRAME_OPTIONS[timeframe_label]

	with st.spinner("Scanning latest insider purchases from public filings..."):
		df = load_insider_buying_data(
			tickers=tuple(selected_tickers),
			timeframe_days=int(timeframe_days),
			min_transaction_value=float(min_value),
			max_rows_per_ticker=int(max_rows),
		)

	if df.empty:
		st.warning(
			"No qualifying insider purchase transactions were found for the selected filters. "
			"Try lowering the threshold or expanding timeframe."
		)
		return

	_render_summary_metrics(df)
	st.divider()
	_render_charts(df)

	show_cols = [
		"Trade Date",
		"Ticker",
		"Insider Name",
		"Title",
		"Trade Type",
		"Transaction Value USD",
		"Value",
		"Qty",
		"Price",
		"Source URL",
	]
	for col in show_cols:
		if col not in df.columns:
			df[col] = np.nan

	table_df = df[show_cols].copy()
	table_df = table_df.sort_values(["Transaction Value USD", "Trade Date"], ascending=[False, False]).reset_index(drop=True)

	st.markdown("### Latest qualifying insider buys")
	st.dataframe(
		table_df,
		use_container_width=True,
		hide_index=True,
		column_config={
			"Trade Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
			"Transaction Value USD": st.column_config.NumberColumn(format="$%,.0f"),
			"Source URL": st.column_config.LinkColumn("Source"),
		},
	)

	total_value = float(table_df["Transaction Value USD"].sum()) if not table_df.empty else 0.0
	st.success(
		f"Tracker complete: {len(table_df):,} qualifying purchases across "
		f"{table_df['Ticker'].nunique()} tickers | Total value {_fmt_money(total_value)}."
	)
