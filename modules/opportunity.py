from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def _analysis_get(df: pd.DataFrame, section: str, key: str, metric: str | None = None) -> str | None:
	if df is None or df.empty:
		return None
	mask = (df.get("section") == section) & (df.get("key") == key)
	if metric is not None:
		mask = mask & (df.get("metric") == metric)
	rows = df.loc[mask]
	if rows.empty:
		return None
	return str(rows.iloc[-1]["value"])


def _parse_float(value: str | None, default=np.nan) -> float:
	try:
		if value is None or value == "":
			return float(default)
		return float(value)
	except Exception:
		return float(default)


def _parse_growth_rates(df: pd.DataFrame) -> list[float]:
	growth_json = _analysis_get(df, "valuation", "growth_rates_json")
	if growth_json:
		try:
			rates_pct = json.loads(growth_json)
			if isinstance(rates_pct, list) and rates_pct:
				return [float(x) / 100 for x in rates_pct]
		except Exception:
			pass

	growth_rate_pct = _parse_float(_analysis_get(df, "valuation", "growth_rate_pct"), np.nan)
	years = int(_parse_float(_analysis_get(df, "valuation", "years"), 5.0))
	if np.isnan(growth_rate_pct):
		growth_rate_pct = 10.0
	return [growth_rate_pct / 100] * max(1, years)


def _project_terminal_per_share_with_growth_path(
	metric_ttm: float,
	shares_outstanding: float,
	growth_rates: list[float],
	buyback_rate: float,
	exit_multiple: float,
) -> float:
	metric_value = float(metric_ttm)
	shares_value = float(shares_outstanding)
	initial_shares = float(shares_outstanding)

	for g in growth_rates:
		metric_value = metric_value * (1 + float(g))
		shares_value = max(shares_value * (1 - float(buyback_rate)), initial_shares * 0.1)

	terminal_value = metric_value * float(exit_multiple)
	return terminal_value / shares_value


def _dcf_enterprise_value(
	fcf_ttm: float,
	years: int,
	explicit_growth_rate: float,
	discount_rate: float,
	terminal_growth_rate: float,
) -> float:
	projected_fcfs = [fcf_ttm * (1 + explicit_growth_rate) ** year for year in range(1, years + 1)]
	discounted_fcfs = [
		fcf / (1 + discount_rate) ** year for year, fcf in enumerate(projected_fcfs, start=1)
	]
	sum_discounted_fcfs = float(sum(discounted_fcfs))

	terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
	terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
	pv_terminal_value = terminal_value / (1 + discount_rate) ** years

	return sum_discounted_fcfs + pv_terminal_value


def _get_snapshot_value(df: pd.DataFrame, key: str, metric_name: str | None = None) -> float:
	if metric_name:
		metric_raw = _analysis_get(df, "snapshot", key, metric_name)
		metric_val = _parse_float(metric_raw, np.nan)
		if not np.isnan(metric_val):
			return metric_val

	rows = df.loc[(df.get("section") == "snapshot") & (df.get("key") == key)]
	if rows.empty:
		return np.nan
	return _parse_float(str(rows.iloc[-1]["value"]), np.nan)


def _compute_opportunity_row(df: pd.DataFrame, ticker: str) -> dict[str, float | str]:
	metric_name = _analysis_get(df, "valuation", "metric_name") or "FCF (Free Cash Flow)"

	price = _get_snapshot_value(df, "price", metric_name)
	shares = _get_snapshot_value(df, "shares_outstanding", metric_name)
	metric_ttm = _get_snapshot_value(df, "metric_ttm", metric_name)

	years = int(_parse_float(_analysis_get(df, "valuation", "years"), 5.0))
	desired_return = _parse_float(_analysis_get(df, "valuation", "desired_return_pct"), 15.0) / 100
	buyback_rate = _parse_float(_analysis_get(df, "valuation", "buyback_rate_pct"), 1.0) / 100
	exit_multiple = _parse_float(_analysis_get(df, "valuation", "exit_multiple"), np.nan)
	growth_rates = _parse_growth_rates(df)

	exit_upside = np.nan
	expected_cagr = np.nan
	if (
		not np.isnan(price)
		and price > 0
		and not np.isnan(shares)
		and shares > 0
		and not np.isnan(metric_ttm)
		and metric_ttm > 0
		and not np.isnan(exit_multiple)
		and exit_multiple > 0
	):
		terminal_per_share = _project_terminal_per_share_with_growth_path(
			metric_ttm=metric_ttm,
			shares_outstanding=shares,
			growth_rates=growth_rates,
			buyback_rate=buyback_rate,
			exit_multiple=exit_multiple,
		)
		expected_cagr = (terminal_per_share / price) ** (1 / max(1, years)) - 1
		fair_value_today = terminal_per_share / (1 + desired_return) ** max(1, years)
		exit_upside = fair_value_today / price - 1

	# Reverse DCF upside
	fcf_price = _get_snapshot_value(df, "price", "FCF (Free Cash Flow)")
	if np.isnan(fcf_price):
		fcf_price = price
	fcf_shares = _get_snapshot_value(df, "shares_outstanding", "FCF (Free Cash Flow)")
	if np.isnan(fcf_shares):
		fcf_shares = shares

	projection_years = int(_parse_float(_analysis_get(df, "reverse", "projection_years"), np.nan))
	starting_fcf_billions = _parse_float(_analysis_get(df, "reverse", "starting_fcf_billions"), np.nan)
	explicit_growth_rate = _parse_float(_analysis_get(df, "reverse", "explicit_growth_rate_pct"), np.nan) / 100
	terminal_growth_rate = _parse_float(_analysis_get(df, "reverse", "terminal_growth_rate_pct"), np.nan) / 100
	discount_rate = _parse_float(_analysis_get(df, "reverse", "discount_rate_pct"), np.nan) / 100
	cash = _parse_float(_analysis_get(df, "company_data", "cash"), np.nan)
	total_debt = _parse_float(_analysis_get(df, "company_data", "total_debt"), np.nan)

	dcf_upside = np.nan
	if (
		not np.isnan(fcf_price)
		and fcf_price > 0
		and not np.isnan(fcf_shares)
		and fcf_shares > 0
		and not np.isnan(starting_fcf_billions)
		and not np.isnan(explicit_growth_rate)
		and not np.isnan(terminal_growth_rate)
		and not np.isnan(discount_rate)
		and not np.isnan(cash)
		and not np.isnan(total_debt)
		and projection_years >= 3
		and discount_rate > terminal_growth_rate
	):
		enterprise_value = _dcf_enterprise_value(
			fcf_ttm=starting_fcf_billions * 1_000_000_000,
			years=projection_years,
			explicit_growth_rate=explicit_growth_rate,
			discount_rate=discount_rate,
			terminal_growth_rate=terminal_growth_rate,
		)
		equity_value = enterprise_value + cash - total_debt
		intrinsic_value_per_share = equity_value / fcf_shares
		dcf_upside = intrinsic_value_per_share / fcf_price - 1

	return {
		"Ticker": ticker,
		"Metric": metric_name,
		"Exit Model Upside %": exit_upside * 100 if not np.isnan(exit_upside) else np.nan,
		"Expected Return (CAGR) %": expected_cagr * 100 if not np.isnan(expected_cagr) else np.nan,
		"Reverse DCF Upside %": dcf_upside * 100 if not np.isnan(dcf_upside) else np.nan,
	}


def load_opportunity_dataframe(analyses_dir: Path | str = "analyses") -> pd.DataFrame:
	root = Path(analyses_dir)
	if not root.exists():
		return pd.DataFrame(columns=["Ticker", "Metric", "Exit Model Upside %", "Expected Return (CAGR) %", "Reverse DCF Upside %"])

	rows: list[dict[str, float | str]] = []
	for file_path in sorted(root.glob("*.csv")):
		if file_path.name.lower() == "portfolio.csv":
			continue
		try:
			df = pd.read_csv(file_path)
			ticker = str(file_path.stem).strip().upper()
			rows.append(_compute_opportunity_row(df, ticker=ticker))
		except Exception:
			continue

	if not rows:
		return pd.DataFrame(columns=["Ticker", "Metric", "Exit Model Upside %", "Expected Return (CAGR) %", "Reverse DCF Upside %"])

	out = pd.DataFrame(rows)
	return out.sort_values("Expected Return (CAGR) %", ascending=False, na_position="last").reset_index(drop=True)


def render_opportunity_cost_tab(analyses_dir: Path | str = "analyses") -> None:
	st.subheader("Opportunity cost")
	st.caption("Ranks saved valuation analyses by upside using both exit-multiple and reverse DCF views.")

	op_df = load_opportunity_dataframe(analyses_dir=analyses_dir)
	if op_df.empty:
		st.info("No saved valuation analyses found yet. Save a few company analyses first.")
		return

	left, right = st.columns(2, gap="large")
	with left:
		st.markdown("#### Highest upside (Exit Multiple model)")
		exit_df = op_df.dropna(subset=["Exit Model Upside %"]).sort_values("Exit Model Upside %", ascending=False)
		if exit_df.empty:
			st.caption("No exit-model values available in saved analyses.")
		else:
			st.dataframe(
				exit_df[["Ticker", "Metric", "Exit Model Upside %", "Expected Return (CAGR) %"]]
				.round(2),
				use_container_width=True,
			)

	with right:
		st.markdown("#### Highest upside (Reverse DCF model)")
		dcf_df = op_df.dropna(subset=["Reverse DCF Upside %"]).sort_values("Reverse DCF Upside %", ascending=False)
		if dcf_df.empty:
			st.caption("No reverse DCF values available in saved analyses.")
		else:
			st.dataframe(
				dcf_df[["Ticker", "Reverse DCF Upside %"]].round(2),
				use_container_width=True,
			)

	st.markdown("#### Stocks with highest expected return")
	plot_df = (
		op_df.dropna(subset=["Expected Return (CAGR) %"])
		.sort_values("Expected Return (CAGR) %", ascending=False)
		.head(10)
	)
	if plot_df.empty:
		st.caption("No expected return values available to plot.")
		return

	fig = px.bar(
		plot_df.sort_values("Expected Return (CAGR) %", ascending=True),
		x="Expected Return (CAGR) %",
		y="Ticker",
		orientation="h",
		color="Expected Return (CAGR) %",
		color_continuous_scale="Viridis",
		title="Top expected-return opportunities",
	)
	fig.update_layout(
		xaxis_title="Expected Return (CAGR) %",
		yaxis_title="",
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		margin=dict(l=10, r=20, t=45, b=20),
	)
	st.plotly_chart(fig, use_container_width=True)
