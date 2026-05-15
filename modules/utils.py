

def fmt_money(value: float) -> str:
	return f"${value:,.2f}"

def _parse_numeric_text(value) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text or text.lower() in {"nan", "none", "null", "-"}:
		return None
	text = (
		text.replace("€", "")
		.replace("$", "")
		.replace("£", "")
		.replace("%", "")
		.replace(",", "")
		.replace(" ", "")
	)
	try:
		return float(text)
	except Exception:
		return None
