"""
Shared ticker characteristic resolution backed by ticker_classification_complete.csv.

Interdependencies:
- Used by `src/app.py` for the holding category override editor seed values.
- Used by `src/insights.py` for spread-analysis style/industry classifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TICKER_CLASSIFICATION_COLUMNS = [
    "instrument_id",
    "ticker",
    "product",
    "currency",
    "asset_class",
    "primary_style",
    "secondary_factor",
    "gics_sector",
    "gics_industry_group",
    "gics_industry",
    "gics_sub_industry",
]

_PRIMARY_STYLE_VALUES = {"Growth", "Value", "Blend", "Dividend"}
_SECONDARY_FACTOR_VALUES = {"", "Quality", "Momentum", "Cyclical", "Defensive"}


def infer_industry_bucket(*, product: str, ticker: str, is_etf: bool) -> str:
    txt = f"{product} {ticker}".upper()
    rules = [
        ("Interactive Media & Services", ["ALPHABET", "GOOGL", "GOOG", "INTERNET"]),
        ("Broadline Retail", ["AMAZON", "AMZN"]),
        ("Semiconductor Equipment", ["ASML"]),
        ("Semiconductors", ["NVIDIA", "NVDA", "MICRON", "MU"]),
        ("Communications Equipment", ["CISCO", "CSCO"]),
        ("Air Freight & Logistics", ["POST", "DHL", "LOGISTICS"]),
        ("Capital Markets", ["FLOW TRADERS", "TRADING"]),
        ("Food Products", ["GENERAL MILLS"]),
        ("Biotechnology", ["GILEAD"]),
        ("IT Services", ["IBM"]),
        ("Beverages", ["PEPSICO", "JDE"]),
        ("Pharmaceuticals", ["PFIZER", "PFE"]),
        ("Household Products", ["PROCTER", "GAMBLE", "PG"]),
        ("Restaurants", ["TAKEAWAY", "RESTAURANT"]),
        ("Electrical Equipment", ["NORDEX"]),
    ]
    for bucket, keywords in rules:
        if any(keyword in txt for keyword in keywords):
            return bucket
    return "Multi-Sector" if is_etf else "Unclassified"


def infer_style_bucket(*, product: str, ticker: str, is_etf: bool) -> str:
    txt = f"{product} {ticker}".upper()
    if any(keyword in txt for keyword in ["DIVIDEND", "ARISTOCRAT", "INCOME", "YIELD"]):
        return "Dividend"
    if any(keyword in txt for keyword in ["VALUE", "QUALITY VALUE", "DEEP VALUE"]):
        return "Value"
    if any(keyword in txt for keyword in ["GROWTH", "NASDAQ", "TECH", "INNOVATION", "MOMENTUM"]):
        return "Growth"
    if is_etf and any(keyword in txt for keyword in ["MSCI", "S&P", "WORLD", "STOXX", "ACWI"]):
        return "Blend"
    return "Blend" if is_etf else "Unclassified"


def infer_secondary_factor_bucket(*, product: str, ticker: str, primary_style: str) -> str:
    txt = f"{product} {ticker}".upper()
    if any(keyword in txt for keyword in ["MOMENTUM", "NASDAQ", "NVDA", "NVIDIA"]):
        return "Momentum"
    if any(keyword in txt for keyword in ["DEFENSIVE", "CONSUMER STAPLES", "PHARMA", "HEALTH", "INSURANCE"]):
        return "Defensive"
    if any(keyword in txt for keyword in ["DIVIDEND", "QUALITY", "ARISTOCRAT"]):
        return "Quality"
    if primary_style in {"Growth", "Value", "Blend", "Dividend"}:
        return "Cyclical"
    return ""


def load_ticker_classifications(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = _resolve_classification_path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=TICKER_CLASSIFICATION_COLUMNS)

    try:
        loaded = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame(columns=TICKER_CLASSIFICATION_COLUMNS)

    return _normalize_classification_df(loaded)


def resolve_ticker_characteristics(
    *,
    instruments_df: pd.DataFrame,
    ticker_classifications_path: Path | str | None = None,
    auto_append_missing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Resolve instrument characteristics from ticker_classification_complete.csv.

    Matching priority:
    1) instrument_id (case-insensitive)
    2) ticker (case-insensitive)

    `auto_append_missing` is accepted for backward compatibility and ignored.
    """
    del auto_append_missing

    base = _normalize_instruments_df(instruments_df)
    if base.empty:
        empty = base.assign(
            asset_class="",
            primary_style="",
            secondary_factor="",
            gics_sector="",
            gics_industry_group="",
            gics_industry="",
            gics_sub_industry="",
            style="",
            industry="",
            matched_by="none",
        )
        return empty, {
            "matched_instrument_id_count": 0,
            "matched_ticker_count": 0,
            "unmatched_count": 0,
            "appended_count": 0,
            "path": str(_resolve_classification_path(ticker_classifications_path)),
        }

    csv_path = _resolve_classification_path(ticker_classifications_path)
    classifications = load_ticker_classifications(csv_path)

    by_instrument_id: dict[str, dict[str, str]] = {}
    by_ticker: dict[str, dict[str, str]] = {}
    for row in classifications.itertuples(index=False):
        record = {
            "instrument_id": _clean_text(getattr(row, "instrument_id", "")),
            "ticker": _clean_text(getattr(row, "ticker", "")),
            "product": _clean_text(getattr(row, "product", "")),
            "currency": _clean_text(getattr(row, "currency", "")).upper(),
            "asset_class": _normalize_asset_class(_clean_text(getattr(row, "asset_class", ""))),
            "primary_style": _normalize_primary_style(_clean_text(getattr(row, "primary_style", ""))),
            "secondary_factor": _normalize_secondary_factor(
                _clean_text(getattr(row, "secondary_factor", ""))
            ),
            "gics_sector": _clean_text(getattr(row, "gics_sector", "")),
            "gics_industry_group": _clean_text(getattr(row, "gics_industry_group", "")),
            "gics_industry": _clean_text(getattr(row, "gics_industry", "")),
            "gics_sub_industry": _clean_text(getattr(row, "gics_sub_industry", "")),
        }
        instrument_key = _normalize_key(record["instrument_id"])
        ticker_key = _normalize_key(record["ticker"])
        if instrument_key and instrument_key not in by_instrument_id:
            by_instrument_id[instrument_key] = record
        if ticker_key and ticker_key not in by_ticker:
            by_ticker[ticker_key] = record

    resolved_rows: list[dict[str, Any]] = []
    matched_instrument_id_count = 0
    matched_ticker_count = 0
    unmatched_count = 0

    for row in base.itertuples(index=False):
        instrument_id = _clean_text(getattr(row, "instrument_id", ""))
        ticker = _clean_text(getattr(row, "ticker", ""))
        product = _clean_text(getattr(row, "product", ""))
        currency = _clean_text(getattr(row, "currency", "")).upper()
        is_etf = bool(getattr(row, "is_etf", False))

        matched_record: dict[str, str] | None = None
        matched_by = "none"
        instrument_key = _normalize_key(instrument_id)
        ticker_key = _normalize_key(ticker)
        if instrument_key and instrument_key in by_instrument_id:
            matched_record = by_instrument_id[instrument_key]
            matched_by = "instrument_id"
            matched_instrument_id_count += 1
        elif ticker_key and ticker_key in by_ticker:
            matched_record = by_ticker[ticker_key]
            matched_by = "ticker"
            matched_ticker_count += 1
        else:
            unmatched_count += 1

        if matched_record is not None:
            primary_style = _normalize_primary_style(matched_record.get("primary_style", ""))
            if primary_style == "":
                primary_style = infer_style_bucket(product=product, ticker=ticker, is_etf=is_etf)
            secondary_factor = _normalize_secondary_factor(matched_record.get("secondary_factor", ""))
            if secondary_factor == "":
                secondary_factor = infer_secondary_factor_bucket(
                    product=product,
                    ticker=ticker,
                    primary_style=primary_style,
                )

            gics_industry = matched_record.get("gics_industry", "") or infer_industry_bucket(
                product=product,
                ticker=ticker,
                is_etf=is_etf,
            )
            gics_sector = matched_record.get("gics_sector", "")
            gics_industry_group = matched_record.get("gics_industry_group", "")
            gics_sub_industry = matched_record.get("gics_sub_industry", "")
            if gics_sector == "":
                gics_sector = "Multi-Sector" if is_etf else "Unclassified"
            if gics_industry_group == "":
                gics_industry_group = gics_industry
            if gics_sub_industry == "":
                gics_sub_industry = gics_industry

            resolved_currency = matched_record.get("currency", "") or currency
            resolved_asset_class = matched_record.get("asset_class", "") or ("ETF" if is_etf else "Equity")
        else:
            primary_style = infer_style_bucket(product=product, ticker=ticker, is_etf=is_etf)
            secondary_factor = infer_secondary_factor_bucket(
                product=product,
                ticker=ticker,
                primary_style=primary_style,
            )
            gics_industry = infer_industry_bucket(product=product, ticker=ticker, is_etf=is_etf)
            gics_sector = "Multi-Sector" if is_etf else "Unclassified"
            gics_industry_group = gics_industry
            gics_sub_industry = gics_industry
            resolved_currency = currency
            resolved_asset_class = "ETF" if is_etf else "Equity"

        resolved_rows.append(
            {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "product": product,
                "currency": resolved_currency,
                "asset_class": _normalize_asset_class(resolved_asset_class),
                "primary_style": _normalize_primary_style(primary_style) or "Unclassified",
                "secondary_factor": _normalize_secondary_factor(secondary_factor),
                "gics_sector": _clean_text(gics_sector),
                "gics_industry_group": _clean_text(gics_industry_group),
                "gics_industry": _clean_text(gics_industry),
                "gics_sub_industry": _clean_text(gics_sub_industry),
                "style": _normalize_primary_style(primary_style) or "Unclassified",
                "industry": _clean_text(gics_industry),
                "matched_by": matched_by,
            }
        )

    resolved = pd.DataFrame(resolved_rows)
    stats = {
        "matched_instrument_id_count": int(matched_instrument_id_count),
        "matched_ticker_count": int(matched_ticker_count),
        "unmatched_count": int(unmatched_count),
        "appended_count": 0,
        "path": str(csv_path),
    }
    return resolved, stats


def _resolve_classification_path(path: Path | str | None) -> Path:
    resolved = Path(path) if path is not None else Path("ticker_classification_complete.csv")
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _normalize_instruments_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["instrument_id", "ticker", "product", "currency", "is_etf"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()
    for col in ["instrument_id", "ticker", "product", "currency"]:
        if col not in out.columns:
            out[col] = ""
        if col == "currency":
            out[col] = out[col].map(_clean_text).str.upper()
        else:
            out[col] = out[col].map(_clean_text)

    if "is_etf" not in out.columns:
        out["is_etf"] = False
    out["is_etf"] = out["is_etf"].fillna(False).astype(bool)
    return out[cols].reset_index(drop=True)


def _normalize_classification_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TICKER_CLASSIFICATION_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(_clean_text)

    out["currency"] = out["currency"].str.upper()
    out["asset_class"] = out["asset_class"].map(_normalize_asset_class)
    out["primary_style"] = out["primary_style"].map(_normalize_primary_style)
    out["secondary_factor"] = out["secondary_factor"].map(_normalize_secondary_factor)
    return out[TICKER_CLASSIFICATION_COLUMNS].reset_index(drop=True)


def _normalize_primary_style(value: Any) -> str:
    text = _clean_text(value)
    if text in _PRIMARY_STYLE_VALUES:
        return text
    return ""


def _normalize_secondary_factor(value: Any) -> str:
    text = _clean_text(value)
    if text in _SECONDARY_FACTOR_VALUES:
        return text
    return ""


def _normalize_asset_class(value: Any) -> str:
    text = _clean_text(value).upper()
    if text == "ETF":
        return "ETF"
    if text in {"EQUITY", "STOCK"}:
        return "Equity"
    return ""


def _normalize_key(value: Any) -> str:
    return _clean_text(value).upper()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text
