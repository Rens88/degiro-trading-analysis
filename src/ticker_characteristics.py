"""
Shared ticker characteristic resolution backed by ticker_classifications.csv.

Interdependencies:
- Used by `src/app.py` for the holding category override editor seed values.
- Used by `src/insights.py` for spread-analysis style/industry classifications.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


TICKER_CLASSIFICATION_COLUMNS = [
    "instrument_id",
    "ticker",
    "product",
    "auto_style",
    "style",
    "auto_industry",
    "industry",
    "classification_description",
    "classification_source",
]
PLACEHOLDER_VALUE = "t.b.d."


def infer_industry_bucket(*, product: str, ticker: str, is_etf: bool) -> str:
    txt = f"{product} {ticker}".upper()
    rules = [
        ("Technology", ["TECH", "SOFTWARE", "SEMICON", "CHIP", "CLOUD", "CYBER", "NASDAQ"]),
        ("Financials", ["BANK", "FINAN", "INSUR", "PAYMENT", "CAPITAL"]),
        ("Healthcare", ["HEALTH", "PHARMA", "BIOTECH", "MEDIC", "THERAP"]),
        ("Consumer", ["CONSUM", "RETAIL", "FOOD", "BEVERAGE", "LUXURY", "APPAREL"]),
        ("Industrials", ["INDUSTR", "AEROSPACE", "DEFEN", "TRANSPORT", "LOGISTICS"]),
        ("Energy", ["ENERGY", "OIL", "GAS", "SOLAR", "WIND", "UTILIT"]),
        ("Materials", ["MINING", "METAL", "STEEL", "CHEMICAL", "MATERIAL"]),
        ("Real Estate", ["REAL ESTATE", "REIT", "PROPERTY"]),
        ("Communication", ["TELECOM", "MEDIA", "COMMUNICATION", "INTERNET"]),
    ]
    for name, keywords in rules:
        if any(keyword in txt for keyword in keywords):
            return name
    if is_etf:
        if any(keyword in txt for keyword in ["MSCI", "S&P", "STOXX", "WORLD", "ALL-WORLD", "ACWI"]):
            return "Broad-market ETF"
        return "ETF (other)"
    return "Unclassified"


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
    return "Unclassified"


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
    Resolve style/industry characteristics for instruments.

    Matching priority:
    1) instrument_id (case-insensitive)
    2) ticker (case-insensitive)
    """
    base = _normalize_instruments_df(instruments_df)
    if base.empty:
        return base.assign(auto_style="", style="", auto_industry="", industry="", matched_by="none"), {
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
    existing_primary_keys: set[str] = set()
    for row in classifications.itertuples(index=False):
        record = {
            "instrument_id": _clean_text(getattr(row, "instrument_id", "")),
            "ticker": _clean_text(getattr(row, "ticker", "")),
            "product": _clean_text(getattr(row, "product", "")),
            "auto_style": _clean_text(getattr(row, "auto_style", "")),
            "style": _clean_text(getattr(row, "style", "")),
            "auto_industry": _clean_text(getattr(row, "auto_industry", "")),
            "industry": _clean_text(getattr(row, "industry", "")),
            "classification_description": _clean_text(getattr(row, "classification_description", "")),
            "classification_source": _clean_text(getattr(row, "classification_source", "")),
        }
        instrument_key = _normalize_key(record["instrument_id"])
        ticker_key = _normalize_key(record["ticker"])
        if instrument_key and instrument_key not in by_instrument_id:
            by_instrument_id[instrument_key] = record
        if ticker_key and ticker_key not in by_ticker:
            by_ticker[ticker_key] = record
        primary_key = _primary_key(record["instrument_id"], record["ticker"])
        if primary_key:
            existing_primary_keys.add(primary_key)

    resolved_rows: list[dict[str, Any]] = []
    rows_to_append: list[dict[str, str]] = []
    append_keys: set[str] = set()
    matched_instrument_id_count = 0
    matched_ticker_count = 0
    unmatched_count = 0

    for row in base.itertuples(index=False):
        instrument_id = _clean_text(getattr(row, "instrument_id", ""))
        ticker = _clean_text(getattr(row, "ticker", ""))
        product = _clean_text(getattr(row, "product", ""))
        is_etf = bool(getattr(row, "is_etf", False))
        inferred_style = infer_style_bucket(product=product, ticker=ticker, is_etf=is_etf)
        inferred_industry = infer_industry_bucket(product=product, ticker=ticker, is_etf=is_etf)

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
            auto_style = matched_record["auto_style"] or inferred_style
            auto_industry = matched_record["auto_industry"] or inferred_industry
            style = matched_record["style"] or auto_style
            industry = matched_record["industry"] or auto_industry
        else:
            auto_style = inferred_style
            auto_industry = inferred_industry
            if auto_append_missing:
                style = PLACEHOLDER_VALUE
                industry = PLACEHOLDER_VALUE
            else:
                style = auto_style
                industry = auto_industry

            if auto_append_missing:
                primary_key = _primary_key(instrument_id, ticker)
                if primary_key and primary_key not in existing_primary_keys and primary_key not in append_keys:
                    rows_to_append.append(
                        {
                            "instrument_id": instrument_id,
                            "ticker": ticker,
                            "product": product,
                            "auto_style": auto_style,
                            "style": PLACEHOLDER_VALUE,
                            "auto_industry": auto_industry,
                            "industry": PLACEHOLDER_VALUE,
                            "classification_description": PLACEHOLDER_VALUE,
                            "classification_source": PLACEHOLDER_VALUE,
                        }
                    )
                    append_keys.add(primary_key)

        resolved_rows.append(
            {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "product": product,
                "is_etf": is_etf,
                "auto_style": auto_style,
                "style": style,
                "auto_industry": auto_industry,
                "industry": industry,
                "matched_by": matched_by,
            }
        )

    appended_count = 0
    if auto_append_missing and rows_to_append:
        append_df = pd.DataFrame(rows_to_append, columns=TICKER_CLASSIFICATION_COLUMNS)
        updated = pd.concat([classifications, append_df], ignore_index=True)
        updated = _normalize_classification_df(updated)
        _atomic_write_classifications(csv_path, updated)
        appended_count = int(len(append_df))

    resolved = pd.DataFrame(resolved_rows)
    stats = {
        "matched_instrument_id_count": int(matched_instrument_id_count),
        "matched_ticker_count": int(matched_ticker_count),
        "unmatched_count": int(unmatched_count),
        "appended_count": int(appended_count),
        "path": str(csv_path),
    }
    return resolved, stats


def _resolve_classification_path(path: Path | str | None) -> Path:
    resolved = Path(path) if path is not None else Path("ticker_classifications.csv")
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _normalize_instruments_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["instrument_id", "ticker", "product", "is_etf"])
    out = df.copy()
    for col in ["instrument_id", "ticker", "product"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(_clean_text)
    if "is_etf" not in out.columns:
        out["is_etf"] = False
    out["is_etf"] = out["is_etf"].fillna(False).astype(bool)
    return out[["instrument_id", "ticker", "product", "is_etf"]].reset_index(drop=True)


def _normalize_classification_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TICKER_CLASSIFICATION_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(_clean_text)
    return out[TICKER_CLASSIFICATION_COLUMNS].reset_index(drop=True)


def _normalize_key(value: Any) -> str:
    return _clean_text(value).upper()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def _primary_key(instrument_id: str, ticker: str) -> str:
    instrument_key = _normalize_key(instrument_id)
    if instrument_key:
        return f"instrument_id::{instrument_key}"
    ticker_key = _normalize_key(ticker)
    if ticker_key:
        return f"ticker::{ticker_key}"
    return ""


def _atomic_write_classifications(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
