"""
AGENT_NOTE: DEGIRO CSV ingestion and normalization.

Interdependencies:
- Produces `LoadedDataset` consumed by `src/app.py` and `src/strategy_check.py`.
- Normalized column names are assumed by `src/reconciliation.py`,
  `src/portfolio_timeseries.py`, `src/tables.py`, and `src/insights.py`.

When editing:
- Treat normalized column names/types as a shared schema.
- If you rename columns here, review all downstream modules and tests.
- See `src/INTERDEPENDENCIES.md` for dependency checklist.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from yaml import YAMLError

from .config import DEFAULT_ETF_ISINS, REQUIRED_DATASET_FILES
from .exceptions import UserFacingError


@dataclass
class LoadedDataset:
    account_label: str
    transactions: pd.DataFrame
    portfolio: pd.DataFrame
    account: pd.DataFrame
    instruments: pd.DataFrame
    warnings: list[str]
    issues: list[dict[str, Any]]


def validate_uploaded_file_set(uploaded_files: list[Any]) -> dict[str, Any]:
    by_name = {f.name: f for f in uploaded_files}
    missing = [name for name in REQUIRED_DATASET_FILES if name not in by_name]
    extras = [name for name in by_name if name not in REQUIRED_DATASET_FILES]
    if missing or extras:
        parts: list[str] = []
        if missing:
            parts.append(f"Missing files: {', '.join(sorted(missing))}")
        if extras:
            parts.append(f"Unexpected files: {', '.join(sorted(extras))}")
        raise UserFacingError(
            "Uploaded files do not match required DEGIRO export names.",
            "Upload exactly these files per dataset: Transactions.csv, Portfolio.csv, Account.csv.\n"
            + "\n".join(parts),
        )
    return by_name


def load_dataset(
    *,
    account_label: str,
    transactions_source: Any,
    portfolio_source: Any,
    account_source: Any,
    mappings_path: str | Path = "mappings.yml",
) -> LoadedDataset:
    warnings: list[str] = []
    issues: list[dict[str, Any]] = []

    transactions_raw = load_csv_generic(transactions_source)
    portfolio_raw = load_csv_generic(portfolio_source)
    account_raw = load_csv_generic(account_source)

    transactions = normalize_transactions(transactions_raw, account_label=account_label, warnings=warnings)
    portfolio = normalize_portfolio(portfolio_raw, account_label=account_label, warnings=warnings)
    account = normalize_account(account_raw, account_label=account_label, warnings=warnings)

    mappings = load_mappings(mappings_path)
    instruments = resolve_instrument_mapping(
        portfolio=portfolio,
        transactions=transactions,
        mappings=mappings,
    )

    transactions = attach_instrument_metadata(transactions, instruments)
    portfolio = attach_instrument_metadata(portfolio, instruments)

    critical_warning_messages, critical_issues = validate_critical_columns(
        transactions=transactions,
        portfolio=portfolio,
        account=account,
    )
    warnings.extend(critical_warning_messages)
    issues.extend(critical_issues)

    return LoadedDataset(
        account_label=account_label,
        transactions=transactions,
        portfolio=portfolio,
        account=account,
        instruments=instruments,
        warnings=warnings,
        issues=issues,
    )


def load_csv_generic(source: Any) -> pd.DataFrame:
    raw_bytes = _read_source_bytes(source)
    text = _decode_bytes(raw_bytes)
    sep = _detect_separator(text)
    repaired_text = _repair_degiro_csv_text(text=text, sep=sep)
    buffer = io.StringIO(repaired_text)
    df = pd.read_csv(buffer, sep=sep, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_source_bytes(source: Any) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _detect_separator(text: str) -> str:
    sample = text[:5000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        return ";" if first_line.count(";") > first_line.count(",") else ","


def _repair_degiro_csv_text(*, text: str, sep: str) -> str:
    if not text.strip():
        return text

    reader = csv.reader(io.StringIO(text), delimiter=sep, quotechar='"')
    rows = list(reader)
    if not rows:
        return text

    header = _pad_row(rows[0], len(rows[0]), sep)
    width = len(header)
    col_date = _find_column_index_from_header(header, ["Date", "Datum"])
    col_time = _find_column_index_from_header(header, ["Time", "Tijd"])
    col_product = _find_column_index_from_header(header, ["Product", "Instrument"])
    col_description = _find_column_index_from_header(header, ["Description", "Omschrijving"])
    col_order_id = _find_column_index_from_header(header, ["Order Id", "Order ID"])

    repaired_rows: list[list[str]] = [header]
    for row in rows[1:]:
        cur = _pad_row(row, width, sep)
        if all(str(v).strip() == "" for v in cur):
            continue

        non_empty = [idx for idx, value in enumerate(cur) if str(value).strip() != ""]
        if (
            len(repaired_rows) > 1
            and _is_continuation_candidate(
                row=cur,
                non_empty=non_empty,
                col_date=col_date,
                col_time=col_time,
            )
            and len(non_empty) == 1
        ):
            prev = repaired_rows[-1]
            only_idx = non_empty[0]
            value = str(cur[only_idx]).strip()
            if col_product is not None and only_idx == col_product:
                prev[col_product] = _merge_text_field(prev[col_product], value)
                continue
            if col_description is not None and only_idx == col_description:
                if _looks_like_amount_currency_fragment(value):
                    continue
                prev[col_description] = _merge_text_field(prev[col_description], value)
                continue
            if col_order_id is not None and only_idx == col_order_id:
                prev[col_order_id] = _merge_concat_field(prev[col_order_id], value)
                continue

        repaired_rows.append(cur)

    out = io.StringIO()
    writer = csv.writer(out, delimiter=sep, quotechar='"', lineterminator="\n")
    writer.writerows(repaired_rows)
    return out.getvalue()


def _find_column_index_from_header(header: list[str], aliases: list[str]) -> int | None:
    lookup = {_normalize_key(col): idx for idx, col in enumerate(header)}
    for alias in aliases:
        idx = lookup.get(_normalize_key(alias))
        if idx is not None:
            return idx
    return None


def _pad_row(row: list[str], width: int, sep: str) -> list[str]:
    if len(row) == width:
        return [str(v).strip() for v in row]
    if len(row) < width:
        return [str(v).strip() for v in row] + [""] * (width - len(row))
    return [str(v).strip() for v in row[: width - 1]] + [sep.join(str(v).strip() for v in row[width - 1 :])]


def _is_continuation_candidate(
    *,
    row: list[str],
    non_empty: list[int],
    col_date: int | None,
    col_time: int | None,
) -> bool:
    if not non_empty:
        return False
    if len(non_empty) > 2:
        return False
    if col_date is not None and row[col_date] != "":
        return False
    if col_time is not None and row[col_time] != "":
        return False
    return True


def _merge_text_field(base: str, tail: str) -> str:
    base_txt = str(base).strip()
    tail_txt = str(tail).strip()
    if base_txt == "":
        return tail_txt
    if tail_txt == "":
        return base_txt
    return f"{base_txt} {tail_txt}".strip()


def _merge_concat_field(base: str, tail: str) -> str:
    return f"{str(base).strip()}{str(tail).strip()}"


def _looks_like_amount_currency_fragment(value: str) -> bool:
    return bool(re.match(r"^[+-]?\d+(?:[.,]\d+)?\s+[A-Z]{3}$", str(value).strip().upper()))


def _normalize_key(value: str) -> str:
    value = re.sub(r"[_\s]+", " ", value.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", value)


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {_normalize_key(col): col for col in df.columns}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = _column_lookup(df)
    for alias in aliases:
        key = _normalize_key(alias)
        if key in lookup:
            return lookup[key]
    return None


def _find_currency_column_near(df: pd.DataFrame, anchor_col: str | None) -> str | None:
    if anchor_col is None or anchor_col not in df.columns:
        return _find_column(df, ["currency", "valuta"])
    columns = list(df.columns)
    anchor_idx = columns.index(anchor_col)
    for offset in (1, 2, -1):
        idx = anchor_idx + offset
        if idx < 0 or idx >= len(columns):
            continue
        candidate = columns[idx]
        values = df[candidate].dropna().astype(str).str.strip().str.upper()
        if values.empty:
            continue
        ratio = values.str.match(r"^[A-Z]{3}$").mean()
        if ratio > 0.50:
            return candidate
    return _find_column(df, ["currency", "valuta"])


def _numeric_like_ratio(series: pd.Series) -> float:
    raw = series.fillna("").astype(str).str.strip()
    valid = raw.ne("")
    if not valid.any():
        return 0.0
    parsed = raw[valid].map(parse_decimal)
    return float(parsed.notna().mean())


def _currency_like_ratio(series: pd.Series) -> float:
    raw = series.fillna("").astype(str).str.strip().str.upper()
    valid = raw.ne("")
    if not valid.any():
        return 0.0
    return float(raw[valid].str.match(r"^[A-Z]{3}$").mean())


def _find_nearby_by_score(
    *,
    df: pd.DataFrame,
    anchor_col: str,
    score_fn: Any,
    offsets: tuple[int, ...] = (1, 2, -1, -2, 3, -3),
    threshold: float = 0.50,
) -> str | None:
    if anchor_col not in df.columns:
        return None
    columns = list(df.columns)
    anchor_idx = columns.index(anchor_col)
    best_col: str | None = None
    best_score = -1.0
    for offset in offsets:
        idx = anchor_idx + offset
        if idx < 0 or idx >= len(columns):
            continue
        candidate = columns[idx]
        score = float(score_fn(df[candidate]))
        if score > best_score:
            best_score = score
            best_col = candidate
        if score >= threshold:
            return candidate
    if best_score > 0:
        return best_col
    return None


def _resolve_amount_currency_columns(
    *,
    df: pd.DataFrame,
    anchor_col: str | None,
) -> tuple[str | None, str | None]:
    if anchor_col is None or anchor_col not in df.columns:
        return None, _find_column(df, ["currency", "valuta"])

    anchor_numeric = _numeric_like_ratio(df[anchor_col])
    anchor_currency = _currency_like_ratio(df[anchor_col])
    nearby_numeric = _find_nearby_by_score(df=df, anchor_col=anchor_col, score_fn=_numeric_like_ratio)
    nearby_currency = _find_nearby_by_score(df=df, anchor_col=anchor_col, score_fn=_currency_like_ratio)
    fallback_currency = _find_column(df, ["currency", "valuta"])

    if anchor_currency >= 0.60 and anchor_currency > anchor_numeric:
        amount_col = nearby_numeric
        currency_col = anchor_col
    elif anchor_numeric >= 0.60 and anchor_numeric >= anchor_currency:
        amount_col = anchor_col
        currency_col = nearby_currency or fallback_currency
    elif anchor_numeric >= anchor_currency:
        amount_col = anchor_col
        currency_col = nearby_currency or fallback_currency
    else:
        amount_col = nearby_numeric or anchor_col
        currency_col = anchor_col if anchor_currency > 0 else (nearby_currency or fallback_currency)

    return amount_col, currency_col


def parse_decimal(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace("\xa0", "")
    if text in {"", "-", "nan", "None"}:
        return np.nan
    text = text.replace('"', "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        result = float(text)
    except ValueError:
        return np.nan
    return -result if negative else result


def parse_datetime_columns(date_values: pd.Series, time_values: pd.Series | None = None) -> pd.Series:
    if time_values is None:
        return pd.to_datetime(date_values, dayfirst=True, errors="coerce")
    combined = (
        date_values.fillna("").astype(str).str.strip()
        + " "
        + time_values.fillna("").astype(str).str.strip()
    ).str.strip()
    return pd.to_datetime(combined, dayfirst=True, errors="coerce")


def normalize_transactions(df: pd.DataFrame, *, account_label: str, warnings: list[str]) -> pd.DataFrame:
    col_date = _find_column(df, ["Date", "Datum"])
    col_time = _find_column(df, ["Time", "Tijd"])
    col_product = _find_column(df, ["Product", "Instrument"])
    col_isin = _find_column(df, ["ISIN"])
    col_symbol = _find_column(df, ["Symbol", "Ticker"])
    col_quantity = _find_column(df, ["Quantity", "Aantal"])
    col_price = _find_column(df, ["Price", "Koers"])
    col_value_eur = _find_column(df, ["Value EUR", "Waarde EUR", "Value in EUR"])
    col_total_eur = _find_column(df, ["Total EUR", "Totaal EUR", "Total"])
    col_fee = _find_column(
        df,
        [
            "Transaction and/or third party fees EUR",
            "Transaction fees EUR",
            "Kosten EUR",
        ],
    )
    col_autofx = _find_column(df, ["AutoFX Fee", "AutoFX kosten"])
    col_description = _find_column(df, ["Description", "Omschrijving"])
    col_type = _find_column(df, ["Type", "Soort", "Venue"])
    col_currency = _find_currency_column_near(df, col_price)

    if col_date is None or col_product is None:
        raise UserFacingError(
            "Could not parse Transactions.csv: required date/product columns not found.",
            "Check if the file is a DEGIRO export and that headers are present.",
        )

    result = pd.DataFrame(
        {
            "datetime": parse_datetime_columns(df[col_date], df[col_time] if col_time else None),
            "account_label": account_label,
            "product": _safe_series(df, col_product).fillna("").astype(str).str.strip(),
            "isin": _safe_series(df, col_isin).fillna("").astype(str).str.strip(),
            "symbol": _safe_series(df, col_symbol).fillna("").astype(str).str.strip(),
            "quantity": _safe_series(df, col_quantity).map(parse_decimal),
            "price": _safe_series(df, col_price).map(parse_decimal),
            "value_eur": _safe_series(df, col_value_eur).map(parse_decimal),
            "total_eur": _safe_series(df, col_total_eur).map(parse_decimal),
            "currency": _safe_series(df, col_currency).fillna("").astype(str).str.upper().str.strip(),
            "description": _safe_series(df, col_description).fillna("").astype(str).str.strip(),
            "type": _safe_series(df, col_type).fillna("").astype(str).str.strip(),
            "fees_eur": _safe_series(df, col_fee).map(parse_decimal).fillna(0.0)
            + _safe_series(df, col_autofx).map(parse_decimal).fillna(0.0),
        }
    )

    result["description"] = np.where(
        result["description"].eq(""),
        result["product"],
        result["description"],
    )
    result["type"] = np.where(result["type"].eq(""), result["description"], result["type"])
    result["isin"] = result["isin"].mask(result["isin"].eq(""), np.nan)
    result["symbol"] = result["symbol"].mask(result["symbol"].eq(""), np.nan)
    result["instrument_id"] = np.where(result["isin"].notna(), result["isin"], result["product"])

    result["quantity"] = infer_quantity_sign(
        quantity=result["quantity"],
        total_eur=result["total_eur"],
        type_text=result["type"],
        description=result["description"],
    )

    return result.sort_values("datetime").reset_index(drop=True)


def infer_quantity_sign(
    *,
    quantity: pd.Series,
    total_eur: pd.Series,
    type_text: pd.Series,
    description: pd.Series,
) -> pd.Series:
    q = quantity.copy()
    if q.isna().all():
        return q
    desc = (
        type_text.fillna("").astype(str).str.lower()
        + " "
        + description.fillna("").astype(str).str.lower()
    )
    sell_hint = desc.str.contains(r"\b(?:sell|verkoop|sold)\b", regex=True) | (total_eur > 0)
    buy_hint = desc.str.contains(r"\b(?:buy|koop|gekocht)\b", regex=True) | (total_eur < 0)
    non_negative = q.notna() & (q >= 0)
    q = np.where(non_negative & sell_hint, -np.abs(q), q)
    q = np.where(non_negative & buy_hint, np.abs(q), q)
    return pd.Series(q, index=quantity.index, dtype="float64")


def normalize_portfolio(df: pd.DataFrame, *, account_label: str, warnings: list[str]) -> pd.DataFrame:
    col_product = _find_column(df, ["Product", "Instrument"])
    col_symbol_isin = _find_column(df, ["Symbol/ISIN", "Symbool/ISIN", "ISIN"])
    col_amount = _find_column(df, ["Amount", "Quantity", "Aantal"])
    col_price = _find_column(df, ["Closing", "Close", "Slotkoers", "Price"])
    col_value_eur = _find_column(df, ["Value in EUR", "Waarde in EUR", "Value EUR"])
    col_currency = _find_currency_column_near(df, col_price)

    if col_product is None or col_value_eur is None:
        raise UserFacingError(
            "Could not parse Portfolio.csv: required product/value columns not found.",
            "Check if Portfolio.csv is a standard DEGIRO export.",
        )

    symbol_or_isin = _safe_series(df, col_symbol_isin).fillna("").astype(str).str.strip()
    isin_mask = symbol_or_isin.str.match(r"^[A-Z]{2}[A-Z0-9]{10}$", na=False)

    result = pd.DataFrame(
        {
            "account_label": account_label,
            "product": _safe_series(df, col_product).fillna("").astype(str).str.strip(),
            "isin": np.where(isin_mask, symbol_or_isin, np.nan),
            "symbol": np.where(isin_mask, np.nan, symbol_or_isin.mask(symbol_or_isin.eq(""), np.nan)),
            "quantity": _safe_series(df, col_amount).map(parse_decimal),
            "price": _safe_series(df, col_price).map(parse_decimal),
            "value_eur": _safe_series(df, col_value_eur).map(parse_decimal),
            "currency": _safe_series(df, col_currency).fillna("").astype(str).str.upper().str.strip(),
        }
    )
    result["currency"] = result["currency"].mask(result["currency"].eq(""), np.nan)
    result["instrument_id"] = np.where(result["isin"].notna(), result["isin"], result["product"])
    result["is_cash_like"] = is_cash_like(result["product"], result["isin"])

    derivable = result["value_eur"].isna() & result["price"].notna() & result["quantity"].notna()
    result.loc[derivable, "value_eur"] = result.loc[derivable, "price"] * result.loc[derivable, "quantity"]

    missing_non_cash = result["value_eur"].isna() & (~result["is_cash_like"])
    if missing_non_cash.any():
        missing_rows = result.loc[missing_non_cash, ["product", "isin"]].fillna("")
        preview = "\n".join(
            f"- {row.product} ({row.isin or 'missing ISIN'})"
            for row in missing_rows.itertuples(index=False)
        )
        raise UserFacingError(
            "Portfolio snapshot has instruments without EUR value.",
            "Ensure Value in EUR is present in Portfolio.csv.\nAffected rows:\n" + preview,
        )

    return result.reset_index(drop=True)


def normalize_account(df: pd.DataFrame, *, account_label: str, warnings: list[str]) -> pd.DataFrame:
    col_date = _find_column(df, ["Date", "Datum"])
    col_time = _find_column(df, ["Time", "Tijd"])
    col_value_date = _find_column(df, ["Value date", "Waardedatum", "Value Date", "Valutadatum"])
    col_description = _find_column(df, ["Description", "Omschrijving"])
    col_type = _find_column(df, ["Type", "Soort"])
    col_change_raw = _find_column(df, ["Change", "Mutatie", "Mutatiebedrag"])
    col_balance_raw = _find_column(df, ["Balance", "Saldo"])
    col_fx = _find_column(df, ["FX", "Exchange rate", "Wisselkoers"])
    col_order_id = _find_column(df, ["Order Id", "Order ID", "OrderId"])
    col_change, col_change_currency = _resolve_amount_currency_columns(
        df=df,
        anchor_col=col_change_raw,
    )
    col_balance, col_balance_currency = _resolve_amount_currency_columns(
        df=df,
        anchor_col=col_balance_raw,
    )

    if col_date is None or col_change is None or col_balance is None:
        raise UserFacingError(
            "Could not parse Account.csv: required date/change/balance columns not found.",
            "Check if Account.csv is a standard DEGIRO export.",
        )

    datetime_values = parse_datetime_columns(df[col_date], df[col_time] if col_time else None)
    value_date_values = (
        pd.to_datetime(_safe_series(df, col_value_date), dayfirst=True, errors="coerce")
        if col_value_date is not None
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    raw_change = _safe_series(df, col_change).map(parse_decimal)
    raw_balance = _safe_series(df, col_balance).map(parse_decimal)
    fx_rate = _safe_series(df, col_fx).map(parse_decimal)
    order_id_series = (
        _safe_series(df, col_order_id).fillna("").astype(str).str.strip()
        if col_order_id is not None
        else pd.Series("", index=df.index, dtype="object")
    )

    change_non_empty = _safe_series(df, col_change).fillna("").astype(str).str.strip().ne("").sum()
    balance_non_empty = _safe_series(df, col_balance).fillna("").astype(str).str.strip().ne("").sum()
    if change_non_empty > 0 and int(raw_change.notna().sum()) == 0:
        raise UserFacingError(
            "Account.csv change amounts could not be parsed as numbers.",
            f"Detected `{col_change}` as change-amount column, but all non-empty values are non-numeric. "
            "Check for malformed rows or swapped CSV columns.",
        )
    if balance_non_empty > 0 and int(raw_balance.notna().sum()) == 0:
        raise UserFacingError(
            "Account.csv balance amounts could not be parsed as numbers.",
            f"Detected `{col_balance}` as balance-amount column, but all non-empty values are non-numeric. "
            "Check for malformed rows or swapped CSV columns.",
        )

    currency = _safe_series(df, col_change_currency).fillna("").astype(str).str.upper().str.strip()
    balance_currency = _safe_series(df, col_balance_currency).fillna("").astype(str).str.upper().str.strip()
    currency = np.where(currency == "", balance_currency, currency)
    currency_series = pd.Series(currency, index=df.index, dtype="object")

    description = _safe_series(df, col_description).fillna("").astype(str).str.strip()
    fx_rate = normalize_account_fx_rate_scale(
        fx_rate=fx_rate,
        order_id=order_id_series,
        description=description,
        currency=currency_series,
        raw_change=raw_change,
        datetime_values=datetime_values,
        warnings=warnings,
    )
    fx_rate = infer_fx_rate_from_order_rows(
        fx_rate=fx_rate,
        order_id=order_id_series,
        description=description,
        currency=currency_series,
        raw_change=raw_change,
        datetime_values=datetime_values,
        value_date=value_date_values,
    )
    inferred_type = infer_account_type(description=description, raw_change=raw_change)
    if col_type is None:
        type_values = inferred_type
    else:
        type_values = _safe_series(df, col_type).fillna("").astype(str).str.strip()
        type_values = np.where(pd.Series(type_values).eq(""), inferred_type, type_values)
        type_values = np.where(
            pd.Series(type_values).astype(str).str.lower().eq("other"),
            inferred_type,
            type_values,
        )

    type_series = pd.Series(type_values, index=df.index, dtype="object")
    is_internal_transfer = type_series.isin(
        {
            "internal_cash_sweep",
            "fx_conversion",
            "reservation_hold",
            "reservation_settlement",
        }
    )
    is_external_flow = type_series.isin({"external_deposit", "external_withdrawal"})
    is_cost = type_series.isin({"transaction_fee", "account_fee", "dividend_tax"})
    is_income = type_series.isin({"dividend", "interest", "promo"})

    change_eur = np.where(currency_series == "EUR", raw_change, np.nan)
    balance_eur = np.where(currency_series == "EUR", raw_balance, np.nan)
    with_fx = (currency_series != "EUR") & (fx_rate > 0)
    change_eur = np.where(with_fx, raw_change / fx_rate, change_eur)
    balance_eur = np.where(with_fx, raw_balance / fx_rate, balance_eur)

    result = pd.DataFrame(
        {
            "datetime": datetime_values,
            "account_label": account_label,
            "order_id": order_id_series.mask(order_id_series.eq(""), np.nan),
            "description": description,
            "type": type_values,
            "currency": currency_series.mask(currency_series.eq(""), np.nan),
            "raw_change": raw_change,
            "raw_balance": raw_balance,
            "fx_rate": fx_rate,
            "change_eur": pd.Series(change_eur, dtype="float64"),
            "balance_eur": pd.Series(balance_eur, dtype="float64"),
            "is_external_flow": pd.Series(is_external_flow, dtype="bool"),
            "is_internal_transfer": pd.Series(is_internal_transfer, dtype="bool"),
            "is_cost": pd.Series(is_cost, dtype="bool"),
            "is_income": pd.Series(is_income, dtype="bool"),
        }
    )

    return result.sort_values("datetime").reset_index(drop=True)


def normalize_account_fx_rate_scale(
    *,
    fx_rate: pd.Series,
    order_id: pd.Series,
    description: pd.Series,
    currency: pd.Series,
    raw_change: pd.Series,
    datetime_values: pd.Series,
    warnings: list[str],
) -> pd.Series:
    """
    Some DEGIRO exports encode FX as tenths (e.g. 11,738 instead of 1,1738).
    Detect that pattern from Valuta Debitering/Creditering amount consistency and
    normalize FX by /10 when strongly indicated.
    """
    fx_out = pd.to_numeric(fx_rate, errors="coerce").copy()
    desc = description.fillna("").astype(str).str.lower().str.strip()
    curr = currency.fillna("").astype(str).str.upper().str.strip()
    change = pd.to_numeric(raw_change, errors="coerce")
    oid = order_id.fillna("").astype(str).str.strip()
    dt = pd.to_datetime(datetime_values, errors="coerce")

    valuta_mask = desc.eq("valuta debitering") | desc.eq("valuta creditering")
    ratios: list[float] = []

    def _collect_ratio(non_eur_idx: Any, eur_idx: Any) -> None:
        non_eur_amount = abs(float(change.loc[non_eur_idx]))
        eur_amount = abs(float(change.loc[eur_idx]))
        fx_value = float(fx_out.loc[non_eur_idx])
        if eur_amount <= 0.0 or non_eur_amount <= 0.0 or fx_value <= 0.0:
            return
        implied = non_eur_amount / eur_amount
        if implied <= 0.0:
            return
        ratio = fx_value / implied
        if np.isfinite(ratio) and ratio > 0.0:
            ratios.append(float(ratio))

    non_empty_orders = oid[oid.ne("")].unique()
    for current_order in non_empty_orders:
        in_order = oid.eq(current_order) & valuta_mask
        eur_idx = fx_out.loc[in_order & curr.eq("EUR") & change.notna() & change.ne(0)].index
        non_eur_idx = fx_out.loc[
            in_order & curr.ne("EUR") & change.notna() & change.ne(0) & fx_out.notna() & fx_out.gt(0)
        ].index
        if len(eur_idx) > 0 and len(non_eur_idx) > 0:
            _collect_ratio(non_eur_idx[0], eur_idx[0])

    no_order_non_eur = fx_out.loc[
        oid.eq("") & valuta_mask & curr.ne("EUR") & change.notna() & change.ne(0) & fx_out.notna() & fx_out.gt(0)
    ].index
    for idx in no_order_non_eur:
        cur_dt = dt.loc[idx]
        if pd.isna(cur_dt):
            continue
        pair_idx = fx_out.loc[
            oid.eq("")
            & valuta_mask
            & curr.eq("EUR")
            & change.notna()
            & change.ne(0)
            & dt.eq(cur_dt)
        ].index
        if len(pair_idx) > 0:
            _collect_ratio(idx, pair_idx[0])

    if not ratios:
        return fx_out

    ratio_series = pd.Series(ratios, dtype="float64")
    near_ten_share = float(ratio_series.between(8.0, 12.0).mean())
    near_one_share = float(ratio_series.between(0.8, 1.2).mean())
    should_divide_by_ten = near_ten_share >= 0.6 and near_ten_share > near_one_share
    if not should_divide_by_ten:
        return fx_out

    adjust_mask = fx_out.gt(5.0)
    adjusted_count = int(adjust_mask.sum())
    if adjusted_count <= 0:
        return fx_out

    fx_out.loc[adjust_mask] = fx_out.loc[adjust_mask] / 10.0
    warnings.append(
        f"Account: normalized FX scale (/10) for {adjusted_count} row(s) based on valuta pair consistency."
    )
    return fx_out


def infer_fx_rate_from_order_rows(
    *,
    fx_rate: pd.Series,
    order_id: pd.Series,
    description: pd.Series,
    currency: pd.Series,
    raw_change: pd.Series,
    datetime_values: pd.Series | None = None,
    value_date: pd.Series | None = None,
) -> pd.Series:
    """
    DEGIRO often reports FX once per Order Id (typically on a Valuta Debitering/Creditering row).
    Propagate that FX rate to other non-EUR rows in the same order when missing.
    """
    fx_out = pd.to_numeric(fx_rate, errors="coerce").copy()
    oid = order_id.fillna("").astype(str).str.strip()

    desc = description.fillna("").astype(str).str.lower()
    curr = currency.fillna("").astype(str).str.upper().str.strip()
    change = pd.to_numeric(raw_change, errors="coerce")

    if oid.ne("").any():
        for current_order in oid[oid.ne("")].unique():
            in_order = oid.eq(current_order)
            fill_mask = in_order & fx_out.isna() & curr.ne("EUR")
            if not fill_mask.any():
                continue

            valid_rates = fx_out.loc[in_order]
            valid_rates = valid_rates[valid_rates > 0].dropna()
            inferred_rate = np.nan
            if not valid_rates.empty:
                inferred_rate = float(valid_rates.iloc[0])
            else:
                inferred_rate = _infer_order_fx_rate_from_amounts(
                    in_order=in_order,
                    description=desc,
                    currency=curr,
                    raw_change=change,
                )

            if np.isfinite(inferred_rate) and float(inferred_rate) > 0.0:
                fx_out.loc[fill_mask] = float(inferred_rate)

    fx_out = _infer_fx_rate_from_no_order_valuta_pairs(
        fx_rate=fx_out,
        order_id=oid,
        description=desc,
        currency=curr,
        raw_change=change,
        datetime_values=datetime_values,
        value_date=value_date,
    )

    return fx_out


def _infer_fx_rate_from_no_order_valuta_pairs(
    *,
    fx_rate: pd.Series,
    order_id: pd.Series,
    description: pd.Series,
    currency: pd.Series,
    raw_change: pd.Series,
    datetime_values: pd.Series | None,
    value_date: pd.Series | None,
) -> pd.Series:
    """
    Infer FX for no-order rows by pairing:
    - non-EUR 'Valuta Debitering' (with FX),
    - EUR 'Valuta Creditering' at the same timestamp,
    and then mapping the FX to matching non-EUR source rows (including dividends and cash-sweep rows)
    primarily by value date and otherwise by a tight datetime window around the FX event.
    """
    fx_out = pd.to_numeric(fx_rate, errors="coerce").copy()
    oid = order_id.fillna("").astype(str).str.strip()
    desc = description.fillna("").astype(str).str.lower().str.strip()
    curr = currency.fillna("").astype(str).str.upper().str.strip()
    change = pd.to_numeric(raw_change, errors="coerce")
    dt = pd.to_datetime(datetime_values, errors="coerce")
    vd = pd.to_datetime(value_date, errors="coerce").dt.normalize()

    no_order = oid.eq("")
    is_valuta_debit = (
        no_order
        & desc.eq("valuta debitering")
        & curr.ne("EUR")
        & change.notna()
        & change.ne(0)
        & fx_out.gt(0)
    )
    is_valuta_credit = (
        no_order
        & desc.eq("valuta creditering")
        & curr.eq("EUR")
        & change.notna()
        & change.ne(0)
    )
    candidate_source = (
        no_order
        & fx_out.isna()
        & curr.ne("EUR")
        & change.notna()
        & change.ne(0)
        & ~desc.eq("valuta debitering")
        & ~desc.eq("valuta creditering")
    )

    if not is_valuta_debit.any():
        return fx_out

    eur_tolerance = 0.05
    non_eur_tolerance = 0.03
    lookaround = pd.Timedelta(days=3)

    debit_indices = (
        pd.DataFrame({"idx": fx_out.index[is_valuta_debit], "datetime": dt.loc[is_valuta_debit].values})
        .sort_values(["datetime", "idx"], na_position="last")
        ["idx"]
        .tolist()
    )

    for debit_idx in debit_indices:
        rate = float(fx_out.loc[debit_idx])
        if not np.isfinite(rate) or rate <= 0.0:
            continue
        debit_change = float(change.loc[debit_idx])
        if not np.isfinite(debit_change) or debit_change == 0.0:
            continue

        target_non_eur = abs(debit_change)
        target_eur = target_non_eur / rate

        credit_mask = is_valuta_credit.copy()
        debit_dt = dt.loc[debit_idx]
        debit_vd = vd.loc[debit_idx]
        if pd.notna(debit_dt):
            credit_mask &= dt.eq(debit_dt)
        elif pd.notna(debit_vd):
            credit_mask &= vd.eq(debit_vd)
        else:
            continue
        if not credit_mask.any():
            continue

        eur_leg_delta = (change.loc[credit_mask].abs() - target_eur).abs()
        if eur_leg_delta.empty or float(eur_leg_delta.min()) > eur_tolerance:
            continue

        source_mask = candidate_source & curr.eq(curr.loc[debit_idx])
        if not source_mask.any():
            continue

        matched_indices: list[Any] = []
        if pd.notna(debit_vd):
            matched_indices = _match_unique_subset_to_target(
                changes=change,
                candidate_mask=source_mask & vd.eq(debit_vd),
                target_abs=target_non_eur,
                tolerance=non_eur_tolerance,
            )
        if not matched_indices and pd.notna(debit_dt):
            matched_indices = _match_unique_subset_to_target(
                changes=change,
                candidate_mask=source_mask & dt.notna() & dt.ge(debit_dt - lookaround) & dt.le(debit_dt + lookaround),
                target_abs=target_non_eur,
                tolerance=non_eur_tolerance,
            )

        if matched_indices:
            fx_out.loc[matched_indices] = rate
            candidate_source.loc[matched_indices] = False

    return fx_out


def _match_unique_subset_to_target(
    *,
    changes: pd.Series,
    candidate_mask: pd.Series,
    target_abs: float,
    tolerance: float,
    max_candidates: int = 20,
) -> list[Any]:
    if not np.isfinite(target_abs) or target_abs <= 0.0:
        return []

    candidate_values = changes.loc[candidate_mask].dropna()
    if candidate_values.empty:
        return []

    ordered = sorted(
        candidate_values.items(),
        key=lambda item: (abs(float(item[1])), item[0]),
        reverse=True,
    )
    if len(ordered) > max_candidates:
        ordered = ordered[:max_candidates]

    indices = [idx for idx, _ in ordered]
    values = [float(val) for _, val in ordered]
    n = len(values)
    if n == 0:
        return []

    matches: list[list[Any]] = []
    for mask in range(1, 1 << n):
        subset_sum = 0.0
        for pos, value in enumerate(values):
            if mask & (1 << pos):
                subset_sum += value
        if abs(abs(subset_sum) - target_abs) <= tolerance:
            matched = [indices[pos] for pos in range(n) if mask & (1 << pos)]
            matches.append(matched)
            if len(matches) > 1:
                return []

    return matches[0] if len(matches) == 1 else []


def _infer_order_fx_rate_from_amounts(
    *,
    in_order: pd.Series,
    description: pd.Series,
    currency: pd.Series,
    raw_change: pd.Series,
) -> float:
    valuta_mask = in_order & description.str.contains(r"valuta (?:debitering|creditering)", regex=True)
    eur_mask = valuta_mask & currency.eq("EUR") & raw_change.notna() & raw_change.ne(0)
    non_eur_mask = valuta_mask & currency.ne("EUR") & raw_change.notna() & raw_change.ne(0)
    if eur_mask.any() and non_eur_mask.any():
        eur_amount = abs(float(raw_change.loc[eur_mask].iloc[0]))
        non_eur_amount = abs(float(raw_change.loc[non_eur_mask].iloc[0]))
        if eur_amount > 0.0 and non_eur_amount > 0.0:
            return non_eur_amount / eur_amount

    trade_mask = in_order & description.str.contains(r"\b(?:koop|verkoop|buy|sell)\b", regex=True)
    trade_mask = trade_mask & currency.ne("EUR") & raw_change.notna() & raw_change.ne(0)
    if eur_mask.any() and trade_mask.any():
        eur_amount = abs(float(raw_change.loc[eur_mask].iloc[0]))
        trade_amount = abs(float(raw_change.loc[trade_mask].iloc[0]))
        if eur_amount > 0.0 and trade_amount > 0.0:
            return trade_amount / eur_amount
    return float("nan")


def infer_account_type(description: pd.Series, raw_change: pd.Series | None = None) -> pd.Series:
    txt = description.fillna("").astype(str).str.lower()
    change = raw_change if raw_change is not None else pd.Series(np.nan, index=description.index)
    out = pd.Series("other", index=description.index, dtype="object")

    internal_sweep = txt.str.contains("degiro cash sweep", regex=False)
    internal_overboeking = txt.str.contains(
        r"overboeking (?:van|naar) uw geldrekening bij flatexdegiro bank"
    )
    out = np.where(internal_sweep | internal_overboeking, "internal_cash_sweep", out)
    out = np.where(txt.str.contains(r"(?:valuta debitering|valuta creditering)"), "fx_conversion", out)

    out = np.where(
        txt.str.contains(r"(?:degiro transactiekosten|kosten van derden)"),
        "transaction_fee",
        out,
    )
    out = np.where(
        txt.str.contains(r"(?:aansluitingskosten|minimale activiteitskosten pensioenrekening)"),
        "account_fee",
        out,
    )
    out = np.where(txt.str.contains("dividendbelasting"), "dividend_tax", out)
    out = np.where(txt.str.contains("dividend") & ~txt.str.contains("dividendbelasting"), "dividend", out)
    out = np.where(txt.str.contains(r"(?:flatex interest income|interest income)"), "interest", out)
    out = np.where(txt.str.contains("verrekening promotie"), "promo", out)

    reservation = txt.str.contains(
        r"(?:reservation ideal|reservation sofort|reservation i?deal / sofort deposit)"
    )
    out = np.where(reservation & (change > 0), "external_deposit", out)
    out = np.where(reservation & (change < 0), "reservation_hold", out)
    out = np.where(
        txt.str.contains("flatex storting", regex=False) & ~reservation,
        "external_deposit",
        out,
    )
    out = np.where(
        txt.str.contains(r"(?:ideal deposit|sofort deposit)") & ~reservation,
        "reservation_settlement",
        out,
    )
    out = np.where(
        txt.str.contains(r"(?:flatex terugstorting|withdraw|opname|uitbetaling)"),
        "external_withdrawal",
        out,
    )
    out = np.where(
        txt.str.contains(r"\b(?:koop|verkoop|buy|sell)\b"),
        "trade_settlement",
        out,
    )
    return pd.Series(out, index=description.index)


def load_mappings(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {"symbols": {}, "currencies": {}, "is_etf": {}, "is_not_etf": {}}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except YAMLError as exc:
        raise UserFacingError(
            "mappings.yml has invalid YAML syntax.",
            f"Fix YAML syntax in mappings.yml. Parser error: {exc}",
        ) from exc

    if not isinstance(data, dict):
        raise UserFacingError(
            "mappings.yml must contain a top-level mapping (dictionary).",
            "Ensure mappings.yml starts with sections like `symbols:`, `currencies:`, `is_etf:`, `is_not_etf:`.",
        )

    required_sections = ["symbols", "currencies", "is_etf", "is_not_etf"]
    for section in required_sections:
        value = data.get(section, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise UserFacingError(
                f"mappings.yml section `{section}` has invalid structure.",
                f"Section `{section}` must be a key-value mapping, not {type(value).__name__}.",
            )

    _validate_boolean_section(data.get("is_etf", {}), "is_etf")
    _validate_boolean_section(data.get("is_not_etf", {}), "is_not_etf")

    return {
        "symbols": _to_upper_key_dict(data.get("symbols", {})),
        "currencies": _to_upper_key_dict(data.get("currencies", {})),
        "is_etf": _to_upper_key_dict(data.get("is_etf", {})),
        "is_not_etf": _to_upper_key_dict(data.get("is_not_etf", {})),
    }


def resolve_instrument_mapping(
    *,
    portfolio: pd.DataFrame,
    transactions: pd.DataFrame,
    mappings: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    base_cols = ["instrument_id", "product", "isin", "symbol", "currency"]
    from_pf = portfolio[base_cols].copy()
    from_tx = transactions[base_cols].copy()
    merged = (
        pd.concat([from_pf, from_tx], ignore_index=True)
        .sort_values(["instrument_id", "product"])
        .drop_duplicates(subset=["instrument_id"], keep="first")
        .reset_index(drop=True)
    )

    merged["is_cash_like"] = is_cash_like(merged["product"], merged["isin"])
    merged["ticker"] = merged["symbol"].where(~merged["symbol"].astype(str).str.match(r"^[A-Z]{2}[A-Z0-9]{10}$"))

    symbol_map = mappings.get("symbols", {})
    currency_map = mappings.get("currencies", {})
    is_etf_map = mappings.get("is_etf", {})
    is_not_etf_map = mappings.get("is_not_etf", {})

    for idx in merged.index:
        isin = str(merged.at[idx, "isin"]).upper() if pd.notna(merged.at[idx, "isin"]) else ""
        product_key = str(merged.at[idx, "product"]).upper()
        if pd.isna(merged.at[idx, "ticker"]) or str(merged.at[idx, "ticker"]).strip() == "":
            mapped = symbol_map.get(isin) or symbol_map.get(product_key)
            if mapped:
                merged.at[idx, "ticker"] = str(mapped).strip()

        if pd.isna(merged.at[idx, "currency"]) or str(merged.at[idx, "currency"]).strip() == "":
            mapped_cur = currency_map.get(isin) or currency_map.get(product_key)
            if mapped_cur:
                merged.at[idx, "currency"] = str(mapped_cur).strip().upper()

    heuristic_etf = merged["product"].fillna("").str.contains(
        r"(?:ETF|UCITS|ISHARES|VANGUARD|SPDR|VANECK)",
        case=False,
        regex=True,
    ) | merged["isin"].fillna("").isin(DEFAULT_ETF_ISINS)
    merged["is_etf"] = heuristic_etf
    merged["is_not_etf"] = False
    missing_classification: list[str] = []
    for idx in merged.index:
        isin = str(merged.at[idx, "isin"]).upper() if pd.notna(merged.at[idx, "isin"]) else ""
        product_key = str(merged.at[idx, "product"]).upper()
        mapped_etf_value = is_etf_map.get(isin, is_etf_map.get(product_key))
        mapped_not_etf_value = is_not_etf_map.get(isin, is_not_etf_map.get(product_key))

        has_etf = mapped_etf_value is not None
        has_not_etf = mapped_not_etf_value is not None
        if has_etf and has_not_etf and _as_bool(mapped_etf_value) and _as_bool(mapped_not_etf_value):
            raise UserFacingError(
                "Conflicting ETF classification in mappings.yml.",
                f"Instrument {merged.at[idx, 'product']} ({isin or product_key}) is marked as both "
                "`is_etf` and `is_not_etf`. Keep only one explicit classification.",
            )
        if has_etf:
            merged.at[idx, "is_etf"] = _as_bool(mapped_etf_value)
        if has_not_etf:
            merged.at[idx, "is_etf"] = not _as_bool(mapped_not_etf_value)

        if (not has_etf) and (not has_not_etf) and (not bool(merged.at[idx, "is_cash_like"])):
            identifier = isin or product_key
            product_name = str(merged.at[idx, "product"]).strip()
            missing_classification.append(
                f"- Product: {product_name} | Identifier: {identifier} | "
                "Add either `is_etf: true` or `is_not_etf: true` in mappings.yml."
            )

        merged.at[idx, "is_not_etf"] = not bool(merged.at[idx, "is_etf"])

    # Cash-like rows should not be treated as ETF/non-ETF holdings.
    merged.loc[merged["is_cash_like"], "is_etf"] = False
    merged.loc[merged["is_cash_like"], "is_not_etf"] = False

    if missing_classification:
        raise UserFacingError(
            "ETF classification is incomplete for one or more instruments.",
            "Every non-cash holding must be explicitly classified in mappings.yml under "
            "`is_etf` or `is_not_etf`.\n" + "\n".join(missing_classification),
        )

    ticker_blank = merged["ticker"].isna() | merged["ticker"].astype(str).str.strip().isin({"", "nan", "None"})
    unresolved = merged[(~merged["is_cash_like"]) & ticker_blank]
    if not unresolved.empty:
        lines = []
        for row in unresolved.itertuples(index=False):
            identifier = row.isin if pd.notna(row.isin) and str(row.isin) else row.product
            product_name = str(row.product).strip() if pd.notna(row.product) else "(missing product name)"
            lines.append(
                f"- Product: {product_name} | Identifier: {identifier} | "
                f"Yahoo lookup: https://finance.yahoo.com/ (search '{product_name}')"
            )
        raise UserFacingError(
            "Ticker mapping is incomplete for one or more instruments.",
            "Edit mappings.yml and add entries under `symbols`.\n"
            "Use the product names below to search Yahoo Finance for the correct ticker.\n"
            + "\n".join(lines),
        )

    return merged


def attach_instrument_metadata(df: pd.DataFrame, instruments: pd.DataFrame) -> pd.DataFrame:
    cols = ["instrument_id", "ticker", "currency", "is_etf", "is_not_etf", "is_cash_like"]
    merged = df.merge(instruments[cols], on="instrument_id", how="left", suffixes=("", "_instrument"))
    if "currency_instrument" in merged.columns:
        merged["currency"] = merged["currency"].fillna(merged["currency_instrument"])
        merged = merged.drop(columns=["currency_instrument"])
    return merged


def is_cash_like(product: pd.Series, isin: pd.Series) -> pd.Series:
    p = product.fillna("").astype(str).str.upper()
    i = isin.fillna("").astype(str).str.upper()
    cash_like_product_pattern = r"(?:CASH|CASH SWEEP|CASH FUND|MONEY MARKET|GELDREKENING|BANKACCOUNT|FLATEX)"
    by_product = p.str.contains(cash_like_product_pattern, regex=True)
    by_known_cash_isin = i.isin({"NLFLATEXACNT"})
    missing_isin = i.eq("")
    currency_only = p.str.fullmatch(r"(?:EUR|USD|GBP|CHF|JPY|AUD|CAD)(?:\s+CASH)?", na=False)
    missing_isin_cash_like = missing_isin & (
        p.str.contains(r"(?:CASH|GELD|MONEY MARKET)", regex=True) | currency_only
    )
    return by_product | by_known_cash_isin | missing_isin_cash_like


def validate_critical_columns(
    *,
    transactions: pd.DataFrame,
    portfolio: pd.DataFrame,
    account: pd.DataFrame,
) -> tuple[list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    issues: list[dict[str, Any]] = []

    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Transactions: invalid datetime rows (ignored in time-series)",
        df=transactions,
        mask=transactions["datetime"].isna(),
        columns=["product", "isin", "quantity", "total_eur", "description"],
    )
    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Account: invalid datetime rows",
        df=account,
        mask=account["datetime"].isna(),
        columns=["description", "type", "currency", "raw_change", "raw_balance"],
    )
    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Transactions: NaN in total_eur",
        df=transactions,
        mask=transactions["total_eur"].isna(),
        columns=["datetime", "product", "isin", "quantity", "price", "total_eur"],
    )
    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Portfolio: NaN in value_eur",
        df=portfolio,
        mask=portfolio["value_eur"].isna(),
        columns=["product", "isin", "quantity", "price", "currency", "is_cash_like"],
    )
    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Account: NaN in balance_eur (FX missing in CSV; fallback may apply later)",
        df=account,
        mask=account["balance_eur"].isna() & account["raw_balance"].notna(),
        columns=["datetime", "description", "currency", "raw_balance", "fx_rate", "balance_eur"],
    )
    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Account: NaN in change_eur (FX missing in CSV; fallback may apply later)",
        df=account,
        mask=account["change_eur"].isna() & account["raw_change"].notna(),
        columns=["datetime", "description", "currency", "raw_change", "fx_rate", "change_eur"],
    )
    _append_row_issue_warning(
        warnings=warnings,
        issues=issues,
        label="Transactions: NaN in quantity",
        df=transactions,
        mask=transactions["quantity"].isna(),
        columns=["datetime", "product", "isin", "quantity", "total_eur", "type"],
    )
    return warnings, issues


def _to_upper_key_dict(mapping: dict[Any, Any]) -> dict[str, Any]:
    return {str(k).upper(): v for k, v in mapping.items()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(value)


def _validate_boolean_section(section: dict[Any, Any], section_name: str) -> None:
    for raw_key, raw_value in section.items():
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, (int, float)) and raw_value in {0, 1}:
            continue
        if isinstance(raw_value, str) and raw_value.strip().lower() in {
            "true",
            "false",
            "yes",
            "no",
            "1",
            "0",
        }:
            continue
        raise UserFacingError(
            f"mappings.yml section `{section_name}` has invalid boolean value.",
            f"Key `{raw_key}` has value `{raw_value}`. Use true/false.",
        )


def _append_row_issue_warning(
    *,
    warnings: list[str],
    issues: list[dict[str, Any]],
    label: str,
    df: pd.DataFrame,
    mask: pd.Series,
    columns: list[str],
    max_examples: int = 3,
) -> None:
    issue_count = int(mask.fillna(False).sum())
    if issue_count <= 0:
        return
    cols = [c for c in columns if c in df.columns]
    sample_df = _format_preview_df(df.loc[mask, cols], max_rows=max_examples)
    warnings.append(f"{label}: {issue_count} row(s).")
    issues.append(
        {
            "label": label,
            "count": issue_count,
            "examples": sample_df,
        }
    )


def _format_preview_df(df: pd.DataFrame, max_rows: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sample = df.head(max_rows).copy()
    sample.insert(0, "row_number", [_row_number_value(idx) for idx in sample.index])
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return sample.reset_index(drop=True)


def _row_number_value(index_value: Any) -> Any:
    if isinstance(index_value, (int, np.integer)):
        return int(index_value) + 1
    return str(index_value)


def _safe_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype="object")
    return df[column]
