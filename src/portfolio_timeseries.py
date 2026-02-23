"""
AGENT_NOTE: Time-series engine for positions, prices, and portfolio metrics.

Interdependencies:
- Consumes normalized transactions/account/instruments from `data_import`.
- `TimeSeriesResult` is consumed by `src/app.py`, `src/insights.py`, and
  `src/strategy_check.py` (optional correlation checks).
- Uses Yahoo/cache utilities that also influence offline-mode UI messaging.

When editing:
- Keep `TimeSeriesResult` fields stable or update all consumers together.
- Preserve offline/cache runtime state keys used by `app.py`.
- See `src/INTERDEPENDENCIES.md` for the shared contract map.
"""

from __future__ import annotations

import json
import logging
import re
import socket
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import UserFacingError

try:
    import yfinance as yf
except Exception:  # pragma: no cover - tested through runtime behavior
    yf = None


_CACHE_RUNTIME_STATE: dict[str, Any] = {
    "offline_mode": False,
    "offline_cached_from": None,
    "offline_tickers": set(),
}


@dataclass
class TimeSeriesResult:
    metrics: pd.DataFrame
    positions: pd.DataFrame
    prices_eur: pd.DataFrame
    fill_stats: pd.DataFrame
    warnings: list[str]
    issues: list[dict[str, object]]


def reset_cache_runtime_state() -> None:
    _CACHE_RUNTIME_STATE["offline_mode"] = False
    _CACHE_RUNTIME_STATE["offline_cached_from"] = None
    _CACHE_RUNTIME_STATE["offline_tickers"] = set()


def prime_cache_runtime_state(
    *,
    cache_dir: str = "cache",
    logger: logging.Logger | None = None,
) -> None:
    reset_cache_runtime_state()
    if _is_yahoo_reachable():
        return
    _CACHE_RUNTIME_STATE["offline_mode"] = True
    _CACHE_RUNTIME_STATE["offline_cached_from"] = _read_last_online_fetch_timestamp(cache_dir=cache_dir)
    if logger:
        logger.warning("Yahoo Finance appears unreachable; running in offline cache mode.")


def get_cache_runtime_state(*, cache_dir: str = "cache") -> dict[str, Any]:
    state = {
        "offline_mode": bool(_CACHE_RUNTIME_STATE.get("offline_mode", False)),
        "offline_cached_from": _CACHE_RUNTIME_STATE.get("offline_cached_from"),
        "offline_tickers": sorted(_CACHE_RUNTIME_STATE.get("offline_tickers", set())),
    }
    if state["offline_mode"] and state["offline_cached_from"] is None:
        state["offline_cached_from"] = _read_last_online_fetch_timestamp(cache_dir=cache_dir)
    return state


def _is_yahoo_reachable(timeout_seconds: float = 1.5) -> bool:
    if yf is None:
        return False
    try:
        with socket.create_connection(("query1.finance.yahoo.com", 443), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def latest_fx_rate(currency: str, cache_dir: str = "cache", logger: logging.Logger | None = None) -> float | None:
    currency = currency.upper()
    if currency == "EUR":
        return 1.0
    end = pd.Timestamp(date.today())
    start = end - pd.Timedelta(days=14)
    try:
        fx = fetch_fx_series(
            currency=currency,
            start=start,
            end=end,
            cache_dir=cache_dir,
            logger=logger,
        )
    except UserFacingError:
        return None
    fx = fx.dropna()
    if fx.empty:
        return None
    return float(fx.iloc[-1])


def compute_portfolio_timeseries(
    *,
    transactions: pd.DataFrame,
    account: pd.DataFrame,
    instruments: pd.DataFrame,
    expected_latest_cash_eur: float | None = None,
    cache_dir: str = "cache",
    logger: logging.Logger | None = None,
) -> TimeSeriesResult:
    warnings: list[str] = []
    issues: list[dict[str, object]] = []
    if transactions.empty:
        raise UserFacingError(
            "No transactions available for time-series processing.",
            "Load at least one dataset with Transactions.csv before running analysis.",
        )

    start_date = transactions["datetime"].dropna().min().normalize()
    if pd.isna(start_date):
        raise UserFacingError(
            "Transactions contain no valid datetime values.",
            "Verify the Date and Time columns in Transactions.csv.",
        )

    end_date = max(
        transactions["datetime"].dropna().max().normalize(),
        account["datetime"].dropna().max().normalize() if not account.empty else start_date,
    )
    daily_index = pd.date_range(start=start_date, end=end_date, freq="D")

    positions = build_daily_positions(
        transactions=transactions,
        instruments=instruments,
        daily_index=daily_index,
    )
    prices_eur, fill_stats, price_warnings, price_issues = build_daily_prices_eur(
        instruments=instruments,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    warnings.extend(price_warnings)
    issues.extend(price_issues)

    shared_cols = [c for c in positions.columns if c in prices_eur.columns]
    if not shared_cols:
        raise UserFacingError(
            "No instruments could be priced from the configured mappings.",
            "Add missing ticker mappings in mappings.yml under `symbols`.",
        )

    positions = positions[shared_cols]
    prices_eur = prices_eur[shared_cols]
    positions_value = (positions * prices_eur).sum(axis=1).rename("positions_value")

    cash_from_statement, cash_warnings, cash_issues = build_daily_cash_series(
        account_df=account,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    warnings.extend(cash_warnings)
    issues.extend(cash_issues)

    cash_from_changes, cash_changes_warnings, cash_changes_issues = build_daily_cash_series_from_changes(
        account_df=account,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    warnings.extend(cash_changes_warnings)
    issues.extend(cash_changes_issues)

    deposits, deposit_warnings, deposit_issues = build_daily_external_deposits(
        account_df=account,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    warnings.extend(deposit_warnings)
    issues.extend(deposit_issues)

    # Primary cash series for timeline analytics: reconstructed from account changes.
    cash_series = cash_from_changes
    portfolio_value = positions_value + cash_series
    profit = portfolio_value - deposits
    simple_return = pd.Series(
        np.where(deposits != 0.0, portfolio_value / deposits - 1.0, np.nan),
        index=daily_index,
        name="simple_return",
    )

    metrics = pd.concat(
        [
            positions_value.rename("positions_value"),
            cash_series.rename("cash"),
            cash_from_changes.rename("cash_from_changes"),
            cash_from_statement.rename("cash_from_statement"),
            profit.rename("profit"),
            deposits.rename("total_deposits"),
            portfolio_value.rename("portfolio_value"),
            simple_return.rename("simple_return"),
        ],
        axis=1,
    )

    cash_gap = (cash_from_statement - cash_from_changes).abs()
    if not cash_gap.empty and float(cash_gap.max()) > 5.0:
        sample = pd.DataFrame(
            {
                "date": cash_gap.index.strftime("%Y-%m-%d"),
                "cash_statement_eur": cash_from_statement.values,
                "cash_from_changes_eur": cash_from_changes.values,
                "abs_diff_eur": cash_gap.values,
            }
        )
        sample = sample[sample["abs_diff_eur"] > 5.0].head(3)
        warnings.append(
            "Cash reconstruction mismatch: statement-based cash differs from change-cumsum cash."
        )
        issues.append(
            {
                "label": "Cash statement vs change-cumsum mismatch",
                "count": int((cash_gap > 5.0).sum()),
                "examples": sample.reset_index(drop=True),
            }
        )

    critical_cols = ["positions_value", "cash", "portfolio_value"]
    if metrics[critical_cols].isna().any().any():
        for col in critical_cols:
            mask = metrics[col].isna()
            if mask.any():
                examples = metrics.loc[mask, [col]].head(3).copy()
                examples.insert(
                    0,
                    "row_number",
                    [int(metrics.index.get_loc(idx)) + 1 for idx in examples.index],
                )
                examples.insert(1, "date", examples.index.strftime("%Y-%m-%d"))
                warnings.append(
                    f"Time-series: NaN values in `{col}` = {int(mask.sum())} row(s). "
                    f"Example dates: {examples['date'].tolist()}"
                )
                issues.append(
                    {
                        "label": f"Time-series NaN in {col}",
                        "count": int(mask.sum()),
                        "examples": examples.reset_index(drop=True),
                    }
                )

    return TimeSeriesResult(
        metrics=metrics,
        positions=positions,
        prices_eur=prices_eur,
        fill_stats=fill_stats,
        warnings=warnings,
        issues=issues,
    )


def build_daily_positions(
    *,
    transactions: pd.DataFrame,
    instruments: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    tx = transactions.copy()
    tx = tx[tx["datetime"].notna()].copy()
    tx["date"] = tx["datetime"].dt.normalize()
    tx = tx[~tx["is_cash_like"].fillna(False)]

    valid_ids = set(instruments.loc[~instruments["is_cash_like"], "instrument_id"])
    tx = tx[tx["instrument_id"].isin(valid_ids)]
    tx["quantity"] = tx["quantity"].fillna(0.0)

    daily_delta = tx.pivot_table(
        index="date",
        columns="instrument_id",
        values="quantity",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    positions = daily_delta.cumsum()
    positions = positions.reindex(daily_index).ffill().fillna(0.0)
    return positions


def build_daily_prices_eur(
    *,
    instruments: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    cache_dir: str,
    logger: logging.Logger | None,
 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, object]]]:
    warnings: list[str] = []
    issues: list[dict[str, object]] = []
    prices: dict[str, pd.Series] = {}
    fill_rows: list[dict[str, object]] = []

    non_cash = instruments[~instruments["is_cash_like"].fillna(False)].copy()
    if non_cash.empty:
        return pd.DataFrame(index=daily_index), pd.DataFrame(), warnings, issues

    start, end = daily_index.min(), daily_index.max()
    for row in non_cash.itertuples(index=False):
        if pd.isna(row.ticker) or str(row.ticker).strip() == "":
            warnings.append(f"Missing ticker for instrument: {row.product}")
            issues.append(
                {
                    "label": "Instrument missing ticker",
                    "count": 1,
                    "examples": pd.DataFrame(
                        [{"instrument_id": row.instrument_id, "product": row.product, "ticker": row.ticker}]
                    ),
                }
            )
            continue

        ticker = str(row.ticker).strip()
        currency = str(row.currency).strip().upper() if pd.notna(row.currency) else "EUR"

        try:
            local = fetch_price_series(
                ticker=ticker,
                start=start,
                end=end,
                cache_dir=cache_dir,
                logger=logger,
            )
        except UserFacingError as exc:
            warnings.append(str(exc))
            continue

        local = local.reindex(daily_index)
        original_missing = int(local.isna().sum())

        if currency != "EUR":
            try:
                fx = fetch_fx_series(
                    currency=currency,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    logger=logger,
                ).reindex(daily_index)
            except UserFacingError as exc:
                warnings.append(str(exc))
                continue
            series_eur = local * fx
        else:
            series_eur = local

        filled = series_eur.ffill().bfill()
        fills_applied = int((series_eur.isna() & filled.notna()).sum())
        remaining_na = int(filled.isna().sum())
        if remaining_na > 0:
            warnings.append(f"Price series still has {remaining_na} missing values after fill: {ticker}")
            missing_dates = (
                pd.DataFrame({"date": filled.index[filled.isna()]})
                .head(3)
                .assign(ticker=ticker)
            )
            issues.append(
                {
                    "label": f"Price series remaining NaN after fill ({ticker})",
                    "count": remaining_na,
                    "examples": missing_dates,
                }
            )

        prices[str(row.instrument_id)] = filled
        fill_rows.append(
            {
                "instrument_id": row.instrument_id,
                "ticker": ticker,
                "currency": currency,
                "missing_before_fill": original_missing,
                "fills_applied": fills_applied,
                "remaining_missing": remaining_na,
            }
        )
        if logger:
            logger.info(
                "Price fill stats | %s | missing_before=%d fills_applied=%d remaining=%d",
                ticker,
                original_missing,
                fills_applied,
                remaining_na,
            )

    return pd.DataFrame(prices, index=daily_index), pd.DataFrame(fill_rows), warnings, issues


def build_daily_cash_series(
    *,
    account_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    cache_dir: str,
    logger: logging.Logger | None,
) -> tuple[pd.Series, list[str], list[dict[str, object]]]:
    warnings: list[str] = []
    issues: list[dict[str, object]] = []
    if account_df.empty:
        return pd.Series(0.0, index=daily_index), warnings, issues

    df = account_df.copy().sort_values("datetime")
    df = df[df["datetime"].notna()].copy()
    df["date"] = df["datetime"].dt.normalize()
    if df.empty:
        return pd.Series(0.0, index=daily_index), warnings, issues

    start, end = daily_index.min(), daily_index.max()
    df["balance_eur_filled"] = df["balance_eur"]
    missing = df["balance_eur_filled"].isna() & df["raw_balance"].notna()
    currencies = sorted(set(df.loc[missing, "currency"].dropna().astype(str).str.upper()))
    for currency in currencies:
        if currency == "EUR":
            df.loc[missing & df["currency"].eq("EUR"), "balance_eur_filled"] = df.loc[
                missing & df["currency"].eq("EUR"), "raw_balance"
            ]
            continue
        try:
            fx = fetch_fx_series(
                currency=currency,
                start=start,
                end=end,
                cache_dir=cache_dir,
                logger=logger,
            ).reindex(daily_index).ffill().bfill()
        except UserFacingError:
            mask = missing & df["currency"].astype(str).str.upper().eq(currency)
            warnings.append(
                f"Account cash conversion: missing FX series for currency {currency}. "
                f"Affected rows: {int(mask.sum())}."
            )
            issues.append(
                {
                    "label": f"Account cash conversion missing FX ({currency})",
                    "count": int(mask.sum()),
                    "examples": _format_preview_df(
                        df.loc[mask, ["datetime", "description", "currency", "raw_balance", "fx_rate"]]
                    ),
                }
            )
            continue

        mask = missing & df["currency"].astype(str).str.upper().eq(currency)
        rates = df.loc[mask, "date"].map(fx)
        df.loc[mask, "balance_eur_filled"] = df.loc[mask, "raw_balance"].values * rates.values

    remaining_mask = df["balance_eur_filled"].isna() & df["raw_balance"].notna()
    if remaining_mask.any():
        warnings.append(
            f"Account balance EUR conversion incomplete: {int(remaining_mask.sum())} row(s)."
        )
        issues.append(
            {
                "label": "Account balance EUR conversion incomplete",
                "count": int(remaining_mask.sum()),
                "examples": _format_preview_df(
                    df.loc[
                        remaining_mask,
                        ["datetime", "description", "currency", "raw_balance", "fx_rate", "balance_eur_filled"],
                    ]
                ),
            }
        )

    eod = (
        df.sort_values("datetime")
        .groupby(["date", "currency"], as_index=False)
        .tail(1)[["date", "currency", "balance_eur_filled"]]
    )
    cash = eod.groupby("date")["balance_eur_filled"].sum()
    cash = cash.reindex(daily_index).ffill().fillna(0.0)
    return cash.astype(float), warnings, issues


def build_daily_external_deposits(
    *,
    account_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    cache_dir: str,
    logger: logging.Logger | None,
) -> tuple[pd.Series, list[str], list[dict[str, object]]]:
    df, warnings, issues = _fill_change_eur_with_fx(
        account_df=account_df,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    if df.empty:
        return pd.Series(0.0, index=daily_index), warnings, issues

    if "is_external_flow" in df.columns:
        external_mask = df["is_external_flow"].fillna(False).astype(bool)
    else:
        external_mask = df["type"].astype(str).isin({"external_deposit", "external_withdrawal"})
    missing_external_eur = external_mask & df["change_eur_filled"].isna() & df["raw_change"].notna()
    if missing_external_eur.any():
        warnings.append(
            f"External deposits/wd rows with missing EUR value: {int(missing_external_eur.sum())} row(s)."
        )
        issues.append(
            {
                "label": "External deposits/wd rows with missing EUR value",
                "count": int(missing_external_eur.sum()),
                "examples": _format_preview_df(
                    df.loc[
                        missing_external_eur,
                        ["datetime", "description", "currency", "raw_change", "fx_rate", "change_eur_filled"],
                    ]
                ),
            }
        )

    daily_external = df.loc[external_mask].groupby("date")["change_eur_filled"].sum().sort_index()
    cumulative = daily_external.cumsum()
    if cumulative.empty:
        return pd.Series(0.0, index=daily_index), warnings, issues

    start_date = daily_index.min()
    baseline_before_start = cumulative.loc[cumulative.index < start_date]
    baseline = float(baseline_before_start.iloc[-1]) if not baseline_before_start.empty else 0.0
    cumulative_in_range = cumulative.loc[cumulative.index >= start_date]
    out = cumulative_in_range.reindex(daily_index).ffill().fillna(baseline)
    return out.astype(float), warnings, issues


def build_daily_cash_series_from_changes(
    *,
    account_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    cache_dir: str,
    logger: logging.Logger | None,
) -> tuple[pd.Series, list[str], list[dict[str, object]]]:
    df, warnings, issues = _fill_change_eur_with_fx(
        account_df=account_df,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    if df.empty:
        return pd.Series(0.0, index=daily_index), warnings, issues

    if "is_internal_transfer" in df.columns:
        effective_mask = ~df["is_internal_transfer"].fillna(False).astype(bool)
    else:
        effective_mask = ~df["type"].astype(str).isin({"internal_cash_sweep", "reservation_hold", "reservation_settlement"})

    dropped_internal = int((~effective_mask).sum())
    if dropped_internal > 0:
        warnings.append(
            f"Cash reconstruction: excluded {dropped_internal} internal transfer row(s) from change cumsum."
        )
        issues.append(
            {
                "label": "Cash reconstruction excluded internal transfers",
                "count": dropped_internal,
                "examples": _format_preview_df(
                    df.loc[
                        ~effective_mask,
                        ["datetime", "description", "type", "currency", "raw_change", "change_eur_filled"],
                    ],
                ),
            }
        )

    daily_change = df.loc[effective_mask].groupby("date")["change_eur_filled"].sum().sort_index()
    cumulative = daily_change.cumsum()
    out = cumulative.reindex(daily_index).ffill().fillna(0.0)
    return out.astype(float), warnings, issues


def summarize_account_categories(
    *,
    account_df: pd.DataFrame,
    cache_dir: str,
    logger: logging.Logger | None,
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    if account_df is None or account_df.empty:
        return pd.DataFrame(), [], []

    date_min = account_df["datetime"].dropna().min()
    date_max = account_df["datetime"].dropna().max()
    if pd.isna(date_min) or pd.isna(date_max):
        return pd.DataFrame(), [], []
    daily_index = pd.date_range(start=date_min.normalize(), end=date_max.normalize(), freq="D")
    df, warnings, issues = _fill_change_eur_with_fx(
        account_df=account_df,
        daily_index=daily_index,
        cache_dir=cache_dir,
        logger=logger,
    )
    if df.empty:
        return pd.DataFrame(), warnings, issues

    df = df.copy()
    df["category"] = df.get("type", pd.Series("other", index=df.index)).fillna("other").astype(str)
    df["change_eur_filled"] = df["change_eur_filled"].fillna(0.0)
    df["inflow_eur"] = np.where(df["change_eur_filled"] > 0, df["change_eur_filled"], 0.0)
    df["outflow_eur"] = np.where(df["change_eur_filled"] < 0, -df["change_eur_filled"], 0.0)

    grouped = (
        df.groupby("category", dropna=False)
        .agg(
            row_count=("category", "size"),
            net_eur=("change_eur_filled", "sum"),
            inflow_eur=("inflow_eur", "sum"),
            outflow_eur=("outflow_eur", "sum"),
        )
        .reset_index()
        .sort_values("outflow_eur", ascending=False)
    )

    if "is_internal_transfer" in df.columns:
        internal_map = df.groupby("category")["is_internal_transfer"].any()
        grouped["is_internal_transfer"] = grouped["category"].map(internal_map).fillna(False)
    if "is_external_flow" in df.columns:
        external_map = df.groupby("category")["is_external_flow"].any()
        grouped["is_external_flow"] = grouped["category"].map(external_map).fillna(False)

    return grouped.reset_index(drop=True), warnings, issues


def _fill_change_eur_with_fx(
    *,
    account_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    cache_dir: str,
    logger: logging.Logger | None,
) -> tuple[pd.DataFrame, list[str], list[dict[str, object]]]:
    warnings: list[str] = []
    issues: list[dict[str, object]] = []
    if account_df.empty:
        return pd.DataFrame(), warnings, issues

    df = account_df.copy().sort_values("datetime")
    df = df[df["datetime"].notna()].copy()
    if df.empty:
        return pd.DataFrame(), warnings, issues
    df["date"] = df["datetime"].dt.normalize()

    start, end = daily_index.min(), daily_index.max()
    df["change_eur_filled"] = df["change_eur"]
    missing = df["change_eur_filled"].isna() & df["raw_change"].notna()
    for currency in sorted(set(df.loc[missing, "currency"].dropna().astype(str).str.upper())):
        if currency == "EUR":
            df.loc[missing & df["currency"].eq("EUR"), "change_eur_filled"] = df.loc[
                missing & df["currency"].eq("EUR"), "raw_change"
            ]
            continue
        try:
            fx = fetch_fx_series(
                currency=currency,
                start=start,
                end=end,
                cache_dir=cache_dir,
                logger=logger,
            ).reindex(daily_index).ffill().bfill()
        except UserFacingError:
            mask = missing & df["currency"].astype(str).str.upper().eq(currency)
            warnings.append(
                f"Account change conversion: missing FX series for currency {currency}. "
                f"Affected rows: {int(mask.sum())}."
            )
            issues.append(
                {
                    "label": f"Account change conversion missing FX ({currency})",
                    "count": int(mask.sum()),
                    "examples": _format_preview_df(
                        df.loc[mask, ["datetime", "description", "currency", "raw_change", "fx_rate"]]
                    ),
                }
            )
            continue
        mask = missing & df["currency"].astype(str).str.upper().eq(currency)
        rates = df.loc[mask, "date"].map(fx)
        df.loc[mask, "change_eur_filled"] = df.loc[mask, "raw_change"].values * rates.values

    return df, warnings, issues


def fetch_fx_series(
    *,
    currency: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str,
    logger: logging.Logger | None,
) -> pd.Series:
    currency = currency.upper()
    if currency == "EUR":
        return pd.Series(1.0, index=pd.date_range(start=start, end=end, freq="D"))
    ticker = f"{currency}EUR=X"
    return fetch_price_series(
        ticker=ticker,
        start=start,
        end=end,
        cache_dir=cache_dir,
        logger=logger,
    )


def fetch_price_series(
    *,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str,
    logger: logging.Logger | None,
) -> pd.Series:
    cache_path = _series_cache_path(cache_dir=cache_dir, ticker=ticker)
    cached = _load_cached_series(cache_path)
    cached = cached.sort_index()

    if yf is None:
        result = cached.loc[(cached.index >= start) & (cached.index <= end)]
        if not result.empty:
            _mark_offline_cache_usage(cache_dir=cache_dir, ticker=ticker, cache_path=cache_path)
            result.name = ticker
            return result
        raise UserFacingError(
            "yfinance is not available in this environment.",
            "Install dependencies from requirements.txt.",
        )

    need_fetch = cached.empty or start < cached.index.min() or end > cached.index.max()
    if need_fetch:
        try:
            fetched = _download_close_series(ticker=ticker, start=start, end=end)
        except Exception:
            fetched = pd.Series(dtype="float64")
        if not fetched.empty:
            cached = pd.concat([cached, fetched]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            _save_cached_series(cache_path, cached)
            _record_online_fetch(cache_dir=cache_dir, ticker=ticker)
        elif cached.empty:
            raise UserFacingError(
                f"Could not fetch price history for {ticker}.",
                "Check internet connection, ticker validity, and mappings.yml.",
            )
        else:
            _mark_offline_cache_usage(cache_dir=cache_dir, ticker=ticker, cache_path=cache_path)
            if logger:
                logger.warning("Using cached data only for ticker %s", ticker)

    result = cached.loc[(cached.index >= start) & (cached.index <= end)]
    if result.empty:
        raise UserFacingError(
            f"No cached/fetched prices available for {ticker} in requested date range.",
            "Ensure ticker mapping is correct and rerun with internet access once to populate cache.",
        )
    result.name = ticker
    return result


def _download_close_series(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    end_plus = end + pd.Timedelta(days=1)
    data = yf.download(
        tickers=ticker,
        start=start.date().isoformat(),
        end=end_plus.date().isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if data is None or data.empty:
        return pd.Series(dtype="float64")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker]
    else:
        close = data["Close"]
    close = close.tz_localize(None) if getattr(close.index, "tz", None) is not None else close
    close.index = pd.to_datetime(close.index).normalize()
    close = close.astype(float).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    return close


def _series_cache_path(cache_dir: str, ticker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", ticker)
    path = Path(cache_dir) / "prices" / f"{safe}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache_meta_path(cache_dir: str) -> Path:
    path = Path(cache_dir) / "prices" / "_cache_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _record_online_fetch(*, cache_dir: str, ticker: str) -> None:
    now = datetime.now().replace(microsecond=0)
    path = _cache_meta_path(cache_dir)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    tickers = data.get("tickers")
    if not isinstance(tickers, dict):
        tickers = {}
    timestamp = now.isoformat(timespec="seconds")
    tickers[str(ticker)] = timestamp
    data["tickers"] = tickers
    data["last_online_fetch_at"] = timestamp
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _read_last_online_fetch_timestamp(*, cache_dir: str) -> datetime | None:
    path = _cache_meta_path(cache_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        raw = data.get("last_online_fetch_at") if isinstance(data, dict) else None
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.notna(ts):
            return ts.to_pydatetime().replace(tzinfo=None)
    prices_dir = Path(cache_dir) / "prices"
    if not prices_dir.exists():
        return None
    mtimes = []
    for path_csv in prices_dir.glob("*.csv"):
        if path_csv.name.startswith("_"):
            continue
        try:
            mtimes.append(datetime.fromtimestamp(path_csv.stat().st_mtime))
        except Exception:
            continue
    if not mtimes:
        return None
    return max(mtimes)


def _mark_offline_cache_usage(*, cache_dir: str, ticker: str, cache_path: Path) -> None:
    _CACHE_RUNTIME_STATE["offline_mode"] = True
    offline_tickers = _CACHE_RUNTIME_STATE.get("offline_tickers", set())
    if not isinstance(offline_tickers, set):
        offline_tickers = set()
    offline_tickers.add(str(ticker))
    _CACHE_RUNTIME_STATE["offline_tickers"] = offline_tickers
    if _CACHE_RUNTIME_STATE.get("offline_cached_from") is None:
        cached_from = _read_last_online_fetch_timestamp(cache_dir=cache_dir)
        if cached_from is None and cache_path.exists():
            try:
                cached_from = datetime.fromtimestamp(cache_path.stat().st_mtime)
            except Exception:
                cached_from = None
        _CACHE_RUNTIME_STATE["offline_cached_from"] = cached_from


def _load_cached_series(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return pd.Series(dtype="float64")
    series = pd.Series(df["close"].astype(float).values, index=df["date"].dt.normalize())
    series = series.sort_index()
    return series


def _save_cached_series(path: Path, series: pd.Series) -> None:
    out = pd.DataFrame({"date": series.index, "close": series.values})
    out.to_csv(path, index=False)


def _format_preview_df(df: pd.DataFrame, max_rows: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sample = df.head(max_rows).copy()
    sample.insert(0, "row_number", [_row_number_value(idx) for idx in sample.index])
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return sample.reset_index(drop=True)


def _row_number_value(index_value: object) -> object:
    if isinstance(index_value, (int, np.integer)):
        return int(index_value) + 1
    return str(index_value)
