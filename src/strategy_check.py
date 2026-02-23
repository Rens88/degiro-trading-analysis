"""
AGENT_NOTE: CLI strategy gate used by `run_strategy_check_startup.bat`.

Interdependencies:
- Reuses data ingestion, reconciliation, table logic, and spread analysis from
  the Streamlit pipeline modules.
- Reads strategy defaults from `src/config.py` and optional overrides from
  `strategy/spread_strategy.json`.
- Startup script compatibility depends on `EXIT_ACTION_REQUIRED = 10`.

When editing:
- Keep strategy schema and defaults aligned with `src/app.py` sidebar fields.
- Keep action detection semantics aligned with spread-analysis output columns.
- See `src/INTERDEPENDENCIES.md` for the shared contract map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DESIRED_ETF_HOLDINGS,
    DEFAULT_DESIRED_NON_ETF_HOLDINGS,
    DEFAULT_MAX_PAIR_CORRELATION,
    DEFAULT_MAX_SINGLE_CURRENCY_PCT,
    DEFAULT_MAX_SINGLE_HOLDING_PCT,
    DEFAULT_MAX_SINGLE_INDUSTRY_PCT,
    DEFAULT_MAX_TOP5_HOLDINGS_PCT,
    DEFAULT_MIN_OVER_VALUE_EUR,
    DEFAULT_MIN_TOTAL_HOLDINGS,
    DEFAULT_STRATEGY_DATASET_A_DIR,
    DEFAULT_STRATEGY_DATASET_B_DIR,
    DEFAULT_STRATEGY_FILE_PATH,
    DEFAULT_TARGET_CURRENCY_PCT,
    DEFAULT_TARGET_CASH_PCT,
    DEFAULT_TARGET_ETF_FRACTION,
    DEFAULT_TARGET_INDUSTRY_PCT,
    DEFAULT_TARGET_STYLE_PCT,
    REQUIRED_DATASET_FILES,
)
from .data_import import LoadedDataset, load_dataset
from .exceptions import UserFacingError
from .logging_utils import setup_logger
from .portfolio_timeseries import compute_portfolio_timeseries, latest_fx_rate
from .reconciliation import TotalsResult, combine_totals, reconcile_dataset


# Exit code used by the startup BAT script to decide whether to launch Streamlit.
EXIT_ACTION_REQUIRED = 10


@dataclass
class ResolvedConfig:
    strategy_file: Path
    dataset_a_dir: Path
    dataset_b_dir: Path
    mappings_path: Path
    strategy: dict[str, Any]


def main() -> int:
    args = _parse_args()
    logger = setup_logger("logs/strategy_check.log")
    try:
        cfg = resolve_config(args)
        datasets = load_datasets(cfg)
        if not datasets:
            print("No dataset could be loaded. Nothing to check.")
            return 2

        report = evaluate_strategy(
            datasets=datasets,
            strategy=cfg.strategy,
            mappings_path=cfg.mappings_path,
            logger=logger,
        )
        _print_strategy_report(report)
        if bool(report.get("action_required", False)):
            print("")
            print("Action required/recommended by strategy check.")
            return EXIT_ACTION_REQUIRED
        print("")
        print("No action required based on current strategy thresholds.")
        return 0
    except UserFacingError as exc:
        print(exc.to_ui_text())
        logger.exception("User-facing strategy check error")
        return 3
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Unexpected error in strategy check: {exc}")
        logger.exception("Unexpected strategy check error")
        return 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checks the saved spread strategy against the latest portfolio composition and "
            "returns exit code 10 if action is required."
        )
    )
    parser.add_argument(
        "--strategy-file",
        default=str(DEFAULT_STRATEGY_FILE_PATH),
        help="Path to saved strategy JSON file.",
    )
    parser.add_argument(
        "--dataset-a-dir",
        default=None,
        help="Directory containing Dataset A CSV exports.",
    )
    parser.add_argument(
        "--dataset-b-dir",
        default=None,
        help="Directory containing Dataset B CSV exports.",
    )
    parser.add_argument(
        "--mappings-path",
        default=None,
        help="Path to mappings.yml.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    strategy_path = Path(str(args.strategy_file)).expanduser()
    if not strategy_path.is_absolute():
        strategy_path = Path.cwd() / strategy_path

    loaded = {}
    if strategy_path.exists():
        loaded = _load_strategy_file(strategy_path)

    strategy = _strategy_with_defaults(loaded.get("strategy", {}))
    data_sources = loaded.get("data_sources", {}) if isinstance(loaded.get("data_sources", {}), dict) else {}

    dataset_a_dir = _resolve_path(
        preferred=getattr(args, "dataset_a_dir", None),
        fallback=data_sources.get("dataset_a_dir"),
        default=str(DEFAULT_STRATEGY_DATASET_A_DIR),
    )
    dataset_b_dir = _resolve_path(
        preferred=getattr(args, "dataset_b_dir", None),
        fallback=data_sources.get("dataset_b_dir"),
        default=str(DEFAULT_STRATEGY_DATASET_B_DIR),
    )
    mappings_path = _resolve_path(
        preferred=getattr(args, "mappings_path", None),
        fallback=data_sources.get("mappings_path"),
        default="mappings.yml",
    )

    return ResolvedConfig(
        strategy_file=strategy_path,
        dataset_a_dir=dataset_a_dir,
        dataset_b_dir=dataset_b_dir,
        mappings_path=mappings_path,
        strategy=strategy,
    )


def load_datasets(cfg: ResolvedConfig) -> dict[str, LoadedDataset]:
    datasets: dict[str, LoadedDataset] = {}
    for panel, folder in [("dataset_a", cfg.dataset_a_dir), ("dataset_b", cfg.dataset_b_dir)]:
        if not _has_required_files(folder):
            print(f"Skipping {panel}: required files not found in {folder}")
            continue
        label = f"{panel.replace('_', ' ').title()} ({folder.name})"
        ds = load_dataset(
            account_label=label,
            transactions_source=folder / "Transactions.csv",
            portfolio_source=folder / "Portfolio.csv",
            account_source=folder / "Account.csv",
            mappings_path=cfg.mappings_path,
        )
        datasets[panel] = ds
    return datasets


def evaluate_strategy(
    *,
    datasets: dict[str, LoadedDataset],
    strategy: dict[str, Any],
    mappings_path: Path,
    logger: Any,
) -> dict[str, Any]:
    # AGENT_NOTE: This mirrors app-level spread logic but returns compact action
    # tables for startup gating. Keep in sync with `app.py` strategy parameters.
    del mappings_path  # kept for forward compatibility

    totals_results: dict[str, TotalsResult] = {}
    merged_portfolio = pd.concat([d.portfolio for d in datasets.values()], ignore_index=True)
    merged_transactions = pd.concat([d.transactions for d in datasets.values()], ignore_index=True)
    merged_account = pd.concat([d.account for d in datasets.values()], ignore_index=True)
    merged_instruments = (
        pd.concat([d.instruments for d in datasets.values()], ignore_index=True)
        .drop_duplicates(subset=["instrument_id"], keep="first")
        .copy()
    )

    for dataset in datasets.values():
        _cash_result, totals_result = reconcile_dataset(
            account_label=dataset.account_label,
            portfolio=dataset.portfolio,
            account=dataset.account,
            fx_lookup=lambda ccy: latest_fx_rate(ccy, cache_dir="cache", logger=logger),
        )
        totals_results[dataset.account_label] = totals_result

    combined_totals = combine_totals(totals_results)
    ts_result = None
    prices_eur = pd.DataFrame()
    metrics_df = pd.DataFrame()
    try:
        ts_result = compute_portfolio_timeseries(
            transactions=merged_transactions,
            account=merged_account,
            instruments=merged_instruments,
            cache_dir="cache",
            logger=logger,
        )
        prices_eur = ts_result.prices_eur
        metrics_df = ts_result.metrics
    except Exception:
        # Startup check keeps working without time-series data.
        prices_eur = pd.DataFrame()
        metrics_df = pd.DataFrame()

    latest_price = _build_latest_price_map(prices_eur)
    latest_price_date = _build_latest_price_date_map(prices_eur)
    cost_basis = _compute_open_cost_basis_by_instrument(merged_transactions)
    holdings_df = _build_holdings_snapshot(
        portfolio_df=merged_portfolio,
        combined_total_value=float(combined_totals.total_value_eur),
        latest_price_map=latest_price,
        latest_price_date_map=latest_price_date,
        cost_basis_map=cost_basis,
    )

    growth_df = _build_growth_table(metrics_df)
    non_etf_df = holdings_df.loc[~holdings_df["is_etf"]].copy()
    etf_df = holdings_df.loc[holdings_df["is_etf"]].copy()
    non_etf_df["portfolio_pct"] = pd.to_numeric(non_etf_df.get("value_pct"), errors="coerce")
    etf_df["portfolio_pct"] = pd.to_numeric(etf_df.get("value_pct"), errors="coerce")
    holdings_cols = [
        "ticker",
        "product",
        "qty",
        "last_px_eur",
        "px_date",
        "value_eur",
        "pnl_eur",
        "portfolio_pct",
    ]
    non_etf_df = non_etf_df.sort_values("value_pct", ascending=False).reset_index(drop=True)
    etf_df = etf_df.sort_values("value_pct", ascending=False).reset_index(drop=True)
    non_etf_df = non_etf_df[[c for c in holdings_cols if c in non_etf_df.columns]]
    etf_df = etf_df[[c for c in holdings_cols if c in etf_df.columns]]

    cash_action_df = _build_cash_split_action(
        combined_total_value=float(combined_totals.total_value_eur),
        cash_value=float(combined_totals.cash_value_eur),
        etf_value=float(etf_df["value_eur"].sum()),
        non_etf_value=float(non_etf_df["value_eur"].sum()),
        target_etf_fraction=float(strategy["target_etf_fraction"]),
        target_cash_pct=float(strategy["target_cash_pct"]),
    )
    cash_action_df = cash_action_df[
        [
            c
            for c in [
                "cash_eur",
                "target_cash_eur",
                "deployable_cash_eur",
                "to_etf_eur",
                "to_non_etf_eur",
                "etf_w_pct",
                "non_etf_w_pct",
            ]
            if c in cash_action_df.columns
        ]
    ]
    trim_actions_df = _build_trim_actions_table(
        holdings_df=holdings_df,
        combined_total_value=float(combined_totals.total_value_eur),
        target_etf_fraction=float(strategy["target_etf_fraction"]),
        desired_etf_holdings=int(strategy["desired_etf_holdings"]),
        desired_non_etf_holdings=int(strategy["desired_non_etf_holdings"]),
        min_over_value_eur=float(strategy["min_over_value_eur"]),
    )

    deployable_cash = float(cash_action_df["deployable_cash_eur"].iloc[0]) if not cash_action_df.empty else 0.0
    action_required = (deployable_cash > 1e-6) or (not trim_actions_df.empty)
    return {
        "growth_df": growth_df,
        "non_etf_df": non_etf_df,
        "etf_df": etf_df,
        "cash_action_df": cash_action_df,
        "trim_actions_df": trim_actions_df,
        "action_required": bool(action_required),
    }


def _print_strategy_report(report: dict[str, Any]) -> None:
    growth_df = report.get("growth_df", pd.DataFrame())
    non_etf_df = report.get("non_etf_df", pd.DataFrame())
    etf_df = report.get("etf_df", pd.DataFrame())
    cash_action_df = report.get("cash_action_df", pd.DataFrame())
    trim_actions_df = report.get("trim_actions_df", pd.DataFrame())

    print("")
    print("Portfolio growth")
    print(_format_table(growth_df))

    print("")
    print("Non-ETF holdings (sorted by portfolio weight)")
    print(_format_table(non_etf_df))

    print("")
    print("ETF holdings (sorted by portfolio weight)")
    print(_format_table(etf_df))

    print("")
    print("Suggested cash deployment to target ETF/non-ETF ratio")
    print(_format_table(cash_action_df))

    print("")
    print("Over-target holdings to trim back to intended percentage")
    print(_format_table(trim_actions_df))


def _build_growth_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "period",
        "start_date",
        "end_date",
        "start_eur",
        "end_eur",
        "deposited_cash_eur",
        "chg_eur",
        "growth_eur",
        "chg_pct",
    ]
    if metrics_df is None or metrics_df.empty or "portfolio_value" not in metrics_df.columns:
        return pd.DataFrame(columns=cols)

    series = pd.to_numeric(metrics_df["portfolio_value"], errors="coerce")
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna() & series.notna()].sort_index()
    if series.empty:
        return pd.DataFrame(columns=cols)

    deposits = pd.Series(dtype=float)
    if "total_deposits" in metrics_df.columns:
        deposits = pd.to_numeric(metrics_df["total_deposits"], errors="coerce")
        deposits.index = pd.to_datetime(deposits.index, errors="coerce")
        deposits = deposits[deposits.index.notna() & deposits.notna()].sort_index()

    end_dt = pd.Timestamp(series.index.max())
    end_val = float(series.iloc[-1])
    month_start = end_dt.to_period("M").to_timestamp(how="start")

    def _deposited_between(start_dt: pd.Timestamp, end_dt_local: pd.Timestamp) -> float:
        if deposits.empty:
            return float("nan")
        start_dep = _series_last_on_or_before(deposits, start_dt)
        if not np.isfinite(start_dep):
            start_dep, _ = _series_first_on_or_after(deposits, start_dt)
        end_dep = _series_last_on_or_before(deposits, end_dt_local)
        if not np.isfinite(end_dep):
            end_dep, _ = _series_first_on_or_after(deposits, end_dt_local)
        if not np.isfinite(start_dep) or not np.isfinite(end_dep):
            return float("nan")
        return float(end_dep - start_dep)

    rows: list[dict[str, Any]] = []
    checkpoints = [
        ("1D", end_dt - pd.Timedelta(days=1)),
        ("7D", end_dt - pd.Timedelta(days=7)),
    ]
    for label, start_dt in checkpoints:
        start_val = _series_last_on_or_before(series, start_dt)
        if not np.isfinite(start_val):
            continue
        change = end_val - start_val
        deposited = _deposited_between(pd.Timestamp(start_dt), end_dt)
        growth = change - deposited if np.isfinite(deposited) else float("nan")
        rows.append(
            {
                "period": label,
                "start_date": pd.Timestamp(start_dt).strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
                "start_eur": start_val,
                "end_eur": end_val,
                "deposited_cash_eur": deposited,
                "chg_eur": change,
                "growth_eur": growth,
                "chg_pct": (change / start_val * 100.0) if abs(start_val) > 1e-12 else np.nan,
            }
        )

    # Month-to-date: prefer previous day before month start; fall back to first value in month.
    mtd_start_val = _series_last_on_or_before(series, month_start - pd.Timedelta(days=1))
    mtd_start_dt = month_start - pd.Timedelta(days=1)
    if not np.isfinite(mtd_start_val):
        mtd_start_val, mtd_start_dt = _series_first_on_or_after(series, month_start)
    if np.isfinite(mtd_start_val):
        mtd_change = end_val - mtd_start_val
        mtd_deposited = _deposited_between(pd.Timestamp(mtd_start_dt), end_dt)
        mtd_growth = mtd_change - mtd_deposited if np.isfinite(mtd_deposited) else float("nan")
        rows.append(
            {
                "period": "MTD",
                "start_date": pd.Timestamp(mtd_start_dt).strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
                "start_eur": mtd_start_val,
                "end_eur": end_val,
                "deposited_cash_eur": mtd_deposited,
                "chg_eur": mtd_change,
                "growth_eur": mtd_growth,
                "chg_pct": (mtd_change / mtd_start_val * 100.0) if abs(mtd_start_val) > 1e-12 else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


def _series_last_on_or_before(series: pd.Series, dt: pd.Timestamp) -> float:
    subset = series.loc[series.index <= pd.Timestamp(dt)]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[-1])


def _series_first_on_or_after(series: pd.Series, dt: pd.Timestamp) -> tuple[float, pd.Timestamp]:
    subset = series.loc[series.index >= pd.Timestamp(dt)]
    if subset.empty:
        return float("nan"), pd.Timestamp(dt)
    return float(subset.iloc[0]), pd.Timestamp(subset.index[0])


def _build_latest_price_map(prices_eur: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if prices_eur is None or prices_eur.empty:
        return out
    px = prices_eur.copy()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index()
    for col in px.columns:
        series = pd.to_numeric(px[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[str(col)] = float(series.iloc[-1])
    return out


def _build_latest_price_date_map(prices_eur: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if prices_eur is None or prices_eur.empty:
        return out
    px = prices_eur.copy()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index()
    for col in px.columns:
        series = pd.to_numeric(px[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[str(col)] = pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d")
    return out


def _compute_open_cost_basis_by_instrument(transactions_df: pd.DataFrame) -> dict[str, float]:
    if transactions_df is None or transactions_df.empty:
        return {}
    tx = transactions_df.copy()
    tx["datetime"] = pd.to_datetime(tx.get("datetime"), errors="coerce")
    tx["instrument_id"] = tx.get("instrument_id", pd.Series("", index=tx.index)).astype(str)
    tx["quantity"] = pd.to_numeric(tx.get("quantity"), errors="coerce")
    tx["total_eur"] = pd.to_numeric(tx.get("total_eur"), errors="coerce")
    tx["fees_eur"] = pd.to_numeric(tx.get("fees_eur"), errors="coerce").fillna(0.0).abs()
    tx = tx[tx["datetime"].notna()].sort_values("datetime")
    if "is_cash_like" in tx.columns:
        tx = tx.loc[~tx["is_cash_like"].fillna(False)].copy()
    tx = tx[tx["instrument_id"].astype(str).str.strip().ne("")]
    if tx.empty:
        return {}

    qty_map: dict[str, float] = {}
    cost_map: dict[str, float] = {}
    for row in tx.itertuples(index=False):
        iid = str(getattr(row, "instrument_id", "")).strip()
        if iid == "":
            continue
        qty = float(getattr(row, "quantity", np.nan))
        total_eur = float(getattr(row, "total_eur", np.nan))
        fees_eur = float(getattr(row, "fees_eur", 0.0))
        if not np.isfinite(qty) or abs(qty) < 1e-12:
            continue

        prev_qty = float(qty_map.get(iid, 0.0))
        prev_cost = float(cost_map.get(iid, 0.0))
        if qty > 0.0:
            spend = (-total_eur if np.isfinite(total_eur) else 0.0) + fees_eur
            spend = max(float(spend), 0.0)
            qty_map[iid] = prev_qty + qty
            cost_map[iid] = prev_cost + spend
            continue

        sell_qty = -qty
        if prev_qty <= 1e-12:
            qty_map[iid] = max(prev_qty - sell_qty, 0.0)
            cost_map[iid] = 0.0
            continue
        qty_reduced = min(sell_qty, prev_qty)
        avg_cost = prev_cost / prev_qty if prev_qty > 0.0 else 0.0
        new_qty = prev_qty - qty_reduced
        new_cost = prev_cost - avg_cost * qty_reduced
        qty_map[iid] = max(new_qty, 0.0)
        cost_map[iid] = max(new_cost, 0.0)
        if qty_map[iid] <= 1e-9:
            qty_map[iid] = 0.0
            cost_map[iid] = 0.0

    return {k: float(v) for k, v in cost_map.items() if np.isfinite(v) and v >= 0.0}


def _build_holdings_snapshot(
    *,
    portfolio_df: pd.DataFrame,
    combined_total_value: float,
    latest_price_map: dict[str, float],
    latest_price_date_map: dict[str, str],
    cost_basis_map: dict[str, float],
) -> pd.DataFrame:
    if portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame(
            columns=["instrument_id", "ticker", "product", "is_etf", "qty", "last_px_eur", "px_date", "value_eur", "pnl_eur", "value_pct"]
        )
    df = portfolio_df.copy()
    if "is_cash_like" in df.columns:
        df = df.loc[~df["is_cash_like"].fillna(False)].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["instrument_id", "ticker", "product", "is_etf", "qty", "last_px_eur", "px_date", "value_eur", "pnl_eur", "value_pct"]
        )

    for col in ["instrument_id", "ticker", "product"]:
        if col not in df.columns:
            df[col] = ""
    if "is_etf" not in df.columns:
        df["is_etf"] = False
    df["instrument_id"] = df["instrument_id"].fillna("").astype(str)
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    df["product"] = df["product"].fillna("").astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce")
    df["value_eur"] = pd.to_numeric(df.get("value_eur"), errors="coerce").fillna(0.0)
    df["is_etf"] = df["is_etf"].fillna(False).astype(bool)

    grouped = (
        df.groupby(["instrument_id", "ticker", "product", "is_etf"], as_index=False)
        .agg(qty=("quantity", "sum"), value_eur=("value_eur", "sum"))
        .copy()
    )
    grouped = grouped.loc[(grouped["qty"].abs() > 1e-9) | (grouped["value_eur"].abs() > 1e-6)].copy()
    if grouped.empty:
        return pd.DataFrame(
            columns=["instrument_id", "ticker", "product", "is_etf", "qty", "last_px_eur", "px_date", "value_eur", "pnl_eur", "value_pct"]
        )

    grouped["last_px_eur"] = grouped["instrument_id"].map(lambda iid: latest_price_map.get(str(iid), np.nan))
    fallback_px = np.where(grouped["qty"].abs() > 1e-12, grouped["value_eur"] / grouped["qty"].abs(), np.nan)
    grouped["last_px_eur"] = grouped["last_px_eur"].where(grouped["last_px_eur"].notna(), fallback_px)
    grouped["px_date"] = grouped["instrument_id"].map(lambda iid: latest_price_date_map.get(str(iid), ""))
    grouped["cost_basis_eur"] = grouped["instrument_id"].map(lambda iid: float(cost_basis_map.get(str(iid), np.nan)))
    grouped["pnl_eur"] = grouped["value_eur"] - grouped["cost_basis_eur"]
    grouped["value_pct"] = np.where(
        np.isfinite(combined_total_value) and combined_total_value > 0.0,
        grouped["value_eur"] / combined_total_value * 100.0,
        np.nan,
    )
    grouped["product"] = grouped["product"].map(lambda v: _truncate_text(str(v), max_len=28))
    grouped["qty"] = grouped["qty"].abs().round(0).astype("Int64")
    return grouped.sort_values("value_pct", ascending=False).reset_index(drop=True)


def _build_cash_split_action(
    *,
    combined_total_value: float,
    cash_value: float,
    etf_value: float,
    non_etf_value: float,
    target_etf_fraction: float,
    target_cash_pct: float,
) -> pd.DataFrame:
    if not np.isfinite(combined_total_value) or combined_total_value <= 0.0:
        return pd.DataFrame(
            columns=[
                "cash_eur",
                "target_cash_eur",
                "deployable_cash_eur",
                "to_etf_eur",
                "to_non_etf_eur",
                "etf_w_pct",
                "non_etf_w_pct",
            ]
        )
    target_cash_value = target_cash_pct / 100.0 * combined_total_value
    deployable = max(cash_value - target_cash_value, 0.0)
    target_etf_value = target_etf_fraction * combined_total_value
    target_non_etf_value = (1.0 - target_etf_fraction) * combined_total_value
    gap_etf = max(target_etf_value - etf_value, 0.0)
    gap_non = max(target_non_etf_value - non_etf_value, 0.0)
    if deployable > 0.0:
        gap_total = gap_etf + gap_non
        if gap_total > 0.0:
            to_etf = deployable * gap_etf / gap_total
            to_non = deployable * gap_non / gap_total
        else:
            to_etf = deployable * target_etf_fraction
            to_non = deployable * (1.0 - target_etf_fraction)
    else:
        to_etf = 0.0
        to_non = 0.0

    return pd.DataFrame(
        [
            {
                "cash_eur": float(cash_value),
                "target_cash_eur": float(target_cash_value),
                "deployable_cash_eur": float(deployable),
                "to_etf_eur": float(to_etf),
                "to_non_etf_eur": float(to_non),
                "etf_w_pct": float(etf_value / combined_total_value * 100.0),
                "non_etf_w_pct": float(non_etf_value / combined_total_value * 100.0),
            }
        ]
    )


def _build_trim_actions_table(
    *,
    holdings_df: pd.DataFrame,
    combined_total_value: float,
    target_etf_fraction: float,
    desired_etf_holdings: int,
    desired_non_etf_holdings: int,
    min_over_value_eur: float,
) -> pd.DataFrame:
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(
            columns=["ticker", "product", "segment", "qty", "w_pct", "target_pct", "over_eur", "sell_qty"]
        )
    if not np.isfinite(combined_total_value) or combined_total_value <= 0.0:
        return pd.DataFrame(
            columns=["ticker", "product", "segment", "qty", "w_pct", "target_pct", "over_eur", "sell_qty"]
        )
    n_etf = max(int(desired_etf_holdings), 1)
    n_non = max(int(desired_non_etf_holdings), 1)
    out = holdings_df.copy()
    out["target_pct"] = np.where(
        out["is_etf"],
        target_etf_fraction / n_etf * 100.0,
        (1.0 - target_etf_fraction) / n_non * 100.0,
    )
    out["target_value_eur"] = out["target_pct"] / 100.0 * combined_total_value
    out["over_eur"] = out["value_eur"] - out["target_value_eur"]
    out = out.loc[pd.to_numeric(out["over_eur"], errors="coerce") > float(min_over_value_eur)].copy()
    if out.empty:
        return pd.DataFrame(
            columns=["ticker", "product", "segment", "qty", "w_pct", "target_pct", "over_eur", "sell_qty"]
        )
    out["sell_qty"] = np.where(
        pd.to_numeric(out["last_px_eur"], errors="coerce") > 1e-12,
        out["over_eur"] / out["last_px_eur"],
        np.nan,
    )
    out["segment"] = np.where(out["is_etf"], "ETF", "Non-ETF")
    cols = ["ticker", "product", "segment", "qty", "value_pct", "target_pct", "over_eur", "sell_qty"]
    out = out[cols].rename(columns={"value_pct": "w_pct"}).sort_values("over_eur", ascending=False)
    return out.reset_index(drop=True)


def _truncate_text(value: str, *, max_len: int) -> str:
    txt = str(value).strip()
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3].rstrip() + "..."


def _has_required_files(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and all((folder / name).exists() for name in REQUIRED_DATASET_FILES)


def _load_strategy_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UserFacingError(
            "Saved strategy file is not valid JSON.",
            f"File: {path}\nError: {exc}",
        ) from exc


def _resolve_path(*, preferred: str | None, fallback: str | None, default: str) -> Path:
    raw = preferred if preferred not in {None, ""} else fallback
    if raw in {None, ""}:
        raw = default
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _strategy_with_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    # AGENT_NOTE: Maintain parity with `app.py` sidebar keys and `config.py`
    # defaults so saved strategy JSON can be used by both UI and startup CLI.
    data = raw if isinstance(raw, dict) else {}
    return {
        "target_etf_fraction": _to_float(data.get("target_etf_fraction"), DEFAULT_TARGET_ETF_FRACTION),
        "desired_etf_holdings": _to_int(data.get("desired_etf_holdings"), DEFAULT_DESIRED_ETF_HOLDINGS),
        "desired_non_etf_holdings": _to_int(
            data.get("desired_non_etf_holdings"),
            DEFAULT_DESIRED_NON_ETF_HOLDINGS,
        ),
        "target_cash_pct": _to_float(data.get("target_cash_pct"), DEFAULT_TARGET_CASH_PCT),
        "max_single_holding_pct": _to_float(
            data.get("max_single_holding_pct"),
            DEFAULT_MAX_SINGLE_HOLDING_PCT,
        ),
        "max_top5_holdings_pct": _to_float(
            data.get("max_top5_holdings_pct"),
            DEFAULT_MAX_TOP5_HOLDINGS_PCT,
        ),
        "max_single_currency_pct": _to_float(
            data.get("max_single_currency_pct"),
            DEFAULT_MAX_SINGLE_CURRENCY_PCT,
        ),
        "max_single_industry_pct": _to_float(
            data.get("max_single_industry_pct"),
            DEFAULT_MAX_SINGLE_INDUSTRY_PCT,
        ),
        "max_pair_correlation": _to_float(
            data.get("max_pair_correlation"),
            DEFAULT_MAX_PAIR_CORRELATION,
        ),
        "min_total_holdings": _to_int(data.get("min_total_holdings"), DEFAULT_MIN_TOTAL_HOLDINGS),
        "min_over_value_eur": _to_float(data.get("min_over_value_eur"), DEFAULT_MIN_OVER_VALUE_EUR),
        "target_currency_pct": _normalize_target_pct_map(
            data.get("target_currency_pct"),
            default=DEFAULT_TARGET_CURRENCY_PCT,
        ),
        "target_industry_pct": _normalize_target_pct_map(
            data.get("target_industry_pct"),
            default=DEFAULT_TARGET_INDUSTRY_PCT,
        ),
        "target_style_pct": _normalize_target_pct_map(
            data.get("target_style_pct"),
            default=DEFAULT_TARGET_STYLE_PCT,
        ),
        "holding_category_overrides": _normalize_holding_category_overrides(
            data.get("holding_category_overrides")
        ),
    }


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_target_pct_map(raw: Any, *, default: dict[str, float] | None = None) -> dict[str, float]:
    source = raw if isinstance(raw, dict) else (default or {})
    out: dict[str, float] = {}
    for key, value in source.items():
        label = str(key).strip()
        if label == "":
            continue
        try:
            pct = float(value)
        except Exception:
            continue
        if pd.notna(pct):
            out[label] = max(float(pct), 0.0)
    return out


def _normalize_holding_category_overrides(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        instrument_id = str(key).strip()
        if instrument_id == "" or not isinstance(value, dict):
            continue
        entry: dict[str, str] = {}
        style = str(value.get("style", "")).strip()
        industry = str(value.get("industry", "")).strip()
        if style:
            entry["style"] = style
        if industry:
            entry["industry"] = industry
        if entry:
            out[instrument_id] = entry
    return out


def _format_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(empty)"
    out = df.copy()
    if len(out) > 30:
        out = out.head(30).copy()

    rename_map = {
        "start_eur": "start",
        "end_eur": "end",
        "deposited_cash_eur": "deposited",
        "chg_eur": "chg",
        "growth_eur": "growth",
        "chg_pct": "chg_pct",
        "last_px_eur": "last_px",
        "value_eur": "value",
        "pnl_eur": "plus_minus",
        "target_cash_eur": "target_cash",
        "deployable_cash_eur": "deployable",
        "to_etf_eur": "to_etf",
        "to_non_etf_eur": "to_non_etf",
        "etf_w_pct": "etf_w_pct",
        "non_etf_w_pct": "non_etf_w_pct",
        "target_pct": "target_pct",
        "over_eur": "over_eur",
        "sell_qty": "sell_qty",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for col in out.columns:
        if col in {"qty", "sell_qty"}:
            qty = pd.to_numeric(out[col], errors="coerce")
            out[col] = qty.map(lambda v: f"{int(round(v))}" if pd.notna(v) else "")
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if not values.notna().any():
            if out[col].dtype == "object":
                out[col] = out[col].astype(str).map(lambda v: _truncate_text(v, max_len=28))
            continue

        name = str(col).lower()
        if "pct" in name:
            out[col] = values.map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
        elif any(
            token in name
            for token in [
                "eur",
                "cash",
                "value",
                "px",
                "price",
                "plus_minus",
                "chg",
                "growth",
                "deposit",
                "target",
                "to_etf",
                "to_non_etf",
                "over",
                "start",
                "end",
            ]
        ):
            out[col] = values.map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
        else:
            out[col] = values.map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")

    with pd.option_context(
        "display.max_rows",
        30,
        "display.max_columns",
        None,
        "display.width",
        120,
        "display.max_colwidth",
        28,
    ):
        return out.to_string(index=False)


if __name__ == "__main__":
    raise SystemExit(main())
