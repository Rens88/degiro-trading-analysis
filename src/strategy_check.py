from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_DESIRED_ETF_HOLDINGS,
    DEFAULT_DESIRED_NON_ETF_HOLDINGS,
    DEFAULT_MAX_SINGLE_CURRENCY_PCT,
    DEFAULT_MAX_SINGLE_HOLDING_PCT,
    DEFAULT_MAX_SINGLE_INDUSTRY_PCT,
    DEFAULT_MAX_TOP5_HOLDINGS_PCT,
    DEFAULT_MIN_OVER_VALUE_EUR,
    DEFAULT_MIN_TOTAL_HOLDINGS,
    DEFAULT_STRATEGY_DATASET_A_DIR,
    DEFAULT_STRATEGY_DATASET_B_DIR,
    DEFAULT_STRATEGY_FILE_PATH,
    DEFAULT_TARGET_CASH_PCT,
    DEFAULT_TARGET_ETF_FRACTION,
    REQUIRED_DATASET_FILES,
)
from .data_import import LoadedDataset, load_dataset
from .exceptions import UserFacingError
from .insights import build_ai_spread_analysis
from .logging_utils import setup_logger
from .portfolio_timeseries import latest_fx_rate
from .reconciliation import CashReconciliationResult, TotalsResult, combine_totals, reconcile_dataset
from .tables import build_four_tables


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

        table_over_target, spread_actions = evaluate_strategy(
            datasets=datasets,
            strategy=cfg.strategy,
            mappings_path=cfg.mappings_path,
            logger=logger,
        )

        action_required = (table_over_target is not None and not table_over_target.empty) or (
            spread_actions is not None and not spread_actions.empty
        )
        if not action_required:
            print("No action required based on current strategy thresholds.")
            return 0

        print("Action required/recommended by strategy check.")
        if table_over_target is not None and not table_over_target.empty:
            print("")
            print("Holdings over target threshold (same basis as app table 3.4):")
            print(_format_table(table_over_target))
        if spread_actions is not None and not spread_actions.empty:
            print("")
            print("Spread action recommendations:")
            print(_format_table(spread_actions))
        return EXIT_ACTION_REQUIRED
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del mappings_path  # kept for forward compatibility

    totals_results: dict[str, TotalsResult] = {}
    cash_results: dict[str, CashReconciliationResult] = {}
    merged_portfolio = pd.concat([d.portfolio for d in datasets.values()], ignore_index=True)

    for dataset in datasets.values():
        cash_result, totals_result = reconcile_dataset(
            account_label=dataset.account_label,
            portfolio=dataset.portfolio,
            account=dataset.account,
            fx_lookup=lambda ccy: latest_fx_rate(ccy, cache_dir="cache", logger=logger),
        )
        totals_results[dataset.account_label] = totals_result
        cash_results[dataset.account_label] = cash_result

    combined_totals = combine_totals(totals_results)
    holdings_for_tables = merged_portfolio.loc[~merged_portfolio["is_cash_like"]].copy()
    table_outputs = build_four_tables(
        holdings=holdings_for_tables,
        totals=combined_totals,
        target_etf_fraction=float(strategy["target_etf_fraction"]),
        desired_etf_holdings=int(strategy["desired_etf_holdings"]),
        desired_non_etf_holdings=int(strategy["desired_non_etf_holdings"]),
        min_over_value_eur=float(strategy["min_over_value_eur"]),
    )

    combined_cash_detail = _combine_cash_detail(cash_results)
    spread_analysis = build_ai_spread_analysis(
        holdings_df=holdings_for_tables,
        total_value_eur=float(combined_totals.total_value_eur),
        cash_value_eur=float(combined_totals.cash_value_eur),
        cash_detail_df=combined_cash_detail,
        strategy={
            "target_etf_fraction": float(strategy["target_etf_fraction"]),
            "desired_etf_holdings": int(strategy["desired_etf_holdings"]),
            "desired_non_etf_holdings": int(strategy["desired_non_etf_holdings"]),
            "target_cash_pct": float(strategy["target_cash_pct"]),
            "max_single_holding_pct": float(strategy["max_single_holding_pct"]),
            "max_top5_holdings_pct": float(strategy["max_top5_holdings_pct"]),
            "max_single_currency_pct": float(strategy["max_single_currency_pct"]),
            "max_single_industry_pct": float(strategy["max_single_industry_pct"]),
            "min_total_holdings": int(strategy["min_total_holdings"]),
        },
    )

    over_target = table_outputs.get("over_target", pd.DataFrame())
    spread_actions = (
        spread_analysis.get("action_plan_df", pd.DataFrame())
        if isinstance(spread_analysis, dict)
        else pd.DataFrame()
    )
    return over_target, spread_actions


def _combine_cash_detail(cash_results: dict[str, CashReconciliationResult]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, result in cash_results.items():
        detail = result.detail if isinstance(result.detail, pd.DataFrame) else pd.DataFrame()
        if detail.empty:
            continue
        cur = detail.copy()
        cur.insert(0, "dataset", label)
        rows.append(cur)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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
        "min_total_holdings": _to_int(data.get("min_total_holdings"), DEFAULT_MIN_TOTAL_HOLDINGS),
        "min_over_value_eur": _to_float(data.get("min_over_value_eur"), DEFAULT_MIN_OVER_VALUE_EUR),
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


def _format_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(empty)"
    out = df.copy()
    preferred_cols = [
        "account_label",
        "product",
        "ticker",
        "value_eur",
        "value_pct",
        "target_per_holding_pct",
        "target_value_eur",
        "over_target_eur",
    ]
    present = [c for c in preferred_cols if c in out.columns]
    if present:
        out = out[present]
    money_cols = [
        col
        for col in out.columns
        if (
            ("eur" in str(col).strip().lower()) or (str(col).strip().lower() in {"raw_balance", "raw_change"})
        )
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    for col in money_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(
            lambda value: f"{value:,.2f}" if pd.notna(value) else ""
        )
    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 240):
        return out.to_string(index=False)


if __name__ == "__main__":
    raise SystemExit(main())
