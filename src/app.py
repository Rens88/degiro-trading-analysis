"""
AGENT_NOTE: Streamlit orchestrator and integration hub.

Interdependencies:
- Consumes normalized datasets from `src/data_import.py`.
- Uses `src/reconciliation.py`, `src/portfolio_timeseries.py`, `src/tables.py`,
  `src/insights.py`, and `src/plots.py` to compute all outputs.
- `process_loaded_datasets()` is the single producer for
  `st.session_state["workflow"]["processed"]`.
- `render_main()` is the main consumer of those processed keys and expects stable
  dict/table field names.

When editing:
- Treat `workflow["processed"]` keys as a public interface inside this app.
- If you rename strategy parameters, also update `src/strategy_check.py` and
  defaults in `src/config.py`.
- See `src/INTERDEPENDENCIES.md` for the full contract map.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    DEFAULT_DESIRED_ETF_HOLDINGS,
    DEFAULT_DESIRED_NON_ETF_HOLDINGS,
    DEFAULT_MAX_PAIR_CORRELATION,
    DEFAULT_MAX_SINGLE_CURRENCY_PCT,
    DEFAULT_MAX_SINGLE_HOLDING_PCT,
    DEFAULT_MAX_SINGLE_INDUSTRY_PCT,
    DEFAULT_MAX_TOP5_HOLDINGS_PCT,
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_STRATEGY_DATASET_A_DIR,
    DEFAULT_STRATEGY_DATASET_B_DIR,
    DEFAULT_STRATEGY_FILE_PATH,
    DEFAULT_TARGET_CURRENCY_PCT,
    DEFAULT_MIN_TOTAL_HOLDINGS,
    DEFAULT_MEDIAN_WINDOW_MONTHS,
    DEFAULT_MIN_OVER_VALUE_EUR,
    REQUIRED_DATASET_FILES,
    DEFAULT_TARGET_CASH_PCT,
    DEFAULT_TARGET_ETF_FRACTION,
    DEFAULT_TARGET_INDUSTRY_PCT,
    DEFAULT_TARGET_STYLE_PCT,
    SAMPLE_DATASETS,
    STATE_LOADING_DATA,
    STATE_ORDER,
    STATE_PROCESSING,
    STATE_RELOADING_EXPORT,
    STATE_SELECTING_PARAMS,
    STATE_VIEWING_RESULTS,
)
from .data_import import LoadedDataset, load_dataset, validate_uploaded_file_set
from .exceptions import UserFacingError
from .insights import build_ai_generated_insights, build_ai_spread_analysis, build_performance_dashboard
from .logging_utils import log, setup_logger
from .plots import (
    build_degiro_costs_quarterly_figure,
    build_benchmark_comparison_figure,
    build_cash_allocation_figure,
    build_drawdown_figure,
    build_holdings_over_time_figure,
    build_performance_over_time_figure,
    build_period_decomposition_figure,
    build_normalized_median_figure,
    build_portfolio_over_time_figure,
)
from .portfolio_timeseries import (
    compute_portfolio_timeseries,
    fetch_fx_series,
    fetch_price_series,
    get_cache_runtime_state,
    latest_fx_rate,
    prime_cache_runtime_state,
    summarize_account_categories,
)
from .reconciliation import (
    CashReconciliationResult,
    TotalsResult,
    combine_totals,
    reconcile_dataset,
    reconciliation_table,
)
from .tables import build_four_tables, build_monthly_starting_portfolio_value_table
from .ticker_characteristics import resolve_ticker_characteristics


MANUAL_MONTHLY_TRACKED_VALUES_RAW: list[tuple[str, str]] = [
    ("11/1/2022", "532,07"),
    ("12/1/2022", "1.847,14"),
    ("1/1/2023", "2.826,07"),
    ("1/16/2023", "4.279,67"),
    ("2/1/2023", "4.619,66"),
    ("3/1/2023", "5.738,97"),
    ("4/1/2023", "7.098,35"),
    ("5/1/2023", "8.153,37"),
    ("6/1/2023", "10.439,61"),
    ("7/1/2023", "11.281,32"),
    ("8/1/2023", "12.858,27"),
    ("9/1/2023", "14.115,45"),
    ("10/1/2023", "14.627,36"),
    ("11/1/2023", "16.158,44"),
    ("12/1/2023", "18.781,00"),
    ("1/1/2024", "21.336,00"),
    ("2/1/2024", "22.958,14"),
    ("3/1/2024", "24.559,00"),
    ("4/1/2024", "27.765,00"),
    ("5/1/2024", "28.907,00"),
    ("6/1/2024", "31.337,00"),
    ("7/1/2024", "32.709,32"),
    ("8/1/2024", "34.698,80"),
    ("9/1/2024", "36.077,00"),
    ("10/1/2024", "38.023,30"),
    ("11/1/2024", "39.082,80"),
    ("12/1/2024", "41.511,54"),
    ("1/1/2025", "42.227,00"),
    ("2/1/2025", "45.353,00"),
    ("3/1/2025", "46.428,14"),
    ("4/1/2025", "45.596,60"),
    ("5/1/2025", "45.658,30"),
    ("6/1/2025", "49.974,05"),
    ("7/1/2025", "51.907,51"),
    ("8/1/2025", "53.784,00"),
    ("9/1/2025", "56.203,00"),
    ("10/1/2025", "59.761,00"),
    ("11/1/2025", "64.610,00"),
    ("12/1/2025", "67.019,00"),
    ("1/1/2026", "68.838,09"),
    ("2/1/2026", "74.094,00"),
]


def transition(next_state: str, reason: str) -> None:
    if st.session_state.get("fsm_state") != next_state:
        st.session_state["fsm_state"] = next_state
        st.session_state["workflow"].setdefault("nav", {})["last_transition"] = {
            "state": next_state,
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


def run() -> None:
    st.set_page_config(page_title="DEGIRO Reconciliation Dashboard", layout="wide")
    logger = setup_logger()
    ensure_session_state()

    st.title("DEGIRO Reconciliation Dashboard")
    apply_startup_autoload(logger=logger)
    render_sidebar(logger=logger)
    render_main(logger=logger)


def ensure_session_state() -> None:
    st.session_state.setdefault(
        "workflow",
        {
            "datasets": {},
            "processed": {},
            "summaries": {},
            "warnings": [],
            "nav": {},
        },
    )
    st.session_state.setdefault("fsm_state", STATE_LOADING_DATA)
    st.session_state.setdefault("upload_sig", tuple())
    st.session_state.setdefault("params_sig", "")
    st.session_state.setdefault("processed_sig", None)
    st.session_state.setdefault("export_bytes", None)
    st.session_state["workflow"].setdefault("reported_values", {})
    st.session_state.setdefault("dataset_a_reported_cash_text", "2.24   ")
    st.session_state.setdefault("dataset_a_reported_total_text", "39889.00")
    st.session_state.setdefault("dataset_b_reported_cash_text", "19.40")
    st.session_state.setdefault("dataset_b_reported_total_text", "36425.00")
    st.session_state.setdefault("startup_autoload_applied", False)
    st.session_state.setdefault("startup_autorun_pending", False)
    st.session_state.setdefault("strategy_autoload_applied", False)
    st.session_state.setdefault("strategy_load_requested_path", "")
    st.session_state.setdefault("strategy_load_notice", "")
    st.session_state.setdefault("strategy_load_notice_level", "info")


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_startup_path(raw: str | None, fallback: str | Path) -> Path:
    candidate = str(raw).strip() if raw is not None else ""
    if candidate == "":
        candidate = str(fallback)
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_startup_strategy_payload(strategy_file: Path, *, logger: Any) -> dict[str, Any]:
    if not strategy_file.exists():
        return {}
    try:
        payload = json.loads(strategy_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        logger.exception("Failed reading startup strategy file: %s", strategy_file)
        return {}


def _to_float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_target_pct_map_local(value: Any, *, default: dict[str, float]) -> dict[str, float]:
    source = value if isinstance(value, dict) else default
    out: dict[str, float] = {}
    for raw_key, raw_pct in source.items():
        key = str(raw_key).strip()
        if key == "":
            continue
        pct = pd.to_numeric(raw_pct, errors="coerce")
        if pd.notna(pct):
            out[key] = max(float(pct), 0.0)
    return out


def _normalize_holding_category_overrides_local(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for raw_instrument_id, raw_entry in value.items():
        instrument_id = str(raw_instrument_id).strip()
        if instrument_id == "" or not isinstance(raw_entry, dict):
            continue
        style = str(raw_entry.get("style", "")).strip()
        industry = str(raw_entry.get("industry", "")).strip()
        entry: dict[str, str] = {}
        if style:
            entry["style"] = style
        if industry:
            entry["industry"] = industry
        if entry:
            out[instrument_id] = entry
    return out


def _target_pct_seed_df(map_data: dict[str, float]) -> pd.DataFrame:
    rows = [{"bucket": str(k), "target_pct": float(v)} for k, v in (map_data or {}).items()]
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["bucket", "target_pct"])


def load_strategy_file(
    *,
    strategy_file_path: str,
    logger: Any,
) -> tuple[dict[str, Any], dict[str, str], Path]:
    raw = str(strategy_file_path).strip()
    if raw == "":
        raise ValueError("strategy_file_path cannot be empty.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists() or not path.is_file():
        raise ValueError(f"Strategy file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Strategy file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Strategy file root must be an object: {path}")

    raw_strategy = payload.get("strategy", {})
    if not isinstance(raw_strategy, dict):
        raw_strategy = {}

    strategy = {
        "target_etf_fraction": _to_float_or_default(
            raw_strategy.get("target_etf_fraction"),
            DEFAULT_TARGET_ETF_FRACTION,
        ),
        "desired_etf_holdings": _to_int_or_default(
            raw_strategy.get("desired_etf_holdings"),
            DEFAULT_DESIRED_ETF_HOLDINGS,
        ),
        "desired_non_etf_holdings": _to_int_or_default(
            raw_strategy.get("desired_non_etf_holdings"),
            DEFAULT_DESIRED_NON_ETF_HOLDINGS,
        ),
        "target_cash_pct": _to_float_or_default(raw_strategy.get("target_cash_pct"), DEFAULT_TARGET_CASH_PCT),
        "max_single_holding_pct": _to_float_or_default(
            raw_strategy.get("max_single_holding_pct"),
            DEFAULT_MAX_SINGLE_HOLDING_PCT,
        ),
        "max_top5_holdings_pct": _to_float_or_default(
            raw_strategy.get("max_top5_holdings_pct"),
            DEFAULT_MAX_TOP5_HOLDINGS_PCT,
        ),
        "max_single_currency_pct": _to_float_or_default(
            raw_strategy.get("max_single_currency_pct"),
            DEFAULT_MAX_SINGLE_CURRENCY_PCT,
        ),
        "max_single_industry_pct": _to_float_or_default(
            raw_strategy.get("max_single_industry_pct"),
            DEFAULT_MAX_SINGLE_INDUSTRY_PCT,
        ),
        "max_pair_correlation": _to_float_or_default(
            raw_strategy.get("max_pair_correlation"),
            DEFAULT_MAX_PAIR_CORRELATION,
        ),
        "min_total_holdings": _to_int_or_default(
            raw_strategy.get("min_total_holdings"),
            DEFAULT_MIN_TOTAL_HOLDINGS,
        ),
        "min_over_value_eur": _to_float_or_default(raw_strategy.get("min_over_value_eur"), DEFAULT_MIN_OVER_VALUE_EUR),
        "target_currency_pct": _normalize_target_pct_map_local(
            raw_strategy.get("target_currency_pct"),
            default=DEFAULT_TARGET_CURRENCY_PCT,
        ),
        "target_industry_pct": _normalize_target_pct_map_local(
            raw_strategy.get("target_industry_pct"),
            default=DEFAULT_TARGET_INDUSTRY_PCT,
        ),
        "target_style_pct": _normalize_target_pct_map_local(
            raw_strategy.get("target_style_pct"),
            default=DEFAULT_TARGET_STYLE_PCT,
        ),
        "holding_category_overrides": _normalize_holding_category_overrides_local(
            raw_strategy.get("holding_category_overrides")
        ),
    }

    raw_sources = payload.get("data_sources", {})
    if not isinstance(raw_sources, dict):
        raw_sources = {}
    data_sources = {
        "dataset_a_dir": str(raw_sources.get("dataset_a_dir", DEFAULT_STRATEGY_DATASET_A_DIR)),
        "dataset_b_dir": str(raw_sources.get("dataset_b_dir", DEFAULT_STRATEGY_DATASET_B_DIR)),
        "mappings_path": str(raw_sources.get("mappings_path", "mappings.yml")),
    }

    if logger is not None:
        logger.info("Loaded strategy file: %s", str(path))
    return strategy, data_sources, path


def apply_loaded_strategy_to_session_state(
    *,
    strategy_file_path: str,
    strategy: dict[str, Any],
    data_sources: dict[str, str],
) -> None:
    scalar_keys = [
        "target_etf_fraction",
        "desired_etf_holdings",
        "desired_non_etf_holdings",
        "target_cash_pct",
        "max_single_holding_pct",
        "max_top5_holdings_pct",
        "max_single_currency_pct",
        "max_single_industry_pct",
        "max_pair_correlation",
        "min_total_holdings",
        "min_over_value_eur",
    ]
    for key in scalar_keys:
        if key in strategy:
            st.session_state[key] = strategy[key]

    st.session_state["strategy_file_path"] = str(strategy_file_path)
    st.session_state["strategy_dataset_a_dir"] = str(
        data_sources.get("dataset_a_dir", DEFAULT_STRATEGY_DATASET_A_DIR)
    )
    st.session_state["strategy_dataset_b_dir"] = str(
        data_sources.get("dataset_b_dir", DEFAULT_STRATEGY_DATASET_B_DIR)
    )
    st.session_state["strategy_mappings_path"] = str(data_sources.get("mappings_path", "mappings.yml"))

    st.session_state["target_currency_pct_editor_seed"] = _target_pct_seed_df(
        strategy.get("target_currency_pct", DEFAULT_TARGET_CURRENCY_PCT)
    )
    st.session_state["target_industry_pct_editor_seed"] = _target_pct_seed_df(
        strategy.get("target_industry_pct", DEFAULT_TARGET_INDUSTRY_PCT)
    )
    st.session_state["target_style_pct_editor_seed"] = _target_pct_seed_df(
        strategy.get("target_style_pct", DEFAULT_TARGET_STYLE_PCT)
    )
    st.session_state.pop("target_currency_pct_editor", None)
    st.session_state.pop("target_industry_pct_editor", None)
    st.session_state.pop("target_style_pct_editor", None)

    st.session_state["loaded_holding_category_overrides"] = strategy.get("holding_category_overrides", {})
    st.session_state.pop("holding_override_editor_sig", None)

    st.session_state["workflow"]["processed"] = {}
    st.session_state["workflow"]["summaries"] = {}
    st.session_state["processed_sig"] = None
    st.session_state["export_bytes"] = None


def process_strategy_load_requests(*, logger: Any) -> None:
    requested_path = str(st.session_state.pop("strategy_load_requested_path", "")).strip()
    if requested_path == "" and not st.session_state.get("strategy_autoload_applied", False):
        st.session_state["strategy_autoload_applied"] = True
        default_path = Path(DEFAULT_STRATEGY_FILE_PATH)
        if default_path.is_file():
            requested_path = str(default_path)

    if requested_path == "":
        return

    try:
        strategy, data_sources, resolved_path = load_strategy_file(
            strategy_file_path=requested_path,
            logger=logger,
        )
        apply_loaded_strategy_to_session_state(
            strategy_file_path=str(resolved_path),
            strategy=strategy,
            data_sources=data_sources,
        )
        transition(STATE_SELECTING_PARAMS, "Strategy loaded from file")
        st.session_state["strategy_load_notice_level"] = "success"
        st.session_state["strategy_load_notice"] = f"Strategy loaded: {resolved_path}"
    except Exception as exc:
        st.session_state["strategy_load_notice_level"] = "error"
        st.session_state["strategy_load_notice"] = f"Failed to load strategy: {exc}"


def _has_required_dataset_files(folder: Path) -> bool:
    return all((folder / name).is_file() for name in REQUIRED_DATASET_FILES)


def _autoload_dataset_from_folder(
    *,
    panel_key: str,
    panel_title: str,
    folder: Path,
    mappings_path: Path,
    logger: Any,
) -> bool:
    if not _has_required_dataset_files(folder):
        return False
    transactions_path = folder / "Transactions.csv"
    portfolio_path = folder / "Portfolio.csv"
    account_path = folder / "Account.csv"
    start = time.perf_counter()
    dataset = load_dataset(
        account_label=f"{panel_title} ({folder.name})",
        transactions_source=transactions_path,
        portfolio_source=portfolio_path,
        account_source=account_path,
        mappings_path=mappings_path,
    )
    st.session_state["workflow"]["datasets"][panel_key] = {
        "dataset": dataset,
        "source_sig": (
            "startup_dir",
            str(folder),
            transactions_path.stat().st_size,
            portfolio_path.stat().st_size,
            account_path.stat().st_size,
        ),
    }
    log(f"Startup auto-loaded {panel_title} from {folder}", start, logger)
    return True


def apply_startup_autoload(*, logger: Any) -> None:
    if st.session_state.get("startup_autoload_applied"):
        return
    st.session_state["startup_autoload_applied"] = True

    if not _env_flag("DEGIRO_STARTUP_AUTORUN"):
        return
    if st.session_state["workflow"]["datasets"]:
        st.session_state["startup_autorun_pending"] = True
        return

    strategy_file = _resolve_startup_path(
        os.getenv("DEGIRO_STARTUP_STRATEGY_FILE"),
        DEFAULT_STRATEGY_FILE_PATH,
    )
    strategy_payload = _load_startup_strategy_payload(strategy_file, logger=logger)
    data_sources = (
        strategy_payload.get("data_sources", {})
        if isinstance(strategy_payload.get("data_sources"), dict)
        else {}
    )
    dataset_a_dir = _resolve_startup_path(
        os.getenv("DEGIRO_STARTUP_DATASET_A_DIR"),
        data_sources.get("dataset_a_dir", DEFAULT_STRATEGY_DATASET_A_DIR),
    )
    dataset_b_dir = _resolve_startup_path(
        os.getenv("DEGIRO_STARTUP_DATASET_B_DIR"),
        data_sources.get("dataset_b_dir", DEFAULT_STRATEGY_DATASET_B_DIR),
    )
    mappings_path = _resolve_startup_path(
        os.getenv("DEGIRO_STARTUP_MAPPINGS_PATH"),
        data_sources.get("mappings_path", "mappings.yml"),
    )

    loaded_count = 0
    load_errors: list[str] = []
    for panel_key, panel_title, folder in [
        ("dataset_a", "Dataset A", dataset_a_dir),
        ("dataset_b", "Dataset B", dataset_b_dir),
    ]:
        try:
            if _autoload_dataset_from_folder(
                panel_key=panel_key,
                panel_title=panel_title,
                folder=folder,
                mappings_path=mappings_path,
                logger=logger,
            ):
                loaded_count += 1
        except UserFacingError as exc:
            load_errors.append(f"{panel_title}: {exc.short_message}")
            logger.exception("Startup auto-load user-facing error for %s", panel_key)
        except Exception:
            load_errors.append(f"{panel_title}: unexpected load failure")
            logger.exception("Startup auto-load unexpected error for %s", panel_key)

    if load_errors:
        st.warning("Startup auto-load errors: " + " | ".join(load_errors))
    if loaded_count == 0:
        st.warning(
            "Startup auto-load is enabled, but no dataset folder had all required files "
            "(Transactions.csv, Portfolio.csv, Account.csv)."
        )
        return

    st.session_state["workflow"]["processed"] = {}
    st.session_state["workflow"]["summaries"] = {}
    st.session_state["processed_sig"] = None
    st.session_state["export_bytes"] = None
    st.session_state["upload_sig"] = current_upload_signature()
    st.session_state["startup_autorun_pending"] = True
    transition(STATE_SELECTING_PARAMS, "Startup auto-loaded dataset(s)")
    st.info(f"Startup auto-loaded {loaded_count} dataset(s). Running analysis automatically.")


def at_or_after(target_state: str) -> bool:
    current = st.session_state.get("fsm_state", STATE_LOADING_DATA)
    idx_current = STATE_ORDER.index(current) if current in STATE_ORDER else 0
    idx_target = STATE_ORDER.index(target_state) if target_state in STATE_ORDER else 0
    return idx_current >= idx_target


def render_sidebar(*, logger: Any) -> None:
    # AGENT_NOTE: Sidebar fields define the strategy/processing param schema.
    # Keep parameter keys and defaults aligned with `src/strategy_check.py`
    # (`_strategy_with_defaults`) and persisted strategy JSON.
    process_strategy_load_requests(logger=logger)
    notice = str(st.session_state.pop("strategy_load_notice", "")).strip()
    notice_level = str(st.session_state.pop("strategy_load_notice_level", "info")).strip().lower()
    if notice != "":
        if notice_level == "error":
            st.sidebar.error(notice)
        elif notice_level == "success":
            st.sidebar.success(notice)
        else:
            st.sidebar.info(notice)

    st.sidebar.header("Data Selection")
    datasets_before = len(st.session_state["workflow"]["datasets"])
    changed = False
    changed |= render_dataset_panel(
        panel_key="dataset_a",
        panel_title="Dataset A",
        default_sample="pensioenbeleggen",
        logger=logger,
    )
    changed |= render_dataset_panel(
        panel_key="dataset_b",
        panel_title="Dataset B",
        default_sample="spaarbeleggen",
        logger=logger,
    )
    datasets_after = len(st.session_state["workflow"]["datasets"])
    second_dataset_just_loaded = changed and datasets_before < 2 <= datasets_after

    render_reported_values_form()

    if changed:
        st.session_state["workflow"]["processed"] = {}
        st.session_state["workflow"]["summaries"] = {}
        st.session_state["export_bytes"] = None
        st.session_state["processed_sig"] = None
        transition(STATE_SELECTING_PARAMS, "Dataset content changed")

    upload_sig = current_upload_signature()
    if upload_sig != st.session_state.get("upload_sig"):
        st.session_state["upload_sig"] = upload_sig
        st.session_state["processed_sig"] = None
        st.session_state["workflow"]["processed"] = {}
        st.session_state["export_bytes"] = None
        if not upload_sig:
            transition(STATE_LOADING_DATA, "No datasets loaded")
        else:
            transition(STATE_SELECTING_PARAMS, "Upload signature changed")

    st.sidebar.header("Analysis Parameters")
    lookback_months = st.sidebar.number_input(
        "lookback_months",
        min_value=1,
        max_value=240,
        value=DEFAULT_LOOKBACK_MONTHS,
        step=1,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help=(
            "How many months of historical data are included in returns, charts, "
            "and diagnostics."
        ),
    )
    median_window_months = st.sidebar.number_input(
        "median_window_months",
        min_value=1,
        max_value=60,
        value=DEFAULT_MEDIAN_WINDOW_MONTHS,
        step=1,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help=(
            "Rolling median window (months) used in normalized-price and "
            "deviation views."
        ),
    )
    target_etf_fraction = st.sidebar.slider(
        "target_etf_fraction",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("target_etf_fraction", DEFAULT_TARGET_ETF_FRACTION)),
        step=0.01,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help=(
            "Target fraction of invested holdings in ETFs (0.50 means 50% ETFs, "
            "50% non-ETFs)."
        ),
    )
    desired_etf_holdings = st.sidebar.number_input(
        "desired_etf_holdings",
        min_value=1,
        max_value=200,
        value=int(st.session_state.get("desired_etf_holdings", DEFAULT_DESIRED_ETF_HOLDINGS)),
        step=1,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Target number of ETF positions to hold.",
    )
    desired_non_etf_holdings = st.sidebar.number_input(
        "desired_non_etf_holdings",
        min_value=1,
        max_value=500,
        value=int(st.session_state.get("desired_non_etf_holdings", DEFAULT_DESIRED_NON_ETF_HOLDINGS)),
        step=1,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Target number of non-ETF positions to hold.",
    )
    min_over_value_eur = st.sidebar.number_input(
        "min_over_value_eur",
        min_value=0.0,
        value=float(st.session_state.get("min_over_value_eur", DEFAULT_MIN_OVER_VALUE_EUR)),
        step=50.0,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help=(
            "Minimum EUR deviation before an overweight/underweight position is "
            "flagged for action."
        ),
    )
    st.sidebar.header("Spread Strategy")
    target_cash_pct = st.sidebar.slider(
        "target_cash_pct",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state.get("target_cash_pct", DEFAULT_TARGET_CASH_PCT)),
        step=0.5,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Target cash percentage of total portfolio value.",
    )
    max_single_holding_pct = st.sidebar.slider(
        "max_single_holding_pct",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.get("max_single_holding_pct", DEFAULT_MAX_SINGLE_HOLDING_PCT)),
        step=0.5,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help=(
            "Maximum allowed allocation in one holding. "
            "If a single position exceeds this percentage, concentration warnings "
            "and action suggestions are generated."
        ),
    )
    max_top5_holdings_pct = st.sidebar.slider(
        "max_top5_holdings_pct",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.get("max_top5_holdings_pct", DEFAULT_MAX_TOP5_HOLDINGS_PCT)),
        step=0.5,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Maximum combined allocation of the 5 largest holdings.",
    )
    max_single_currency_pct = st.sidebar.slider(
        "max_single_currency_pct",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.get("max_single_currency_pct", DEFAULT_MAX_SINGLE_CURRENCY_PCT)),
        step=0.5,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Maximum allowed allocation in a single currency bucket.",
    )
    max_single_industry_pct = st.sidebar.slider(
        "max_single_industry_pct",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.get("max_single_industry_pct", DEFAULT_MAX_SINGLE_INDUSTRY_PCT)),
        step=0.5,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Maximum allowed allocation in a single industry bucket.",
    )
    max_pair_correlation = st.sidebar.slider(
        "max_pair_correlation",
        min_value=0.50,
        max_value=0.99,
        value=float(st.session_state.get("max_pair_correlation", DEFAULT_MAX_PAIR_CORRELATION)),
        step=0.01,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help=(
            "Warn when the rolling return correlation between two holdings is "
            "above this threshold."
        ),
    )
    min_total_holdings = st.sidebar.number_input(
        "min_total_holdings",
        min_value=1,
        max_value=500,
        value=int(st.session_state.get("min_total_holdings", DEFAULT_MIN_TOTAL_HOLDINGS)),
        step=1,
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Minimum total number of holdings required by the diversification rules.",
    )
    with st.sidebar.expander("Target allocation maps (%)", expanded=False):
        target_currency_pct = target_pct_map_editor(
            editor_key="target_currency_pct_editor",
            default_map=DEFAULT_TARGET_CURRENCY_PCT,
            disabled=not at_or_after(STATE_SELECTING_PARAMS),
            bucket_label="currency",
        )
        target_industry_pct = target_pct_map_editor(
            editor_key="target_industry_pct_editor",
            default_map=DEFAULT_TARGET_INDUSTRY_PCT,
            disabled=not at_or_after(STATE_SELECTING_PARAMS),
            bucket_label="industry",
        )
        target_style_pct = target_pct_map_editor(
            editor_key="target_style_pct_editor",
            default_map=DEFAULT_TARGET_STYLE_PCT,
            disabled=not at_or_after(STATE_SELECTING_PARAMS),
            bucket_label="style",
        )
    with st.sidebar.expander("Holding category overrides", expanded=False):
        holding_category_overrides = holding_category_overrides_editor(
            datasets=st.session_state["workflow"]["datasets"],
            disabled=not at_or_after(STATE_SELECTING_PARAMS),
        )
    strategy_file_path = st.sidebar.text_input(
        "strategy_file_path",
        value=str(st.session_state.get("strategy_file_path", DEFAULT_STRATEGY_FILE_PATH)),
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="JSON file used by the strategy-check script.",
    )
    strategy_dataset_a_dir = st.sidebar.text_input(
        "strategy_dataset_a_dir",
        value=str(st.session_state.get("strategy_dataset_a_dir", DEFAULT_STRATEGY_DATASET_A_DIR)),
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Folder expected to contain Dataset A exports (Transactions.csv, Portfolio.csv, Account.csv).",
    )
    strategy_dataset_b_dir = st.sidebar.text_input(
        "strategy_dataset_b_dir",
        value=str(st.session_state.get("strategy_dataset_b_dir", DEFAULT_STRATEGY_DATASET_B_DIR)),
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Folder expected to contain Dataset B exports (Transactions.csv, Portfolio.csv, Account.csv).",
    )
    strategy_mappings_path = st.sidebar.text_input(
        "strategy_mappings_path",
        value=str(st.session_state.get("strategy_mappings_path", "mappings.yml")),
        disabled=not at_or_after(STATE_SELECTING_PARAMS),
        help="Mappings file path used by both apps.",
    )

    params = {
        "lookback_months": int(lookback_months),
        "median_window_months": int(median_window_months),
        "target_etf_fraction": float(target_etf_fraction),
        "desired_etf_holdings": int(desired_etf_holdings),
        "desired_non_etf_holdings": int(desired_non_etf_holdings),
        "min_over_value_eur": float(min_over_value_eur),
        "target_cash_pct": float(target_cash_pct),
        "max_single_holding_pct": float(max_single_holding_pct),
        "max_top5_holdings_pct": float(max_top5_holdings_pct),
        "max_single_currency_pct": float(max_single_currency_pct),
        "max_single_industry_pct": float(max_single_industry_pct),
        "max_pair_correlation": float(max_pair_correlation),
        "min_total_holdings": int(min_total_holdings),
        "target_currency_pct": target_currency_pct,
        "target_industry_pct": target_industry_pct,
        "target_style_pct": target_style_pct,
        "holding_category_overrides": holding_category_overrides,
    }
    strategy_action_cols = st.sidebar.columns(2)
    if strategy_action_cols[0].button("Load strategy", disabled=not at_or_after(STATE_SELECTING_PARAMS)):
        st.session_state["strategy_load_requested_path"] = str(strategy_file_path)
        st.rerun()

    if strategy_action_cols[1].button("Save strategy", disabled=not at_or_after(STATE_SELECTING_PARAMS)):
        try:
            strategy_save_path = save_strategy_file(
                strategy_file_path=strategy_file_path,
                strategy={
                    "target_etf_fraction": float(params["target_etf_fraction"]),
                    "desired_etf_holdings": int(params["desired_etf_holdings"]),
                    "desired_non_etf_holdings": int(params["desired_non_etf_holdings"]),
                    "target_cash_pct": float(params["target_cash_pct"]),
                    "max_single_holding_pct": float(params["max_single_holding_pct"]),
                    "max_top5_holdings_pct": float(params["max_top5_holdings_pct"]),
                    "max_single_currency_pct": float(params["max_single_currency_pct"]),
                    "max_single_industry_pct": float(params["max_single_industry_pct"]),
                    "max_pair_correlation": float(params["max_pair_correlation"]),
                    "min_total_holdings": int(params["min_total_holdings"]),
                    "min_over_value_eur": float(params["min_over_value_eur"]),
                    "target_currency_pct": params["target_currency_pct"],
                    "target_industry_pct": params["target_industry_pct"],
                    "target_style_pct": params["target_style_pct"],
                    "holding_category_overrides": params["holding_category_overrides"],
                },
                data_sources={
                    "dataset_a_dir": strategy_dataset_a_dir,
                    "dataset_b_dir": strategy_dataset_b_dir,
                    "mappings_path": strategy_mappings_path,
                },
                logger=logger,
            )
            st.sidebar.success(f"Strategy saved: {strategy_save_path}")
        except Exception as exc:
            st.sidebar.error(f"Failed to save strategy: {exc}")

    params_sig = json.dumps(params, sort_keys=True)
    if params_sig != st.session_state.get("params_sig"):
        st.session_state["params_sig"] = params_sig
        st.session_state["workflow"]["processed"] = {}
        st.session_state["processed_sig"] = None
        st.session_state["export_bytes"] = None

    auto_ran = False
    startup_autorun_pending = bool(st.session_state.pop("startup_autorun_pending", False))
    if startup_autorun_pending and at_or_after(STATE_SELECTING_PARAMS):
        transition(STATE_PROCESSING, "Startup auto-load requested immediate analysis")
        run_analysis(params=params, logger=logger)
        auto_ran = True

    if second_dataset_just_loaded and at_or_after(STATE_SELECTING_PARAMS) and not auto_ran:
        transition(STATE_PROCESSING, "Second dataset loaded; auto-run analysis")
        run_analysis(params=params, logger=logger)
        auto_ran = True

    datasets_loaded = len(st.session_state["workflow"]["datasets"]) > 0
    run_disabled = not datasets_loaded or not at_or_after(STATE_SELECTING_PARAMS)
    if st.sidebar.button("Run analysis", disabled=run_disabled) and not auto_ran:
        transition(STATE_PROCESSING, "User clicked Run analysis")
        run_analysis(params=params, logger=logger)


def render_reported_values_form() -> None:
    workflow = st.session_state["workflow"]
    with st.sidebar.expander("DEGIRO App Reference Values", expanded=False):
        st.caption(
            "Optional bonus check: enter values from the DEGIRO app for each loaded dataset:\n"
            "- cash total (EUR + foreign currency converted)\n"
            "- total portfolio value (EUR)"
        )
        panel_specs = [("dataset_a", "Dataset A"), ("dataset_b", "Dataset B")]
        parsed: dict[str, dict[str, float]] = {}
        parse_errors: list[str] = []
        notes: list[str] = []

        for panel_key, panel_title in panel_specs:
            loaded = panel_key in workflow["datasets"]
            st.markdown(f"**{panel_title}**")
            if not loaded:
                st.caption("Load this dataset first to enable manual reference input.")
            manual_enabled = st.checkbox(
                "Use manual DEGIRO app values (optional)",
                key=f"{panel_key}_reported_enabled",
                disabled=not loaded,
                help="Enable this to compare computed values with values manually copied from the DEGIRO app.",
            )
            st.text_input(
                "Reported cash total (EUR)",
                key=f"{panel_key}_reported_cash_text",
                disabled=(not loaded or not manual_enabled),
                help=(
                    "Disabled until the dataset is loaded and "
                    "'Use manual DEGIRO app values (optional)' is checked."
                ),
            )
            st.text_input(
                "Reported portfolio total (EUR)",
                key=f"{panel_key}_reported_total_text",
                disabled=(not loaded or not manual_enabled),
                help=(
                    "Disabled until the dataset is loaded and "
                    "'Use manual DEGIRO app values (optional)' is checked."
                ),
            )
            if not loaded or not manual_enabled:
                continue

            cash_text = str(st.session_state.get(f"{panel_key}_reported_cash_text", "")).strip()
            total_text = str(st.session_state.get(f"{panel_key}_reported_total_text", "")).strip()

            if cash_text == "" and total_text == "":
                notes.append(f"{panel_title}: manual mode enabled but both values are empty.")
                continue

            cash = parse_user_float(cash_text) if cash_text != "" else None
            total = parse_user_float(total_text) if total_text != "" else None
            if cash is None:
                parse_errors.append(f"{panel_title}: invalid cash value `{cash_text}`.")
            if total is None:
                parse_errors.append(f"{panel_title}: invalid total value `{total_text}`.")
            if cash is not None and total is not None:
                parsed[panel_key] = {
                    "reported_cash_eur": cash,
                    "reported_total_eur": total,
                }

        workflow["reported_values"] = parsed
        if parse_errors:
            st.error("\n".join(parse_errors))
        elif notes:
            st.info("\n".join(notes))


def render_dataset_panel(
    *,
    panel_key: str,
    panel_title: str,
    default_sample: str,
    logger: Any,
) -> bool:
    changed = False
    panel_expanded_key = f"{panel_key}_sidebar_expanded"
    if panel_expanded_key not in st.session_state:
        st.session_state[panel_expanded_key] = True
    with st.sidebar.expander(panel_title, expanded=bool(st.session_state.get(panel_expanded_key, True))):
        use_sample = st.toggle(
            "Use bundled sample data",
            value=True,
            key=f"{panel_key}_use_sample",
        )
        if use_sample:
            sample_choice = st.selectbox(
                "Sample dataset",
                options=list(SAMPLE_DATASETS.keys()),
                index=list(SAMPLE_DATASETS.keys()).index(default_sample),
                key=f"{panel_key}_sample_choice",
            )
            uploaded_files = None
        else:
            sample_choice = None
            uploaded_files = st.file_uploader(
                "Upload Transactions.csv, Portfolio.csv, Account.csv",
                type=["csv"],
                accept_multiple_files=True,
                key=f"{panel_key}_uploader",
            )

        col1, col2 = st.columns(2)
        load_clicked = col1.button("Load", key=f"{panel_key}_load_btn")
        unload_clicked = col2.button("Unload", key=f"{panel_key}_unload_btn")

        if unload_clicked:
            if panel_key in st.session_state["workflow"]["datasets"]:
                del st.session_state["workflow"]["datasets"][panel_key]
                changed = True
            st.session_state[panel_expanded_key] = True

        if load_clicked:
            try:
                start = time.perf_counter()
                if use_sample:
                    sample_dir = SAMPLE_DATASETS[sample_choice]
                    dataset = load_dataset(
                        account_label=f"{panel_title} ({sample_choice})",
                        transactions_source=sample_dir / "Transactions.csv",
                        portfolio_source=sample_dir / "Portfolio.csv",
                        account_source=sample_dir / "Account.csv",
                        mappings_path="mappings.yml",
                    )
                    source_sig = (
                        "sample",
                        sample_choice,
                        (sample_dir / "Transactions.csv").stat().st_size,
                        (sample_dir / "Portfolio.csv").stat().st_size,
                        (sample_dir / "Account.csv").stat().st_size,
                    )
                else:
                    if not uploaded_files:
                        raise UserFacingError(
                            f"{panel_title}: no files uploaded.",
                            "Switch to upload mode and select all required CSV files.",
                        )
                    by_name = validate_uploaded_file_set(uploaded_files)
                    dataset = load_dataset(
                        account_label=f"{panel_title} (uploaded)",
                        transactions_source=by_name["Transactions.csv"],
                        portfolio_source=by_name["Portfolio.csv"],
                        account_source=by_name["Account.csv"],
                        mappings_path="mappings.yml",
                    )
                    source_sig = (
                        "upload",
                        tuple(sorted((f.name, f.size) for f in uploaded_files)),
                    )

                st.session_state["workflow"]["datasets"][panel_key] = {
                    "dataset": dataset,
                    "source_sig": source_sig,
                }
                changed = True
                st.session_state[panel_expanded_key] = False
                log(f"Loaded {panel_title}", start, logger)
                st.success(f"{panel_title} loaded.")
            except UserFacingError as exc:
                st.error(exc.to_ui_text())
                logger.exception("User-facing load error in %s", panel_title)
            except Exception:
                st.error(
                    f"{panel_title} failed to load. Check logs/app.log for technical details."
                )
                logger.exception("Unexpected load error in %s", panel_title)

        if panel_key in st.session_state["workflow"]["datasets"]:
            loaded = st.session_state["workflow"]["datasets"][panel_key]["dataset"]
            st.caption(
                f"Loaded: {loaded.account_label} | "
                f"Tx: {len(loaded.transactions)} | "
                f"Portfolio: {len(loaded.portfolio)} | "
                f"Account: {len(loaded.account)}"
            )
    return changed


def current_upload_signature() -> tuple[Any, ...]:
    rows: list[Any] = []
    for key in sorted(st.session_state["workflow"]["datasets"].keys()):
        row = st.session_state["workflow"]["datasets"][key]
        rows.append((key, row.get("source_sig")))
    return tuple(rows)


def target_pct_map_editor(
    *,
    editor_key: str,
    default_map: dict[str, float],
    disabled: bool,
    bucket_label: str,
) -> dict[str, float]:
    # AGENT_NOTE: Output map is persisted in strategy JSON and reused by the
    # CLI strategy checker. Keep bucket normalization behavior stable.
    seed_key = f"{editor_key}_seed"
    if seed_key not in st.session_state:
        seed_rows = [{"bucket": str(k), "target_pct": float(v)} for k, v in (default_map or {}).items()]
        if seed_rows:
            st.session_state[seed_key] = pd.DataFrame(seed_rows)
        else:
            st.session_state[seed_key] = pd.DataFrame(columns=["bucket", "target_pct"])
    edited = st.data_editor(
        st.session_state[seed_key],
        key=editor_key,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        disabled=disabled,
        column_config={
            "bucket": st.column_config.TextColumn(
                bucket_label,
                required=False,
                help=f"Bucket name for {bucket_label} allocation target.",
            ),
            "target_pct": st.column_config.NumberColumn(
                "target_pct",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                format="%.2f",
                help="Target percentage for this bucket.",
            ),
        },
    )
    out: dict[str, float] = {}
    if isinstance(edited, pd.DataFrame):
        for row in edited.itertuples(index=False):
            bucket = str(getattr(row, "bucket", "")).strip()
            if bucket_label == "currency":
                bucket = bucket.upper()
            pct = pd.to_numeric(getattr(row, "target_pct", np.nan), errors="coerce")
            if bucket and pd.notna(pct):
                out[bucket] = max(float(pct), 0.0)
    return out


def holding_category_overrides_editor(
    *,
    datasets: dict[str, dict[str, Any]],
    disabled: bool,
) -> dict[str, dict[str, str]]:
    # AGENT_NOTE: Overrides are keyed by instrument_id and feed directly into
    # `insights.build_ai_spread_analysis()` for final style/industry bucketing.
    instruments_rows: list[dict[str, Any]] = []
    for row in datasets.values():
        ds = row.get("dataset")
        if ds is None:
            continue
        instruments = getattr(ds, "instruments", pd.DataFrame())
        if not isinstance(instruments, pd.DataFrame) or instruments.empty:
            continue
        if "is_cash_like" in instruments.columns:
            cash_mask = instruments["is_cash_like"].fillna(False).astype(bool)
        else:
            cash_mask = pd.Series(False, index=instruments.index, dtype=bool)
        cur = instruments.loc[~cash_mask].copy()
        if cur.empty:
            continue
        for item in cur.itertuples(index=False):
            instrument_id = str(getattr(item, "instrument_id", "")).strip()
            if instrument_id == "":
                continue
            product = str(getattr(item, "product", "")).strip()
            ticker = str(getattr(item, "ticker", "")).strip()
            is_etf = bool(getattr(item, "is_etf", False))
            instruments_rows.append(
                {
                    "instrument_id": instrument_id,
                    "product": product,
                    "ticker": ticker,
                    "is_etf": is_etf,
                }
            )
    if not instruments_rows:
        st.caption("Load at least one dataset to edit category overrides.")
        return {}

    base = (
        pd.DataFrame(instruments_rows)
        .drop_duplicates(subset=["instrument_id"], keep="first")
        .sort_values("product")
        .reset_index(drop=True)
    )
    base, characteristics_stats = resolve_ticker_characteristics(
        instruments_df=base,
        ticker_classifications_path=Path("ticker_classifications.csv"),
        auto_append_missing=True,
    )
    appended_count = int(characteristics_stats.get("appended_count", 0))
    if appended_count > 0:
        st.caption(
            f"Updated ticker characteristics: appended {appended_count} new instrument(s) to "
            "`ticker_classifications.csv` with `t.b.d.` placeholders."
        )
    base = base[["instrument_id", "ticker", "product", "auto_style", "style", "auto_industry", "industry"]].copy()

    seed_sig = tuple(base["instrument_id"].astype(str).tolist())
    if st.session_state.get("holding_override_editor_sig") != seed_sig:
        prev_df = st.session_state.get("holding_override_editor_df")
        prev_map: dict[str, dict[str, str]] = {}
        if isinstance(prev_df, pd.DataFrame) and not prev_df.empty:
            for row in prev_df.itertuples(index=False):
                key = str(getattr(row, "instrument_id", "")).strip()
                if key == "":
                    continue
                prev_map[key] = {
                    "style": str(getattr(row, "style", "")).strip(),
                    "industry": str(getattr(row, "industry", "")).strip(),
                }
        loaded_overrides = st.session_state.pop("loaded_holding_category_overrides", None)
        if isinstance(loaded_overrides, dict):
            for instrument_id, entry in loaded_overrides.items():
                key = str(instrument_id).strip()
                if key == "" or not isinstance(entry, dict):
                    continue
                prev_map[key] = {
                    "style": str(entry.get("style", "")).strip(),
                    "industry": str(entry.get("industry", "")).strip(),
                }
        default_style = base["style"].fillna("").astype(str)
        default_industry = base["industry"].fillna("").astype(str)
        base["style"] = base["instrument_id"].map(
            lambda k: prev_map.get(str(k), {}).get("style", "")
        )
        base["industry"] = base["instrument_id"].map(
            lambda k: prev_map.get(str(k), {}).get("industry", "")
        )
        base["style"] = np.where(base["style"].astype(str).str.strip().eq(""), default_style, base["style"])
        base["industry"] = np.where(
            base["industry"].astype(str).str.strip().eq(""),
            default_industry,
            base["industry"],
        )
        base["style"] = np.where(base["style"].astype(str).str.strip().eq(""), base["auto_style"], base["style"])
        base["industry"] = np.where(
            base["industry"].astype(str).str.strip().eq(""),
            base["auto_industry"],
            base["industry"],
        )
        st.session_state["holding_override_editor_sig"] = seed_sig
        st.session_state["holding_override_editor_df"] = base

    edited = st.data_editor(
        st.session_state["holding_override_editor_df"],
        key="holding_override_editor_widget",
        hide_index=True,
        width="stretch",
        disabled=[
            "instrument_id",
            "ticker",
            "product",
            "auto_style",
            "auto_industry",
            *(["style", "industry"] if disabled else []),
        ],
        num_rows="fixed",
        column_order=[
            "instrument_id",
            "ticker",
            "product",
            "auto_style",
            "style",
            "auto_industry",
            "industry",
        ],
        column_config={
            "instrument_id": st.column_config.TextColumn(
                "instrument_id",
                required=True,
                help="Unique instrument identifier.",
            ),
            "ticker": st.column_config.TextColumn(
                "ticker",
                help="Mapped ticker symbol used in pricing and analytics.",
            ),
            "product": st.column_config.TextColumn(
                "product",
                help="Original product name from DEGIRO export.",
            ),
            "auto_style": st.column_config.TextColumn(
                "auto_style",
                help="Automatically inferred style bucket.",
            ),
            "style": st.column_config.TextColumn(
                "style",
                help="Manual style override (optional).",
            ),
            "auto_industry": st.column_config.TextColumn(
                "auto_industry",
                help="Automatically inferred industry bucket.",
            ),
            "industry": st.column_config.TextColumn(
                "industry",
                help="Manual industry override (optional).",
            ),
        },
    )
    if isinstance(edited, pd.DataFrame):
        st.session_state["holding_override_editor_df"] = edited
    out: dict[str, dict[str, str]] = {}
    final_df = st.session_state.get("holding_override_editor_df", pd.DataFrame())
    if isinstance(final_df, pd.DataFrame):
        for row in final_df.itertuples(index=False):
            instrument_id = str(getattr(row, "instrument_id", "")).strip()
            if instrument_id == "":
                continue
            style = str(getattr(row, "style", "")).strip()
            industry = str(getattr(row, "industry", "")).strip()
            entry: dict[str, str] = {}
            if style:
                entry["style"] = style
            if industry:
                entry["industry"] = industry
            if entry:
                out[instrument_id] = entry
    return out


def build_benchmark_bundle(
    *,
    metrics_index: pd.Index,
    cache_dir: str,
    logger: Any,
) -> dict[str, Any]:
    # AGENT_NOTE: Returns two linked outputs:
    # - `levels_df` for plotting (`plots.build_benchmark_comparison_figure`)
    # - `returns_df` for alpha/beta stats (`insights.build_performance_dashboard`)
    idx = pd.to_datetime(metrics_index, errors="coerce")
    idx = pd.Index(idx[idx.notna()]).sort_values()
    if idx.empty:
        return {"levels_df": pd.DataFrame(), "returns_df": pd.DataFrame(), "warnings": []}

    start = pd.Timestamp(idx.min()).normalize()
    end = pd.Timestamp(idx.max()).normalize()
    daily_index = pd.date_range(start=start, end=end, freq="D")
    specs = {
        "MSCI World": {"ticker": "URTH", "currency": "USD"},
        "AEX": {"ticker": "^AEX", "currency": "EUR"},
        "Nasdaq": {"ticker": "^IXIC", "currency": "USD"},
    }
    levels: dict[str, pd.Series] = {}
    warnings: list[str] = []
    for name, spec in specs.items():
        ticker = str(spec["ticker"])
        ccy = str(spec["currency"]).upper()
        try:
            px = fetch_price_series(
                ticker=ticker,
                start=start,
                end=end,
                cache_dir=cache_dir,
                logger=logger,
            ).reindex(daily_index).ffill().bfill()
            if ccy != "EUR":
                fx = fetch_fx_series(
                    currency=ccy,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    logger=logger,
                ).reindex(daily_index).ffill().bfill()
                px = px * fx
            if px.dropna().empty:
                warnings.append(f"Benchmark series is empty: {name} ({ticker}).")
                continue
            levels[name] = pd.to_numeric(px, errors="coerce")
        except Exception:
            warnings.append(f"Benchmark unavailable: {name} ({ticker}).")
    if not levels:
        return {"levels_df": pd.DataFrame(), "returns_df": pd.DataFrame(), "warnings": warnings}

    levels_df_raw = pd.DataFrame(levels, index=daily_index).dropna(how="all")
    levels_df = levels_df_raw.copy()
    for col in levels_df.columns:
        first_valid = levels_df[col].dropna()
        if first_valid.empty or float(first_valid.iloc[0]) == 0.0:
            levels_df[col] = np.nan
        else:
            levels_df[col] = levels_df[col] / float(first_valid.iloc[0]) * 100.0
    returns_df = levels_df_raw.pct_change().replace([np.inf, -np.inf], np.nan)
    returns_df = returns_df.dropna(how="all")
    return {
        "levels_df": levels_df,
        "returns_df": returns_df,
        "warnings": warnings,
    }


def build_daily_position_cost_basis(
    *,
    transactions: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    instrument_ids: list[str],
) -> pd.DataFrame:
    # AGENT_NOTE: This approximates net spent on currently held shares and is
    # consumed by `plots.build_holdings_over_time_figure` hover details.
    if (
        transactions is None
        or transactions.empty
        or daily_index is None
        or len(daily_index) == 0
        or len(instrument_ids) == 0
    ):
        return pd.DataFrame(index=daily_index, columns=instrument_ids, dtype="float64")

    tx = transactions.copy()
    tx["datetime"] = pd.to_datetime(tx.get("datetime"), errors="coerce")
    tx = tx[tx["datetime"].notna()].copy()
    tx["date"] = tx["datetime"].dt.normalize()
    tx["instrument_id"] = tx.get("instrument_id", pd.Series("", index=tx.index)).astype(str)
    tx["quantity"] = pd.to_numeric(tx.get("quantity"), errors="coerce")
    tx["total_eur"] = pd.to_numeric(tx.get("total_eur"), errors="coerce")
    if "is_cash_like" in tx.columns:
        tx = tx.loc[~tx["is_cash_like"].fillna(False)].copy()
    tx = tx[tx["instrument_id"].isin([str(v) for v in instrument_ids])].copy()
    tx = tx.sort_values("datetime")
    if tx.empty:
        return pd.DataFrame(0.0, index=daily_index, columns=instrument_ids, dtype="float64")

    qty_map: dict[str, float] = {str(i): 0.0 for i in instrument_ids}
    cost_map: dict[str, float] = {str(i): 0.0 for i in instrument_ids}
    rows: list[dict[str, float]] = []

    grouped = {k: v.copy() for k, v in tx.groupby("date", sort=True)}
    for dt in daily_index:
        day = pd.Timestamp(dt).normalize()
        day_tx = grouped.get(day, pd.DataFrame())
        if not day_tx.empty:
            for row in day_tx.itertuples(index=False):
                iid = str(getattr(row, "instrument_id", ""))
                if iid not in qty_map:
                    continue
                qty = float(getattr(row, "quantity", np.nan))
                total_eur = float(getattr(row, "total_eur", np.nan))
                if not np.isfinite(qty) or abs(qty) < 1e-12:
                    continue
                if qty > 0.0:
                    spend = -total_eur if np.isfinite(total_eur) else 0.0
                    if not np.isfinite(spend):
                        spend = 0.0
                    cost_map[iid] += max(spend, 0.0)
                    qty_map[iid] += qty
                else:
                    sell_qty = -qty
                    prev_qty = qty_map[iid]
                    if prev_qty <= 1e-12:
                        qty_map[iid] = max(prev_qty - sell_qty, 0.0)
                        cost_map[iid] = 0.0
                        continue
                    avg_cost = cost_map[iid] / prev_qty if prev_qty > 0 else 0.0
                    qty_reduced = min(sell_qty, prev_qty)
                    qty_new = prev_qty - qty_reduced
                    cost_new = cost_map[iid] - avg_cost * qty_reduced
                    qty_map[iid] = max(qty_new, 0.0)
                    cost_map[iid] = max(cost_new, 0.0)
                    if qty_map[iid] <= 1e-9:
                        qty_map[iid] = 0.0
                        cost_map[iid] = 0.0
        rows.append({iid: float(cost_map.get(str(iid), 0.0)) for iid in instrument_ids})

    return pd.DataFrame(rows, index=daily_index, columns=instrument_ids, dtype="float64")


def run_analysis(*, params: dict[str, Any], logger: Any) -> None:
    workflow = st.session_state["workflow"]
    datasets_entries = workflow["datasets"]
    if not datasets_entries:
        st.error("No datasets loaded.")
        return

    upload_sig = current_upload_signature()
    params_sig = st.session_state.get("params_sig")
    processed_sig = (upload_sig, params_sig)
    if workflow.get("processed") and st.session_state.get("processed_sig") == processed_sig:
        transition(STATE_VIEWING_RESULTS, "Using cached processed results")
        return

    start = time.perf_counter()
    try:
        with st.spinner("Processing datasets..."):
            log("Starting reconciliation workflow", start, logger)
            loaded_datasets: dict[str, LoadedDataset] = {
                key: value["dataset"] for key, value in datasets_entries.items()
            }
            results = process_loaded_datasets(
                datasets=loaded_datasets,
                params=params,
                logger=logger,
            )

            workflow["processed"] = results
            workflow["summaries"] = {"updated_at": datetime.now().isoformat(timespec="seconds")}
            workflow["warnings"] = results.get("warnings", [])
            st.session_state["processed_sig"] = processed_sig
            transition(STATE_VIEWING_RESULTS, "Processing completed")
            log("Finished processing workflow", start, logger)
    except UserFacingError as exc:
        st.error(exc.to_ui_text())
        logger.exception("User-facing processing error")
        transition(STATE_SELECTING_PARAMS, "Processing failed with user-facing error")
    except Exception:
        st.error(
            "Analysis failed due to an unexpected error. Check logs/app.log for technical details."
        )
        logger.exception("Unexpected processing error")
        transition(STATE_SELECTING_PARAMS, "Processing failed unexpectedly")


def process_loaded_datasets(
    *,
    datasets: dict[str, LoadedDataset],
    params: dict[str, Any],
    logger: Any,
) -> dict[str, Any]:
    # AGENT_NOTE: High-impact integration point.
    # Any change to returned keys/structures must be reflected in `render_main`
    # and tests that assert downstream tables/figures.
    prime_cache_runtime_state(cache_dir="cache", logger=logger)
    cash_results: dict[str, CashReconciliationResult] = {}
    totals_results: dict[str, TotalsResult] = {}
    per_dataset_timeseries: dict[str, Any] = {}
    issue_tables: list[dict[str, Any]] = []

    for key, dataset in datasets.items():
        cash_result, totals_result = reconcile_dataset(
            account_label=dataset.account_label,
            portfolio=dataset.portfolio,
            account=dataset.account,
            fx_lookup=lambda ccy: latest_fx_rate(ccy, cache_dir="cache", logger=logger),
        )
        cash_results[dataset.account_label] = cash_result
        totals_results[dataset.account_label] = totals_result
        for issue in dataset.issues:
            issue_tables.append(
                {
                    "dataset": dataset.account_label,
                    "label": issue["label"],
                    "count": issue["count"],
                    "examples": issue["examples"],
                }
            )

    merged_transactions = pd.concat(
        [d.transactions for d in datasets.values()],
        ignore_index=True,
    )
    merged_portfolio = pd.concat([d.portfolio for d in datasets.values()], ignore_index=True)
    merged_account = pd.concat([d.account for d in datasets.values()], ignore_index=True)
    merged_instruments = merge_instruments([d.instruments for d in datasets.values()])

    _combined_cash_direct, combined_totals_direct = reconcile_dataset(
        account_label="Combined",
        portfolio=merged_portfolio,
        account=merged_account,
        fx_lookup=lambda ccy: latest_fx_rate(ccy, cache_dir="cache", logger=logger),
    )
    cash_results["Combined"] = combine_cash_results_from_datasets(cash_results)

    combined_totals_sum = combine_totals(totals_results)
    totals_results["Combined"] = combined_totals_sum

    ts_result = compute_portfolio_timeseries(
        transactions=merged_transactions,
        account=merged_account,
        instruments=merged_instruments,
        cache_dir="cache",
        logger=logger,
    )
    for issue in ts_result.issues:
        issue_tables.append(
            {
                "dataset": "Combined",
                "label": issue["label"],
                "count": issue["count"],
                "examples": issue["examples"],
            }
        )

    for panel_key, dataset in datasets.items():
        cur_ts = compute_portfolio_timeseries(
            transactions=dataset.transactions,
            account=dataset.account,
            instruments=dataset.instruments,
            cache_dir="cache",
            logger=logger,
        )
        per_dataset_timeseries[dataset.account_label] = cur_ts
        for issue in cur_ts.issues:
            issue_tables.append(
                {
                    "dataset": dataset.account_label,
                    "label": issue["label"],
                    "count": issue["count"],
                    "examples": issue["examples"],
                }
            )

    holdings_for_tables = merged_portfolio.loc[~merged_portfolio["is_cash_like"]].copy()
    table_outputs = build_four_tables(
        holdings=holdings_for_tables,
        totals=combined_totals_sum,
        target_etf_fraction=float(params["target_etf_fraction"]),
        desired_etf_holdings=int(params["desired_etf_holdings"]),
        desired_non_etf_holdings=int(params["desired_non_etf_holdings"]),
        min_over_value_eur=float(params["min_over_value_eur"]),
    )
    ai_insights = build_ai_generated_insights(
        metrics_df=ts_result.metrics,
        over_target_df=table_outputs.get("over_target", pd.DataFrame()),
    )
    spread_analysis = build_ai_spread_analysis(
        holdings_df=holdings_for_tables,
        total_value_eur=float(combined_totals_sum.total_value_eur),
        cash_value_eur=float(combined_totals_sum.cash_value_eur),
        cash_detail_df=(
            cash_results["Combined"].detail
            if "Combined" in cash_results and hasattr(cash_results["Combined"], "detail")
            else pd.DataFrame()
        ),
        strategy={
            "target_etf_fraction": float(params["target_etf_fraction"]),
            "desired_etf_holdings": int(params["desired_etf_holdings"]),
            "desired_non_etf_holdings": int(params["desired_non_etf_holdings"]),
            "target_cash_pct": float(params["target_cash_pct"]),
            "max_single_holding_pct": float(params["max_single_holding_pct"]),
            "max_top5_holdings_pct": float(params["max_top5_holdings_pct"]),
            "max_single_currency_pct": float(params["max_single_currency_pct"]),
            "max_single_industry_pct": float(params["max_single_industry_pct"]),
            "max_pair_correlation": float(params["max_pair_correlation"]),
            "min_total_holdings": int(params["min_total_holdings"]),
            "min_over_value_eur": float(params["min_over_value_eur"]),
            "target_currency_pct": params.get("target_currency_pct", {}),
            "target_industry_pct": params.get("target_industry_pct", {}),
            "target_style_pct": params.get("target_style_pct", {}),
            "holding_category_overrides": params.get("holding_category_overrides", {}),
        },
        prices_eur=ts_result.prices_eur,
        ticker_classifications_path=Path("ticker_classifications.csv"),
        auto_append_ticker_characteristics=True,
    )
    benchmark_bundle = build_benchmark_bundle(
        metrics_index=ts_result.metrics.index,
        cache_dir="cache",
        logger=logger,
    )
    performance_dashboard = build_performance_dashboard(
        metrics_df=ts_result.metrics,
        transactions_df=merged_transactions,
        account_df=merged_account,
        benchmark_returns_df=benchmark_bundle.get("returns_df", pd.DataFrame()),
    )
    ai_period_fig = build_period_decomposition_figure(
        ai_insights.get("period_performance_df", pd.DataFrame())
    )
    ai_drawdown_fig = build_drawdown_figure(ts_result.metrics)
    ai_cash_allocation_fig = build_cash_allocation_figure(
        ts_result.metrics,
        target_cash_pct=float(params["target_cash_pct"]),
    )

    fig_portfolio = build_portfolio_over_time_figure(ts_result.metrics)
    fig_performance_over_time = build_performance_over_time_figure(ts_result.metrics)
    holdings_cost_basis_df = build_daily_position_cost_basis(
        transactions=merged_transactions,
        daily_index=ts_result.positions.index,
        instrument_ids=list(ts_result.positions.columns),
    )
    fig_holdings_over_time = build_holdings_over_time_figure(
        positions=ts_result.positions,
        prices_eur=ts_result.prices_eur,
        instruments=merged_instruments,
        cash_series=ts_result.metrics.get("cash"),
        cost_basis_eur=holdings_cost_basis_df,
    )
    fig_benchmark = build_benchmark_comparison_figure(benchmark_bundle.get("levels_df", pd.DataFrame()))
    fig_normalized = build_normalized_median_figure(
        ts_result.prices_eur,
        instruments=merged_instruments,
        lookback_months=int(params["lookback_months"]),
        median_window_months=int(params["median_window_months"]),
    )

    warnings: list[str] = []
    for dataset in datasets.values():
        warnings.extend(dataset.warnings)
    warnings.extend(ts_result.warnings)
    for ts_single in per_dataset_timeseries.values():
        warnings.extend(ts_single.warnings)
    warnings.extend(benchmark_bundle.get("warnings", []))

    snapshot_vs_rebuilt_df = build_snapshot_vs_rebuilt_table(
        totals_results=totals_results,
        per_dataset_timeseries=per_dataset_timeseries,
        combined_timeseries=ts_result,
    )
    positions_reconciliation_rows: list[pd.DataFrame] = []
    for dataset in datasets.values():
        per_table = build_positions_reconciliation_table(
            portfolio_df=dataset.portfolio,
            ts_result=per_dataset_timeseries.get(dataset.account_label),
        )
        if not per_table.empty:
            per_table.insert(0, "dataset", dataset.account_label)
            positions_reconciliation_rows.append(per_table)
    combined_positions_table = build_positions_reconciliation_table(
        portfolio_df=merged_portfolio,
        ts_result=ts_result,
    )
    if not combined_positions_table.empty:
        combined_positions_table.insert(0, "dataset", "Combined")
        positions_reconciliation_rows.append(combined_positions_table)
    positions_reconciliation_df = (
        pd.concat(positions_reconciliation_rows, ignore_index=True)
        if positions_reconciliation_rows
        else pd.DataFrame()
    )
    latest_balance_check_df = build_latest_account_balance_check_table(cash_results)

    category_summary_rows: list[pd.DataFrame] = []
    for dataset in datasets.values():
        category_df, category_warnings, category_issues = summarize_account_categories(
            account_df=dataset.account,
            cache_dir="cache",
            logger=logger,
        )
        warnings.extend(category_warnings)
        for issue in category_issues:
            issue_tables.append(
                {
                    "dataset": dataset.account_label,
                    "label": issue["label"],
                    "count": issue["count"],
                    "examples": issue["examples"],
                }
            )
        if not category_df.empty:
            category_df.insert(0, "dataset", dataset.account_label)
            category_summary_rows.append(category_df)

    combined_category_df, combined_category_warnings, combined_category_issues = summarize_account_categories(
        account_df=merged_account,
        cache_dir="cache",
        logger=logger,
    )
    warnings.extend(combined_category_warnings)
    for issue in combined_category_issues:
        issue_tables.append(
            {
                "dataset": "Combined",
                "label": issue["label"],
                "count": issue["count"],
                "examples": issue["examples"],
            }
        )
    if not combined_category_df.empty:
        combined_category_df.insert(0, "dataset", "Combined")
        category_summary_rows.append(combined_category_df)

    account_category_summary_df = (
        pd.concat(category_summary_rows, ignore_index=True) if category_summary_rows else pd.DataFrame()
    )
    panel_to_account_label = {panel: dataset.account_label for panel, dataset in datasets.items()}
    account_label_to_dataset_name = {
        account_label: panel.replace("_", " ").title()
        for panel, account_label in panel_to_account_label.items()
    }
    costs_summary_df = build_costs_summary_table(account_category_summary_df)
    degiro_costs_quarterly_df = build_quarterly_costs_table(
        account_df=merged_account,
        transactions_df=merged_transactions,
        account_label_to_dataset_name=account_label_to_dataset_name,
    )
    fig_degiro_costs_quarterly = build_degiro_costs_quarterly_figure(degiro_costs_quarterly_df)
    cash_source_raw_df = build_cash_source_raw_table(merged_account)
    cash_timeline_df = build_cash_timeline_table(ts_result.metrics)
    daily_close_holdings_df = build_daily_close_holdings_table(
        ts_result=ts_result,
        instruments=merged_instruments,
    )
    manual_monthly_reference_df = build_manual_monthly_reference_df()
    monthly_start_portfolio_value_df = build_monthly_starting_portfolio_value_table(
        per_dataset_metrics={
            label: ts.metrics
            for label, ts in per_dataset_timeseries.items()
            if ts is not None and hasattr(ts, "metrics")
        },
        manual_tracked_values=manual_monthly_reference_df,
    )
    cache_runtime_state = get_cache_runtime_state(cache_dir="cache")

    return {
        "panel_to_account_label": panel_to_account_label,
        "cash_results": cash_results,
        "totals_results": totals_results,
        "combined_totals_direct": combined_totals_direct,
        "reconciliation_df": reconciliation_table(cash_results),
        "transactions_df": merged_transactions,
        "account_df": merged_account,
        "timeseries": ts_result,
        "per_dataset_timeseries": per_dataset_timeseries,
        "snapshot_vs_rebuilt_df": snapshot_vs_rebuilt_df,
        "positions_reconciliation_df": positions_reconciliation_df,
        "latest_balance_check_df": latest_balance_check_df,
        "account_category_summary_df": account_category_summary_df,
        "costs_summary_df": costs_summary_df,
        "degiro_costs_quarterly_df": degiro_costs_quarterly_df,
        "cash_source_raw_df": cash_source_raw_df,
        "cash_timeline_df": cash_timeline_df,
        "daily_close_holdings_df": daily_close_holdings_df,
        "monthly_start_portfolio_value_df": monthly_start_portfolio_value_df,
        "fig_portfolio": fig_portfolio,
        "fig_performance_over_time": fig_performance_over_time,
        "fig_holdings_over_time": fig_holdings_over_time,
        "fig_benchmark": fig_benchmark,
        "fig_degiro_costs_quarterly": fig_degiro_costs_quarterly,
        "fig_normalized": fig_normalized,
        "tables": table_outputs,
        "ai_insights": ai_insights,
        "spread_analysis": spread_analysis,
        "performance_dashboard": performance_dashboard,
        "benchmark_levels_df": benchmark_bundle.get("levels_df", pd.DataFrame()),
        "benchmark_returns_df": benchmark_bundle.get("returns_df", pd.DataFrame()),
        "ai_period_fig": ai_period_fig,
        "ai_drawdown_fig": ai_drawdown_fig,
        "ai_cash_allocation_fig": ai_cash_allocation_fig,
        "offline_mode": bool(cache_runtime_state.get("offline_mode", False)),
        "offline_cached_from": cache_runtime_state.get("offline_cached_from"),
        "warnings": warnings,
        "issue_tables": issue_tables,
    }


def merge_instruments(instruments_list: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(instruments_list, ignore_index=True)
    merged = merged.sort_values(["instrument_id", "ticker"], na_position="last")

    def first_notna(values: pd.Series) -> Any:
        for value in values:
            if pd.notna(value) and str(value) != "":
                return value
        return np.nan

    out = merged.groupby("instrument_id", as_index=False).agg(
        product=("product", first_notna),
        isin=("isin", first_notna),
        symbol=("symbol", first_notna),
        currency=("currency", first_notna),
        ticker=("ticker", first_notna),
        is_etf=("is_etf", "max"),
        is_cash_like=("is_cash_like", "max"),
    )
    return out


def money_format_columns(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    out: list[str] = []
    for col in df.columns:
        name = str(col).strip().lower()
        if "eur" not in name and name not in {"raw_balance", "raw_change"}:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            out.append(col)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().any():
            out.append(col)
    return out


def pct_format_columns(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    out: list[str] = []
    for col in df.columns:
        name = str(col).strip().lower()
        if "pct" not in name and "percent" not in name:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            out.append(col)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().any():
            out.append(col)
    return out


def quantity_format_columns(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    out: list[str] = []
    quantity_names = {"quantity", "qty"}
    for col in df.columns:
        name = str(col).strip().lower()
        if name not in quantity_names:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            out.append(col)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().any():
            out.append(col)
    return out


def four_table_format_map(df: pd.DataFrame) -> dict[str, str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    out: dict[str, str] = {}
    for col in quantity_format_columns(df):
        out[col] = "{:,.0f}"
    for col in pct_format_columns(df):
        out[col] = "{:,.2f}"
    return out


def render_dataframe(
    df: pd.DataFrame,
    *,
    width: str = "stretch",
    highlight_date: str | None = None,
    column_formats: dict[str, str] | None = None,
) -> None:
    if not isinstance(df, pd.DataFrame):
        st.dataframe(df, width=width)
        return
    money_cols = money_format_columns(df)
    active_custom_formats = {
        col: fmt for col, fmt in (column_formats or {}).items() if col in df.columns and isinstance(fmt, str)
    }
    needs_styling = bool(money_cols or active_custom_formats) or (
        highlight_date is not None and "date" in df.columns
    )
    if not needs_styling:
        st.dataframe(df, width=width)
        return
    display_df = df.copy()
    for col in sorted(set(money_cols).union(active_custom_formats.keys())):
        if not pd.api.types.is_numeric_dtype(display_df[col]):
            converted = pd.to_numeric(display_df[col], errors="coerce")
            if converted.notna().any():
                display_df[col] = converted

    styler = display_df.style
    format_map: dict[str, str] = {}
    if money_cols:
        format_map.update({col: "{:,.2f}" for col in money_cols})
    format_map.update(active_custom_formats)
    if format_map:
        styler = styler.format(format_map)
    if highlight_date is not None and "date" in df.columns:
        styler = styler.apply(
            lambda row: (
                ["background-color: rgba(59, 130, 246, 0.20)"] * len(row)
                if str(row.get("date", "")) == highlight_date
                else [""] * len(row)
            ),
            axis=1,
        )
    st.dataframe(styler, width=width)


def render_main(*, logger: Any) -> None:
    # AGENT_NOTE: This function is tightly coupled to keys returned by
    # `process_loaded_datasets()`. Keep key names stable unless you also update
    # all section rendering and related tests.
    del logger
    workflow = st.session_state["workflow"]
    processed = workflow.get("processed", {})
    if not processed or not at_or_after(STATE_VIEWING_RESULTS):
        st.info("Load Dataset A and/or Dataset B in the sidebar, then click Run analysis.")
        return

    per_dataset_timeseries = processed.get("per_dataset_timeseries", {})
    panel_to_account_label = processed.get("panel_to_account_label", {})
    spread_analysis = processed.get("spread_analysis", {})
    performance_dashboard = processed.get("performance_dashboard", {})
    tables_data = processed.get("tables", {})
    offline_mode = bool(processed.get("offline_mode", False))
    offline_cached_from = processed.get("offline_cached_from")

    def _fmt_day(value: pd.Timestamp) -> str:
        return value.strftime("%d-%b-%Y").lower()

    def _to_float(value: Any) -> float:
        try:
            out = float(value)
        except Exception:
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    def _fmt_eur(value: float) -> str:
        if not np.isfinite(value):
            return "N/A"
        return f"EUR {value:,.2f}"

    def _fmt_pct(value: float) -> str:
        if not np.isfinite(value):
            return "N/A"
        return f"{value:,.1f}%"

    def _fmt_offline_cached_from(value: Any) -> str:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return "unknown time"
        return ts.strftime("%H:%M:%S %d-%m-%Y")

    def _delta_mask(df: pd.DataFrame, delta_col: str) -> pd.Series:
        if not isinstance(df, pd.DataFrame):
            return pd.Series(dtype=bool)
        if df.empty or delta_col not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        values = pd.to_numeric(df[delta_col], errors="coerce")
        return values.round(2).ne(0.0) & values.notna()

    def _render_delta_table(
        *,
        df: pd.DataFrame,
        delta_col: str,
        empty_message: str,
        warning_message: str,
    ) -> bool:
        if df is None or df.empty:
            st.info(empty_message)
            return False
        display_df = df.copy()
        money_cols = money_format_columns(display_df)
        for col in money_cols:
            if not pd.api.types.is_numeric_dtype(display_df[col]):
                converted = pd.to_numeric(display_df[col], errors="coerce")
                if converted.notna().any():
                    display_df[col] = converted

        diff_mask = _delta_mask(display_df, delta_col)
        styler = display_df.style
        if money_cols:
            styler = styler.format({col: "{:,.2f}" for col in money_cols})
        styler = styler.apply(
            lambda row: (
                ["background-color: rgba(255, 99, 71, 0.20)"] * len(row)
                if bool(diff_mask.get(row.name, False))
                else [""] * len(row)
            ),
            axis=1,
        )
        st.dataframe(styler, width="stretch")
        if bool(diff_mask.any()):
            st.warning(warning_message)
        return bool(diff_mask.any())

    def _holding_key(row: pd.Series) -> str:
        isin = str(row.get("isin", "")).strip().upper()
        if isin and isin not in {"NAN", "NONE"}:
            return f"isin::{isin}"
        product = str(row.get("product", "")).strip().lower()
        ticker = str(row.get("ticker", "")).strip().lower()
        return f"product::{product}|ticker::{ticker}"

    def _count_over_target_by_segment() -> tuple[int, int]:
        over_df = tables_data.get("over_target", pd.DataFrame())
        etf_df = tables_data.get("etf", pd.DataFrame())
        non_etf_df = tables_data.get("non_etf", pd.DataFrame())
        if any(not isinstance(df, pd.DataFrame) or df.empty for df in [over_df, etf_df, non_etf_df]):
            return 0, 0
        etf_keys = {_holding_key(row) for _, row in etf_df.iterrows()}
        non_etf_keys = {_holding_key(row) for _, row in non_etf_df.iterrows()}
        etf_count = 0
        non_etf_count = 0
        for _, row in over_df.iterrows():
            key = _holding_key(row)
            if key in etf_keys:
                etf_count += 1
            elif key in non_etf_keys:
                non_etf_count += 1
        return int(etf_count), int(non_etf_count)

    date_lines: list[str] = []
    if offline_mode:
        st.warning(
            f"OFFLINE MODUS, using cached data from {_fmt_offline_cached_from(offline_cached_from)}"
        )
    for panel_key in sorted(panel_to_account_label.keys()):
        account_label = panel_to_account_label[panel_key]
        ts_single = per_dataset_timeseries.get(account_label)
        metrics_df = getattr(ts_single, "metrics", pd.DataFrame())
        if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
            continue
        first_date = pd.to_datetime(metrics_df.index.min(), errors="coerce")
        last_date = pd.to_datetime(metrics_df.index.max(), errors="coerce")
        if pd.isna(first_date) or pd.isna(last_date):
            continue
        panel_name = panel_key.replace("_", " ").title().strip()
        account_label_text = str(account_label).strip()
        if account_label_text.lower().startswith(panel_name.lower()):
            display_label = account_label_text
        elif account_label_text:
            display_label = f"{panel_name} ({account_label_text})"
        else:
            display_label = panel_name
        date_lines.append(f"{display_label}: {_fmt_day(first_date)} to {_fmt_day(last_date)}")
    if date_lines:
        st.caption("Loaded data date ranges (first to last):")
        for line in date_lines:
            st.caption(f"- {line}")

    cash_reconciliation_df = build_cash_reconciliation_overview_table(
        cash_results=processed.get("cash_results", {}),
        combined_timeseries=processed.get("timeseries"),
        panel_to_account_label=processed.get("panel_to_account_label", {}),
        reported_values=workflow.get("reported_values", {}),
    )
    portfolio_reconciliation_df = build_portfolio_reconciliation_overview_table(
        totals_results=processed["totals_results"],
        combined_totals_direct=processed["combined_totals_direct"],
        cash_results=processed.get("cash_results", {}),
        panel_to_account_label=processed.get("panel_to_account_label", {}),
        reported_values=workflow.get("reported_values", {}),
    )
    cash_recon_has_warning = bool(_delta_mask(cash_reconciliation_df, "delta_vs_ground_truth_eur").any())
    portfolio_recon_has_warning = bool(_delta_mask(portfolio_reconciliation_df, "delta_vs_ground_truth_eur").any())
    validation_expanded = False

    latest_metrics = pd.Series(dtype="float64")
    metrics_df = getattr(processed.get("timeseries"), "metrics", pd.DataFrame())
    if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        latest_metrics = metrics_df.iloc[-1]
    latest_total_deposits = _to_float(latest_metrics.get("total_deposits", np.nan))
    latest_total_value = _to_float(latest_metrics.get("portfolio_value", np.nan))
    latest_cash = _to_float(latest_metrics.get("cash", np.nan))

    etf_pct = float("nan")
    non_etf_pct = float("nan")
    if isinstance(spread_analysis, dict):
        etf_non_etf_df = spread_analysis.get("etf_non_etf_df", pd.DataFrame())
    else:
        etf_non_etf_df = pd.DataFrame()
    if isinstance(etf_non_etf_df, pd.DataFrame) and not etf_non_etf_df.empty:
        etf_row = etf_non_etf_df.loc[etf_non_etf_df["segment"] == "ETF"]
        non_etf_row = etf_non_etf_df.loc[etf_non_etf_df["segment"] == "Non-ETF"]
        if not etf_row.empty:
            etf_pct = _to_float(etf_row.iloc[0].get("share_pct_total", np.nan))
        if not non_etf_row.empty:
            non_etf_pct = _to_float(non_etf_row.iloc[0].get("share_pct_total", np.nan))

    etf_over_count, non_etf_over_count = _count_over_target_by_segment()
    performance_summary = (
        performance_dashboard.get("summary", {})
        if isinstance(performance_dashboard, dict)
        else {}
    )
    savings_interest_equivalent_eur = _to_float(
        performance_summary.get("all_time_investment_pnl_eur", np.nan)
    )
    savings_xirr_pct = _to_float(performance_summary.get("all_time_xirr_pct", np.nan))

    latest_price_by_instrument: dict[str, dict[str, Any]] = {}
    prices_eur_df = getattr(processed.get("timeseries"), "prices_eur", pd.DataFrame())
    if isinstance(prices_eur_df, pd.DataFrame) and not prices_eur_df.empty:
        px = prices_eur_df.copy()
        px.index = pd.to_datetime(px.index, errors="coerce")
        px = px[px.index.notna()].sort_index()
        for instrument_id in px.columns:
            series = pd.to_numeric(px[instrument_id], errors="coerce").dropna()
            if series.empty:
                continue
            latest_idx = pd.to_datetime(series.index[-1], errors="coerce")
            latest_price_by_instrument[str(instrument_id)] = {
                "price_eur": float(series.iloc[-1]),
                "date": latest_idx.strftime("%Y-%m-%d") if pd.notna(latest_idx) else "",
            }

    def _prepare_recent_trades(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["quantity"] = pd.to_numeric(out.get("quantity"), errors="coerce")
        out["total_eur"] = pd.to_numeric(out.get("total_eur"), errors="coerce")
        out["fees_eur"] = pd.to_numeric(out.get("fees_eur"), errors="coerce").fillna(0.0).abs()
        qty_abs = out["quantity"].abs()
        gross_abs = out["total_eur"].abs()
        effective_total_eur = np.where(
            out["side"].eq("Buy"),
            gross_abs + out["fees_eur"],
            np.where(
                out["side"].eq("Sell"),
                np.maximum(gross_abs - out["fees_eur"], 0.0),
                gross_abs,
            ),
        )
        out["trade_price_eur_incl_costs"] = np.where(
            qty_abs > 1e-12,
            effective_total_eur / qty_abs,
            np.nan,
        )
        out["trade_price_eur_incl_costs"] = np.round(out["trade_price_eur_incl_costs"], 2)
        out["quantity"] = np.where(qty_abs.notna(), np.round(qty_abs, 0), np.nan)
        out["quantity"] = pd.Series(out["quantity"], index=out.index).astype("Int64")
        instrument_keys = out.get("instrument_id", pd.Series("", index=out.index)).astype(str)
        out["latest_known_price_eur"] = instrument_keys.map(
            lambda key: latest_price_by_instrument.get(key, {}).get("price_eur", np.nan)
        )
        out["latest_known_price_date"] = instrument_keys.map(
            lambda key: latest_price_by_instrument.get(key, {}).get("date", "")
        )
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        ordered_cols = [
            "datetime",
            "side",
            "product",
            "ticker",
            "quantity",
            "trade_price_eur_incl_costs",
            "latest_known_price_eur",
            "latest_known_price_date",
            "total_eur",
            "account_label",
        ]
        return out[[col for col in ordered_cols if col in out.columns]]

    transactions_df = processed.get("transactions_df", pd.DataFrame())
    recent_buys_df = pd.DataFrame()
    recent_sells_df = pd.DataFrame()
    if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
        tx = transactions_df.copy()
        tx["datetime"] = pd.to_datetime(tx.get("datetime"), errors="coerce")
        tx["quantity"] = pd.to_numeric(tx.get("quantity"), errors="coerce")
        tx["fees_eur"] = pd.to_numeric(tx.get("fees_eur"), errors="coerce")
        tx["total_eur"] = pd.to_numeric(tx.get("total_eur"), errors="coerce")
        tx = tx[tx["datetime"].notna()].copy()
        if "is_cash_like" in tx.columns:
            tx = tx.loc[~tx["is_cash_like"].fillna(False)].copy()
        tx["side"] = np.where(tx["quantity"] > 0.0, "Buy", np.where(tx["quantity"] < 0.0, "Sell", "Other"))
        keep_cols = [
            c
            for c in [
                "datetime",
                "side",
                "product",
                "ticker",
                "instrument_id",
                "quantity",
                "fees_eur",
                "total_eur",
                "account_label",
            ]
            if c in tx.columns
        ]
        buys = tx.loc[tx["side"].eq("Buy"), keep_cols].sort_values("datetime", ascending=False).head(5).copy()
        sells = tx.loc[tx["side"].eq("Sell"), keep_cols].sort_values("datetime", ascending=False).head(5).copy()
        recent_buys_df = _prepare_recent_trades(buys)
        recent_sells_df = _prepare_recent_trades(sells)

    with st.expander("Section 1: Overview", expanded=True):
        over_threshold_total = int(etf_over_count + non_etf_over_count)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.button(
                f"Total net deposited\n{_fmt_eur(latest_total_deposits)}",
                key="section1_kpi_net_deposited",
                disabled=True,
                width="stretch",
            )
        with c2:
            st.button(
                f"Current total value\n{_fmt_eur(latest_total_value)}",
                key="section1_kpi_total_value",
                disabled=True,
                width="stretch",
            )
        with c3:
            st.button(
                f"Current cash position\n{_fmt_eur(latest_cash)}",
                key="section1_kpi_cash",
                disabled=True,
                width="stretch",
            )
        with c4:
            st.button(
                f"ETF / non-ETF split\n{_fmt_pct(etf_pct)} / {_fmt_pct(non_etf_pct)}",
                key="section1_kpi_split",
                disabled=True,
                width="stretch",
            )
        with c5:
            btn_text = f"Above strategy threshold\nETF {etf_over_count} | non-ETF {non_etf_over_count}"
            try:
                st.button(
                    btn_text,
                    key="section1_kpi_above_target",
                    width="stretch",
                    type="primary" if over_threshold_total > 0 else "secondary",
                )
            except TypeError:
                st.button(
                    btn_text,
                    key="section1_kpi_above_target",
                    width="stretch",
                )
        with c6:
            st.button(
                (
                    f"Savings-equivalent interest\n{_fmt_eur(savings_interest_equivalent_eur)}"
                    f" ({_fmt_pct(savings_xirr_pct)}/yr)"
                ),
                key="section1_kpi_savings_equivalent",
                disabled=True,
                width="stretch",
            )

        st.subheader("Performance over time (deposits, value, simple return)")
        plotly_chart_stretch(processed.get("fig_performance_over_time"))
        st.subheader("Holdings and cash over time (includes closed holdings)")
        plotly_chart_stretch(processed.get("fig_holdings_over_time"))

        trades_left, trades_right = st.columns(2)
        with trades_left:
            st.markdown("**5 most recent buys**")
            if recent_buys_df.empty:
                st.info("No buy trades available.")
            else:
                render_dataframe(recent_buys_df, width="stretch")
        with trades_right:
            st.markdown("**5 most recent sells**")
            if recent_sells_df.empty:
                st.info("No sell trades available.")
            else:
                render_dataframe(recent_sells_df, width="stretch")

    validation_title = (
        "Section 2: Validation "
        f"(cash diff: {'YES' if cash_recon_has_warning else 'no'} | "
        f"portfolio diff: {'YES' if portfolio_recon_has_warning else 'no'})"
    )
    with st.expander(validation_title, expanded=validation_expanded):
        st.subheader("Daily Close Holdings (Combined, EUR)")
        st.caption(
            "Per-day close snapshot of combined holdings: count and EUR value per non-cash holding, "
            "cash as value-only, plus total portfolio value and cumulative net external flow."
        )
        daily_close_holdings_df = processed.get("daily_close_holdings_df", pd.DataFrame())
        if daily_close_holdings_df.empty:
            st.info("No completed daily close rows available for holdings table.")
        else:
            latest_table_date = pd.to_datetime(daily_close_holdings_df.get("date"), errors="coerce").max()
            if pd.notna(latest_table_date):
                st.caption(
                    f"Most recent date in this table: **{latest_table_date.strftime('%d-%b-%Y').lower()}**"
                )
                latest_is_today = latest_table_date.normalize() == pd.Timestamp.now().normalize()
                if latest_is_today:
                    render_dataframe(
                        daily_close_holdings_df,
                        width="stretch",
                        highlight_date=latest_table_date.strftime("%Y-%m-%d"),
                    )
                    st.info("Latest row is for today and may be intraday (day not fully closed yet).")
                else:
                    render_dataframe(daily_close_holdings_df, width="stretch")
            else:
                render_dataframe(daily_close_holdings_df, width="stretch")

        st.subheader("Cash Reconciliation")
        st.caption(
            "Combined totals only (Dataset A + B). Rows compare cash definitions; "
            "delta is shown versus portfolio ground truth."
        )
        _render_delta_table(
            df=cash_reconciliation_df,
            delta_col="delta_vs_ground_truth_eur",
            empty_message="No cash reconciliation overview available.",
            warning_message="Differences detected. Check out where this difference may have come from.",
        )

        st.subheader("Portfolio Reconciliation")
        st.caption(
            "Rows compare combined portfolio totals. Ground truth is the sum from Portfolio.csv "
            "(non-cash positions + cash rows)."
        )
        _render_delta_table(
            df=portfolio_reconciliation_df,
            delta_col="delta_vs_ground_truth_eur",
            empty_message="No portfolio reconciliation table available.",
            warning_message=(
                "Differences vs Portfolio.csv ground truth detected. "
                "Check where this difference may have come from."
            ),
        )

        st.subheader("Totals Check")
        st.text(build_totals_check_text(processed["totals_results"], processed["combined_totals_direct"]))

        st.subheader("Positions Reconciliation (Quantity vs Price Attribution)")
        st.caption(
            "Decomposes portfolio value delta into quantity mismatch and pricing mismatch. "
            "Large `value_delta_from_qty_eur` indicates missing/wrong transactions; "
            "large `value_delta_from_price_eur` indicates pricing/date mismatch."
        )
        positions_reconciliation_df = processed.get("positions_reconciliation_df", pd.DataFrame())
        if positions_reconciliation_df.empty:
            st.info("No positions reconciliation table available.")
        else:
            render_dataframe(positions_reconciliation_df, width="stretch")

        st.subheader("Latest Account Balance Check (Per Currency)")
        st.caption(
            "Current-cash check from Account.csv statements: latest EUR and USD balances per dataset, "
            "converted to EUR and summed."
        )
        latest_balance_check_df = processed.get("latest_balance_check_df", pd.DataFrame())
        if latest_balance_check_df.empty:
            st.info("No latest-balance check data available.")
        else:
            render_dataframe(latest_balance_check_df, width="stretch")

        st.subheader("Account Flow Summary by Category (Costs/Income)")
        st.caption(
            "Category totals from Account.csv rows. Outflow includes transaction fees, account fees and dividend tax; "
            "inflow includes dividends and interest."
        )
        category_summary_df = processed.get("account_category_summary_df", pd.DataFrame())
        if category_summary_df.empty:
            st.info("No category summary available.")
        else:
            render_dataframe(category_summary_df, width="stretch")

        st.subheader("Cash Derivation Debug (Raw vs Derived)")
        st.caption(
            "Left: normalized Account rows used to derive cash. Right: derived daily cash timeline."
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Raw account rows (cash source)**")
            raw_cash_df = processed.get("cash_source_raw_df", pd.DataFrame())
            if raw_cash_df.empty:
                st.info("No account rows available.")
            else:
                render_dataframe(raw_cash_df, width="stretch")
        with right:
            st.markdown("**Derived cash evolution (daily)**")
            timeline_df = processed.get("cash_timeline_df", pd.DataFrame())
            if timeline_df.empty:
                st.info("No cash timeline available.")
            else:
                render_dataframe(timeline_df, width="stretch")

        st.subheader("Data Quality Diagnostics (Examples as Tables)")
        if processed.get("warnings"):
            st.warning("\n".join(f"- {w}" for w in processed["warnings"][:30]))
        issue_tables = processed.get("issue_tables", [])
        if issue_tables:
            for idx, issue in enumerate(issue_tables[:40], start=1):
                dataset = issue.get("dataset", "Unknown dataset")
                label = issue.get("label", "Issue")
                count = issue.get("count", 0)
                with st.expander(f"{idx}. {dataset} | {label} | rows={count}", expanded=False):
                    examples = issue.get("examples")
                    if isinstance(examples, pd.DataFrame) and not examples.empty:
                        render_dataframe(examples, width="stretch")
                    else:
                        st.caption("No example rows available.")
        elif not processed.get("warnings"):
            st.info("No data quality diagnostics available.")

    with st.expander("Section 3: DEGIRO costs", expanded=True):
        st.subheader("Costs & Promo Income Totals Check")
        st.caption(
            "Absolute EUR subtotals per dataset. "
            "Identity: total_costs_eur - total_income_eur = net_cost_minus_income_eur. "
            "Income here includes only promo income (dividend and interest income omitted)."
        )
        costs_summary_df = processed.get("costs_summary_df", pd.DataFrame())
        if costs_summary_df.empty:
            st.info("No cost summary available.")
        else:
            render_dataframe(costs_summary_df, width="stretch")

        st.subheader("Quarterly costs over time (stacked by dataset)")
        st.caption(
            "Quarterly totals of transaction fees, account fees, and dividend tax (stacked bars), "
            "plus secondary-axis lines for number of trades and markets participated in "
            "(markets counted separately per dataset)."
        )
        plotly_chart_stretch(processed.get("fig_degiro_costs_quarterly"))

    with st.expander("Section 4: Outlook (simple)", expanded=True):
        st.subheader("Stock Development Normalized to Rolling Median (Plotly)")
        plotly_chart_stretch(processed.get("fig_normalized"))

        st.subheader("Four tables")
        st.caption(
            "target_per_holding_pct logic: first split by target_etf_fraction, then divide ETF share by "
            "`desired_etf_holdings` and non-ETF share by `desired_non_etf_holdings`."
        )
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "ETFs characteristics",
                "non-ETFs characteristics",
                "Summary with subtotals",
                "Holdings over target threshold",
            ]
        )
        with tab1:
            etf_table = tables_data.get("etf", pd.DataFrame())
            render_dataframe(
                etf_table,
                width="stretch",
                column_formats=four_table_format_map(etf_table),
            )
        with tab2:
            non_etf_table = tables_data.get("non_etf", pd.DataFrame())
            render_dataframe(
                non_etf_table,
                width="stretch",
                column_formats=four_table_format_map(non_etf_table),
            )
        with tab3:
            summary_df = tables_data.get("summary", pd.DataFrame())
            render_dataframe(
                summary_df,
                width="stretch",
                column_formats=four_table_format_map(summary_df),
            )
            combined = value_from_summary(summary_df, "combined value")
            non_etf = value_from_summary(summary_df, "non-ETF value")
            etf = value_from_summary(summary_df, "ETF value")
            cash = value_from_summary(summary_df, "cash position")
            st.text(
                f"Identity check: EUR {etf:,.2f} + EUR {non_etf:,.2f} + EUR {cash:,.2f} "
                f"= EUR {combined:,.2f}"
            )
        with tab4:
            over_target_table = tables_data.get("over_target", pd.DataFrame())
            render_dataframe(
                over_target_table,
                width="stretch",
                column_formats=four_table_format_map(over_target_table),
            )

    with st.expander("Section 5: Performance", expanded=True):
        st.caption(
            "All-time, yearly and quarterly performance with start value, net deposited, end value, "
            "platform costs, trade statistics, and TWR/IRR/XIRR."
        )
        if not isinstance(performance_dashboard, dict) or not performance_dashboard:
            st.info("No performance dashboard available for the current dataset.")
        else:
            summary = performance_dashboard.get("summary", {})
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("All-time TWR", _fmt_pct(_to_float(summary.get("all_time_twr_pct", np.nan))))
            with k2:
                st.metric("All-time IRR", _fmt_pct(_to_float(summary.get("all_time_irr_pct", np.nan))))
            with k3:
                st.metric("All-time XIRR", _fmt_pct(_to_float(summary.get("all_time_xirr_pct", np.nan))))
            with k4:
                st.metric("Start amount", _fmt_eur(_to_float(summary.get("all_time_start_value_eur", np.nan))))
            with k5:
                st.metric("End amount", _fmt_eur(_to_float(summary.get("all_time_end_value_eur", np.nan))))
            st.caption(
                f"All-time net deposited: {_fmt_eur(_to_float(summary.get('all_time_net_deposit_eur', np.nan)))} | "
                f"investment result: {_fmt_eur(_to_float(summary.get('all_time_investment_pnl_eur', np.nan)))}"
            )

            all_time_df = performance_dashboard.get("all_time_df", pd.DataFrame())
            yearly_df = performance_dashboard.get("yearly_df", pd.DataFrame())
            quarterly_df = performance_dashboard.get("quarterly_df", pd.DataFrame())
            benchmark_stats_df = performance_dashboard.get("benchmark_stats_df", pd.DataFrame())

            tab_all, tab_year, tab_quarter, tab_benchmark = st.tabs(
                ["All-time", "Yearly", "Quarterly", "Benchmarks (alpha/beta)"]
            )
            with tab_all:
                if isinstance(all_time_df, pd.DataFrame) and not all_time_df.empty:
                    render_dataframe(all_time_df, width="stretch")
                    all_fig = all_time_df.rename(columns={"net_deposited_eur": "net_flow_eur"})
                    plotly_chart_stretch(build_period_decomposition_figure(all_fig))
                else:
                    st.info("No all-time performance data available.")

            with tab_year:
                if isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty:
                    render_dataframe(yearly_df, width="stretch")
                    y_fig = yearly_df.rename(columns={"net_deposited_eur": "net_flow_eur"})
                    plotly_chart_stretch(build_period_decomposition_figure(y_fig))
                else:
                    st.info("No yearly performance data available.")

            with tab_quarter:
                if isinstance(quarterly_df, pd.DataFrame) and not quarterly_df.empty:
                    render_dataframe(quarterly_df, width="stretch")
                    q_fig = quarterly_df.rename(columns={"net_deposited_eur": "net_flow_eur"})
                    plotly_chart_stretch(build_period_decomposition_figure(q_fig))
                else:
                    st.info("No quarterly performance data available.")

            with tab_benchmark:
                plotly_chart_stretch(processed.get("fig_benchmark"))
                if isinstance(benchmark_stats_df, pd.DataFrame) and not benchmark_stats_df.empty:
                    render_dataframe(benchmark_stats_df, width="stretch")
                else:
                    st.info("No benchmark statistics available.")

    with st.expander("Section 6: AI Generated Spread Analysis", expanded=True):
        st.caption(
            "Spread diagnostics for ETF/non-ETF strategy, cash target, concentration risk, "
            "currency mix, industry/style allocation, correlation overlap risk, and concrete next actions."
        )
        if not isinstance(spread_analysis, dict) or not spread_analysis:
            st.info("No spread analysis available for the current dataset.")
            return

        spread_lines = spread_analysis.get("summary_lines", [])
        if spread_lines:
            st.markdown("**Summary**")
            for line in spread_lines:
                st.write(f"- {line}")

        tab_strategy, tab_alloc, tab_spread_actions, tab_next = st.tabs(
            ["Strategy checks", "Currency/industry/style", "Concentration and actions", "What to do next"]
        )
        with tab_strategy:
            strategy_df = spread_analysis.get("strategy_checks_df", pd.DataFrame())
            if isinstance(strategy_df, pd.DataFrame) and not strategy_df.empty:
                st.markdown("**Strategy checks vs configured targets**")
                render_dataframe(strategy_df, width="stretch")
            else:
                st.info("No strategy checks available.")

            etf_non_etf_df = spread_analysis.get("etf_non_etf_df", pd.DataFrame())
            if isinstance(etf_non_etf_df, pd.DataFrame) and not etf_non_etf_df.empty:
                st.markdown("**ETF and non-ETF mix**")
                render_dataframe(etf_non_etf_df, width="stretch")
            else:
                st.info("No ETF/non-ETF spread table available.")

        with tab_alloc:
            currency_df = spread_analysis.get("currency_allocation_df", pd.DataFrame())
            if isinstance(currency_df, pd.DataFrame) and not currency_df.empty:
                st.markdown("**Currency allocation (EUR converted)**")
                render_dataframe(currency_df, width="stretch")
            else:
                st.info("No currency allocation table available.")

            industry_df = spread_analysis.get("industry_allocation_df", pd.DataFrame())
            if isinstance(industry_df, pd.DataFrame) and not industry_df.empty:
                st.markdown("**Industry allocation**")
                render_dataframe(industry_df, width="stretch")
            else:
                st.info("No industry allocation table available.")

            style_df = spread_analysis.get("style_allocation_df", pd.DataFrame())
            if isinstance(style_df, pd.DataFrame) and not style_df.empty:
                st.markdown("**Style allocation (growth/value/dividend/etc.)**")
                render_dataframe(style_df, width="stretch")
            else:
                st.info("No style allocation table available.")

        with tab_spread_actions:
            concentration_df = spread_analysis.get("concentration_df", pd.DataFrame())
            if isinstance(concentration_df, pd.DataFrame) and not concentration_df.empty:
                st.markdown("**Top holdings concentration**")
                render_dataframe(concentration_df, width="stretch")
            else:
                st.info("No concentration table available.")

            spread_action_df = spread_analysis.get("action_plan_df", pd.DataFrame())
            if isinstance(spread_action_df, pd.DataFrame) and not spread_action_df.empty:
                st.markdown("**Spread action plan**")
                render_dataframe(spread_action_df, width="stretch")
            else:
                st.info("No spread action plan available.")

            corr_df = spread_analysis.get("correlation_warnings_df", pd.DataFrame())
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                st.markdown("**High-correlation warning pairs**")
                render_dataframe(corr_df, width="stretch")
            else:
                st.info("No high-correlation warning pairs detected.")

        with tab_next:
            next_df = spread_analysis.get("what_to_do_next_df", pd.DataFrame())
            if isinstance(next_df, pd.DataFrame) and not next_df.empty:
                render_dataframe(next_df, width="stretch")
            else:
                st.info("No next-step plan available.")


def plotly_chart_stretch(fig: Any) -> None:
    if fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig)


def build_totals_check_text(
    totals_results: dict[str, TotalsResult],
    combined_direct: TotalsResult,
) -> str:
    lines: list[str] = []
    labels = [k for k in totals_results.keys() if k != "Combined"]
    for label in labels:
        t = totals_results[label]
        lines.append(
            f"{label}: EUR {t.positions_value_eur:,.2f} + EUR {t.cash_value_eur:,.2f} "
            f"= EUR {t.total_value_eur:,.2f}"
        )

    combined = totals_results["Combined"]
    lines.append(
        f"Combined (sum of datasets): EUR {combined.positions_value_eur:,.2f} + "
        f"EUR {combined.cash_value_eur:,.2f} = EUR {combined.total_value_eur:,.2f}"
    )
    lines.append(
        f"Combined (direct from merged exports): EUR {combined_direct.positions_value_eur:,.2f} + "
        f"EUR {combined_direct.cash_value_eur:,.2f} = EUR {combined_direct.total_value_eur:,.2f}"
    )

    if len(labels) == 2:
        a = totals_results[labels[0]].total_value_eur
        b = totals_results[labels[1]].total_value_eur
        c = combined.total_value_eur
        lines.append(
            f"Cross-check: EUR {a:,.2f} + EUR {b:,.2f} = EUR {c:,.2f} "
            f"(delta EUR {(a + b - c):,.2f})"
        )
    return "\n".join(lines)


def combine_cash_results_from_datasets(
    per_dataset: dict[str, CashReconciliationResult],
) -> CashReconciliationResult:
    labels = [label for label in per_dataset.keys() if label != "Combined"]
    account_values = [per_dataset[label].cash_from_account_eur for label in labels]
    portfolio_values = [per_dataset[label].cash_from_portfolio_snapshot_eur for label in labels]
    account_total = float(np.nansum(account_values))
    portfolio_total = float(np.nansum(portfolio_values))
    if account_values and not np.isfinite(account_values).any():
        account_total = float("nan")
    if portfolio_values and not np.isfinite(portfolio_values).any():
        portfolio_total = float("nan")
    delta = account_total - portfolio_total

    detail_rows: list[dict[str, Any]] = []
    for label in labels:
        detail = per_dataset[label].detail
        if detail is None or detail.empty:
            continue
        cur = detail.copy()
        cur.insert(0, "dataset", label)
        detail_rows.extend(cur.to_dict(orient="records"))
    combined_detail = pd.DataFrame(detail_rows)

    diagnosis = "Combined by summing per-dataset latest per-currency cash checks."
    return CashReconciliationResult(
        account_label="Combined",
        cash_from_account_eur=account_total,
        cash_from_portfolio_snapshot_eur=portfolio_total,
        cash_delta_eur=delta,
        diagnosis=diagnosis,
        detail=combined_detail,
    )


def value_from_summary(summary_df: pd.DataFrame, metric: str) -> float:
    match = summary_df.loc[summary_df["metric"] == metric, "value_eur"]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def parse_user_float(text: str) -> float | None:
    t = str(text).strip().replace(" ", "")
    if t == "":
        return None
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def save_strategy_file(
    *,
    strategy_file_path: str,
    strategy: dict[str, Any],
    data_sources: dict[str, str],
    logger: Any,
) -> Path:
    raw = str(strategy_file_path).strip()
    if raw == "":
        raise ValueError("strategy_file_path cannot be empty.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "data_sources": data_sources,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if logger is not None:
        logger.info("Saved strategy file: %s", str(path))
    return path


def build_manual_monthly_reference_df() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_text, value_text in MANUAL_MONTHLY_TRACKED_VALUES_RAW:
        tracked_date = pd.to_datetime(date_text, format="%m/%d/%Y", errors="coerce")
        value = parse_user_float(value_text)
        if pd.isna(tracked_date) or value is None:
            continue
        rows.append(
            {
                "tracked_date": tracked_date,
                "manual_tracked_total_eur": float(value),
            }
        )
    return pd.DataFrame(rows)


def build_portfolio_reconciliation_overview_table(
    *,
    totals_results: dict[str, TotalsResult],
    combined_totals_direct: TotalsResult,
    cash_results: dict[str, CashReconciliationResult],
    panel_to_account_label: dict[str, str],
    reported_values: dict[str, dict[str, float]],
) -> pd.DataFrame:
    loaded_panels = sorted(panel_to_account_label.keys())
    reported_total_values: list[float] = []
    for panel_key in loaded_panels:
        values = reported_values.get(panel_key, {})
        if not isinstance(values, dict):
            continue
        raw_total = values.get("reported_total_eur")
        try:
            reported_total = float(raw_total)
        except Exception:
            continue
        if np.isfinite(reported_total):
            reported_total_values.append(reported_total)

    ground_truth_positions = float(combined_totals_direct.positions_value_eur)
    ground_truth_cash = float("nan")
    combined_cash = cash_results.get("Combined")
    if isinstance(combined_cash, CashReconciliationResult):
        ground_truth_cash = float(combined_cash.cash_from_portfolio_snapshot_eur)
    ground_truth_total = (
        float(ground_truth_positions + ground_truth_cash)
        if np.isfinite(ground_truth_positions) and np.isfinite(ground_truth_cash)
        else float("nan")
    )

    if "Combined" in totals_results:
        combined_totals = totals_results["Combined"]
        computed_total = float(combined_totals.total_value_eur)
    else:
        account_labels = [
            panel_to_account_label.get(panel_key)
            for panel_key in loaded_panels
            if panel_to_account_label.get(panel_key) in totals_results
        ]
        computed_total = float(np.nansum([totals_results[label].total_value_eur for label in account_labels]))

    reported_total_sum = float(np.nansum(reported_total_values)) if reported_total_values else float("nan")

    expected = len(loaded_panels)
    provided = len(reported_total_values)
    if expected > 0 and provided == expected:
        reported_note = f"Manual input provided for {provided}/{expected} loaded dataset(s)."
    elif expected > 0:
        reported_note = f"Partial manual input: {provided}/{expected} loaded dataset(s) provided."
    else:
        reported_note = "Manual reported values."

    def delta_vs_ground_truth(value: float) -> float:
        if not np.isfinite(value) or not np.isfinite(ground_truth_total):
            return np.nan
        return float(value - ground_truth_total)

    rows: list[dict[str, Any]] = [
        {
            "portfolio_definition": "Portfolio snapshot (ground truth)",
            "value_eur": ground_truth_total,
            "delta_vs_ground_truth_eur": 0.0 if np.isfinite(ground_truth_total) else np.nan,
            "note": "Sum of Portfolio.csv values (non-cash positions + cash rows) for Dataset A + B.",
        },
        {
            "portfolio_definition": "Computed total",
            "value_eur": computed_total,
            "delta_vs_ground_truth_eur": delta_vs_ground_truth(computed_total),
            "note": "Computed from combined loaded datasets.",
        },
    ]
    if np.isfinite(reported_total_sum):
        rows.append(
            {
                "portfolio_definition": "Reported total (manual)",
                "value_eur": reported_total_sum,
                "delta_vs_ground_truth_eur": delta_vs_ground_truth(reported_total_sum),
                "note": reported_note,
            }
        )
    return pd.DataFrame(rows)


def build_cash_reconciliation_overview_table(
    *,
    cash_results: dict[str, CashReconciliationResult],
    combined_timeseries: Any,
    panel_to_account_label: dict[str, str],
    reported_values: dict[str, dict[str, float]],
) -> pd.DataFrame:
    combined = cash_results.get("Combined")
    if not isinstance(combined, CashReconciliationResult):
        return pd.DataFrame()

    ground_truth = float(combined.cash_from_portfolio_snapshot_eur)
    account_cash = float(combined.cash_from_account_eur)

    reconstructed_cash = float("nan")
    metrics = getattr(combined_timeseries, "metrics", pd.DataFrame())
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        if "cash_from_changes" in metrics.columns:
            reconstructed_cash = float(pd.to_numeric(metrics["cash_from_changes"], errors="coerce").iloc[-1])
        elif "cash" in metrics.columns:
            reconstructed_cash = float(pd.to_numeric(metrics["cash"], errors="coerce").iloc[-1])

    def delta_vs(value: float, reference: float) -> float:
        if not np.isfinite(value) or not np.isfinite(reference):
            return np.nan
        return float(value - reference)

    rows: list[dict[str, Any]] = [
        {
            "cash_definition": "Portfolio snapshot (ground truth)",
            "value_eur": ground_truth,
            "delta_vs_ground_truth_eur": 0.0 if np.isfinite(ground_truth) else np.nan,
            "note": "Sum of cash rows from Portfolio.csv (Dataset A + B).",
        },
        {
            "cash_definition": "Account latest balances",
            "value_eur": account_cash,
            "delta_vs_ground_truth_eur": delta_vs(account_cash, ground_truth),
            "note": "Latest per-currency balances from Account.csv, converted to EUR.",
        },
        {
            "cash_definition": "Reconstructed cash (change cumsum)",
            "value_eur": reconstructed_cash,
            "delta_vs_ground_truth_eur": delta_vs(reconstructed_cash, ground_truth),
            "note": "Daily reconstruction from Account.csv change rows.",
        },
    ]

    loaded_panels = sorted(panel_to_account_label.keys())
    reported_cash_values: list[float] = []
    for panel_key in loaded_panels:
        values = reported_values.get(panel_key, {})
        if not isinstance(values, dict):
            continue
        raw_cash = values.get("reported_cash_eur")
        try:
            cash = float(raw_cash)
        except Exception:
            continue
        if np.isfinite(cash):
            reported_cash_values.append(cash)

    if reported_cash_values:
        reported_total = float(np.nansum(reported_cash_values))
        expected = len(loaded_panels)
        provided = len(reported_cash_values)
        if expected > 0 and provided == expected:
            note = f"Sum of manual reported cash for {provided}/{expected} loaded dataset(s)."
        elif expected > 0:
            note = f"Partial manual input: {provided}/{expected} loaded dataset(s) reported."
        else:
            note = "Manual reported cash."
        rows.append(
            {
                "cash_definition": "Reported cash (manual)",
                "value_eur": reported_total,
                "delta_vs_ground_truth_eur": delta_vs(reported_total, ground_truth),
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def build_snapshot_vs_rebuilt_table(
    *,
    totals_results: dict[str, TotalsResult],
    per_dataset_timeseries: dict[str, Any],
    combined_timeseries: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label, totals in totals_results.items():
        if label == "Combined":
            continue
        ts = per_dataset_timeseries.get(label)
        if ts is None or ts.metrics.empty:
            continue
        rebuilt_cash = float(ts.metrics.iloc[-1]["cash"])
        rebuilt_positions = float(ts.metrics.iloc[-1]["positions_value"])
        rebuilt_total = float(ts.metrics.iloc[-1]["portfolio_value"])
        rows.append(
            {
                "dataset": label,
                "snapshot_cash_eur": totals.cash_value_eur,
                "rebuilt_cash_eur": rebuilt_cash,
                "cash_delta_eur": rebuilt_cash - totals.cash_value_eur,
                "snapshot_positions_eur": totals.positions_value_eur,
                "rebuilt_positions_eur": rebuilt_positions,
                "positions_delta_eur": rebuilt_positions - totals.positions_value_eur,
                "snapshot_total_eur": totals.total_value_eur,
                "rebuilt_total_eur": rebuilt_total,
                "total_delta_eur": rebuilt_total - totals.total_value_eur,
            }
        )

    if "Combined" in totals_results and combined_timeseries is not None and not combined_timeseries.metrics.empty:
        totals = totals_results["Combined"]
        rebuilt_cash = float(combined_timeseries.metrics.iloc[-1]["cash"])
        rebuilt_positions = float(combined_timeseries.metrics.iloc[-1]["positions_value"])
        rebuilt_total = float(combined_timeseries.metrics.iloc[-1]["portfolio_value"])
        rows.append(
            {
                "dataset": "Combined",
                "snapshot_cash_eur": totals.cash_value_eur,
                "rebuilt_cash_eur": rebuilt_cash,
                "cash_delta_eur": rebuilt_cash - totals.cash_value_eur,
                "snapshot_positions_eur": totals.positions_value_eur,
                "rebuilt_positions_eur": rebuilt_positions,
                "positions_delta_eur": rebuilt_positions - totals.positions_value_eur,
                "snapshot_total_eur": totals.total_value_eur,
                "rebuilt_total_eur": rebuilt_total,
                "total_delta_eur": rebuilt_total - totals.total_value_eur,
            }
        )

    return pd.DataFrame(rows)


def build_positions_reconciliation_table(
    *,
    portfolio_df: pd.DataFrame,
    ts_result: Any,
    max_rows: int = 200,
) -> pd.DataFrame:
    if portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()
    if ts_result is None:
        return pd.DataFrame()
    if getattr(ts_result, "positions", pd.DataFrame()).empty:
        return pd.DataFrame()
    if getattr(ts_result, "prices_eur", pd.DataFrame()).empty:
        return pd.DataFrame()

    holdings = portfolio_df.loc[~portfolio_df["is_cash_like"].fillna(False)].copy()
    if holdings.empty:
        return pd.DataFrame()

    def _first_non_empty(values: pd.Series) -> Any:
        for value in values:
            if pd.notna(value) and str(value).strip() != "":
                return value
        return np.nan

    snapshot = (
        holdings.groupby("instrument_id", as_index=False)
        .agg(
            product=("product", _first_non_empty),
            snapshot_quantity=("quantity", "sum"),
            snapshot_value_eur=("value_eur", "sum"),
        )
        .copy()
    )

    latest_pos = ts_result.positions.iloc[-1].astype(float)
    latest_price = ts_result.prices_eur.iloc[-1].astype(float)
    rebuilt = pd.DataFrame(
        {
            "instrument_id": latest_pos.index.astype(str),
            "rebuilt_quantity": latest_pos.values,
            "rebuilt_price_eur": latest_price.reindex(latest_pos.index).values,
        }
    )
    rebuilt["rebuilt_value_eur"] = rebuilt["rebuilt_quantity"] * rebuilt["rebuilt_price_eur"]

    out = snapshot.merge(rebuilt, on="instrument_id", how="outer")
    out["snapshot_quantity"] = out["snapshot_quantity"].fillna(0.0)
    out["snapshot_value_eur"] = out["snapshot_value_eur"].fillna(0.0)
    out["rebuilt_quantity"] = out["rebuilt_quantity"].fillna(0.0)
    out["rebuilt_value_eur"] = out["rebuilt_value_eur"].fillna(0.0)

    out["snapshot_price_eur"] = np.where(
        out["snapshot_quantity"] != 0,
        out["snapshot_value_eur"] / out["snapshot_quantity"],
        np.nan,
    )
    out["quantity_delta"] = out["rebuilt_quantity"] - out["snapshot_quantity"]
    out["value_delta_eur"] = out["rebuilt_value_eur"] - out["snapshot_value_eur"]
    out["value_delta_from_qty_eur"] = out["quantity_delta"] * out["rebuilt_price_eur"]
    out["value_delta_from_price_eur"] = out["snapshot_quantity"] * (
        out["rebuilt_price_eur"] - out["snapshot_price_eur"]
    )
    out["decomposition_residual_eur"] = out["value_delta_eur"] - (
        out["value_delta_from_qty_eur"] + out["value_delta_from_price_eur"]
    )

    out["abs_value_delta_eur"] = out["value_delta_eur"].abs()
    out = out.sort_values("abs_value_delta_eur", ascending=False).drop(columns=["abs_value_delta_eur"])
    if len(out) > max_rows:
        out = out.head(max_rows)
    return out.reset_index(drop=True)


def build_latest_account_balance_check_table(
    cash_results: dict[str, CashReconciliationResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, result in cash_results.items():
        detail = result.detail.copy() if isinstance(result.detail, pd.DataFrame) else pd.DataFrame()
        if detail.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "source_dataset": dataset,
                    "currency": "<none>",
                    "latest_datetime": "<none>",
                    "raw_balance": np.nan,
                    "balance_eur_csv": np.nan,
                    "balance_eur_computed": np.nan,
                    "note": "No account balance rows",
                }
            )
            continue
        for row in detail.itertuples(index=False):
            source_dataset = getattr(row, "dataset", dataset)
            rows.append(
                {
                    "dataset": dataset,
                    "source_dataset": source_dataset,
                    "currency": row.currency,
                    "latest_datetime": row.datetime.strftime("%Y-%m-%d %H:%M:%S")
                    if pd.notna(row.datetime)
                    else "<NaN>",
                    "raw_balance": row.raw_balance,
                    "balance_eur_csv": row.balance_eur,
                    "balance_eur_computed": row.balance_eur_computed,
                    "note": "",
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "source_dataset": dataset,
                "currency": "TOTAL",
                "latest_datetime": "",
                "raw_balance": np.nan,
                "balance_eur_csv": np.nan,
                "balance_eur_computed": result.cash_from_account_eur,
                "note": "sum of latest per currency",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["raw_balance", "balance_eur_csv", "balance_eur_computed"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_costs_summary_table(account_category_summary_df: pd.DataFrame) -> pd.DataFrame:
    if account_category_summary_df is None or account_category_summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    by_dataset = account_category_summary_df.groupby("dataset", dropna=False)
    for dataset, group in by_dataset:
        def outflow_of(category: str) -> float:
            sel = group[group["category"] == category]
            if sel.empty:
                return 0.0
            return float(sel["outflow_eur"].sum())

        def inflow_of(category: str) -> float:
            sel = group[group["category"] == category]
            if sel.empty:
                return 0.0
            return float(sel["inflow_eur"].sum())

        transaction_fee = outflow_of("transaction_fee")
        account_fee = outflow_of("account_fee")
        dividend_tax = outflow_of("dividend_tax")
        total_costs = transaction_fee + account_fee + dividend_tax

        promo_income = inflow_of("promo")
        total_income = promo_income

        rows.append(
            {
                "dataset": dataset,
                "transaction_fee_eur": transaction_fee,
                "account_fee_eur": account_fee,
                "dividend_tax_eur": dividend_tax,
                "total_costs_eur": total_costs,
                "promo_income_eur": promo_income,
                "total_income_eur": total_income,
                "net_cost_minus_income_eur": total_costs - total_income,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_quarterly_costs_table(
    *,
    account_df: pd.DataFrame,
    transactions_df: pd.DataFrame | None = None,
    account_label_to_dataset_name: dict[str, str] | None = None,
) -> pd.DataFrame:
    if account_df is None or account_df.empty:
        return pd.DataFrame()

    df = account_df.copy()
    df["datetime"] = pd.to_datetime(df.get("datetime"), errors="coerce")
    df = df[df["datetime"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["type"] = df.get("type", pd.Series("other", index=df.index)).fillna("other").astype(str)
    df["change_eur"] = pd.to_numeric(df.get("change_eur"), errors="coerce").fillna(0.0)
    cost_types = {"transaction_fee", "account_fee", "dividend_tax"}
    df = df[df["type"].isin(cost_types)].copy()
    if df.empty:
        return pd.DataFrame()

    labels = account_label_to_dataset_name or {}
    df["dataset"] = (
        df.get("account_label", pd.Series("", index=df.index))
        .astype(str)
        .map(lambda value: labels.get(value, value))
    )
    df["quarter"] = df["datetime"].dt.to_period("Q").astype(str)
    df["cost_eur"] = np.where(df["change_eur"] < 0.0, -df["change_eur"], 0.0)
    df["transaction_fee_eur"] = np.where(df["type"].eq("transaction_fee"), df["cost_eur"], 0.0)
    df["account_fee_eur"] = np.where(df["type"].eq("account_fee"), df["cost_eur"], 0.0)
    df["dividend_tax_eur"] = np.where(df["type"].eq("dividend_tax"), df["cost_eur"], 0.0)

    grouped = (
        df.groupby(["quarter", "dataset"], dropna=False, as_index=False)
        .agg(
            transaction_fee_eur=("transaction_fee_eur", "sum"),
            account_fee_eur=("account_fee_eur", "sum"),
            dividend_tax_eur=("dividend_tax_eur", "sum"),
            total_costs_eur=("cost_eur", "sum"),
        )
        .sort_values(["quarter", "dataset"])
        .reset_index(drop=True)
    )

    if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
        tx = transactions_df.copy()
        tx["datetime"] = pd.to_datetime(tx.get("datetime"), errors="coerce")
        tx = tx[tx["datetime"].notna()].copy()
        if not tx.empty:
            tx["dataset"] = (
                tx.get("account_label", pd.Series("", index=tx.index))
                .astype(str)
                .map(lambda value: labels.get(value, value))
            )
            tx["quarter"] = tx["datetime"].dt.to_period("Q").astype(str)

            quantity = pd.to_numeric(tx.get("quantity"), errors="coerce")
            total_eur = pd.to_numeric(tx.get("total_eur"), errors="coerce")
            trade_mask = (quantity.notna() & quantity.ne(0.0)) | (
                quantity.isna() & total_eur.notna() & total_eur.ne(0.0)
            )
            tx = tx.loc[trade_mask].copy()

            if not tx.empty:
                tx["market"] = (
                    tx.get("type", pd.Series("", index=tx.index))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )
                tx["market"] = tx["market"].replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})

                activity = (
                    tx.groupby(["quarter", "dataset"], dropna=False, as_index=False)
                    .agg(
                        trade_count=("quarter", "size"),
                        market_count=("market", lambda s: int(pd.Series(s).dropna().nunique())),
                    )
                    .reset_index(drop=True)
                )
                grouped = grouped.merge(activity, on=["quarter", "dataset"], how="outer")

    for col in [
        "transaction_fee_eur",
        "account_fee_eur",
        "dividend_tax_eur",
        "total_costs_eur",
        "trade_count",
        "market_count",
    ]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)

    grouped["quarter"] = grouped["quarter"].astype(str)
    grouped["dataset"] = grouped["dataset"].fillna("Unknown").astype(str)
    grouped = grouped.sort_values(["quarter", "dataset"]).reset_index(drop=True)
    return grouped


def build_cash_source_raw_table(account_df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if account_df is None or account_df.empty:
        return pd.DataFrame()
    cols = [
        "account_label",
        "datetime",
        "type",
        "description",
        "currency",
        "raw_change",
        "change_eur",
        "raw_balance",
        "balance_eur",
        "fx_rate",
        "is_external_flow",
        "is_internal_transfer",
        "is_cost",
        "is_income",
    ]
    present = [c for c in cols if c in account_df.columns]
    out = account_df[present].copy().sort_values("datetime")
    out.insert(0, "row_number", range(1, len(out) + 1))
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["raw_change", "change_eur", "raw_balance", "balance_eur", "fx_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if len(out) > max_rows:
        return out.tail(max_rows).reset_index(drop=True)
    return out.reset_index(drop=True)


def build_cash_timeline_table(metrics_df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty or "cash" not in metrics_df.columns:
        return pd.DataFrame()
    out = metrics_df.copy()
    out = out.loc[
        :,
        [
            c
            for c in [
                "cash",
                "cash_from_changes",
                "cash_from_statement",
                "total_deposits",
                "positions_value",
                "portfolio_value",
            ]
            if c in out.columns
        ],
    ]
    out = out.rename(
        columns={
            "cash": "cash_eur",
            "cash_from_changes": "cash_from_changes_eur",
            "cash_from_statement": "cash_from_statement_eur",
            "total_deposits": "total_deposits_eur",
            "positions_value": "positions_value_eur",
            "portfolio_value": "portfolio_value_eur",
        }
    )
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.insert(1, "cash_day_change_eur", out["cash_eur"].diff().fillna(0.0))
    if "total_deposits_eur" in out.columns:
        out.insert(
            2,
            "external_flow_day_eur",
            out["total_deposits_eur"].diff().fillna(out["total_deposits_eur"]),
        )
    if "cash_from_statement_eur" in out.columns and "cash_from_changes_eur" in out.columns:
        out["cash_statement_vs_changes_delta_eur"] = (
            out["cash_from_statement_eur"] - out["cash_from_changes_eur"]
        )
    out.insert(0, "row_number", range(1, len(out) + 1))
    if len(out) > max_rows:
        return out.tail(max_rows).reset_index(drop=True)
    return out.reset_index(drop=True)


def build_daily_close_holdings_table(
    *,
    ts_result: Any,
    instruments: pd.DataFrame,
    max_rows: int = 5000,
) -> pd.DataFrame:
    if ts_result is None:
        return pd.DataFrame()
    if getattr(ts_result, "positions", pd.DataFrame()).empty:
        return pd.DataFrame()
    if getattr(ts_result, "prices_eur", pd.DataFrame()).empty:
        return pd.DataFrame()
    if getattr(ts_result, "metrics", pd.DataFrame()).empty:
        return pd.DataFrame()

    positions = ts_result.positions.copy()
    prices_eur = ts_result.prices_eur.copy()
    metrics = ts_result.metrics.copy()

    shared_cols = [c for c in positions.columns if c in prices_eur.columns]
    if not shared_cols:
        return pd.DataFrame()

    positions = positions[shared_cols].sort_index()
    prices_eur = prices_eur[shared_cols].reindex(positions.index)
    metrics = metrics.reindex(positions.index)

    # Completed days: all required totals are present and every held instrument has a EUR close.
    held_missing_price = (positions.ne(0.0) & prices_eur.isna()).any(axis=1)
    required_metric_cols = ["cash", "portfolio_value", "total_deposits"]
    if not set(required_metric_cols).issubset(set(metrics.columns)):
        return pd.DataFrame()
    metric_complete = metrics[required_metric_cols].notna().all(axis=1)
    completed_days = (~held_missing_price) & metric_complete
    if not completed_days.any():
        return pd.DataFrame()

    positions = positions.loc[completed_days]
    prices_eur = prices_eur.loc[completed_days]
    metrics = metrics.loc[completed_days]
    values_eur = positions * prices_eur

    instrument_lookup = pd.DataFrame()
    if isinstance(instruments, pd.DataFrame) and not instruments.empty and "instrument_id" in instruments.columns:
        instrument_lookup = instruments.drop_duplicates(subset=["instrument_id"], keep="first").copy()

    id_to_label: dict[str, str] = {}
    used_labels: set[str] = set()
    for instrument_id in shared_cols:
        label = str(instrument_id)
        if not instrument_lookup.empty:
            match = instrument_lookup.loc[instrument_lookup["instrument_id"].eq(instrument_id)]
            if not match.empty:
                row = match.iloc[0]
                product = str(row.get("product", "")).strip()
                ticker = str(row.get("ticker", "")).strip()
                if product and product.lower() != "nan":
                    label = product
                if ticker and ticker.lower() != "nan":
                    label = f"{label} ({ticker})"
        unique = label
        suffix = 2
        while unique in used_labels:
            unique = f"{label} #{suffix}"
            suffix += 1
        used_labels.add(unique)
        id_to_label[instrument_id] = unique

    out = pd.DataFrame(index=positions.index)
    for instrument_id in shared_cols:
        label = id_to_label.get(instrument_id, str(instrument_id))
        out[f"count::{label}"] = pd.to_numeric(positions[instrument_id], errors="coerce")
        out[f"value_eur::{label}"] = pd.to_numeric(values_eur[instrument_id], errors="coerce")

    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.insert(1, "cumulative_net_external_flow_eur", pd.to_numeric(metrics["total_deposits"], errors="coerce"))
    out.insert(2, "value_eur::Cash", pd.to_numeric(metrics["cash"], errors="coerce"))
    out.insert(3, "total_portfolio_value_eur", pd.to_numeric(metrics["portfolio_value"], errors="coerce"))
    out = out.reset_index(drop=True)

    if len(out) > max_rows:
        return out.tail(max_rows).reset_index(drop=True)
    return out
