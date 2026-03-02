"""
AGENT_NOTE: Shared configuration and constants.

Interdependencies:
- Defaults here are consumed by `src/app.py` sidebar controls and
  `src/strategy_check.py::_strategy_with_defaults`.
- Plot color constants are used by `src/plots.py`.
- State constants are used by Streamlit FSM helpers in `src/app.py`.

When editing:
- Keep strategy defaults synchronized across app UI, CLI strategy check, and
  persisted JSON (`strategy/spread_strategy.json`).
- See `src/INTERDEPENDENCIES.md` for the cross-module contract map.
"""

from __future__ import annotations

from pathlib import Path

# Required DEGIRO export files for each dataset.
REQUIRED_DATASET_FILES = ("Transactions.csv", "Portfolio.csv", "Account.csv")

# Bundled sample datasets.
SAMPLE_DATASETS = {
    "pensioenbeleggen": Path("data/pensioenbeleggen"),
    "spaarbeleggen": Path("data/spaarbeleggen"),
    "pensioenbeleggen_fixed": Path("data/pensioenbeleggen_fixed"),
    "spaarbeleggen_fixed": Path("data/spaarbeleggen_fixed"),
    "pensioenbeleggen_broken": Path("data/pensioenbeleggen_broken"),
    "spaarbeleggen_broken": Path("data/spaarbeleggen_broken"),
}

# FSM state names.
STATE_LOADING_DATA = "STATE_LOADING_DATA"
STATE_SELECTING_PARAMS = "STATE_SELECTING_PARAMS"
STATE_PROCESSING = "STATE_PROCESSING"
STATE_VIEWING_RESULTS = "STATE_VIEWING_RESULTS"
STATE_EXPORTING = "STATE_EXPORTING"
STATE_RELOADING_EXPORT = "STATE_RELOADING_EXPORT"

STATE_ORDER = [
    STATE_LOADING_DATA,
    STATE_SELECTING_PARAMS,
    STATE_PROCESSING,
    STATE_VIEWING_RESULTS,
    STATE_EXPORTING,
    STATE_RELOADING_EXPORT,
]

# Plot colors.
BASE_BLUE = "#01378A"
BASE_RED = "#E1011A"
BASE_ORANGE = "#EA6D08"
BASE_YELLOW = "#F4C300"
BASE_GREEN = "#009F3D"

# Defaults.
DEFAULT_LOOKBACK_MONTHS = 24
DEFAULT_MEDIAN_WINDOW_MONTHS = 6
DEFAULT_TARGET_ETF_FRACTION = 0.50
DEFAULT_MIN_OVER_VALUE_EUR = 400.0
DEFAULT_DESIRED_ETF_HOLDINGS = 4
DEFAULT_DESIRED_NON_ETF_HOLDINGS = 12
DEFAULT_TARGET_CASH_PCT = 10.0
DEFAULT_MAX_SINGLE_HOLDING_PCT = 12.0
DEFAULT_MAX_TOP5_HOLDINGS_PCT = 55.0
DEFAULT_MAX_SINGLE_CURRENCY_PCT = 65.0
DEFAULT_MAX_SINGLE_INDUSTRY_PCT = 35.0
DEFAULT_MAX_PAIR_CORRELATION = 0.90
DEFAULT_MIN_TOTAL_HOLDINGS = 12
DEFAULT_STRATEGY_FILE_PATH = Path("strategy/spread_strategy.json")
DEFAULT_STRATEGY_DATASET_A_DIR = Path("data/pensioenbeleggen")
DEFAULT_STRATEGY_DATASET_B_DIR = Path("data/spaarbeleggen")
DEFAULT_TICKER_CLASSIFICATION_PATH = Path("ticker_classification_complete.csv")
DEFAULT_TARGET_CURRENCY_PCT = {"EUR": 50.0, "USD": 50.0}
DEFAULT_TARGET_INDUSTRY_PCT: dict[str, float] = {}
DEFAULT_TARGET_STYLE_PCT = {"Growth": 34.0, "Value": 33.0, "Dividend": 33.0}

# Fallback ETF list from legacy project.
DEFAULT_ETF_ISINS = {
    "IE00B0M63284",
    "IE00B1XNHC34",
    "IE00B4L5Y983",
    "IE00B5M1WJ87",
    "NL0010408704",
    "IE00BKY58G26",
    "IE00B3XXRP09",
    "FR0000021842",
}
