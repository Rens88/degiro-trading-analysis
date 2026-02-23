# Interdependencies Map (AI + Human)

This file documents high-impact module contracts and cross-module dependencies.
Keep this updated whenever you rename shared keys, change table columns, or alter strategy/config schema.

## End-to-end Data Flow

1) `src/data_import.py`
- Parses CSV files and normalizes columns.
- Produces `LoadedDataset` objects.

2) `src/reconciliation.py`
- Computes per-dataset and combined cash/total values.
- Produces `CashReconciliationResult` and `TotalsResult`.

3) `src/portfolio_timeseries.py`
- Builds daily positions, prices (EUR), and portfolio metrics.
- Produces `TimeSeriesResult`.

4) `src/tables.py`
- Computes ETF/non-ETF tables and over-target holdings table.
- Uses `TotalsResult` and normalized holdings data.

5) `src/insights.py`
- Computes performance dashboard and spread analysis.
- Consumes metrics/transactions/account + holdings + strategy + optional prices.

6) `src/plots.py`
- Renders Plotly figures from already-processed frames.

7) `src/app.py`
- Orchestrates Streamlit FSM/session state.
- Single producer of `workflow["processed"]` via `process_loaded_datasets`.
- Renders all sections from keys in `workflow["processed"]`.

8) `src/strategy_check.py`
- Reuses import/reconciliation/tables/insights pipeline in CLI mode.
- Startup integration contract: returns exit code `10` when action is required.

## Shared Contract: `workflow["processed"]` (producer: `src/app.py`)

Primary consumer: `render_main` in `src/app.py`.
If you rename/remove keys below, update `render_main` and tests.

- `timeseries`, `per_dataset_timeseries`
- `tables`
- `spread_analysis`
- `performance_dashboard`
- `fig_performance_over_time`, `fig_holdings_over_time`, `fig_benchmark`, `fig_normalized`
- `cash_results`, `totals_results`, `combined_totals_direct`
- `warnings`, `issue_tables`

## Shared Contract: `spread_analysis` dict (producer: `build_ai_spread_analysis`)

Consumers:
- Streamlit section 5 in `src/app.py`
- CLI action detection in `src/strategy_check.py`

Important keys:
- `summary_lines`
- `strategy_checks_df`, `etf_non_etf_df`, `currency_allocation_df`, `industry_allocation_df`, `style_allocation_df`
- `concentration_df`, `action_plan_df`, `what_to_do_next_df`
- `correlation_warnings_df`

Action-gating compatibility:
- `src/strategy_check.py::_spread_actions_require_action` prefers a boolean `requires_action` column when present.

## Shared Contract: `performance_dashboard` dict (producer: `build_performance_dashboard`)

Consumer:
- Streamlit section 4 in `src/app.py`

Important keys:
- `all_time_df`, `yearly_df`, `quarterly_df`, `benchmark_stats_df`
- `summary` with:
  - `all_time_twr_pct`
  - `all_time_irr_pct`
  - `all_time_xirr_pct`
  - `all_time_start_value_eur`
  - `all_time_end_value_eur`
  - `all_time_net_deposit_eur`
  - `all_time_investment_pnl_eur`

## Shared Strategy Schema (config + UI + CLI)

Source of defaults:
- `src/config.py`

Must stay aligned in:
- `src/app.py` sidebar params
- `src/strategy_check.py::_strategy_with_defaults`
- `strategy/spread_strategy.json` saved values

High-impact fields:
- `target_etf_fraction`
- `desired_etf_holdings`, `desired_non_etf_holdings`
- `min_over_value_eur`
- `target_cash_pct`
- `max_single_holding_pct`, `max_top5_holdings_pct`, `max_single_currency_pct`, `max_single_industry_pct`
- `max_pair_correlation`
- `min_total_holdings`
- `target_currency_pct`, `target_industry_pct`, `target_style_pct`
- `holding_category_overrides`

## Edit Checklist for Risky Changes

- If you change normalized column names in `src/data_import.py`:
  - Review `src/reconciliation.py`, `src/portfolio_timeseries.py`, `src/tables.py`, `src/insights.py`, and tests.

- If you change `TimeSeriesResult` fields:
  - Review `src/app.py`, `src/insights.py`, `src/plots.py`, and tests.

- If you change over-target or spread action columns:
  - Review Streamlit rendering and CLI action gating in `src/strategy_check.py`.

- If you change any `workflow["processed"]` key:
  - Review all section rendering in `src/app.py` and update tests.

- If you change strategy exit logic:
  - Keep `EXIT_ACTION_REQUIRED = 10` behavior for `run_strategy_check_startup.bat`.
