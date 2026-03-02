# DEGIRO Trading Analysis (Reconciliation-First)

Streamlit app for DEGIRO exports with strict subtotal checks and reconciliation-first analytics.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Inputs

Each dataset must include exactly:

- `Transactions.csv`
- `Portfolio.csv`
- `Account.csv`

The sidebar supports up to 2 datasets (`Dataset A`, `Dataset B`), each loaded independently:

- `Use bundled sample data` mode:
  - Reads from `data/pensioenbeleggen` or `data/spaarbeleggen`.
- Upload mode:
  - Upload exactly the three required files above.
- Click `Load` per dataset (nothing is auto-loaded).

## App Output Sections (Main Panel)

1. `Section 1 — Portfolio over time (Plotly)`
   - positions value, cash, profit, total deposits, portfolio value, simple return (%)
   - latest totals check with explicit identity: `positions + cash = total`
   - cash reconciliation table and diagnosis
   - dataset and combined totals checks (including `A + B = combined`)
2. `Section 2 — Stock development normalized to rolling median (Plotly)`
   - `normalized_price(t) = price(t) / rolling_median(price, median_window_months)`
3. `Section 3 — Four tables`
   - ETFs characteristics
   - non-ETFs characteristics
   - summary with explicit subtotals and identity:
     - `ETF value + non-ETF value + cash = combined value`
   - holdings above target allocation threshold

## Reconciliation Logic

### Cash Reconciliation

Per dataset and combined:

- `cash_from_account_eur`: latest account balances per currency converted to EUR and summed
- `cash_from_portfolio_snapshot_eur`: sum of cash-like rows in portfolio snapshot
- `cash_delta_eur = cash_from_account_eur - cash_from_portfolio_snapshot_eur`

A diagnosis hint is shown for likely causes:

- missing FX conversion
- missing cash rows in portfolio snapshot
- timing/rounding gaps

### Totals Check

Per dataset and combined:

- `positions_value_eur`
- `cash_value_eur` (prefers account cash; falls back to portfolio cash if needed)
- `total_value_eur = positions_value_eur + cash_value_eur`

If two datasets are loaded:

- shows Dataset A totals
- shows Dataset B totals
- shows Combined totals
- verifies `A total + B total = combined total`

## Instrument Classification (`ticker_classification_complete.csv`)

Ticker and metadata resolution uses `ticker_classification_complete.csv` as the
single source of truth:

- `ticker`, `currency`, and `asset_class` are used during ingestion.
- `primary_style`, `secondary_factor`, and `gics_*` columns are used in spread
  analysis and Section 6 allocation views.

If a row is missing, the app raises a user-facing error listing the affected
instrument(s) and asks you to add the row in
`ticker_classification_complete.csv`.

## Tests

```bash
pytest
```

Covered:

- CSV normalization for comma/semicolon and English/Dutch header variants
- cash reconciliation math invariants
- combined totals invariant (`A + B = combined`)
- warnings for critical NaNs (`value_eur`, `balance_eur`, prices path warnings)
