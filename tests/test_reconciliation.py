from __future__ import annotations

import pandas as pd

from src.reconciliation import TotalsResult, cash_row_mask, combine_totals, reconcile_dataset


def test_cash_reconciliation_basic_math() -> None:
    portfolio = pd.DataFrame(
        [
            {"product": "CASH & CASH FUND & FTX CASH (EUR)", "isin": None, "value_eur": 100.0, "is_cash_like": True},
            {"product": "TEST STOCK", "isin": "US0000000001", "value_eur": 900.0, "is_cash_like": False},
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-02-18 10:00:00"),
                "currency": "EUR",
                "raw_balance": 110.0,
                "balance_eur": 110.0,
            }
        ]
    )
    cash_result, totals_result = reconcile_dataset(
        account_label="A",
        portfolio=portfolio,
        account=account,
        fx_lookup=None,
    )

    assert cash_result.cash_from_account_eur == 110.0
    assert cash_result.cash_from_portfolio_snapshot_eur == 100.0
    assert cash_result.cash_delta_eur == 10.0
    assert totals_result.positions_value_eur == 900.0
    assert totals_result.cash_value_eur == 110.0
    assert totals_result.total_value_eur == 1010.0


def test_combined_totals_sum_invariant() -> None:
    totals = {
        "Dataset A": TotalsResult(
            account_label="Dataset A",
            positions_value_eur=1000.0,
            cash_value_eur=200.0,
            cash_source="account balance",
            total_value_eur=1200.0,
        ),
        "Dataset B": TotalsResult(
            account_label="Dataset B",
            positions_value_eur=300.0,
            cash_value_eur=50.0,
            cash_source="account balance",
            total_value_eur=350.0,
        ),
    }
    combined = combine_totals(totals)
    assert combined.positions_value_eur == 1300.0
    assert combined.cash_value_eur == 250.0
    assert combined.total_value_eur == 1550.0
    assert totals["Dataset A"].total_value_eur + totals["Dataset B"].total_value_eur == combined.total_value_eur


def test_cash_row_mask_does_not_treat_etf_currency_suffix_as_cash() -> None:
    portfolio = pd.DataFrame(
        [
            {"product": "ISHARES EUROPEAN PROPERTY YIELD UCITS ETF EUR D", "isin": "IE00B0M63284"},
            {"product": "FLATEX EURO BANKACCOUNT", "isin": "NLFLATEXACNT"},
        ]
    )
    mask = cash_row_mask(portfolio)
    assert bool(mask.iloc[0]) is False
    assert bool(mask.iloc[1]) is True
