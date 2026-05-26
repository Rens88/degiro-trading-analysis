from __future__ import annotations

import pandas as pd

from src.app import _dataset_integrity_warnings
from src.data_import import LoadedDataset


def _dataset(
    *,
    account_label: str,
    tx_rows: list[dict[str, object]],
    portfolio_rows: list[dict[str, object]],
    account_rows: list[dict[str, object]],
) -> LoadedDataset:
    return LoadedDataset(
        account_label=account_label,
        transactions=pd.DataFrame(tx_rows),
        portfolio=pd.DataFrame(portfolio_rows),
        account=pd.DataFrame(account_rows),
        instruments=pd.DataFrame(),
        warnings=[],
        issues=[],
    )


def test_dataset_integrity_warnings_flag_identical_account_and_portfolio_files() -> None:
    account_rows = [
        {
            "datetime": pd.Timestamp("2026-01-01 09:00:00"),
            "description": "flatex Storting",
            "currency": "EUR",
            "raw_change": 500.0,
            "raw_balance": 500.0,
            "account_label": "shared",
        }
    ]
    portfolio_rows = [
        {
            "instrument_id": "ETF1",
            "product": "ETF One",
            "quantity": 2.0,
            "value_eur": 250.0,
            "account_label": "shared",
        }
    ]
    left = _dataset(
        account_label="Dataset A",
        tx_rows=[
            {
                "datetime": pd.Timestamp("2026-01-02 10:00:00"),
                "instrument_id": "ETF1",
                "quantity": 2.0,
                "total_eur": -250.0,
                "account_label": "A",
            }
        ],
        portfolio_rows=portfolio_rows,
        account_rows=account_rows,
    )
    right = _dataset(
        account_label="Dataset B",
        tx_rows=[
            {
                "datetime": pd.Timestamp("2026-01-03 10:00:00"),
                "instrument_id": "STOCK2",
                "quantity": 1.0,
                "total_eur": -100.0,
                "account_label": "B",
            }
        ],
        portfolio_rows=portfolio_rows,
        account_rows=account_rows,
    )

    warnings = _dataset_integrity_warnings({"dataset_a": left, "dataset_b": right})

    assert any("`Account.csv`" in warning for warning in warnings)
    assert any("`Portfolio.csv`" in warning for warning in warnings)
