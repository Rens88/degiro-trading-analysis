from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio_timeseries import build_daily_cash_series_from_changes, compute_portfolio_timeseries


def test_price_nans_emit_explicit_warnings(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([np.nan, np.nan, np.nan], index=index, name=ticker)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 09:00:00"),
                "currency": "EUR",
                "raw_balance": 100.0,
                "balance_eur": 100.0,
                "raw_change": 100.0,
                "change_eur": 100.0,
                "type": "external_deposit",
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        expected_latest_cash_eur=100.0,
        cache_dir="cache",
        logger=None,
    )
    assert any("missing values after fill" in w for w in out.warnings)


def test_cash_deposits_profit_and_return_are_computed(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([100.0, 100.0, 100.0], index=index, name=ticker)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 09:00:00"),
                "currency": "EUR",
                "raw_balance": 1000.0,
                "balance_eur": 1000.0,
                "raw_change": 1000.0,
                "change_eur": 1000.0,
                "type": "external_deposit",
            },
            {
                "datetime": pd.Timestamp("2026-01-02 09:00:00"),
                "currency": "EUR",
                "raw_balance": 990.0,
                "balance_eur": 990.0,
                "raw_change": -10.0,
                "change_eur": -10.0,
                "type": "fee",
            },
            {
                "datetime": pd.Timestamp("2026-01-03 09:00:00"),
                "currency": "EUR",
                "raw_balance": 890.0,
                "balance_eur": 890.0,
                "raw_change": -100.0,
                "change_eur": -100.0,
                "type": "external_withdrawal",
            },
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        expected_latest_cash_eur=None,
        cache_dir="cache",
        logger=None,
    )
    metrics = out.metrics
    assert metrics["cash"].nunique() > 1
    assert float(metrics.iloc[-1]["cash"]) == 890.0
    assert float(metrics.iloc[-1]["total_deposits"]) == 900.0
    assert not pd.isna(metrics.iloc[-1]["simple_return"])
    assert float(metrics.iloc[-1]["profit"]) != float(metrics.iloc[-1]["portfolio_value"])


def test_expected_latest_cash_parameter_does_not_shift_history(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([10.0, 10.0], index=index, name=ticker)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 09:00:00"),
                "currency": "EUR",
                "raw_balance": 500.0,
                "balance_eur": 500.0,
                "raw_change": 500.0,
                "change_eur": 500.0,
                "type": "external_deposit",
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        expected_latest_cash_eur=5000.0,
        cache_dir="cache",
        logger=None,
    )
    assert float(out.metrics.iloc[0]["cash"]) == 500.0


def test_cash_before_first_account_row_starts_at_zero(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([10.0, 10.0, 10.0], index=index, name=ticker)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-03 09:00:00"),
                "currency": "EUR",
                "raw_balance": 500.0,
                "balance_eur": 500.0,
                "raw_change": 500.0,
                "change_eur": 500.0,
                "type": "external_deposit",
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        end_date_override=pd.Timestamp("2026-01-03"),
        cache_dir="cache",
        logger=None,
    )
    assert float(out.metrics.iloc[0]["cash"]) == 0.0
    assert float(out.metrics.iloc[1]["cash"]) == 0.0
    assert float(out.metrics.iloc[2]["cash"]) == 500.0


def test_deposits_start_on_first_account_date_before_first_trade(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([10.0, 10.0, 10.0], index=index, name=ticker)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-03 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 09:00:00"),
                "currency": "EUR",
                "raw_balance": 500.0,
                "balance_eur": 500.0,
                "raw_change": 500.0,
                "change_eur": 500.0,
                "type": "external_deposit",
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        end_date_override=pd.Timestamp("2026-01-03"),
        cache_dir="cache",
        logger=None,
    )

    assert list(out.metrics.index) == list(index)
    assert list(out.metrics["total_deposits"].astype(float)) == [500.0, 500.0, 500.0]


def test_end_date_override_trims_to_latest_real_price_date(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([10.0, 11.0, 12.0], index=index, name=ticker)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 09:00:00"),
                "currency": "EUR",
                "raw_balance": 100.0,
                "balance_eur": 100.0,
                "raw_change": 100.0,
                "change_eur": 100.0,
                "type": "external_deposit",
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        end_date_override=pd.Timestamp("2026-01-10"),
        cache_dir="cache",
        logger=None,
    )

    assert out.metrics.index.max() == pd.Timestamp("2026-01-03")
    assert out.prices_eur.index.max() == pd.Timestamp("2026-01-03")
    assert out.prices_local.index.max() == pd.Timestamp("2026-01-03")
    assert out.latest_price_data_date == pd.Timestamp("2026-01-03")


def test_foreign_currency_local_prices_are_preserved(monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    def fake_fetch_price_series(*, ticker, start, end, cache_dir, logger):
        return pd.Series([50.0, 51.0, 52.0], index=index, name=ticker)

    def fake_fetch_fx_series(*, currency, start, end, cache_dir, logger):
        return pd.Series([0.9, 0.9, 0.9], index=index, name=currency)

    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_price_series",
        fake_fetch_price_series,
    )
    monkeypatch.setattr(
        "src.portfolio_timeseries.fetch_fx_series",
        fake_fetch_fx_series,
    )

    transactions = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "instrument_id": "US0000000001",
                "quantity": 1.0,
                "is_cash_like": False,
            }
        ]
    )
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 09:00:00"),
                "currency": "EUR",
                "raw_balance": 100.0,
                "balance_eur": 100.0,
                "raw_change": 100.0,
                "change_eur": 100.0,
                "type": "external_deposit",
            }
        ]
    )
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST",
                "ticker": "TEST",
                "currency": "USD",
                "is_cash_like": False,
            }
        ]
    )

    out = compute_portfolio_timeseries(
        transactions=transactions,
        account=account,
        instruments=instruments,
        end_date_override=pd.Timestamp("2026-01-03"),
        cache_dir="cache",
        logger=None,
    )

    assert float(out.prices_local.iloc[-1, 0]) == 52.0
    assert np.isclose(float(out.prices_eur.iloc[-1, 0]), 46.8)


def test_cash_cumsum_excludes_internal_sweep_rows() -> None:
    daily_index = pd.date_range("2026-01-01", periods=2, freq="D")
    account = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-01 10:00:00"),
                "currency": "EUR",
                "raw_change": 500.0,
                "change_eur": 500.0,
                "type": "external_deposit",
                "is_internal_transfer": False,
            },
            {
                "datetime": pd.Timestamp("2026-01-01 10:01:00"),
                "currency": "EUR",
                "raw_change": 500.0,
                "change_eur": 500.0,
                "type": "internal_cash_sweep",
                "is_internal_transfer": True,
            },
        ]
    )

    cash, warnings, _issues = build_daily_cash_series_from_changes(
        account_df=account,
        daily_index=daily_index,
        cache_dir="cache",
        logger=None,
    )

    assert float(cash.iloc[-1]) == 500.0
    assert any("excluded" in w.lower() for w in warnings)
