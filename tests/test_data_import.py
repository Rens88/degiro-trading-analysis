from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_import import (
    infer_account_type,
    is_cash_like,
    load_csv_generic,
    load_dataset,
    load_mappings,
    normalize_account,
    normalize_transactions,
    resolve_instrument_mapping,
    validate_critical_columns,
)
from src.exceptions import UserFacingError


def test_transactions_normalization_comma_separator_english_headers() -> None:
    csv_text = """Date,Time,Product,ISIN,Quantity,Price,Value EUR,Total EUR,Transaction and/or third party fees EUR
18-02-2026,10:00,TEST ETF,IE00TEST00001,2,"10,50","-21,00","-21,50","-0,50"
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_transactions(raw, account_label="A", warnings=warnings)

    assert out.shape[0] == 1
    assert out.loc[0, "product"] == "TEST ETF"
    assert out.loc[0, "isin"] == "IE00TEST00001"
    assert out.loc[0, "quantity"] == 2.0
    assert out.loc[0, "price"] == 10.5
    assert out.loc[0, "total_eur"] == -21.5
    assert out.loc[0, "fees_eur"] == -0.5


def test_transactions_normalization_semicolon_separator_dutch_headers() -> None:
    csv_text = """Datum;Tijd;Product;ISIN;Aantal;Koers;Waarde EUR;Totaal EUR
18-02-2026;10:00;TEST STOCK;US0000000001;3;12,00;-36,00;-36,00
"""
    raw = pd.read_csv(io.StringIO(csv_text), sep=";", dtype=str)
    warnings: list[str] = []
    out = normalize_transactions(raw, account_label="B", warnings=warnings)

    assert out.shape[0] == 1
    assert out.loc[0, "quantity"] == 3.0
    assert out.loc[0, "price"] == 12.0
    assert out.loc[0, "value_eur"] == -36.0
    assert out.loc[0, "total_eur"] == -36.0


def test_account_normalization_uses_fx_for_non_eur_balance() -> None:
    csv_text = """Date,Time,Description,FX,Change,,Balance,
18-02-2026,10:00,Dividend,"1,2000","12,00",USD,"24,00",USD
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_account(raw, account_label="A", warnings=warnings)

    assert out.shape[0] == 1
    assert out.loc[0, "currency"] == "USD"
    assert round(float(out.loc[0, "change_eur"]), 6) == 10.0
    assert round(float(out.loc[0, "balance_eur"]), 6) == 20.0


def test_account_normalization_handles_currency_in_change_balance_columns() -> None:
    csv_text = """Date,Time,Description,FX,Change,,Balance,
18-02-2026,10:00,Dividend,"1,2000",USD,"12,00",USD,"24,00"
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_account(raw, account_label="A", warnings=warnings)

    assert out.shape[0] == 1
    assert out.loc[0, "currency"] == "USD"
    assert round(float(out.loc[0, "change_eur"]), 6) == 10.0
    assert round(float(out.loc[0, "balance_eur"]), 6) == 20.0


def test_account_normalization_infers_fx_from_no_order_valuta_pair_for_dividends() -> None:
    csv_text = """Date,Time,Value date,Product,ISIN,Description,FX,Change,,Balance,,Order Id
19-02-2026,06:32,18-02-2026,,,Valuta Creditering,,EUR,"12,93",EUR,"1022,94",
19-02-2026,06:32,18-02-2026,,,Valuta Debitering,"1,1812",USD,"-15,27",USD,"0,00",
18-02-2026,06:55,17-02-2026,PROCTER & GAMBLE CO,US7427181091,Dividendbelasting,,USD,"-2,70",USD,"15,27",
18-02-2026,06:54,17-02-2026,PROCTER & GAMBLE CO,US7427181091,Dividend,,USD,"17,97",USD,"17,97",
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_account(raw, account_label="A", warnings=warnings)

    dividend = out[out["description"] == "Dividend"].iloc[0]
    tax = out[out["description"] == "Dividendbelasting"].iloc[0]
    assert round(float(dividend["fx_rate"]), 4) == 1.1812
    assert round(float(tax["fx_rate"]), 4) == 1.1812
    assert round(float(dividend["change_eur"]), 2) == 15.21
    assert round(float(tax["change_eur"]), 2) == -2.29


def test_account_normalization_infers_fx_for_no_order_cash_sweep_like_row() -> None:
    csv_text = """Date,Time,Value date,Product,ISIN,Description,FX,Change,,Balance,,Order Id
03-12-2023,07:55,01-12-2023,,,Valuta Creditering,,EUR,"8,40",EUR,"1134,41",
03-12-2023,07:55,01-12-2023,,,Valuta Debitering,"1,0911",USD,"-9,17",USD,"0,00",
04-12-2023,12:02,01-12-2023,FLATEX USD BANKACCOUNT,NLFLATEXACNT,Degiro Cash Sweep Transfer,,USD,"9,17",USD,"9,17",
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_account(raw, account_label="A", warnings=warnings)

    sweep = out[out["description"] == "Degiro Cash Sweep Transfer"].iloc[0]
    assert round(float(sweep["fx_rate"]), 4) == 1.0911
    assert round(float(sweep["change_eur"]), 2) == 8.40


def test_account_normalization_normalizes_fx_scale_tenths_in_order_rows() -> None:
    csv_text = """Date,Time,Description,FX,Change,,Balance,,Order Id
01-01-2026,10:00,Valuta Debitering,"11,738","-117,38",USD,"0,00",USD,order-1
01-01-2026,10:00,Valuta Creditering,,"100,00",EUR,"100,00",EUR,order-1
01-01-2026,10:00,Koop 1 @ 117,38 USD,,"-117,38",USD,"-117,38",USD,order-1
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_account(raw, account_label="A", warnings=warnings)

    trade = out[out["description"].str.contains("Koop", na=False)].iloc[0]
    assert round(float(trade["fx_rate"]), 4) == 1.1738
    assert round(float(trade["change_eur"]), 2) == -100.00
    assert any("normalized FX scale" in w for w in warnings)


def test_account_normalization_normalizes_fx_scale_tenths_for_no_order_dividend_pair() -> None:
    csv_text = """Date,Time,Value date,Product,ISIN,Description,FX,Change,,Balance,,Order Id
19-02-2026,06:32,18-02-2026,,,Valuta Creditering,,EUR,"10,00",EUR,"1000,00",
19-02-2026,06:32,18-02-2026,,,Valuta Debitering,"11,812",USD,"-11,81",USD,"0,00",
18-02-2026,06:55,17-02-2026,TEST CO,US0000000001,Dividendbelasting,,USD,"-2,09",USD,"11,81",
18-02-2026,06:54,17-02-2026,TEST CO,US0000000001,Dividend,,USD,"13,90",USD,"13,90",
"""
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    warnings: list[str] = []
    out = normalize_account(raw, account_label="A", warnings=warnings)

    dividend = out[out["description"] == "Dividend"].iloc[0]
    tax = out[out["description"] == "Dividendbelasting"].iloc[0]
    assert round(float(dividend["fx_rate"]), 4) == 1.1812
    assert round(float(tax["fx_rate"]), 4) == 1.1812
    assert round(float(dividend["change_eur"]) + float(tax["change_eur"]), 2) == 10.00
    assert any("normalized FX scale" in w for w in warnings)


def test_infer_account_type_ideal_deposit_uses_reservation_flow() -> None:
    description = pd.Series(
        [
            "Reservation iDEAL",
            "Reservation iDEAL",
            "iDEAL Deposit",
            "flatex Storting",
        ]
    )
    raw_change = pd.Series([500.0, -500.0, 500.0, 300.0])
    out = infer_account_type(description=description, raw_change=raw_change)

    assert out.iloc[0] == "external_deposit"
    assert out.iloc[1] == "reservation_hold"
    assert out.iloc[2] == "reservation_settlement"
    assert out.iloc[3] == "external_deposit"


def test_critical_nan_fields_emit_warnings() -> None:
    tx = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-01-01"), pd.NaT],
            "product": ["A", "B"],
            "isin": ["US1", "US2"],
            "quantity": [1.0, None],
            "price": [10.0, 12.0],
            "total_eur": [1.0, None],
            "description": ["desc", "desc2"],
            "type": ["buy", "sell"],
        }
    )
    pf = pd.DataFrame(
        {
            "product": ["A", "B"],
            "isin": ["US1", "US2"],
            "quantity": [1.0, 2.0],
            "price": [10.0, 11.0],
            "currency": ["EUR", "USD"],
            "is_cash_like": [False, False],
            "value_eur": [2.0, None],
        }
    )
    acc = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-01-01"), pd.NaT],
            "description": ["desc", "desc2"],
            "type": ["external_deposit", "other"],
            "currency": ["EUR", "USD"],
            "raw_change": [10.0, 5.0],
            "raw_balance": [3.0, 4.0],
            "fx_rate": [1.0, None],
            "change_eur": [10.0, None],
            "balance_eur": [3.0, None],
        }
    )
    warnings, issues = validate_critical_columns(transactions=tx, portfolio=pf, account=acc)
    assert any("total_eur" in w for w in warnings)
    assert any("value_eur" in w for w in warnings)
    assert any("balance_eur" in w for w in warnings)
    assert len(issues) > 0
    assert all("examples" in issue for issue in issues)


def test_missing_ticker_error_contains_product_name() -> None:
    portfolio = pd.DataFrame(
        [
            {
                "instrument_id": "DE0005552004",
                "product": "DEUTSCHE POST AG",
                "isin": "DE0005552004",
                "symbol": None,
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )
    transactions = portfolio.copy()
    mappings = {"symbols": {}, "currencies": {}, "is_etf": {}, "is_not_etf": {}}

    try:
        resolve_instrument_mapping(portfolio=portfolio, transactions=transactions, mappings=mappings)
        raise AssertionError("Expected UserFacingError due to missing ticker mapping.")
    except UserFacingError as exc:
        text = exc.to_ui_text()
        assert "DEUTSCHE POST AG" in text
        assert "DE0005552004" in text
        assert "Yahoo" in text


def test_is_not_etf_explicit_override() -> None:
    portfolio = pd.DataFrame(
        [
            {
                "instrument_id": "TEST1",
                "product": "SOME UCITS STYLE NAME",
                "isin": "NL0012345678",
                "symbol": None,
                "currency": "EUR",
                "is_cash_like": False,
            }
        ]
    )
    transactions = portfolio.copy()
    mappings = {
        "symbols": {"NL0012345678": "TEST.AS"},
        "currencies": {},
        "is_etf": {},
        "is_not_etf": {"NL0012345678": True},
    }
    out = resolve_instrument_mapping(portfolio=portfolio, transactions=transactions, mappings=mappings)
    assert bool(out.loc[0, "is_etf"]) is False
    assert bool(out.loc[0, "is_not_etf"]) is True


def test_missing_is_etf_or_is_not_etf_raises_error() -> None:
    portfolio = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "product": "TEST STOCK",
                "isin": "US0000000001",
                "symbol": None,
                "currency": "USD",
                "is_cash_like": False,
            }
        ]
    )
    transactions = portfolio.copy()
    mappings = {
        "symbols": {"US0000000001": "TEST"},
        "currencies": {"US0000000001": "USD"},
        "is_etf": {},
        "is_not_etf": {},
    }

    try:
        resolve_instrument_mapping(portfolio=portfolio, transactions=transactions, mappings=mappings)
        raise AssertionError("Expected classification error for missing is_etf/is_not_etf entry.")
    except UserFacingError as exc:
        text = exc.to_ui_text()
        assert "ETF classification is incomplete" in text
        assert "US0000000001" in text
        assert "TEST STOCK" in text


def test_load_mappings_invalid_yaml_syntax_raises_user_error(tmp_path) -> None:
    path = tmp_path / "mappings.yml"
    path.write_text("symbols:\n  A: B\nis_etf: [oops\n", encoding="utf-8")
    try:
        load_mappings(path)
        raise AssertionError("Expected syntax validation error.")
    except UserFacingError as exc:
        assert "invalid YAML syntax" in exc.to_ui_text()


def test_is_cash_like_does_not_mark_etf_with_eur_in_name() -> None:
    product = pd.Series(["ISHARES EUROPEAN PROPERTY YIELD UCITS ETF EUR D"])
    isin = pd.Series(["IE00B0M63284"])
    out = is_cash_like(product, isin)
    assert bool(out.iloc[0]) is False


def test_is_cash_like_marks_flatex_bank_account_as_cash() -> None:
    product = pd.Series(["FLATEX EURO BANKACCOUNT"])
    isin = pd.Series(["NLFLATEXACNT"])
    out = is_cash_like(product, isin)
    assert bool(out.iloc[0]) is True


def test_load_csv_generic_repairs_transactions_product_suffix_row() -> None:
    header = ["Date", "Time", "Product", "ISIN"]
    rows = [
        header,
        ["08-01-2025", "14:45", "ISHARES EUROPEAN PROPERTY YIELD UCITS ETF", "IE00B0M63284"],
        ["", "", "EUR D", ""],
    ]
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=",", quotechar='"', lineterminator="\n").writerows(rows)

    out = load_csv_generic(buffer.getvalue().encode("utf-8"))
    assert out.shape[0] == 1
    assert out.loc[0, "Product"] == "ISHARES EUROPEAN PROPERTY YIELD UCITS ETF EUR D"
    assert out.loc[0, "ISIN"] == "IE00B0M63284"


def test_load_csv_generic_repairs_account_continuation_rows() -> None:
    header = ["Date", "Time", "Value date", "Product", "ISIN", "Description", "FX", "Change", "", "Balance", "", "Order Id"]
    rows = [
        header,
        [
            "12-09-2024",
            "11:00",
            "11-09-2024",
            "",
            "",
            "Overboeking naar uw geldrekening bij flatexDEGIRO Bank:",
            "",
            "",
            "",
            "USD",
            "64,67",
            "",
        ],
        ["", "", "", "", "", "9,94 USD", "", "", "", "", "", ""],
        [
            "04-11-2025",
            "16:56",
            "04-11-2025",
            "VANECK WORLD EQUAL WEIGHT SCREENED",
            "NL0010408704",
            "Koop 20 @ 36,5 EUR",
            "",
            "EUR",
            "-730,00",
            "EUR",
            "482,03",
            "3b98a95d-5681-4c8e-",
        ],
        ["", "", "", "", "", "", "", "", "", "", "", "ad0c-8c6ca8768e10"],
    ]
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=",", quotechar='"', lineterminator="\n").writerows(rows)

    out = load_csv_generic(buffer.getvalue().encode("utf-8"))
    assert out.shape[0] == 2
    assert (
        out.loc[0, "Description"]
        == "Overboeking naar uw geldrekening bij flatexDEGIRO Bank:"
    )
    assert out.loc[1, "Order Id"] == "3b98a95d-5681-4c8e-ad0c-8c6ca8768e10"


def test_load_csv_generic_merges_non_amount_description_continuation() -> None:
    header = ["Date", "Time", "Value date", "Product", "ISIN", "Description", "FX", "Change", "", "Balance", "", "Order Id"]
    rows = [
        header,
        ["01-01-2025", "10:00", "01-01-2025", "", "", "Some description", "", "", "", "EUR", "10,00", ""],
        ["", "", "", "", "", "continued text", "", "", "", "", "", ""],
    ]
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=",", quotechar='"', lineterminator="\n").writerows(rows)

    out = load_csv_generic(buffer.getvalue().encode("utf-8"))
    assert out.shape[0] == 1
    assert out.loc[0, "Description"] == "Some description continued text"


@pytest.mark.parametrize(
    ("clean_name", "broken_name"),
    [
        ("pensioenbeleggen", "pensioenbeleggen_broken"),
        ("spaarbeleggen", "spaarbeleggen_broken"),
    ],
)
def test_broken_datasets_import_same_as_clean(clean_name: str, broken_name: str) -> None:
    clean_base = Path("data") / clean_name
    broken_base = Path("data") / broken_name

    clean = load_dataset(
        account_label=clean_name,
        transactions_source=clean_base / "Transactions.csv",
        portfolio_source=clean_base / "Portfolio.csv",
        account_source=clean_base / "Account.csv",
        mappings_path="mappings.yml",
    )
    broken = load_dataset(
        account_label=broken_name,
        transactions_source=broken_base / "Transactions.csv",
        portfolio_source=broken_base / "Portfolio.csv",
        account_source=broken_base / "Account.csv",
        mappings_path="mappings.yml",
    )

    assert_frame_equal(
        clean.transactions.reset_index(drop=True),
        broken.transactions.reset_index(drop=True),
        check_dtype=False,
    )
    assert_frame_equal(
        clean.portfolio.reset_index(drop=True),
        broken.portfolio.reset_index(drop=True),
        check_dtype=False,
    )
    assert_frame_equal(
        clean.account.reset_index(drop=True),
        broken.account.reset_index(drop=True),
        check_dtype=False,
    )
    assert_frame_equal(
        clean.instruments.reset_index(drop=True),
        broken.instruments.reset_index(drop=True),
        check_dtype=False,
    )

    assert broken.warnings == clean.warnings
    assert [(i["label"], i["count"]) for i in broken.issues] == [
        (i["label"], i["count"]) for i in clean.issues
    ]
