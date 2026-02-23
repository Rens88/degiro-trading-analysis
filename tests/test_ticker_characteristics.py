from __future__ import annotations

import pandas as pd

from src.ticker_characteristics import PLACEHOLDER_VALUE, resolve_ticker_characteristics


def _classification_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    defaults = {
        "instrument_id": "",
        "ticker": "",
        "product": "",
        "auto_style": "",
        "style": "",
        "auto_industry": "",
        "industry": "",
        "classification_description": "",
        "classification_source": "",
    }
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        item = defaults.copy()
        item.update(row)
        normalized_rows.append(item)
    return pd.DataFrame(normalized_rows, columns=list(defaults.keys()))


def test_resolver_prefers_instrument_id_before_ticker(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classifications.csv"
    _classification_frame(
        [
            {
                "instrument_id": "ID_MATCH",
                "ticker": "AAA",
                "style": "Style from instrument id",
                "industry": "Industry from instrument id",
            },
            {
                "instrument_id": "OTHER_ID",
                "ticker": "SHARED",
                "style": "Style from ticker",
                "industry": "Industry from ticker",
            },
        ]
    ).to_csv(csv_path, index=False)

    instruments = pd.DataFrame(
        [{"instrument_id": "id_match", "ticker": "shared", "product": "Any", "is_etf": False}]
    )
    resolved, stats = resolve_ticker_characteristics(
        instruments_df=instruments,
        ticker_classifications_path=csv_path,
        auto_append_missing=False,
    )

    row = resolved.iloc[0]
    assert str(row["style"]) == "Style from instrument id"
    assert str(row["industry"]) == "Industry from instrument id"
    assert str(row["matched_by"]) == "instrument_id"
    assert int(stats["matched_instrument_id_count"]) == 1


def test_resolver_falls_back_to_ticker_when_instrument_id_missing(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classifications.csv"
    _classification_frame(
        [
            {
                "instrument_id": "SOME_OTHER_ID",
                "ticker": "TKR_FALLBACK",
                "style": "Ticker style",
                "industry": "Ticker industry",
            }
        ]
    ).to_csv(csv_path, index=False)

    instruments = pd.DataFrame(
        [{"instrument_id": "UNKNOWN_ID", "ticker": "tkr_fallback", "product": "Any", "is_etf": False}]
    )
    resolved, stats = resolve_ticker_characteristics(
        instruments_df=instruments,
        ticker_classifications_path=csv_path,
        auto_append_missing=False,
    )

    row = resolved.iloc[0]
    assert str(row["style"]) == "Ticker style"
    assert str(row["industry"]) == "Ticker industry"
    assert str(row["matched_by"]) == "ticker"
    assert int(stats["matched_ticker_count"]) == 1


def test_resolver_auto_appends_unknown_rows_with_placeholders(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classifications.csv"
    _classification_frame([]).to_csv(csv_path, index=False)

    instruments = pd.DataFrame(
        [{"instrument_id": "NEW_ID", "ticker": "NEWTKR", "product": "MSCI World ETF", "is_etf": True}]
    )
    resolved, stats = resolve_ticker_characteristics(
        instruments_df=instruments,
        ticker_classifications_path=csv_path,
        auto_append_missing=True,
    )

    assert int(stats["appended_count"]) == 1
    row = resolved.iloc[0]
    assert str(row["style"]) == PLACEHOLDER_VALUE
    assert str(row["industry"]) == PLACEHOLDER_VALUE
    assert str(row["auto_style"]) == "Blend"
    assert str(row["auto_industry"]) == "Broad-market ETF"

    saved = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert saved.shape[0] == 1
    saved_row = saved.iloc[0]
    assert str(saved_row["instrument_id"]) == "NEW_ID"
    assert str(saved_row["ticker"]) == "NEWTKR"
    assert str(saved_row["product"]) == "MSCI World ETF"
    assert str(saved_row["style"]) == PLACEHOLDER_VALUE
    assert str(saved_row["industry"]) == PLACEHOLDER_VALUE
    assert str(saved_row["classification_description"]) == PLACEHOLDER_VALUE
    assert str(saved_row["classification_source"]) == PLACEHOLDER_VALUE


def test_resolver_auto_append_is_idempotent(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classifications.csv"
    _classification_frame([]).to_csv(csv_path, index=False)
    instruments = pd.DataFrame(
        [{"instrument_id": "NEW_ID", "ticker": "NEWTKR", "product": "Any", "is_etf": False}]
    )

    _resolved_first, stats_first = resolve_ticker_characteristics(
        instruments_df=instruments,
        ticker_classifications_path=csv_path,
        auto_append_missing=True,
    )
    _resolved_second, stats_second = resolve_ticker_characteristics(
        instruments_df=instruments,
        ticker_classifications_path=csv_path,
        auto_append_missing=True,
    )

    assert int(stats_first["appended_count"]) == 1
    assert int(stats_second["appended_count"]) == 0
    saved = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert saved.shape[0] == 1
