from __future__ import annotations

import pandas as pd

from src.ticker_characteristics import resolve_ticker_characteristics


def _classification_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    defaults = {
        "instrument_id": "",
        "ticker": "",
        "product": "",
        "currency": "",
        "asset_class": "",
        "primary_style": "",
        "secondary_factor": "",
        "gics_sector": "",
        "gics_industry_group": "",
        "gics_industry": "",
        "gics_sub_industry": "",
    }
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        item = defaults.copy()
        item.update(row)
        normalized_rows.append(item)
    return pd.DataFrame(normalized_rows, columns=list(defaults.keys()))


def test_resolver_prefers_instrument_id_before_ticker(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classification_complete.csv"
    _classification_frame(
        [
            {
                "instrument_id": "ID_MATCH",
                "ticker": "AAA",
                "asset_class": "Equity",
                "primary_style": "Growth",
                "secondary_factor": "Momentum",
                "gics_industry": "Industry by instrument",
            },
            {
                "instrument_id": "OTHER_ID",
                "ticker": "SHARED",
                "asset_class": "ETF",
                "primary_style": "Blend",
                "secondary_factor": "Quality",
                "gics_industry": "Industry by ticker",
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
    assert str(row["primary_style"]) == "Growth"
    assert str(row["gics_industry"]) == "Industry by instrument"
    assert str(row["style"]) == "Growth"
    assert str(row["industry"]) == "Industry by instrument"
    assert str(row["matched_by"]) == "instrument_id"
    assert int(stats["matched_instrument_id_count"]) == 1


def test_resolver_falls_back_to_ticker_when_instrument_id_missing(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classification_complete.csv"
    _classification_frame(
        [
            {
                "instrument_id": "SOME_OTHER_ID",
                "ticker": "TKR_FALLBACK",
                "asset_class": "Equity",
                "primary_style": "Value",
                "secondary_factor": "Defensive",
                "gics_industry": "Ticker industry",
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
    assert str(row["primary_style"]) == "Value"
    assert str(row["gics_industry"]) == "Ticker industry"
    assert str(row["matched_by"]) == "ticker"
    assert int(stats["matched_ticker_count"]) == 1


def test_resolver_unmatched_rows_use_inference_and_do_not_append(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classification_complete.csv"
    _classification_frame([]).to_csv(csv_path, index=False)

    instruments = pd.DataFrame(
        [{"instrument_id": "NEW_ID", "ticker": "NEWTKR", "product": "MSCI World ETF", "is_etf": True}]
    )
    resolved, stats = resolve_ticker_characteristics(
        instruments_df=instruments,
        ticker_classifications_path=csv_path,
        auto_append_missing=True,
    )

    row = resolved.iloc[0]
    assert str(row["asset_class"]) == "ETF"
    assert str(row["primary_style"]) == "Blend"
    assert str(row["gics_industry"]) == "Multi-Sector"
    assert str(row["matched_by"]) == "none"
    assert int(stats["appended_count"]) == 0
    saved = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert saved.shape[0] == 0
