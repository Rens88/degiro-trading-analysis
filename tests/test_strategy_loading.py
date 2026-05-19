from __future__ import annotations

import json

from src.app import load_strategy_file


def test_load_strategy_file_migrates_legacy_target_split(tmp_path) -> None:
    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategy": {
                    "target_cash_pct": 1.0,
                    "target_etf_fraction": 0.6,
                    "desired_etf_holdings": 5,
                    "desired_non_etf_holdings": 10,
                },
                "data_sources": {
                    "dataset_a_dir": "data/a",
                    "dataset_b_dir": "data/b",
                    "classification_path": "ticker_classification_complete.csv",
                },
            }
        ),
        encoding="utf-8",
    )

    strategy, data_sources, resolved = load_strategy_file(
        strategy_file_path=str(strategy_path),
        logger=None,
    )

    assert resolved == strategy_path
    assert strategy["target_cash_pct"] == 1.0
    assert round(float(strategy["target_etf_pct"]), 6) == round(99.0 * 0.6, 6)
    assert round(float(strategy["target_non_etf_pct"]), 6) == round(99.0 * 0.4, 6)
    assert strategy["desired_etf_holdings"] == 5
    assert strategy["desired_non_etf_holdings"] == 10
    assert data_sources["dataset_a_dir"] == "data/a"
    assert data_sources["dataset_b_dir"] == "data/b"
    assert data_sources["classification_path"] == "ticker_classification_complete.csv"


def test_load_strategy_file_reads_explicit_three_way_target_split(tmp_path) -> None:
    strategy_path = tmp_path / "strategy_new.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategy": {
                    "target_etf_pct": 42.0,
                    "target_non_etf_pct": 38.0,
                    "target_cash_pct": 20.0,
                }
            }
        ),
        encoding="utf-8",
    )

    strategy, _, _ = load_strategy_file(strategy_file_path=str(strategy_path), logger=None)

    assert strategy["target_etf_pct"] == 42.0
    assert strategy["target_non_etf_pct"] == 38.0
    assert strategy["target_cash_pct"] == 20.0
