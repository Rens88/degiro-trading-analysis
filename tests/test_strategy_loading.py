from __future__ import annotations

import json

from src.app import load_strategy_file


def test_load_strategy_file_reads_target_cash_pct(tmp_path) -> None:
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
                    "mappings_path": "mappings.yml",
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
    assert strategy["target_etf_fraction"] == 0.6
    assert strategy["desired_etf_holdings"] == 5
    assert strategy["desired_non_etf_holdings"] == 10
    assert data_sources["dataset_a_dir"] == "data/a"
    assert data_sources["dataset_b_dir"] == "data/b"
    assert data_sources["mappings_path"] == "mappings.yml"
