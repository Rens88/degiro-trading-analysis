from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class CashReconciliationResult:
    account_label: str
    cash_from_account_eur: float
    cash_from_portfolio_snapshot_eur: float
    cash_delta_eur: float
    diagnosis: str
    detail: pd.DataFrame


@dataclass
class TotalsResult:
    account_label: str
    positions_value_eur: float
    cash_value_eur: float
    cash_source: str
    total_value_eur: float


def reconcile_dataset(
    *,
    account_label: str,
    portfolio: pd.DataFrame,
    account: pd.DataFrame,
    fx_lookup: Callable[[str], float | None] | None = None,
) -> tuple[CashReconciliationResult, TotalsResult]:
    account_cash, account_detail, missing_fx = cash_from_account(
        account_df=account,
        fx_lookup=fx_lookup,
    )
    portfolio_cash = cash_from_portfolio_snapshot(portfolio)
    delta = account_cash - portfolio_cash

    diagnosis = diagnose_cash_delta(
        cash_delta_eur=delta,
        missing_fx=missing_fx,
        portfolio_cash_rows=int(cash_row_mask(portfolio).sum()),
    )
    cash_result = CashReconciliationResult(
        account_label=account_label,
        cash_from_account_eur=account_cash,
        cash_from_portfolio_snapshot_eur=portfolio_cash,
        cash_delta_eur=delta,
        diagnosis=diagnosis,
        detail=account_detail,
    )

    positions_value = float(portfolio.loc[~cash_row_mask(portfolio), "value_eur"].fillna(0.0).sum())
    if np.isfinite(account_cash):
        preferred_cash = account_cash
        source = "account balance"
    else:
        preferred_cash = portfolio_cash
        source = "portfolio cash fallback"

    total = positions_value + preferred_cash
    totals_result = TotalsResult(
        account_label=account_label,
        positions_value_eur=positions_value,
        cash_value_eur=preferred_cash,
        cash_source=source,
        total_value_eur=total,
    )
    return cash_result, totals_result


def cash_row_mask(portfolio: pd.DataFrame) -> pd.Series:
    if "is_cash_like" in portfolio.columns:
        return portfolio["is_cash_like"].fillna(False).astype(bool)
    product = portfolio["product"].fillna("").astype(str).str.upper()
    isin = portfolio["isin"].fillna("").astype(str).str.upper()
    by_product = product.str.contains(
        r"(CASH|CASH SWEEP|CASH FUND|MONEY MARKET|GELDREKENING|BANKACCOUNT|FLATEX)",
        regex=True,
    )
    by_known_cash_isin = isin.isin({"NLFLATEXACNT"})
    missing_isin = isin.eq("")
    currency_only = product.str.fullmatch(r"(EUR|USD|GBP|CHF|JPY|AUD|CAD)(\s+CASH)?", na=False)
    missing_isin_cash_like = missing_isin & (
        product.str.contains(r"(CASH|GELD|MONEY MARKET)", regex=True) | currency_only
    )
    return by_product | by_known_cash_isin | missing_isin_cash_like


def cash_from_portfolio_snapshot(portfolio: pd.DataFrame) -> float:
    mask = cash_row_mask(portfolio)
    return float(portfolio.loc[mask, "value_eur"].fillna(0.0).sum())


def cash_from_account(
    *,
    account_df: pd.DataFrame,
    fx_lookup: Callable[[str], float | None] | None = None,
) -> tuple[float, pd.DataFrame, int]:
    if account_df.empty:
        return float("nan"), pd.DataFrame(columns=["currency", "raw_balance", "balance_eur"]), 0

    df = account_df.copy().sort_values("datetime")
    latest_per_currency = (
        df.groupby("currency", dropna=False, as_index=False)
        .tail(1)
        .loc[:, ["currency", "raw_balance", "balance_eur", "datetime"]]
        .reset_index(drop=True)
    )

    missing_fx = 0
    computed_values: list[float] = []
    for row in latest_per_currency.itertuples(index=False):
        if pd.isna(row.currency):
            computed_values.append(np.nan)
            continue
        currency = str(row.currency).upper()
        if currency == "EUR":
            val = row.raw_balance if pd.notna(row.raw_balance) else row.balance_eur
            computed_values.append(float(val) if pd.notna(val) else np.nan)
            continue
        if pd.notna(row.balance_eur):
            computed_values.append(float(row.balance_eur))
            continue
        if pd.notna(row.raw_balance) and fx_lookup is not None:
            rate = fx_lookup(currency)
            if rate and np.isfinite(rate):
                computed_values.append(float(row.raw_balance) * float(rate))
                continue
        missing_fx += 1
        computed_values.append(np.nan)

    latest_per_currency["balance_eur_computed"] = computed_values
    total = float(np.nansum(latest_per_currency["balance_eur_computed"]))
    if latest_per_currency["balance_eur_computed"].isna().all():
        total = float("nan")
    return total, latest_per_currency, missing_fx


def diagnose_cash_delta(cash_delta_eur: float, missing_fx: int, portfolio_cash_rows: int) -> str:
    abs_delta = abs(cash_delta_eur) if np.isfinite(cash_delta_eur) else np.inf
    if missing_fx > 0:
        return (
            f"FX conversion missing for {missing_fx} latest non-EUR cash balance row(s); "
            "only those rows are excluded from account-cash total."
        )
    if portfolio_cash_rows == 0:
        return "Portfolio snapshot cash rows missing."
    if not np.isfinite(cash_delta_eur):
        return "Unable to compute cash delta from inputs."
    if abs_delta < 1.0:
        return "Cash reconciliation looks good."
    if abs_delta < 50.0:
        return "Small mismatch; likely timing/rounding between exports."
    return "Large mismatch; verify cash row detection and FX conversion."


def combine_totals(
    per_dataset: dict[str, TotalsResult],
) -> TotalsResult:
    positions = float(sum(item.positions_value_eur for item in per_dataset.values()))
    cash = float(sum(item.cash_value_eur for item in per_dataset.values()))
    total = positions + cash
    return TotalsResult(
        account_label="Combined",
        positions_value_eur=positions,
        cash_value_eur=cash,
        cash_source="sum(dataset cash)",
        total_value_eur=total,
    )


def reconciliation_table(
    cash_results: dict[str, CashReconciliationResult],
) -> pd.DataFrame:
    rows = []
    for label, result in cash_results.items():
        rows.append(
            {
                "dataset": label,
                "cash_from_account_eur": result.cash_from_account_eur,
                "cash_from_portfolio_snapshot_eur": result.cash_from_portfolio_snapshot_eur,
                "cash_delta_eur": result.cash_delta_eur,
                "diagnosis": result.diagnosis,
            }
        )
    return pd.DataFrame(rows)
