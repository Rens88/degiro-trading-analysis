from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import BASE_BLUE, BASE_GREEN, BASE_ORANGE, BASE_RED, BASE_YELLOW


def build_portfolio_over_time_figure(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["positions_value"],
            name="Positions value",
            line=dict(color=BASE_BLUE, width=2),
            hovertemplate="Positions value: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["cash"],
            name="Cash",
            line=dict(color=BASE_GREEN, width=2),
            hovertemplate="Cash: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["profit"],
            name="Profit",
            line=dict(color=BASE_RED, width=2),
            hovertemplate="Profit: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["total_deposits"],
            name="Total deposits (cumulative)",
            line=dict(color=BASE_ORANGE, width=2, dash="dot"),
            hovertemplate="Total deposits: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["portfolio_value"],
            name="Portfolio value",
            line=dict(color=BASE_YELLOW, width=3),
            hovertemplate="Portfolio value: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["simple_return"] * 100.0,
            name="Simple return (%)",
            yaxis="y2",
            line=dict(color="#222222", width=2),
            hovertemplate="Simple return: %{y:,.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=120),
        title="Portfolio Over Time",
        yaxis=dict(title="EUR"),
        yaxis2=dict(title="Return (%)", overlaying="y", side="right"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.26,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_normalized_median_figure(
    prices_eur: pd.DataFrame,
    *,
    instruments: pd.DataFrame | None = None,
    lookback_months: int,
    median_window_months: int,
) -> go.Figure:
    fig = go.Figure()
    if prices_eur.empty:
        fig.update_layout(template="plotly_white", title="No price series available")
        return fig

    end_date = prices_eur.index.max()
    start_date = end_date - pd.DateOffset(months=lookback_months)
    window_days = max(5, int(round(median_window_months * 30.44)))
    lookback = prices_eur.loc[prices_eur.index >= start_date]

    instrument_meta: dict[str, dict[str, str]] = {}
    if instruments is not None and not instruments.empty and "instrument_id" in instruments.columns:
        tmp = instruments.copy()
        tmp["instrument_id"] = tmp["instrument_id"].astype(str)
        for row in tmp.itertuples(index=False):
            instrument_id = str(getattr(row, "instrument_id", ""))
            if instrument_id == "":
                continue
            ticker = str(getattr(row, "ticker", "")).strip()
            product = str(getattr(row, "product", "")).strip()
            instrument_meta[instrument_id] = {
                "ticker": ticker if ticker and ticker.lower() not in {"nan", "none"} else instrument_id,
                "product": product if product and product.lower() not in {"nan", "none"} else instrument_id,
            }

    for col in lookback.columns:
        series = lookback[col].astype(float)
        rolling_median = series.rolling(window=window_days, min_periods=max(5, window_days // 3)).median()
        normalized = series / rolling_median
        col_key = str(col)
        meta = instrument_meta.get(col_key, {"ticker": col_key, "product": col_key})
        ticker_label = meta["ticker"]
        product_label = meta["product"]
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized,
                mode="lines",
                name=ticker_label,
                customdata=np.full(len(normalized), product_label, dtype=object),
                hovertemplate=(
                    "Ticker: %{fullData.name}<br>"
                    "Product: %{customdata}<br>"
                    "Normalized: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=(
            f"Stock Development Normalized to Rolling Median "
            f"({median_window_months}m window, {lookback_months}m lookback)"
        ),
        yaxis=dict(title="Normalized price"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=10, r=10, t=50, b=110),
    )
    return fig


def format_latest_totals_text(
    *,
    positions_value: float,
    cash: float,
    total_value: float,
    deposits: float,
    profit: float,
    simple_return: float,
) -> str:
    return (
        f"- positions value: EUR {positions_value:,.2f}\n"
        f"- cash: EUR {cash:,.2f}\n"
        f"- total value: EUR {total_value:,.2f}\n"
        f"- total deposits: EUR {deposits:,.2f}\n"
        f"- profit: EUR {profit:,.2f}\n"
        f"- simple return: {simple_return * 100.0:,.2f}%\n"
        f"- identity: EUR {positions_value:,.2f} + EUR {cash:,.2f} = EUR {total_value:,.2f}"
    )


def build_period_decomposition_figure(period_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if period_df is None or period_df.empty:
        fig.update_layout(template="plotly_white", title="No period decomposition available")
        return fig

    x = period_df["period"]
    fig.add_trace(
        go.Bar(
            x=x,
            y=period_df["net_flow_eur"],
            name="Net external flow",
            marker_color=BASE_ORANGE,
            hovertemplate="Period: %{x}<br>Net flow: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=period_df["investment_pnl_eur"],
            name="Investment P/L (ex-flows)",
            marker_color=BASE_BLUE,
            hovertemplate="Period: %{x}<br>Investment P/L: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=period_df["total_change_eur"],
            name="Total value change",
            mode="lines+markers",
            line=dict(color=BASE_GREEN, width=2),
            hovertemplate="Period: %{x}<br>Total change: EUR %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        barmode="group",
        title="Portfolio Value Change Decomposition",
        yaxis=dict(title="EUR"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def build_drawdown_figure(metrics: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if metrics is None or metrics.empty or "portfolio_value" not in metrics.columns:
        fig.update_layout(template="plotly_white", title="No drawdown data available")
        return fig

    equity = pd.to_numeric(metrics["portfolio_value"], errors="coerce").dropna()
    if equity.empty:
        fig.update_layout(template="plotly_white", title="No drawdown data available")
        return fig

    running_peak = equity.cummax()
    drawdown_pct = ((equity / running_peak) - 1.0) * 100.0

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Portfolio value",
            line=dict(color=BASE_BLUE, width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: EUR %{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            name="Drawdown (%)",
            line=dict(color=BASE_RED, width=2),
            fill="tozeroy",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:,.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title="Portfolio Value and Drawdown",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    fig.update_yaxes(title_text="EUR", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
    return fig


def build_cash_allocation_figure(metrics: pd.DataFrame, *, target_cash_pct: float = 10.0) -> go.Figure:
    fig = go.Figure()
    required_cols = {"cash", "portfolio_value"}
    if metrics is None or metrics.empty or not required_cols.issubset(set(metrics.columns)):
        fig.update_layout(template="plotly_white", title="No cash allocation data available")
        return fig

    cash = pd.to_numeric(metrics["cash"], errors="coerce")
    total = pd.to_numeric(metrics["portfolio_value"], errors="coerce")
    cash_pct = np.where(total > 0.0, (cash / total) * 100.0, np.nan)
    cash_pct = pd.Series(cash_pct, index=metrics.index, dtype="float64")
    if cash_pct.dropna().empty:
        fig.update_layout(template="plotly_white", title="No cash allocation data available")
        return fig

    fig.add_trace(
        go.Scatter(
            x=cash_pct.index,
            y=cash_pct.values,
            name="Cash %",
            line=dict(color=BASE_BLUE, width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Cash: %{y:,.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cash_pct.index,
            y=np.full(len(cash_pct), float(target_cash_pct)),
            name="Target cash %",
            line=dict(color=BASE_ORANGE, width=2, dash="dash"),
            hovertemplate="Target cash: %{y:,.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title="Cash Allocation vs Target",
        yaxis=dict(title="Cash share (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig
