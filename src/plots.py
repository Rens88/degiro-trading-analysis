"""
AGENT_NOTE: Plotly-only rendering module.

Interdependencies:
- Input frames are produced by `src/app.py` processing pipeline and
  `src/insights.py`.
- Color constants come from `src/config.py`.
- Function outputs are rendered in Streamlit sections via `app.py`.

When editing:
- Avoid data mutation; keep this module presentation-focused.
- If required input columns change, update producer code and tests together.
- See `src/INTERDEPENDENCIES.md` for contract details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import BASE_BLUE, BASE_GREEN, BASE_ORANGE, BASE_RED, BASE_YELLOW


def build_performance_over_time_figure(metrics: pd.DataFrame) -> go.Figure:
    # AGENT_NOTE: Expects metrics columns:
    # `total_deposits`, `portfolio_value`, `simple_return`.
    fig = go.Figure()
    if metrics is None or metrics.empty:
        fig.update_layout(template="plotly_white", title="No performance data available")
        return fig

    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["total_deposits"],
            name="Total deposits (cumulative)",
            line=dict(color=BASE_ORANGE, width=2, dash="dot"),
            hovertemplate="Deposited: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["portfolio_value"],
            name="Portfolio value",
            line=dict(color=BASE_BLUE, width=3),
            hovertemplate="Portfolio value: EUR %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.index,
            y=metrics["simple_return"] * 100.0,
            name="Simple return (%)",
            yaxis="y2",
            line=dict(color=BASE_RED, width=2),
            hovertemplate="Simple return: %{y:,.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title="Performance Over Time",
        margin=dict(l=10, r=10, t=45, b=120),
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


def build_holdings_over_time_figure(
    *,
    positions: pd.DataFrame,
    prices_eur: pd.DataFrame,
    instruments: pd.DataFrame,
    cash_series: pd.Series | None = None,
    cost_basis_eur: pd.DataFrame | None = None,
) -> go.Figure:
    # AGENT_NOTE: Expects positions and prices with matching instrument_id columns.
    # Optional `cost_basis_eur` is used only for hover diagnostics.
    fig = go.Figure()
    if positions is None or prices_eur is None or positions.empty or prices_eur.empty:
        fig.update_layout(template="plotly_white", title="No holdings time-series available")
        return fig

    shared_cols = [c for c in positions.columns if c in prices_eur.columns]
    if not shared_cols:
        fig.update_layout(template="plotly_white", title="No holdings time-series available")
        return fig

    pos = positions[shared_cols].copy()
    px = prices_eur[shared_cols].reindex(pos.index).copy()
    values = pos * px
    if cost_basis_eur is None or cost_basis_eur.empty:
        cost_basis = pd.DataFrame(np.nan, index=pos.index, columns=shared_cols)
    else:
        cost_basis = cost_basis_eur.reindex(index=pos.index, columns=shared_cols)

    meta_map: dict[str, dict[str, str]] = {}
    if isinstance(instruments, pd.DataFrame) and not instruments.empty and "instrument_id" in instruments.columns:
        tmp = instruments.copy()
        tmp["instrument_id"] = tmp["instrument_id"].astype(str)
        for row in tmp.itertuples(index=False):
            instrument_id = str(getattr(row, "instrument_id", ""))
            if instrument_id == "":
                continue
            product = str(getattr(row, "product", "")).strip()
            ticker = str(getattr(row, "ticker", "")).strip()
            if not product:
                product = instrument_id
            if not ticker:
                ticker = instrument_id
            meta_map[instrument_id] = {"product": product, "ticker": ticker}

    # Keep all holdings including those that closed during the period.
    cols_sorted = sorted(shared_cols, key=lambda c: float(pd.to_numeric(values[c], errors="coerce").max()), reverse=True)
    for col in cols_sorted:
        col_key = str(col)
        meta = meta_map.get(col_key, {"product": col_key, "ticker": col_key})
        name = f"{meta['product']} ({meta['ticker']})"
        y = pd.to_numeric(values[col], errors="coerce")
        cb = pd.to_numeric(cost_basis[col], errors="coerce")
        custom = np.column_stack(
            [
                cb.to_numpy(dtype="float64"),
                np.array([meta["product"]] * len(y), dtype=object),
                np.array([meta["ticker"]] * len(y), dtype=object),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y,
                name=name,
                mode="lines",
                customdata=custom,
                hovertemplate=(
                    "Product: %{customdata[1]}<br>"
                    "Ticker: %{customdata[2]}<br>"
                    "Value: EUR %{y:,.2f}<br>"
                    "Net spent for current shares: EUR %{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

    if cash_series is not None and len(cash_series) > 0:
        cash = pd.to_numeric(cash_series, errors="coerce").reindex(pos.index)
        fig.add_trace(
            go.Scatter(
                x=cash.index,
                y=cash.values,
                name="Cash",
                line=dict(color=BASE_GREEN, width=2, dash="dash"),
                hovertemplate="Cash: EUR %{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title="Holdings and Cash Over Time",
        height=900,
        yaxis=dict(title="EUR"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=120),
    )
    return fig


def build_benchmark_comparison_figure(levels_df: pd.DataFrame) -> go.Figure:
    # AGENT_NOTE: Expects benchmark levels normalized to a common base (e.g. 100).
    fig = go.Figure()
    if levels_df is None or levels_df.empty:
        fig.update_layout(template="plotly_white", title="No benchmark data available")
        return fig
    for col in levels_df.columns:
        series = pd.to_numeric(levels_df[col], errors="coerce")
        if series.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=levels_df.index,
                y=series,
                name=str(col),
                mode="lines",
                hovertemplate=f"{col}: %{{y:,.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title="Benchmark Comparison (EUR normalized to 100)",
        yaxis=dict(title="Index (start=100)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def build_degiro_costs_quarterly_figure(costs_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    required_cols = {"quarter", "dataset", "total_costs_eur"}
    if costs_df is None or costs_df.empty or not required_cols.issubset(set(costs_df.columns)):
        fig.update_layout(template="plotly_white", title="No quarterly DEGIRO costs available")
        return fig

    df = costs_df.copy()
    df["dataset"] = df["dataset"].fillna("Unknown").astype(str)
    df["quarter"] = df["quarter"].fillna("").astype(str)
    df["total_costs_eur"] = pd.to_numeric(df["total_costs_eur"], errors="coerce").fillna(0.0)
    if "trade_count" in df.columns:
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0.0)
    if "market_count" in df.columns:
        df["market_count"] = pd.to_numeric(df["market_count"], errors="coerce").fillna(0.0)
    if df.empty:
        fig.update_layout(template="plotly_white", title="No quarterly DEGIRO costs available")
        return fig

    try:
        quarter_ts = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="start")
        df["quarter_sort"] = quarter_ts
        quarter_order = (
            df.sort_values("quarter_sort")
            .loc[:, ["quarter", "quarter_sort"]]
            .drop_duplicates(subset=["quarter"], keep="first")
            .sort_values("quarter_sort")["quarter"]
            .tolist()
        )
    except Exception:
        df["quarter_sort"] = df["quarter"]
        quarter_order = sorted(df["quarter"].dropna().unique().tolist())

    agg_map: dict[str, str] = {"total_costs_eur": "sum"}
    if "trade_count" in df.columns:
        agg_map["trade_count"] = "sum"
    if "market_count" in df.columns:
        agg_map["market_count"] = "sum"
    grouped = df.groupby(["quarter", "dataset"], dropna=False, as_index=False).agg(agg_map).reset_index(drop=True)
    grouped["quarter"] = pd.Categorical(grouped["quarter"], categories=quarter_order, ordered=True)
    grouped = grouped.sort_values(["quarter", "dataset"]).reset_index(drop=True)

    palette = [BASE_BLUE, BASE_ORANGE, BASE_GREEN, BASE_RED, BASE_YELLOW]
    datasets = grouped["dataset"].dropna().astype(str).unique().tolist()
    for idx, dataset in enumerate(datasets):
        sub = grouped[grouped["dataset"] == dataset]
        fig.add_trace(
            go.Bar(
                x=sub["quarter"].astype(str),
                y=sub["total_costs_eur"],
                name=dataset,
                marker_color=palette[idx % len(palette)],
                hovertemplate=(
                    "Quarter: %{x}<br>"
                    f"Dataset: {dataset}<br>"
                    "Costs: EUR %{y:,.2f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    line_quarterly = (
        grouped.groupby("quarter", dropna=False, as_index=False)
        .agg(
            trade_count=("trade_count", "sum") if "trade_count" in grouped.columns else ("total_costs_eur", "size"),
            market_count=("market_count", "sum")
            if "market_count" in grouped.columns
            else ("total_costs_eur", "size"),
        )
        .sort_values("quarter")
    )
    line_quarterly["quarter"] = line_quarterly["quarter"].astype(str)
    has_trade_line = "trade_count" in grouped.columns
    has_market_line = "market_count" in grouped.columns
    if has_trade_line:
        fig.add_trace(
            go.Scatter(
                x=line_quarterly["quarter"],
                y=line_quarterly["trade_count"],
                name="Trades (#)",
                mode="lines+markers",
                line=dict(color=BASE_RED, width=2),
                marker=dict(size=7),
                hovertemplate="Quarter: %{x}<br>Trades: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
    if has_market_line:
        fig.add_trace(
            go.Scatter(
                x=line_quarterly["quarter"],
                y=line_quarterly["market_count"],
                name="Markets (#)",
                mode="lines+markers",
                line=dict(color=BASE_GREEN, width=2, dash="dot"),
                marker=dict(size=7),
                hovertemplate="Quarter: %{x}<br>Markets: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        hovermode="x unified",
        title="DEGIRO Quarterly Costs and Activity (stacked)",
        xaxis_title="Quarter",
        yaxis=dict(title="EUR"),
        yaxis2=dict(title="Count", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=10, r=10, t=50, b=60),
    )
    return fig


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


def build_allocation_pie_figure(
    *,
    allocation_df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    if (
        allocation_df is None
        or allocation_df.empty
        or category_col not in allocation_df.columns
        or value_col not in allocation_df.columns
    ):
        fig.update_layout(template="plotly_white", title=f"No {title.lower()} data available")
        return fig

    data = allocation_df[[category_col, value_col]].copy()
    data[category_col] = data[category_col].fillna("").astype(str).str.strip().replace("", "Unclassified")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0.0)
    data = data[data[value_col] > 0.0].copy()
    if data.empty:
        fig.update_layout(template="plotly_white", title=f"No {title.lower()} data available")
        return fig

    colors = [BASE_BLUE, BASE_ORANGE, BASE_GREEN, BASE_RED, BASE_YELLOW]
    fig.add_trace(
        go.Pie(
            labels=data[category_col],
            values=data[value_col],
            hole=0.35,
            sort=False,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="%{label}<br>EUR %{value:,.2f}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    )
    return fig


def build_holdings_segment_pie_figure(
    *,
    holdings_df: pd.DataFrame,
    title: str,
    total_portfolio_value_eur: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if holdings_df is None or holdings_df.empty or "value_eur" not in holdings_df.columns:
        fig.update_layout(template="plotly_white", title=f"No {title.lower()} data available")
        return fig

    data = holdings_df.copy()
    data["value_eur"] = pd.to_numeric(data["value_eur"], errors="coerce")
    data = data[data["value_eur"] > 0.0].copy()
    if data.empty:
        fig.update_layout(template="plotly_white", title=f"No {title.lower()} data available")
        return fig

    portfolio_total = pd.to_numeric(total_portfolio_value_eur, errors="coerce")
    if not np.isfinite(portfolio_total) or float(portfolio_total) <= 0.0:
        portfolio_total = float(data["value_eur"].sum())
    else:
        portfolio_total = float(portfolio_total)
    sleeve_total = float(data["value_eur"].sum())

    ticker = (
        data["ticker"].fillna("").astype(str).str.strip()
        if "ticker" in data.columns
        else pd.Series("", index=data.index)
    )
    product = (
        data["product"].fillna("").astype(str).str.strip()
        if "product" in data.columns
        else pd.Series("", index=data.index)
    )
    isin = (
        data["isin"].fillna("").astype(str).str.strip()
        if "isin" in data.columns
        else pd.Series("", index=data.index)
    )
    labels = ticker.where(ticker != "", product)
    labels = labels.where(labels != "", isin)
    labels = labels.where(labels != "", "Unlabeled holding")

    if "is_over_target_threshold" in data.columns:
        over_target_mask = data["is_over_target_threshold"].fillna(False).astype(bool)
    elif "over_target_eur" in data.columns:
        over_target_mask = pd.to_numeric(data["over_target_eur"], errors="coerce").fillna(0.0) > 0.0
    else:
        over_target_mask = pd.Series(False, index=data.index, dtype=bool)

    over_target_eur = (
        pd.to_numeric(data["over_target_eur"], errors="coerce").fillna(0.0)
        if "over_target_eur" in data.columns
        else pd.Series(0.0, index=data.index, dtype="float64")
    )
    status_text = np.where(over_target_mask.to_numpy(), "Above target threshold", "At/below target threshold")
    portfolio_pct = np.where(
        portfolio_total > 0.0,
        data["value_eur"].to_numpy(dtype="float64") / portfolio_total * 100.0,
        np.nan,
    )
    sleeve_pct = np.where(
        sleeve_total > 0.0,
        data["value_eur"].to_numpy(dtype="float64") / sleeve_total * 100.0,
        np.nan,
    )
    label_pct_text = [f"{v:.2f}% portfolio" if np.isfinite(v) else "N/A" for v in portfolio_pct]
    hover_text = [
        (
            f"{label}<br>"
            f"EUR {value:,.2f}<br>"
            f"Portfolio weight: {pf_pct:.2f}%<br>"
            f"ETF/non-ETF sleeve weight: {seg_pct:.2f}%<br>"
            f"Status: {status}<br>"
            f"Over target: EUR {over:,.2f}"
        )
        for label, value, pf_pct, seg_pct, status, over in zip(
            labels.tolist(),
            data["value_eur"].to_numpy(dtype="float64"),
            portfolio_pct,
            sleeve_pct,
            status_text.tolist(),
            over_target_eur.to_numpy(dtype="float64"),
        )
    ]

    normal_palette = [BASE_BLUE, BASE_ORANGE, BASE_GREEN, BASE_YELLOW]
    colors: list[str] = []
    normal_idx = 0
    for is_over in over_target_mask.tolist():
        if is_over:
            colors.append(BASE_RED)
        else:
            colors.append(normal_palette[normal_idx % len(normal_palette)])
            normal_idx += 1
    pull = [0.10 if is_over else 0.0 for is_over in over_target_mask.tolist()]

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=data["value_eur"],
            hole=0.35,
            sort=False,
            pull=pull,
            marker=dict(colors=colors, line=dict(color="white", width=1)),
            text=label_pct_text,
            textinfo="label+text",
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    )
    return fig
