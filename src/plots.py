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

ETF_NORMALIZED_COLORS = (
    BASE_BLUE,
    "#1F4AA8",
    "#3A63BE",
    "#5A52C6",
    "#7459C9",
    "#9271D6",
)

NON_ETF_NORMALIZED_COLORS = (
    BASE_ORANGE,
    "#F28C28",
    "#E85D04",
    "#D95F0E",
    "#D1495B",
    BASE_RED,
)

DISTINCT_NORMALIZED_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _normalized_group_color(*, is_etf: bool, group_index: int) -> str:
    palette = ETF_NORMALIZED_COLORS if is_etf else NON_ETF_NORMALIZED_COLORS
    return str(palette[group_index % len(palette)])


def _distinct_normalized_color(group_index: int) -> str:
    return str(DISTINCT_NORMALIZED_COLORS[group_index % len(DISTINCT_NORMALIZED_COLORS)])


def _apply_lowpass_butterworth(
    series: pd.Series,
    *,
    normalized_cutoff_ratio: float = 0.18,
    order: int = 2,
) -> pd.Series:
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:
        return series

    if not isinstance(series, pd.Series) or series.empty:
        return series
    if normalized_cutoff_ratio <= 0.0 or normalized_cutoff_ratio >= 1.0:
        return series

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    out = values.copy()
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < max(20, 12 * order):
        return pd.Series(out, index=series.index, dtype="float64")

    if isinstance(series.index, pd.DatetimeIndex):
        time_seconds = (series.index.view("int64") / 1_000_000_000.0).astype("float64")
    else:
        time_seconds = np.arange(len(series), dtype="float64")

    start = None
    segment_count = len(values)
    for idx in range(segment_count + 1):
        is_finite = idx < segment_count and finite_mask[idx]
        if is_finite and start is None:
            start = idx
            continue
        if is_finite or start is None:
            continue

        stop = idx
        segment_slice = slice(start, stop)
        segment_values = values[segment_slice]
        segment_times = time_seconds[segment_slice]

        if len(segment_values) >= max(20, 12 * order):
            dt = np.diff(segment_times)
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            if dt.size > 0:
                dt_median = float(np.median(dt))
                fs = 1.0 / dt_median if dt_median > 0.0 else np.nan
                if np.isfinite(fs) and fs > 0.0:
                    nyquist = 0.5 * fs
                    cutoff_hz = float(normalized_cutoff_ratio) * nyquist
                    if 0.0 < cutoff_hz < nyquist:
                        try:
                            tu = np.arange(segment_times[0], segment_times[-1] + dt_median * 0.5, dt_median)
                            yu = np.interp(tu, segment_times, segment_values)
                            sos = butter(order, cutoff_hz, fs=fs, output="sos")
                            yu_filtered = sosfiltfilt(sos, yu)
                            out[segment_slice] = np.interp(segment_times, tu, yu_filtered)
                        except Exception:
                            pass
        start = None

    return pd.Series(out, index=series.index, dtype="float64")


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
        # hovermode="x unified",
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
        grouped.groupby("quarter", dropna=False, as_index=False, observed=False)
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
    prices_local: pd.DataFrame | None = None,
    instruments: pd.DataFrame | None = None,
    holdings_catalog: pd.DataFrame | None = None,
    lookback_months: int,
    median_window_months: int,
    apply_lowpass_filter: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if prices_eur.empty:
        fig.update_layout(template="plotly_white", title="No price series available")
        return fig

    end_date = prices_eur.index.max()
    window_days = max(5, int(round(median_window_months * 30.44)))
    lookback_value = int(lookback_months)
    if lookback_value <= 0:
        lookback = prices_eur
        lookback_label = "all-time"
    else:
        start_date = end_date - pd.DateOffset(months=lookback_value)
        lookback = prices_eur.loc[prices_eur.index >= start_date]
        lookback_label = f"{lookback_value}m"

    instrument_meta: dict[str, dict[str, object]] = {}
    if holdings_catalog is not None and not holdings_catalog.empty and "instrument_id" in holdings_catalog.columns:
        tmp = holdings_catalog.copy()
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
                "currency": str(getattr(row, "currency", "EUR")).strip().upper() or "EUR",
                "is_etf": bool(getattr(row, "is_etf", False)),
                "over_target_eur": float(pd.to_numeric(getattr(row, "over_target_eur", np.nan), errors="coerce")),
                "is_over_target_threshold": bool(getattr(row, "is_over_target_threshold", False)),
                "target_status": str(getattr(row, "target_status", "")).strip(),
            }
    if instruments is not None and not instruments.empty and "instrument_id" in instruments.columns:
        tmp = instruments.copy()
        tmp["instrument_id"] = tmp["instrument_id"].astype(str)
        for row in tmp.itertuples(index=False):
            instrument_id = str(getattr(row, "instrument_id", ""))
            if instrument_id == "":
                continue
            ticker = str(getattr(row, "ticker", "")).strip()
            product = str(getattr(row, "product", "")).strip()
            merged = instrument_meta.get(instrument_id, {}).copy()
            merged.setdefault(
                "ticker",
                ticker if ticker and ticker.lower() not in {"nan", "none"} else instrument_id,
            )
            merged.setdefault(
                "product",
                product if product and product.lower() not in {"nan", "none"} else instrument_id,
            )
            merged.setdefault("currency", str(getattr(row, "currency", "EUR")).strip().upper() or "EUR")
            merged.setdefault("is_etf", bool(getattr(row, "is_etf", False)))
            merged.setdefault("over_target_eur", float("nan"))
            merged.setdefault("is_over_target_threshold", False)
            merged.setdefault("target_status", "")
            instrument_meta[instrument_id] = merged

    trace_specs: list[dict[str, object]] = []
    for col in lookback.columns:
        series = lookback[col].astype(float)
        rolling_median = series.rolling(window=window_days, min_periods=max(5, window_days // 3)).median()
        normalized = series / rolling_median
        full_series = pd.to_numeric(prices_eur[col], errors="coerce")
        full_rolling_median = full_series.rolling(
            window=window_days,
            min_periods=max(5, window_days // 3),
        ).median()
        full_normalized = full_series / full_rolling_median
        if apply_lowpass_filter:
            normalized = _apply_lowpass_butterworth(normalized)
            full_normalized = _apply_lowpass_butterworth(full_normalized)
        col_key = str(col)
        meta = instrument_meta.get(
            col_key,
            {
                "ticker": col_key,
                "product": col_key,
                "currency": "EUR",
                "is_etf": False,
                "over_target_eur": float("nan"),
                "is_over_target_threshold": False,
                "target_status": "",
            },
        )
        ticker_label = str(meta["ticker"])
        product_label = str(meta["product"])
        currency_code = str(meta.get("currency", "EUR")).strip().upper() or "EUR"
        is_etf = bool(meta.get("is_etf", False))
        over_target_eur = float(pd.to_numeric(meta.get("over_target_eur", np.nan), errors="coerce"))
        is_over_target_threshold = bool(meta.get("is_over_target_threshold", False))
        target_status = str(meta.get("target_status", "")).strip()
        if target_status == "":
            if np.isfinite(over_target_eur) and over_target_eur > 0.0:
                target_status = "Over target"
            elif np.isfinite(over_target_eur) and over_target_eur < 0.0:
                target_status = "Under target"
            else:
                target_status = "At target"
        latest_price_series = full_series.dropna()
        latest_normalized_series = full_normalized.dropna()
        latest_price_eur = float(latest_price_series.iloc[-1]) if not latest_price_series.empty else np.nan
        latest_price_date = (
            pd.Timestamp(latest_price_series.index[-1]).strftime("%b %d, %Y")
            if not latest_price_series.empty
            else "N/A"
        )
        latest_normalized = (
            float(latest_normalized_series.iloc[-1]) if not latest_normalized_series.empty else np.nan
        )
        local_series = None
        latest_local_price = np.nan
        if isinstance(prices_local, pd.DataFrame) and col in prices_local.columns:
            local_series = pd.to_numeric(prices_local[col], errors="coerce").reindex(normalized.index)
            latest_local_series = pd.to_numeric(prices_local[col], errors="coerce").dropna()
            if not latest_local_series.empty:
                latest_local_price = float(latest_local_series.iloc[-1])
        customdata = np.column_stack(
            [
                np.full(len(normalized), product_label, dtype=object),
                series.to_numpy(dtype=float, copy=False),
                np.full(len(normalized), latest_price_eur, dtype=float),
                np.full(len(normalized), latest_price_date, dtype=object),
                np.full(len(normalized), latest_normalized, dtype=float),
                np.full(len(normalized), "ETF" if is_etf else "Non-ETF", dtype=object),
                np.full(len(normalized), target_status, dtype=object),
                np.full(
                    len(normalized),
                    "Above threshold" if is_over_target_threshold else "At/below threshold",
                    dtype=object,
                ),
                (
                    local_series.to_numpy(dtype=float, copy=False)
                    if isinstance(local_series, pd.Series)
                    else np.full(len(normalized), np.nan, dtype=float)
                ),
                np.full(len(normalized), latest_local_price, dtype=float),
                np.full(len(normalized), currency_code, dtype=object),
            ]
        )
        line_width = (
            3.0
            if is_over_target_threshold
            else (2.3 if np.isfinite(over_target_eur) and over_target_eur > 0.0 else 1.6)
        )
        line_opacity = (
            1.0
            if is_over_target_threshold
            else (0.95 if np.isfinite(over_target_eur) and over_target_eur > 0.0 else 0.78)
        )
        trace_specs.append(
            {
                "instrument_id": col_key,
                "ticker": ticker_label,
                "currency_code": currency_code,
                "is_etf": is_etf,
                "legend_rank": 0 if is_etf else 1,
                "x": normalized.index,
                "y": normalized,
                "customdata": customdata,
                "line_width": line_width,
                "line_opacity": line_opacity,
            }
        )

    trace_specs.sort(key=lambda item: (int(item["legend_rank"]), str(item["ticker"]).upper()))
    unique_types = {bool(spec["is_etf"]) for spec in trace_specs}
    etf_color_idx = 0
    non_etf_color_idx = 0
    distinct_color_idx = 0
    for spec in trace_specs:
        is_etf = bool(spec["is_etf"])
        currency_code = str(spec.get("currency_code", "EUR"))
        if len(unique_types) == 1:
            line_color = _distinct_normalized_color(distinct_color_idx)
            distinct_color_idx += 1
        else:
            line_color = _normalized_group_color(
                is_etf=is_etf,
                group_index=etf_color_idx if is_etf else non_etf_color_idx,
            )
            if is_etf:
                etf_color_idx += 1
            else:
                non_etf_color_idx += 1
        local_price_hover = (
            f"Price ({currency_code}): %{{customdata[8]:,.2f}}<br>"
            if currency_code != "EUR"
            else ""
        )
        latest_local_price_hover = (
            f"Most recent price ({currency_code}): %{{customdata[9]:,.2f}}<br>"
            if currency_code != "EUR"
            else ""
        )
        hovertemplate = (
            "Ticker: %{fullData.name}<br>"
            "Product: %{customdata[0]}<br>"
            "Price (EUR): %{customdata[1]:,.2f}<br>"
            + local_price_hover
            + "Normalized price: %{y:.4f}<br>"
            + "Most recent price (EUR): %{customdata[2]:,.2f}<br>"
            + latest_local_price_hover
            + "Most recent price date: %{customdata[3]}<br>"
            + "Most recent normalized price: %{customdata[4]:.4f}<br>"
            + "Holding type: %{customdata[5]}<br>"
            + "Target status: %{customdata[6]}<br>"
            + "Threshold status: %{customdata[7]}<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=spec["x"],
                y=spec["y"],
                mode="lines",
                name=str(spec["ticker"]),
                uid=str(spec["instrument_id"]),
                customdata=spec["customdata"],
                line=dict(
                    color=line_color,
                    dash="solid",
                    width=float(spec["line_width"]),
                ),
                opacity=float(spec["line_opacity"]),
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        uirevision="normalized-median-chart",
        title=(
            f"Stock Development Normalized to Rolling Median "
            f"({median_window_months}m window, {lookback_label} lookback)"
            f"{' | Butterworth smoothed' if apply_lowpass_filter else ''}"
        ),
        yaxis=dict(title="Normalized price"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=10, r=240, t=50, b=40),
    )
    return fig


def build_normalized_median_window_switcher_figure(
    figures_by_window: dict[int, go.Figure],
    *,
    default_window_months: int | None = None,
) -> go.Figure:
    valid_figures = {
        int(window_months): figure
        for window_months, figure in figures_by_window.items()
        if isinstance(figure, go.Figure)
    }
    if not valid_figures:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No normalized price series available")
        return fig

    available_windows = sorted(valid_figures)
    default_window = (
        int(default_window_months)
        if default_window_months is not None and int(default_window_months) in valid_figures
        else available_windows[0]
    )
    base_figure = valid_figures[default_window]
    combined = go.Figure()
    titles_by_window: dict[int, str] = {}
    window_trace_indexes: dict[int, list[int]] = {}

    for window_months in available_windows:
        current_figure = valid_figures[window_months]
        titles_by_window[window_months] = str(
            current_figure.layout.title.text
            if current_figure.layout.title.text is not None
            else f"Stock Development Normalized to Rolling Median ({window_months}m window)"
        )
        window_trace_indexes[window_months] = []
        for trace in current_figure.data:
            payload = trace.to_plotly_json()
            payload["visible"] = window_months == default_window
            uid = payload.get("uid")
            if uid is not None:
                payload["uid"] = f"{uid}__{window_months}m"
            combined.add_trace(go.Scatter(**payload))
            window_trace_indexes[window_months].append(len(combined.data) - 1)

    base_layout = base_figure.layout.to_plotly_json()
    base_layout.pop("updatemenus", None)
    base_layout.pop("template", None)
    combined.update_layout(template="plotly_white", **base_layout)
    base_margin = dict(base_layout.get("margin", {}))
    base_margin["t"] = max(int(base_margin.get("t", 0) or 0), 120)
    title_layout = {
        "x": 0.01,
        "xanchor": "left",
        "y": 0.98,
        "yanchor": "top",
    }
    combined.update_layout(
        title={"text": titles_by_window[default_window], **title_layout},
        margin=base_margin,
    )

    if len(available_windows) > 1:
        total_traces = len(combined.data)
        buttons: list[dict[str, object]] = []
        for window_months in available_windows:
            visible = [False] * total_traces
            for idx in window_trace_indexes[window_months]:
                visible[idx] = True
            buttons.append(
                {
                    "label": (
                        f"{window_months} months"
                        if window_months in {3, 6, 12}
                        else f"Custom ({window_months} months)"
                    ),
                    "method": "update",
                    "args": [
                        {"visible": visible},
                        {"title": {"text": titles_by_window[window_months], **title_layout}},
                    ],
                }
            )
        combined.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "showactive": True,
                    "active": available_windows.index(default_window),
                    "buttons": buttons,
                    "x": 1.0,
                    "xanchor": "right",
                    "y": 1.20,
                    "yanchor": "top",
                    "pad": {"r": 8, "t": 0},
                }
            ]
        )

    return combined


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
    target_per_holding_pct: float | None = None,
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
    target_value = pd.to_numeric(target_per_holding_pct, errors="coerce")
    if not np.isfinite(target_value) and "target_per_holding_pct" in data.columns:
        target_series = pd.to_numeric(data["target_per_holding_pct"], errors="coerce")
        positive_targets = target_series[target_series > 0.0]
        if not positive_targets.empty:
            target_value = float(positive_targets.max())
        elif target_series.notna().any():
            target_value = float(target_series.fillna(0.0).max())
    if np.isfinite(target_value):
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Target:<br>{float(target_value):.1f}%",
            showarrow=False,
            font=dict(size=14, color="#4b5563"),
            align="center",
        )
    return fig
