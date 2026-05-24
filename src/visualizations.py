# src/visualizations.py
# All Plotly chart functions for the Streamlit dashboard
#
# Includes:
#   1. Correlation matrix heatmap (top-stock return correlations)
#   2. Factor attribution bar chart (what drove each stock's composite score)
#   3. Regime indicator gauge
#   4. Rank history line chart (how a stock's rank changed over time)
#
# All functions return a plotly Figure object. In app.py, call:
#   st.plotly_chart(fig, use_container_width=True)
#
# We'll build this in Weeks 7–9.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION MATRIX HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(price_df: pd.DataFrame, tickers: list, period_days: int = 252) -> go.Figure:
    """
    Create a Plotly heatmap showing pairwise return correlations between stocks.

    Why this matters:
        If your top 10 stocks are all correlated > 0.8, they'll all fall together
        in a crash. This heatmap reveals hidden concentration risk.
        Ideal: pick stocks with correlation < 0.5 to each other.

    Args:
        price_df: DataFrame of closing prices (rows=dates, columns=tickers)
        tickers: List of ticker symbols to include (top 15–20 recommended)
        period_days: How many days of returns to use (252 = 1 year)

    Returns:
        Plotly Figure (heatmap)
    """
    # Get the subset of stocks and compute daily returns
    available = [t for t in tickers if t in price_df.columns]
    prices = price_df[available].tail(period_days)
    returns = prices.pct_change().dropna()

    # Compute the correlation matrix
    corr_matrix = returns.corr().round(2)

    # Color scale: red for high correlation (bad — similar), blue for low (good — diversified)
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",   # Red = high corr, Blue = low corr
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Correlation", thickness=15, len=0.8)
    ))

    fig.update_layout(
        title=dict(text="Return Correlation Matrix — Top Ranked Stocks", font=dict(size=14)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        height=500,
        margin=dict(l=100, r=80, t=60, b=100),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. FACTOR ATTRIBUTION BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_attribution(ranked_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Stacked bar chart showing what each factor contributed to each stock's
    composite score.

    This is the single most impressive visualization for Modulor — it shows
    exactly WHY each stock ranked highly (e.g. "RELIANCE is high because of
    strong momentum and EPS momentum, not value").

    Args:
        ranked_df: DataFrame with columns:
            ticker, value_score, growth_score, quality_score,
            momentum_score, eps_momentum_score (all z-scored)
        top_n: How many top stocks to show

    Returns:
        Plotly Figure (stacked bar chart)
    """
    df = ranked_df.head(top_n).copy()

    factor_cols = ["value_score", "growth_score", "quality_score",
                   "momentum_score", "eps_momentum_score"]
    labels = ["Value", "Growth", "Quality", "Momentum", "EPS Momentum"]
    colors = ["#378ADD", "#1D9E75", "#7F77DD", "#BA7517", "#D85A30"]

    fig = go.Figure()

    for col, label, color in zip(factor_cols, labels, colors):
        if col not in df.columns:
            continue
        fig.add_trace(go.Bar(
            name=label,
            x=df["ticker"],
            y=df[col].round(2),
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        barmode="relative",          # stacked with negatives going down
        title=dict(text="Factor Attribution — What Drives Each Stock's Rank", font=dict(size=14)),
        xaxis=dict(title="Stock", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title="Z-Score Contribution", zeroline=True, zerolinecolor="#ccc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=60, r=20, t=80, b=100),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. MARKET REGIME GAUGE
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_gauge(regime: str, nifty_pct_from_ma: float) -> go.Figure:
    """
    A dial/gauge showing the current market regime.

    Args:
        regime: "Risk-On", "Neutral", or "Risk-Off"
        nifty_pct_from_ma: How far Nifty 50 is from its 200-day MA (%)

    Returns:
        Plotly Figure (gauge chart)
    """
    regime_colors = {
        "Risk-On": "#1D9E75",    # green
        "Neutral": "#BA7517",    # amber
        "Risk-Off": "#D85A30",   # red/coral
    }
    color = regime_colors.get(regime, "#888")

    # Clamp the value to the gauge range
    gauge_value = max(-15, min(15, nifty_pct_from_ma))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        number={"suffix": "%", "font": {"size": 24}},
        title={"text": f"Market Regime: <b>{regime}</b>", "font": {"size": 14}},
        delta={"reference": 0, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [-15, 15], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [-15, -5], "color": "#FAECE7"},   # red zone
                {"range": [-5, 5],   "color": "#FAEEDA"},   # amber zone
                {"range": [5, 15],   "color": "#E1F5EE"},   # green zone
            ],
            "threshold": {
                "line": {"color": "#333", "width": 2},
                "thickness": 0.75,
                "value": gauge_value,
            },
        }
    ))

    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. STOCK RANK HISTORY LINE CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_rank_history(ticker: str, history_df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing how a stock's rank changed across screener runs.
    Used in the single-stock deep-dive view.

    Args:
        ticker: Stock symbol
        history_df: Output of history_tracker.get_stock_history(ticker)
                    Columns: run_date, rank, composite_score

    Returns:
        Plotly Figure (line chart, rank axis inverted so #1 is at the top)
    """
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"{ticker} not found in history", showarrow=False)
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_df["run_date"],
        y=history_df["rank"],
        mode="lines+markers",
        line=dict(color="#378ADD", width=2),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Rank: #%{y}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Rank History", font=dict(size=14)),
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Rank",
            autorange="reversed",   # Rank #1 at TOP of chart
            tickformat="d",
        ),
        height=300,
        margin=dict(l=60, r=20, t=50, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. FACTOR SCORE RADAR CHART (single stock)
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_radar(ticker: str, scores: dict) -> go.Figure:
    """
    Spider/radar chart showing a single stock's factor scores.
    Great for the stock detail page.

    Args:
        ticker: Stock symbol
        scores: dict with keys value, growth, quality, momentum, eps_momentum
                (z-scores — values typically between -3 and +3)

    Returns:
        Plotly Figure (radar chart)
    """
    categories = ["Value", "Growth", "Quality", "Momentum", "EPS\nMomentum"]
    values = [
        scores.get("value", 0),
        scores.get("growth", 0),
        scores.get("quality", 0),
        scores.get("momentum", 0),
        scores.get("eps_momentum", 0),
    ]
    # Close the radar loop
    values_loop = values + [values[0]]
    categories_loop = categories + [categories[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_loop,
        theta=categories_loop,
        fill="toself",
        fillcolor="rgba(55, 138, 221, 0.2)",
        line=dict(color="#378ADD", width=2),
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2]),
        ),
        title=dict(text=f"{ticker} — Factor Profile", font=dict(size=14)),
        height=350,
        margin=dict(l=60, r=60, t=60, b=40),
        showlegend=False,
    )

    return fig
