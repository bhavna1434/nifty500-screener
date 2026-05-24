# src/technical_filter.py
# Stage 4: Green-Flag Technical Entry Signals
# Applies RSI, Moving Average, and 52-week proximity filters
# We'll build this in Week 7

import pandas as pd
import ta  # pip install ta


# ── Technical Filter Thresholds ───────────────────────────────────────────────
TECH_RULES = {
    "rsi_max": 70,              # RSI must be below 70 (not overbought)
    "price_above_ma50": True,   # Price must be above 50-day moving average
    "pct_from_52w_high_max": 20, # Price within 20% of 52-week high
}


def compute_rsi(prices: pd.Series, window: int = 14) -> float:
    """
    Compute the Relative Strength Index (RSI) for a stock.

    RSI > 70: Overbought (avoid buying)
    RSI < 30: Oversold (potential buy opportunity)
    RSI 30-70: Neutral

    Args:
        prices: Series of closing prices
        window: RSI lookback period (standard = 14 days)

    Returns:
        Latest RSI value (float)
    """
    rsi_series = ta.momentum.RSIIndicator(close=prices.squeeze(), window=window).rsi()
    return rsi_series.iloc[-1]


def compute_ma(prices: pd.Series, window: int = 50) -> float:
    """Compute the N-day simple moving average."""
    return prices.rolling(window=window).mean().iloc[-1]


def compute_52w_proximity(prices: pd.Series) -> float:
    """
    Calculate how far the current price is from the 52-week high (as a %).
    0% = at the 52-week high. 20% = 20% below the 52-week high.
    """
    lookback = min(252, len(prices))
    high_52w = prices.iloc[-lookback:].max()
    current = prices.iloc[-1]
    return ((high_52w - current) / high_52w) * 100


def apply_green_flag_filter(top_stocks: list, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply technical filters to the top-ranked stocks from Stage 3.

    Args:
        top_stocks: List of tickers (top 50 from factor model)
        price_df: DataFrame of closing prices

    Returns:
        DataFrame of stocks that pass all technical checks, with signal columns
    """
    results = []

    for ticker in top_stocks:
        if ticker not in price_df.columns:
            continue

        prices = price_df[ticker].dropna()
        if len(prices) < 50:
            continue  # need at least enough data for MA50 and RSI

        rsi = compute_rsi(prices)
        ma50 = compute_ma(prices, window=50)
        current_price = prices.iloc[-1]
        above_ma50 = current_price >= ma50
        pct_from_high = compute_52w_proximity(prices)

        # RSI must be strictly below 70 — at or above 70 means overbought
        passes = (
            rsi < TECH_RULES["rsi_max"]
            and above_ma50
            and pct_from_high <= TECH_RULES["pct_from_52w_high_max"]
        )

        results.append({
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "rsi": round(rsi, 1),
            "ma_50": round(ma50, 2),
            "above_ma50": above_ma50,
            "pct_from_52w_high": round(pct_from_high, 1),
            "passes": passes,
        })

    df = pd.DataFrame(results)
    return df.reset_index(drop=True)
