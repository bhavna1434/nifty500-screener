# src/factor_model.py
# Stage 3: Yellow-Flag Factor Ranking
#
# Ranks stocks on 5 factors using cross-sectional z-score normalization:
#   1. Value      — P/E + EV/EBITDA (lower is better → sign is flipped)
#   2. Growth     — Revenue CAGR + EPS CAGR (higher is better)
#   3. Quality    — ROE + ROCE (higher is better)
#   4. Momentum   — 6M price return, skipping most recent month (higher is better)
#   5. Earnings Surprise — PEAD signal (higher surprise % is better)
#
# Why cross-sectional z-scoring? See 02_THEORY_DEEP_DIVE.md Section 7.
# Short: all factors live on different scales; z-scoring puts them on the
# same scale (mean=0, std=1) so they can be combined fairly.

import pandas as pd
import numpy as np
from src.utils import z_score, cagr, pct_change_n_months


# ── Default factor weights (sum to 1.0) ───────────────────────────────────────
DEFAULT_WEIGHTS = {
    "value":    0.20,
    "growth":   0.20,
    "quality":  0.20,
    "momentum": 0.20,
    "surprise": 0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum(price_df: pd.DataFrame, months: int = 6) -> pd.Series:
    """
    Compute N-month price return for each stock, skipping the most recent month.

    Why skip the most recent month?
    Jegadeesh & Titman (1993) and subsequent research show stocks exhibit
    1-month SHORT-TERM REVERSAL (microstructure effects, bid-ask bounce).
    Including it weakens the momentum signal. The standard convention is:
        momentum = return from (months + 1) months ago to 1 month ago

    Formula:
        momentum_i = (P_i[t-21] / P_i[t-21*(months+1)] - 1) × 100

    Where t is today, 21 trading days ≈ 1 calendar month.

    Args:
        price_df: DataFrame of daily closing prices (rows=dates, cols=tickers)
        months:   Lookback in months (default 6). 6M is standard for India.

    Returns:
        pd.Series indexed by ticker — raw momentum return (%, not yet z-scored)
    """
    skip_days    = 21                    # skip most recent 1 month
    lookback_days = months * 21          # e.g. 6 months = 126 trading days

    min_required = lookback_days + skip_days + 5   # buffer

    if len(price_df) < min_required:
        raise ValueError(
            f"Need at least {min_required} days of data for {months}-month momentum. "
            f"Got {len(price_df)} rows."
        )

    # Price 1 month ago (the "recent" end, skipping reversal period)
    price_end   = price_df.iloc[-(skip_days + 1)]
    # Price (months + 1) months ago (the "start" of the lookback)
    price_start = price_df.iloc[-(lookback_days + skip_days + 1)]

    momentum = ((price_end - price_start) / price_start.replace(0, np.nan)) * 100
    return momentum.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL FACTORS (from fundamentals_df)
# ══════════════════════════════════════════════════════════════════════════════
#
# fundamentals_df expected columns:
#   ticker, pe_ratio, ev_ebitda, roe, roce,
#   revenue_cagr_3y, eps_cagr_3y
#
# These are scraped from Screener.in by fundamental_filter.py
# ══════════════════════════════════════════════════════════════════════════════

def compute_value_score(fundamentals_df: pd.DataFrame) -> pd.Series:
    """
    Compute value factor z-score from P/E and EV/EBITDA.

    Value = lower ratio → better (cheaper stock).
    We average the two metrics then FLIP the sign so that a higher z-score
    means a CHEAPER (better) stock.

    Formula:
        raw_value_i = (PE_i + EVEBITDA_i) / 2    [average of two ratios]
        value_zscore_i = -z_score(raw_value_i)    [flipped: lower PE → higher score]

    Outlier handling: winsorise at 2nd–98th percentile to prevent a stock with
    PE = 500 from dominating the cross-sectional distribution.

    Returns:
        pd.Series indexed by ticker (higher = cheaper = better)
    """
    df = fundamentals_df.copy().set_index("ticker")

    # Fill missing ev_ebitda with NaN so we can average available metrics
    if "ev_ebitda" not in df.columns:
        df["ev_ebitda"] = float("nan")
    else:
        df["ev_ebitda"] = pd.to_numeric(df["ev_ebitda"], errors="coerce")
    df["pe_ratio"] = pd.to_numeric(df["pe_ratio"], errors="coerce")

    # Keep stocks with at least a valid, positive P/E
    df = df[df["pe_ratio"] > 0].copy()
    if df.empty:
        return pd.Series(dtype=float)

    # Winsorise each metric independently at 2nd–98th percentile
    for col in ["pe_ratio", "ev_ebitda"]:
        valid = df[col].dropna()
        if len(valid) >= 4:
            lo, hi = valid.quantile(0.02), valid.quantile(0.98)
            df[col] = df[col].clip(lo, hi)

    # Average whichever metrics are available per stock
    raw = df[["pe_ratio", "ev_ebitda"]].mean(axis=1, skipna=True)
    return -z_score(raw)   # negative sign: low valuation → high z-score


def compute_quality_score(fundamentals_df: pd.DataFrame) -> pd.Series:
    """
    Compute quality factor z-score from ROE and ROCE.

    Why both?
    - ROE can be inflated by leverage (DuPont decomposition)
    - ROCE is leverage-neutral — measures return on ALL capital
    Using both together rewards companies with high returns that are
    NOT driven by excessive debt.

    Formula:
        raw_quality_i = (ROE_i + ROCE_i) / 2
        quality_zscore_i = z_score(raw_quality_i)

    Returns:
        pd.Series indexed by ticker (higher = better quality)
    """
    df = fundamentals_df.copy().set_index("ticker")
    df = df.dropna(subset=["roe", "roce"])

    # Winsorise at 2nd–98th percentile
    for col in ["roe", "roce"]:
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        df[col] = df[col].clip(lo, hi)

    raw = (df["roe"] + df["roce"]) / 2
    return z_score(raw)


def compute_growth_score(fundamentals_df: pd.DataFrame) -> pd.Series:
    """
    Compute growth factor z-score from 3-year Revenue CAGR and EPS CAGR.

    Formula:
        CAGR = (End / Start)^(1/3) - 1   (see utils.py)
        raw_growth_i = (rev_cagr_i + eps_cagr_i) / 2
        growth_zscore_i = z_score(raw_growth_i)

    Why 3-year CAGR and not 1-year YoY?
    1-year growth is noisy — one bad year tanks the score even for a great
    business. 3-year CAGR smooths out the noise and captures the trend.

    Returns:
        pd.Series indexed by ticker (higher = faster growing)
    """
    df = fundamentals_df.copy().set_index("ticker")

    # Old cache entries may contain complex-number strings like "(7.87+0j)"
    # (produced when a negative EPS base was raised to 1/3 in an earlier version).
    # Extract the real part; anything that can't be parsed becomes NaN.
    for col in ["revenue_cagr_3y", "eps_cagr_3y"]:
        if col in df.columns:
            def _to_real(x):
                try:
                    return float(x)
                except (ValueError, TypeError):
                    try:
                        return complex(str(x)).real
                    except Exception:
                        return np.nan
            df[col] = df[col].apply(_to_real)

    df = df.dropna(subset=["revenue_cagr_3y", "eps_cagr_3y"])

    # Winsorise
    for col in ["revenue_cagr_3y", "eps_cagr_3y"]:
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        df[col] = df[col].clip(lo, hi)

    raw = (df["revenue_cagr_3y"] + df["eps_cagr_3y"]) / 2
    return z_score(raw)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE RANKING
# ══════════════════════════════════════════════════════════════════════════════

def rank_stocks(
    universe: list,
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    surprise_scores: pd.Series = None,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Main function: rank all stocks in the universe by composite factor score.

    Steps:
      1. Compute each factor signal (raw)
      2. Z-score each signal cross-sectionally
      3. Combine into weighted composite score
      4. Sort descending (highest composite = best rank)

    Args:
        universe:        List of tickers passing Stage 2 (Red-Flag filter)
        price_df:        Daily closing prices DataFrame (all stocks)
        fundamentals_df: Fundamental data DataFrame with columns:
                         ticker, pe_ratio, ev_ebitda, roe, roce,
                         revenue_cagr_3y, eps_cagr_3y
        surprise_scores: Optional pd.Series of PEAD z-scores (from earnings_surprise.py)
                         If None, earnings surprise factor is set to 0 for all stocks.
        weights:         Optional dict overriding DEFAULT_WEIGHTS.
                         Must contain: value, growth, quality, momentum, surprise.
                         Values must sum to 1.0.

    Returns:
        DataFrame sorted by rank (rank=1 is best), with columns:
        ticker, rank, composite_score, value_score, growth_score,
        quality_score, momentum_score, surprise_score
    """
    w = weights if weights else DEFAULT_WEIGHTS

    # Validate weights sum to ~1.0
    total_w = sum(w.values())
    if abs(total_w - 1.0) > 0.01:
        # Normalise if they don't sum to 1
        w = {k: v / total_w for k, v in w.items()}

    # ── Filter price_df and fundamentals_df to universe ───────────────────────
    avail_price = [t for t in universe if t in price_df.columns]
    price_sub   = price_df[avail_price].copy()

    fund_sub = fundamentals_df[fundamentals_df["ticker"].isin(universe)].copy()

    results = []

    # ── Compute factor z-scores ───────────────────────────────────────────────
    try:
        momentum_raw = compute_momentum(price_sub, months=6)
        momentum_z   = z_score(momentum_raw)
    except Exception as e:
        print(f"  Warning: Momentum factor failed: {e}")
        momentum_z = pd.Series(0.0, index=avail_price)

    try:
        value_z = compute_value_score(fund_sub)
    except Exception as e:
        print(f"  Warning: Value factor failed: {e}")
        value_z = pd.Series(0.0, index=fund_sub["ticker"].tolist())

    try:
        quality_z = compute_quality_score(fund_sub)
    except Exception as e:
        print(f"  Warning: Quality factor failed: {e}")
        quality_z = pd.Series(0.0, index=fund_sub["ticker"].tolist())

    try:
        growth_z = compute_growth_score(fund_sub)
    except Exception as e:
        print(f"  Warning: Growth factor failed: {e}")
        growth_z = pd.Series(0.0, index=fund_sub["ticker"].tolist())

    # Earnings surprise — optional; default to 0 if not provided
    if surprise_scores is not None:
        surprise_z = surprise_scores.reindex(universe).fillna(0.0)
    else:
        surprise_z = pd.Series(0.0, index=universe)

    # ── Build composite score for each stock ──────────────────────────────────
    for ticker in universe:
        vs = float(value_z.get(ticker, 0.0))
        gs = float(growth_z.get(ticker, 0.0))
        qs = float(quality_z.get(ticker, 0.0))
        ms = float(momentum_z.get(ticker, 0.0))
        ss = float(surprise_z.get(ticker, 0.0))

        composite = (
            w["value"]    * vs +
            w["growth"]   * gs +
            w["quality"]  * qs +
            w["momentum"] * ms +
            w["surprise"] * ss
        )

        results.append({
            "ticker":          ticker,
            "composite_score": round(composite, 4),
            "value_score":     round(vs, 4),
            "growth_score":    round(gs, 4),
            "quality_score":   round(qs, 4),
            "momentum_score":  round(ms, 4),
            "surprise_score":  round(ss, 4),
        })

    # ── Sort by composite score descending, assign rank ───────────────────────
    ranked = (
        pd.DataFrame(results)
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    return ranked


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    print("Testing factor_model.py with live data...\n")

    # Small sample to test quickly
    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
               "WIPRO", "HINDUNILVR", "KOTAKBANK", "LT", "AXISBANK"]
    nse = [t + ".NS" for t in tickers]

    print("Downloading price data...")
    raw = yf.download(nse, period="1y", progress=False)["Close"]
    raw.columns = [c.replace(".NS", "") for c in raw.columns]

    # Dummy fundamentals for quick test
    fund_data = pd.DataFrame([
        {"ticker": t, "pe_ratio": 15 + i*3, "ev_ebitda": 10 + i*2,
         "roe": 20 - i, "roce": 18 - i,
         "revenue_cagr_3y": 12 + i, "eps_cagr_3y": 10 + i}
        for i, t in enumerate(tickers)
    ])

    ranked = rank_stocks(
        universe=tickers,
        price_df=raw,
        fundamentals_df=fund_data,
    )

    print("\nTop 10 ranked stocks (equal-weight factors, dummy fundamentals):")
    print(ranked[["rank", "ticker", "composite_score",
                  "value_score", "growth_score",
                  "quality_score", "momentum_score"]].to_string(index=False))
    print("\n✅ factor_model.py working correctly!")
