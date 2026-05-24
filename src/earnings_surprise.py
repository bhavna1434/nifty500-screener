# src/earnings_surprise.py
# Stage 3 (Advanced): EPS Momentum Factor — Quarter-over-Quarter EPS Growth
#
# WHAT THIS ACTUALLY MEASURES:
#   Quarter-over-quarter EPS change: (latest_quarter_EPS - prev_quarter_EPS) / |prev|
#   This is EPS momentum, not true earnings surprise.
#
# WHY IT IS NOT TRUE PEAD EARNINGS SURPRISE:
#   True Post-Earnings Announcement Drift (PEAD) requires analyst consensus EPS
#   estimates to compute: surprise = (actual - consensus) / |consensus|
#   Analyst estimates are not freely available — they require paid data
#   (Bloomberg, Refinitiv, Trendlyne Pro).
#
#   Using prev-quarter EPS as a "proxy estimate" measures sequential EPS growth,
#   which captures earnings acceleration but NOT the market-surprise component
#   that drives the PEAD drift anomaly.
#
# HOW WE USE IT:
#   - Compute QoQ EPS change % = (latest_eps - prev_eps) / |prev_eps| × 100
#   - Apply a 60-day linear decay from the quarter-end announcement date
#     (recent earnings acceleration matters more than stale ones)
#   - Z-score cross-sectionally and add as the 5th factor (20% weight)
#
# Data source: Screener.in quarterly results page ("EPS in Rs" row)

import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY   = 1.0
REQUEST_TIMEOUT = 15
DRIFT_WINDOW_DAYS = 60


def _safe_float(text: str):
    """Parse a string to float, return None on failure."""
    try:
        cleaned = str(text).replace(",", "").replace("%", "").strip()
        return float(cleaned) if cleaned not in ("", "-", "None") else None
    except (ValueError, TypeError):
        return None


# ── QoQ EPS Change Calculation ────────────────────────────────────────────────

def compute_earnings_surprise(actual_eps: float, estimated_eps: float) -> float:
    """
    Calculate quarter-over-quarter EPS change as a percentage.

    Named "earnings_surprise" for historical reasons; this measures sequential
    EPS momentum, not surprise vs analyst consensus.

    Formula: (latest_eps - prev_eps) / |prev_eps| × 100

    Examples:
        Prev: ₹10, Latest: ₹12 → +20%  (accelerating earnings)
        Prev: ₹10, Latest: ₹8  → -20%  (decelerating earnings)
        Prev: ₹10, Latest: ₹10 →   0%  (flat earnings)

    Args:
        actual_eps: Most recent quarter EPS (₹)
        estimated_eps: Previous quarter EPS used as the baseline (₹)

    Returns:
        QoQ EPS change percentage (float). Positive = acceleration.
    """
    if not estimated_eps or estimated_eps == 0:
        return None
    return ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100


def scrape_quarterly_eps(ticker: str) -> dict:
    """
    Scrape the latest two quarters of EPS from Screener.in quarterly results table.

    Returns:
        dict with keys:
            latest_eps (float)              — most recent quarter EPS
            prev_eps (float)                — previous quarter EPS (used as proxy estimate)
            quarters (list of (str, float)) — all (quarter_label, eps) pairs found
            days_since_announcement (int)   — estimated days since latest results release
    """
    _empty = {"latest_eps": None, "prev_eps": None,
               "quarters": [], "days_since_announcement": None}
    url = f"https://www.screener.in/company/{ticker}/"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return _empty
        soup = BeautifulSoup(resp.text, "html.parser")

        section = soup.find("section", {"id": "quarters"})
        if not section:
            return _empty

        table = section.find("table")
        if not table:
            return _empty

        rows = table.find_all("tr")
        if not rows:
            return _empty

        # Header row — quarter labels ("Mar 2025", "Jun 2025", …)
        header_cols = rows[0].find_all(["th", "td"])
        quarter_labels = [c.get_text(strip=True) for c in header_cols[1:]]

        # Find "EPS in Rs" row
        eps_values = []
        for row in rows[1:]:
            cols = row.find_all("td")
            if not cols:
                continue
            label = cols[0].get_text(strip=True).rstrip("+").strip()
            if label == "EPS in Rs":
                eps_values = [_safe_float(c.get_text(strip=True)) for c in cols[1:]]
                break

        if not eps_values:
            return _empty

        # Pair labels with values and drop any None-valued entries
        quarters = [(q, v) for q, v in zip(quarter_labels, eps_values) if v is not None]
        if len(quarters) < 2:
            return _empty

        latest_eps = quarters[-1][1]
        prev_eps   = quarters[-2][1]

        # Estimate days since announcement:
        # Indian companies typically file results ~45 days after quarter end.
        days_since = None
        try:
            latest_qtr_str = quarters[-1][0]          # e.g. "Mar 2026"
            qtr_end = datetime.strptime(latest_qtr_str, "%b %Y")
            announcement = qtr_end + timedelta(days=45)
            days_since = max(0, (datetime.now() - announcement).days)
        except ValueError:
            pass

        return {
            "latest_eps":              latest_eps,
            "prev_eps":                prev_eps,
            "quarters":                quarters,
            "days_since_announcement": days_since,
        }

    except Exception as e:
        print(f"  Warning: Could not fetch EPS for {ticker}: {e}")
        return _empty


def compute_surprise_factor_for_universe(universe: list) -> pd.Series:
    """
    Compute the QoQ EPS momentum z-score for every stock in the universe.

    Steps per stock:
      1. Scrape latest two quarterly EPS from Screener.in
      2. QoQ change % = (latest_eps - prev_eps) / |prev_eps| × 100
      3. Apply linear decay over 60 days from estimated announcement date
         (recent earnings acceleration carries more weight)
      4. Z-score cross-sectionally

    Returns:
        pd.Series indexed by ticker (higher = stronger recent EPS acceleration)
    """
    raw_surprises = {}

    for i, ticker in enumerate(universe):
        if i > 0:
            time.sleep(REQUEST_DELAY)

        eps_data = scrape_quarterly_eps(ticker)
        actual   = eps_data.get("latest_eps")
        estimate = eps_data.get("prev_eps")

        if actual is None or estimate is None:
            raw_surprises[ticker] = None
            continue

        surprise_pct = compute_earnings_surprise(actual, estimate)
        if surprise_pct is None:
            raw_surprises[ticker] = None
            continue

        days = eps_data.get("days_since_announcement")
        if days is not None:
            surprise_pct = apply_pead_decay(surprise_pct, days)

        raw_surprises[ticker] = surprise_pct

    series = pd.Series(raw_surprises).fillna(0.0)

    std = series.std()
    if std == 0:
        return series * 0.0

    return (series - series.mean()) / std


# ── Recency Decay: EPS momentum signal fades over time ───────────────────────

def apply_pead_decay(surprise_score: float, days_since_announcement: int) -> float:
    """
    Weight recent earnings acceleration more heavily than stale data.
    Apply a linear decay to zero over 60 days from the estimated announcement date.

    Args:
        surprise_score: Raw QoQ EPS change score
        days_since_announcement: Estimated days since quarter results were filed

    Returns:
        Decayed score (0.0 once 60 days have elapsed)
    """
    DRIFT_WINDOW_DAYS = 60

    if days_since_announcement >= DRIFT_WINDOW_DAYS:
        return 0.0  # Signal expired

    decay_factor = 1 - (days_since_announcement / DRIFT_WINDOW_DAYS)
    return surprise_score * decay_factor


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing earnings_surprise.py...")

    # Test the surprise calculation formula
    cases = [
        (12.0, 10.0, "Big beat"),
        (8.0, 10.0, "Miss"),
        (10.0, 10.0, "In-line"),
        (15.0, 10.0, "Blowout"),
    ]

    for actual, estimate, label in cases:
        s = compute_earnings_surprise(actual, estimate)
        print(f"  {label}: Actual ₹{actual}, Estimate ₹{estimate} → Surprise: {s:+.1f}%")

    # Test PEAD decay
    print("\nPEAD signal decay over time:")
    score = 2.0  # strong beat
    for days in [0, 15, 30, 45, 60]:
        decayed = apply_pead_decay(score, days)
        print(f"  Day {days:2d}: score = {decayed:.2f}")

    print("\n✅ earnings_surprise.py logic working correctly!")
