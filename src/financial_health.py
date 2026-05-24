# src/financial_health.py
# Stage 2 (Advanced): Piotroski F-Score + Altman Z-Score
#
# These are two of the most respected models in quantitative finance.
# Both are purely rule-based — no ML, no magic — just well-tested formulas.
#
# We'll build this in Week 4 after the basic Red-Flag filter is working.

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE
# ══════════════════════════════════════════════════════════════════════════════
#
# Created by Joseph Piotroski (Stanford, 2000).
# Scores a company 0–9 across 9 binary criteria in 3 buckets:
#   A. Profitability (4 criteria)
#   B. Leverage / Liquidity (3 criteria)
#   C. Operating Efficiency (2 criteria)
#
# Score interpretation:
#   0–2  → Very weak (avoid)
#   3–5  → Neutral
#   6–9  → Strong (Piotroski called these "winners")
#
# We use F-Score >= 5 as our minimum threshold for passing Stage 2.
# ══════════════════════════════════════════════════════════════════════════════

# Minimum F-Score to pass Stage 2
PIOTROSKI_THRESHOLD = 5


def compute_piotroski_fscore(financials: dict) -> dict:
    """
    Compute the Piotroski F-Score for a single stock.

    Args:
        financials: dict with the following keys (all floats, from Screener.in):
            - net_income          (this year and last year)
            - total_assets        (this year and last year)
            - operating_cash_flow (this year)
            - long_term_debt      (this year and last year)
            - current_assets      (this year and last year)
            - current_liabilities (this year and last year)
            - shares_outstanding  (this year and last year)
            - gross_profit        (this year and last year)
            - revenue             (this year and last year)

    Returns:
        dict with keys:
            - score (int 0-9)
            - breakdown (dict showing each of the 9 criteria)
            - passes (bool: True if score >= PIOTROSKI_THRESHOLD)
    """
    score = 0
    breakdown = {}

    # Normalise: replace None with 0 so missing data scores 0, not a crash
    _NUMERIC_KEYS = [
        "net_income", "net_income_prev", "total_assets", "total_assets_prev",
        "operating_cash_flow", "long_term_debt", "long_term_debt_prev",
        "current_assets", "current_assets_prev", "current_liabilities",
        "current_liabilities_prev", "shares_outstanding", "shares_outstanding_prev",
        "gross_profit", "gross_profit_prev", "revenue", "revenue_prev",
        "ebit", "working_capital", "retained_earnings", "book_value_equity",
        "total_liabilities",
    ]
    financials = {k: (financials.get(k) or 0) for k in _NUMERIC_KEYS}

    # ── A. PROFITABILITY ──────────────────────────────────────────────────────

    # A1. ROA > 0 (Return on Assets is positive this year)
    roa = (financials.get("net_income") or 0) / max(financials.get("total_assets") or 1, 1)
    a1 = 1 if roa > 0 else 0
    breakdown["A1_roa_positive"] = a1
    score += a1

    # A2. Operating Cash Flow > 0
    a2 = 1 if financials.get("operating_cash_flow", 0) > 0 else 0
    breakdown["A2_cfo_positive"] = a2
    score += a2

    # A3. ROA improved vs last year
    roa_prev = (financials.get("net_income_prev") or 0) / max(financials.get("total_assets_prev") or 1, 1)
    a3 = 1 if roa > roa_prev else 0
    breakdown["A3_roa_improved"] = a3
    score += a3

    # A4. Cash flow from operations > Net Income (earnings quality / no accruals)
    a4 = 1 if financials.get("operating_cash_flow", 0) > financials.get("net_income", 0) else 0
    breakdown["A4_cfo_gt_ni"] = a4
    score += a4

    # ── B. LEVERAGE / LIQUIDITY ───────────────────────────────────────────────

    # B5. Long-term debt ratio decreased vs last year
    debt_ratio_now = (financials.get("long_term_debt") or 0) / max(financials.get("total_assets") or 1, 1)
    debt_ratio_prev = (financials.get("long_term_debt_prev") or 0) / max(financials.get("total_assets_prev") or 1, 1)
    b5 = 1 if debt_ratio_now < debt_ratio_prev else 0
    breakdown["B5_leverage_reduced"] = b5
    score += b5

    # B6. Current ratio (current assets / current liabilities) improved
    cr_now = (financials.get("current_assets") or 0) / max(financials.get("current_liabilities") or 1, 1)
    cr_prev = (financials.get("current_assets_prev") or 0) / max(financials.get("current_liabilities_prev") or 1, 1)
    b6 = 1 if cr_now > cr_prev else 0
    breakdown["B6_liquidity_improved"] = b6
    score += b6

    # B7. No new shares issued (dilution check)
    shares_now = financials.get("shares_outstanding") or 1
    shares_prev = financials.get("shares_outstanding_prev") or 1
    b7 = 1 if shares_now <= shares_prev else 0
    breakdown["B7_no_dilution"] = b7
    score += b7

    # ── C. OPERATING EFFICIENCY ───────────────────────────────────────────────

    # C8. Gross margin improved vs last year
    gm_now = (financials.get("gross_profit") or 0) / max(financials.get("revenue") or 1, 1)
    gm_prev = (financials.get("gross_profit_prev") or 0) / max(financials.get("revenue_prev") or 1, 1)
    c8 = 1 if gm_now > gm_prev else 0
    breakdown["C8_margin_improved"] = c8
    score += c8

    # C9. Asset turnover improved (revenue / total assets)
    at_now = (financials.get("revenue") or 0) / max(financials.get("total_assets") or 1, 1)
    at_prev = (financials.get("revenue_prev") or 0) / max(financials.get("total_assets_prev") or 1, 1)
    c9 = 1 if at_now > at_prev else 0
    breakdown["C9_asset_turnover_improved"] = c9
    score += c9

    return {
        "score": score,
        "breakdown": breakdown,
        "passes": score >= PIOTROSKI_THRESHOLD
    }


# ══════════════════════════════════════════════════════════════════════════════
# ALTMAN Z'' SCORE — EMERGING MARKETS MODEL (1995)
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY Z'' AND NOT THE ORIGINAL Z-SCORE:
#
# The original Altman Z (1968) was built on 66 US manufacturing companies.
# It has two problems for Indian Nifty 500 stocks:
#
#   1. Manufacturing bias: X5 = Revenue/Total Assets penalises service companies
#      (IT, finance) that have few hard assets but high revenue productivity.
#      Nifty 500 is ~30% IT and financial services — the original Z is unfair.
#
#   2. Market cap in X4: Uses Market Cap / Book Value of Total Liabilities.
#      Original Z uses MARKET cap. Z'' uses BOOK VALUE of equity — more stable
#      and not distorted by short-term price swings.
#
#   3. Calibrated for US defaults, not Indian IBC (2016) resolution cycles.
#
# Altman, Hartzell & Peck (1995) "Emerging Market Corporate Bonds — A New
# Rating System" introduced Z'' specifically for non-manufacturing / emerging
# market companies:
#
#   Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4
#
# Where:
#   X1 = Working Capital / Total Assets        (liquidity)
#   X2 = Retained Earnings / Total Assets      (cumulative profitability)
#   X3 = EBIT / Total Assets                   (operating efficiency)
#   X4 = Book Value of Equity / Total Liabilities  (leverage cushion)
#
# NOTE: X5 (Revenue/Total Assets) is intentionally REMOVED in Z''.
#
# Z'' Thresholds (different from original Z):
#   Z'' > 2.60  → Safe zone
#   1.10–2.60   → Grey zone (caution)
#   Z'' < 1.10  → Distress zone → FAIL Stage 2
#
# FINANCIAL SECTOR EXCLUSION:
#   Banks, NBFCs, Insurance companies must NOT use Z''.
#   Their balance sheets are structurally different (deposits ≠ debt for banks).
#   For financial sector, use sector-specific metrics (CAR, NPA ratio, ROA).
# ══════════════════════════════════════════════════════════════════════════════

# Z'' thresholds (Altman 1995 — emerging markets model)
ALTMAN_SAFE_ZONE     = 2.60
ALTMAN_DISTRESS_ZONE = 1.10

# Sectors where Z'' should NOT be applied
FINANCIAL_SECTORS = {"Banking", "Finance", "NBFCs", "Insurance", "Financial Services"}


def compute_altman_zscore(financials: dict, sector: str = "") -> dict:
    """
    Compute Altman Z'' Score (1995 Emerging Markets model) for a single stock.

    Use Z'' — NOT the original 1968 Z — for Nifty 500 stocks.
    See the extensive comments above explaining why.

    Args:
        financials: dict with keys (all floats, in ₹ crore or consistent unit):
            - working_capital     = Current Assets - Current Liabilities
            - total_assets        = Total Assets from balance sheet
            - retained_earnings   = Accumulated retained earnings (P&L reserves)
            - ebit                = Earnings Before Interest & Tax
            - book_value_equity   = Total Shareholders' Equity (book value)
            - total_liabilities   = Total Debt + Other Liabilities

        sector: The stock's GICS or NSE sector (string).
                If it's a financial sector, Z'' is skipped.

    Returns:
        dict with keys:
            - model:   "Z'' (Emerging Markets)" or "Skipped (Financial Sector)"
            - z_score: (float) Z'' score
            - zone:    "Safe", "Grey", or "Distress"
            - passes:  (bool) True if not Distress
            - components: dict with X1–X4 values
            - interpretation: plain-English explanation
    """
    # ── Financial sector exclusion ────────────────────────────────────────────
    if sector in FINANCIAL_SECTORS:
        return {
            "model":   "Skipped (Financial Sector)",
            "z_score": None,
            "zone":    "N/A",
            "passes":  True,   # don't penalise banks — use separate check
            "components": {},
            "interpretation": (
                f"{sector} companies should not be evaluated with Z''. "
                "Use Capital Adequacy Ratio (CAR > 12%), NPA Ratio (< 3%), "
                "and Net Interest Margin as alternative financial health metrics."
            )
        }

    ta = max(financials.get("total_assets") or 1, 1)

    # X1: Working Capital / Total Assets
    x1 = (financials.get("working_capital") or 0) / ta

    # X2: Retained Earnings / Total Assets
    x2 = (financials.get("retained_earnings") or 0) / ta

    # X3: EBIT / Total Assets
    x3 = (financials.get("ebit") or 0) / ta

    # X4: Book Value of Equity / Total Liabilities
    x4 = (financials.get("book_value_equity") or 0) / max(financials.get("total_liabilities") or 1, 1)

    # Z'' formula — Altman, Hartzell & Peck (1995)
    z_prime_prime = 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4

    # Zone classification
    if z_prime_prime > ALTMAN_SAFE_ZONE:
        zone = "Safe"
        interp = f"Z'' = {z_prime_prime:.2f} > 2.60. Company appears financially healthy. Low bankruptcy risk over the next 2 years."
    elif z_prime_prime > ALTMAN_DISTRESS_ZONE:
        zone = "Grey"
        interp = f"Z'' = {z_prime_prime:.2f} (1.10–2.60). Grey zone — elevated risk. Monitor closely. Will still PASS Stage 2 but flagged."
    else:
        zone = "Distress"
        interp = f"Z'' = {z_prime_prime:.2f} < 1.10. High financial distress signal. Company rejected at Stage 2."

    return {
        "model":   "Z'' (Altman 1995, Emerging Markets)",
        "z_score": round(z_prime_prime, 2),
        "zone":    zone,
        "passes":  zone != "Distress",
        "components": {
            "X1 (Working Capital / TA)":         round(x1, 3),
            "X2 (Retained Earnings / TA)":       round(x2, 3),
            "X3 (EBIT / TA)":                    round(x3, 3),
            "X4 (Book Equity / Total Liab)":     round(x4, 3),
        },
        "interpretation": interp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED HEALTH GATE (call this from fundamental_filter.py)
# ══════════════════════════════════════════════════════════════════════════════

def passes_health_gate(financials: dict, sector: str = "") -> tuple[bool, dict]:
    """
    Run both Piotroski and Altman Z'' checks. Stock must pass BOTH to proceed.

    For financial sector stocks, Altman is skipped (always passes that gate).
    Piotroski still applies to all sectors.

    Args:
        financials: dict with all financial data (see individual functions)
        sector: The stock's NSE sector string (used to skip Z'' for banks/NBFCs)

    Returns:
        (passes: bool, report: dict with both scores for display)
    """
    piotroski = compute_piotroski_fscore(financials)
    altman = compute_altman_zscore(financials, sector=sector)

    passes = piotroski["passes"] and altman["passes"]

    report = {
        "piotroski_score":       piotroski["score"],
        "piotroski_passes":      piotroski["passes"],
        "piotroski_breakdown":   piotroski["breakdown"],
        "altman_model":          altman["model"],
        "altman_zscore":         altman["z_score"],
        "altman_zone":           altman["zone"],
        "altman_passes":         altman["passes"],
        "altman_components":     altman["components"],
        "altman_interpretation": altman["interpretation"],
        "overall_health_pass":   passes,
    }

    return passes, report


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy financials to verify the formulas work
    dummy = {
        "net_income": 500, "net_income_prev": 400,
        "total_assets": 5000, "total_assets_prev": 4800,
        "operating_cash_flow": 600,
        "long_term_debt": 800, "long_term_debt_prev": 900,
        "current_assets": 1200, "current_assets_prev": 1100,
        "current_liabilities": 600, "current_liabilities_prev": 620,
        "shares_outstanding": 100, "shares_outstanding_prev": 100,
        "gross_profit": 1200, "gross_profit_prev": 1100,
        "revenue": 3000, "revenue_prev": 2800,
        "retained_earnings": 1500,
        "ebit": 700,
        "market_cap": 8000,
        "total_liabilities": 2000,
        "working_capital": 600,
    }

    # Test non-financial company (Z'' applies)
    passes, report = passes_health_gate(dummy, sector="IT")

    print("── Non-Financial Company (IT sector) ──")
    print(f"Piotroski F-Score: {report['piotroski_score']}/9 → {'PASS' if report['piotroski_passes'] else 'FAIL'}")
    print(f"Altman Model:      {report['altman_model']}")
    print(f"Altman Z'' Score:  {report['altman_zscore']} ({report['altman_zone']}) → {'PASS' if report['altman_passes'] else 'FAIL'}")
    print(f"Z'' Interpretation:{report['altman_interpretation']}")
    print(f"Overall:           {'✅ PASSES health gate' if passes else '❌ FAILS health gate'}")

    # Test financial company (Z'' skipped)
    print("\n── Financial Company (Banking sector) ──")
    passes_bank, report_bank = passes_health_gate(dummy, sector="Banking")
    print(f"Piotroski F-Score: {report_bank['piotroski_score']}/9 → {'PASS' if report_bank['piotroski_passes'] else 'FAIL'}")
    print(f"Altman:            {report_bank['altman_model']} — Z'' not applied")
    print(f"Overall:           {'✅ PASSES health gate' if passes_bank else '❌ FAILS health gate'}")
