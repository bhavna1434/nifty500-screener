# src/fundamental_filter.py
# Stage 2: Red-Flag Filter — Fundamental Hard Gates
#
# Scrapes fundamental data from Screener.in and applies four hard filters.
# Stocks failing ANY filter are rejected before factor ranking begins.
#
# The four gates:
#   1. ROCE ≥ 10%          (sector-adjusted — see SECTOR_ROCE_FLOORS)
#   2. Debt/Equity ≤ 2.0x  (sector-adjusted)
#   3. Interest Coverage ≥ 1.5x
#   4. Promoter Pledge ≤ 20%
#
# Plus the Piotroski F-Score and Altman Z'' gate from financial_health.py
#
# Why Screener.in? It aggregates NSE/BSE financial data and exposes it as
# a clean webpage. It's free, widely used by Indian retail investors, and
# has consistent HTML structure across all companies.
#
# Limitation: ~1–3 month data lag. Not suitable for real-time trading.
# For production use, replace with Trendlyne Pro or Bloomberg terminal.

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from src.financial_health import passes_health_gate

# ── Request settings ──────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5    # seconds between requests — be polite to Screener.in
REQUEST_TIMEOUT = 15   # seconds

# ── Sector-adjusted thresholds ────────────────────────────────────────────────
# See 03_LIMITATIONS_AND_CALIBRATION.md Section 4.1 for full rationale.
# Capital-intensive sectors (infra, utilities) legitimately carry more debt
# and earn lower ROCE — a blanket threshold would unfairly exclude them.

SECTOR_ROCE_FLOORS = {
    "Technology":          25.0,
    "IT":                  25.0,
    "FMCG":                20.0,
    "Consumer Goods":      20.0,
    "Pharmaceuticals":     18.0,
    "Healthcare":          18.0,
    "Automobile":          12.0,
    "Manufacturing":       10.0,
    "Metals & Mining":     10.0,
    "Chemicals":           12.0,
    "Infrastructure":       8.0,
    "Power":                8.0,
    "Utilities":            8.0,
    "Telecom":              8.0,
    "Default":             10.0,   # fallback for unrecognised sectors
}

SECTOR_DE_CEILINGS = {
    "Infrastructure":  3.0,
    "Power":           3.0,
    "Utilities":       3.0,
    "Real Estate":     3.0,
    "Default":         2.0,
}

# Financial sector — use different health checks (not Altman Z'')
FINANCIAL_SECTORS = {"Banking", "Finance", "NBFCs", "Insurance", "Financial Services"}


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER.IN SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fundamentals(ticker: str) -> dict | None:
    """
    Scrape fundamental metrics for one stock from Screener.in.

    URL pattern: https://www.screener.in/company/{TICKER}/

    Data extracted:
        - ROCE (%)
        - ROE (%)
        - Debt/Equity ratio
        - Interest Coverage ratio
        - Promoter pledge (%)
        - Sector / industry label
        - P/E ratio (for factor model)
        - EV/EBITDA (for factor model)
        - Revenue CAGR 3Y (%)
        - EPS CAGR 3Y (%)
        - Net Income, Total Assets, EBIT, etc. (for Piotroski/Altman)

    Args:
        ticker: NSE symbol WITHOUT .NS (e.g. "RELIANCE")

    Returns:
        dict of fundamental data, or None if the fetch/parse fails
    """
    url = f"https://www.screener.in/company/{ticker.upper()}/"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            print(f"  {ticker}: Not found on Screener.in (404). Skipping.")
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  {ticker}: Request failed — {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    data = {"ticker": ticker}

    # ── 1. Top Ratios (the summary metrics block) ─────────────────────────────
    # Screener.in shows these as <li> items in the #top section
    top_section = soup.find("section", {"id": "top-ratios"})
    if not top_section:
        top_section = soup.find("ul", {"id": "top-ratios"})

    if top_section:
        for item in top_section.find_all("li"):
            label_tag = item.find("span", {"class": "name"})
            value_tag = item.find("span", {"class": "number"})
            if not label_tag or not value_tag:
                continue
            label = label_tag.get_text(strip=True).lower()
            raw_val = value_tag.get_text(strip=True).replace(",", "").replace("%", "").strip()

            val = _safe_float(raw_val)
            if val is None:
                continue

            if "stock p/e" in label or "p/e" in label:
                data["pe_ratio"] = val
            elif "market cap" in label:
                data["market_cap"] = val        # crores
            elif "roce" in label:
                data["roce"] = val
            elif "roe" in label:
                data["roe"] = val
            elif "book value" in label:
                data["book_value_per_share"] = val

    # ── 2. Company description / sector ──────────────────────────────────────
    about = soup.find("div", {"class": "company-info"})
    if about:
        sector_tag = about.find("a", {"class": "breadcrumb-item"})
        if sector_tag:
            data["sector"] = sector_tag.get_text(strip=True)

    # ── 3. Financial ratios table ─────────────────────────────────────────────
    # Screener.in has a "Ratios" section with Debt/Equity, Interest Coverage, etc.
    ratios_section = soup.find("section", {"id": "ratios"})
    if ratios_section:
        rows = ratios_section.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            label = cols[0].get_text(strip=True).lower()
            # Latest value is the last <td>
            raw_val = cols[-1].get_text(strip=True).replace(",", "").replace("%", "").strip()
            val = _safe_float(raw_val)
            if val is None:
                continue

            if "debtor days" in label:
                pass  # skip
            elif any(x in label for x in ("debt / equity", "debt to equity", "d/e ratio", "d/e")):
                data["debt_equity"] = val
            elif "interest coverage" in label or "int. coverage" in label:
                data["interest_coverage"] = val
            elif "ev/ebitda" in label:
                data["ev_ebitda"] = val

    # ── 4. Shareholding / promoter pledge ────────────────────────────────────
    # Screener.in shows pledging in the shareholding section
    shareholding = soup.find("section", {"id": "shareholding"})
    if shareholding:
        pledge_text = shareholding.find(
            text=lambda t: t and "pledge" in t.lower()
        )
        if pledge_text:
            parent = pledge_text.find_parent()
            if parent:
                next_val = parent.find_next_sibling()
                if next_val:
                    val = _safe_float(
                        next_val.get_text(strip=True).replace("%", "").replace(",", "")
                    )
                    if val is not None:
                        data["promoter_pledge"] = val

        # Promoter holding %
        for row in shareholding.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 2:
                label = cols[0].get_text(strip=True).lower()
                if "promoter" in label and "pledge" not in label:
                    val = _safe_float(cols[-1].get_text(strip=True).replace("%", ""))
                    if val is not None:
                        data["promoter_holding"] = val

    # ── 5. P&L data for CAGR computation ─────────────────────────────────────
    # Screener.in P&L labels (after stripping trailing '+'):
    #   Sales, Expenses, Operating Profit, OPM %, Other Income,
    #   Interest, Depreciation, Profit before tax, Tax %, Net Profit, EPS in Rs
    pl_section = soup.find("section", {"id": "profit-loss"})
    if pl_section:
        table = pl_section.find("table")
        if table:
            pl_data = _parse_screener_table(table)

            # Revenue CAGR (3-year): compare latest vs 3 years ago
            if "Sales" in pl_data and len(pl_data["Sales"]) >= 4:
                rev_latest = pl_data["Sales"][-1]
                rev_3y_ago = pl_data["Sales"][-4]
                if rev_3y_ago and rev_latest and rev_3y_ago > 0:
                    data["revenue_cagr_3y"] = ((rev_latest / rev_3y_ago) ** (1/3) - 1) * 100

            # EPS CAGR (3-year)
            if "EPS in Rs" in pl_data and len(pl_data["EPS in Rs"]) >= 4:
                eps_latest = pl_data["EPS in Rs"][-1]
                eps_3y_ago = pl_data["EPS in Rs"][-4]
                if eps_3y_ago and eps_latest and eps_3y_ago > 0:
                    data["eps_cagr_3y"] = ((eps_latest / eps_3y_ago) ** (1/3) - 1) * 100

            # Net income for Piotroski A1/A3/A4
            if "Net Profit" in pl_data and pl_data["Net Profit"]:
                data["net_income"]      = pl_data["Net Profit"][-1]
                data["net_income_prev"] = pl_data["Net Profit"][-2] if len(pl_data["Net Profit"]) > 1 else None

            # Revenue + gross profit proxy (Operating Profit) for Piotroski C8/C9
            if "Sales" in pl_data and pl_data["Sales"]:
                data["revenue"]      = pl_data["Sales"][-1]
                data["revenue_prev"] = pl_data["Sales"][-2] if len(pl_data["Sales"]) > 1 else None
            if "Operating Profit" in pl_data and pl_data["Operating Profit"]:
                data["gross_profit"]      = pl_data["Operating Profit"][-1]
                data["gross_profit_prev"] = pl_data["Operating Profit"][-2] if len(pl_data["Operating Profit"]) > 1 else None

            # EBIT for Altman Z'' (Operating Profit - Depreciation)
            op_profit    = (pl_data.get("Operating Profit") or [None])[-1]
            depreciation = (pl_data.get("Depreciation") or [None])[-1]
            if op_profit is not None and depreciation is not None:
                data["ebit"] = op_profit - depreciation

            # Shares outstanding proxy: Net Profit / EPS (for Piotroski B7 dilution check)
            eps_vals = pl_data.get("EPS in Rs", [])
            np_vals  = pl_data.get("Net Profit", [])
            if eps_vals and np_vals:
                eps_now  = eps_vals[-1]
                eps_prev = eps_vals[-2] if len(eps_vals) > 1 else None
                ni_now   = np_vals[-1]
                ni_prev  = np_vals[-2] if len(np_vals) > 1 else None
                if eps_now and ni_now and eps_now != 0:
                    data["shares_outstanding"]      = (ni_now * 1e7) / eps_now
                if eps_prev and ni_prev and eps_prev != 0:
                    data["shares_outstanding_prev"] = (ni_prev * 1e7) / eps_prev

    # ── 6. Balance sheet data for Piotroski / Altman Z'' ─────────────────────
    # Screener.in balance sheet labels (actual):
    #   'Equity Capital', 'Reserves', 'Borrowings+', 'Other Liabilities+',
    #   'Total Liabilities', 'Fixed Assets+', 'CWIP', 'Investments',
    #   'Other Assets+', 'Total Assets'
    # No 'Current Assets' / 'Current Liabilities' in this condensed view.
    bs_section = soup.find("section", {"id": "balance-sheet"})
    if bs_section:
        table = bs_section.find("table")
        if table:
            bs_data = _parse_screener_table(table)

            def _latest(key, default=None):
                # Match exact key or key with trailing '+'
                vals = bs_data.get(key) or bs_data.get(key + "+", [])
                return vals[-1] if vals else default

            def _prev(key, default=None):
                vals = bs_data.get(key) or bs_data.get(key + "+", [])
                return vals[-2] if len(vals) >= 2 else default

            data["total_assets"]           = _latest("Total Assets")
            data["total_assets_prev"]      = _prev("Total Assets")
            data["total_liabilities"]      = _latest("Total Liabilities")
            data["long_term_debt"]         = _latest("Borrowings")
            data["long_term_debt_prev"]    = _prev("Borrowings")
            data["retained_earnings"]      = _latest("Reserves")

            # Shareholders equity = Equity Capital + Reserves
            eq_cap  = _latest("Equity Capital") or 0
            eq_cap_prev = _prev("Equity Capital") or 0
            reserves = _latest("Reserves") or 0
            reserves_prev = _prev("Reserves") or 0
            data["book_value_equity"]      = eq_cap + reserves if (eq_cap or reserves) else None
            data["book_value_equity_prev"] = eq_cap_prev + reserves_prev if (eq_cap_prev or reserves_prev) else None

            # Current assets/liabilities not in condensed view — leave as 0
            # so Piotroski B6 (liquidity) scores 0 rather than crashing
            data["current_assets"]         = None
            data["current_assets_prev"]    = None
            data["current_liabilities"]    = None
            data["current_liabilities_prev"] = None
            data["working_capital"]        = None

    # ── 7. Cash flow data for Piotroski ──────────────────────────────────────
    cf_section = soup.find("section", {"id": "cash-flow"})
    if cf_section:
        table = cf_section.find("table")
        if table:
            cf_data = _parse_screener_table(table)
            # Actual label: "Cash from Operating Activity" (trailing + stripped)
            cf_key = next((k for k in cf_data if "operating activity" in k.lower() or k == "Cash from Operations"), None)
            if cf_key:
                vals = cf_data[cf_key]
                data["operating_cash_flow"] = vals[-1] if vals else None

    # ── 8. Fallback: compute D/E from balance sheet if not scraped directly ──
    if not data.get("debt_equity"):
        debt  = data.get("long_term_debt")   or 0
        equity = data.get("book_value_equity")
        if equity and equity > 0:
            data["debt_equity"] = round(debt / equity, 2)

    # ── 9. Compute EV/EBITDA from scraped components ─────────────────────────
    # EV  = Market Cap + Borrowings  (simplified: ignores cash)
    # EBITDA = Operating Profit (gross_profit field, which = Sales - Expenses)
    market_cap   = data.get("market_cap")       # crores
    borrowings   = data.get("long_term_debt")   # crores (from balance sheet)
    ebitda       = data.get("gross_profit")     # crores (Operating Profit from P&L)
    if market_cap and ebitda and ebitda > 0:
        ev = market_cap + (borrowings or 0)
        data["ev_ebitda"] = round(ev / ebitda, 2)

    return data if len(data) > 2 else None


def _safe_float(text: str) -> float | None:
    """Convert a string to float, returning None on failure."""
    try:
        cleaned = str(text).replace(",", "").replace("%", "").strip()
        if cleaned in ("", "-", "—", "N/A", "na", "nan"):
            return None
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_screener_table(table) -> dict:
    """
    Parse a Screener.in financial table into a dict of {row_label: [values]}.
    Trailing '+' is stripped from labels (Screener.in uses it as an expand marker).
    """
    result = {}
    rows = table.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        label = cols[0].get_text(strip=True).rstrip("+").strip()
        values = []
        for col in cols[1:]:
            val = _safe_float(col.get_text(strip=True).replace(",", ""))
            values.append(val)
        if label and values:
            result[label] = values
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RED-FLAG FILTER — apply to full universe
# ══════════════════════════════════════════════════════════════════════════════

def apply_red_flag_filter(
    universe: list,
    sector_map: dict = None,
    delay: float = REQUEST_DELAY,
    verbose: bool = True,
) -> tuple:
    """
    Apply all Stage 2 Red-Flag filters to the full Nifty 500 universe.

    Filters applied (in order):
      1. Data availability check (skip if can't fetch data)
      2. ROCE ≥ sector floor
      3. Debt/Equity ≤ sector ceiling
      4. Interest Coverage ≥ 1.5x
      5. Promoter Pledge ≤ 20%
      6. Piotroski F-Score ≥ 5
      7. Altman Z'' not in Distress zone

    Args:
        universe:   List of NSE ticker symbols (Nifty 500 constituent list)
        sector_map: Optional dict mapping ticker → sector string.
                    If provided, sector-adjusted thresholds are applied.
        delay:      Seconds to wait between Screener.in requests
        verbose:    If True, prints pass/fail for each stock

    Returns:
        (passing: list, rejected_log: pd.DataFrame)
        passing:      List of tickers that passed all filters
        rejected_log: DataFrame showing why each stock was rejected
    """
    passing = []
    rejected_log = []
    fundamentals_store = []   # cache for use in factor model

    total = len(universe)
    for i, ticker in enumerate(universe, 1):
        if verbose:
            print(f"[{i:3d}/{total}] {ticker:<15}", end="")

        sector = (sector_map or {}).get(ticker, "Default")

        # ── Fetch data ────────────────────────────────────────────────────────
        data = fetch_fundamentals(ticker)
        time.sleep(delay)

        if data is None:
            if verbose:
                print("SKIP — data unavailable")
            rejected_log.append({"ticker": ticker, "reason": "data_unavailable"})
            continue

        # ── Gate 1: ROCE ──────────────────────────────────────────────────────
        roce_floor = SECTOR_ROCE_FLOORS.get(sector, SECTOR_ROCE_FLOORS["Default"])
        roce = data.get("roce")
        if roce is None or roce < roce_floor:
            if verbose:
                print(f"FAIL — ROCE {roce}% < {roce_floor}% floor for {sector}")
            rejected_log.append({"ticker": ticker, "reason": f"roce_{roce}_below_{roce_floor}"})
            continue

        # ── Gate 2: Debt/Equity ───────────────────────────────────────────────
        de_ceiling = SECTOR_DE_CEILINGS.get(sector, SECTOR_DE_CEILINGS["Default"])
        de = data.get("debt_equity", 0)
        if de is not None and de > de_ceiling:
            if verbose:
                print(f"FAIL — D/E {de:.1f}x > {de_ceiling}x ceiling for {sector}")
            rejected_log.append({"ticker": ticker, "reason": f"de_{de:.1f}_above_{de_ceiling}"})
            continue

        # ── Gate 3: Interest Coverage ─────────────────────────────────────────
        icr = data.get("interest_coverage")
        if icr is not None and icr < 1.5:
            if verbose:
                print(f"FAIL — Interest Coverage {icr:.1f}x < 1.5x")
            rejected_log.append({"ticker": ticker, "reason": f"icr_{icr:.1f}_below_1.5"})
            continue

        # ── Gate 4: Promoter Pledge ───────────────────────────────────────────
        pledge = data.get("promoter_pledge", 0)
        if pledge is not None and pledge > 20.0:
            if verbose:
                print(f"FAIL — Promoter pledge {pledge:.1f}% > 20%")
            rejected_log.append({"ticker": ticker, "reason": f"pledge_{pledge:.0f}pct"})
            continue

        # ── Gate 5+6: Piotroski F-Score + Altman Z'' ─────────────────────────
        if sector not in FINANCIAL_SECTORS:
            health_pass, health_report = passes_health_gate(data, sector=sector)
            if not health_pass:
                reason = []
                if not health_report["piotroski_passes"]:
                    reason.append(f"piotroski_{health_report['piotroski_score']}")
                if not health_report["altman_passes"]:
                    reason.append(f"altman_{health_report['altman_zone']}")
                if verbose:
                    print(f"FAIL — {', '.join(reason)}")
                rejected_log.append({"ticker": ticker, "reason": " + ".join(reason)})
                continue

            data["piotroski_score"] = health_report["piotroski_score"]
            data["altman_zscore"]   = health_report["altman_zscore"]
            data["altman_zone"]     = health_report["altman_zone"]

        # ── PASSED all gates ──────────────────────────────────────────────────
        passing.append(ticker)
        fundamentals_store.append(data)
        if verbose:
            print(f"PASS  (ROCE:{roce:.0f}% D/E:{de if de else 'N/A'} "
                  f"ICR:{icr if icr else 'N/A'} Pledge:{pledge if pledge else 0:.0f}%)")

    if verbose:
        print(f"\n{'─'*50}")
        print(f"Universe:  {total} stocks")
        print(f"Passing:   {len(passing)} stocks ({len(passing)/total*100:.1f}%)")
        print(f"Rejected:  {total - len(passing)} stocks")

    rejected_df = pd.DataFrame(rejected_log) if rejected_log else pd.DataFrame()
    fundamentals_df = pd.DataFrame(fundamentals_store) if fundamentals_store else pd.DataFrame()

    return passing, rejected_df, fundamentals_df


# ── Quick test on 5 stocks ────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickers = ["RELIANCE", "TCS", "YESBANK", "INFY", "TATASTEEL"]
    print(f"Testing fundamental_filter.py on: {test_tickers}\n")

    passing, rejected, fundamentals = apply_red_flag_filter(
        test_tickers, verbose=True
    )

    print(f"\nPassing: {passing}")
    if not rejected.empty:
        print(f"\nRejection log:\n{rejected.to_string(index=False)}")
