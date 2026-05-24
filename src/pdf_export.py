# src/pdf_export.py
# PDF Tearsheet Generator
#
# Generates a professional 1-page PDF summary for any stock in the screener.
# In the Streamlit app, users click a "Download Tearsheet" button and get a
# clean PDF with all factor scores, fundamental data, and regime context.
#
# This is the "polish" feature that makes your project look like a real product.
# Uses fpdf2 (already in requirements.txt).
#
# We'll build this in Week 9.

from fpdf import FPDF
from datetime import date
import io


# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE (RGB tuples)
# ══════════════════════════════════════════════════════════════════════════════
COLOR_DARK    = (30, 30, 30)
COLOR_MUTED   = (100, 100, 100)
COLOR_BORDER  = (220, 220, 220)
COLOR_GREEN   = (29, 158, 117)    # Safe / positive
COLOR_AMBER   = (186, 117, 23)    # Neutral / warning
COLOR_RED     = (216, 90, 48)     # Danger / negative
COLOR_BLUE    = (55, 138, 221)    # Accent / headers
COLOR_BG_LIGHT = (248, 249, 250)  # Light section backgrounds


def _color_for_piotroski(score: int) -> tuple:
    if score >= 7: return COLOR_GREEN
    if score >= 5: return COLOR_AMBER
    return COLOR_RED

def _color_for_altman(zone: str) -> tuple:
    if zone == "Safe": return COLOR_GREEN
    if zone == "Grey": return COLOR_AMBER
    return COLOR_RED

def _color_for_zscore(z: float) -> tuple:
    if z >= 0.5:  return COLOR_GREEN
    if z >= -0.5: return COLOR_AMBER
    return COLOR_RED


# ══════════════════════════════════════════════════════════════════════════════
# TEARSHEET PDF CLASS
# ══════════════════════════════════════════════════════════════════════════════

class TearsheetPDF(FPDF):
    """Custom PDF class with a consistent header and footer."""

    def header(self):
        # Blue top bar
        self.set_fill_color(*COLOR_BLUE)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 2)
        self.cell(0, 8, "QUANTAMENTAL NIFTY 500 SCREENER - Stock Tearsheet", ln=0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*COLOR_MUTED)
        self.cell(0, 5,
            f"Generated {date.today().isoformat()} | For informational purposes only. Not investment advice.",
            align="C"
        )


def generate_tearsheet(stock_data: dict) -> bytes:
    """
    Generate a 1-page PDF tearsheet for a single stock.

    Args:
        stock_data: dict containing all the stock's information. Keys:
            ticker            (str)   e.g. "RELIANCE"
            company_name      (str)   e.g. "Reliance Industries Ltd"
            sector            (str)   e.g. "Energy"
            rank              (int)   e.g. 3
            regime            (str)   e.g. "Risk-On"
            composite_score   (float) e.g. 1.82
            value_score       (float) z-score
            growth_score      (float) z-score
            quality_score     (float) z-score
            momentum_score    (float) z-score
            surprise_score    (float) z-score
            piotroski_score   (int)   0-9
            altman_zone       (str)   "Safe", "Grey", or "Distress"
            altman_zscore     (float) e.g. 3.4
            current_price     (float) e.g. 2850.0
            pe_ratio          (float) e.g. 22.4
            roce              (float) % e.g. 18.5
            debt_equity       (float) e.g. 0.4
            revenue_cagr_3y   (float) % e.g. 12.3
            eps_cagr_3y       (float) % e.g. 15.1
            rsi               (float) e.g. 55.2
            ma_50             (float) e.g. 2780.0
            pct_from_52w_high (float) % e.g. 5.2

    Returns:
        PDF as bytes (use with Streamlit's st.download_button)
    """
    pdf = TearsheetPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # ── STOCK HEADER ─────────────────────────────────────────────────────────
    pdf.set_xy(10, 16)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*COLOR_DARK)
    pdf.cell(0, 10, stock_data.get("ticker", ""), ln=0)

    pdf.set_xy(10, 26)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*COLOR_MUTED)
    company = stock_data.get("company_name", "")
    sector  = stock_data.get("sector", "")
    pdf.cell(0, 6, f"{company}  |  {sector}", ln=1)

    # Rank badge
    pdf.set_xy(160, 16)
    pdf.set_fill_color(*COLOR_BLUE)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(40, 14, f"Rank  #{stock_data.get('rank', '-')}", border=0, fill=True, align="C")

    # Regime label
    regime = stock_data.get("regime", "Neutral")
    r_color = COLOR_GREEN if regime == "Risk-On" else (COLOR_AMBER if regime == "Neutral" else COLOR_RED)
    pdf.set_xy(160, 32)
    pdf.set_fill_color(*r_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(40, 8, f"Regime: {regime}", border=0, fill=True, align="C")

    # Divider line
    pdf.set_draw_color(*COLOR_BORDER)
    pdf.set_xy(10, 40)
    pdf.line(10, 42, 200, 42)

    # ── COMPOSITE SCORE + FINANCIAL HEALTH ───────────────────────────────────
    _section_header(pdf, "Composite Score & Financial Health", y=45)

    composite = stock_data.get("composite_score", 0)
    piotroski  = stock_data.get("piotroski_score", 0)
    altman_z   = stock_data.get("altman_zscore", 0)
    altman_zone = stock_data.get("altman_zone", "-")

    _metric_box(pdf, x=10,  y=53, w=58, label="Composite Score", value=f"{composite:.2f}", color=COLOR_BLUE)
    _metric_box(pdf, x=72,  y=53, w=58, label="Piotroski F-Score",
                value=f"{piotroski}/9", color=_color_for_piotroski(piotroski))
    _metric_box(pdf, x=134, y=53, w=58, label="Altman Z-Score",
                value=f"{altman_z:.1f} ({altman_zone})", color=_color_for_altman(altman_zone))

    # ── FACTOR SCORES ─────────────────────────────────────────────────────────
    _section_header(pdf, "Factor Scores (Z-Score vs Nifty 500 Universe)", y=72)

    factors = [
        ("Value",            stock_data.get("value_score", 0)),
        ("Growth",           stock_data.get("growth_score", 0)),
        ("Quality",          stock_data.get("quality_score", 0)),
        ("Momentum",         stock_data.get("momentum_score", 0)),
        ("Earnings Surprise",stock_data.get("surprise_score", 0)),
    ]

    x_start = 10
    box_w = 37
    gap = 2
    for i, (label, score) in enumerate(factors):
        _factor_bar(pdf, x=x_start + i*(box_w+gap), y=80, w=box_w,
                    label=label, score=score)

    # ── FUNDAMENTAL SNAPSHOT ─────────────────────────────────────────────────
    _section_header(pdf, "Fundamental Snapshot", y=110)

    fundamentals = [
        ("Current Price",   f"Rs.{stock_data.get('current_price', 0):,.0f}"),
        ("P/E Ratio",       f"{stock_data.get('pe_ratio', 0):.1f}x"),
        ("ROCE",            f"{stock_data.get('roce', 0):.1f}%"),
        ("Debt / Equity",   f"{stock_data.get('debt_equity', 0):.2f}x"),
        ("Revenue CAGR 3Y", f"{stock_data.get('revenue_cagr_3y', 0):+.1f}%"),
        ("EPS CAGR 3Y",     f"{stock_data.get('eps_cagr_3y', 0):+.1f}%"),
    ]

    _two_column_table(pdf, fundamentals, y=118)

    # ── TECHNICAL SIGNALS ────────────────────────────────────────────────────
    _section_header(pdf, "Technical Entry Signals (Green-Flag Check)", y=148)

    rsi = stock_data.get("rsi", 0)
    ma50 = stock_data.get("ma_50", 0)
    price = stock_data.get("current_price", 0)
    pct_52w = stock_data.get("pct_from_52w_high", 0)

    tech_checks = [
        ("RSI (14-day)",         f"{rsi:.1f}",  rsi <= 70),
        ("Price vs 50-day MA",   f"{'Above' if price >= ma50 else 'Below'} (MA: Rs.{ma50:,.0f})",
                                  price >= ma50),
        ("From 52-week High",    f"{pct_52w:.1f}% below",  pct_52w <= 20),
    ]

    _checklist_table(pdf, tech_checks, y=156)

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    pdf.set_xy(10, 270)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.multi_cell(190, 4,
        "This tearsheet is generated by an automated quantamental screener for educational and research purposes only. "
        "It does not constitute investment advice. Past factor scores do not guarantee future performance. "
        "Always conduct your own due diligence before making investment decisions.",
        align="L"
    )

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# HELPER DRAWING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _section_header(pdf, text: str, y: float):
    pdf.set_xy(10, y)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*COLOR_BLUE)
    pdf.cell(0, 5, text.upper(), ln=1)
    pdf.line(10, y + 5, 200, y + 5)


def _metric_box(pdf, x: float, y: float, w: float, label: str, value: str, color: tuple):
    """Draws a metric box: label on top, value below in color."""
    pdf.set_fill_color(*COLOR_BG_LIGHT)
    pdf.rect(x, y, w, 14, "F")
    pdf.set_xy(x + 2, y + 1)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.cell(w - 4, 4, label, ln=1)
    pdf.set_xy(x + 2, y + 6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*color)
    pdf.cell(w - 4, 7, value)


def _factor_bar(pdf, x: float, y: float, w: float, label: str, score: float):
    """Draws a single factor score with a mini progress bar."""
    # Label
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.cell(w, 4, label, align="C", ln=1)

    # Score value
    pdf.set_xy(x, y + 4)
    pdf.set_font("Helvetica", "B", 12)
    color = _color_for_zscore(score)
    pdf.set_text_color(*color)
    pdf.cell(w, 7, f"{score:+.2f}", align="C", ln=1)

    # Mini bar: normalize score from [-3, 3] to [0, w]
    bar_full_w = w - 4
    bar_x = x + 2
    bar_y = y + 13
    pdf.set_fill_color(*COLOR_BORDER)
    pdf.rect(bar_x, bar_y, bar_full_w, 3, "F")

    # Fill portion
    clipped = max(-3, min(3, score))
    if clipped >= 0:
        fill_w = (clipped / 3) * (bar_full_w / 2)
        fill_x = bar_x + bar_full_w / 2
    else:
        fill_w = (abs(clipped) / 3) * (bar_full_w / 2)
        fill_x = bar_x + bar_full_w / 2 - fill_w

    pdf.set_fill_color(*color)
    pdf.rect(fill_x, bar_y, fill_w, 3, "F")

    # Center line
    pdf.set_draw_color(*COLOR_DARK)
    pdf.line(bar_x + bar_full_w/2, bar_y, bar_x + bar_full_w/2, bar_y + 3)


def _two_column_table(pdf, rows: list, y: float):
    """Renders a 2-column key-value table."""
    col_w = 90
    row_h = 7
    for i, (label, value) in enumerate(rows):
        col = i % 2
        row = i // 2
        rx = 10 + col * col_w
        ry = y + row * row_h

        if i % 4 in [0, 1]:
            pdf.set_fill_color(*COLOR_BG_LIGHT)
            pdf.rect(rx, ry, col_w, row_h, "F")

        pdf.set_xy(rx + 2, ry + 1)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*COLOR_MUTED)
        pdf.cell(42, 5, label)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*COLOR_DARK)
        pdf.cell(col_w - 46, 5, value)


def _checklist_table(pdf, checks: list, y: float):
    """Renders a checklist table with pass/fail indicators."""
    for i, (label, value, passes) in enumerate(checks):
        ry = y + i * 8
        pdf.set_xy(10, ry)

        # Pass/Fail indicator
        icon_color = COLOR_GREEN if passes else COLOR_RED
        pdf.set_fill_color(*icon_color)
        pdf.rect(10, ry + 1, 4, 4, "F")

        pdf.set_xy(17, ry)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*COLOR_MUTED)
        pdf.cell(60, 6, label)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*COLOR_DARK)
        pdf.cell(60, 6, value)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*icon_color)
        pdf.cell(30, 6, "PASS" if passes else "FAIL")


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def add_download_button(stock_data: dict):
    """
    Add a Streamlit download button that generates and downloads the PDF.

    Call this from app.py for each stock in the ranked table:

        from src.pdf_export import add_download_button
        add_download_button(stock_row_dict)

    Args:
        stock_data: dict with all required fields (see generate_tearsheet)
    """
    import streamlit as st

    ticker = stock_data.get("ticker", "stock")
    pdf_bytes = generate_tearsheet(stock_data)

    st.download_button(
        label=f"📄 Download {ticker} Tearsheet",
        data=pdf_bytes,
        file_name=f"{ticker}_tearsheet_{date.today().isoformat()}.pdf",
        mime="application/pdf",
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "ticker": "RELIANCE", "company_name": "Reliance Industries Ltd",
        "sector": "Energy", "rank": 1, "regime": "Risk-On",
        "composite_score": 1.82,
        "value_score": 0.8, "growth_score": 1.5, "quality_score": 1.2,
        "momentum_score": 2.3, "surprise_score": 1.1,
        "piotroski_score": 7, "altman_zscore": 3.8, "altman_zone": "Safe",
        "current_price": 2850.0, "pe_ratio": 22.4,
        "roce": 18.5, "debt_equity": 0.4,
        "revenue_cagr_3y": 12.3, "eps_cagr_3y": 15.1,
        "rsi": 55.2, "ma_50": 2780.0, "pct_from_52w_high": 5.2,
    }

    pdf_bytes = generate_tearsheet(sample)

    with open("RELIANCE_tearsheet_test.pdf", "wb") as f:
        f.write(pdf_bytes)

    print(f"✅ Tearsheet generated: {len(pdf_bytes):,} bytes")
    print("   Saved as: RELIANCE_tearsheet_test.pdf")
