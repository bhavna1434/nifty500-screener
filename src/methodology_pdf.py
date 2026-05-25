# src/methodology_pdf.py
# Generates a one-page A4 methodology PDF for the Nifty 500 screener.
# Uses fpdf2 with Helvetica (Latin-1 only — no Unicode/emoji).

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Colour palette
BLUE_BG   = (232, 244, 253)   # #E8F4FD — title banner fill
BLUE_HDR  = (41, 98, 162)     # #2962A2 — section header text
DARK_TEXT = (30, 30, 30)
MID_GREY  = (120, 120, 120)
ROW_ALT   = (245, 249, 253)   # alternating table row tint

# Page geometry (A4 = 210 x 297 mm)
MARGIN     = 14.0
COL_W      = 210 - 2 * MARGIN   # 182 mm usable width
TITLE_H    = 30.0


def _set(pdf: FPDF, r, g, b, text=True):
    """Set draw/fill/text colour in one call."""
    if text:
        pdf.set_text_color(r, g, b)
    else:
        pdf.set_fill_color(r, g, b)
        pdf.set_draw_color(r, g, b)


def _section_header(pdf: FPDF, title: str):
    """Bold coloured section heading with a thin rule beneath."""
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    _set(pdf, *BLUE_HDR)
    pdf.cell(COL_W, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _set(pdf, *BLUE_HDR, text=False)
    pdf.set_line_width(0.3)
    y = pdf.get_y()
    pdf.line(MARGIN, y, MARGIN + COL_W, y)
    pdf.ln(2)
    _set(pdf, *DARK_TEXT)


def _body(pdf: FPDF, text: str, indent: float = 0, h: float = 5):
    pdf.set_font("Helvetica", "", 9)
    _set(pdf, *DARK_TEXT)
    pdf.set_x(MARGIN + indent)
    pdf.multi_cell(COL_W - indent, h, text)


def generate_methodology_pdf(output_path: str = "data/Nifty500_Methodology.pdf"):
    """
    Render a single-page A4 methodology document and save to output_path.
    All text is Latin-1 safe (no Unicode symbols, em dashes, or special chars).
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(MARGIN, 0, MARGIN)
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # ── Title banner ─────────────────────────────────────────────────────────
    pdf.set_fill_color(*BLUE_BG)
    pdf.set_draw_color(*BLUE_BG)
    pdf.rect(0, 0, 210, TITLE_H, "F")

    pdf.set_xy(MARGIN, 6)
    pdf.set_font("Helvetica", "B", 15)
    _set(pdf, *BLUE_HDR)
    pdf.cell(COL_W, 8, "Quantamental Nifty 500 Screener - Methodology",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

    pdf.set_x(MARGIN)
    pdf.set_font("Helvetica", "", 8)
    _set(pdf, *MID_GREY)
    pdf.cell(COL_W, 5, "Built by Bhavna Sharma  |  Data: Yahoo Finance + Screener.in",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(TITLE_H + 3)
    _set(pdf, *DARK_TEXT)

    # ── Section 1: Pipeline ───────────────────────────────────────────────────
    _section_header(pdf, "1.  The 5-Stage Pipeline")

    stages = [
        ("Stage 1", "Market Regime   -",
         "Nifty 50 vs 200-day MA + market breadth filter -> Risk-On / Neutral / Risk-Off"),
        ("Stage 2", "Red-Flag Filter -",
         "Piotroski F-Score >= 5, Altman Z'' >= 1.10, ROCE/D-E/Pledge/Market Cap/Liquidity gates"),
        ("Stage 3", "Factor Model    -",
         "Cross-sectional z-scores: Value, Growth, Quality, Momentum, EPS Momentum (equal-weighted)"),
        ("Stage 4", "Technical Filter-",
         "RSI < 70, price above 50-day MA, within 20% of 52-week high"),
        ("Stage 5", "Dashboard       -",
         "Interactive Streamlit app with factor sliders, history log, and PDF tearsheets"),
    ]

    for stage_id, label, desc in stages:
        pdf.set_font("Helvetica", "B", 9)
        _set(pdf, *DARK_TEXT)
        pdf.set_x(MARGIN + 2)
        pdf.cell(22, 5, stage_id, border=0)
        pdf.set_font("Helvetica", "B", 9)
        _set(pdf, *BLUE_HDR)
        pdf.cell(36, 5, label, border=0)
        pdf.set_font("Helvetica", "", 9)
        _set(pdf, *DARK_TEXT)
        pdf.multi_cell(COL_W - 60, 5, desc)

    # ── Section 2: Factor Definitions table ───────────────────────────────────
    _section_header(pdf, "2.  Factor Definitions")

    col_w = [28, 98, 56]   # Factor | Metric | Direction
    headers = ["Factor", "Metric", "Direction"]

    # Header row
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*BLUE_HDR)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 5.5, h, border=0, fill=True, align="C")
    pdf.ln()

    rows = [
        ("Value",        "z(-PE) + z(-EV/EBITDA) / 2",    "Lower is better"),
        ("Growth",       "z(Rev CAGR 3Y) + z(EPS CAGR 3Y) / 2", "Higher is better"),
        ("Quality",      "z(ROE) + z(ROCE) / 2",           "Higher is better"),
        ("Momentum",     "6-month price return, skip 1 month",    "Higher is better"),
        ("EPS Momentum", "Quarter-over-quarter EPS % change",     "Higher is better"),
    ]

    for i, (factor, metric, direction) in enumerate(rows):
        if i % 2 == 0:
            pdf.set_fill_color(*ROW_ALT)
        else:
            pdf.set_fill_color(255, 255, 255)
        _set(pdf, *DARK_TEXT)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_w[0], 5, factor, border=0, fill=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_w[1], 5, metric, border=0, fill=True)
        pdf.cell(col_w[2], 5, direction, border=0, fill=True)
        pdf.ln()

    # ── Section 3: Risk Scores ────────────────────────────────────────────────
    _section_header(pdf, "3.  Risk Scores")

    _body(pdf,
        "Piotroski F-Score (Piotroski 2000): 9 binary criteria derived from annual financial "
        "statements, covering profitability (4 signals), leverage/liquidity (3 signals), and "
        "operating efficiency (2 signals). Score >= 5 required to pass Stage 2.")
    pdf.ln(1)
    _body(pdf,
        "Altman Z''-Score (Altman 1995, emerging markets model): "
        "6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4, where X1=Working Capital/Total Assets, "
        "X2=Retained Earnings/Total Assets, X3=EBIT/Total Assets, X4=Book Equity/Total Liabilities. "
        "Score >= 1.10 required (above Distress zone).")

    # ── Section 4: Backtest Summary ───────────────────────────────────────────
    _section_header(pdf, "4.  Backtest Summary (price-momentum factor only, 2019-2024)")

    bt_lines = [
        "Universe: top 100 Nifty 500 stocks  |  Rebalancing: monthly  |  Signal: 6-month price momentum",
        "CAGR: 33.0%  vs  Benchmark (Nifty 500 TRI): 14.3%  |  Alpha: +16.7% p.a.  |  Beta: 1.05",
        "Sharpe Ratio: 1.27  |  Sortino Ratio: 2.10  |  Max Drawdown: -15.2%  |  Win Rate: 72% of months",
        "Transaction costs: 0.5% per rebalance included.  Survivorship bias present - returns overstated.",
        "Note: Single-factor (momentum) backtest only. Full 5-factor model not backtested. Not a trading signal.",
    ]
    for line in bt_lines:
        _body(pdf, line, indent=2)

    # ── Section 5: Disclaimer ─────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7.5)
    _set(pdf, *MID_GREY)
    pdf.set_x(MARGIN)
    pdf.multi_cell(
        COL_W, 4.5,
        "DISCLAIMER: For educational purposes only. Not investment advice. "
        "Past performance does not guarantee future results. "
        "Screener.in data may have a 1-3 month lag. "
        "This tool does not account for taxes, brokerage, or market impact costs.",
        align="C",
    )

    # ── Footer rule ───────────────────────────────────────────────────────────
    pdf.set_draw_color(*BLUE_HDR)
    pdf.set_line_width(0.4)
    pdf.line(MARGIN, 289, MARGIN + COL_W, 289)
    pdf.set_y(290)
    pdf.set_font("Helvetica", "", 7)
    _set(pdf, *MID_GREY)
    pdf.cell(COL_W / 2, 4, "Bhavna Sharma  -  bhavnasharma.1404@gmail.com", align="L")
    pdf.cell(COL_W / 2, 4, "github.com/bhavna1434/Quantamental-nifty500-screener", align="R")

    pdf.output(output_path)
    print(f"Methodology PDF saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_methodology_pdf()
    print("Done.")
