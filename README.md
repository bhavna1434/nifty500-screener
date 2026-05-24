# Quantamental Nifty 500 Screener

**A 5-stage QVGS-style stock screener for the Indian market**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit)](https://bhavna-nifty500-screener.streamlit.app)

---

## What This Is

A quantamental stock screener that combines systematic fundamental analysis — Piotroski F-Score, Altman Z''-Score, and a 5-factor cross-sectional model — with technical timing signals to filter the full Nifty 500 universe down to a ranked shortlist of actionable ideas. The pipeline follows the Red-Flag → Yellow-Flag → Green-Flag methodology used by practitioners such as Modulor Capital: first eliminate structurally weak businesses, then rank survivors by factor quality, then confirm with price-based entry signals. Built as a portfolio project targeting quantitative equity research roles, May–July 2026.

---

## The 5-Stage Pipeline

```
Nifty 500 (504 stocks)
      │
      ▼
Stage 1 · Market Regime Detection
         200-day MA on Nifty 50 + market breadth %
         → Risk-On / Neutral / Risk-Off classification
      │
      ▼
Stage 2 · Red-Flag Filter
         Piotroski F-Score ≥ 5
         Altman Z''-Score ≥ 1.10 (not in Distress zone)
         ROCE ≥ sector floor · D/E ≤ sector ceiling
         Interest Coverage ≥ 1.5x · Promoter Pledge ≤ 20%
         Market Cap ≥ ₹500 Cr · Avg Daily Liquidity ≥ ₹5 Cr
      │
      ▼
Stage 3 · Factor Model  (cross-sectional z-scores, equal-weighted)
         Value     — P/E + EV/EBITDA (lower is better)
         Growth    — 3Y Revenue CAGR + 3Y EPS CAGR
         Quality   — ROE + ROCE
         Momentum  — 6M price return, skipping most recent month
         EPS Mom   — QoQ EPS acceleration (recency-decayed)
      │
      ▼
Stage 4 · Technical Filter (Green-Flag entry signals)
         RSI < 70  ·  Price above 50-day MA
         Within 20% of 52-week high
      │
      ▼
Stage 5 · Streamlit Dashboard
         Ranked table with factor sliders
         Sector concentration analysis
         Factor attribution + correlation heatmap
         Week-over-week history log (SQLite + CSV persistence)
         Per-stock PDF tearsheets
         Momentum backtest vs Nifty 500
```

---

## Academic Foundation

| Paper | Factor | Key Finding |
|---|---|---|
| Piotroski (2000) | Quality | 9-point F-Score separates improving from deteriorating value stocks; long-short earns ~23% annually |
| Altman (1968, 1995) | Quality | Z-Score and Z''-Score predict bankruptcy 1–2 years ahead with >80% accuracy |
| Jegadeesh & Titman (1993) | Momentum | Stocks with strong 3–12M returns continue to outperform for the next 3–12M |
| Fama & French (1992) | Value / Size | Book-to-market ratio and market cap explain cross-sectional return variation beyond beta |
| Ball & Brown (1968) | EPS Momentum | Earnings surprises drive post-announcement price drift (PEAD) for 30–60 days |

---

## Data Sources

- **Price data:** Yahoo Finance via `yfinance` — NSE tickers with `.NS` suffix; bulk downloads for speed
- **Fundamentals:** Screener.in — scraped with `requests` + `BeautifulSoup4`; results cached locally for 7 days to avoid repeated scraping

---

## How to Run Locally

```bash
git clone https://github.com/bhavna1434/nifty500-screener
cd nifty500-screener
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** The first run scrapes fundamentals for all 500 stocks from Screener.in (~10–15 minutes, ~1.5s delay per request to be polite). Subsequent runs within 7 days load from `data/fundamentals_cache.csv` and complete in under a minute.

---

## Known Limitations

- **Scraper fragility:** The Screener.in scraper will break if the site changes its HTML structure. It is not production-grade — for a live system, replace with Trendlyne Pro or a Bloomberg terminal feed.
- **EPS momentum ≠ PEAD:** The 5th factor measures quarter-over-quarter EPS change, not surprise vs analyst consensus. True PEAD requires paid analyst estimates (Bloomberg, Refinitiv). This is EPS acceleration, not the classical Ball & Brown anomaly.
- **Backtest is price-momentum only:** Historical fundamental data (Piotroski, Altman scores by year) is not freely available. The backtest is a single-factor momentum simulation and should not be read as a forecast of the full model's live performance.
- **Survivorship bias:** The Nifty 500 constituent list used is current. Stocks delisted or dropped from the index between 2019–2024 are absent from the backtest universe, which mechanically inflates simulated returns.

---

## Built By

Bhavna Sharma · bhavnasharma.1404@gmail.com · [GitHub](https://github.com/bhavna1434)

Built as a quantitative finance portfolio project, May–July 2026.
