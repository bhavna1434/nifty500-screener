# app.py — Quantamental Nifty 500 Screener
# Main Streamlit application
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="Nifty 500 Screener",
    page_icon="📈",
    layout="wide"
)

from src.data_loader import load_nifty500_list
from src.regime_detector import get_current_regime
from src.fundamental_filter import apply_red_flag_filter
from src.factor_model import rank_stocks
from src.technical_filter import apply_green_flag_filter


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Screener Settings")
    st.caption("Adjust factor weights to create your own custom tilt")

    st.subheader("Factor Weights")
    st.caption("Each slider sets the weight for that factor. They are normalised automatically.")

    w_value    = st.slider("📊 Value (P/E, EV/EBITDA)",  min_value=0, max_value=100, value=20, step=5)
    w_growth   = st.slider("📈 Growth (Rev + EPS CAGR)", min_value=0, max_value=100, value=20, step=5)
    w_quality  = st.slider("🏆 Quality (ROE, ROCE)",     min_value=0, max_value=100, value=20, step=5)
    w_momentum = st.slider("⚡ Momentum (6M return)",    min_value=0, max_value=100, value=20, step=5)
    w_surprise = st.slider("📣 Earnings Surprise (PEAD)",min_value=0, max_value=100, value=20, step=5)

    total = w_value + w_growth + w_quality + w_momentum + w_surprise
    if total == 0:
        st.error("At least one factor must have a non-zero weight.")
        weights = None
    else:
        weights = {
            "value":    w_value    / total,
            "growth":   w_growth   / total,
            "quality":  w_quality  / total,
            "momentum": w_momentum / total,
            "surprise": w_surprise / total,
        }
        st.caption("Effective weights (sum = 100%):")
        for name, w in weights.items():
            st.caption(f"  {name.capitalize()}: {w*100:.1f}%")

    st.divider()
    st.subheader("Filters")
    top_n            = st.number_input("Show top N stocks",        min_value=5, max_value=50, value=20, step=5)
    piotroski_min    = st.slider("Piotroski F-Score minimum",      min_value=0, max_value=9,  value=5)
    exclude_distress = st.checkbox("Exclude Altman Distress zone stocks", value=True)

    st.divider()

    run_clicked = st.button("▶ Run Screener", type="primary", use_container_width=True)
    st.caption("Scans first 50 tickers · ~2–3 min")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — runs when button is clicked
# ══════════════════════════════════════════════════════════════════════════════

if run_clicked and weights:
    with st.spinner("Running screener pipeline…"):

        # Stage 0 — load tickers
        with st.spinner("Loading Nifty 500 ticker list…"):
            all_tickers = load_nifty500_list()
            universe_50 = all_tickers[:50]

        # Stage 1 — download price data
        with st.spinner(f"Downloading 1y price data for {len(universe_50)} stocks…"):
            tickers_ns = [t + ".NS" for t in universe_50]
            raw = yf.download(tickers_ns, period="1y", progress=False, auto_adjust=True)["Close"]
            raw.columns = [c.replace(".NS", "") for c in raw.columns]

        # Stage 2 — fundamental red-flag filter
        with st.spinner("Stage 2: Running fundamental red-flag filter (Screener.in)…"):
            passing, rejected_df, fundamentals_df = apply_red_flag_filter(
                universe_50, verbose=False
            )

        # Stage 3 — factor model ranking
        if passing:
            with st.spinner(f"Stage 3: Ranking {len(passing)} stocks by factor model…"):
                ranked_df = rank_stocks(
                    universe=passing,
                    price_df=raw,
                    fundamentals_df=fundamentals_df,
                    weights=weights,
                )
            top20 = ranked_df["ticker"].head(20).tolist()
        else:
            ranked_df = pd.DataFrame()
            top20 = []

        # Stage 4 — technical green-flag filter
        if top20:
            with st.spinner("Stage 4: Applying technical green-flag filter…"):
                tech_df = apply_green_flag_filter(top20, raw)
            final_picks = tech_df[tech_df["passes"]]["ticker"].tolist()
        else:
            tech_df = pd.DataFrame()
            final_picks = []

        # Merge technical signals into ranked_df for display
        if not ranked_df.empty and not tech_df.empty:
            ranked_df = ranked_df.merge(
                tech_df[["ticker", "rsi", "above_ma50", "pct_from_52w_high", "passes"]],
                on="ticker", how="left"
            )

        # Persist in session state
        st.session_state["ranked_df"]      = ranked_df
        st.session_state["tech_df"]        = tech_df
        st.session_state["final_picks"]    = final_picks
        st.session_state["passing_count"]  = len(passing)
        st.session_state["universe_count"] = len(universe_50)

    st.success(f"Done! {len(final_picks)} stocks passed all 4 stages.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.title("📈 Quantamental Nifty 500 Screener")
st.caption("QVGS-style pipeline: Red-Flag filters → 5-Factor ranking → Technical timing")

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    try:
        regime_data  = get_current_regime()
        regime       = regime_data["regime"]
    except Exception:
        regime = "Neutral"
    regime_emoji = {"Risk-On": "🟢", "Neutral": "🟡", "Risk-Off": "🔴"}.get(regime, "🟡")
    st.metric("Market Regime", f"{regime_emoji} {regime}")

with col2:
    universe_count = st.session_state.get("universe_count", 500)
    st.metric("Universe Scanned", str(universe_count), help="Stocks passed into the pipeline")

with col3:
    passing_count = st.session_state.get("passing_count", None)
    st.metric(
        "After Red-Flag Filter",
        str(passing_count) if passing_count is not None else "—",
        help="Passing Piotroski + Altman + ROCE/Debt/Pledge"
    )

with col4:
    final_picks = st.session_state.get("final_picks", None)
    st.metric(
        "Final Top Picks",
        str(len(final_picks)) if final_picks is not None else "—",
        help=f"Top {top_n} after all 4 stages"
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Ranked Stocks",
    "📊 Factor Analysis",
    "📅 History & Changes",
    "🔎 Stock Deep-Dive",
])


# ── TAB 1: Ranked Stocks ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Top Ranked Stocks")
    st.caption("Sorted by composite factor score. Green rows passed the technical filter.")

    ranked_df = st.session_state.get("ranked_df", None)

    if ranked_df is None or ranked_df.empty:
        st.info("Click **▶ Run Screener** in the sidebar to load real data.")
    else:
        display_cols = ["rank", "ticker", "composite_score",
                        "value_score", "growth_score", "quality_score",
                        "momentum_score", "surprise_score"]
        if "rsi" in ranked_df.columns:
            display_cols += ["rsi", "above_ma50", "pct_from_52w_high", "passes"]

        display_df = ranked_df[display_cols].head(int(top_n)).copy()
        display_df.columns = [c.replace("_score", "").replace("_", " ").title()
                               for c in display_df.columns]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Composite": st.column_config.ProgressColumn(
                    "Composite Score", min_value=-3, max_value=3, format="%.2f"
                ),
                "Passes": st.column_config.CheckboxColumn("Tech ✅"),
            }
        )
        st.caption(f"Showing top {min(int(top_n), len(ranked_df))} of {len(ranked_df)} ranked stocks.")

    st.divider()
    st.subheader("Download Tearsheets")
    st.caption("One-page PDF summary per stock — available after Week 9 build.")
    st.info("🚧 PDF tearsheets coming in Week 9.")


# ── TAB 2: Factor Analysis ────────────────────────────────────────────────────
with tab2:
    st.subheader("Factor Attribution")
    st.info("🚧 Factor attribution chart — build in Week 8 (visualizations.py).")
    st.divider()
    st.subheader("Return Correlation Matrix")
    st.info("🚧 Correlation heatmap — build in Week 7 (visualizations.py).")


# ── TAB 3: History & Changes ──────────────────────────────────────────────────
with tab3:
    st.subheader("Week-over-Week Changes")
    st.info("🚧 Screener history — build in Week 8 (history_tracker.py).")


# ── TAB 4: Stock Deep-Dive ────────────────────────────────────────────────────
with tab4:
    st.subheader("Single Stock Analysis")
    search_ticker = st.text_input("Enter NSE ticker (e.g. RELIANCE, TCS, INFY)", value="RELIANCE")
    if search_ticker:
        col_l, col_r = st.columns(2)
        with col_l:
            st.info("🚧 Factor radar chart — build in Week 9 (visualizations.py).")
        with col_r:
            st.info("🚧 Rank history chart — build in Week 9 (history_tracker.py).")
        st.caption(f"Showing data for: **{search_ticker.upper()}**")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Bhavna Sharma · "
    "[GitHub](https://github.com/bhavna1434/nifty500-screener) · "
    "Data: Yahoo Finance + Screener.in · For educational purposes only."
)
