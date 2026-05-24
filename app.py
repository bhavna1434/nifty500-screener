# app.py — Quantamental Nifty 500 Screener
# Main Streamlit application
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(
    page_title="Nifty 500 Screener",
    page_icon="📈",
    layout="wide"
)

from src.data_loader import load_nifty500_list
from src.regime_detector import get_current_regime
from src.fundamental_filter import apply_red_flag_filter, fetch_fundamentals
from src.factor_model import rank_stocks
from src.technical_filter import apply_green_flag_filter
from src.history_tracker import (
    init_database, save_run, render_history_section, get_stock_history,
    import_history_from_csv, export_history_to_csv,
)
from src.pdf_export import generate_tearsheet
from src.visualizations import (
    plot_correlation_heatmap,
    plot_factor_attribution,
    plot_regime_gauge,
    plot_factor_radar,
    plot_rank_history,
)
from src.backtesting import run_momentum_backtest, plot_backtest_results

init_database()
import_history_from_csv()   # re-seed SQLite from CSV after cloud restarts

CACHE_PATH   = Path("data/fundamentals_cache.csv")
CACHE_MAX_AGE_DAYS = 7


def load_nifty500_df(filepath: str = "data/nifty500_list.csv") -> pd.DataFrame:
    """Return the full CSV with Company Name, Industry, Symbol columns."""
    path = Path(filepath)
    if not path.exists():
        st.error(
            f"**Missing file: `{filepath}`**\n\n"
            "The Nifty 500 constituent list is required to run the screener. "
            "Please add `data/nifty500_list.csv` to the repository and redeploy. "
            "Download it from: NSE India → Indices → Nifty 500 → Download."
        )
        st.stop()
    return pd.read_csv(filepath)


def build_sector_map(nifty_df: pd.DataFrame) -> dict:
    """Build {ticker: industry} dict from the CSV's Symbol and Industry columns."""
    return dict(zip(nifty_df["Symbol"], nifty_df["Industry"]))


def load_fundamentals_cache() -> pd.DataFrame | None:
    """Return cached fundamentals if file exists and is < 7 days old, else None."""
    if not CACHE_PATH.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
    if age > timedelta(days=CACHE_MAX_AGE_DAYS):
        return None
    return pd.read_csv(CACHE_PATH)


def scrape_and_cache_fundamentals(tickers: list, sector_map: dict,
                                   progress_bar) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """
    Scrape fundamentals for all tickers, updating a Streamlit progress bar.
    Saves results to cache after completion.
    Returns (passing, rejected_df, fundamentals_df).
    """
    from src.fundamental_filter import apply_red_flag_filter
    passing, rejected_df, fundamentals_df = apply_red_flag_filter(
        tickers, sector_map=sector_map, verbose=False
    )
    if not fundamentals_df.empty:
        fundamentals_df.to_csv(CACHE_PATH, index=False)
    return passing, rejected_df, fundamentals_df


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
    cache_exists = CACHE_PATH.exists()
    cache_age    = (
        datetime.now() - datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
        if cache_exists else None
    )
    if cache_exists and cache_age and cache_age <= timedelta(days=CACHE_MAX_AGE_DAYS):
        st.caption(f"Cache: {cache_age.days}d {cache_age.seconds//3600}h old — run will be fast ⚡")
    else:
        st.caption("First run scrapes all 500 stocks (~15 min). Subsequent runs use cache.")


# ── Regime (computed once per page load, stored so pipeline can use it) ──────
try:
    _regime_data = get_current_regime()
    regime = _regime_data["regime"]
    st.session_state["regime_data"] = _regime_data
except Exception:
    regime = "Neutral"
    st.session_state["regime_data"] = {"regime": "Neutral", "nifty_data": {}, "breadth_pct": None}
st.session_state["regime"] = regime


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — runs when button is clicked
# ══════════════════════════════════════════════════════════════════════════════

if run_clicked and weights:
    with st.spinner("Running screener pipeline…"):

        # ── Stage 0: load tickers + sector map ───────────────────────────────
        nifty_df    = load_nifty500_df()
        all_tickers = nifty_df["Symbol"].tolist()
        sector_map  = build_sector_map(nifty_df)
        st.toast(f"Loaded {len(all_tickers)} tickers from CSV")

        # ── Stage 1: download price data for full universe ────────────────────
        with st.spinner(f"Downloading 1y price data for {len(all_tickers)} stocks…"):
            tickers_ns = [t + ".NS" for t in all_tickers]
            raw = yf.download(
                tickers_ns, period="1y", progress=False, auto_adjust=True
            )["Close"]
            raw.columns = [c.replace(".NS", "") for c in raw.columns]
            st.session_state["price_df"] = raw

        # ── Stage 2: fundamentals — use cache if fresh, else scrape ──────────
        cached_df = load_fundamentals_cache()
        if cached_df is not None:
            st.toast("Using cached fundamentals ⚡ (< 7 days old)")
            fundamentals_df = cached_df
            # Re-apply filter logic using cached data to get passing list
            passing = fundamentals_df["ticker"].tolist() if not fundamentals_df.empty else []
            rejected_df = pd.DataFrame()
        else:
            with st.spinner(
                f"Stage 2: Scraping fundamentals for {len(all_tickers)} stocks from "
                f"Screener.in… (first run — this takes ~15 min)"
            ):
                passing, rejected_df, fundamentals_df = scrape_and_cache_fundamentals(
                    all_tickers, sector_map, progress_bar=None
                )
            st.toast(f"Scraped {len(all_tickers)} stocks — cache saved to data/fundamentals_cache.csv")

        # ── Stage 2b: earnings surprise scores ───────────────────────────────────
        surprise_scores = None
        if passing:
            from src.earnings_surprise import compute_surprise_factor_for_universe
            with st.spinner(f"Stage 2b: Fetching earnings surprise for {len(passing)} stocks…"):
                surprise_scores = compute_surprise_factor_for_universe(passing)
            st.toast(f"Earnings surprise computed for {(surprise_scores != 0).sum()} stocks")

        # ── Stage 3: factor model ranking ─────────────────────────────────────
        if passing:
            with st.spinner(f"Stage 3: Ranking {len(passing)} stocks by factor model…"):
                ranked_df, excluded_df = rank_stocks(
                    universe=passing,
                    price_df=raw,
                    fundamentals_df=fundamentals_df,
                    surprise_scores=surprise_scores,
                    weights=weights,
                )
            if not excluded_df.empty:
                st.toast(f"{len(excluded_df)} stocks excluded — insufficient factor data")
            top_n_tickers = ranked_df["ticker"].head(int(top_n)).tolist()
        else:
            ranked_df    = pd.DataFrame()
            excluded_df  = pd.DataFrame()
            top_n_tickers = []

        # ── Stage 4: technical green-flag filter ──────────────────────────────
        if top_n_tickers:
            with st.spinner("Stage 4: Applying technical green-flag filter…"):
                tech_df = apply_green_flag_filter(top_n_tickers, raw)
            final_picks = tech_df[tech_df["passes"]]["ticker"].tolist()
        else:
            tech_df     = pd.DataFrame()
            final_picks = []

        # Merge technical signals into ranked_df for display
        if not ranked_df.empty and not tech_df.empty:
            ranked_df = ranked_df.merge(
                tech_df[["ticker", "rsi", "above_ma50", "pct_from_52w_high", "passes"]],
                on="ticker", how="left"
            )

        # Persist in session state
        st.session_state["ranked_df"]       = ranked_df
        st.session_state["excluded_df"]     = excluded_df
        st.session_state["tech_df"]         = tech_df
        st.session_state["final_picks"]     = final_picks
        st.session_state["passing_count"]   = len(passing)
        st.session_state["universe_count"]  = len(all_tickers)
        st.session_state["fundamentals_df"] = fundamentals_df
        st.session_state["nifty_df"]        = nifty_df

        # Save run to history database, then flush to CSV for git persistence
        if not ranked_df.empty:
            save_run(
                ranked_df=ranked_df,
                regime=st.session_state.get("regime", "Neutral"),
                n_universe=len(all_tickers),
            )
            export_history_to_csv()

    st.success(f"Done! {len(final_picks)} stocks passed all 4 stages.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.title("📈 Quantamental Nifty 500 Screener")
st.caption("QVGS-style pipeline: Red-Flag filters → 5-Factor ranking → Technical timing")

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    regime = st.session_state.get("regime", "Neutral")
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

# ── Regime gauge ──────────────────────────────────────────────────────────────
_rd = st.session_state.get("regime_data", {})
_nifty_pct = _rd.get("nifty_data", {}).get("pct_from_ma", 0.0) or 0.0
st.plotly_chart(
    plot_regime_gauge(regime, _nifty_pct),
    use_container_width=True,
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Ranked Stocks",
    "📊 Factor Analysis",
    "📅 History & Changes",
    "🔎 Stock Deep-Dive",
    "⏱ Backtest",
])


# ── TAB 1: Ranked Stocks ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Top Ranked Stocks")
    st.caption("Only stocks that passed all 4 stages are shown (RSI < 70, above MA50, within 20% of 52w high).")

    ranked_df   = st.session_state.get("ranked_df", None)
    excluded_df = st.session_state.get("excluded_df", pd.DataFrame())

    if ranked_df is None or ranked_df.empty:
        st.info("Click **▶ Run Screener** in the sidebar to load real data.")
    else:
        # Only show stocks that passed the Stage 4 technical filter
        if "passes" in ranked_df.columns:
            display_df_full = ranked_df[ranked_df["passes"] == True].copy()
        else:
            display_df_full = ranked_df.copy()

        display_cols = ["rank", "ticker", "composite_score",
                        "value_score", "growth_score", "quality_score",
                        "momentum_score", "surprise_score"]
        if "rsi" in display_df_full.columns:
            display_cols += ["rsi", "above_ma50", "pct_from_52w_high"]

        display_df = display_df_full[display_cols].head(int(top_n)).copy()
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
            }
        )
        st.caption(
            f"Showing {len(display_df)} stocks that passed all 4 stages "
            f"(out of {len(ranked_df)} ranked)."
        )

        if not excluded_df.empty:
            with st.expander(f"⚠️ {len(excluded_df)} stocks excluded — insufficient factor data"):
                st.dataframe(excluded_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Download Tearsheets")
    st.caption("One-page PDF summary for each stock that passed all 4 stages.")

    _final    = st.session_state.get("final_picks", [])
    _rdf      = st.session_state.get("ranked_df", pd.DataFrame())
    _fdf      = st.session_state.get("fundamentals_df", pd.DataFrame())
    _tdf      = st.session_state.get("tech_df", pd.DataFrame())
    _ndf      = st.session_state.get("nifty_df", pd.DataFrame())
    _regime   = st.session_state.get("regime", "Neutral")

    if not _final:
        st.info("Run the screener to generate tearsheets.")
    else:
        # Build per-ticker lookup dicts once
        _ranked_lu = _rdf.set_index("ticker").to_dict("index") if not _rdf.empty else {}
        _fund_lu   = _fdf.set_index("ticker").to_dict("index") if not _fdf.empty else {}
        _tech_lu   = _tdf.set_index("ticker").to_dict("index") if not _tdf.empty else {}
        _name_lu   = dict(zip(_ndf["Symbol"], _ndf["Company Name"])) if not _ndf.empty else {}
        _sector_lu = dict(zip(_ndf["Symbol"], _ndf["Industry"]))    if not _ndf.empty else {}

        # Render 3 buttons per row
        for _i in range(0, len(_final), 3):
            _row_tickers = _final[_i : _i + 3]
            _cols = st.columns(len(_row_tickers))
            for _col, _tk in zip(_cols, _row_tickers):
                with _col:
                    _r = _ranked_lu.get(_tk, {})
                    _f = _fund_lu.get(_tk, {})
                    _t = _tech_lu.get(_tk, {})

                    def _safe(v, default=0):
                        try:
                            f = float(v) if v is not None else None
                            # f != f is True only for NaN (IEEE 754)
                            return default if f is None or f != f else f
                        except (TypeError, ValueError):
                            return default

                    _stock_data = {
                        "ticker":           _tk,
                        "company_name":     _name_lu.get(_tk, _tk),
                        "sector":           _sector_lu.get(_tk, ""),
                        "rank":             _r.get("rank", "—"),
                        "regime":           _regime,
                        "composite_score":  _safe(_r.get("composite_score")),
                        "value_score":      _safe(_r.get("value_score")),
                        "growth_score":     _safe(_r.get("growth_score")),
                        "quality_score":    _safe(_r.get("quality_score")),
                        "momentum_score":   _safe(_r.get("momentum_score")),
                        "surprise_score":   _safe(_r.get("surprise_score")),
                        "piotroski_score":  int(_safe(_f.get("piotroski_score"))),
                        "altman_zone":      _f.get("altman_zone", "—"),
                        "altman_zscore":    _safe(_f.get("altman_zscore")),
                        "current_price":    _safe(_t.get("current_price")),
                        "pe_ratio":         _safe(_f.get("pe_ratio")),
                        "roce":             _safe(_f.get("roce")),
                        "debt_equity":      _safe(_f.get("debt_equity")),
                        "revenue_cagr_3y":  _safe(_f.get("revenue_cagr_3y")),
                        "eps_cagr_3y":      _safe(_f.get("eps_cagr_3y")),
                        "rsi":              _safe(_r.get("rsi", _t.get("rsi"))),
                        "ma_50":            _safe(_t.get("ma_50")),
                        "pct_from_52w_high": _safe(_r.get("pct_from_52w_high",
                                                           _t.get("pct_from_52w_high"))),
                    }

                    _pdf_bytes = generate_tearsheet(_stock_data)
                    st.download_button(
                        label=f"📄 {_tk}",
                        data=_pdf_bytes,
                        file_name=f"{_tk}_tearsheet_{pd.Timestamp.today().date()}.pdf",
                        mime="application/pdf",
                        key=f"dl_{_tk}",
                        use_container_width=True,
                    )


# ── TAB 2: Factor Analysis ────────────────────────────────────────────────────
with tab2:
    st.subheader("Factor Attribution")
    if "ranked_df" in st.session_state and not st.session_state["ranked_df"].empty:
        st.plotly_chart(
            plot_factor_attribution(st.session_state["ranked_df"], top_n=15),
            use_container_width=True,
        )
    else:
        st.info("Run the screener to see factor attribution.")

    st.divider()
    st.subheader("Return Correlation Matrix")
    if "ranked_df" in st.session_state and "price_df" in st.session_state:
        _top15 = st.session_state["ranked_df"]["ticker"].tolist()[:15]
        st.plotly_chart(
            plot_correlation_heatmap(st.session_state["price_df"], _top15),
            use_container_width=True,
        )
    else:
        st.info("Run the screener to see correlation matrix.")


# ── TAB 3: History & Changes ──────────────────────────────────────────────────
with tab3:
    st.subheader("Week-over-Week Changes")
    st.caption("Which stocks entered and exited the top list since the last run?")
    if "ranked_df" in st.session_state:
        render_history_section(st.session_state["ranked_df"])
    else:
        st.info("Run the screener first to see history.")


# ── TAB 4: Stock Deep-Dive ────────────────────────────────────────────────────
with tab4:
    st.subheader("Single Stock Analysis")
    search_ticker = st.text_input("Enter NSE ticker (e.g. RELIANCE, TCS, INFY)", value="RELIANCE")
    if search_ticker:
        _tick = search_ticker.upper()
        st.caption(f"Showing data for: **{_tick}**")
        col_l, col_r = st.columns(2)
        with col_l:
            _rdf = st.session_state.get("ranked_df")
            if _rdf is not None and not _rdf.empty:
                _row = _rdf[_rdf["ticker"] == _tick]
                if not _row.empty:
                    _scores = {
                        "value":    float(_row["value_score"].iloc[0]),
                        "growth":   float(_row["growth_score"].iloc[0]),
                        "quality":  float(_row["quality_score"].iloc[0]),
                        "momentum": float(_row["momentum_score"].iloc[0]),
                        "surprise": float(_row["surprise_score"].iloc[0]),
                    }
                    st.plotly_chart(plot_factor_radar(_tick, _scores), use_container_width=True)
                else:
                    st.info(f"{_tick} not in current screener results.")
            else:
                st.info("Run the screener to see the factor profile.")
        with col_r:
            _hist = get_stock_history(_tick)
            if not _hist.empty:
                st.plotly_chart(plot_rank_history(_tick, _hist), use_container_width=True)
            else:
                st.info(f"No history yet for {_tick}. Run the screener to record it.")


# ── TAB 5: Backtest ──────────────────────────────────────────────────────────
with tab5:
    st.subheader("Momentum Strategy Backtest")
    st.caption(
        "Simulates running a price-momentum screener monthly on the Nifty 500 universe, "
        "holding the top N stocks for one month, then rebalancing. "
        "Transaction costs of 0.5% per rebalance are included."
    )

    with st.expander("⚠️ Important Limitations — Read Before Interpreting Results", expanded=False):
        st.warning(
            "**Survivorship bias:** This backtest uses the *current* Nifty 500 constituent list. "
            "Stocks that were delisted, merged, or dropped from the index between 2019–2024 are "
            "excluded, which artificially inflates returns. Real-world results would be lower.\n\n"
            "**Price-momentum only:** The live screener uses 5 factors (value, growth, quality, "
            "momentum, earnings surprise). This backtest uses price momentum only — historical "
            "fundamental data requires a paid source (Trendlyne Pro, Bloomberg).\n\n"
            "**No look-ahead bias on momentum:** The momentum signal uses only prices available "
            "at each rebalancing date. The most recent month is excluded to avoid the short-term "
            "reversal effect.\n\n"
            "**For educational purposes only.** Past simulated performance is not a reliable "
            "indicator of future returns."
        )

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_universe_n = st.slider("Universe size (top N Nifty 500 stocks)", 50, 500, 100, step=50)
    with bt_col2:
        bt_top_n = st.slider("Portfolio size (top N per rebalance)", 10, 50, 20, step=5)
    with bt_col3:
        bt_momentum_months = st.slider("Momentum lookback (months)", 3, 12, 6)

    run_bt = st.button("Run Backtest", type="primary", key="run_backtest_btn")

    if run_bt:
        with st.spinner(f"Downloading 5y prices for {bt_universe_n} stocks and running backtest…"):
            try:
                _bt_tickers = load_nifty500_list()[:bt_universe_n]
                _bt_nse = [t + ".NS" for t in _bt_tickers]
                _bt_prices = yf.download(
                    _bt_nse, period="5y", progress=False, auto_adjust=True
                )["Close"]
                _bt_prices.columns = [c.replace(".NS", "") for c in _bt_prices.columns]
                _bt_prices.index = pd.to_datetime(_bt_prices.index).tz_localize(None)

                _bt_nifty = yf.download(
                    "^CRSLDX", period="5y", progress=False, auto_adjust=True
                )["Close"].squeeze()
                _bt_nifty.index = pd.to_datetime(_bt_nifty.index).tz_localize(None)

                _bt_results = run_momentum_backtest(
                    price_df=_bt_prices,
                    nifty_prices=_bt_nifty,
                    top_n=bt_top_n,
                    momentum_months=bt_momentum_months,
                    start_date="2019-01-01",
                    end_date="2024-12-31",
                )
                st.session_state["backtest_results"] = _bt_results
            except Exception as _e:
                st.error(f"Backtest failed: {_e}")

    if "backtest_results" in st.session_state:
        _res = st.session_state["backtest_results"]

        if "error" in _res:
            st.error(_res["error"])
        else:
            # ── Metrics table ─────────────────────────────────────────────────
            st.subheader("Performance Metrics")
            _metrics = _res["metrics"]
            _rows = []
            for k, v in _metrics.items():
                if v == "":
                    _rows.append({"Metric": f"**{k}**", "Value": ""})
                else:
                    _rows.append({"Metric": k, "Value": v})
            st.table(pd.DataFrame(_rows).set_index("Metric"))

            st.divider()

            # ── Cumulative return chart ───────────────────────────────────────
            st.subheader("Cumulative Returns vs Nifty 500")
            st.plotly_chart(plot_backtest_results(_res), use_container_width=True)

            # ── Monthly holdings history ──────────────────────────────────────
            with st.expander("Monthly Portfolio Holdings (last 12 months)"):
                _hist = _res.get("portfolio_history", [])
                if _hist:
                    for _entry in _hist[-12:]:
                        st.write(f"**{_entry['date']}** — {', '.join(_entry['holdings'])}")
    else:
        st.info("Configure the parameters above and click **Run Backtest** to see results.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Bhavna Sharma · "
    "[GitHub](https://github.com/bhavna1434/nifty500-screener) · "
    "Data: Yahoo Finance + Screener.in · For educational purposes only."
)
