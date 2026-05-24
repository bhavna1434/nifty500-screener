# src/history_tracker.py
# Screener History Log — tracks weekly top-20 results in a SQLite database
#
# Why this matters:
#   Most student projects are one-shot scripts. A history log makes yours a
#   living system — you can show "RELIANCE entered the top 20 on May 12th" or
#   "INFY dropped out after earnings miss on June 3rd". That's what a real
#   quantamental product looks like.
#
# SQLite is built into Python — no install needed, no external database.
# The database file lives at data/screener_history.db in your project folder.
#
# We'll build this in Week 8 alongside the Streamlit dashboard.

import sqlite3
import pandas as pd
from datetime import date, datetime
import os


# Path to the database file (relative to project root)
DB_PATH = "data/screener_history.db"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def init_database():
    """
    Create the SQLite database and tables if they don't already exist.
    Call this once at app startup — it's safe to call multiple times
    (CREATE TABLE IF NOT EXISTS is idempotent).
    """
    os.makedirs("data", exist_ok=True)  # create data/ folder if needed

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: screener_runs — one row per weekly run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screener_runs (
            run_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date  TEXT NOT NULL,          -- e.g. "2026-05-12"
            regime    TEXT,                   -- e.g. "Risk-On"
            n_stocks_universe  INTEGER,       -- stocks before filtering
            n_stocks_passed    INTEGER        -- stocks in final top-20
        )
    """)

    # Table 2: top_stocks — one row per stock per run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS top_stocks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER REFERENCES screener_runs(run_id),
            ticker          TEXT NOT NULL,
            rank            INTEGER,
            composite_score REAL,
            value_score     REAL,
            growth_score    REAL,
            quality_score   REAL,
            momentum_score  REAL,
            surprise_score  REAL,
            piotroski       INTEGER,
            altman_zone     TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database ready at {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVING RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_run(ranked_df: pd.DataFrame, regime: str, n_universe: int) -> int:
    """
    Save the results of a screener run to the database.

    Args:
        ranked_df: DataFrame with top-ranked stocks (output of factor_model.rank_stocks)
                   Expected columns: ticker, rank, composite_score, value_score,
                   growth_score, quality_score, momentum_score, surprise_score,
                   piotroski, altman_zone
        regime: Current market regime string (e.g. "Risk-On")
        n_universe: Total stocks before Red-Flag filtering

    Returns:
        run_id of the saved run (int)
    """
    init_database()  # ensure tables exist

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = date.today().isoformat()

    # Insert the run metadata
    cursor.execute("""
        INSERT INTO screener_runs (run_date, regime, n_stocks_universe, n_stocks_passed)
        VALUES (?, ?, ?, ?)
    """, (today, regime, n_universe, len(ranked_df)))

    run_id = cursor.lastrowid  # get the auto-generated ID

    # Insert each stock's data
    for _, row in ranked_df.iterrows():
        cursor.execute("""
            INSERT INTO top_stocks
            (run_id, ticker, rank, composite_score, value_score, growth_score,
             quality_score, momentum_score, surprise_score, piotroski, altman_zone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            row.get("ticker"),
            row.get("rank"),
            row.get("composite_score"),
            row.get("value_score"),
            row.get("growth_score"),
            row.get("quality_score"),
            row.get("momentum_score"),
            row.get("surprise_score"),
            row.get("piotroski"),
            row.get("altman_zone"),
        ))

    conn.commit()
    conn.close()

    print(f"  Saved run #{run_id} ({today}): {len(ranked_df)} stocks, regime={regime}")
    return run_id


# ══════════════════════════════════════════════════════════════════════════════
# READING HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_run() -> pd.DataFrame:
    """
    Retrieve the most recent screener run results.

    Returns:
        DataFrame of stocks from the last run, sorted by rank
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT ts.*, sr.run_date, sr.regime
        FROM top_stocks ts
        JOIN screener_runs sr ON ts.run_id = sr.run_id
        WHERE ts.run_id = (SELECT MAX(run_id) FROM screener_runs)
        ORDER BY ts.rank
    """, conn)
    conn.close()
    return df


def get_all_runs_summary() -> pd.DataFrame:
    """
    Get a summary table of all historical runs (one row per run).

    Returns:
        DataFrame with run_date, regime, n_stocks_universe, n_stocks_passed
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT run_id, run_date, regime, n_stocks_universe, n_stocks_passed
        FROM screener_runs
        ORDER BY run_date DESC
    """, conn)
    conn.close()
    return df


def compare_runs(run_id_new: int, run_id_old: int) -> dict:
    """
    Compare two runs to find which stocks entered/exited the top list.

    Args:
        run_id_new: The more recent run
        run_id_old: The previous run to compare against

    Returns:
        dict with keys:
            - new_entries: List of tickers that are NEW in the recent run
            - exits: List of tickers that DROPPED OUT since the last run
            - held: List of tickers present in both runs
    """
    conn = sqlite3.connect(DB_PATH)

    new_tickers = set(pd.read_sql_query(
        "SELECT ticker FROM top_stocks WHERE run_id = ?", conn, params=(run_id_new,)
    )["ticker"])

    old_tickers = set(pd.read_sql_query(
        "SELECT ticker FROM top_stocks WHERE run_id = ?", conn, params=(run_id_old,)
    )["ticker"])

    conn.close()

    return {
        "new_entries": sorted(new_tickers - old_tickers),
        "exits":       sorted(old_tickers - new_tickers),
        "held":        sorted(new_tickers & old_tickers),
    }


def get_stock_history(ticker: str) -> pd.DataFrame:
    """
    Get the full history of a single stock's appearances in the screener.
    Useful for the 'single stock deep-dive' view in the dashboard.

    Args:
        ticker: NSE symbol (e.g. "RELIANCE")

    Returns:
        DataFrame showing the stock's rank and scores across all runs it appeared in
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT sr.run_date, ts.rank, ts.composite_score,
               ts.value_score, ts.growth_score, ts.quality_score,
               ts.momentum_score, ts.surprise_score
        FROM top_stocks ts
        JOIN screener_runs sr ON ts.run_id = sr.run_id
        WHERE ts.ticker = ?
        ORDER BY sr.run_date
    """, conn, params=(ticker,))
    conn.close()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT HELPER: call this from app.py
# ══════════════════════════════════════════════════════════════════════════════

def render_history_section(current_df: pd.DataFrame):
    """
    Render the history section in the Streamlit app.
    Shows new entries, exits, and the run history table.

    Call this from app.py after running the screener:
        from src.history_tracker import render_history_section
        render_history_section(top_stocks_df)

    Args:
        current_df: The current run's ranked DataFrame
    """
    import streamlit as st

    runs = get_all_runs_summary()

    if len(runs) < 2:
        st.info("Run the screener at least twice to see what changed week-over-week.")
        return

    latest_id = runs.iloc[0]["run_id"]
    prev_id = runs.iloc[1]["run_id"]
    diff = compare_runs(latest_id, prev_id)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("New entries", len(diff["new_entries"]))
        if diff["new_entries"]:
            for t in diff["new_entries"]:
                st.success(f"↑ {t}")

    with col2:
        st.metric("Exits", len(diff["exits"]))
        if diff["exits"]:
            for t in diff["exits"]:
                st.error(f"↓ {t}")

    with col3:
        st.metric("Held from last week", len(diff["held"]))

    st.caption(f"Comparing run on {runs.iloc[0]['run_date']} vs {runs.iloc[1]['run_date']}")

    with st.expander("View full run history"):
        st.dataframe(runs, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSV PERSISTENCE — survives Streamlit Cloud ephemeral filesystem restarts
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = "data/screener_history.csv"

# Columns exported to / imported from CSV
_CSV_COLS = [
    "run_date", "regime", "universe_size",
    "ticker", "rank", "composite_score",
    "value_score", "growth_score", "quality_score",
    "momentum_score", "surprise_score",
]


def export_history_to_csv():
    """
    Dump the full SQLite history to data/screener_history.csv.

    Called after every save_run() so the CSV always reflects the latest state.
    The CSV is committed to git, giving history a free persistent store that
    survives Streamlit Cloud's ephemeral filesystem across restarts.
    """
    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            sr.run_date,
            sr.regime,
            sr.n_stocks_universe  AS universe_size,
            ts.ticker,
            ts.rank,
            ts.composite_score,
            ts.value_score,
            ts.growth_score,
            ts.quality_score,
            ts.momentum_score,
            ts.surprise_score
        FROM top_stocks ts
        JOIN screener_runs sr ON ts.run_id = sr.run_id
        ORDER BY sr.run_date DESC, ts.rank ASC
    """, conn)
    conn.close()

    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"  History exported: {len(df)} rows -> {CSV_PATH}")


def import_history_from_csv():
    """
    Seed the SQLite database from data/screener_history.csv on startup.

    Only runs when:
      - The CSV file exists (committed to git / uploaded)
      - The screener_runs table is empty (fresh DB after a cloud restart)

    This is idempotent: if the DB already has data it does nothing.
    """
    init_database()

    if not os.path.exists(CSV_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    existing = pd.read_sql_query("SELECT COUNT(*) AS n FROM screener_runs", conn).iloc[0]["n"]
    if existing > 0:
        conn.close()
        return  # DB already populated — nothing to do

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        conn.close()
        return

    cursor = conn.cursor()
    imported_runs = 0
    imported_stocks = 0

    for run_date, group in df.groupby("run_date", sort=False):
        regime = group["regime"].iloc[0]
        universe_size = int(group["universe_size"].iloc[0])
        n_passed = len(group)

        cursor.execute("""
            INSERT INTO screener_runs (run_date, regime, n_stocks_universe, n_stocks_passed)
            VALUES (?, ?, ?, ?)
        """, (run_date, regime, universe_size, n_passed))
        run_id = cursor.lastrowid

        for _, row in group.iterrows():
            cursor.execute("""
                INSERT INTO top_stocks
                    (run_id, ticker, rank, composite_score,
                     value_score, growth_score, quality_score,
                     momentum_score, surprise_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                row.get("ticker"),
                row.get("rank"),
                row.get("composite_score"),
                row.get("value_score"),
                row.get("growth_score"),
                row.get("quality_score"),
                row.get("momentum_score"),
                row.get("surprise_score"),
            ))
            imported_stocks += 1
        imported_runs += 1

    conn.commit()
    conn.close()
    print(f"  History imported from CSV: {imported_runs} runs, {imported_stocks} stock rows")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_database()

    # Create some dummy data to test with
    dummy_df = pd.DataFrame([
        {"ticker": "RELIANCE", "rank": 1, "composite_score": 1.8, "value_score": 1.2,
         "growth_score": 2.1, "quality_score": 1.5, "momentum_score": 2.3,
         "surprise_score": 1.1, "piotroski": 7, "altman_zone": "Safe"},
        {"ticker": "TCS", "rank": 2, "composite_score": 1.6, "value_score": 0.9,
         "growth_score": 1.8, "quality_score": 2.0, "momentum_score": 1.4,
         "surprise_score": 1.8, "piotroski": 8, "altman_zone": "Safe"},
        {"ticker": "HDFCBANK", "rank": 3, "composite_score": 1.4, "value_score": 1.5,
         "growth_score": 1.2, "quality_score": 1.8, "momentum_score": 0.9,
         "surprise_score": 0.5, "piotroski": 6, "altman_zone": "Safe"},
    ])

    run_id = save_run(dummy_df, regime="Risk-On", n_universe=500)
    print(f"\nSaved run ID: {run_id}")

    history = get_latest_run()
    print(f"\nLatest run:\n{history[['ticker', 'rank', 'composite_score']].to_string()}")

    ticker_hist = get_stock_history("RELIANCE")
    print(f"\nRELIANCE history:\n{ticker_hist}")

    print("\n✅ history_tracker.py working correctly!")
