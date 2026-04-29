"""
main.py
=======
ICT Algo Trader — Main Automation Script

Runs every 15 minutes:
  1. Downloads latest EURUSD 1H + 4H data
  2. Runs EntryModel to detect signals
  3. Logs new signals to Google Sheets
  4. Saves chart.html and signals.html
  5. Checks signal expiry
  6. Sleeps 15 minutes and repeats
"""

import sys
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from strategies.entry_model import EntryModel, TradeSignal
from strategies.market_structure import MarketStructure
from utils.signal_viewer import plot_all_signals

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these values
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = dict(
    # Instruments — add or remove as needed
    instruments = [
        {"ticker": "EURUSD=X", "name": "EURUSD", "pip": 0.0001, "pip_value": 7.0},
        {"ticker": "GBPUSD=X", "name": "GBPUSD", "pip": 0.0001, "pip_value": 7.0},
        {"ticker": "AUDUSD=X", "name": "AUDUSD", "pip": 0.0001, "pip_value": 7.0},
        {"ticker": "USDCAD=X", "name": "USDCAD", "pip": 0.0001, "pip_value": 7.0},
    ],

    account_balance  = 10_000.0,
    risk_pct         = 0.01,
    daily_loss_limit = 0.03,
    max_drawdown     = 0.10,
    tp1_rr           = 1.0,
    tp2_rr           = 2.0,
    sl_buffer_pips   = 3.0,
    min_rr           = 1.5,
    swing_lookback   = 5,

    # Schedule
    interval_minutes = 15,
    signal_expiry_hours = 4,       # cancel signal if not triggered in X hours

    # Data
    htf_period       = "120d",     # 4H data lookback
    ltf_period       = "60d",      # 1H data lookback

    # Google Sheets
    credentials_file = str(ROOT / "credentials.json"),
    sheet_id         = "1Ea3gkpcUzzIlXUb9nX5q0Qb_jIcFVvxXIQfaqA7qGuE",
    sheet_name       = "Signals",  # tab name in your Google Sheet

    # Output files
    chart_html       = str(ROOT / "chart.html"),
    signals_html     = str(ROOT / "signals.html"),
    log_file         = str(ROOT / "logs" / "trader.log"),
)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

Path(CONFIG["log_file"]).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"], encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS
# ──────────────────────────────────────────────────────────────────────────────

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_HEADERS = [
    "Timestamp", "Instrument", "Direction",
    "Entry", "SL", "TP1", "TP2",
    "SL Pips", "RR TP1", "RR TP2",
    "Size (lots)", "Risk $", "Risk %",
    "Trend", "Zone", "Confirmation",
    "Status", "Expiry",
]


def get_sheet():
    """Connect to Google Sheets and return the worksheet."""
    creds = Credentials.from_service_account_file(
        CONFIG["credentials_file"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sheet  = client.open_by_key(CONFIG["sheet_id"])

    # Get or create the signals tab
    try:
        ws = sheet.worksheet(CONFIG["sheet_name"])
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(CONFIG["sheet_name"], rows=1000, cols=20)
        ws.append_row(SHEET_HEADERS)
        # Format header row bold
        ws.format("A1:R1", {"textFormat": {"bold": True}})
        log.info(f"Created sheet tab: {CONFIG['sheet_name']}")

    return ws


def signal_to_row(sig: TradeSignal) -> list:
    """Convert a TradeSignal to a Google Sheets row."""
    expiry = pd.Timestamp(sig.timestamp) + pd.Timedelta(hours=CONFIG["signal_expiry_hours"])
    return [
        str(sig.timestamp),
        sig.instrument,
        sig.direction.upper(),
        round(sig.entry, 5),
        round(sig.sl, 5),
        round(sig.tp1, 5),
        round(sig.tp2, 5),
        round(sig.sl_pips, 1),
        round(sig.rr_tp1, 2),
        round(sig.rr_tp2, 2),
        round(sig.position_size, 2),
        round(sig.risk_amount, 2),
        f"{sig.risk_pct * 100:.1f}%",
        sig.trend_source,
        sig.entry_zone,
        sig.confirmation,
        "PENDING",
        str(expiry),
    ]


def log_signals_to_sheet(ws, signals: list[TradeSignal], seen_timestamps: set) -> set:
    """Log new signals to Google Sheets. Returns updated seen_timestamps."""
    new_signals = [s for s in signals if f"{s.instrument}_{s.timestamp}" not in seen_timestamps]

    if not new_signals:
        return seen_timestamps

    for sig in new_signals:
        row = signal_to_row(sig)
        ws.append_row(row)
        seen_timestamps.add(f"{sig.instrument}_{sig.timestamp}")
        log.info(f"Logged to Sheets: {sig.direction.upper()} {sig.instrument} @ {sig.entry}")

    return seen_timestamps


def update_expired_signals(ws, expiry_hours: int) -> None:
    """Mark signals as EXPIRED if past their expiry time."""
    try:
        records = ws.get_all_records()
        now = datetime.now(timezone.utc)

        for i, row in enumerate(records, start=2):  # row 1 is header
            if row.get("Status") != "PENDING":
                continue
            expiry_str = row.get("Expiry", "")
            if not expiry_str:
                continue
            try:
                expiry_dt = pd.Timestamp(expiry_str)
                if expiry_dt.tzinfo is None:
                    expiry_dt = expiry_dt.tz_localize("UTC")
                if now > expiry_dt:
                    ws.update_cell(i, SHEET_HEADERS.index("Status") + 1, "EXPIRED")
                    log.info(f"Signal row {i} marked EXPIRED")
            except Exception:
                pass
    except Exception as e:
        log.warning(f"Could not update expired signals: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download and normalise OHLCV data from yfinance."""
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    else:
        df.columns = df.columns.str.lower()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_once(ws, seen_timestamps: set) -> set:
    """Single run — loop through all instruments, analyse, log, save charts."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    log.info(f"── Run at {now} ──────────────────────────")

    all_signals = []

    for inst in CONFIG["instruments"]:
        name   = inst["name"]
        ticker = inst["ticker"]
        pip_value = inst["pip_value"]

        try:
            log.info(f"Processing {name}...")

            # 1. Download data
            df_1h = download_data(ticker, CONFIG["ltf_period"], "1h")
            df_4h = download_data(ticker, CONFIG["htf_period"], "4h")
            log.info(f"  {name} — 1H: {len(df_1h)} rows  4H: {len(df_4h)} rows")

            # 2. Run entry model
            model = EntryModel(
                df_high          = df_4h,
                df_low           = df_1h,
                instrument       = name,
                account_balance  = CONFIG["account_balance"],
                risk_pct         = CONFIG["risk_pct"],
                daily_loss_limit = CONFIG["daily_loss_limit"],
                max_drawdown     = CONFIG["max_drawdown"],
                tp1_rr           = CONFIG["tp1_rr"],
                tp2_rr           = CONFIG["tp2_rr"],
                sl_buffer_pips   = CONFIG["sl_buffer_pips"],
                min_rr           = CONFIG["min_rr"],
                swing_lookback   = CONFIG["swing_lookback"],
                pip_value        = pip_value,
            )
            log.info(f"  {name} — Signals: {len(model.signals)}")

            if model.latest_signal():
                log.info(str(model.latest_signal()))

            # 3. Log to Google Sheets
            seen_timestamps = log_signals_to_sheet(ws, model.signals, seen_timestamps)

            # 4. Save per-instrument signal charts
            if model.signals:
                sig_html = str(ROOT / f"signals_{name}.html")
                plot_all_signals(
                    df=df_1h,
                    signals=model.signals,
                    candles_before=60,
                    candles_after=20,
                    save_html=sig_html,
                )
                log.info(f"  {name} — Charts saved → signals_{name}.html")

            # 5. Save market structure chart
            ms = MarketStructure(df_1h, CONFIG["swing_lookback"])
            ms.plot(last_n=300, title=f"{name} 1H — ICT Structure")\
              .write_html(str(ROOT / f"chart_{name}.html"))

            all_signals.extend(model.signals)

        except Exception as e:
            log.error(f"  {name} failed: {e}")
            continue

    # 6. Update expired signals
    update_expired_signals(ws, CONFIG["signal_expiry_hours"])

    log.info(f"Total signals across all instruments: {len(all_signals)}")
    return seen_timestamps


def main():
    log.info("=" * 60)
    log.info("  ICT Algo Trader — Starting")
    log.info(f"  Instrument : {CONFIG['instrument']}")
    log.info(f"  Interval   : every {CONFIG['interval_minutes']} minutes")
    log.info(f"  Risk/trade : {CONFIG['risk_pct']*100:.0f}%")
    log.info("=" * 60)

    # Connect to Google Sheets once
    log.info("Connecting to Google Sheets...")
    ws = get_sheet()
    log.info("Connected ✓")

    seen_timestamps: set = set()
    interval_seconds = CONFIG["interval_minutes"] * 60

    while True:
        try:
            seen_timestamps = run_once(ws, seen_timestamps)
        except Exception as e:
            log.error(f"Run failed: {e}")
            log.error(traceback.format_exc())

        log.info(f"Sleeping {CONFIG['interval_minutes']} minutes...\n")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()