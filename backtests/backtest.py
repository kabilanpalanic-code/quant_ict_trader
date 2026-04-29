"""
backtests/backtest.py
=====================
ICT Strategy Backtester

Walks through historical data bar by bar.
At each bar, runs EntryModel on data up to that point.
Simulates trade execution and tracks results.

Usage:
    from backtests.backtest import Backtest
    bt = Backtest(instrument="EURUSD", start="2025-01-01", end="2025-06-30")
    bt.run()
    bt.report()
    bt.plot().show()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime
import yfinance as yf
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from strategies.entry_model import EntryModel, TradeSignal


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """A completed backtest trade with entry, exit and P&L."""
    signal:         TradeSignal
    entry_bar:      int
    entry_time:     pd.Timestamp
    entry_price:    float
    direction:      Literal["long", "short"]
    sl:             float
    tp1:            float
    tp2:            float
    sl_pips:        float
    position_size:  float
    risk_amount:    float

    # Exit info — filled when trade closes
    exit_bar:       int   = -1
    exit_time:      pd.Timestamp | None = None
    exit_price:     float = 0.0
    result:         Literal["WIN", "LOSS", "PARTIAL", "EXPIRED", "OPEN"] = "OPEN"
    tp1_hit:        bool  = False
    tp2_hit:        bool  = False
    sl_hit:         bool  = False
    pnl:            float = 0.0       # $ profit/loss
    pnl_pips:       float = 0.0
    bars_held:      int   = 0
    rr_achieved:    float = 0.0


@dataclass
class BacktestResult:
    """Summary statistics for a completed backtest."""
    instrument:     str
    start:          str
    end:            str
    initial_balance: float
    final_balance:  float

    total_trades:   int
    wins:           int
    losses:         int
    partials:       int
    expired:        int

    total_pnl:      float
    total_profit:   float
    total_loss:     float
    win_rate:       float
    profit_factor:  float
    avg_win:        float
    avg_loss:       float
    best_trade:     float
    worst_trade:    float
    max_drawdown:   float
    max_drawdown_pct: float
    avg_bars_held:  float
    trades:         list[BacktestTrade] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  BACKTEST RESULTS — {self.instrument}\n"
            f"  {self.start}  →  {self.end}\n"
            f"{'='*55}\n"
            f"  Total trades   : {self.total_trades}\n"
            f"  Wins           : {self.wins}  ({self.win_rate:.1f}%)\n"
            f"  Losses         : {self.losses}\n"
            f"  Partials (TP1) : {self.partials}\n"
            f"  Expired        : {self.expired}\n"
            f"{'─'*55}\n"
            f"  Total P&L      : ${self.total_pnl:+.2f}\n"
            f"  Total profit   : ${self.total_profit:.2f}\n"
            f"  Total loss     : ${self.total_loss:.2f}\n"
            f"  Profit factor  : {self.profit_factor:.2f}\n"
            f"{'─'*55}\n"
            f"  Avg win        : ${self.avg_win:.2f}\n"
            f"  Avg loss       : ${self.avg_loss:.2f}\n"
            f"  Best trade     : ${self.best_trade:.2f}\n"
            f"  Worst trade    : ${self.worst_trade:.2f}\n"
            f"{'─'*55}\n"
            f"  Starting bal   : ${self.initial_balance:.2f}\n"
            f"  Final balance  : ${self.final_balance:.2f}\n"
            f"  Return         : {((self.final_balance/self.initial_balance)-1)*100:+.2f}%\n"
            f"  Max drawdown   : ${self.max_drawdown:.2f} ({self.max_drawdown_pct:.1f}%)\n"
            f"{'─'*55}\n"
            f"  Avg bars held  : {self.avg_bars_held:.1f}\n"
            f"{'='*55}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ──────────────────────────────────────────────────────────────────────────────

class Backtest:
    """
    ICT Strategy Backtester.

    Parameters
    ----------
    instrument      : e.g. "EURUSD"
    ticker          : yfinance ticker e.g. "EURUSD=X"
    start           : start date "YYYY-MM-DD"
    end             : end date "YYYY-MM-DD" (default: today)
    account_balance : starting balance
    risk_pct        : risk per trade
    tp1_rr          : TP1 risk:reward
    tp2_rr          : TP2 risk:reward
    tp1_close_pct   : % of position closed at TP1
    sl_buffer_pips  : SL buffer
    min_rr          : minimum RR to take trade
    swing_lookback  : swing detection lookback
    signal_expiry_h : hours before pending signal expires
    min_warmup_bars : minimum bars needed before first signal
    step_bars       : check for signals every N bars (1 = every bar, 4 = every 4 hours)
    pip_value       : $ per pip per lot
    """

    INSTRUMENTS = {
        "EURUSD": {"ticker": "EURUSD=X", "pip": 0.0001, "pip_value": 7.0},
        "GBPUSD": {"ticker": "GBPUSD=X", "pip": 0.0001, "pip_value": 7.0},
        "AUDUSD": {"ticker": "AUDUSD=X", "pip": 0.0001, "pip_value": 7.0},
        "USDCAD": {"ticker": "USDCAD=X", "pip": 0.0001, "pip_value": 7.0},
    }

    def __init__(
        self,
        instrument: str = "EURUSD",
        start: str = "2025-01-01",
        end: str | None = None,
        account_balance: float = 10_000.0,
        risk_pct: float = 0.01,
        tp1_rr: float = 1.0,
        tp2_rr: float = 2.0,
        tp1_close_pct: float = 0.5,
        sl_buffer_pips: float = 3.0,
        min_rr: float = 1.5,
        swing_lookback: int = 5,
        signal_expiry_h: int = 4,
        min_warmup_bars: int = 100,
        step_bars: int = 4,
        pip_value: float | None = None,
    ):
        self.instrument     = instrument.upper()
        self.start          = start
        self.end            = end or datetime.today().strftime("%Y-%m-%d")
        self.initial_balance = account_balance
        self.balance        = account_balance
        self.risk_pct       = risk_pct
        self.tp1_rr         = tp1_rr
        self.tp2_rr         = tp2_rr
        self.tp1_close_pct  = tp1_close_pct
        self.sl_buffer_pips = sl_buffer_pips
        self.min_rr         = min_rr
        self.swing_lookback = swing_lookback
        self.signal_expiry_h = signal_expiry_h
        self.min_warmup_bars = min_warmup_bars
        self.step_bars      = step_bars

        inst = self.INSTRUMENTS.get(self.instrument, {})
        self.ticker     = inst.get("ticker", f"{instrument}=X")
        self.pip_value  = pip_value or inst.get("pip_value", 7.0)

        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = []
        self.result: BacktestResult | None = None

        self._df_1h: pd.DataFrame | None = None
        self._df_4h: pd.DataFrame | None = None

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Run the full backtest. Returns BacktestResult."""
        print(f"\nBacktest: {self.instrument}  {self.start} → {self.end}")
        print("Downloading data...")

        self._download_data()
        print(f"1H bars: {len(self._df_1h)}  4H bars: {len(self._df_4h)}")

        # Filter to backtest period
        df_period = self._df_1h[
            (self._df_1h.index >= pd.Timestamp(self.start, tz="UTC")) &
            (self._df_1h.index <= pd.Timestamp(self.end,   tz="UTC"))
        ]

        if len(df_period) < self.min_warmup_bars:
            raise ValueError(f"Not enough bars in period: {len(df_period)}")

        print(f"Bars in period: {len(df_period)}")
        print("Running bar by bar simulation...")

        open_trades: list[BacktestTrade] = []
        seen_signals: set = set()
        self.equity_curve = [self.initial_balance]

        for i in range(self.min_warmup_bars, len(df_period), self.step_bars):
            current_bar = df_period.index[i]

            # Slice data up to current bar — no future peeking
            df_1h_slice = self._df_1h[self._df_1h.index <= current_bar]
            df_4h_slice = self._df_4h[self._df_4h.index <= current_bar]

            if len(df_1h_slice) < self.min_warmup_bars:
                continue

            # Check open trades against current bar OHLC
            bar_high  = df_period["high"].iloc[i]
            bar_low   = df_period["low"].iloc[i]
            bar_close = df_period["close"].iloc[i]

            for trade in open_trades[:]:
                self._check_trade_exit(trade, i, current_bar,
                                       bar_high, bar_low, bar_close)
                if trade.result != "OPEN":
                    open_trades.remove(trade)
                    self.trades.append(trade)
                    self.balance += trade.pnl
                    self.equity_curve.append(self.balance)

            # Check signal expiry
            for trade in open_trades[:]:
                expiry = trade.entry_time + pd.Timedelta(hours=self.signal_expiry_h)
                if current_bar > expiry:
                    trade.result    = "EXPIRED"
                    trade.exit_bar  = i
                    trade.exit_time = current_bar
                    trade.pnl       = 0.0
                    open_trades.remove(trade)
                    self.trades.append(trade)

            # Only check for new signals every step_bars
            if i % self.step_bars != 0:
                continue

            # Run entry model on current slice
            try:
                model = EntryModel(
                    df_high          = df_4h_slice,
                    df_low           = df_1h_slice,
                    instrument       = self.instrument,
                    account_balance  = self.balance,
                    risk_pct         = self.risk_pct,
                    tp1_rr           = self.tp1_rr,
                    tp2_rr           = self.tp2_rr,
                    tp1_close_pct    = self.tp1_close_pct,
                    sl_buffer_pips   = self.sl_buffer_pips,
                    min_rr           = self.min_rr,
                    swing_lookback   = self.swing_lookback,
                    pip_value        = self.pip_value,
                )
            except Exception:
                continue

            for sig in model.signals:
                sig_key = f"{sig.direction}_{sig.entry:.5f}_{current_bar.date()}"
                if sig_key in seen_signals:
                    continue
                if len(open_trades) >= 2:   # max 2 open trades
                    continue
                seen_signals.add(sig_key)

                trade = BacktestTrade(
                    signal        = sig,
                    entry_bar     = i,
                    entry_time    = current_bar,
                    entry_price   = sig.entry,
                    direction     = sig.direction,
                    sl            = sig.sl,
                    tp1           = sig.tp1,
                    tp2           = sig.tp2,
                    sl_pips       = sig.sl_pips,
                    position_size = sig.position_size,
                    risk_amount   = sig.risk_amount,
                )
                open_trades.append(trade)

        # Close any remaining open trades at last bar
        for trade in open_trades:
            last_close = df_period["close"].iloc[-1]
            trade.result     = "OPEN"
            trade.exit_bar   = len(df_period) - 1
            trade.exit_time  = df_period.index[-1]
            trade.exit_price = last_close
            pips = (last_close - trade.entry_price) / 0.0001
            if trade.direction == "short":
                pips = -pips
            trade.pnl_pips = pips
            trade.pnl      = round(pips * self.pip_value * trade.position_size, 2)
            self.trades.append(trade)

        self.result = self._calculate_results()
        return self.result

    def report(self) -> None:
        """Print backtest results."""
        if self.result:
            print(self.result)
        else:
            print("Run backtest first: bt.run()")

    def trades_df(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "direction":   t.direction,
                "entry":       t.entry_price,
                "sl":          t.sl,
                "tp1":         t.tp1,
                "tp2":         t.tp2,
                "result":      t.result,
                "tp1_hit":     t.tp1_hit,
                "pnl_$":       t.pnl,
                "pnl_pips":    t.pnl_pips,
                "bars_held":   t.bars_held,
                "rr":          t.rr_achieved,
                "zone":        t.signal.entry_zone,
            })
        return pd.DataFrame(rows)

    def plot(self, height: int = 800) -> go.Figure:
        """Plot equity curve and trade distribution."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Equity Curve",
                "Trade P&L Distribution",
                "Win/Loss by Month",
                "Cumulative P&L",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 1. Equity curve
        fig.add_trace(go.Scatter(
            y=self.equity_curve,
            mode="lines",
            line=dict(color="#26DE81", width=2),
            name="Equity",
            fill="tozeroy",
            fillcolor="rgba(38,222,129,0.1)",
        ), row=1, col=1)

        # 2. P&L distribution
        pnls = [t.pnl for t in self.trades if t.result != "OPEN"]
        colours = ["#26DE81" if p > 0 else "#FC5C65" for p in pnls]
        fig.add_trace(go.Bar(
            y=pnls,
            marker_color=colours,
            name="Trade P&L",
            showlegend=False,
        ), row=1, col=2)

        # 3. Monthly P&L
        if self.trades:
            df_t = self.trades_df()
            df_t["month"] = pd.to_datetime(df_t["entry_time"]).dt.to_period("M").astype(str)
            monthly = df_t.groupby("month")["pnl_$"].sum()
            fig.add_trace(go.Bar(
                x=monthly.index.tolist(),
                y=monthly.values.tolist(),
                marker_color=["#26DE81" if v > 0 else "#FC5C65" for v in monthly.values],
                name="Monthly P&L",
                showlegend=False,
            ), row=2, col=1)

        # 4. Cumulative P&L
        cumulative = np.cumsum([t.pnl for t in self.trades if t.result != "OPEN"])
        fig.add_trace(go.Scatter(
            y=cumulative.tolist(),
            mode="lines",
            line=dict(color="#45AAF2", width=2),
            name="Cumulative P&L",
            fill="tozeroy",
            fillcolor="rgba(69,170,242,0.1)",
        ), row=2, col=2)

        # Add zero line
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)",
                      line_dash="dash", row=2, col=2)

        title = (
            f"{self.instrument} Backtest  |  "
            f"{self.start} → {self.end}  |  "
            f"Trades: {len(self.trades)}  |  "
            f"Return: {((self.balance/self.initial_balance)-1)*100:+.1f}%"
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color="#E8E8E8"), x=0.01),
            paper_bgcolor="#0F1117",
            plot_bgcolor="#0F1117",
            height=height,
            margin=dict(l=60, r=20, t=60, b=20),
            showlegend=False,
        )
        axis_style = dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color="#AAA", size=10),
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

        return fig

    # ── private ────────────────────────────────────────────────────────────────

    def _download_data(self) -> None:
        """Download 1H and 4H data covering the backtest period + warmup."""
        def fetch(interval: str, period: str) -> pd.DataFrame:
            raw = yf.download(self.ticker, period=period,
                              interval=interval, auto_adjust=True, progress=False)
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0).str.lower()
            else:
                df.columns = df.columns.str.lower()
            df.index = pd.to_datetime(df.index, utc=True)
            return df.dropna()

        self._df_1h = fetch("1h",  "730d")   # max yfinance allows for 1H
        self._df_4h = fetch("4h",  "730d")

    def _check_trade_exit(
        self,
        trade: BacktestTrade,
        bar_idx: int,
        bar_time: pd.Timestamp,
        high: float,
        low: float,
        close: float,
    ) -> None:
        """Check if current bar triggers SL, TP1, or TP2 for a trade."""
        if trade.result != "OPEN":
            return

        trade.bars_held = bar_idx - trade.entry_bar

        if trade.direction == "long":
            # SL hit
            if low <= trade.sl:
                if not trade.tp1_hit:
                    # Full loss
                    trade.result     = "LOSS"
                    trade.sl_hit     = True
                    trade.exit_price = trade.sl
                    trade.exit_time  = bar_time
                    trade.exit_bar   = bar_idx
                    pips = (trade.sl - trade.entry_price) / 0.0001
                    trade.pnl_pips   = pips
                    trade.pnl        = round(pips * self.pip_value * trade.position_size, 2)
                    trade.rr_achieved = pips / trade.sl_pips if trade.sl_pips else 0
                else:
                    # TP1 already hit — SL moved to breakeven
                    trade.result     = "PARTIAL"
                    trade.exit_price = trade.entry_price
                    trade.exit_time  = bar_time
                    trade.exit_bar   = bar_idx
                    trade.pnl        = round(trade.pnl * self.tp1_close_pct, 2)
                    trade.rr_achieved = 1.0

            # TP1 hit
            elif high >= trade.tp1 and not trade.tp1_hit:
                trade.tp1_hit = True
                tp1_pnl = abs(trade.tp1 - trade.entry_price) / 0.0001
                tp1_pnl *= self.pip_value * trade.position_size * self.tp1_close_pct
                trade.pnl += tp1_pnl

            # TP2 hit
            elif high >= trade.tp2 and trade.tp1_hit:
                trade.result     = "WIN"
                trade.tp2_hit    = True
                trade.exit_price = trade.tp2
                trade.exit_time  = bar_time
                trade.exit_bar   = bar_idx
                tp2_pnl = abs(trade.tp2 - trade.entry_price) / 0.0001
                tp2_pnl *= self.pip_value * trade.position_size * (1 - self.tp1_close_pct)
                trade.pnl += tp2_pnl
                trade.pnl        = round(trade.pnl, 2)
                trade.rr_achieved = self.tp2_rr

        else:  # short
            # SL hit
            if high >= trade.sl:
                if not trade.tp1_hit:
                    trade.result     = "LOSS"
                    trade.sl_hit     = True
                    trade.exit_price = trade.sl
                    trade.exit_time  = bar_time
                    trade.exit_bar   = bar_idx
                    pips = (trade.entry_price - trade.sl) / 0.0001
                    trade.pnl_pips   = pips
                    trade.pnl        = round(pips * self.pip_value * trade.position_size, 2)
                    trade.rr_achieved = pips / trade.sl_pips if trade.sl_pips else 0
                else:
                    trade.result     = "PARTIAL"
                    trade.exit_price = trade.entry_price
                    trade.exit_time  = bar_time
                    trade.exit_bar   = bar_idx
                    trade.pnl        = round(trade.pnl * self.tp1_close_pct, 2)
                    trade.rr_achieved = 1.0

            # TP1 hit
            elif low <= trade.tp1 and not trade.tp1_hit:
                trade.tp1_hit = True
                tp1_pnl = abs(trade.entry_price - trade.tp1) / 0.0001
                tp1_pnl *= self.pip_value * trade.position_size * self.tp1_close_pct
                trade.pnl += tp1_pnl

            # TP2 hit
            elif low <= trade.tp2 and trade.tp1_hit:
                trade.result     = "WIN"
                trade.tp2_hit    = True
                trade.exit_price = trade.tp2
                trade.exit_time  = bar_time
                trade.exit_bar   = bar_idx
                tp2_pnl = abs(trade.entry_price - trade.tp2) / 0.0001
                tp2_pnl *= self.pip_value * trade.position_size * (1 - self.tp1_close_pct)
                trade.pnl += tp2_pnl
                trade.pnl        = round(trade.pnl, 2)
                trade.rr_achieved = self.tp2_rr

    def _calculate_results(self) -> BacktestResult:
        """Calculate summary statistics from completed trades."""
        closed = [t for t in self.trades if t.result != "OPEN"]

        wins     = [t for t in closed if t.result == "WIN"]
        losses   = [t for t in closed if t.result == "LOSS"]
        partials = [t for t in closed if t.result == "PARTIAL"]
        expired  = [t for t in closed if t.result == "EXPIRED"]

        total_profit = sum(t.pnl for t in closed if t.pnl > 0)
        total_loss   = abs(sum(t.pnl for t in closed if t.pnl < 0))
        total_pnl    = sum(t.pnl for t in closed)

        win_rate = len(wins) / len(closed) * 100 if closed else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        avg_win  = total_profit / len(wins)   if wins   else 0
        avg_loss = total_loss   / len(losses) if losses else 0

        pnls = [t.pnl for t in closed]
        best_trade  = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0

        # Max drawdown from equity curve
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        max_dd = float(dd.max()) if len(dd) > 0 else 0
        max_dd_pct = (max_dd / self.initial_balance * 100) if self.initial_balance > 0 else 0

        avg_bars = np.mean([t.bars_held for t in closed]) if closed else 0

        return BacktestResult(
            instrument      = self.instrument,
            start           = self.start,
            end             = self.end,
            initial_balance = self.initial_balance,
            final_balance   = self.balance,
            total_trades    = len(closed),
            wins            = len(wins),
            losses          = len(losses),
            partials        = len(partials),
            expired         = len(expired),
            total_pnl       = round(total_pnl, 2),
            total_profit    = round(total_profit, 2),
            total_loss      = round(total_loss, 2),
            win_rate        = round(win_rate, 1),
            profit_factor   = round(profit_factor, 2),
            avg_win         = round(avg_win, 2),
            avg_loss        = round(avg_loss, 2),
            best_trade      = round(best_trade, 2),
            worst_trade     = round(worst_trade, 2),
            max_drawdown    = round(max_dd, 2),
            max_drawdown_pct= round(max_dd_pct, 1),
            avg_bars_held   = round(float(avg_bars), 1),
            trades          = self.trades,
        )