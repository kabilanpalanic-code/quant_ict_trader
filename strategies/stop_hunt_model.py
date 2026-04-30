"""
strategies/stop_hunt_model.py
==============================
ICT Stop Hunt Model

Rules:
  1. Mark Asian Kill Zone range (8PM-10PM NY)
  2. Wait for price to SWEEP Asian High or Low (close beyond)
  3. Wait for FIRST CHoCH after sweep (close confirms, not just wick)
  4. Wait for BPR to form after CHoCH
  5. Enter at BPR midpoint
  6. SL = beyond the sweep wick
  7. TP = opposite Asian range level

Skip if:
  - No BPR forms after CHoCH
  - Second or later CHoCH
  - CHoCH before sweep
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Literal
from zoneinfo import ZoneInfo

from strategies.asian_range import AsianRange, detect_asian_ranges
from strategies.bpr import BPR, detect_bprs, FVGZone

NY_TZ = ZoneInfo("America/New_York")


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StopHuntSignal:
    """A complete Stop Hunt Model trade signal."""
    timestamp:      pd.Timestamp   # when signal fired (BPR touched)
    direction:      Literal["long", "short"]
    instrument:     str

    # Levels
    entry:          float          # BPR midpoint
    sl:             float          # beyond sweep wick
    tp:             float          # opposite Asian level

    # Context
    asian_range:    AsianRange
    sweep_price:    float          # the extreme wick of the sweep
    sweep_time:     pd.Timestamp
    choch_time:     pd.Timestamp
    bpr:            BPR

    # Risk
    sl_pips:        float
    tp_pips:        float
    rr:             float

    @property
    def is_valid(self) -> bool:
        return self.rr >= 1.0 and self.sl_pips >= 5.0

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  STOP HUNT — {self.direction.upper()} {self.instrument}\n"
            f"{'='*50}\n"
            f"  Entry  : {self.entry:.5f}  (BPR midpoint)\n"
            f"  SL     : {self.sl:.5f}  ({self.sl_pips:.1f} pips)\n"
            f"  TP     : {self.tp:.5f}  ({self.tp_pips:.1f} pips)\n"
            f"  RR     : {self.rr:.2f}:1\n"
            f"  Sweep  : {self.sweep_price:.5f} at {self.sweep_time}\n"
            f"  CHoCH  : {self.choch_time}\n"
            f"  Asian  : H={self.asian_range.high:.5f} L={self.asian_range.low:.5f}\n"
            f"{'='*50}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# STOP HUNT MODEL
# ──────────────────────────────────────────────────────────────────────────────

class StopHuntModel:
    """
    ICT Stop Hunt Model signal generator.

    Parameters
    ----------
    df          : 5min OHLCV DataFrame with UTC DatetimeIndex
    instrument  : e.g. "EURUSD"
    sl_buffer   : extra pips beyond sweep wick for SL
    min_rr      : minimum RR to take signal
    min_bpr_pips: minimum BPR size to qualify
    swing_bars  : bars each side for swing detection (CHoCH)
    """

    COLOUR = dict(
        bg="#0F1117",
        grid="rgba(255,255,255,0.06)",
        up="#26DE81",
        down="#FC5C65",
        asian="#F7B731",
        sweep="#A55EEA",
        choch="#45AAF2",
        bpr="#F7B731",
        sl="#FC5C65",
        tp="#26DE81",
        entry="#FFFFFF",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        instrument: str = "EURUSD",
        sl_buffer_pips: float = 3.0,
        min_rr: float = 1.0,
        min_bpr_pips: float = 1.0,
        swing_bars: int = 3,
    ):
        self._validate(df)
        self.df         = df.copy()
        self.instrument = instrument
        self.sl_buffer  = sl_buffer_pips * 0.0001
        self.min_rr     = min_rr
        self.min_bpr_pips = min_bpr_pips
        self.swing_bars = swing_bars

        self.asian_ranges: list[AsianRange] = []
        self.signals:      list[StopHuntSignal] = []

        self._run()

    # ── public API ─────────────────────────────────────────────────────────────

    def latest_signal(self) -> StopHuntSignal | None:
        valid = [s for s in self.signals if s.is_valid]
        return valid[-1] if valid else None

    def plot(
        self,
        last_n: int | None = 500,
        title: str | None = None,
        height: int = 780,
    ) -> go.Figure:
        """Plot 5min chart with all Stop Hunt signals."""
        df   = self.df.iloc[-last_n:] if last_n else self.df
        title = title or f"{self.instrument} 5min — Stop Hunt Model"
        start_ts = df.index[0]

        fig = make_subplots(rows=1, cols=1)

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing_line_color=self.COLOUR["up"],
            decreasing_line_color=self.COLOUR["down"],
            increasing_fillcolor=self.COLOUR["up"],
            decreasing_fillcolor=self.COLOUR["down"],
            line=dict(width=1), whiskerwidth=0.3,
            showlegend=False, name="OHLC",
        ))

        # Asian ranges
        for ar in self.asian_ranges:
            if ar.end_time < start_ts:
                continue
            # Asian range box
            fig.add_shape(type="rect",
                x0=ar.start_time, x1=df.index[-1],
                y0=ar.low, y1=ar.high,
                fillcolor="rgba(247,183,49,0.06)",
                line=dict(color=self.COLOUR["asian"], width=1, dash="dot"),
            )
            fig.add_annotation(
                x=ar.start_time, y=ar.high,
                text=f"Asian H {ar.high:.5f}",
                font=dict(size=9, color=self.COLOUR["asian"]),
                showarrow=False, xanchor="left", yanchor="bottom",
                bgcolor="rgba(15,17,23,0.7)", borderpad=2,
            )
            fig.add_annotation(
                x=ar.start_time, y=ar.low,
                text=f"Asian L {ar.low:.5f}",
                font=dict(size=9, color=self.COLOUR["asian"]),
                showarrow=False, xanchor="left", yanchor="top",
                bgcolor="rgba(15,17,23,0.7)", borderpad=2,
            )

        # Signals
        for sig in self.signals:
            if sig.timestamp < start_ts or not sig.is_valid:
                continue
            is_long = sig.direction == "long"

            # BPR zone
            fig.add_shape(type="rect",
                x0=sig.bpr.formed_at, x1=df.index[-1],
                y0=sig.bpr.bottom, y1=sig.bpr.top,
                fillcolor="rgba(247,183,49,0.15)",
                line=dict(color=self.COLOUR["bpr"], width=1),
            )

            # Entry line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.entry, y1=sig.entry,
                line=dict(color=self.COLOUR["entry"], width=1.5),
            )

            # SL line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.sl, y1=sig.sl,
                line=dict(color=self.COLOUR["sl"], width=1, dash="dash"),
            )

            # TP line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.tp, y1=sig.tp,
                line=dict(color=self.COLOUR["tp"], width=1.5, dash="dash"),
            )

            # Sweep marker
            fig.add_trace(go.Scatter(
                x=[sig.sweep_time], y=[sig.sweep_price],
                mode="markers",
                marker=dict(
                    symbol="x", size=12,
                    color=self.COLOUR["sweep"],
                    line=dict(color=self.COLOUR["sweep"], width=2),
                ),
                name="Sweep", showlegend=False,
                hovertemplate=f"<b>Sweep</b><br>{sig.sweep_price:.5f}<extra></extra>",
            ))

            # CHoCH marker
            choch_price = df.loc[sig.choch_time, "close"] if sig.choch_time in df.index else sig.entry
            fig.add_trace(go.Scatter(
                x=[sig.choch_time], y=[choch_price],
                mode="markers+text",
                marker=dict(symbol="diamond", size=10, color=self.COLOUR["choch"]),
                text=["CHoCH"],
                textposition="top center" if is_long else "bottom center",
                textfont=dict(size=9, color=self.COLOUR["choch"]),
                name="CHoCH", showlegend=False,
            ))

            # Entry marker
            fig.add_trace(go.Scatter(
                x=[sig.timestamp], y=[sig.entry],
                mode="markers",
                marker=dict(
                    symbol="triangle-up" if is_long else "triangle-down",
                    size=14, color=self.COLOUR["up"] if is_long else self.COLOUR["down"],
                    line=dict(color="white", width=1.5),
                ),
                name="Entry", showlegend=False,
                hovertemplate=(
                    f"<b>{'LONG' if is_long else 'SHORT'}</b><br>"
                    f"Entry: {sig.entry:.5f}<br>"
                    f"SL: {sig.sl:.5f}<br>"
                    f"TP: {sig.tp:.5f}<br>"
                    f"RR: {sig.rr:.2f}:1<extra></extra>"
                ),
            ))

            # Annotations
            fig.add_annotation(
                x=sig.timestamp, y=sig.entry,
                text=f"  {'L' if is_long else 'S'} {sig.rr:.1f}R",
                font=dict(size=10, color=self.COLOUR["up"] if is_long else self.COLOUR["down"]),
                showarrow=False, xanchor="left",
                bgcolor="rgba(15,17,23,0.8)", borderpad=2,
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color="#E8E8E8"), x=0.01),
            paper_bgcolor=self.COLOUR["bg"],
            plot_bgcolor=self.COLOUR["bg"],
            height=height,
            margin=dict(l=60, r=20, t=48, b=20),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8"),
        )
        axis_style = dict(
            gridcolor=self.COLOUR["grid"],
            zerolinecolor=self.COLOUR["grid"],
            tickfont=dict(color="#AAA", size=10),
            showspikes=True, spikecolor="#555", spikethickness=1,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style, tickformat=".5f")
        return fig

    def summary(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame()
        rows = []
        for s in self.signals:
            rows.append({
                "timestamp":  s.timestamp,
                "direction":  s.direction,
                "entry":      s.entry,
                "sl":         s.sl,
                "tp":         s.tp,
                "sl_pips":    s.sl_pips,
                "tp_pips":    s.tp_pips,
                "rr":         s.rr,
                "valid":      s.is_valid,
            })
        return pd.DataFrame(rows)

    # ── private pipeline ───────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex.")

    def _run(self) -> None:
        self.asian_ranges = detect_asian_ranges(self.df)
        for ar in self.asian_ranges:
            self._process_range(ar)

    def _process_range(self, ar: AsianRange) -> None:
        df = self.df

        # Only look at bars within same trading day (max 200 bars = ~16 hours on 5min)
        after_asian = df[
            (df.index > ar.end_time) &
            (df.index <= ar.end_time + pd.Timedelta(hours=16))
        ]
        if len(after_asian) < 10:
            return

        highs  = after_asian["high"].values
        lows   = after_asian["low"].values
        closes = after_asian["close"].values

        sweep_found = False
        sweep_dir   = None
        sweep_price = 0.0
        sweep_time  = None
        sweep_bar   = 0
        choch_found = False
        choch_time  = None
        choch_bar   = 0

        for i, (bar_time, row) in enumerate(after_asian.iterrows()):
            # Step 1: Sweep
            if not sweep_found:
                if row["close"] > ar.high:
                    sweep_found = True
                    sweep_dir   = "high"
                    sweep_price = highs[i]
                    sweep_time  = bar_time
                    sweep_bar   = i
                elif row["close"] < ar.low:
                    sweep_found = True
                    sweep_dir   = "low"
                    sweep_price = lows[i]
                    sweep_time  = bar_time
                    sweep_bar   = i
                continue

            # Step 2: First CHoCH
            if not choch_found:
                if sweep_dir == "high":
                    recent_low = self._recent_swing_low(after_asian, i, lookback=20)
                    if recent_low and row["close"] < recent_low:
                        choch_found = True
                        choch_time  = bar_time
                        choch_bar   = i
                elif sweep_dir == "low":
                    recent_high = self._recent_swing_high(after_asian, i, lookback=20)
                    if recent_high and row["close"] > recent_high:
                        choch_found = True
                        choch_time  = bar_time
                        choch_bar   = i
                continue

            # Step 3: BPR — run ONCE on slice after CHoCH (max 50 bars)
            if choch_found:
                end_bar   = min(choch_bar + 50, len(after_asian))
                df_window = after_asian.iloc[choch_bar:end_bar]
                bprs      = detect_bprs(df_window, self.min_bpr_pips)

                current_price = row["close"]
                valid_bprs = []
                for bpr in bprs:
                    if bpr.filled:
                        continue
                    if sweep_dir == "low" and bpr.midpoint < current_price:
                        valid_bprs.append(bpr)
                    elif sweep_dir == "high" and bpr.midpoint > current_price:
                        valid_bprs.append(bpr)

                if not valid_bprs:
                    return  # no BPR — skip this range

                bpr = sorted(valid_bprs, key=lambda b: b.formed_at)[-1]
                direction = "long" if sweep_dir == "low" else "short"
                entry     = bpr.midpoint

                if direction == "long":
                    sl = sweep_price - self.sl_buffer
                    tp = ar.high
                else:
                    sl = sweep_price + self.sl_buffer
                    tp = ar.low

                sl_pips = abs(entry - sl) / 0.0001
                tp_pips = abs(entry - tp) / 0.0001
                rr      = tp_pips / sl_pips if sl_pips > 0 else 0

                self.signals.append(StopHuntSignal(
                    timestamp   = bar_time,
                    direction   = direction,
                    instrument  = self.instrument,
                    entry       = round(entry, 5),
                    sl          = round(sl, 5),
                    tp          = round(tp, 5),
                    asian_range = ar,
                    sweep_price = round(sweep_price, 5),
                    sweep_time  = sweep_time,
                    choch_time  = choch_time,
                    bpr         = bpr,
                    sl_pips     = round(sl_pips, 1),
                    tp_pips     = round(tp_pips, 1),
                    rr          = round(rr, 2),
                ))
                return  # one signal per Asian range

    def _recent_swing_high(
        self,
        df: pd.DataFrame,
        current_bar: int,
        lookback: int = 20,
    ) -> float | None:
        """Find most recent swing high before current bar."""
        n = self.swing_bars
        start = max(0, current_bar - lookback)
        highs = df["high"].values

        for i in range(current_bar - 1, start + n, -1):
            if i + n >= len(highs) or i - n < 0:
                continue
            window = highs[i-n : i+n+1]
            if highs[i] == window.max():
                return float(highs[i])
        return None

    def _recent_swing_low(
        self,
        df: pd.DataFrame,
        current_bar: int,
        lookback: int = 20,
    ) -> float | None:
        """Find most recent swing low before current bar."""
        n = self.swing_bars
        start = max(0, current_bar - lookback)
        lows = df["low"].values

        for i in range(current_bar - 1, start + n, -1):
            if i + n >= len(lows) or i - n < 0:
                continue
            window = lows[i-n : i+n+1]
            if lows[i] == window.min():
                return float(lows[i])
        return None