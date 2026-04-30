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
        height: int = 820,
    ) -> go.Figure:
        """Plot 5min chart with session zones and Stop Hunt signals."""
        df    = self.df.iloc[-last_n:] if last_n else self.df
        title = title or f"{self.instrument} 5min — Stop Hunt Model"
        start_ts = df.index[0]
        end_ts   = df.index[-1]

        fig = make_subplots(rows=1, cols=1)

        # ── Session zone shading ──────────────────────────────────────────────
        # Walk through each day and shade Asian KZ and NY Morning session
        df_ny = df.copy()
        df_ny.index = df_ny.index.tz_convert(NY_TZ)

        dates_seen = set()
        for ts in df_ny.index:
            d = ts.date()
            if d in dates_seen:
                continue
            dates_seen.add(d)

            # Asian Kill Zone: 8PM-10PM NY
            asian_start = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                       hour=20, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            asian_end   = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                       hour=22, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            if asian_start >= start_ts and asian_start <= end_ts:
                fig.add_vrect(x0=asian_start, x1=asian_end,
                    fillcolor="rgba(247,183,49,0.08)",
                    layer="below", line_width=0,
                    annotation_text="Asian KZ",
                    annotation_position="top left",
                    annotation_font=dict(size=9, color="#F7B731"),
                )

            # NY Morning session: 7AM-10AM NY (next day after Asian)
            import datetime
            next_d = d + datetime.timedelta(days=1)
            ny_start = pd.Timestamp(year=next_d.year, month=next_d.month, day=next_d.day,
                                    hour=7, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            ny_end   = pd.Timestamp(year=next_d.year, month=next_d.month, day=next_d.day,
                                    hour=10, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            if ny_start >= start_ts and ny_start <= end_ts:
                fig.add_vrect(x0=ny_start, x1=ny_end,
                    fillcolor="rgba(69,170,242,0.06)",
                    layer="below", line_width=0,
                    annotation_text="NY Morning",
                    annotation_position="top left",
                    annotation_font=dict(size=9, color="#45AAF2"),
                )

        # ── Candlesticks ──────────────────────────────────────────────────────
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

        # ── Asian range levels ────────────────────────────────────────────────
        for idx_ar, ar in enumerate(self.asian_ranges):
            if ar.end_time < start_ts:
                continue

            # Line ends at next Asian KZ start or chart end
            next_asian_start = end_ts
            if idx_ar + 1 < len(self.asian_ranges):
                next_asian_start = min(
                    self.asian_ranges[idx_ar + 1].start_time,
                    end_ts
                )

            fig.add_shape(type="line",
                x0=ar.end_time, x1=next_asian_start,
                y0=ar.high, y1=ar.high,
                line=dict(color=self.COLOUR["asian"], width=1.5, dash="dot"),
            )
            fig.add_shape(type="line",
                x0=ar.end_time, x1=next_asian_start,
                y0=ar.low, y1=ar.low,
                line=dict(color=self.COLOUR["asian"], width=1.5, dash="dot"),
            )
            fig.add_annotation(x=ar.end_time, y=ar.high,
                text=f"Asian H {ar.high:.5f}",
                font=dict(size=10, color=self.COLOUR["asian"]),
                showarrow=False, xanchor="left", yanchor="bottom",
                bgcolor="rgba(15,17,23,0.8)", borderpad=3,
                bordercolor=self.COLOUR["asian"], borderwidth=1,
            )
            fig.add_annotation(x=ar.end_time, y=ar.low,
                text=f"Asian L {ar.low:.5f}",
                font=dict(size=10, color=self.COLOUR["asian"]),
                showarrow=False, xanchor="left", yanchor="top",
                bgcolor="rgba(15,17,23,0.8)", borderpad=3,
                bordercolor=self.COLOUR["asian"], borderwidth=1,
            )

        # ── Signals ───────────────────────────────────────────────────────────
        for sig in self.signals:
            if sig.timestamp < start_ts or not sig.is_valid:
                continue
            is_long = sig.direction == "long"
            ec = self.COLOUR["up"] if is_long else self.COLOUR["down"]

            # Signal expires at next Asian KZ start
            next_ar = next((a for a in self.asian_ranges
                           if a.start_time > sig.asian_range.start_time), None)
            sig_end = next_ar.start_time if next_ar and next_ar.start_time <= end_ts else end_ts

            # BPR zone
            fig.add_shape(type="rect",
                x0=sig.bpr.formed_at, x1=sig_end,
                y0=sig.bpr.bottom, y1=sig.bpr.top,
                fillcolor="rgba(247,183,49,0.2)",
                line=dict(color=self.COLOUR["bpr"], width=1.5),
            )
            fig.add_annotation(x=sig.bpr.formed_at, y=sig.bpr.top,
                text="BPR",
                font=dict(size=10, color=self.COLOUR["bpr"]),
                showarrow=False, xanchor="left", yanchor="bottom",
                bgcolor="rgba(15,17,23,0.8)", borderpad=2,
            )

            # Entry line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=sig_end,
                y0=sig.entry, y1=sig.entry,
                line=dict(color=ec, width=2),
            )
            fig.add_annotation(x=sig_end, y=sig.entry,
                text=f"ENTRY {sig.entry:.5f}",
                font=dict(size=11, color=ec),
                showarrow=False, xanchor="right",
                yanchor="bottom" if is_long else "top",
                bgcolor="rgba(15,17,23,0.85)", borderpad=3,
                bordercolor=ec, borderwidth=1,
            )

            # SL line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=sig_end,
                y0=sig.sl, y1=sig.sl,
                line=dict(color=self.COLOUR["sl"], width=1.5, dash="dash"),
            )
            fig.add_annotation(x=sig_end, y=sig.sl,
                text=f"SL {sig.sl:.5f} ({sig.sl_pips:.0f}p)",
                font=dict(size=11, color=self.COLOUR["sl"]),
                showarrow=False, xanchor="right",
                yanchor="top" if is_long else "bottom",
                bgcolor="rgba(15,17,23,0.85)", borderpad=3,
            )

            # TP line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=sig_end,
                y0=sig.tp, y1=sig.tp,
                line=dict(color=self.COLOUR["tp"], width=2, dash="dash"),
            )
            fig.add_annotation(x=sig_end, y=sig.tp,
                text=f"TP {sig.tp:.5f} ({sig.tp_pips:.0f}p) RR:{sig.rr}",
                font=dict(size=11, color=self.COLOUR["tp"]),
                showarrow=False, xanchor="right",
                yanchor="bottom" if is_long else "top",
                bgcolor="rgba(15,17,23,0.85)", borderpad=3,
            )

            # Risk/reward zones
            fig.add_shape(type="rect",
                x0=sig.timestamp, x1=sig_end,
                y0=min(sig.entry, sig.sl), y1=max(sig.entry, sig.sl),
                fillcolor="rgba(252,92,101,0.08)", line=dict(width=0),
            )
            fig.add_shape(type="rect",
                x0=sig.timestamp, x1=sig_end,
                y0=min(sig.entry, sig.tp), y1=max(sig.entry, sig.tp),
                fillcolor="rgba(38,222,129,0.06)", line=dict(width=0),
            )

            # Sweep marker
            fig.add_trace(go.Scatter(
                x=[sig.timestamp], y=[sig.entry],
                mode="markers",
                marker=dict(
                    symbol="triangle-up" if is_long else "triangle-down",
                    size=16, color=ec,
                    line=dict(color="white", width=2),
                ),
                showlegend=False,
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

        # Search window: Asian close to 10 AM NY NEXT DAY (end of NY Morning session)
        # Asian ends 10 PM NY → next day 10 AM NY = 12 hours window
        ar_end_ny = ar.end_time.astimezone(NY_TZ)
        next_day  = ar_end_ny.date() + pd.Timedelta(days=1)
        ny_morning_end = pd.Timestamp(
            year=next_day.year, month=next_day.month, day=next_day.day,
            hour=10, minute=0, tzinfo=NY_TZ
        )
        ny_morning_end_utc = ny_morning_end.tz_convert("UTC")

        after_asian = df[
            (df.index > ar.end_time) &
            (df.index <= ny_morning_end_utc)
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

            # Step 3: BPR — detect on full window after Asian close
            if choch_found:
                end_bar   = min(50, len(after_asian))
                df_window = after_asian.iloc[:end_bar]
                bprs      = detect_bprs(df_window, self.min_bpr_pips)

                current_price = row["close"]
                valid_bprs = []
                for bpr in bprs:
                    if bpr.filled:
                        continue
                    # Long (swept low) → BPR must be ABOVE sweep wick
                    # AND below or at Asian Low (pullback into value zone)
                    if sweep_dir == "low":
                        if bpr.midpoint > sweep_price and bpr.midpoint <= ar.low:
                            valid_bprs.append(bpr)
                    # Short (swept high) → BPR must be BELOW sweep wick
                    # AND above or at Asian High (pullback into value zone)
                    elif sweep_dir == "high":
                        if bpr.midpoint < sweep_price and bpr.midpoint >= ar.high:
                            valid_bprs.append(bpr)

                if not valid_bprs:
                    return

                bpr = sorted(valid_bprs, key=lambda b: b.formed_at)[-1]
                direction = "long" if sweep_dir == "low" else "short"
                entry     = bpr.midpoint

                # SL = beyond the FULL sweep candle wick + buffer
                # Use the extreme wick of the sweep candle, not just close
                sweep_candle = after_asian.iloc[sweep_bar]
                if direction == "long":
                    sl = sweep_candle["low"] - self.sl_buffer   # below sweep wick
                    tp = ar.high
                else:
                    sl = sweep_candle["high"] + self.sl_buffer  # above sweep wick
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