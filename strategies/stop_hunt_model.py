"""
strategies/stop_hunt_model.py
==============================
ICT Stop Hunt Model — Clean rewrite

Rules:
  SHORT:
    1. Price sweeps ABOVE Asian High (close beyond)
    2. First BEARISH CHoCH after sweep (close breaks below recent swing low)
    3. FVG or BPR must be ABOVE Asian High
    4. Enter at zone midpoint
    5. SL = above swing high after CHoCH
    6. TP = Asian Low

  LONG:
    1. Price sweeps BELOW Asian Low (close beyond)
    2. First BULLISH CHoCH after sweep (close breaks above recent swing high)
    3. FVG or BPR must be BELOW Asian Low
    4. Enter at zone midpoint
    5. SL = below swing low after CHoCH
    6. TP = Asian High

Skip if:
  - Asian range < min_asian_pips
  - No sweep
  - No CHoCH after sweep
  - No FVG/BPR in correct location
  - Signal not fired before NY Morning end (10AM NY)
  - No carry forward to next day
"""

from __future__ import annotations

import datetime
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo

from strategies.asian_range import AsianRange, detect_asian_ranges
from strategies.bpr import detect_bprs, detect_fvgs, BPR, FVGZone

NY_TZ = ZoneInfo("America/New_York")


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class StopHuntSignal:
    timestamp:   pd.Timestamp
    direction:   Literal["long", "short"]
    instrument:  str
    entry:       float
    sl:          float
    tp:          float
    sl_pips:     float
    tp_pips:     float
    rr:          float
    asian_range: AsianRange
    sweep_price: float
    sweep_time:  pd.Timestamp
    choch_time:  pd.Timestamp
    zone_type:   str   # "BPR" or "FVG"
    zone_top:    float
    zone_bottom: float

    @property
    def is_valid(self) -> bool:
        return self.rr >= 1.0 and self.sl_pips >= 5.0

    @property
    def zone_midpoint(self) -> float:
        return (self.zone_top + self.zone_bottom) / 2

    def __str__(self) -> str:
        return (
            f"\n{'='*52}\n"
            f"  STOP HUNT — {self.direction.upper()} {self.instrument}\n"
            f"{'='*52}\n"
            f"  Entry  : {self.entry:.5f}  ({self.zone_type})\n"
            f"  SL     : {self.sl:.5f}  ({self.sl_pips:.1f} pips)\n"
            f"  TP     : {self.tp:.5f}  ({self.tp_pips:.1f} pips)\n"
            f"  RR     : {self.rr:.2f}:1\n"
            f"  Sweep  : {self.sweep_price:.5f} at {self.sweep_time}\n"
            f"  CHoCH  : {self.choch_time}\n"
            f"  Zone   : {self.zone_bottom:.5f} - {self.zone_top:.5f}\n"
            f"  Asian  : H={self.asian_range.high:.5f} L={self.asian_range.low:.5f}\n"
            f"{'='*52}"
        )


# ── Model ──────────────────────────────────────────────────────────────────────

class StopHuntModel:

    COLOUR = dict(
        bg="#0F1117", grid="rgba(255,255,255,0.06)",
        up="#26DE81", down="#FC5C65",
        asian="#F7B731", sweep="#A55EEA",
        choch="#45AAF2", zone="#F7B731",
        sl="#FC5C65", tp="#26DE81",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        instrument: str = "EURUSD",
        sl_buffer_pips: float = 3.0,
        min_rr: float = 1.0,
        min_asian_pips: float = 10.0,
        swing_bars: int = 3,
        min_zone_pips: float = 0.5,
    ):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")
        required = {"open", "high", "low", "close"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")

        self.df             = df.copy()
        self.instrument     = instrument
        self.sl_buffer      = sl_buffer_pips * 0.0001
        self.min_rr         = min_rr
        self.min_asian_pips = min_asian_pips
        self.swing_bars     = swing_bars
        self.min_zone_pips  = min_zone_pips

        self.asian_ranges: list[AsianRange] = []
        self.signals:      list[StopHuntSignal] = []

        self._run()

    # ── public ─────────────────────────────────────────────────────────────────

    def latest_signal(self) -> StopHuntSignal | None:
        valid = [s for s in self.signals if s.is_valid]
        return valid[-1] if valid else None

    def summary(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": s.timestamp,
            "direction": s.direction,
            "entry":     s.entry,
            "sl":        s.sl,
            "tp":        s.tp,
            "sl_pips":   s.sl_pips,
            "tp_pips":   s.tp_pips,
            "rr":        s.rr,
            "zone":      s.zone_type,
            "valid":     s.is_valid,
        } for s in self.signals])

    def plot(self, last_n: int = 2000, title: str | None = None, height: int = 820) -> go.Figure:
        df       = self.df.iloc[-last_n:] if last_n else self.df
        title    = title or f"{self.instrument} 5min — Stop Hunt Model"
        start_ts = df.index[0]
        end_ts   = df.index[-1]

        fig = go.Figure()

        # Session shading
        df_ny = df.copy()
        df_ny.index = df_ny.index.tz_convert(NY_TZ)
        dates_seen = set()
        for ts in df_ny.index:
            d = ts.date()
            if d in dates_seen:
                continue
            dates_seen.add(d)
            asian_s = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                   hour=20, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            asian_e = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                   hour=22, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            if asian_s >= start_ts and asian_s <= end_ts:
                fig.add_vrect(x0=asian_s, x1=asian_e,
                    fillcolor="rgba(247,183,49,0.08)", layer="below", line_width=0,
                    annotation_text="Asian KZ", annotation_position="top left",
                    annotation_font=dict(size=9, color="#F7B731"))
            nd = d + datetime.timedelta(days=1)
            ny_s = pd.Timestamp(year=nd.year, month=nd.month, day=nd.day,
                                hour=7, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            ny_e = pd.Timestamp(year=nd.year, month=nd.month, day=nd.day,
                                hour=10, minute=0, tzinfo=NY_TZ).tz_convert("UTC")
            if ny_s >= start_ts and ny_s <= end_ts:
                fig.add_vrect(x0=ny_s, x1=ny_e,
                    fillcolor="rgba(69,170,242,0.06)", layer="below", line_width=0,
                    annotation_text="NY AM", annotation_position="top left",
                    annotation_font=dict(size=9, color="#45AAF2"))

        # Candles
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color=self.COLOUR["up"],
            decreasing_line_color=self.COLOUR["down"],
            increasing_fillcolor=self.COLOUR["up"],
            decreasing_fillcolor=self.COLOUR["down"],
            line=dict(width=1), whiskerwidth=0.3, showlegend=False))

        # Asian range levels — expire at next Asian KZ
        for i, ar in enumerate(self.asian_ranges):
            if ar.end_time < start_ts:
                continue
            next_start = end_ts
            if i + 1 < len(self.asian_ranges):
                next_start = min(self.asian_ranges[i+1].start_time, end_ts)
            for y, label in [(ar.high, f"Asian H {ar.high:.5f}"),
                             (ar.low,  f"Asian L {ar.low:.5f}")]:
                fig.add_shape(type="line", x0=ar.end_time, x1=next_start,
                    y0=y, y1=y, line=dict(color=self.COLOUR["asian"], width=1.5, dash="dot"))
                fig.add_annotation(x=ar.end_time, y=y, text=label,
                    font=dict(size=9, color=self.COLOUR["asian"]),
                    showarrow=False, xanchor="left",
                    yanchor="bottom" if y == ar.high else "top",
                    bgcolor="rgba(15,17,23,0.8)", borderpad=2)

        # Signals
        for sig in self.signals:
            if sig.timestamp < start_ts or not sig.is_valid:
                continue
            is_long = sig.direction == "long"
            ec = self.COLOUR["up"] if is_long else self.COLOUR["down"]
            next_ar = next((a for a in self.asian_ranges
                           if a.start_time > sig.asian_range.start_time), None)
            sig_end = next_ar.start_time if next_ar and next_ar.start_time <= end_ts else end_ts

            # Entry zone
            fig.add_shape(type="rect", x0=sig.choch_time, x1=sig_end,
                y0=sig.zone_bottom, y1=sig.zone_top,
                fillcolor="rgba(247,183,49,0.2)", line=dict(color=self.COLOUR["zone"], width=1.5))
            fig.add_annotation(x=sig.choch_time, y=sig.zone_top,
                text=sig.zone_type, font=dict(size=10, color=self.COLOUR["zone"]),
                showarrow=False, xanchor="left", yanchor="bottom",
                bgcolor="rgba(15,17,23,0.8)", borderpad=2)

            # Entry / SL / TP lines
            for y, col, lbl, dash, anchor in [
                (sig.entry, ec,                  f"ENTRY {sig.entry:.5f}", "solid", "bottom" if is_long else "top"),
                (sig.sl,    self.COLOUR["sl"],   f"SL {sig.sl:.5f} ({sig.sl_pips:.0f}p)", "dash", "top" if is_long else "bottom"),
                (sig.tp,    self.COLOUR["tp"],   f"TP {sig.tp:.5f} ({sig.tp_pips:.0f}p) RR:{sig.rr}", "dash", "bottom" if is_long else "top"),
            ]:
                fig.add_shape(type="line", x0=sig.timestamp, x1=sig_end,
                    y0=y, y1=y, line=dict(color=col, width=2 if dash=="solid" else 1.5,
                    dash="solid" if dash=="solid" else "dash"))
                fig.add_annotation(x=sig_end, y=y, text=lbl,
                    font=dict(size=10, color=col), showarrow=False,
                    xanchor="right", yanchor=anchor,
                    bgcolor="rgba(15,17,23,0.85)", borderpad=3,
                    bordercolor=col if dash=="solid" else "rgba(0,0,0,0)", borderwidth=1)

            # Sweep marker
            fig.add_trace(go.Scatter(x=[sig.sweep_time], y=[sig.sweep_price],
                mode="markers+text",
                marker=dict(symbol="x", size=14, color=self.COLOUR["sweep"],
                            line=dict(color=self.COLOUR["sweep"], width=2.5)),
                text=["SWEEP"], textposition="top right",
                textfont=dict(size=10, color=self.COLOUR["sweep"]), showlegend=False))

            # CHoCH marker
            choch_price = df["close"].get(sig.choch_time, sig.entry)
            fig.add_trace(go.Scatter(x=[sig.choch_time], y=[choch_price],
                mode="markers+text",
                marker=dict(symbol="diamond", size=12, color=self.COLOUR["choch"],
                            line=dict(color="white", width=1)),
                text=["CHoCH"], textposition="top center" if is_long else "bottom center",
                textfont=dict(size=10, color=self.COLOUR["choch"]), showlegend=False))

            # Entry marker
            fig.add_trace(go.Scatter(x=[sig.timestamp], y=[sig.entry],
                mode="markers",
                marker=dict(symbol="triangle-up" if is_long else "triangle-down",
                            size=16, color=ec, line=dict(color="white", width=2)),
                showlegend=False))

        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color="#E8E8E8"), x=0.01),
            paper_bgcolor=self.COLOUR["bg"], plot_bgcolor=self.COLOUR["bg"],
            height=height, margin=dict(l=60, r=20, t=48, b=20),
            xaxis_rangeslider_visible=False, hovermode="x unified",
            hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8"))
        axis = dict(gridcolor=self.COLOUR["grid"], zerolinecolor=self.COLOUR["grid"],
                    tickfont=dict(color="#AAA", size=10),
                    showspikes=True, spikecolor="#555", spikethickness=1)
        fig.update_xaxes(**axis)
        fig.update_yaxes(**axis, tickformat=".5f")
        return fig

    # ── private ────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        self.asian_ranges = detect_asian_ranges(self.df)
        for ar in self.asian_ranges:
            self._process_range(ar)

    def _process_range(self, ar: AsianRange) -> None:
        # Skip small Asian ranges
        if ar.size_pips < self.min_asian_pips:
            return

        # Window: after Asian KZ (10PM NY) to NY Morning end (10AM NY next day)
        ar_end_ny  = ar.end_time.astimezone(NY_TZ)
        next_day   = ar_end_ny.date() + datetime.timedelta(days=1)
        ny_end_utc = pd.Timestamp(
            year=next_day.year, month=next_day.month, day=next_day.day,
            hour=10, minute=0, tzinfo=NY_TZ).tz_convert("UTC")

        after = self.df[
            (self.df.index > ar.end_time) &
            (self.df.index <= ny_end_utc)
        ]
        if len(after) < 10:
            return

        # ── Step 1: Find sweep ─────────────────────────────────────────────────
        sweep_dir   = None
        sweep_bar   = None
        sweep_price = None
        sweep_time  = None

        for i, (ts, row) in enumerate(after.iterrows()):
            if row["close"] > ar.high:
                sweep_dir   = "high"
                sweep_bar   = i
                sweep_price = after["high"].iloc[i]
                sweep_time  = ts
                break
            elif row["close"] < ar.low:
                sweep_dir   = "low"
                sweep_bar   = i
                sweep_price = after["low"].iloc[i]
                sweep_time  = ts
                break

        if sweep_dir is None:
            return

        # ── Step 2: First CHoCH after sweep ───────────────────────────────────
        # SHORT setup → first BEARISH CHoCH (close below recent swing low)
        # LONG  setup → first BULLISH CHoCH (close above recent swing high)
        after_sweep = after.iloc[sweep_bar + 1:]
        choch_bar   = None
        choch_time  = None

        for i, (ts, row) in enumerate(after_sweep.iterrows()):
            if sweep_dir == "high":
                # Bearish CHoCH — close breaks below recent swing low
                recent_low = self._swing_low(after_sweep, i)
                if recent_low and row["close"] < recent_low:
                    choch_bar  = i
                    choch_time = ts
                    break
            else:
                # Bullish CHoCH — close breaks above recent swing high
                recent_high = self._swing_high(after_sweep, i)
                if recent_high and row["close"] > recent_high:
                    choch_bar  = i
                    choch_time = ts
                    break

        if choch_bar is None:
            return

        # ── Step 3: Find entry zone (BPR or FVG) in correct location ──────────
        # SHORT: BEARISH FVG/BPR ABOVE Asian High
        # LONG:  BULLISH FVG/BPR BELOW Asian Low
        zone       = None
        zone_type  = None
        min_gap    = self.min_zone_pips * 0.0001

        # Try BPR first (stronger signal)
        # BPR must contain the correct FVG direction
        bprs = detect_bprs(after, min_gap)
        for bpr in bprs:
            if sweep_dir == "high":
                # SHORT → BPR must be above Asian High AND contain bearish FVG
                if (bpr.midpoint >= ar.high and
                        bpr.bearish_fvg.top >= ar.high):
                    zone      = (bpr.bottom, bpr.top)
                    zone_type = "BPR"
                    break
            elif sweep_dir == "low":
                # LONG → BPR must be below Asian Low AND contain bullish FVG
                if (bpr.midpoint <= ar.low and
                        bpr.bullish_fvg.bottom <= ar.low):
                    zone      = (bpr.bottom, bpr.top)
                    zone_type = "BPR"
                    break

        # Fallback to single FVG if no valid BPR
        if zone is None:
            fvgs = detect_fvgs(after, min_gap)
            for fvg in fvgs:
                mid = (fvg.top + fvg.bottom) / 2
                if sweep_dir == "high" and fvg.kind == "bearish" and mid >= ar.high:
                    zone      = (fvg.bottom, fvg.top)
                    zone_type = "FVG"
                    break
                elif sweep_dir == "low" and fvg.kind == "bullish" and mid <= ar.low:
                    zone      = (fvg.bottom, fvg.top)
                    zone_type = "FVG"
                    break

        if zone is None:
            return

        zone_bottom, zone_top = zone
        entry = (zone_top + zone_bottom) / 2

        # ── Step 4: SL from swing high/low after CHoCH ────────────────────────
        after_choch = after_sweep.iloc[choch_bar:choch_bar + 20]
        buf = self.sl_buffer

        if sweep_dir == "low":  # LONG
            lows = after_choch["low"].values
            swings = [lows[k] for k in range(self.swing_bars, len(lows) - self.swing_bars)
                      if lows[k] == lows[k-self.swing_bars:k+self.swing_bars+1].min()]
            sl = (swings[-1] - buf) if swings else (after_choch["low"].min() - buf)
            if abs(entry - sl) / 0.0001 < 5:
                sl = sweep_price - buf
            tp = ar.high

        else:  # SHORT
            highs = after_choch["high"].values
            swings = [highs[k] for k in range(self.swing_bars, len(highs) - self.swing_bars)
                      if highs[k] == highs[k-self.swing_bars:k+self.swing_bars+1].max()]
            sl = (swings[-1] + buf) if swings else (after_choch["high"].max() + buf)
            if abs(entry - sl) / 0.0001 < 5:
                sl = sweep_price + buf
            tp = ar.low

        sl_pips = abs(entry - sl) / 0.0001
        tp_pips = abs(entry - tp) / 0.0001
        rr      = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0

        direction = "long" if sweep_dir == "low" else "short"

        # Signal timestamp = CHoCH bar (that's when we know setup is valid)
        self.signals.append(StopHuntSignal(
            timestamp   = choch_time,
            direction   = direction,
            instrument  = self.instrument,
            entry       = round(entry, 5),
            sl          = round(sl, 5),
            tp          = round(tp, 5),
            sl_pips     = round(sl_pips, 1),
            tp_pips     = round(tp_pips, 1),
            rr          = rr,
            asian_range = ar,
            sweep_price = round(sweep_price, 5),
            sweep_time  = sweep_time,
            choch_time  = choch_time,
            zone_type   = zone_type,
            zone_top    = round(zone_top, 5),
            zone_bottom = round(zone_bottom, 5),
        ))

    def _swing_high(self, df: pd.DataFrame, current: int, lookback: int = 20) -> float | None:
        n     = self.swing_bars
        start = max(0, current - lookback)
        highs = df["high"].values
        for i in range(current - 1, start + n - 1, -1):
            if i + n >= len(highs) or i - n < 0:
                continue
            if highs[i] == highs[i-n:i+n+1].max():
                return float(highs[i])
        return None

    def _swing_low(self, df: pd.DataFrame, current: int, lookback: int = 20) -> float | None:
        n     = self.swing_bars
        start = max(0, current - lookback)
        lows  = df["low"].values
        for i in range(current - 1, start + n - 1, -1):
            if i + n >= len(lows) or i - n < 0:
                continue
            if lows[i] == lows[i-n:i+n+1].min():
                return float(lows[i])
        return None