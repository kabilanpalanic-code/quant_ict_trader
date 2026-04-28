"""
strategies/market_structure.py
================================
ICT Market Structure Analysis Module

Concepts implemented:
  - Swing High / Swing Low detection (fractal-based)
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
  - Order Block (OB) marking — bullish & bearish
  - Interactive Plotly chart for Jupyter Lab

Author : Quant ICT Trader
Python : 3.12+
Deps   : pandas, numpy, plotly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Literal

# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]
    confirmed: bool = False          # True once the next opposite swing forms


@dataclass
class StructureEvent:
    """Represents a BOS or CHoCH event."""
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["BOS_bull", "BOS_bear", "CHoCH_bull", "CHoCH_bear"]
    broken_swing_price: float        # the swing level that was breached


@dataclass
class OrderBlock:
    """
    ICT Order Block — the last opposing candle before an impulse move.
    
    Bullish OB  : last bearish candle before a bullish impulse that breaks structure.
    Bearish OB  : last bullish candle before a bearish impulse that breaks structure.
    """
    index: int
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    kind: Literal["bullish", "bearish"]
    mitigated: bool = False          # True once price returns into the OB zone
    mitigated_at: pd.Timestamp | None = None


# ──────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS CLASS
# ──────────────────────────────────────────────────────────────────────────────

class MarketStructure:
    """
    Detects ICT market structure elements on an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close (+ optionally volume).
        Index should be DatetimeIndex.
    swing_lookback : int
        Number of bars each side to confirm a fractal swing.
        Default 5 — good for 1H/4H EURUSD.
    """

    # colour palette (Plotly-friendly)
    COLOUR = dict(
        swing_high="#F7B731",
        swing_low="#45AAF2",
        bos_bull="#26DE81",
        bos_bear="#FC5C65",
        choch_bull="#FD9644",
        choch_bear="#A55EEA",
        ob_bull="rgba(38,222,129,0.18)",
        ob_bear="rgba(252,92,101,0.18)",
        ob_bull_border="#26DE81",
        ob_bear_border="#FC5C65",
        structure_line="rgba(255,255,255,0.25)",
        bg="#0F1117",
        grid="rgba(255,255,255,0.06)",
        candle_up="#26DE81",
        candle_down="#FC5C65",
        wick="rgba(255,255,255,0.5)",
    )

    def __init__(self, df: pd.DataFrame, swing_lookback: int = 5):
        self._validate(df)
        self.df = df.copy()
        self.lookback = swing_lookback

        self.swing_highs: list[SwingPoint] = []
        self.swing_lows:  list[SwingPoint] = []
        self.structure_events: list[StructureEvent] = []
        self.order_blocks: list[OrderBlock] = []

        self._run()

    # ── public API ─────────────────────────────────────────────────────────────

    def plot(
        self,
        last_n: int | None = None,
        title: str = "ICT Market Structure",
        height: int = 780,
        show_volume: bool = True,
        max_events: int = 10,
        show_mitigated_ob: bool = False,
    ) -> go.Figure:
        """
        Build and return a Plotly Figure.

        Parameters
        ----------
        last_n            : only show last N candles
        max_events        : max BOS/CHoCH labels shown (most recent only)
        show_mitigated_ob : show OBs that price has already revisited
        """
        df = self.df.iloc[-last_n:] if last_n else self.df
        start_ts = df.index[0]

        rows = 2 if show_volume else 1
        row_heights = [0.78, 0.22] if show_volume else [1.0]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
        )

        # 1. Candlesticks ──────────────────────────────────────────────────────
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"], high=df["high"],
                low=df["low"],  close=df["close"],
                increasing_line_color=self.COLOUR["candle_up"],
                decreasing_line_color=self.COLOUR["candle_down"],
                increasing_fillcolor=self.COLOUR["candle_up"],
                decreasing_fillcolor=self.COLOUR["candle_down"],
                line=dict(width=1),
                whiskerwidth=0.3,
                name="OHLC",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # 2. Structure lines (zig-zag through confirmed swings) ────────────────
        all_swings = sorted(
            self.swing_highs + self.swing_lows,
            key=lambda s: s.index,
        )
        # only show swings within the visible range
        visible_swings = [s for s in all_swings if s.timestamp >= start_ts]
        if len(visible_swings) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp for s in visible_swings],
                    y=[s.price for s in visible_swings],
                    mode="lines",
                    line=dict(color=self.COLOUR["structure_line"], width=1, dash="dot"),
                    name="Structure",
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # 3. Swing markers — single batch trace each (not one trace per swing)
        sh_vis = [s for s in self.swing_highs if s.timestamp >= start_ts]
        sl_vis = [s for s in self.swing_lows  if s.timestamp >= start_ts]

        if sh_vis:
            fig.add_trace(go.Scatter(
                x=[s.timestamp for s in sh_vis],
                y=[s.price * 1.0004 for s in sh_vis],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=8, color=self.COLOUR["swing_high"]),
                text=["SH"] * len(sh_vis),
                textposition="top center",
                textfont=dict(size=8, color=self.COLOUR["swing_high"]),
                name="Swing High",
                showlegend=True,
                hovertemplate="<b>Swing High</b><br>%{x}<br>%{customdata:.5f}<extra></extra>",
                customdata=[s.price for s in sh_vis],
            ), row=1, col=1)

        if sl_vis:
            fig.add_trace(go.Scatter(
                x=[s.timestamp for s in sl_vis],
                y=[s.price * 0.9996 for s in sl_vis],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=8, color=self.COLOUR["swing_low"]),
                text=["SL"] * len(sl_vis),
                textposition="bottom center",
                textfont=dict(size=8, color=self.COLOUR["swing_low"]),
                name="Swing Low",
                showlegend=True,
                hovertemplate="<b>Swing Low</b><br>%{x}<br>%{customdata:.5f}<extra></extra>",
                customdata=[s.price for s in sl_vis],
            ), row=1, col=1)

        # 4. BOS / CHoCH — most recent max_events only, no horizontal lines ────
        _event_colours = {
            "BOS_bull":   self.COLOUR["bos_bull"],
            "BOS_bear":   self.COLOUR["bos_bear"],
            "CHoCH_bull": self.COLOUR["choch_bull"],
            "CHoCH_bear": self.COLOUR["choch_bear"],
        }
        _event_labels = {
            "BOS_bull":   "BOS↑",
            "BOS_bear":   "BOS↓",
            "CHoCH_bull": "CHoCH↑",
            "CHoCH_bear": "CHoCH↓",
        }

        visible_events = [e for e in self.structure_events if e.timestamp >= start_ts]
        visible_events = visible_events[-max_events:]   # keep only most recent

        for ev in visible_events:
            col = _event_colours[ev.kind]
            lbl = _event_labels[ev.kind]
            is_bull = ev.kind.endswith("bull")
            fig.add_annotation(
                x=ev.timestamp,
                y=ev.price * (1.0006 if is_bull else 0.9994),
                text=lbl,
                font=dict(size=10, color=col, family="monospace"),
                showarrow=False,
                bgcolor="rgba(15,17,23,0.82)",
                bordercolor=col,
                borderwidth=1,
                borderpad=3,
                row=1, col=1,
            )

        # 5. Order Blocks — skip mitigated, dedupe nearby price levels ──────────
        seen_levels: set[float] = set()

        obs_to_show = [
            ob for ob in self.order_blocks
            if ob.timestamp >= start_ts
            and (show_mitigated_ob or not ob.mitigated)
        ]

        for ob in obs_to_show:
            mid = round((ob.high + ob.low) / 2, 5)
            if any(abs(mid - seen) < 0.00015 for seen in seen_levels):
                continue
            seen_levels.add(mid)

            is_bull = ob.kind == "bullish"
            fill   = self.COLOUR["ob_bull"] if is_bull else self.COLOUR["ob_bear"]
            border = self.COLOUR["ob_bull_border"] if is_bull else self.COLOUR["ob_bear_border"]
            lbl    = "OB+" if is_bull else "OB-"

            fig.add_shape(
                type="rect",
                x0=ob.timestamp, x1=df.index[-1],
                y0=ob.low, y1=ob.high,
                fillcolor=fill,
                line=dict(color=border, width=1, dash="dot"),
                row=1, col=1,
            )
            fig.add_annotation(
                x=ob.timestamp,
                y=ob.high,
                text=lbl,
                font=dict(size=9, color=border),
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(15,17,23,0.7)",
                borderpad=2,
                row=1, col=1,
            )

        # 6. Volume bars ───────────────────────────────────────────────────────
        if show_volume and "volume" in df.columns:
            vol_colours = [
                self.COLOUR["candle_up"] if c >= o else self.COLOUR["candle_down"]
                for o, c in zip(df["open"], df["close"])
            ]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["volume"],
                    marker_color=vol_colours,
                    marker_line_width=0,
                    name="Volume",
                    showlegend=False,
                    opacity=0.6,
                ),
                row=2, col=1,
            )

        # 7. Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#E8E8E8"), x=0.02),
            paper_bgcolor=self.COLOUR["bg"],
            plot_bgcolor=self.COLOUR["bg"],
            height=height,
            margin=dict(l=60, r=20, t=48, b=20),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8", bordercolor="#333"),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
                font=dict(color="#CCC"),
            ),
        )

        axis_style = dict(
            gridcolor=self.COLOUR["grid"],
            zerolinecolor=self.COLOUR["grid"],
            tickfont=dict(color="#AAA", size=10),
            title_font=dict(color="#888"),
            showspikes=True,
            spikecolor="#555",
            spikethickness=1,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style, tickformat=".5f")

        return fig

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summary of all detected events."""
        rows = []
        for ev in self.structure_events:
            rows.append({
                "timestamp": ev.timestamp,
                "event": ev.kind,
                "price": ev.price,
                "broken_level": ev.broken_swing_price,
            })
        return pd.DataFrame(rows)

    def order_blocks_df(self) -> pd.DataFrame:
        """Return a DataFrame of all order blocks."""
        rows = []
        for ob in self.order_blocks:
            rows.append({
                "timestamp": ob.timestamp,
                "kind":      ob.kind,
                "high":      ob.high,
                "low":       ob.low,
                "open":      ob.open,
                "close":     ob.close,
                "mitigated": ob.mitigated,
                "mitigated_at": ob.mitigated_at,
            })
        return pd.DataFrame(rows)

    # ── private pipeline ───────────────────────────────────────────────────────

    def _run(self) -> None:
        self._detect_swings()
        self._detect_structure()
        self._detect_order_blocks()
        self._check_ob_mitigation()

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex.")

    def _detect_swings(self) -> None:
        """
        Fractal-based swing detection.
        A swing HIGH at bar i means:  high[i] = max(high[i-n : i+n+1])
        A swing LOW  at bar i means:  low[i]  = min(low[i-n  : i+n+1])
        """
        n = self.lookback
        highs = self.df["high"].values
        lows  = self.df["low"].values
        idx   = self.df.index

        for i in range(n, len(self.df) - n):
            window_h = highs[i-n : i+n+1]
            window_l = lows[i-n  : i+n+1]

            if highs[i] == window_h.max():
                self.swing_highs.append(
                    SwingPoint(index=i, timestamp=idx[i],
                               price=highs[i], kind="high", confirmed=True)
                )
            if lows[i] == window_l.min():
                self.swing_lows.append(
                    SwingPoint(index=i, timestamp=idx[i],
                               price=lows[i], kind="low", confirmed=True)
                )

    def _detect_structure(self) -> None:
        """
        BOS  — price continues in the prevailing trend direction by breaking
               the most recent swing in that direction.
        CHoCH — price breaks the most recent swing in the OPPOSITE direction,
                signalling a potential trend reversal.

        We walk through swing points in chronological order and track the
        current swing high / low.  When a close breaks through one of them,
        we classify the event.
        """
        closes = self.df["close"].values
        idx    = self.df.index

        all_swings = sorted(
            self.swing_highs + self.swing_lows,
            key=lambda s: s.index,
        )
        if len(all_swings) < 2:
            return

        # Simple trend state: +1 = bullish, -1 = bearish, 0 = undefined
        trend = 0
        last_sh: SwingPoint | None = None
        last_sl: SwingPoint | None = None

        for swing in all_swings:
            if swing.kind == "high":
                last_sh = swing
            else:
                last_sl = swing

            # Check every bar after this swing for a break
            start_bar = swing.index + 1
            end_bar   = len(closes)

            # find next swing to limit look-ahead (avoid re-detecting same break)
            next_swing_bar = end_bar
            for s2 in all_swings:
                if s2.index > swing.index:
                    next_swing_bar = s2.index
                    break

            for i in range(start_bar, min(end_bar, next_swing_bar)):
                c = closes[i]

                # Break above most recent SH → bullish continuation (BOS) or reversal
                if last_sh and c > last_sh.price:
                    kind = "BOS_bull" if trend >= 0 else "CHoCH_bull"
                    self.structure_events.append(
                        StructureEvent(
                            index=i,
                            timestamp=idx[i],
                            price=c,
                            kind=kind,
                            broken_swing_price=last_sh.price,
                        )
                    )
                    trend = 1
                    last_sh = None          # consumed — wait for new SH
                    break

                # Break below most recent SL → bearish continuation (BOS) or reversal
                if last_sl and c < last_sl.price:
                    kind = "BOS_bear" if trend <= 0 else "CHoCH_bear"
                    self.structure_events.append(
                        StructureEvent(
                            index=i,
                            timestamp=idx[i],
                            price=c,
                            kind=kind,
                            broken_swing_price=last_sl.price,
                        )
                    )
                    trend = -1
                    last_sl = None
                    break

    def _detect_order_blocks(self) -> None:
        """
        For each BOS/CHoCH, look back to find the last opposing candle
        before the impulse move — that's the Order Block.

        Bullish OB  : last bearish (close < open) candle before a bullish BOS/CHoCH.
        Bearish OB  : last bullish (close > open) candle before a bearish BOS/CHoCH.
        """
        df = self.df

        for ev in self.structure_events:
            is_bull = ev.kind.endswith("bull")
            look_back_bars = min(ev.index, 20)   # search up to 20 bars back

            ob_bar: int | None = None
            for j in range(ev.index - 1, ev.index - look_back_bars, -1):
                o, c = df["open"].iat[j], df["close"].iat[j]
                if is_bull and c < o:            # bearish candle before bullish impulse
                    ob_bar = j
                    break
                if not is_bull and c > o:        # bullish candle before bearish impulse
                    ob_bar = j
                    break

            if ob_bar is not None:
                self.order_blocks.append(
                    OrderBlock(
                        index=ob_bar,
                        timestamp=df.index[ob_bar],
                        open=df["open"].iat[ob_bar],
                        high=df["high"].iat[ob_bar],
                        low=df["low"].iat[ob_bar],
                        close=df["close"].iat[ob_bar],
                        kind="bullish" if is_bull else "bearish",
                    )
                )

    def _check_ob_mitigation(self) -> None:
        """
        Mark an OB as mitigated once price re-enters its high-low zone
        after it was created.
        """
        df = self.df

        for ob in self.order_blocks:
            for i in range(ob.index + 1, len(df)):
                l, h = df["low"].iat[i], df["high"].iat[i]
                # price wicks into the OB zone
                if l <= ob.high and h >= ob.low:
                    ob.mitigated    = True
                    ob.mitigated_at = df.index[i]
                    break


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame, swing_lookback: int = 5) -> MarketStructure:
    """Shortcut: analyse(df).plot().show()"""
    return MarketStructure(df, swing_lookback)