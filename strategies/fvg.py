"""
strategies/fvg.py
=================
ICT Fair Value Gap (FVG) Detection

A FVG is a 3-candle imbalance where the middle candle moves so fast
that candle 1 and candle 3 don't overlap — leaving an unfilled price gap.

Bullish FVG : high[i-2] < low[i]   → gap between candle 1 top and candle 3 bottom
Bearish FVG : low[i-2]  > high[i]  → gap between candle 1 bottom and candle 3 top

Price tends to return and "fill" the gap — used as entry zones in ICT.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FVG:
    index: int                          # index of the middle candle
    timestamp: pd.Timestamp
    top: float                          # upper boundary of the gap
    bottom: float                       # lower boundary of the gap
    kind: Literal["bullish", "bearish"]
    filled: bool = False                # True once price enters the gap zone
    filled_at: pd.Timestamp | None = None

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        """Gap size in price (pips * 0.0001 for EURUSD)."""
        return self.top - self.bottom


# ──────────────────────────────────────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────────────────────────────────────

class FairValueGap:
    """
    Detects and tracks Fair Value Gaps on an OHLCV DataFrame.

    Parameters
    ----------
    df          : OHLCV DataFrame with DatetimeIndex
    min_gap_pips: minimum gap size to qualify (filters micro-gaps / noise)
    """

    COLOUR = dict(
        bull_fill="rgba(38,222,129,0.15)",
        bull_border="#26DE81",
        bear_fill="rgba(252,92,101,0.15)",
        bear_border="#FC5C65",
        filled_fill="rgba(150,150,150,0.08)",
        filled_border="rgba(150,150,150,0.4)",
        bg="#0F1117",
        grid="rgba(255,255,255,0.06)",
        candle_up="#26DE81",
        candle_down="#FC5C65",
    )

    def __init__(self, df: pd.DataFrame, min_gap_pips: float = 2.0):
        self._validate(df)
        self.df = df.copy()
        self.min_gap = min_gap_pips * 0.0001   # convert pips to price
        self.fvgs: list[FVG] = []
        self._detect()
        self._check_fills()

    # ── public API ─────────────────────────────────────────────────────────────

    def unfilled(self) -> list[FVG]:
        """Return only FVGs price has not yet revisited."""
        return [f for f in self.fvgs if not f.filled]

    def plot(
        self,
        last_n: int | None = None,
        title: str = "ICT Fair Value Gaps",
        height: int = 720,
        show_filled: bool = False,
    ) -> go.Figure:
        """
        Returns a Plotly figure with candles + FVG zones.

        Parameters
        ----------
        show_filled : show FVGs that price has already filled (greyed out)
        """
        df = self.df.iloc[-last_n:] if last_n else self.df
        start_ts = df.index[0]

        fig = make_subplots(rows=1, cols=1)

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing_line_color=self.COLOUR["candle_up"],
            decreasing_line_color=self.COLOUR["candle_down"],
            increasing_fillcolor=self.COLOUR["candle_up"],
            decreasing_fillcolor=self.COLOUR["candle_down"],
            line=dict(width=1),
            whiskerwidth=0.3,
            name="OHLC",
            showlegend=False,
        ))

        # FVG zones
        fvgs_to_show = [
            f for f in self.fvgs
            if f.timestamp >= start_ts and (show_filled or not f.filled)
        ]

        for fvg in fvgs_to_show:
            is_bull = fvg.kind == "bullish"

            if fvg.filled:
                fill   = self.COLOUR["filled_fill"]
                border = self.COLOUR["filled_border"]
                lbl    = "FVG✓"
            else:
                fill   = self.COLOUR["bull_fill"] if is_bull else self.COLOUR["bear_fill"]
                border = self.COLOUR["bull_border"] if is_bull else self.COLOUR["bear_border"]
                lbl    = "FVG+" if is_bull else "FVG-"

            # zone rectangle — extends to right edge
            x1 = fvg.filled_at if fvg.filled else df.index[-1]
            fig.add_shape(
                type="rect",
                x0=fvg.timestamp, x1=x1,
                y0=fvg.bottom,    y1=fvg.top,
                fillcolor=fill,
                line=dict(color=border, width=1, dash="dot"),
            )
            fig.add_annotation(
                x=fvg.timestamp,
                y=fvg.top,
                text=lbl,
                font=dict(size=9, color=border),
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(15,17,23,0.7)",
                borderpad=2,
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#E8E8E8"), x=0.02),
            paper_bgcolor=self.COLOUR["bg"],
            plot_bgcolor=self.COLOUR["bg"],
            height=height,
            margin=dict(l=60, r=20, t=48, b=20),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8", bordercolor="#333"),
        )
        axis_style = dict(
            gridcolor=self.COLOUR["grid"],
            zerolinecolor=self.COLOUR["grid"],
            tickfont=dict(color="#AAA", size=10),
            showspikes=True,
            spikecolor="#555",
            spikethickness=1,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style, tickformat=".5f")

        return fig

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "timestamp": f.timestamp,
            "kind":      f.kind,
            "top":       f.top,
            "bottom":    f.bottom,
            "size_pips": round(f.size / 0.0001, 1),
            "filled":    f.filled,
            "filled_at": f.filled_at,
        } for f in self.fvgs])

    # ── private ────────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex.")

    def _detect(self) -> None:
        highs = self.df["high"].values
        lows  = self.df["low"].values
        idx   = self.df.index

        for i in range(2, len(self.df)):
            # Bullish FVG: gap between top of candle[i-2] and bottom of candle[i]
            if lows[i] > highs[i-2]:
                gap = lows[i] - highs[i-2]
                if gap >= self.min_gap:
                    self.fvgs.append(FVG(
                        index=i-1,
                        timestamp=idx[i-1],   # middle candle = the impulse
                        top=lows[i],
                        bottom=highs[i-2],
                        kind="bullish",
                    ))

            # Bearish FVG: gap between bottom of candle[i-2] and top of candle[i]
            elif highs[i] < lows[i-2]:
                gap = lows[i-2] - highs[i]
                if gap >= self.min_gap:
                    self.fvgs.append(FVG(
                        index=i-1,
                        timestamp=idx[i-1],
                        top=lows[i-2],
                        bottom=highs[i],
                        kind="bearish",
                    ))

    def _check_fills(self) -> None:
        """Mark FVG as filled once price wicks into the gap zone."""
        highs = self.df["high"].values
        lows  = self.df["low"].values
        idx   = self.df.index

        for fvg in self.fvgs:
            for i in range(fvg.index + 2, len(self.df)):
                if lows[i] <= fvg.top and highs[i] >= fvg.bottom:
                    fvg.filled    = True
                    fvg.filled_at = idx[i]
                    break