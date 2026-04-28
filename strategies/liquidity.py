"""
strategies/liquidity.py
========================
ICT Liquidity Detection

Concepts implemented:
  - Equal Highs (EQH) — buy-side liquidity above
  - Equal Lows  (EQL) — sell-side liquidity below
  - Liquidity Grab — price sweeps a liquidity level then reverses
  - Liquidity Pool — cluster of swing highs/lows within a tight range
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EqualLevel:
    """Two or more swing points at nearly the same price — a liquidity cluster."""
    timestamps: list[pd.Timestamp]
    price: float                          # average level
    kind: Literal["high", "low"]          # EQH or EQL
    swept: bool = False                   # True once price breaks through
    swept_at: pd.Timestamp | None = None


@dataclass
class LiquidityGrab:
    """
    Price wicks beyond a swing high/low (sweeping stops),
    then closes back inside — a classic ICT liquidity grab.
    """
    index: int
    timestamp: pd.Timestamp
    price: float                          # the wick extreme
    kind: Literal["bull_grab", "bear_grab"]
    swept_level: float                    # the swing level that was swept


# ──────────────────────────────────────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────────────────────────────────────

class Liquidity:
    """
    Detects ICT liquidity concepts on an OHLCV DataFrame.

    Parameters
    ----------
    df              : OHLCV DataFrame with DatetimeIndex
    swing_lookback  : bars each side for swing detection (match market_structure.py)
    equal_threshold : max pip difference to consider two levels "equal"
    """

    COLOUR = dict(
        eqh="#F7B731",
        eql="#45AAF2",
        grab_bull="#26DE81",
        grab_bear="#FC5C65",
        swept="rgba(150,150,150,0.5)",
        bg="#0F1117",
        grid="rgba(255,255,255,0.06)",
        candle_up="#26DE81",
        candle_down="#FC5C65",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 5,
        equal_threshold: float = 3.0,    # pips
    ):
        self._validate(df)
        self.df = df.copy()
        self.lookback  = swing_lookback
        self.threshold = equal_threshold * 0.0001

        self.equal_highs: list[EqualLevel] = []
        self.equal_lows:  list[EqualLevel] = []
        self.grabs:       list[LiquidityGrab] = []

        self._swing_highs: list[tuple[int, pd.Timestamp, float]] = []
        self._swing_lows:  list[tuple[int, pd.Timestamp, float]] = []

        self._detect_swings()
        self._detect_equal_levels()
        self._detect_grabs()
        self._check_swept()

    # ── public API ─────────────────────────────────────────────────────────────

    def active_buyside(self) -> list[EqualLevel]:
        """Unswept equal highs — buy-side liquidity still available."""
        return [e for e in self.equal_highs if not e.swept]

    def active_sellside(self) -> list[EqualLevel]:
        """Unswept equal lows — sell-side liquidity still available."""
        return [e for e in self.equal_lows if not e.swept]

    def plot(
        self,
        last_n: int | None = None,
        title: str = "ICT Liquidity",
        height: int = 720,
        show_swept: bool = False,
    ) -> go.Figure:
        df = self.df.iloc[-last_n:] if last_n else self.df
        start_ts = df.index[0]

        fig = go.Figure()

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

        # Equal Highs — dashed horizontal lines
        for eq in self.equal_highs:
            if eq.timestamps[-1] < start_ts:
                continue
            if eq.swept and not show_swept:
                continue
            colour = self.COLOUR["swept"] if eq.swept else self.COLOUR["eqh"]
            fig.add_shape(
                type="line",
                x0=eq.timestamps[0], x1=df.index[-1],
                y0=eq.price, y1=eq.price,
                line=dict(color=colour, width=1.2, dash="dash"),
            )
            fig.add_annotation(
                x=eq.timestamps[-1],
                y=eq.price * 1.0003,
                text="EQH" + (" ✓" if eq.swept else ""),
                font=dict(size=9, color=colour),
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(15,17,23,0.7)",
                borderpad=2,
            )

        # Equal Lows — dashed horizontal lines
        for eq in self.equal_lows:
            if eq.timestamps[-1] < start_ts:
                continue
            if eq.swept and not show_swept:
                continue
            colour = self.COLOUR["swept"] if eq.swept else self.COLOUR["eql"]
            fig.add_shape(
                type="line",
                x0=eq.timestamps[0], x1=df.index[-1],
                y0=eq.price, y1=eq.price,
                line=dict(color=colour, width=1.2, dash="dash"),
            )
            fig.add_annotation(
                x=eq.timestamps[-1],
                y=eq.price * 0.9997,
                text="EQL" + (" ✓" if eq.swept else ""),
                font=dict(size=9, color=colour),
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(15,17,23,0.7)",
                borderpad=2,
            )

        # Liquidity Grabs — arrow markers
        for grab in [g for g in self.grabs if g.timestamp >= start_ts]:
            is_bull = grab.kind == "bull_grab"
            colour  = self.COLOUR["grab_bull"] if is_bull else self.COLOUR["grab_bear"]
            fig.add_trace(go.Scatter(
                x=[grab.timestamp],
                y=[grab.price],
                mode="markers",
                marker=dict(
                    symbol="arrow-up" if is_bull else "arrow-down",
                    size=12,
                    color=colour,
                    line=dict(color=colour, width=1),
                ),
                name="Liq Grab",
                showlegend=False,
                hovertemplate=(
                    f"<b>{'Bull' if is_bull else 'Bear'} Liquidity Grab</b><br>"
                    f"Swept: {grab.swept_level:.5f}<br>"
                    f"%{{x}}<extra></extra>"
                ),
            ))

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
        rows = []
        for eq in self.equal_highs + self.equal_lows:
            rows.append({
                "kind":    "EQH" if eq.kind == "high" else "EQL",
                "price":   eq.price,
                "touches": len(eq.timestamps),
                "swept":   eq.swept,
                "swept_at": eq.swept_at,
            })
        return pd.DataFrame(rows).sort_values("price", ascending=False)

    # ── private ────────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex.")

    def _detect_swings(self) -> None:
        n     = self.lookback
        highs = self.df["high"].values
        lows  = self.df["low"].values
        idx   = self.df.index

        for i in range(n, len(self.df) - n):
            if highs[i] == highs[i-n:i+n+1].max():
                self._swing_highs.append((i, idx[i], highs[i]))
            if lows[i] == lows[i-n:i+n+1].min():
                self._swing_lows.append((i, idx[i], lows[i]))

    def _detect_equal_levels(self) -> None:
        """
        Group swing points that are within `threshold` pips of each other.
        Two or more in the same group = an equal high/low.
        """
        def group_levels(points: list, kind: Literal["high", "low"]) -> list[EqualLevel]:
            if not points:
                return []
            # sort by price
            sorted_pts = sorted(points, key=lambda x: x[2])
            groups: list[list] = [[sorted_pts[0]]]

            for pt in sorted_pts[1:]:
                if abs(pt[2] - groups[-1][-1][2]) <= self.threshold:
                    groups[-1].append(pt)
                else:
                    groups.append([pt])

            levels = []
            for grp in groups:
                if len(grp) >= 2:
                    avg_price = float(np.mean([p[2] for p in grp]))
                    timestamps = [p[1] for p in sorted(grp, key=lambda x: x[0])]
                    levels.append(EqualLevel(
                        timestamps=timestamps,
                        price=avg_price,
                        kind=kind,
                    ))
            return levels

        self.equal_highs = group_levels(self._swing_highs, "high")
        self.equal_lows  = group_levels(self._swing_lows,  "low")

    def _detect_grabs(self) -> None:
        """
        A liquidity grab = candle wicks beyond a swing level but CLOSES back inside.

        Bull grab: wick below a swing low, close above it — sell-side stops swept.
        Bear grab: wick above a swing high, close below it — buy-side stops swept.
        """
        highs  = self.df["high"].values
        lows   = self.df["low"].values
        closes = self.df["close"].values
        idx    = self.df.index

        swing_high_prices = [p[2] for p in self._swing_highs]
        swing_low_prices  = [p[2] for p in self._swing_lows]

        for i in range(1, len(self.df)):
            # Bear grab — wick above a swing high, close below it
            for sh in swing_high_prices:
                if highs[i] > sh and closes[i] < sh:
                    self.grabs.append(LiquidityGrab(
                        index=i,
                        timestamp=idx[i],
                        price=highs[i],
                        kind="bear_grab",
                        swept_level=sh,
                    ))
                    break

            # Bull grab — wick below a swing low, close above it
            for sl in swing_low_prices:
                if lows[i] < sl and closes[i] > sl:
                    self.grabs.append(LiquidityGrab(
                        index=i,
                        timestamp=idx[i],
                        price=lows[i],
                        kind="bull_grab",
                        swept_level=sl,
                    ))
                    break

    def _check_swept(self) -> None:
        """Mark equal levels as swept once price closes beyond them."""
        closes = self.df["close"].values
        idx    = self.df.index

        for eq in self.equal_highs:
            last_touch_bar = self.df.index.get_loc(eq.timestamps[-1])
            for i in range(last_touch_bar + 1, len(self.df)):
                if closes[i] > eq.price:
                    eq.swept    = True
                    eq.swept_at = idx[i]
                    break

        for eq in self.equal_lows:
            last_touch_bar = self.df.index.get_loc(eq.timestamps[-1])
            for i in range(last_touch_bar + 1, len(self.df)):
                if closes[i] < eq.price:
                    eq.swept    = True
                    eq.swept_at = idx[i]
                    break