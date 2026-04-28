"""
strategies/entry_model.py
==========================
ICT Entry Model — Multi-Timeframe Signal Generation

Flow:
  1. Check risk limits (daily loss, max drawdown)
  2. Determine trend on higher timeframe (4H)
  3. Find value zones on lower timeframe (1H) — FVG or Order Block
  4. Confirm with liquidity grab
  5. Calculate entry, SL, TP1, TP2, position size
  6. Return TradeSignal or None

Supports multiple models via config parameters.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime, date

from strategies.market_structure import MarketStructure
from strategies.fvg import FairValueGap, FVG
from strategies.liquidity import Liquidity


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    """Tracks daily and overall risk usage."""
    account_balance: float
    daily_start_balance: float
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    open_trades: int = 0
    max_open_trades: int = 2
    daily_losses: float = 0.0
    total_losses: float = 0.0

    @property
    def daily_loss_limit(self) -> float:
        return self.daily_start_balance * self.daily_loss_limit_pct

    @property
    def max_drawdown(self) -> float:
        return self.account_balance * self.max_drawdown_pct

    def can_trade(self) -> tuple[bool, str]:
        """Returns (can_trade, reason_if_not)."""
        if self.daily_losses >= self.daily_loss_limit:
            return False, f"Daily loss limit hit (${self.daily_losses:.2f} / ${self.daily_loss_limit:.2f})"
        if self.total_losses >= self.max_drawdown:
            return False, f"Max drawdown hit (${self.total_losses:.2f} / ${self.max_drawdown:.2f})"
        if self.open_trades >= self.max_open_trades:
            return False, f"Max open trades reached ({self.open_trades})"
        return True, "OK"

    def record_loss(self, amount: float) -> None:
        self.daily_losses  += amount
        self.total_losses  += amount
        self.account_balance -= amount

    def reset_daily(self) -> None:
        self.daily_losses = 0.0
        self.daily_start_balance = self.account_balance


@dataclass
class TradeSignal:
    """A fully calculated trade signal ready for execution."""
    timestamp: pd.Timestamp
    instrument: str
    direction: Literal["long", "short"]

    # Price levels
    entry: float
    sl: float
    tp1: float                          # partial exit — 50% of position
    tp2: float                          # final exit — remaining 50%

    # Position sizing
    account_balance: float
    risk_pct: float
    risk_amount: float                  # $ at risk
    sl_pips: float
    position_size: float                # in lots
    tp1_size: float                     # lots to close at TP1
    tp2_size: float                     # lots to close at TP2

    # Risk:Reward
    rr_tp1: float
    rr_tp2: float

    # Context — why this signal fired
    trend_source: str                   # e.g. "BOS_bull on 4H"
    entry_zone: str                     # e.g. "Bullish FVG" or "Bullish OB"
    confirmation: str                   # e.g. "Bull liquidity grab"

    # Session info
    session: str = ""

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  {self.direction.upper()} {self.instrument}\n"
            f"{'='*50}\n"
            f"  Entry  : {self.entry:.5f}\n"
            f"  SL     : {self.sl:.5f}  ({self.sl_pips:.1f} pips)\n"
            f"  TP1    : {self.tp1:.5f}  (RR {self.rr_tp1:.1f}:1)\n"
            f"  TP2    : {self.tp2:.5f}  (RR {self.rr_tp2:.1f}:1)\n"
            f"  Size   : {self.position_size:.2f} lots\n"
            f"  Risk   : ${self.risk_amount:.2f} ({self.risk_pct*100:.1f}%)\n"
            f"  Trend  : {self.trend_source}\n"
            f"  Zone   : {self.entry_zone}\n"
            f"  Confirm: {self.confirmation}\n"
            f"{'='*50}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY MODEL
# ──────────────────────────────────────────────────────────────────────────────

class EntryModel:
    """
    ICT Multi-Timeframe Entry Model.

    Parameters
    ----------
    df_high         : higher timeframe DataFrame (e.g. 4H) — trend direction
    df_low          : lower timeframe DataFrame  (e.g. 1H) — entry precision
    instrument      : trading pair name (e.g. "EURUSD")
    account_balance : current account balance in USD
    risk_pct        : risk per trade as decimal (0.01 = 1%)
    daily_loss_limit: max daily loss as decimal (0.03 = 3%)
    max_drawdown    : max total drawdown as decimal (0.10 = 10%)
    tp1_rr          : take profit 1 risk:reward ratio
    tp2_rr          : take profit 2 risk:reward ratio
    tp1_close_pct   : portion of position to close at TP1 (0.5 = 50%)
    sl_buffer_pips  : extra pips below/above SL level for safety
    min_rr          : minimum acceptable RR — skip trade if below this
    pip_value       : value per pip per lot in USD (7.0 default for EURUSD)
    swing_lookback  : swing detection lookback (match your other modules)
    fvg_min_pips    : minimum FVG size to consider
    liq_threshold   : equal high/low pip threshold
    """

    def __init__(
        self,
        df_high: pd.DataFrame,
        df_low: pd.DataFrame,
        instrument: str = "EURUSD",
        account_balance: float = 10_000.0,
        risk_pct: float = 0.01,
        daily_loss_limit: float = 0.03,
        max_drawdown: float = 0.10,
        tp1_rr: float = 1.0,
        tp2_rr: float = 2.0,
        tp1_close_pct: float = 0.5,
        sl_buffer_pips: float = 3.0,
        min_rr: float = 1.5,
        pip_value: float = 7.0,
        swing_lookback: int = 5,
        fvg_min_pips: float = 2.0,
        liq_threshold: float = 3.0,
    ):
        self.df_high    = df_high
        self.df_low     = df_low
        self.instrument = instrument
        self.risk_pct   = risk_pct
        self.tp1_rr     = tp1_rr
        self.tp2_rr     = tp2_rr
        self.tp1_close_pct = tp1_close_pct
        self.sl_buffer  = sl_buffer_pips * 0.0001
        self.min_rr     = min_rr
        self.pip_value  = pip_value

        # Risk state
        self.risk = RiskState(
            account_balance=account_balance,
            daily_start_balance=account_balance,
            daily_loss_limit_pct=daily_loss_limit,
            max_drawdown_pct=max_drawdown,
        )

        # Run all detectors
        self.ms_high = MarketStructure(df_high, swing_lookback)
        self.ms_low  = MarketStructure(df_low,  swing_lookback)
        self.fvg     = FairValueGap(df_low, fvg_min_pips)
        self.liq     = Liquidity(df_low, swing_lookback, liq_threshold)

        # Results
        self.signals: list[TradeSignal] = []
        self._generate()

    # ── public API ─────────────────────────────────────────────────────────────

    def latest_signal(self) -> TradeSignal | None:
        """Most recent valid signal."""
        return self.signals[-1] if self.signals else None

    def signals_df(self) -> pd.DataFrame:
        """All signals as a DataFrame for review/backtesting."""
        if not self.signals:
            return pd.DataFrame()
        rows = []
        for s in self.signals:
            rows.append({
                "timestamp":  s.timestamp,
                "direction":  s.direction,
                "entry":      s.entry,
                "sl":         s.sl,
                "tp1":        s.tp1,
                "tp2":        s.tp2,
                "sl_pips":    s.sl_pips,
                "rr_tp1":     s.rr_tp1,
                "rr_tp2":     s.rr_tp2,
                "size_lots":  s.position_size,
                "risk_$":     s.risk_amount,
                "zone":       s.entry_zone,
                "confirm":    s.confirmation,
                "trend":      s.trend_source,
            })
        return pd.DataFrame(rows)

    def plot(
        self,
        last_n: int | None = None,
        title: str | None = None,
        height: int = 820,
    ) -> go.Figure:
        """Plot 1H chart with all signals marked."""
        df   = self.df_low.iloc[-last_n:] if last_n else self.df_low
        title = title or f"{self.instrument} — ICT Entry Model"
        start_ts = df.index[0]

        fig = make_subplots(rows=1, cols=1)

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing_line_color="#26DE81",
            decreasing_line_color="#FC5C65",
            increasing_fillcolor="#26DE81",
            decreasing_fillcolor="#FC5C65",
            line=dict(width=1),
            whiskerwidth=0.3,
            name="OHLC",
            showlegend=False,
        ))

        # FVG zones (unfilled only)
        for fvg in [f for f in self.fvg.fvgs if not f.filled and f.timestamp >= start_ts]:
            is_bull = fvg.kind == "bullish"
            fig.add_shape(
                type="rect",
                x0=fvg.timestamp, x1=df.index[-1],
                y0=fvg.bottom, y1=fvg.top,
                fillcolor="rgba(38,222,129,0.1)" if is_bull else "rgba(252,92,101,0.1)",
                line=dict(color="#26DE81" if is_bull else "#FC5C65", width=1, dash="dot"),
            )

        # Liquidity levels
        for eq in self.liq.active_buyside():
            if eq.timestamps[-1] >= start_ts:
                fig.add_shape(
                    type="line",
                    x0=eq.timestamps[0], x1=df.index[-1],
                    y0=eq.price, y1=eq.price,
                    line=dict(color="#F7B731", width=1, dash="dash"),
                )

        for eq in self.liq.active_sellside():
            if eq.timestamps[-1] >= start_ts:
                fig.add_shape(
                    type="line",
                    x0=eq.timestamps[0], x1=df.index[-1],
                    y0=eq.price, y1=eq.price,
                    line=dict(color="#45AAF2", width=1, dash="dash"),
                )

        # Trade signals
        for sig in [s for s in self.signals if s.timestamp >= start_ts]:
            is_long = sig.direction == "long"
            colour  = "#26DE81" if is_long else "#FC5C65"

            # Entry line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.entry, y1=sig.entry,
                line=dict(color=colour, width=1.5),
            )
            # SL line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.sl, y1=sig.sl,
                line=dict(color="#FC5C65", width=1, dash="dot"),
            )
            # TP1 line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.tp1, y1=sig.tp1,
                line=dict(color="#26DE81", width=1, dash="dot"),
            )
            # TP2 line
            fig.add_shape(type="line",
                x0=sig.timestamp, x1=df.index[-1],
                y0=sig.tp2, y1=sig.tp2,
                line=dict(color="#26DE81", width=1.5, dash="dot"),
            )
            # Signal marker
            fig.add_trace(go.Scatter(
                x=[sig.timestamp],
                y=[sig.entry],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up" if is_long else "triangle-down",
                    size=14, color=colour,
                ),
                text=[f"{'L' if is_long else 'S'} {sig.rr_tp2:.1f}R"],
                textposition="top center" if is_long else "bottom center",
                textfont=dict(size=10, color=colour),
                name=f"{sig.direction.upper()} signal",
                showlegend=False,
                hovertemplate=(
                    f"<b>{sig.direction.upper()} {self.instrument}</b><br>"
                    f"Entry: {sig.entry:.5f}<br>"
                    f"SL: {sig.sl:.5f}<br>"
                    f"TP1: {sig.tp1:.5f}<br>"
                    f"TP2: {sig.tp2:.5f}<br>"
                    f"RR: {sig.rr_tp2:.1f}:1<br>"
                    f"Zone: {sig.entry_zone}<br>"
                    f"%{{x}}<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#E8E8E8"), x=0.02),
            paper_bgcolor="#0F1117",
            plot_bgcolor="#0F1117",
            height=height,
            margin=dict(l=60, r=20, t=48, b=20),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8", bordercolor="#333"),
        )
        axis_style = dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#AAA", size=10),
            showspikes=True, spikecolor="#555", spikethickness=1,
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style, tickformat=".5f")

        return fig

    # ── signal generation ──────────────────────────────────────────────────────

    def _generate(self) -> None:
        """Walk through 1H bars and check for valid setups."""

        # Step 1 — get higher timeframe trend
        htf_trend = self._get_htf_trend()
        if htf_trend == 0:
            return  # no clear trend on HTF

        # Step 2 — get latest structure on LTF
        ltf_trend = self._get_ltf_trend()

        # Trends must agree
        if htf_trend != ltf_trend:
            return

        direction: Literal["long", "short"] = "long" if htf_trend == 1 else "short"

        # Step 3 — find valid entry zones on LTF
        zones = self._find_entry_zones(direction)
        if not zones:
            return

        # Step 4 — check liquidity confirmation
        for zone_price, zone_label, zone_sl in zones:
            confirmation = self._find_confirmation(direction, zone_price)
            if not confirmation:
                continue

            # Step 5 — calculate full signal
            signal = self._build_signal(
                direction=direction,
                entry=zone_price,
                sl_raw=zone_sl,
                zone_label=zone_label,
                confirmation=confirmation,
                htf_trend=htf_trend,
            )
            if signal:
                self.signals.append(signal)

    def _get_htf_trend(self) -> int:
        """
        Returns +1 (bullish), -1 (bearish), 0 (no clear trend)
        based on last structure event on higher timeframe.
        """
        events = self.ms_high.structure_events
        if not events:
            return 0
        last = events[-1]
        if last.kind in ("BOS_bull", "CHoCH_bull"):
            return 1
        if last.kind in ("BOS_bear", "CHoCH_bear"):
            return -1
        return 0

    def _get_ltf_trend(self) -> int:
        """Same but for lower timeframe."""
        events = self.ms_low.structure_events
        if not events:
            return 0
        last = events[-1]
        if last.kind in ("BOS_bull", "CHoCH_bull"):
            return 1
        if last.kind in ("BOS_bear", "CHoCH_bear"):
            return -1
        return 0

    def _find_entry_zones(
        self,
        direction: Literal["long", "short"],
    ) -> list[tuple[float, str, float]]:
        """
        Returns list of (entry_price, label, sl_price) for valid zones.
        Entry  = FVG midpoint or OB midpoint
        SL     = nearest swing low (for longs) or swing high (for shorts)
        """
        zones = []
        current_price = self.df_low["close"].iloc[-1]

        # Get recent swing points for SL placement
        recent_sh = sorted(self.ms_low.swing_highs, key=lambda s: s.index)[-5:]
        recent_sl = sorted(self.ms_low.swing_lows,  key=lambda s: s.index)[-5:]

        if direction == "long":
            # SL = lowest recent swing low below current price
            sl_candidates = [s.price for s in recent_sl if s.price < current_price]
            if not sl_candidates:
                return []
            swing_sl = min(sl_candidates) - self.sl_buffer

            for fvg in self.fvg.unfilled():
                if fvg.kind == "bullish" and fvg.midpoint < current_price + 0.0100:
                    zones.append((fvg.midpoint, "Bullish FVG", swing_sl))

            for ob in self.ms_low.order_blocks:
                if not ob.mitigated and ob.kind == "bullish" and ob.high < current_price + 0.0100:
                    zones.append(((ob.high + ob.low) / 2, "Bullish OB", swing_sl))

        else:  # short
            # SL = highest recent swing high above current price
            sl_candidates = [s.price for s in recent_sh if s.price > current_price]
            if not sl_candidates:
                return []
            swing_sl = max(sl_candidates) + self.sl_buffer

            for fvg in self.fvg.unfilled():
                if fvg.kind == "bearish" and fvg.midpoint > current_price - 0.0100:
                    zones.append((fvg.midpoint, "Bearish FVG", swing_sl))

            for ob in self.ms_low.order_blocks:
                if not ob.mitigated and ob.kind == "bearish" and ob.low > current_price - 0.0100:
                    zones.append(((ob.high + ob.low) / 2, "Bearish OB", swing_sl))

        return zones

    def _find_confirmation(
        self,
        direction: Literal["long", "short"],
        zone_price: float,
    ) -> str | None:
        """
        Looks for a liquidity grab near the zone as confirmation.
        Returns description string or None.
        """
        if not self.liq.grabs:
            return None

        last_grab = self.liq.grabs[-1]
        price_tolerance = 0.0050   # within 50 pips of zone

        if direction == "long" and last_grab.kind == "bull_grab":
            if abs(last_grab.price - zone_price) <= price_tolerance:
                return f"Bull liquidity grab at {last_grab.swept_level:.5f}"

        if direction == "short" and last_grab.kind == "bear_grab":
            if abs(last_grab.price - zone_price) <= price_tolerance:
                return f"Bear liquidity grab at {last_grab.swept_level:.5f}"

        # Fallback — no grab required if zone aligns with EQH/EQL
        if direction == "long":
            for eq in self.liq.active_sellside():
                if abs(eq.price - zone_price) <= price_tolerance:
                    return f"EQL confluence at {eq.price:.5f}"

        if direction == "short":
            for eq in self.liq.active_buyside():
                if abs(eq.price - zone_price) <= price_tolerance:
                    return f"EQH confluence at {eq.price:.5f}"

        return None

    def _build_signal(
        self,
        direction: Literal["long", "short"],
        entry: float,
        sl_raw: float,
        zone_label: str,
        confirmation: str,
        htf_trend: int,
    ) -> TradeSignal | None:
        """Calculate full signal with position size and TP levels."""

        # Check risk limits first
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            print(f"[Risk block] {reason}")
            return None

        # SL distance
        sl_distance = abs(entry - sl_raw)
        sl_pips     = sl_distance / 0.0001

        if sl_pips < 15:    # minimum 15 pips SL for 1H EURUSD
            return None

        # TP levels — based on fixed RR first
        if direction == "long":
            tp1 = entry + (sl_distance * self.tp1_rr)
            tp2 = entry + (sl_distance * self.tp2_rr)

            # Use next liquidity level as TP2 only if within 3x RR
            next_liq = self._next_liquidity_above(entry)
            if next_liq and tp1 < next_liq <= entry + (sl_distance * 3.0):
                tp2 = next_liq
        else:
            tp1 = entry - (sl_distance * self.tp1_rr)
            tp2 = entry - (sl_distance * self.tp2_rr)

            next_liq = self._next_liquidity_below(entry)
            if next_liq and tp1 > next_liq >= entry - (sl_distance * 3.0):
                tp2 = next_liq

        # Check minimum RR
        rr_tp1 = abs(entry - tp1) / sl_distance
        rr_tp2 = abs(entry - tp2) / sl_distance

        if rr_tp2 < self.min_rr:
            return None

        # Position sizing
        risk_amount   = self.risk.account_balance * self.risk_pct
        pip_risk      = sl_pips * self.pip_value
        position_size = round(risk_amount / pip_risk, 2) if pip_risk > 0 else 0

        if position_size <= 0:
            return None

        tp1_size = round(position_size * self.tp1_close_pct, 2)
        tp2_size = round(position_size * (1 - self.tp1_close_pct), 2)

        # HTF trend label
        htf_label = self.ms_high.structure_events[-1].kind if self.ms_high.structure_events else "unknown"
        trend_source = f"{htf_label} on HTF"

        return TradeSignal(
            timestamp=self.df_low.index[-1],
            instrument=self.instrument,
            direction=direction,
            entry=round(entry, 5),
            sl=round(sl_raw, 5),
            tp1=round(tp1, 5),
            tp2=round(tp2, 5),
            account_balance=self.risk.account_balance,
            risk_pct=self.risk_pct,
            risk_amount=round(risk_amount, 2),
            sl_pips=round(sl_pips, 1),
            position_size=position_size,
            tp1_size=tp1_size,
            tp2_size=tp2_size,
            rr_tp1=round(rr_tp1, 2),
            rr_tp2=round(rr_tp2, 2),
            trend_source=trend_source,
            entry_zone=zone_label,
            confirmation=confirmation,
        )

    def _next_liquidity_above(self, price: float) -> float | None:
        """Nearest unswept EQH above current price — used as TP2 target."""
        candidates = [
            eq.price for eq in self.liq.active_buyside()
            if eq.price > price
        ]
        return min(candidates) if candidates else None

    def _next_liquidity_below(self, price: float) -> float | None:
        """Nearest unswept EQL below current price — used as TP2 target."""
        candidates = [
            eq.price for eq in self.liq.active_sellside()
            if eq.price < price
        ]
        return max(candidates) if candidates else None