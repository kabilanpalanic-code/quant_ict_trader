"""
strategies/bpr.py
==================
Balanced Price Range (BPR) Detection

BPR = overlap zone between a bullish FVG and a bearish FVG.
This overlap is the highest probability entry zone in ICT.

Bullish FVG : high[i-2] < low[i]   — gap below current price
Bearish FVG : low[i-2]  > high[i]  — gap above current price

BPR forms when these two zones overlap.
Entry = BPR midpoint.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class FVGZone:
    index:     int
    timestamp: pd.Timestamp
    top:       float
    bottom:    float
    kind:      Literal["bullish", "bearish"]

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size_pips(self) -> float:
        return (self.top - self.bottom) / 0.0001


@dataclass
class BPR:
    """
    Balanced Price Range — overlap of bullish and bearish FVG.
    Entry at midpoint, highest probability reversal zone.
    """
    top:           float        # top of overlap zone
    bottom:        float        # bottom of overlap zone
    midpoint:      float        # entry price
    bullish_fvg:   FVGZone
    bearish_fvg:   FVGZone
    formed_at:     pd.Timestamp
    filled:        bool = False
    filled_at:     pd.Timestamp | None = None

    @property
    def size_pips(self) -> float:
        return (self.top - self.bottom) / 0.0001


def detect_fvgs(
    df: pd.DataFrame,
    min_gap_pips: float = 1.0,
) -> list[FVGZone]:
    """Detect all FVGs in the DataFrame."""
    fvgs: list[FVGZone] = []
    highs  = df["high"].values
    lows   = df["low"].values
    idx    = df.index
    min_gap = min_gap_pips * 0.0001

    for i in range(2, len(df)):
        # Bullish FVG
        if lows[i] > highs[i-2] and (lows[i] - highs[i-2]) >= min_gap:
            fvgs.append(FVGZone(
                index=i-1, timestamp=idx[i-1],
                top=lows[i], bottom=highs[i-2],
                kind="bullish",
            ))
        # Bearish FVG
        elif highs[i] < lows[i-2] and (lows[i-2] - highs[i]) >= min_gap:
            fvgs.append(FVGZone(
                index=i-1, timestamp=idx[i-1],
                top=lows[i-2], bottom=highs[i],
                kind="bearish",
            ))

    return fvgs


def detect_bprs(
    df: pd.DataFrame,
    min_gap_pips: float = 1.0,
    max_lookback: int = 50,
) -> list[BPR]:
    """
    Detect BPR zones — overlapping bullish and bearish FVGs.

    Parameters
    ----------
    max_lookback : only check FVGs within last N bars
    """
    fvgs = detect_fvgs(df, min_gap_pips)
    if not fvgs:
        return []

    # Only recent FVGs
    recent_fvgs = [f for f in fvgs if f.index >= len(df) - max_lookback]

    bullish = [f for f in recent_fvgs if f.kind == "bullish"]
    bearish = [f for f in recent_fvgs if f.kind == "bearish"]

    bprs: list[BPR] = []

    for bull in bullish:
        for bear in bearish:
            # Check overlap
            overlap_top    = min(bull.top,    bear.top)
            overlap_bottom = max(bull.bottom, bear.bottom)

            if overlap_bottom >= overlap_top:
                continue  # no overlap

            overlap_pips = (overlap_top - overlap_bottom) / 0.0001
            if overlap_pips < 1.0:
                continue  # overlap too small

            # BPR forms at the later of the two FVGs
            formed_at = max(bull.timestamp, bear.timestamp)

            bpr = BPR(
                top          = overlap_top,
                bottom       = overlap_bottom,
                midpoint     = (overlap_top + overlap_bottom) / 2,
                bullish_fvg  = bull,
                bearish_fvg  = bear,
                formed_at    = formed_at,
            )
            bprs.append(bpr)

    # Check fills — BPR is filled once price closes inside it
    closes = df["close"].values
    idx    = df.index

    for bpr in bprs:
        start_bar = df.index.searchsorted(bpr.formed_at)
        for i in range(start_bar + 1, len(df)):
            if df["low"].iloc[i] <= bpr.top and df["high"].iloc[i] >= bpr.bottom:
                bpr.filled    = True
                bpr.filled_at = idx[i]
                break

    return bprs


def get_active_bprs(df: pd.DataFrame, min_gap_pips: float = 1.0) -> list[BPR]:
    """Return only unfilled BPRs."""
    return [b for b in detect_bprs(df, min_gap_pips) if not b.filled]