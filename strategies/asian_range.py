"""
strategies/asian_range.py
==========================
Detects Asian Kill Zone high and low on 5min OHLCV data.

Asian Kill Zone: 8:00 PM - 10:00 PM New York time
- Asian High = highest point during kill zone
- Asian Low  = lowest point during kill zone

These levels become the liquidity targets for the Stop Hunt Model.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

ASIAN_START_HOUR = 20  # 8 PM NY
ASIAN_END_HOUR   = 22  # 10 PM NY


@dataclass
class AsianRange:
    date:       pd.Timestamp   # date this range belongs to
    high:       float
    low:        float
    high_time:  pd.Timestamp
    low_time:   pd.Timestamp
    start_time: pd.Timestamp
    end_time:   pd.Timestamp

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2

    @property
    def size_pips(self) -> float:
        return (self.high - self.low) / 0.0001


def detect_asian_ranges(df: pd.DataFrame) -> list[AsianRange]:
    """
    Detect all Asian Kill Zone ranges in the DataFrame.

    Parameters
    ----------
    df : OHLCV DataFrame with UTC DatetimeIndex (5min recommended)
    """
    # Convert index to NY time for session filtering
    df_ny = df.copy()
    df_ny.index = df_ny.index.tz_convert(NY_TZ)

    ranges: list[AsianRange] = []

    # Group by date in NY time
    for date, group in df_ny.groupby(df_ny.index.date):
        asian_bars = group[
            (group.index.hour >= ASIAN_START_HOUR) &
            (group.index.hour <  ASIAN_END_HOUR)
        ]

        if len(asian_bars) < 4:  # need at least 4 bars (20 min)
            continue

        high_idx = asian_bars["high"].idxmax()
        low_idx  = asian_bars["low"].idxmin()

        ranges.append(AsianRange(
            date       = pd.Timestamp(date),
            high       = float(asian_bars["high"].max()),
            low        = float(asian_bars["low"].min()),
            high_time  = high_idx,
            low_time   = low_idx,
            start_time = asian_bars.index[0],
            end_time   = asian_bars.index[-1],
        ))

    return ranges


def get_latest_asian_range(df: pd.DataFrame) -> AsianRange | None:
    """Get the most recent completed Asian range."""
    ranges = detect_asian_ranges(df)
    return ranges[-1] if ranges else None