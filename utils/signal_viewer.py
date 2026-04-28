"""
utils/signal_viewer.py
======================
Focused signal chart viewer.
Shows 50 candles before and after each signal with clear SL/TP levels.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategies.entry_model import TradeSignal


def plot_signal(
    df: pd.DataFrame,
    signal: TradeSignal,
    candles_before: int = 50,
    candles_after: int = 20,
    height: int = 600,
) -> go.Figure:
    """
    Plot a focused chart around a single signal.
    Shows candles_before bars before entry and candles_after bars after.
    """
    # Find signal bar index
    try:
        sig_idx = df.index.get_loc(signal.timestamp)
    except KeyError:
        # find nearest bar
        sig_idx = df.index.searchsorted(signal.timestamp)

    start = max(0, sig_idx - candles_before)
    end   = min(len(df), sig_idx + candles_after)
    view  = df.iloc[start:end]

    is_long = signal.direction == "long"
    colour  = "#26DE81" if is_long else "#FC5C65"

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=view.index,
        open=view["open"], high=view["high"],
        low=view["low"],   close=view["close"],
        increasing_line_color="#26DE81",
        decreasing_line_color="#FC5C65",
        increasing_fillcolor="#26DE81",
        decreasing_fillcolor="#FC5C65",
        line=dict(width=1),
        whiskerwidth=0.3,
        showlegend=False,
        name="OHLC",
    ))

    x0 = view.index[0]
    x1 = view.index[-1]

    # Entry line
    fig.add_shape(type="line", x0=x0, x1=x1,
        y0=signal.entry, y1=signal.entry,
        line=dict(color=colour, width=2))
    fig.add_annotation(x=x1, y=signal.entry,
        text=f"  ENTRY {signal.entry:.5f}",
        font=dict(size=11, color=colour),
        showarrow=False, xanchor="left",
        bgcolor="rgba(15,17,23,0.8)", borderpad=3)

    # SL line
    fig.add_shape(type="line", x0=x0, x1=x1,
        y0=signal.sl, y1=signal.sl,
        line=dict(color="#FC5C65", width=1.5, dash="dash"))
    fig.add_annotation(x=x1, y=signal.sl,
        text=f"  SL {signal.sl:.5f} ({signal.sl_pips:.0f} pips)",
        font=dict(size=11, color="#FC5C65"),
        showarrow=False, xanchor="left",
        bgcolor="rgba(15,17,23,0.8)", borderpad=3)

    # TP1 line
    fig.add_shape(type="line", x0=x0, x1=x1,
        y0=signal.tp1, y1=signal.tp1,
        line=dict(color="#26DE81", width=1.5, dash="dash"))
    fig.add_annotation(x=x1, y=signal.tp1,
        text=f"  TP1 {signal.tp1:.5f} ({signal.rr_tp1:.1f}R)",
        font=dict(size=11, color="#26DE81"),
        showarrow=False, xanchor="left",
        bgcolor="rgba(15,17,23,0.8)", borderpad=3)

    # TP2 line
    fig.add_shape(type="line", x0=x0, x1=x1,
        y0=signal.tp2, y1=signal.tp2,
        line=dict(color="#A8FF78", width=2, dash="dash"))
    fig.add_annotation(x=x1, y=signal.tp2,
        text=f"  TP2 {signal.tp2:.5f} ({signal.rr_tp2:.1f}R)",
        font=dict(size=11, color="#A8FF78"),
        showarrow=False, xanchor="left",
        bgcolor="rgba(15,17,23,0.8)", borderpad=3)

    # Signal marker
    fig.add_trace(go.Scatter(
        x=[signal.timestamp],
        y=[signal.entry],
        mode="markers",
        marker=dict(
            symbol="triangle-up" if is_long else "triangle-down",
            size=16, color=colour,
            line=dict(color="white", width=1),
        ),
        showlegend=False,
    ))

    # Risk zone shading between entry and SL
    fig.add_shape(type="rect",
        x0=x0, x1=x1,
        y0=min(signal.entry, signal.sl),
        y1=max(signal.entry, signal.sl),
        fillcolor="rgba(252,92,101,0.08)",
        line=dict(width=0),
    )

    # Reward zone shading between entry and TP2
    fig.add_shape(type="rect",
        x0=x0, x1=x1,
        y0=min(signal.entry, signal.tp2),
        y1=max(signal.entry, signal.tp2),
        fillcolor="rgba(38,222,129,0.06)",
        line=dict(width=0),
    )

    # Title with signal info
    title = (
        f"{signal.direction.upper()} {signal.instrument} | "
        f"Entry: {signal.entry:.5f} | "
        f"SL: {signal.sl_pips:.0f} pips | "
        f"RR: {signal.rr_tp2:.1f}:1 | "
        f"Risk: ${signal.risk_amount:.0f} | "
        f"Size: {signal.position_size:.2f} lots"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#E8E8E8"), x=0.01),
        paper_bgcolor="#0F1117",
        plot_bgcolor="#0F1117",
        height=height,
        margin=dict(l=60, r=160, t=50, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8"),
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


def plot_all_signals(
    df: pd.DataFrame,
    signals: list[TradeSignal],
    candles_before: int = 50,
    candles_after: int = 20,
    height: int = 600,
    save_html: str | None = "signals.html",
) -> None:
    """
    Saves one HTML file with all signals as separate tabs/sections.
    Open signals.html in browser to review each trade.
    """
    if not signals:
        print("No signals to plot.")
        return

    html_parts = []
    for i, sig in enumerate(signals):
        fig = plot_signal(df, sig, candles_before, candles_after, height)
        html_parts.append(f"<h2 style='color:#E8E8E8;font-family:monospace;padding:10px'>Signal {i+1} — {sig.timestamp}</h2>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))

    full_html = f"""
    <html>
    <head>
        <title>ICT Signals</title>
        <style>
            body {{ background: #0F1117; margin: 0; padding: 20px; }}
            h2 {{ margin-top: 30px; }}
        </style>
    </head>
    <body>
        <h1 style='color:#E8E8E8;font-family:monospace'>
            ICT Signal Review — {len(signals)} signals
        </h1>
        {''.join(html_parts)}
    </body>
    </html>
    """

    if save_html:
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Saved {len(signals)} signal charts → {save_html}")