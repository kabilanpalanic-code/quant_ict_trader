"""
utils/signal_viewer.py
======================
ICT Signal Chart Viewer — shows entry, SL, TP levels with trade result.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from strategies.entry_model import TradeSignal

# ── colour palette ─────────────────────────────────────────────────────────────
C = dict(
    bg="#0F1117", grid="rgba(255,255,255,0.06)",
    up="#26DE81", down="#FC5C65",
    entry_long="#26DE81", entry_short="#FC5C65",
    sl="#FC5C65", tp1="#26DE81", tp2="#A8FF78",
    win_bg="rgba(38,222,129,0.04)",
    loss_bg="rgba(252,92,101,0.04)",
    expired_bg="rgba(150,150,150,0.04)",
    open_bg="rgba(69,170,242,0.04)",
    vline="rgba(255,255,255,0.3)",
)

RESULT_COLOUR = {
    "WIN":     "#26DE81",
    "LOSS":    "#FC5C65",
    "PARTIAL": "#F7B731",
    "EXPIRED": "#888888",
    "OPEN":    "#45AAF2",
}


def plot_signal(
    df: pd.DataFrame,
    signal: TradeSignal,
    trade=None,
    candles_before: int = 60,
    candles_after: int = 40,
    height: int = 620,
) -> go.Figure:
    try:
        sig_idx = df.index.get_loc(signal.timestamp)
    except KeyError:
        sig_idx = df.index.searchsorted(signal.timestamp)

    start = max(0, sig_idx - candles_before)
    end   = min(len(df), sig_idx + candles_after)
    view  = df.iloc[start:end]

    is_long = signal.direction == "long"
    entry_colour = C["entry_long"] if is_long else C["entry_short"]

    result  = trade.result if trade else "OPEN"
    res_col = RESULT_COLOUR.get(result, "#888")
    res_bg  = C.get(f"{result.lower()}_bg", C["open_bg"])

    fig = go.Figure()

    # Background tint
    fig.add_shape(type="rect",
        x0=view.index[0], x1=view.index[-1],
        y0=0, y1=1, xref="x", yref="paper",
        fillcolor=res_bg, line=dict(width=0), layer="below")

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=view.index,
        open=view["open"], high=view["high"],
        low=view["low"],   close=view["close"],
        increasing_line_color=C["up"], decreasing_line_color=C["down"],
        increasing_fillcolor=C["up"],  decreasing_fillcolor=C["down"],
        line=dict(width=1), whiskerwidth=0.3,
        showlegend=False, name="OHLC",
    ))

    entry_ts = signal.timestamp
    x1 = view.index[-1]

    # Vertical entry line
    fig.add_shape(type="line",
        x0=entry_ts, x1=entry_ts, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=C["vline"], width=1.5, dash="dot"))

    # Entry line
    fig.add_shape(type="line",
        x0=view.index[0], x1=entry_ts,
        y0=signal.entry, y1=signal.entry,
        line=dict(color=entry_colour, width=1, dash="dot"), opacity=0.3)
    fig.add_shape(type="line",
        x0=entry_ts, x1=x1,
        y0=signal.entry, y1=signal.entry,
        line=dict(color=entry_colour, width=2))
    fig.add_annotation(x=entry_ts, y=signal.entry,
        text=f"ENTRY {signal.entry:.5f}",
        font=dict(size=10, color=entry_colour), showarrow=False,
        xanchor="left", yanchor="bottom" if is_long else "top",
        bgcolor="rgba(15,17,23,0.85)", borderpad=3,
        bordercolor=entry_colour, borderwidth=1)

    # SL line
    fig.add_shape(type="line", x0=entry_ts, x1=x1,
        y0=signal.sl, y1=signal.sl,
        line=dict(color=C["sl"], width=1.5, dash="dash"))
    fig.add_annotation(x=entry_ts, y=signal.sl,
        text=f"SL {signal.sl:.5f} ({signal.sl_pips:.0f}p)",
        font=dict(size=9, color=C["sl"]), showarrow=False,
        xanchor="left", yanchor="top" if is_long else "bottom",
        bgcolor="rgba(15,17,23,0.85)", borderpad=2)

    # TP1 line
    fig.add_shape(type="line", x0=entry_ts, x1=x1,
        y0=signal.tp1, y1=signal.tp1,
        line=dict(color=C["tp1"], width=1.5, dash="dash"))
    fig.add_annotation(x=entry_ts, y=signal.tp1,
        text=f"TP1 {signal.tp1:.5f} ({signal.rr_tp1:.1f}R)",
        font=dict(size=9, color=C["tp1"]), showarrow=False,
        xanchor="left", yanchor="bottom" if is_long else "top",
        bgcolor="rgba(15,17,23,0.85)", borderpad=2)

    # TP2 line
    fig.add_shape(type="line", x0=entry_ts, x1=x1,
        y0=signal.tp2, y1=signal.tp2,
        line=dict(color=C["tp2"], width=2, dash="dash"))
    fig.add_annotation(x=entry_ts, y=signal.tp2,
        text=f"TP2 {signal.tp2:.5f} ({signal.rr_tp2:.1f}R)",
        font=dict(size=9, color=C["tp2"]), showarrow=False,
        xanchor="left", yanchor="bottom" if is_long else "top",
        bgcolor="rgba(15,17,23,0.85)", borderpad=2)

    # Risk/reward zones
    fig.add_shape(type="rect", x0=entry_ts, x1=x1,
        y0=min(signal.entry, signal.sl), y1=max(signal.entry, signal.sl),
        fillcolor="rgba(252,92,101,0.07)", line=dict(width=0))
    fig.add_shape(type="rect", x0=entry_ts, x1=x1,
        y0=min(signal.entry, signal.tp2), y1=max(signal.entry, signal.tp2),
        fillcolor="rgba(38,222,129,0.05)", line=dict(width=0))

    # Entry marker
    fig.add_trace(go.Scatter(
        x=[entry_ts], y=[signal.entry], mode="markers",
        marker=dict(
            symbol="triangle-up" if is_long else "triangle-down",
            size=16, color=entry_colour,
            line=dict(color="white", width=1.5),
        ), showlegend=False))

    # Exit marker
    if trade and trade.exit_time and result not in ("OPEN",):
        exit_price = trade.exit_price if trade.exit_price else signal.entry
        fig.add_trace(go.Scatter(
            x=[trade.exit_time], y=[exit_price],
            mode="markers+text",
            marker=dict(
                symbol="x" if result in ("LOSS", "EXPIRED") else "star",
                size=14, color=res_col,
                line=dict(color="white", width=1),
            ),
            text=[result],
            textposition="top center" if is_long else "bottom center",
            textfont=dict(size=10, color=res_col),
            showlegend=False))
        fig.add_shape(type="line",
            x0=trade.exit_time, x1=trade.exit_time,
            y0=0, y1=1, xref="x", yref="paper",
            line=dict(color=res_col, width=1, dash="dot"), opacity=0.5)

    # Result badge
    badge = f"● {result}"
    if trade:
        pnl_str = f"  ${trade.pnl:+.2f}" if result not in ("EXPIRED", "OPEN") else ""
        badge = f"● {result}{pnl_str}  |  {trade.bars_held} bars"

    fig.add_annotation(x=0.99, y=0.98, xref="paper", yref="paper",
        text=badge, font=dict(size=13, color=res_col, family="monospace"),
        showarrow=False, xanchor="right", yanchor="top",
        bgcolor="rgba(15,17,23,0.9)",
        bordercolor=res_col, borderwidth=2, borderpad=6)

    title = (
        f"{signal.direction.upper()} {signal.instrument}  |  "
        f"Entry: {signal.entry:.5f}  |  SL: {signal.sl_pips:.0f}p  |  "
        f"RR: {signal.rr_tp2:.1f}:1  |  "
        f"Risk: ${signal.risk_amount:.0f}  |  {signal.entry_zone}"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#E8E8E8"), x=0.01),
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=height, margin=dict(l=60, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A1D2E", font_color="#E8E8E8"),
    )
    axis_style = dict(
        gridcolor=C["grid"], zerolinecolor=C["grid"],
        tickfont=dict(color="#AAA", size=10),
        showspikes=True, spikecolor="#555", spikethickness=1,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style, tickformat=".5f")
    return fig


def plot_all_signals(
    df: pd.DataFrame,
    signals: list[TradeSignal],
    trades: list | None = None,
    candles_before: int = 60,
    candles_after: int = 40,
    height: int = 620,
    save_html: str | None = "signals.html",
) -> None:
    if not signals:
        print("No signals to plot.")
        return

    html_parts = []
    for i, sig in enumerate(signals):
        trade = trades[i] if trades and i < len(trades) else None
        result = trade.result if trade else "OPEN"
        res_col = RESULT_COLOUR.get(result, "#888")

        fig = plot_signal(df, sig, trade, candles_before, candles_after, height)
        header = (
            f"<div style='display:flex;align-items:center;gap:16px;"
            f"padding:12px 16px;background:#1A1D2E;margin-top:24px;"
            f"border-left:4px solid {res_col}'>"
            f"<span style='color:#888;font-size:13px'>Signal {i+1}</span>"
            f"<span style='color:#E8E8E8;font-size:14px;font-weight:bold'>"
            f"{sig.direction.upper()} {sig.instrument}</span>"
            f"<span style='color:#888'>{sig.timestamp}</span>"
            f"<span style='color:{res_col};font-weight:bold'>{result}</span>"
            f"</div>"
        )
        html_parts.append(header)
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))

    wins = losses = partials = expired = 0
    total_pnl = 0.0
    if trades:
        for t in trades:
            if t.result == "WIN": wins += 1
            elif t.result == "LOSS": losses += 1
            elif t.result == "PARTIAL": partials += 1
            elif t.result == "EXPIRED": expired += 1
            total_pnl += t.pnl

    pnl_col = "#26DE81" if total_pnl >= 0 else "#FC5C65"
    summary = (
        f"<div style='background:#1A1D2E;padding:16px;margin-bottom:16px;"
        f"border-radius:8px;font-family:monospace'>"
        f"<span style='color:#E8E8E8;font-size:16px;font-weight:bold'>"
        f"ICT Backtest — {len(signals)} signals</span><br><br>"
        f"<span style='color:#26DE81'>✅ Wins: {wins}</span> &nbsp;"
        f"<span style='color:#FC5C65'>❌ Losses: {losses}</span> &nbsp;"
        f"<span style='color:#F7B731'>◑ Partials: {partials}</span> &nbsp;"
        f"<span style='color:#888'>⏱ Expired: {expired}</span> &nbsp;"
        f"<span style='color:{pnl_col}'>💰 P&L: ${total_pnl:+.2f}</span>"
        f"</div>"
    ) if trades else ""

    full_html = f"""<html>
<head><title>ICT Signals</title>
<style>body{{background:#0F1117;margin:0;padding:20px;font-family:monospace;}}</style>
</head>
<body>{summary}{''.join(html_parts)}</body></html>"""

    if save_html:
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Saved {len(signals)} signal charts → {save_html}")