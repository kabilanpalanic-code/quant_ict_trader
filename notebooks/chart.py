import plotly.io as pio
pio.renderers.default = "jupyterlab"

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go


def fetch_eurusd(interval: str = "1h", period: str = "30d") -> pd.DataFrame:
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(interval=interval, period=period)
    df = df[["Open", "High", "Low", "Close"]].round(5)
    return df


def plot_candlestick(df: pd.DataFrame) -> None:
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="EURUSD"
        )
    ])

    fig.update_layout(
        title="EURUSD 1H Chart — Last 30 Days",
        yaxis_title="Price",
        xaxis_title="Time",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
    )

    fig.show()


if __name__ == "__main__":
    print("Fetching EURUSD data...")
    df = fetch_eurusd(interval="1h", period="30d")
    print(f"Got {len(df)} candles. Opening chart...")
    plot_candlestick(df)