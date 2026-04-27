import yfinance as yf
import pandas as pd

def fetch_eurusd(interval: str = "1h", period: str = "30d") -> pd.DataFrame:
    """
    Fetch EURUSD OHLCV data from Yahoo Finance.
    
    Args:
        interval: candle size - "15m", "1h", "4h", "1d"
        period:   how far back  - "7d", "30d", "60d", "1y"
    
    Returns:
        DataFrame with Open, High, Low, Close, Volume
    """
    print(f"Fetching EURUSD {interval} data for last {period}...")
    
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(interval=interval, period=period)
    
    # Clean up columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index)
    
    # Round to 5 decimal places (standard for Forex)
    df = df.round(5)
    
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a clean summary of the fetched data."""
    
    print("\n" + "="*60)
    print("EURUSD DATA SUMMARY")
    print("="*60)
    print(f"Total candles    : {len(df)}")
    print(f"From             : {df.index[0]}")
    print(f"To               : {df.index[-1]}")
    print(f"Highest price    : {df['High'].max()}")
    print(f"Lowest price     : {df['Low'].min()}")
    print(f"Latest close     : {df['Close'].iloc[-1]}")
    print(f"Latest open      : {df['Open'].iloc[-1]}")
    print("="*60)
    
    print("\nLAST 10 CANDLES:")
    print("-"*60)
    print(df.tail(10).to_string())
    print("-"*60)


if __name__ == "__main__":
    # Fetch 1H data for last 30 days
    df = fetch_eurusd(interval="1h", period="30d")
    
    # Print summary
    print_summary(df)
    
    print("\nData fetch complete. Shape:", df.shape)