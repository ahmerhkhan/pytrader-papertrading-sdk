"""
Example: Running backtests with CSV data files.

This example demonstrates how to use CSV files for backtesting,
supporting various formats including simple (symbol, timestamp, price, volume).
"""

from pathlib import Path
from pytrader import Trader, Strategy
from pytrader.indicators import SMA


class SimpleMomentumStrategy(Strategy):
    """Simple momentum strategy using moving average."""
    
    def on_start(self):
        self.sma_period = 20
        self.position_size = 100
    
    def on_data(self, data):
        for symbol, df in data.items():
            if len(df) < self.sma_period:
                continue
            
            sma = SMA(df['close'], self.sma_period)
            current_price = df['close'].iloc[-1]
            
            if current_price > sma.iloc[-1]:
                self.buy(symbol, self.position_size)
            else:
                self.sell(symbol, self.position_size)


def example_1_simple_csv_format():
    """
    Example 1: CSV with simple format (symbol, timestamp, price, volume)
    
    CSV format:
    symbol,timestamp,price,volume
    OGDC,2024-01-01 09:00:00,95.50,1000000
    OGDC,2024-01-01 09:15:00,95.75,1200000
    ...
    """
    print("\n=== Example 1: Simple CSV Format ===")
    
    # Create a sample CSV file (in real usage, you'd have your own CSV)
    csv_path = Path("example_data.csv")
    
    # Create sample data
    sample_data = """symbol,timestamp,price,volume
OGDC,2024-01-01 09:00:00,95.50,1000000
OGDC,2024-01-01 09:15:00,95.75,1200000
OGDC,2024-01-01 09:30:00,96.00,1100000
OGDC,2024-01-01 09:45:00,95.25,1300000
OGDC,2024-01-01 10:00:00,96.50,1500000
"""
    csv_path.write_text(sample_data)
    
    trader = Trader(
        strategy=SimpleMomentumStrategy,
        symbols=["OGDC"],
        initial_cash=1_000_000.0,
    )
    
    # Run backtest with CSV - auto-detects column mapping
    result = trader.run_backtest(
        csv_path=csv_path,
        # No API token needed when using CSV!
    )
    
    print(f"Final Portfolio Value: {result['summary']['final_portfolio_value']:,.2f}")
    print(f"Total Return: {result['summary']['total_return_pct']:.2f}%")
    print(f"Total Trades: {len(result['trades'])}")
    
    # Cleanup
    csv_path.unlink()


def example_2_custom_column_mapping():
    """
    Example 2: CSV with custom column names
    
    CSV format:
    ticker,ts,close_price,vol
    OGDC,2024-01-01 09:00:00,95.50,1000000
    ...
    """
    print("\n=== Example 2: Custom Column Mapping ===")
    
    csv_path = Path("example_data_custom.csv")
    
    sample_data = """ticker,ts,close_price,vol
OGDC,2024-01-01 09:00:00,95.50,1000000
OGDC,2024-01-01 09:15:00,95.75,1200000
OGDC,2024-01-01 09:30:00,96.00,1100000
"""
    csv_path.write_text(sample_data)
    
    trader = Trader(
        strategy=SimpleMomentumStrategy,
        symbols=["OGDC"],
        initial_cash=1_000_000.0,
    )
    
    # Specify custom column mapping
    result = trader.run_backtest(
        csv_path=csv_path,
        csv_column_mapping={
            "timestamp": "ts",      # Map "ts" column to timestamp
            "price": "close_price", # Map "close_price" to price
            "volume": "vol",        # Map "vol" to volume
            "symbol": "ticker",     # Map "ticker" to symbol (if filtering needed)
        },
    )
    
    print(f"Final Portfolio Value: {result['summary']['final_portfolio_value']:,.2f}")
    
    # Cleanup
    csv_path.unlink()


def example_3_ohlcv_format():
    """
    Example 3: CSV with full OHLCV format
    
    CSV format:
    symbol,datetime,open,high,low,close,volume
    OGDC,2024-01-01 09:00:00,95.00,96.00,94.50,95.50,1000000
    ...
    """
    print("\n=== Example 3: OHLCV Format ===")
    
    csv_path = Path("example_data_ohlcv.csv")
    
    sample_data = """symbol,datetime,open,high,low,close,volume
OGDC,2024-01-01 09:00:00,95.00,96.00,94.50,95.50,1000000
OGDC,2024-01-01 09:15:00,95.50,96.25,95.00,95.75,1200000
OGDC,2024-01-01 09:30:00,95.75,96.50,95.25,96.00,1100000
"""
    csv_path.write_text(sample_data)
    
    trader = Trader(
        strategy=SimpleMomentumStrategy,
        symbols=["OGDC"],
        initial_cash=1_000_000.0,
    )
    
    # Auto-detects OHLCV format
    result = trader.run_backtest(
        csv_path=csv_path,
    )
    
    print(f"Final Portfolio Value: {result['summary']['final_portfolio_value']:,.2f}")
    
    # Cleanup
    csv_path.unlink()


def example_4_multiple_symbols_in_csv():
    """
    Example 4: CSV with multiple symbols
    
    CSV format:
    symbol,timestamp,price,volume
    OGDC,2024-01-01 09:00:00,95.50,1000000
    HBL,2024-01-01 09:00:00,150.00,2000000
    OGDC,2024-01-01 09:15:00,95.75,1200000
    HBL,2024-01-01 09:15:00,151.00,1800000
    ...
    """
    print("\n=== Example 4: Multiple Symbols in CSV ===")
    
    csv_path = Path("example_data_multi.csv")
    
    sample_data = """symbol,timestamp,price,volume
OGDC,2024-01-01 09:00:00,95.50,1000000
HBL,2024-01-01 09:00:00,150.00,2000000
OGDC,2024-01-01 09:15:00,95.75,1200000
HBL,2024-01-01 09:15:00,151.00,1800000
OGDC,2024-01-01 09:30:00,96.00,1100000
HBL,2024-01-01 09:30:00,152.00,1900000
"""
    csv_path.write_text(sample_data)
    
    trader = Trader(
        strategy=SimpleMomentumStrategy,
        symbols=["OGDC", "HBL"],  # Multiple symbols
        initial_cash=1_000_000.0,
    )
    
    # CSV provider will filter by symbol automatically
    result = trader.run_backtest(
        csv_path=csv_path,
        csv_column_mapping={
            "timestamp": "timestamp",
            "price": "price",
            "volume": "volume",
            "symbol": "symbol",  # Important: specify symbol column
        },
    )
    
    print(f"Final Portfolio Value: {result['summary']['final_portfolio_value']:,.2f}")
    print(f"Trades for OGDC: {sum(1 for t in result['trades'] if t.get('symbol') == 'OGDC')}")
    print(f"Trades for HBL: {sum(1 for t in result['trades'] if t.get('symbol') == 'HBL')}")
    
    # Cleanup
    csv_path.unlink()


if __name__ == "__main__":
    print("CSV Backtest Examples")
    print("=" * 50)
    
    try:
        example_1_simple_csv_format()
        example_2_custom_column_mapping()
        example_3_ohlcv_format()
        example_4_multiple_symbols_in_csv()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

