"""
Calculate technical indicators (RSI, MACD, etc.) for Schedule 13D/G filings.

For each filing in scoring_model_with_tickers.csv:
- Calculate indicators as of filed_as_of_date + 1 day
- Requires historical data (typically 50-200 days before) to calculate indicators
- Outputs CSV with indicators added

Indicators calculated:
- RSI (14-day Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA (Simple Moving Averages: 20, 50, 200 day)
- Volume indicators (Volume vs 20-day average)
- Bollinger Bands
- ATR (Average True Range)
"""

import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: Required libraries not installed")
    print("Please run: pip install yfinance pandas numpy")
    exit(1)


# -----------------------
# Config
# -----------------------

BASE = Path("./sec_13dg_txt_2025").resolve()
IN_FILE = BASE / "scoring_model_with_tickers.csv"
OUT_FILE = BASE / "scoring_model_with_indicators.csv"

# Need historical data to calculate indicators
LOOKBACK_DAYS = 200  # days of historical data needed for 200-day SMA

# Rate limiting
DELAY_BETWEEN_REQUESTS = 0.5  # seconds


# -----------------------
# Technical Indicator Functions
# -----------------------

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    if len(prices) < slow + signal:
        return None, None, None
    
    prices_series = pd.Series(prices)
    exp1 = prices_series.ewm(span=fast, adjust=False).mean()
    exp2 = prices_series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return (
        round(macd.iloc[-1], 2) if len(macd) > 0 else None,
        round(signal_line.iloc[-1], 2) if len(signal_line) > 0 else None,
        round(histogram.iloc[-1], 2) if len(histogram) > 0 else None
    )


def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return round(np.mean(prices[-period:]), 2)


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return round(sma, 2), round(upper, 2), round(lower, 2)


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
        return None
    
    tr_list = []
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr = max(hl, hc, lc)
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return None
    
    atr = np.mean(tr_list[-period:])
    return round(atr, 2)


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date from YYYYMMDD format"""
    if not date_str or date_str.strip() == "":
        return None
    try:
        date_str = str(date_str).strip()
        return datetime.strptime(date_str, "%Y%m%d")
    except:
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except:
            return None


def get_indicators_for_ticker(ticker: str, target_date: datetime) -> Optional[Dict]:
    """
    Fetch historical data and calculate indicators as of target_date
    
    Returns dict with all calculated indicators
    """
    if not ticker or ticker.strip() == "":
        return None
    
    try:
        # Get data from lookback period to target date
        start_date = target_date - timedelta(days=LOOKBACK_DAYS)
        end_date = target_date + timedelta(days=5)  # buffer to ensure we get target date
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_str, end=end_str, interval="1d")
        
        if df.empty:
            return None
        
        # Find the row closest to target_date (or next trading day)
        target_date_str = target_date.strftime("%Y-%m-%d")
        df.index = pd.to_datetime(df.index).date
        
        # Get all dates up to and including target date
        available_dates = [d for d in df.index if d <= target_date.date()]
        if not available_dates:
            # Target date is before all available data
            return None
        
        # Use the last available date up to target_date
        calc_date = max(available_dates)
        calc_idx = df.index.get_loc(calc_date)
        
        # Extract price arrays up to calculation date
        closes = df['Close'].iloc[:calc_idx+1].values
        highs = df['High'].iloc[:calc_idx+1].values
        lows = df['Low'].iloc[:calc_idx+1].values
        volumes = df['Volume'].iloc[:calc_idx+1].values
        
        if len(closes) < 2:
            return None
        
        current_price = closes[-1]
        
        # Calculate indicators
        rsi = calculate_rsi(closes, period=14)
        macd, macd_signal, macd_hist = calculate_macd(closes)
        
        sma_20 = calculate_sma(closes, 20)
        sma_50 = calculate_sma(closes, 50)
        sma_200 = calculate_sma(closes, 200)
        
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(closes, 20, 2)
        
        atr = calculate_atr(highs, lows, closes, 14)
        
        # Volume metrics
        avg_volume_20 = calculate_sma(volumes, 20) if len(volumes) >= 20 else None
        volume_ratio = round(volumes[-1] / avg_volume_20, 2) if avg_volume_20 and avg_volume_20 > 0 else None
        
        # Price vs moving averages
        price_vs_sma20 = round((current_price / sma_20 - 1) * 100, 2) if sma_20 else None
        price_vs_sma50 = round((current_price / sma_50 - 1) * 100, 2) if sma_50 else None
        price_vs_sma200 = round((current_price / sma_200 - 1) * 100, 2) if sma_200 else None
        
        # Bollinger Band position
        bb_position = None
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_position = round((current_price - bb_lower) / (bb_upper - bb_lower) * 100, 2)
        
        return {
            "calculation_date": calc_date.strftime("%Y-%m-%d"),
            "price_at_filing": round(current_price, 2),
            "rsi_14": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "price_vs_sma20_pct": price_vs_sma20,
            "price_vs_sma50_pct": price_vs_sma50,
            "price_vs_sma200_pct": price_vs_sma200,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "bb_position_pct": bb_position,
            "atr_14": atr,
            "volume": int(volumes[-1]) if len(volumes) > 0 else None,
            "avg_volume_20": int(avg_volume_20) if avg_volume_20 else None,
            "volume_ratio": volume_ratio,
        }
    
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


# -----------------------
# Main Processing
# -----------------------

def main():
    if not IN_FILE.exists():
        print(f"ERROR: Input file not found: {IN_FILE}")
        return
    
    print("=" * 70)
    print("TECHNICAL INDICATORS CALCULATOR")
    print("=" * 70)
    print(f"Input:  {IN_FILE}")
    print(f"Output: {OUT_FILE}")
    print(f"Calculation date: filed_as_of_date + 1 day")
    print()
    
    # Read input CSV
    rows_processed = 0
    rows_with_ticker = 0
    indicators_calculated = 0
    indicators_failed = 0
    
    output_rows = []
    
    with IN_FILE.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Add new indicator columns
        indicator_columns = [
            "indicator_calc_date", "price_at_filing",
            "rsi_14", "macd", "macd_signal", "macd_histogram",
            "sma_20", "sma_50", "sma_200",
            "price_vs_sma20_pct", "price_vs_sma50_pct", "price_vs_sma200_pct",
            "bb_upper", "bb_middle", "bb_lower", "bb_position_pct",
            "atr_14", "volume", "avg_volume_20", "volume_ratio"
        ]
        
        output_fieldnames = list(fieldnames) + indicator_columns
        
        for row in reader:
            rows_processed += 1
            
            ticker = row.get('ticker', '').strip().strip('"')
            filed_date_str = row.get('filed_as_of_date', '')
            issuer_name = row.get('issuer_name', '')
            accession = row.get('accession_number', '')
            
            # Initialize indicator columns
            for col in indicator_columns:
                row[col] = ''
            
            # Skip if no ticker
            if not ticker:
                output_rows.append(row)
                continue
            
            rows_with_ticker += 1
            
            # Parse filing date
            filed_date = parse_date(filed_date_str)
            if not filed_date:
                print(f"  SKIP [{rows_processed}]: Invalid date for {ticker}")
                output_rows.append(row)
                continue
            
            # Calculate for filing date + 1
            calc_date = filed_date + timedelta(days=1)
            
            print(f"[{rows_processed}] {ticker} ({issuer_name[:40]}...) | Filing: {filed_date_str} -> Calc: {calc_date.strftime('%Y-%m-%d')}")
            
            # Get indicators
            indicators = get_indicators_for_ticker(ticker, calc_date)
            
            if indicators:
                # Add indicators to row
                for key, value in indicators.items():
                    row[key.replace('calculation_date', 'indicator_calc_date')] = value if value is not None else ''
                
                indicators_calculated += 1
                print(f"  OK: RSI={indicators.get('rsi_14')}, Price=${indicators.get('price_at_filing')}")
            else:
                indicators_failed += 1
                print(f"  SKIP: No data available")
            
            output_rows.append(row)
            
            # Rate limiting
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            # Progress update
            if rows_processed % 100 == 0:
                print(f"  Progress: {rows_processed} rows | {indicators_calculated} calculated | {indicators_failed} failed")
    
    # Write output CSV
    print(f"\nWriting output to {OUT_FILE}...")
    with OUT_FILE.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows processed:     {rows_processed:,}")
    print(f"Rows with tickers:        {rows_with_ticker:,}")
    print(f"Indicators calculated:    {indicators_calculated:,} ({indicators_calculated/rows_with_ticker*100:.1f}%)")
    print(f"Failed to calculate:      {indicators_failed:,}")
    print()
    print(f"Output saved to: {OUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
