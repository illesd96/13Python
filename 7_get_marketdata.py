"""
Fetch daily market data for Schedule 13D/G filings.

For each filing in scoring_model_with_tickers.csv:
- Get 1 month of data BEFORE filed_as_of_date
- Get 3 months of data AFTER filed_as_of_date
- Save as JSON with accession_number as key

Market data includes: Open, High, Low, Close, Volume
"""

import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: Required libraries not installed")
    print("Please run: pip install yfinance pandas")
    exit(1)


# -----------------------
# Config
# -----------------------

BASE = Path("./sec_13dg_txt_2025").resolve()
IN_FILE = BASE / "scoring_model_with_tickers.csv"
OUT_FILE = BASE / "market_data.json"

# Date windows
DAYS_BEFORE = 30  # 1 month before filing
DAYS_AFTER = 90   # 3 months after filing

# Rate limiting (be nice to Yahoo Finance)
DELAY_BETWEEN_REQUESTS = 0.5  # seconds


# -----------------------
# Helper Functions
# -----------------------

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date from YYYYMMDD format"""
    if not date_str or date_str.strip() == "":
        return None
    try:
        # Handle YYYYMMDD format
        date_str = str(date_str).strip()
        return datetime.strptime(date_str, "%Y%m%d")
    except:
        try:
            # Try MM/DD/YYYY format
            return datetime.strptime(date_str, "%m/%d/%Y")
        except:
            return None


def get_market_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[List[Dict]]:
    """
    Fetch daily market data from Yahoo Finance
    
    Returns list of daily data dicts with:
    - date, open, high, low, close, volume
    """
    if not ticker or ticker.strip() == "":
        return None
    
    try:
        # Format dates for yfinance
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_str, end=end_str, interval="1d")
        
        if df.empty:
            return None
        
        # Convert to list of dicts
        data = []
        for date, row in df.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row['Open']), 2) if 'Open' in row and not pd.isna(row['Open']) else None,
                "high": round(float(row['High']), 2) if 'High' in row and not pd.isna(row['High']) else None,
                "low": round(float(row['Low']), 2) if 'Low' in row and not pd.isna(row['Low']) else None,
                "close": round(float(row['Close']), 2) if 'Close' in row and not pd.isna(row['Close']) else None,
                "volume": int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else None,
            })
        
        return data if data else None
    
    except Exception as e:
        print(f"    ERROR fetching {ticker}: {e}")
        return None


# -----------------------
# Main Processing
# -----------------------

def main():
    if not IN_FILE.exists():
        print(f"ERROR: Input file not found: {IN_FILE}")
        return
    
    print("=" * 70)
    print("MARKET DATA FETCHER FOR SCHEDULE 13D/G FILINGS")
    print("=" * 70)
    print(f"Input:  {IN_FILE}")
    print(f"Output: {OUT_FILE}")
    print(f"Window: {DAYS_BEFORE} days before, {DAYS_AFTER} days after filing")
    print()
    
    # Load existing data if any (to resume from failures)
    existing_data = {}
    if OUT_FILE.exists():
        try:
            with OUT_FILE.open('r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing entries from {OUT_FILE.name}")
        except:
            pass
    
    # Read CSV and fetch market data
    rows_processed = 0
    rows_with_ticker = 0
    data_fetched = 0
    data_failed = 0
    skipped = 0
    
    market_data = existing_data.copy()
    
    with IN_FILE.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            rows_processed += 1
            
            accession = row.get('accession_number', '')
            ticker = row.get('ticker', '').strip().strip('"')
            filed_date_str = row.get('filed_as_of_date', '')
            issuer_name = row.get('issuer_name', '')
            
            # Skip if no ticker
            if not ticker:
                continue
            
            rows_with_ticker += 1
            
            # Skip if already fetched
            if accession in market_data:
                skipped += 1
                if rows_processed % 100 == 0:
                    print(f"[{rows_processed}/{rows_with_ticker} with ticker] Skipped {skipped}, Fetched {data_fetched}, Failed {data_failed}")
                continue
            
            # Parse filing date
            filed_date = parse_date(filed_date_str)
            if not filed_date:
                print(f"  SKIP: Invalid date for {accession}: {filed_date_str}")
                continue
            
            # Calculate date range
            start_date = filed_date - timedelta(days=DAYS_BEFORE)
            end_date = filed_date + timedelta(days=DAYS_AFTER)
            
            # Fetch market data
            print(f"[{rows_processed}] Fetching {ticker} ({issuer_name[:40]}...)")
            print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            daily_data = get_market_data(ticker, start_date, end_date)
            
            if daily_data:
                market_data[accession] = {
                    "accession_number": accession,
                    "ticker": ticker,
                    "issuer_name": issuer_name,
                    "filed_as_of_date": filed_date_str,
                    "filed_date_parsed": filed_date.strftime("%Y-%m-%d"),
                    "data_start_date": start_date.strftime("%Y-%m-%d"),
                    "data_end_date": end_date.strftime("%Y-%m-%d"),
                    "days_before_filing": DAYS_BEFORE,
                    "days_after_filing": DAYS_AFTER,
                    "total_trading_days": len(daily_data),
                    "daily_data": daily_data
                }
                data_fetched += 1
                print(f"  OK: Fetched {len(daily_data)} trading days")
            else:
                data_failed += 1
                print(f"  SKIP: No data available")
            
            # Rate limiting
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            # Save progress every 50 rows
            if rows_processed % 50 == 0:
                with OUT_FILE.open('w', encoding='utf-8') as f:
                    json.dump(market_data, f, indent=2)
                print(f"  >> Progress saved ({len(market_data)} entries)")
    
    # Final save
    with OUT_FILE.open('w', encoding='utf-8') as f:
        json.dump(market_data, f, indent=2)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows processed:     {rows_processed:,}")
    print(f"Rows with tickers:        {rows_with_ticker:,}")
    print(f"Market data fetched:      {data_fetched:,}")
    print(f"Failed to fetch:          {data_failed:,}")
    print(f"Skipped (already exist):  {skipped:,}")
    print(f"Total in JSON:            {len(market_data):,}")
    print()
    print(f"Output saved to: {OUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
