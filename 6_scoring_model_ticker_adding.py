"""
Add ticker symbols to scoring_model.csv by matching issuer_name with company_tickers.csv

Matching strategy:
1. First try exact CIK match (most reliable)
2. If no CIK match, use fuzzy string matching on company names (95% similarity threshold)
3. Handle case-insensitive matching and common variations

Output: scoring_model_with_tickers.csv
"""

import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from rapidfuzz import fuzz, process
    USE_RAPIDFUZZ = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        USE_RAPIDFUZZ = False
    except ImportError:
        print("ERROR: Please install rapidfuzz or fuzzywuzzy:")
        print("  pip install rapidfuzz")
        print("  or")
        print("  pip install fuzzywuzzy python-Levenshtein")
        exit(1)


# -----------------------
# Config
# -----------------------

BASE = Path("./sec_13dg_txt_2025").resolve()
TICKERS_FILE = Path("./company_tickers.csv").resolve()
IN_FILE = BASE / "scoring_model.csv"
OUT_FILE = BASE / "scoring_model_with_tickers.csv"

FUZZY_THRESHOLD = 95  # Minimum similarity score (0-100)


# -----------------------
# Helper Functions
# -----------------------

def normalize_cik(cik) -> Optional[str]:
    """Normalize CIK to 10-digit zero-padded format"""
    if not cik or str(cik).strip() == "":
        return None
    try:
        # Remove any non-digit characters and convert to int
        cik_int = int(re.sub(r'\D', '', str(cik)))
        return str(cik_int).zfill(10)
    except (ValueError, TypeError):
        return None


def normalize_name(name: str) -> str:
    """Normalize company name for better matching"""
    if not name:
        return ""
    
    # Convert to uppercase
    name = str(name).upper().strip()
    
    # Remove common suffixes and punctuation
    name = re.sub(r'\s+(INC\.?|CORP\.?|LTD\.?|LLC|LP|L\.P\.)\s*$', '', name)
    name = re.sub(r'\s+(CORPORATION|INCORPORATED|COMPANY|LIMITED)\s*$', '', name)
    
    # Remove special characters but keep spaces
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def load_ticker_lookup(tickers_file: Path) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """
    Load company tickers and create lookup dictionaries
    
    Returns:
        - cik_to_ticker: {normalized_cik: ticker}
        - name_to_data: {normalized_name: (ticker, original_name)}
    """
    cik_to_ticker = {}
    name_to_data = {}
    
    with tickers_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cik = normalize_cik(row.get('cik'))
            ticker = (row.get('ticker') or '').strip()
            company_name = (row.get('company_name') or '').strip()
            
            if not ticker:
                continue
            
            # CIK lookup
            if cik:
                cik_to_ticker[cik] = ticker
            
            # Name lookup (normalized)
            if company_name:
                norm_name = normalize_name(company_name)
                if norm_name:
                    name_to_data[norm_name] = (ticker, company_name)
    
    print(f"Loaded {len(cik_to_ticker)} CIK mappings and {len(name_to_data)} name mappings")
    return cik_to_ticker, name_to_data


def find_ticker_by_name(issuer_name: str, name_to_data: Dict[str, Tuple[str, str]], 
                       threshold: int = FUZZY_THRESHOLD) -> Optional[Tuple[str, str, int]]:
    """
    Find ticker using fuzzy name matching
    
    Returns:
        (ticker, matched_name, score) or None
    """
    if not issuer_name:
        return None
    
    norm_issuer = normalize_name(issuer_name)
    if not norm_issuer:
        return None
    
    # Try exact match first (after normalization)
    if norm_issuer in name_to_data:
        ticker, orig_name = name_to_data[norm_issuer]
        return (ticker, orig_name, 100)
    
    # Fuzzy match
    choices = list(name_to_data.keys())
    
    if USE_RAPIDFUZZ:
        # rapidfuzz returns (choice, score, index)
        result = process.extractOne(
            norm_issuer, 
            choices, 
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        if result:
            matched_name, score, _ = result
            ticker, orig_name = name_to_data[matched_name]
            return (ticker, orig_name, score)
    else:
        # fuzzywuzzy returns (choice, score)
        result = process.extractOne(
            norm_issuer, 
            choices, 
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        if result:
            matched_name, score = result
            ticker, orig_name = name_to_data[matched_name]
            return (ticker, orig_name, score)
    
    return None


# -----------------------
# Main Processing
# -----------------------

def main():
    if not TICKERS_FILE.exists():
        print(f"ERROR: Tickers file not found: {TICKERS_FILE}")
        return
    
    if not IN_FILE.exists():
        print(f"ERROR: Input file not found: {IN_FILE}")
        return
    
    print("=" * 70)
    print("TICKER MATCHING SCRIPT")
    print("=" * 70)
    print(f"Input:   {IN_FILE}")
    print(f"Tickers: {TICKERS_FILE}")
    print(f"Output:  {OUT_FILE}")
    print(f"Fuzzy threshold: {FUZZY_THRESHOLD}%")
    print()
    
    # Load ticker lookup tables
    cik_to_ticker, name_to_data = load_ticker_lookup(TICKERS_FILE)
    
    # Process scoring model
    rows_processed = 0
    matched_by_cik = 0
    matched_by_name = 0
    no_match = 0
    
    output_rows = []
    
    with IN_FILE.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        # Add new columns
        output_fieldnames = list(fieldnames) + ['ticker', 'ticker_match_method', 'ticker_match_score']
        
        for row in reader:
            rows_processed += 1
            
            issuer_cik = row.get('issuer_cik', '')
            issuer_name = row.get('issuer_name', '')
            
            ticker = None
            match_method = None
            match_score = None
            
            # Strategy 1: Exact CIK match
            norm_cik = normalize_cik(issuer_cik)
            if norm_cik and norm_cik in cik_to_ticker:
                ticker = cik_to_ticker[norm_cik]
                match_method = 'CIK'
                match_score = 100
                matched_by_cik += 1
            
            # Strategy 2: Fuzzy name match
            elif issuer_name:
                result = find_ticker_by_name(issuer_name, name_to_data, FUZZY_THRESHOLD)
                if result:
                    ticker, matched_name, score = result
                    match_method = 'NAME_FUZZY'
                    match_score = score
                    matched_by_name += 1
                    
                    if rows_processed <= 10 or score < 98:
                        print(f"  Fuzzy match ({score}%): '{issuer_name}' -> '{matched_name}' [{ticker}]")
            
            # No match found
            if not ticker:
                no_match += 1
                if no_match <= 10:
                    print(f"  NO MATCH: CIK={issuer_cik}, Name='{issuer_name}'")
            
            # Add new fields to row
            row['ticker'] = ticker or ''
            row['ticker_match_method'] = match_method or ''
            row['ticker_match_score'] = match_score if match_score is not None else ''
            
            output_rows.append(row)
            
            # Progress indicator
            if rows_processed % 500 == 0:
                print(f"Processed {rows_processed} rows...")
    
    # Write output
    with OUT_FILE.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows processed:    {rows_processed:,}")
    print(f"Matched by CIK:          {matched_by_cik:,} ({matched_by_cik/rows_processed*100:.1f}%)")
    print(f"Matched by fuzzy name:   {matched_by_name:,} ({matched_by_name/rows_processed*100:.1f}%)")
    print(f"No match found:          {no_match:,} ({no_match/rows_processed*100:.1f}%)")
    print(f"Total matched:           {matched_by_cik + matched_by_name:,} ({(matched_by_cik + matched_by_name)/rows_processed*100:.1f}%)")
    print()
    print(f"Output written to: {OUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
