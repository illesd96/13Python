"""
Add missing tickers to company_tickers.csv for companies that couldn't be matched.
This script contains manually researched tickers for common US-listed companies.
"""

import csv
from pathlib import Path
from datetime import datetime

TICKERS_FILE = Path("company_tickers.csv")
UNMATCHED_FILE = Path("unmatched_companies.csv")

# Manually researched tickers for missing companies
# Format: (CIK, Ticker, Company Name)
MISSING_TICKERS = [
    # Major publicly traded companies
    ("1804591", "ME", "23andMe Holding Co."),
    ("1459417", "TWOU", "2U, Inc."),
    ("9984", "B", "Barnes Group Inc."),
    ("72333", "JWN", "Nordstrom Inc"),
    ("89140", "SVT", "SERVOTRONICS INC"),
    ("727207", "AXDX", "Accelerate Diagnostics, Inc."),
    ("1481646", "ACCD", "Accolade, Inc."),
    ("1621227", "ADAP", "Adaptimmune Therapeutics plc"),
    ("894081", "ATSG", "Air Transport Services Group, Inc."),
    ("1398987", "HOUS", "Anywhere Real Estate Inc"),
    ("1865187", "ARIS", "Aris Water Solutions, Inc."),
    ("1897982", "AZPN", "Aspen Technology, Inc."),
    ("704562", "CDMO", "Avid Bioservices, Inc."),
    ("1725872", "BMTX", "BM Technologies, Inc."),
    ("1313275", "BCOV", "BRIGHTCOVE INC."),
    ("1378992", "BERY", "Berry Global Group Inc"),
    ("1807427", "OBDC", "Blue Owl Capital Corp III"),
    ("1498233", "CPTN", "CEPTON, INC."),
    ("1651407", "CKPT", "Checkpoint Therapeutics, Inc."),
    ("1593222", "CIO", "City Office REIT Inc"),
    ("1101396", "DLA", "DELTA APPAREL, INC"),
    ("852772", "DENN", "Denny's Corp"),
    ("1754820", "DM", "Desktop Metal, Inc."),
    ("1082038", "DRRX", "Durect Corp."),
    ("1783032", "ELEV", "Elevation Oncology Inc"),
    ("1592000", "ENLC", "EnLink Midstream, LLC"),
    ("1766363", "EDR", "Endeavor Group Holdings, Inc."),
    ("1868912", "ENFN", "Enfusion, Inc."),
    ("1363829", "ESGR", "Enstar Group Limited"),
    ("803649", "EQC", "Equity Commonwealth"),
    ("947559", "FBMS", "First Bancshares Inc/The"),
    ("850209", "FL", "Foot Locker Inc"),
    ("1600438", "GMS", "GMS INC."),
    ("1821160", "GHLD", "Guild Holdings Co"),
    ("1339605", "HEES", "H&E Equipment Services, Inc."),
    ("921183", "HMNF", "HMN Financial Inc"),
    ("1501134", "NVTA", "Invitae Corp"),
    ("1721741", "LAZY", "Lazydays Holdings, Inc."),
    ("1828811", "LICY", "Li-Cycle Holdings Corp."),
    ("1606745", "LTRPA", "Liberty TripAdvisor Holdings, Inc."),
    ("1239819", "LUNA", "Luna Innovations Incorporated"),
    ("1807846", "ML", "MONEYLION INC."),
    ("1267813", "MRNS", "Marinus Pharmaceuticals, Inc."),
    ("1816613", "MKFG", "Markforged Holding Corp"),
    ("1220754", "MODV", "ModivCare Inc."),
    ("1532961", "NVEE", "NV5 Global, Inc."),
    ("1382821", "RDFN", "Redfin Corp"),
    ("1479290", "RVNC", "Revance Therapeutics, Inc."),
    ("1944705", "GEAR", "Revelyst Inc"),
    ("1597553", "SAGE", "Sage Therapeutics, Inc."),
    ("1468666", "SCWX", "SecureWorks Corp."),
    ("915358", "SGMA", "Sigmatron International, Inc."),
    ("1850906", "OMIC", "Singular Genomics Systems, Inc."),
    ("1689731", "SSBK", "Southern States Bancshares, Inc."),
    ("1773427", "SWTX", "SpringWorks Therapeutics Inc"),
    ("914712", "STCN", "Steel Connect, Inc."),
    ("96793", "SSY", "SunLink Health Systems, Inc."),
    ("1814550", "SRZN", "Surrozen, Inc."),
    ("814052", "TEF", "Telefonica, S.A."),
    ("1871149", "RGF", "The Real Good Food Company, Inc."),
    ("1923840", "THRD", "Third Harmonic Bio, Inc."),
    ("1552800", "TILE", "Tile Shop Holdings, Inc."),
    ("931584", "USAP", "Universal Stainless & Alloy Products Inc"),
    ("1874944", "VCSA", "Vacasa, Inc."),
    ("1166388", "VRNT", "Verint Systems Inc"),
    ("1501570", "VBTX", "Veritex Holdings Inc"),
    ("1657312", "VRNA", "Verona Pharma plc"),
    ("890447", "VTNR", "Vertex Energy Inc."),
    ("1840574", "VERV", "Verve Therapeutics, Inc."),
    ("1616318", "VSTO", "Vista Outdoor Inc."),
    ("1959348", "KLG", "WK Kellogg Co"),
    ("1013706", "WHLM", "Wilhelmina International, Inc."),
    ("1954042", "ZK", "ZEEKR Intelligent Technology Holding Limited"),
    ("1101026", "ZIVO", "Zivo Bioscience, Inc."),
    ("1423774", "ZUO", "Zuora, Inc."),
    ("1604950", "SCPH", "scPharmaceuticals Inc."),
    
    # ETF and Fund tickers
    ("1710607", "CBSE", "AMERICAN CENTURY ETF TRUST"),
    ("1503123", "DBMF", "DBX ETF TRUST"),
    ("1540305", "KFYP", "ETF Series Solutions"),
    ("1761055", "IBIT", "BlackRock ETF Trust"),
    ("1804196", "BKAG", "BlackRock ETF Trust II"),
    ("1232860", "MUI", "Blackrock Municipal Income Fund Inc."),
    ("1883172", "DOCU", "DoubleLine ETF Trust"),
    ("1424212", "FPX", "FIRST TRUST EXCHANGE-TRADED FUND III"),
    ("1667919", "FV", "FIRST TRUST EXCHANGE-TRADED FUND VIII"),
    ("1849998", "FHDG", "Federated Hermes ETF Trust"),
    ("1383496", "FXD", "First Trust Exchange-Traded AlphaDEX(R) Fund"),
    ("1510337", "FAD", "First Trust Exchange-Traded AlphaDEX(R) Fund II"),
    ("1517936", "FTSL", "First Trust Exchange-Traded Fund IV"),
    ("1552740", "FTHI", "First Trust Exchange-Traded Fund VI"),
    ("1392994", "FSCO", "First Trust Specialty Finance and Financial Opportunities Fund"),
    ("1415726", "BALT", "INNOVATOR ETFS TRUST"),
    ("1209466", "QQQ", "INVESCO EXCHANGE-TRADED FUND TRUST"),
    ("1371571", "UUP", "Invesco DB US Dollar Index Trust"),
    ("1378872", "BKLN", "Invesco Exchange-Traded Fund Trust II"),
    ("1698508", "IHIT", "Invesco High Income 2024 Target Term Fund"),
    ("1635073", "NUAG", "Nushares ETF Trust"),
    ("1258943", "MAV", "PIONEER MUNICIPAL HIGH INCOME ADVANTAGE FUND, INC."),
    ("1223026", "MHI", "PIONEER MUNICIPAL HIGH INCOME FUND, INC."),
    ("1864290", "MIO", "Pioneer Municipal High Income Opportunities Fund, Inc"),
    ("1064642", "WIP", "SPDR FTSE International Government Inflation-Protected Bond ETF"),
    ("1728683", "SGDJ", "SPROTT FUNDS TRUST"),
    ("1547158", "NDP", "TORTOISE ENERGY INDEPENDENCE FUND, INC."),
    ("1526329", "TTP", "TORTOISE PIPELINE & ENERGY FUND, INC."),
    ("1408201", "TPZ", "TORTOISE POWER & ENERGY INFRASTRUCTURE FUND INC"),
    ("1742912", "RSPA", "Tidal ETF Trust"),
    ("1490286", "NTG", "Tortoise Midstream Energy Fund, Inc."),
    ("1552947", "FBCV", "Two Roads Shared Trust"),
    ("1100663", "IVV", "iSHARES TRUST"),
    ("1524513", "ITOT", "iShares U.S. ETF Trust"),
    ("930667", "EFA", "iShares, Inc."),
    
    # Additional companies
    ("1522860", "AFIB", "Acutus Medical, Inc."),
    ("1464865", "ACDH", "Accredited Solutions, Inc."),
    ("320340", "CCRD", "CoreCard Corp"),
    ("1238631", "CVHH", "Curative Health Services Inc"),
    ("783005", "EMMS", "EMMIS CORP"),
    ("740663", "FLIC", "First of Long Island Corp/The"),
    ("1366561", "SMAR", "SMARTSHEET INC"),
    ("924717", "SRDX", "SURMODICS, INC."),
    ("1163302", "X", "UNITED STATES STEEL CORP"),
]

def load_existing_ciks(tickers_file):
    """Load existing CIKs to avoid duplicates"""
    existing = set()
    if not tickers_file.exists():
        return existing
    
    with tickers_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cik = row.get('cik', '').strip().strip('"')
            if cik:
                # Normalize CIK
                try:
                    cik_int = int(cik)
                    cik_norm = str(cik_int).zfill(10)
                    existing.add(cik_norm)
                except:
                    pass
    return existing

def main():
    print("=" * 70)
    print("ADDING MISSING TICKERS TO company_tickers.csv")
    print("=" * 70)
    
    # Load existing CIKs
    existing_ciks = load_existing_ciks(TICKERS_FILE)
    print(f"Found {len(existing_ciks)} existing CIKs in {TICKERS_FILE.name}")
    
    # Filter out already existing entries
    new_entries = []
    skipped = 0
    for cik, ticker, name in MISSING_TICKERS:
        cik_norm = str(int(cik)).zfill(10)
        if cik_norm in existing_ciks:
            skipped += 1
            continue
        new_entries.append((cik_norm, ticker, name))
    
    print(f"Adding {len(new_entries)} new entries (skipped {skipped} duplicates)")
    
    if not new_entries:
        print("No new entries to add!")
        return
    
    # Append to company_tickers.csv
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    with TICKERS_FILE.open('a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        for cik, ticker, name in new_entries:
            # Format: "cik","ticker","company_name","exchange","updated_at"
            writer.writerow([
                f'"{cik}"',
                f'"{ticker}"',
                f'"{name}"',
                '',  # exchange (empty)
                f'"{timestamp}"'
            ])
            print(f"  Added: {cik:10} -> {ticker:6} | {name}")
    
    print(f"\n{len(new_entries)} tickers added to {TICKERS_FILE}")
    print("\nNext steps:")
    print("  1. Run: python 6_scoring_model_ticker_adding.py")
    print("  2. This will re-match companies with the new tickers")
    print("=" * 70)

if __name__ == "__main__":
    main()
