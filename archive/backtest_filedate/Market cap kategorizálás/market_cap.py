#!/usr/bin/env python3
"""
Market cap besorolás (micro vs mid_large) Excelből beolvasott tickerlistára.

Bemenet (Excel): legalább ezek az oszlopok:
- form_type
- file_date
- issuer_name
- ticker

Kimenet: két fájl az outdir-ben:
- enriched_with_cap.csv
- enriched_with_cap.xlsx

Használat:
python3 classify_market_cap.py --excel schedule13_events.xlsx --sheet 0 --outdir cap_outputs

Előfeltétel:
pip install pandas yfinance openpyxl
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


CAP_MICRO_MAX = 300_000_000  # < $300M => micro
CHUNK_SIZE = 200             # yfinance.Tickers batch méret
SLEEP_BETWEEN_CHUNKS = 0.10  # kímélet a Yahoo felé


def parse_args():
    p = argparse.ArgumentParser(description="Market cap kategorizálás (micro vs mid_large)")
    p.add_argument("--excel", type=str, default="schedule13_big.xlsx",
                   help="Bemeneti Excel fájl (form_type, file_date, issuer_name, ticker)")
    p.add_argument("--sheet", type=str, default="0",
                   help="Lap neve vagy indexe (alap: 0)")
    p.add_argument("--outdir", type=str, default="cap_outputs",
                   help="Kimeneti mappa")
    p.add_argument("--tickers-limit", type=int, default=0,
                   help="Csak az első N egyedi tickeren fut (debug/gyors teszt)")
    return p.parse_args()


def clean_ticker(t: str) -> Optional[str]:
    """ISIN és warrant/rights szűrés. Külföldi suffix (pl. .DE, .L) maradhat."""
    t = (str(t) if t is not None else "").strip().upper()
    if not t:
        return None
    import re
    # ISIN-szerű: 2 betű + 10 alfanumerikus (utolsó számjegy)
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", t):
        return None
    # warrants/rights gyakran nem elérhető YF-en
    if re.search(r"(-WT|\.WT| W$| WS$|/WS$|/W$)", t):
        return None
    return t


def cap_bucket_from_mc(mc) -> str:
    if mc is None or (isinstance(mc, float) and np.isnan(mc)):
        return "unknown"
    try:
        return "micro" if float(mc) < CAP_MICRO_MAX else "mid_large"
    except Exception:
        return "unknown"


def fetch_market_caps_fast(tickers: List[str]) -> pd.DataFrame:
    """
    Market cap lekérés gyorsan:
    - Elsődlegesen fast_info.market_cap (batch: yf.Tickers)
    - Fallback: .info.get("marketCap") egyenként, ha fast_info hiányzik
    Vissza: DataFrame [ticker, market_cap]
    """
    tickers = sorted({t for t in (clean_ticker(x) for x in tickers) if t})
    rows = []
    if not tickers:
        return pd.DataFrame(columns=["ticker", "market_cap"])

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i+CHUNK_SIZE]
        joined = " ".join(chunk)
        try:
            bundle = yf.Tickers(joined)
            for t in chunk:
                mc = np.nan
                try:
                    fi = bundle.tickers[t].fast_info
                    mc = fi.get("market_cap", np.nan)
                    # ha fast_info hiányos, fallback .info
                    if mc is None or (isinstance(mc, float) and np.isnan(mc)):
                        try:
                            info = bundle.tickers[t].info or {}
                            mc = info.get("marketCap", np.nan)
                        except Exception:
                            mc = np.nan
                except Exception:
                    # teljes fallback: egyenként
                    try:
                        tk = yf.Ticker(t)
                        mc = tk.fast_info.get("market_cap", np.nan)
                        if mc is None or (isinstance(mc, float) and np.isnan(mc)):
                            info = tk.info or {}
                            mc = info.get("marketCap", np.nan)
                    except Exception:
                        mc = np.nan
                rows.append({"ticker": t, "market_cap": mc})
        except Exception:
            # teljes chunk fallback egyenként
            for t in chunk:
                mc = np.nan
                try:
                    tk = yf.Ticker(t)
                    mc = tk.fast_info.get("market_cap", np.nan)
                    if mc is None or (isinstance(mc, float) and np.isnan(mc)):
                        info = tk.info or {}
                        mc = info.get("marketCap", np.nan)
                except Exception:
                    mc = np.nan
                rows.append({"ticker": t, "market_cap": mc})
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Excel beolvasás
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet, engine="openpyxl")

    cols = {c.lower().strip(): c for c in df.columns}
    required = ["form_type", "file_date", "issuer_name", "ticker"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Hiányzó oszlop(ok) az Excelben: {missing}. Kell: {required}")

    df = df.rename(columns={
        cols["form_type"]: "form_type",
        cols["file_date"]: "file_date",
        cols["issuer_name"]: "issuer_name",
        cols["ticker"]: "ticker",
    })

    # alaptisztítás
    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    # file_date -> dátum
    try:
        df["file_date"] = pd.to_datetime(df["file_date"]).dt.date
    except Exception:
        pass  # ha szöveges, maradhat — a cap lekéréshez nem szükséges

    # 2) Egyedi tickerek (opcionális limit)
    uniq_tickers = sorted({t for t in (clean_ticker(x) for x in df["ticker"]) if t})
    if args.tickers_limit and args.tickers_limit > 0:
        uniq_tickers = uniq_tickers[:args.tickers_limit]

    if not uniq_tickers:
        print("Nincs érvényes ticker a fájlban a tisztítás után.")
        return

    # 3) Market cap lekérés
    caps_df = fetch_market_caps_fast(uniq_tickers)
    if caps_df.empty:
        print("Nem sikerült market cap-et lekérni (üres válasz).")
        # akkor is írjunk enriched fájlt unknown-nal
        df["market_cap"] = np.nan
        df["cap_bucket"] = "unknown"
    else:
        # 4) Visszamappelés az eredeti DF-re
        caps_map = {row["ticker"]: row["market_cap"] for _, row in caps_df.iterrows()}
        df["market_cap"] = df["ticker"].map(lambda t: caps_map.get(clean_ticker(t), np.nan))
        df["cap_bucket"] = df["market_cap"].map(cap_bucket_from_mc)

    # 5) Mentés
    csv_path = outdir / "enriched_with_cap.csv"
    xlsx_path = outdir / "enriched_with_cap.xlsx"
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        # ha nincs openpyxl, legalább CSV meglegyen
        pass

    # Extra: listázzuk a sikertelen tickereket
    failed = [t for t in uniq_tickers if t not in set(caps_df["ticker"])] if not caps_df.empty else uniq_tickers
    if failed:
        pd.Series(failed, name="failed_ticker").to_csv(outdir / "failed_tickers.csv", index=False)

    print("Kész.")
    print(f"- Bemenet:  {args.excel} [{sheet}]")
    print(f"- Egyedi tickerek lekérve: {len(uniq_tickers)}")
    print(f"- Kimenetek: {csv_path}  és  {xlsx_path}")
    if failed:
        print(f"- Nem sikerült cap-et lekérni {len(failed)} tickerhez (failed_tickers.csv).")


if __name__ == "__main__":
    main()
