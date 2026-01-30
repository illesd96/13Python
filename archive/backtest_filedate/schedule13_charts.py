#!/usr/bin/env python3
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Schedule13: file_date utáni árfolyamok vizualizálása")
    p.add_argument("--excel", type=str, default="schedule13_events.xlsx",
                   help="Bemeneti Excel (oszlopok: file_date, ticker)")
    p.add_argument("--sheet", type=str, default="0",
                   help="Lap neve vagy indexe (alap: 0)")
    p.add_argument("--days", type=int, default=30,
                   help="file_date utáni kereskedési napok száma (alap: 30)")
    p.add_argument("--offset-days", type=int, default=1,
                   help="file_date + offset_naptól kezdjük (alap: 1 = T+1)")
    p.add_argument("--price-field", type=str, default="Adj Close",
                   choices=["Open","Close","Adj Close"],
                   help="Melyik mezőt rajzoljuk (alap: Adj Close)")
    p.add_argument("--outdir", type=str, default="post30_charts",
                   help="Kimeneti mappa (PNG & CSV)")
    p.add_argument("--max-tickers", type=int, default=0,
                   help="Csak az első N tickert dolgozza fel (0 = mind)")
    return p.parse_args()

# ------------------------------
# Segédek
# ------------------------------
def clean_ticker(t: str) -> str | None:
    t = (str(t) if t is not None else "").strip().upper()
    if not t:
        return None
    import re
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", t):  # ISIN-szerű
        return None
    if re.search(r"(-WT|\.WT| W$| WS$|/WS$|/W$)", t):  # warrants/rights
        return None
    return t

def ensure_multiindex(pr: pd.DataFrame, tickers_hint: list[str]) -> pd.DataFrame:
    if isinstance(pr.columns, pd.MultiIndex):
        # lehet (Field, Ticker) vagy (Ticker, Field) — itt nem bolygatjuk,
        # a későbbiekben mindkettőt kezeljük.
        return pr
    else:
        # egyszintű -> MultiIndex
        only = tickers_hint[0] if tickers_hint else "TICKER"
        pr.columns = pd.MultiIndex.from_product([pr.columns, [only]])
        return pr

def slice_field(pr: pd.DataFrame, field: str, ticker: str) -> pd.Series | None:
    """Visszaadja a kiválasztott field sorozatát tickerre, akár (Field, Ticker), akár (Ticker, Field) az oszloprend."""
    if not isinstance(pr.columns, pd.MultiIndex):
        # egyszintű eset (egy ticker)
        if field in pr.columns:
            return pr[field]
        return None
    cols = pr.columns
    if (field, ticker) in cols:
        return pr[(field, ticker)]
    if (ticker, field) in cols:
        return pr[(ticker, field)]
    return None

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "individual").mkdir(parents=True, exist_ok=True)
    (outdir / "combined").mkdir(parents=True, exist_ok=True)
    (outdir / "data").mkdir(parents=True, exist_ok=True)

    # 1) Excel beolvasás
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet, engine="openpyxl")

    cols = {c.lower().strip(): c for c in df.columns}
    assert "file_date" in cols and "ticker" in cols, "Excelben kell: file_date, ticker"

    df = df.rename(columns={cols["file_date"]: "file_date", cols["ticker"]: "ticker"})
    df = df.dropna(subset=["file_date","ticker"]).copy()
    df["file_date"] = pd.to_datetime(df["file_date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # opcionális limit
    if args.max_tickers and args.max_tickers > 0:
        df = df.groupby("ticker").head(1)  # 1 esemény/ticker
        df = df.head(args.max_tickers)

    # 2) Letöltési ablak: minden file_date köré kell adat
    min_file = df["file_date"].min()
    max_file = df["file_date"].max()
    start = min_file - dt.timedelta(days=10)  # kis puffer
    end   = max_file + dt.timedelta(days=args.days + 10)

    # 3) Ticker lista és letöltés
    tickers = sorted({t for t in (clean_ticker(x) for x in df["ticker"]) if t})
    if not tickers:
        print("Nincs érvényes ticker.")
        return

    px = yf.download(
        tickers=tickers,
        start=start, end=end, interval="1d",
        auto_adjust=False, progress=False, threads=True
    )
    if px is None or px.empty:
        print("Nem sikerült árfolyamot letölteni.")
        return
    px = ensure_multiindex(px, tickers)

    # 4) Egyedi chartok + panel összeállítás
    # A panel egy "relatív nap" indexen lesz: 0..args.days (T+offset .. T+offset+days)
    rel_panel = {}  # ticker -> Series (len = args.days+1), normalizált 100-ra
    used = []

    for _, row in df.iterrows():
        tkr = clean_ticker(row["ticker"])
        if not tkr:
            continue
        fdate = row["file_date"]
        # belépő nap: file_date + offset (ált. T+1)
        entry_day = fdate + dt.timedelta(days=args.offset_days)

        s = slice_field(px, args.price_field, tkr)
        if s is None:
            continue

        # Indexet dátummá
        s = s.copy()
        s.index = pd.to_datetime(s.index).date

        # szűrés: entry_day..entry_day+days
        # kereskedési napok szerint (vannak kimaradó napok) – vegyük az első (args.days+1) értéket
        win = s.loc[entry_day: entry_day + dt.timedelta(days=args.days)]
        # ha túl kevés adat, ugorjuk
        if win.dropna().shape[0] < 2:
            continue

        # normalizálás a kezdő értékre
        try:
            start_val = float(win.dropna().iloc[0])
        except Exception:
            continue
        if not np.isfinite(start_val) or start_val <= 0:
            continue

        norm = (win / start_val) * 100.0  # 100-ról indul
        rel = norm.reset_index(drop=True)  # relatív napok 0..N
        rel_panel[tkr] = rel
        used.append((tkr, fdate))

        # Egyedi chart
        plt.figure(figsize=(8, 4.5))
        plt.plot(win.index, win.values)
        plt.title(f"{tkr} – {args.price_field} from {entry_day} (+{args.days}d)")
        plt.xlabel("Date")
        plt.ylabel(args.price_field)
        plt.grid(True, alpha=0.3)
        fname = f"{tkr}_{fdate}_field-{args.price_field.replace(' ','_')}_d{args.days}.png"
        plt.tight_layout()
        plt.savefig(outdir / "individual" / fname, dpi=120)
        plt.close()

    if not rel_panel:
        print("Nem készült egyetlen sorozat sem (ellenőrizd a tickereket / dátumot).")
        return

    # 5) Panel mentése CSV-be (relatív nap index, oszlopok: tickerek, érték: normalizált 100)
    panel_df = pd.DataFrame(rel_panel)
    panel_df.index.name = "rel_day"
    panel_path = outdir / "data" / f"normalized_panel_{args.price_field.replace(' ','_')}_d{args.days}.csv"
    panel_df.to_csv(panel_path)

    # 6) Összesített chart (normalizált 100-ról induló görbék)
    plt.figure(figsize=(10, 6))
    for tkr, ser in rel_panel.items():
        plt.plot(ser.index, ser.values, linewidth=1.0)
    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"Normalized {args.price_field} (100 at entry), next {args.days} trading days")
    plt.xlabel("Relative day (0 = entry)")
    plt.ylabel("Index (100 = entry)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    comb_name = f"combined_norm_{args.price_field.replace(' ','_')}_d{args.days}.png"
    plt.savefig(outdir / "combined" / comb_name, dpi=140)
    plt.close()

    print("Kész.")
    print(f"- Egyedi chartok: {outdir / 'individual'}")
    print(f"- Összesített chart: {outdir / 'combined' / comb_name}")
    print(f"- Normalizált panel CSV: {panel_path}")
    if args.max_tickers:
        print(f"(Csak az első {args.max_tickers} ticker került feldolgozásra.)")

if __name__ == "__main__":
    main()
