#!/usr/bin/env python3
"""
Schedule 13 backtest: T+1 entry, stop-loss (ATR vagy fix %), opcionális TP/trailing.
Bemenet: Excel (file_date, ticker) – min. két oszlop szükséges.

pip install pandas numpy yfinance openpyxl

Kimenetek: backtest_outputs/ alá CSV-k
- schedule13_stop_trades.csv   (trade szintű eredmények)
- schedule13_stop_summary.csv  (PnL összefoglaló)

Megjegyzések:
- priority: ha ugyanazon napon a High >= TP és Low <= Stop is megtörténik, melyik legyen előbb?
  választható: stop_first (konzervatív) / tp_first (optimista).
- Gap kezelése: ha a nyitó <= stop → stop az Open-ön; ha a nyitó >= target → TP az Open-ön.
- ATR alaphoz napi OHLC szükséges (High/Low/Close).
"""

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------------
# Globális beállítások
# -----------------------------------
PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
YF_TIMEOUT = 60  # növelve a timeout hibák csökkentésére

# -----------------------------------
# CLI
# -----------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Schedule 13 backtest: T+1 entry, stop-loss (ATR vagy fix %), opcionális TP/trailing."
    )
    p.add_argument("--excel", type=str, default="schedule13_events.xlsx",
                   help="Bemeneti Excel (oszlopok: file_date, ticker).")
    p.add_argument("--sheet", type=str, default="0",
                   help="Lap neve vagy indexe (alap: 0).")
    p.add_argument("--price-field", type=str, default="Open",
                   choices=["Open","Close","Adj Close"],
                   help="Belépő ár mező (Open/Close/Adj Close).")

    p.add_argument("--atr-period", type=int, default=14,
                   help="ATR periódus (Wilder).")

    p.add_argument("--stop-type", type=str, default="atr",
                   choices=["atr","pct","none"],
                   help=(
                       "Stop típusa:\n"
                       "  atr  = k*ATR távolság (pl. --stop-k 2.0 → 2×ATR)\n"
                       "  pct  = fix százalék (pl. --stop-k 0.08 → 8%%)\n"
                       "  none = nincs stop"
                   ))
    p.add_argument("--stop-k", type=float, default=2.0,
                   help="ATR szorzó (atr esetén), vagy fix százalék (pct esetén), pl. 0.08 = 8%%.")
    p.add_argument("--tp-pct", type=float, default=0,
                   help="Take-profit százalék (0 = kikapcsolva), pl. 0.10 = 10%%.")
    p.add_argument("--trail-k", type=float, default=5,
                   help="Követő stop: highest_close − (trail_k × ATR). 0 = kikapcsolva.")
    p.add_argument("--max-hold-days", type=int, default=0,
                   help="Max tartási idő (kereskedési napokban). 0 = nincs időzárás (csak Stop/TP).")
    p.add_argument("--priority", type=str, default="stop_first",
                   choices=["stop_first","tp_first"],
                   help=(
                       "Ha ugyanazon a napon stop és TP is érintett:\n"
                       "  stop_first = konzervatív (előbb stop)\n"
                       "  tp_first   = optimista (előbb TP)"
                   ))
    p.add_argument("--outdir", type=str, default="backtest_outputs",
                   help="Kimeneti mappa.")
    p.add_argument("--dedupe-days", type=int, default=5,
                   help="Ugyanazon ticker eseményei között minimum nap (rolling filter).")

    # Időzárás kikapcsolása + letöltési ablak külön paraméterekkel
    p.add_argument("--no-time-exit", action="store_true",
                   help="Ha meg van adva, nem zár időre (csak Stop/TP).")
    p.add_argument("--pre-days", type=int, default=30,
                   help="Ennyi nappal a legkorábbi file_date előtt töltsön adatot.")
    p.add_argument("--post-days", type=int, default=180,
                   help="Ennyi nappal a legkésőbbi file_date után töltsön adatot (ha nincs időzárás).")

    # Debug / limit / batching
    p.add_argument("--debug", action="store_true",
                   help="Részletes naplózás (tickerlista, dátumtartomány, batch jegyzetek).")
    p.add_argument("--tickers-limit", type=int, default=0,
                   help="Csak az első N tickert dolgozza fel (debug/gyors teszthez).")
    p.add_argument("--chunk-size", type=int, default=50,
                   help="Yahoo batch letöltés csomagméret (nagy listánál csökkentsd, pl. 25).")

    return p.parse_args()

# -----------------------------------
# Segédfüggvények
# -----------------------------------
def clean_ticker(t: str) -> Optional[str]:
    """ISIN és warrant/rights szűrés. Külföldi suffix maradhat (Yahoo használja)."""
    t = (str(t) if t is not None else "").strip().upper()
    if not t:
        return None
    import re
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", t):  # ISIN-szerű
        return None
    if re.search(r"(-WT|\.WT| W$| WS$|/WS$|/W$)", t):  # warrant/rights
        return None
    return t

def ensure_multiindex(pr: pd.DataFrame, tickers_hint: List[str]) -> pd.DataFrame:
    """
    Garantáljuk a (Field, Ticker) MultiIndex oszlopokat.
    Kezeli a (Ticker, Field) és az egyszintű eseteket is.
    """
    if isinstance(pr.columns, pd.MultiIndex):
        lev0 = list(pr.columns.get_level_values(0))
        lev1 = list(pr.columns.get_level_values(1))
        fields = set(PRICE_FIELDS)
        # ha (Ticker, Field) sorrend: felcseréljük
        if set(lev0) >= set(tickers_hint) and any(f in lev1 for f in fields):
            new_cols = [(f, t) for (t, f) in pr.columns.to_list()]
            pr = pr.copy()
            pr.columns = pd.MultiIndex.from_tuples(new_cols)
        return pr
    else:
        # Egy ticker eset: tegyük vissza MultiIndexre
        pr = pr.copy()
        only = tickers_hint[0] if tickers_hint else "TICKER"
        pr.columns = pd.MultiIndex.from_product([pr.columns, [only]])
        return pr

def fetch_prices(tickers, start, end, interval="1d", timeout=YF_TIMEOUT, chunk_size=50, debug=False):
    """
    Árfolyam letöltés yfinance-ról:
      - chunkokra vágott batch letöltés
      - batch üresség esetén egyenkénti letöltés
    Vissza: (DataFrame, failed_list, notes_str)
    """
    import math
    uniq = sorted({t for t in (clean_ticker(x) for x in tickers) if t})
    if debug:
        print(f"[DEBUG] Letöltendő tickerek (tisztítás után): {len(uniq)}")

    if not uniq:
        return pd.DataFrame(), [], "[fetch] Üres tickerlista a tisztítás után."

    frames = []
    failed = []
    notes = []

    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i:i+chunk_size]
        if debug:
            print(f"[DEBUG] Batch {i//chunk_size+1}/{math.ceil(len(uniq)/chunk_size)}: {len(chunk)} ticker")

        # próbáljuk batch-ben
        try:
            df = yf.download(
                tickers=chunk, start=start, end=end, interval=interval,
                auto_adjust=False, threads=True, progress=False, timeout=timeout
            )
            if df is not None and not df.empty:
                df = ensure_multiindex(df, chunk)
                frames.append(df)
                continue
            else:
                notes.append(f"[batch empty] {chunk[:5]}..")
        except Exception as e:
            notes.append(f"[batch error] {type(e).__name__}: {e}")

        # egyenként
        for t in chunk:
            try:
                s = yf.download(
                    t, start=start, end=end, interval=interval,
                    auto_adjust=False, threads=False, progress=False, timeout=timeout
                )
                if s is None or s.empty:
                    failed.append(t)
                    continue
                s = ensure_multiindex(s, [t])
                frames.append(s)
            except Exception as e:
                failed.append(t)
                notes.append(f"[single {t}] {type(e).__name__}: {e}")

        time.sleep(0.4)  # kímélet

    if frames:
        out = pd.concat(frames, axis=1).sort_index(axis=1)
        return out, failed, "; ".join(notes) if notes else ""
    else:
        return pd.DataFrame(), failed, "; ".join(notes) if notes else "[fetch] Nincs adat sem batch-ben, sem egyenként."

def compute_atr(pr: pd.DataFrame, period: int, tickers_hint: List[str]) -> pd.DataFrame:
    """
    Wilder ATR (napi). Bemenet: árfolyamok MultiIndex oszlopokkal (Field, Ticker).
    Kimenet: DataFrame MultiIndex oszlopokkal ("ATR", Ticker).
    """
    pr = ensure_multiindex(pr, tickers_hint)

    have = set(pr.columns.get_level_values(0))
    required = {"High", "Low", "Close"}
    missing = required - have
    if missing:
        raise RuntimeError(f"Hiányzó mezők az ATR-hez: {missing}")

    tickers = list(pr.columns.get_level_values(1).unique())
    atr_cols = pd.MultiIndex.from_product([["ATR"], tickers])
    atr_df = pd.DataFrame(index=pr.index, columns=atr_cols, dtype=float)

    for t in tickers:
        # kiválasztjuk a ticker sorozatait
        h = pr[("High", t)].astype(float)
        l = pr[("Low",  t)].astype(float)
        c = pr[("Close", t)].astype(float)
        pc = c.shift(1)

        # True Range (három komponens max-a soronként)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

        # Wilder-féle ATR simítás
        vals = tr.to_numpy()
        atr_vals = np.full_like(vals, np.nan, dtype=float)
        valid_idx = np.where(np.isfinite(vals))[0]
        if len(valid_idx) >= period:
            start = valid_idx[0]
            atr_vals[start + period - 1] = np.nanmean(vals[start:start + period])
            for i in range(start + period, len(vals)):
                if np.isfinite(atr_vals[i - 1]) and np.isfinite(vals[i]):
                    atr_vals[i] = (atr_vals[i - 1] * (period - 1) + vals[i]) / period

        atr_df[("ATR", t)] = atr_vals

    return atr_df

def next_trading_day(d: dt.date, trading_days: List[dt.date]) -> Optional[dt.date]:
    import bisect
    i = bisect.bisect_left(trading_days, d)
    if i < len(trading_days) and trading_days[i] == d:
        return d
    if i < len(trading_days):
        return trading_days[i]
    return None

# -----------------------------------
# Stop/TP logika napi baron
# -----------------------------------
def simulate_trade(dates: List[dt.date],
                   open_s: pd.Series, high_s: pd.Series, low_s: pd.Series, close_s: pd.Series,
                   atr_s: pd.Series,
                   entry_day: dt.date, entry_price: float,
                   stop_type: str, stop_k: float,
                   tp_pct: float, trail_k: float,
                   max_hold_days: int, priority: str):
    """
    Visszatér: dict(exit_day, exit_price, exit_reason, bars_held, pnl_pct, max_drawdown_pct, peak_gain_pct)
    """
    def initial_stop():
        if stop_type == "none":
            return -np.inf
        if stop_type == "pct":
            return entry_price * (1 - stop_k)
        if stop_type == "atr":
            atr0 = atr_s.get(entry_day, np.nan)
            if not np.isfinite(atr0):
                return entry_price * (1 - 0.1)  # fallback: 10%
            return entry_price - stop_k * atr0
        return entry_price * (1 - 0.1)

    stop_level = initial_stop()
    tp_level = entry_price * (1 + tp_pct) if tp_pct and tp_pct > 0 else np.inf
    highest_close = close_s.get(entry_day, entry_price)
    peak_gain = 0.0
    max_dd = 0.0

    try:
        idx0 = dates.index(entry_day)
    except ValueError:
        return None

    exit_day = None
    exit_px = None
    exit_reason = None
    bars = 0

    # end_i: ha nincs időzárás (max_hold_days==0), menjen a teljes adatablak végéig
    if max_hold_days and max_hold_days > 0:
        end_i = min(idx0 + max_hold_days, len(dates) - 1)
    else:
        end_i = len(dates) - 1

    for i in range(idx0 + 1, end_i + 1):
        d = dates[i]
        # OHLC értékek lok-kal, hogy hiány esetén KeyError-t kapjunk
        try:
            o = float(open_s.loc[d])
            h = float(high_s.loc[d])
            l = float(low_s.loc[d])
            c = float(close_s.loc[d])
        except KeyError:
            # nincs bar ezen a napon ebben a sorozatban → ugorjuk
            continue

        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
            continue

        bars += 1

        # trailing frissítés (előző záró után)
        if trail_k and trail_k > 0:
            atr_today = float(atr_s.get(d, np.nan))
            if np.isfinite(atr_today):
                highest_close = max(highest_close, c)
                trail_level = highest_close - trail_k * atr_today
                stop_level = max(stop_level, trail_level)

        # Nyitó gap kezelése
        if o <= stop_level:
            exit_day, exit_px, exit_reason = d, o, "stop_gap_open"
            peak_gain = max(peak_gain, (c / entry_price) - 1.0)
            max_dd = min(max_dd, (c / entry_price) - 1.0)
            break
        if o >= tp_level:
            exit_day, exit_px, exit_reason = d, o, "tp_gap_open"
            peak_gain = max(peak_gain, (c / entry_price) - 1.0)
            max_dd = min(max_dd, (c / entry_price) - 1.0)
            break

        # Intraday érintések
        hit_stop = (l <= stop_level)
        hit_tp   = (h >= tp_level)

        if hit_stop and hit_tp:
            if priority == "stop_first":
                exit_day, exit_px, exit_reason = d, stop_level, "stop_intraday"
            else:
                exit_day, exit_px, exit_reason = d, tp_level, "tp_intraday"
            peak_gain = max(peak_gain, (h / entry_price) - 1.0)
            max_dd    = min(max_dd, (l / entry_price) - 1.0)
            break
        elif hit_stop:
            exit_day, exit_px, exit_reason = d, stop_level, "stop_intraday"
            peak_gain = max(peak_gain, (h / entry_price) - 1.0)
            max_dd    = min(max_dd, (l / entry_price) - 1.0)
            break
        elif hit_tp:
            exit_day, exit_px, exit_reason = d, tp_level, "tp_intraday"
            peak_gain = max(peak_gain, (h / entry_price) - 1.0)
            max_dd    = min(max_dd, (l / entry_price) - 1.0)
            break

        # nap végi statok
        peak_gain = max(peak_gain, (c / entry_price) - 1.0)
        max_dd    = min(max_dd, (c / entry_price) - 1.0)

    if exit_day is None:
        # nincs jel → időzárás vagy adatablak vége
        exit_day = dates[end_i]
        exit_px = float(close_s.get(exit_day, np.nan))
        exit_reason = "time_exit" if (max_hold_days and max_hold_days > 0) else "data_end"

    pnl = (exit_px / entry_price) - 1.0 if (np.isfinite(exit_px) and entry_price > 0) else np.nan
    return {
        "entry_day": entry_day,
        "entry_price": entry_price,
        "exit_day": exit_day,
        "exit_price": exit_px,
        "exit_reason": exit_reason,
        "bars_held": bars,
        "pnl_pct": pnl,
        "peak_gain_pct": peak_gain,
        "max_drawdown_pct": max_dd
    }

# -----------------------------------
# Main
# -----------------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Excel beolvasás
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    df = pd.read_excel(args.excel, sheet_name=sheet, engine="openpyxl")
    cols = {c.lower().strip(): c for c in df.columns}
    assert "file_date" in cols and "ticker" in cols, "Excel-nek tartalmaznia kell: file_date, ticker"

    rename_map = {
        cols["file_date"]: "file_date",
        cols["ticker"]: "ticker",
    }
    # opcionális oszlopok:
    if "form_type" in cols:
        rename_map[cols["form_type"]] = "form_type"
    if "cap_bucket" in cols:
        rename_map[cols["cap_bucket"]] = "cap_bucket"

    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["file_date","ticker"]).copy()
    df["file_date"] = pd.to_datetime(df["file_date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # ha hiányoznak, hozzuk létre üresen
    if "form_type" not in df.columns:
        df["form_type"] = np.nan
    if "cap_bucket" not in df.columns:
        df["cap_bucket"] = np.nan

    # deduplikáció ugyanarra a tickerre
    df = df.sort_values(["ticker","file_date"])
    if args.dedupe_days > 0:
        keep_mask = []
        for tkr, g in df.groupby("ticker"):
            last = None
            for d in g["file_date"]:
                if last is None or (d - last).days >= args.dedupe_days:
                    keep_mask.append(True)
                    last = d
                else:
                    keep_mask.append(False)
        df = df.loc[keep_mask].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if df.empty:
        print("Nincs esemény az Excelben.")
        return

    # 2) Időintervallum az árfolyamokhoz – a pre/post paramétereket használjuk
    min_date = df["file_date"].min() - dt.timedelta(days=args.pre_days)
    if args.no_time_exit or args.max_hold_days == 0:
        post_days = args.post_days
    else:
        post_days = args.max_hold_days
    max_date = df["file_date"].max() + dt.timedelta(days=post_days + 5)

    tickers = sorted(df["ticker"].unique().tolist())
    if args.tickers_limit and args.tickers_limit > 0:
        tickers = tickers[:args.tickers_limit]

    if args.debug:
        print(f"[DEBUG] Események száma: {len(df)}")
        print(f"[DEBUG] file_date tartomány: {df['file_date'].min()} .. {df['file_date'].max()}")
        print(f"[DEBUG] Tickerek (egyedi, limit után): {len(set(tickers))}")
        print(f"[DEBUG] Letöltési ablak: {min_date} .. {max_date}")
        print(f"[DEBUG] Minta tickerek: {tickers[:10]}")

    # 3) Árfolyamok letöltése
    px, failed, notes = fetch_prices(
        tickers, start=min_date, end=max_date,
        interval="1d", timeout=YF_TIMEOUT,
        chunk_size=args.chunk_size, debug=args.debug
    )
    if args.debug:
        print(f"[DEBUG] fetch notes: {notes}")
        print(f"[DEBUG] failed tickers: {len(failed)} (példa: {failed[:10]})")

    if px.empty:
        print("Nem sikerült árfolyamot letölteni.")
        if failed:
            print(f"Részletek – sikertelen tickerek száma: {len(failed)}. Példa: {failed[:10]}")
        if notes:
            print(notes)
        return

    # rendezett kereskedési napok
    trade_days = sorted(pd.to_datetime(px.index).date)

    # 4) ATR számítás
    px = ensure_multiindex(px, tickers)
    atr_df = compute_atr(px, args.atr_period, tickers)

    trades = []
    skipped = 0

    # 5) Backtest eseményenként
    for _, row in df.iterrows():
        tkr = clean_ticker(row["ticker"])
        if not tkr:
            continue
        file_d = row["file_date"]

        # belépő nap = file_date + 1 kereskedési nap
        t_plus_1 = next_trading_day(file_d + dt.timedelta(days=1), trade_days)
        if t_plus_1 is None:
            skipped += 1
            continue

        # belépő ár
        pf = args.price_field
        try:
            entry_price = float(px[(pf, tkr)].loc[pd.to_datetime(t_plus_1)])
        except Exception:
            skipped += 1
            continue

        # sorozatok
        try:
            open_s  = px[("Open",  tkr)].copy()
            high_s  = px[("High",  tkr)].copy()
            low_s   = px[("Low",   tkr)].copy()
            close_s = px[("Close", tkr)].copy()
        except KeyError:
            skipped += 1
            continue

        try:
            atr_s = atr_df[("ATR", tkr)].copy()
        except KeyError:
            skipped += 1
            continue

        # indexeket dátummá
        for s in (open_s, high_s, low_s, close_s, atr_s):
            s.index = pd.to_datetime(s.index).date

        sim = simulate_trade(
            dates=trade_days,
            open_s=open_s, high_s=high_s, low_s=low_s, close_s=close_s,
            atr_s=atr_s,
            entry_day=t_plus_1, entry_price=entry_price,
            stop_type=args.stop_type, stop_k=args.stop_k,
            tp_pct=args.tp_pct, trail_k=args.trail_k,
            max_hold_days=0 if args.no_time_exit else args.max_hold_days,
            priority=args.priority
        )
        if sim is None:
            skipped += 1
            continue

        trades.append({
            "ticker": tkr,
            "file_date": file_d,
            "form_type": row.get("form_type", np.nan),
            "cap_bucket": row.get("cap_bucket", np.nan),
            **sim
        })

    trades_df = pd.DataFrame(trades)
    trades_path = outdir / "schedule13_stop_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    # 6) Gyors összefoglaló
    def summ(s: pd.Series):
        s = s.dropna()
        return pd.Series({
            "N": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "stdev": s.std(),
            "hit_%_>0": (s > 0).mean()*100 if len(s) else np.nan,
            "p25": s.quantile(0.25) if len(s) else np.nan,
            "p75": s.quantile(0.75) if len(s) else np.nan
        })

    summary = summ(trades_df["pnl_pct"]) if not trades_df.empty else pd.Series(dtype=float)
    summary_df = pd.DataFrame(summary, columns=["pnl_pct"])
    summary_path = outdir / "schedule13_stop_summary.csv"
    summary_df.to_csv(summary_path)

    print("Kész!")
    print(f"Trade-ok száma: {len(trades_df)} | Kihagyva: {skipped} | Hibás letöltések: {len(failed)}")
    if not trades_df.empty:
        winrate = (trades_df["pnl_pct"] > 0).mean() * 100
        print(f"Átlag PnL: {trades_df['pnl_pct'].mean():.4f} | Medián: {trades_df['pnl_pct'].median():.4f} | Win%: {winrate:.1f}%")
        print(f"Eredmények: {trades_path}")
        print(f"Összefoglaló: {summary_path}")
    else:
        print("Nem született érvényes trade – próbáld nagyobb post window-val (pl. --post-days 240), vagy ellenőrizd a TP/stop beállításokat.")

if __name__ == "__main__":
    main()
