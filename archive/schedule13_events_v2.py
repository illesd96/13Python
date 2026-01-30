import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import re
import time

# pip install yfinance openpyxl
import yfinance as yf

# ==== 0) KIMENETI MAPPÁT HOZZUK LÉTRE MÁR A LEGELEJÉN ====
OUT_DIR = Path("backtest_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ==== 1) PARAMÉTEREK ====
EXCEL_PATH = "schedule13_events.xlsx"   # <-- ide tedd a saját fájlod nevét
SHEET_NAME = 0                          # vagy név szerint pl. "Sheet1"
DATE_COL = "file_date"
TICKER_COL = "ticker"
ISSUER_COL = "issuer"                   # opcionális
FORM_COL = "form_type"                  # opcionális

START_PAD_DAYS = 30
WINDOWS = [1, 5, 20]                    # +1d, +5d, +20d (belépő->kilépő)
DEDUPE_DAYS = 5

YF_TIMEOUT = 30
PRICE_FIELDS = ["Open", "Adj Close", "Close"]  # fallback sorrend

# Piaci kapitalizáció
CAP_FETCH = True                        # ha lassúnak érzed, állítsd False-ra
CAP_MICRO_MAX = 300_000_000            # < $300M = micro

# ==== 2) SEGÉDFÜGGVÉNYEK ====
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower().strip(): c for c in df.columns}
    assert DATE_COL in lower_map, f"Hiányzik oszlop: {DATE_COL}"
    assert TICKER_COL in lower_map, f"Hiányzik oszlop: {TICKER_COL}"

    rename_dict = {
        lower_map[DATE_COL]: "file_date",
        lower_map[TICKER_COL]: "ticker"
    }
    if ISSUER_COL in lower_map:
        rename_dict[lower_map[ISSUER_COL]] = "issuer"
    if FORM_COL in lower_map:
        rename_dict[lower_map[FORM_COL]] = "form_type"

    df = df.rename(columns=rename_dict)
    if "issuer" not in df.columns:
        df["issuer"] = np.nan
    if "form_type" not in df.columns:
        df["form_type"] = np.nan
    return df

def clean_ticker(t: str) -> str | None:
    """ISIN és warrant/rights kiszűrése. Külföldi suffix maradhat (Yahoo használja)."""
    t = (str(t) if t is not None else "").strip().upper()
    if not t:
        return None
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", t):  # ISIN-szerű
        return None
    if re.search(r"(-WT|\.WT| W$| WS$|/WS$|/W$)", t):  # warrant/rights
        return None
    return t

def clean_tickers(tickers):
    uniq, seen = [], set()
    for t in tickers:
        ct = clean_ticker(t)
        if ct and ct not in seen:
            seen.add(ct)
            uniq.append(ct)
    return uniq

def ensure_multiindex(df: pd.DataFrame, tickers_hint: list[str]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lev0 = list(df.columns.get_level_values(0))
        lev1 = list(df.columns.get_level_values(1))
        # ha (Ticker, Field), cseréljük fel (heurisztika)
        if set(lev0) >= set(tickers_hint) and any(f in lev1 for f in PRICE_FIELDS):
            new_cols = [(f, t) for (t, f) in df.columns.to_list()]
            df.columns = pd.MultiIndex.from_tuples(new_cols)
        return df
    else:
        cols = pd.MultiIndex.from_product([df.columns, tickers_hint])
        df.columns = cols
        return df

def fetch_prices(tickers, start, end, interval="1d", max_retries=2, sleep_sec=1):
    tickers = clean_tickers(tickers)
    if not tickers:
        return pd.DataFrame(), []

    fails = []

    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                threads=True,
                progress=False,
                timeout=YF_TIMEOUT
            )
            if df is not None and not df.empty:
                df = ensure_multiindex(df, tickers)
                return df, fails
        except Exception:
            time.sleep(sleep_sec)

    frames = []
    for t in tickers:
        try:
            s = yf.download(
                t, start=start, end=end, interval=interval,
                auto_adjust=False, threads=False, progress=False, timeout=YF_TIMEOUT
            )
            if s is None or s.empty:
                fails.append(t)
                continue
            s = ensure_multiindex(s, [t])
            frames.append(s)
        except Exception:
            fails.append(t)

    if frames:
        df = pd.concat(frames, axis=1).sort_index(axis=1)
        return df, fails
    return pd.DataFrame(), fails

def pick_first_available_field(df: pd.DataFrame, fields: list[str]) -> str | None:
    if not isinstance(df.columns, pd.MultiIndex):
        return None
    level0 = list(df.columns.get_level_values(0))
    for f in fields:
        if f in level0:
            return f
    return None

def ticker_has_any_prices(df: pd.DataFrame, tkr: str, fields: list[str]) -> tuple[bool, str | None]:
    if not isinstance(df.columns, pd.MultiIndex):
        return False, None
    for f in fields:
        col = (f, tkr)
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.Series) and s.notna().any():
                return True, f
    return False, None

def next_trading_day(d, available_dates: set):
    cur = d
    for _ in range(7):
        if cur in available_dates:
            return cur
        cur = (pd.to_datetime(cur) + pd.Timedelta(days=1)).date()
    return None

# ---- Market cap (fast_info + limited .info fallback) ----
def cap_bucket_from_mc(mc: float | None, micro_max=CAP_MICRO_MAX) -> str:
    if mc is None or (isinstance(mc, float) and np.isnan(mc)):
        return "unknown"
    try:
        return "micro" if float(mc) < micro_max else "mid_large"
    except Exception:
        return "unknown"

def fetch_market_caps_fast(tickers: list[str], fallback_info: bool = True, info_limit: int = 400) -> pd.DataFrame:
    """
    Fast market cap fetch:
      1) Batch via Tickers(...).fast_info
      2) Optional fallback to .info for only the tickers still missing (slow), capped by info_limit
    Returns: DataFrame[ticker, market_cap]
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "market_cap"])

    uq = sorted({str(t).upper() for t in tickers if t and str(t).upper() != "SPY"})
    chunks = [uq[i:i+200] for i in range(0, len(uq), 200)]
    rows = []

    # Pass 1: fast_info (batch)
    for chunk in chunks:
        tickstr = " ".join(chunk)
        try:
            bundle = yf.Tickers(tickstr)
            for t in chunk:
                mc = np.nan
                try:
                    fi = bundle.tickers[t].fast_info
                    mc = fi.get("market_cap", np.nan)
                except Exception:
                    mc = np.nan
                if mc in (None, 0):
                    mc = np.nan
                rows.append({"ticker": t, "market_cap": mc})
        except Exception:
            # fallback: per-ticker fast_info
            for t in chunk:
                try:
                    fi = yf.Ticker(t).fast_info
                    mc = fi.get("market_cap", np.nan)
                except Exception:
                    mc = np.nan
                if mc in (None, 0):
                    mc = np.nan
                rows.append({"ticker": t, "market_cap": mc})
        time.sleep(0.1)  # be polite

    df_fast = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "market_cap"])
    if not fallback_info or df_fast.empty:
        return df_fast

    # Pass 2: .info fallback only for missing
    need = df_fast[df_fast["market_cap"].isna()]["ticker"].tolist()
    if not need:
        return df_fast

    need = need[:max(0, int(info_limit))]  # cap the slow fallback
    slow_rows = []
    for t in need:
        mc = np.nan
        try:
            inf = yf.Ticker(t).info  # slow
            mc = inf.get("marketCap", np.nan)
        except Exception:
            mc = np.nan
        if mc in (None, 0):
            mc = np.nan
        slow_rows.append({"ticker": t, "market_cap": mc})
        time.sleep(0.05)

    df_slow = pd.DataFrame(slow_rows)
    if not df_slow.empty:
        merged = df_fast.merge(df_slow, on="ticker", how="left", suffixes=("_fast", "_slow"))
        merged["market_cap"] = merged["market_cap_fast"].where(~merged["market_cap_slow"].notna(),
                                                               merged["market_cap_slow"])
        return merged[["ticker", "market_cap"]]

    return df_fast

def attach_caps_to_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Csak azokhoz kérünk cap-et, amik ténylegesen bekerültek a 'res'-be. fast_info + limited .info fallback."""
    used_tickers = sorted(set(res_df["ticker"].dropna().astype(str).str.upper()))
    caps_df = fetch_market_caps_fast(used_tickers, fallback_info=True, info_limit=400)
    caps_map = {str(k).upper(): v for k, v in zip(caps_df.get("ticker", []), caps_df.get("market_cap", []))}
    res_df = res_df.copy()
    res_df["market_cap"] = res_df["ticker"].map(lambda t: caps_map.get(str(t).upper(), np.nan))
    res_df["cap_bucket"] = res_df["market_cap"].map(lambda mc: cap_bucket_from_mc(mc))
    resolved = res_df["market_cap"].notna().sum()
    total = res_df["ticker"].nunique()
    print(f"[cap] resolved caps for {resolved} events; unique_tickers={total}")
    return res_df

# ==== 3) ESEMÉNYEK BEOLVASÁSA ====
df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df = normalize_columns(df_raw)

df = df.dropna(subset=["ticker", "file_date"])
df["file_date"] = pd.to_datetime(df["file_date"]).dt.date
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

# Időszak a letöltéshez
min_date = df["file_date"].min() - dt.timedelta(days=START_PAD_DAYS)
max_date = df["file_date"].max() + dt.timedelta(days=max(WINDOWS) + 5)

# ==== 4) ÁRFOLYAM ADATOK LETÖLTÉSE (SPY mint piaci bázis) ====
tickers = sorted(df["ticker"].unique().tolist())
all_tickers = tickers + ["SPY"]

px, failed_batch = fetch_prices(all_tickers, start=min_date, end=max_date, interval="1d")
if px.empty:
    pd.DataFrame().to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
    pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / "summary_raw_returns.csv")
    pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / "summary_abnormal_returns.csv")
    if failed_batch:
        pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    raise RuntimeError("Árfolyamadatok üresek.")

use_field = pick_first_available_field(px, PRICE_FIELDS)
if use_field is None:
    pd.DataFrame().to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
    for name in ["summary_raw_returns.csv", "summary_abnormal_returns.csv"]:
        pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / name)
    if failed_batch:
        pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    raise SystemExit("Nem találtam Open/Adj Close/Close mezőt a letöltött adatokban.")

prices = px[use_field].copy()
# ha szeletelés miatt egyszintű lett, tegyük vissza (mező, ticker) MultiIndexre
if not isinstance(prices.columns, pd.MultiIndex):
	    prices.columns = pd.MultiIndex.from_product([[use_field], prices.columns])
price_wide = prices.copy()
price_wide.index = pd.to_datetime(price_wide.index).date
price_index = set(price_wide.index)
sorted_dates = sorted(price_index)

# ==== 5) DEDUPLIKÁCIÓ (ugyanazon ticker sűrű eseményei) ====
df = df.sort_values(["ticker", "file_date"])
df["keep"] = True
for tkr, g in df.groupby("ticker", sort=False):
    last_kept = None
    keep_mask = []
    for d in g["file_date"]:
        if last_kept is None or (d - last_kept).days >= DEDUPE_DAYS:
            keep_mask.append(True)
            last_kept = d
        else:
            keep_mask.append(False)
    df.loc[g.index, "keep"] = keep_mask
events = df[df["keep"]].drop(columns=["keep"]).reset_index(drop=True)

# ==== 6) HOZAMOK + 20D PEAK GAIN / MAX DROP ====
def summarize(col, dfres):
    if col not in dfres.columns:
        return pd.Series({
            "N": 0, "mean": np.nan, "median": np.nan, "stdev": np.nan,
            "hit_%_>0": np.nan, "p25": np.nan, "p75": np.nan
        })
    s = dfres[col].dropna()
    return pd.Series({
        "N": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "stdev": s.std(),
        "hit_%_>0": (s > 0).mean()*100 if len(s) else np.nan,
        "p25": s.quantile(0.25) if len(s) else np.nan,
        "p75": s.quantile(0.75) if len(s) else np.nan
    })

records = []
skipped_no_price = 0
skipped_no_entry = 0

for _, row in events.iterrows():
    tkr = clean_ticker(row["ticker"])
    if not tkr:
        skipped_no_price += 1
        continue

    fdate = row["file_date"]
    # entry nap
    entry_day = next_trading_day((pd.to_datetime(fdate) + pd.Timedelta(days=1)).date(), price_index)
    if entry_day is None:
        skipped_no_entry += 1
        continue

    has_any, field_for_ticker = ticker_has_any_prices(price_wide, tkr, PRICE_FIELDS)
    if not has_any:
        skipped_no_price += 1
        continue
    f_field = field_for_ticker or use_field

    try:
        entry_px = float(price_wide.loc[entry_day, (f_field, tkr)])
        entry_spy = float(price_wide.loc[entry_day, (use_field, "SPY")])
    except KeyError:
        skipped_no_price += 1
        continue
    if not np.isfinite(entry_px) or entry_px <= 0:
        skipped_no_price += 1
        continue

    out = {
        "file_date": fdate,
        "entry_day": entry_day,
        "ticker": tkr,
        "issuer": row.get("issuer", np.nan),
        "form_type": row.get("form_type", np.nan),
        "price_field": f_field,
        "entry_price": entry_px,
        "entry_price_spy": entry_spy,
    }

    # kimenő napok és hozamok
    for w in WINDOWS:
        exit_day = None
        try:
            idx = sorted_dates.index(entry_day)
            if idx + w < len(sorted_dates):
                exit_day = sorted_dates[idx + w]
        except ValueError:
            exit_day = None

        if exit_day is None:
            out[f"ret_{w}d"] = np.nan
            out[f"aret_{w}d"] = np.nan
            continue
        try:
            exit_px = float(price_wide.loc[exit_day, (f_field, tkr)])
            exit_spy = float(price_wide.loc[exit_day, (use_field, "SPY")])
        except KeyError:
            out[f"ret_{w}d"] = np.nan
            out[f"aret_{w}d"] = np.nan
            continue

        ret = (exit_px / entry_px) - 1.0
        mkt = (exit_spy / entry_spy) - 1.0
        out[f"ret_{w}d"] = ret
        out[f"aret_{w}d"] = ret - mkt

    # 20 napos peak gain / max drop (belépőhöz képest)
    try:
        idx0 = sorted_dates.index(entry_day)
        idx_end = min(idx0 + 20, len(sorted_dates) - 1)
        lookahead_days = sorted_dates[idx0+1: idx_end+1]  # T+1 .. T+20
        if lookahead_days:
            series = price_wide.loc[lookahead_days, (f_field, tkr)].astype(float)
            rel = series / entry_px - 1.0
            out["peak_gain_20d"] = np.nanmax(rel.values)
            out["max_drop_20d"] = np.nanmin(rel.values)
        else:
            out["peak_gain_20d"] = np.nan
            out["max_drop_20d"] = np.nan
    except Exception:
        out["peak_gain_20d"] = np.nan
        out["max_drop_20d"] = np.nan

    records.append(out)

res = pd.DataFrame(records)

# ---- Piaci kapitalizáció hozzárendelés (csak a bekerült tickerekre) ----
if CAP_FETCH:
    print(f"[cap] Ténylegesen használt tickerek: {res['ticker'].nunique()}")
    res = attach_caps_to_results(res)
else:
    res["market_cap"] = np.nan
    res["cap_bucket"] = res["market_cap"].map(lambda mc: cap_bucket_from_mc(mc))

# ==== 7) ÖSSZEFOGLALÓK ====
if res.empty:
    pd.DataFrame().to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
    empty_summary = pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"])
    empty_summary.to_csv(OUT_DIR / "summary_raw_returns.csv")
    empty_summary.to_csv(OUT_DIR / "summary_abnormal_returns.csv")
    if 'failed_batch' in locals() and failed_batch:
        pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    print("Kész (de nem volt egyetlen érvényes esemény sem).")
    raise SystemExit(0)

def summarize_table(dfres: pd.DataFrame, windows=WINDOWS, abnormal=False):
    cols = [f"aret_{w}d" for w in windows] if abnormal else [f"ret_{w}d" for w in windows]
    tables = {c: summarize(c, dfres) for c in cols}
    return pd.concat(tables, axis=1)

def grouped_summary(dfres, by_cols, abnormal=False):
    cols = [f"aret_{w}d" for w in WINDOWS] if abnormal else [f"ret_{w}d" for w in WINDOWS]
    out_frames = []
    for name, sub in dfres.groupby(by_cols, dropna=False):
        tbl = {c: summarize(c, sub) for c in cols}
        gsum = pd.concat(tbl, axis=1)
        if not isinstance(name, tuple):
            name = (name,)
        idx = pd.MultiIndex.from_product([['|'.join(map(lambda x: str(x) if x==x else "NaN", name))], gsum.index])
        gsum.index = idx
        out_frames.append(gsum)
    if out_frames:
        return pd.concat(out_frames, axis=0)
    return pd.DataFrame()

def summarize_cols(dfres: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    tables = {c: summarize(c, dfres) for c in cols}
    return pd.concat(tables, axis=1)

def grouped_summary_custom(dfres: pd.DataFrame, by_cols: list[str], cols: list[str]) -> pd.DataFrame:
    out_frames = []
    for name, sub in dfres.groupby(by_cols, dropna=False):
        tbl = {c: summarize(c, sub) for c in cols}
        gsum = pd.concat(tbl, axis=1)
        if not isinstance(name, tuple):
            name = (name,)
        idx = pd.MultiIndex.from_product([['|'.join(map(lambda x: str(x) if x==x else "NaN", name))], gsum.index])
        gsum.index = idx
        out_frames.append(gsum)
    if out_frames:
        return pd.concat(out_frames, axis=0)
    return pd.DataFrame()

# alap összefoglalók
summary = summarize_table(res, abnormal=False)
ab_summary = summarize_table(res, abnormal=True)

# form-type szerinti
summary_by_form_raw = grouped_summary(res, by_cols=["form_type"], abnormal=False)
summary_by_form_ab  = grouped_summary(res, by_cols=["form_type"], abnormal=True)

# cap_bucket × form_type
summary_by_cap_form_raw = grouped_summary(res, by_cols=["cap_bucket","form_type"], abnormal=False)
summary_by_cap_form_ab  = grouped_summary(res, by_cols=["cap_bucket","form_type"], abnormal=True)

# --- 20 napos peak gain / max drop összefoglalók ---
PEAK_COLS = ["peak_gain_20d", "max_drop_20d"]
summary_peaks = summarize_cols(res, PEAK_COLS)
summary_peaks_by_form = grouped_summary_custom(res, by_cols=["form_type"], cols=PEAK_COLS)
summary_peaks_by_cap_form = grouped_summary_custom(res, by_cols=["cap_bucket", "form_type"], cols=PEAK_COLS)

# ==== 8) MENTÉS ====
res.to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
summary.to_csv(OUT_DIR / "summary_raw_returns.csv")
ab_summary.to_csv(OUT_DIR / "summary_abnormal_returns.csv")
summary_by_form_raw.to_csv(OUT_DIR / "summary_by_form_raw.csv")
summary_by_form_ab.to_csv(OUT_DIR / "summary_by_form_ab.csv")
summary_by_cap_form_raw.to_csv(OUT_DIR / "summary_by_cap_form_raw.csv")
summary_by_cap_form_ab.to_csv(OUT_DIR / "summary_by_cap_form_ab.csv")
summary_peaks.to_csv(OUT_DIR / "summary_peaks.csv")
summary_peaks_by_form.to_csv(OUT_DIR / "summary_peaks_by_form.csv")
summary_peaks_by_cap_form.to_csv(OUT_DIR / "summary_peaks_by_cap_form.csv")

if 'failed_batch' in locals() and failed_batch:
    pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)

print("Kész!")
print(f"Események száma: {len(res)}")
print("Példasor a kimenetből:")
print(res.head(3))
