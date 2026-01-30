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
WINDOWS = [1, 5, 20]                    # +1d, +5d, +20d (nyitó->nyitó)
DEDUPE_DAYS = 5

YF_TIMEOUT = 20

# 🔁 Ha nincs Open, próbáljuk Adj Close-t, majd Close-t
PRICE_FIELDS = ["Open", "Adj Close", "Close"]

# ==== 2) SEGÉDFÜGGVÉNYEK ====
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """A bemenet oszlopait kisbetűs kulcsokra mapeljük és átnevezzük a kód elvárásai szerint."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    # kötelezők
    assert DATE_COL in lower_map, f"Hiányzik oszlop: {DATE_COL}"
    assert TICKER_COL in lower_map, f"Hiányzik oszlop: {TICKER_COL}"

    rename_dict = {
        lower_map[DATE_COL]: "file_date",
        lower_map[TICKER_COL]: "ticker"
    }
    # opcionálisak
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
    # ISIN-szerű? (pl. USxxxxxxxxxx) -> dobd
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", t):
        return None
    # Warrants / rights – gyakran nincs Yahoo adat
    if re.search(r"(-WT|\.WT| W$| WS$|/WS$|/W$)", t):
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
    """
    A yfinance különböző alakban adhat vissza:
     - több ticker: MultiIndex (Field, Ticker) vagy (Ticker, Field)
     - egy ticker: egyszintű oszlopok [Open, High, ...]
    Ezt egységesítjük (Field, Ticker) formára.
    """
    if isinstance(df.columns, pd.MultiIndex):
        lev0 = list(df.columns.get_level_values(0))
        lev1 = list(df.columns.get_level_values(1))
        # ha (Ticker, Field), cseréljük fel
        # Heurisztika: a ticker nevek jellemzően a tickers_hint halmazban vannak
        if set(lev0) >= set(tickers_hint) and any(f in lev1 for f in PRICE_FIELDS):
            new_cols = [(f, t) for (t, f) in df.columns.to_list()]
            df.columns = pd.MultiIndex.from_tuples(new_cols)
        return df
    else:
        # egy ticker esete: alakítsuk MultiIndexre
        cols = pd.MultiIndex.from_product([df.columns, tickers_hint])
        df.columns = cols
        return df

def fetch_prices(tickers, start, end, interval="1d", max_retries=2, sleep_sec=1):
    """
    Árfolyamok letöltése:
    - Előbb batch-ben próbál, ha üres/hibás, egyesével is megpróbálja.
    - A visszatérő DataFrame oszlopait (Field, Ticker) MultiIndexre normalizálja.
    Visszaad: (DataFrame, [sikertelen_tickerek])
    """
    tickers = clean_tickers(tickers)
    if not tickers:
        return pd.DataFrame(), []

    fails = []

    # Batch kísérlet
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

    # Egyenként
    frames = []
    for t in tickers:
        try:
            s = yf.download(
                t,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                threads=False,
                progress=False,
                timeout=YF_TIMEOUT
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
    """Visszaadja az első olyan mezőnevet, ami ténylegesen szerepel a MultiIndex első szintjén."""
    if not isinstance(df.columns, pd.MultiIndex):
        return None
    level0 = list(df.columns.get_level_values(0))
    for f in fields:
        if f in level0:
            return f
    return None

def ticker_has_any_prices(df: pd.DataFrame, tkr: str, fields: list[str]) -> tuple[bool, str | None]:
    """
    Van-e bármelyik preferált mezőből (fields) nem-NaN sor a tickerre?
    Vissza (van?, mezőnév).
    """
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
    for _ in range(7):  # max 1 hét csúszás
        if cur in available_dates:
            return cur
        cur = (pd.to_datetime(cur) + pd.Timedelta(days=1)).date()
    return None

# ==== 3) ESEMÉNYEK BEOLVASÁSA ====
df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df = normalize_columns(df_raw)

# Dátum és ticker normalizálás
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
    # mentünk üres kimenetet is, hogy következetes legyen
    pd.DataFrame().to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
    pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / "summary_raw_returns.csv")
    pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / "summary_abnormal_returns.csv")
    if failed_batch:
        pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    raise RuntimeError("Árfolyamadatok üresek. Ellenőrizd a hálózatot, a dátumokat, a tickereket, vagy a tisztítási szabályokat.")

# 🧠 Ármező választás (Open/Adj Close/Close sorrendben)
use_field = pick_first_available_field(px, PRICE_FIELDS)
if use_field is None:
    pd.DataFrame().to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
    pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / "summary_raw_returns.csv")
    pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"]).to_csv(OUT_DIR / "summary_abnormal_returns.csv")
    if failed_batch:
        pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    raise SystemExit("Nem találtam Open/Adj Close/Close mezőt a letöltött adatokban.")

# Ezt fogjuk használni: a kiválasztott ármező
prices = px[use_field].copy()
# Ha szeletelés miatt egyszintű lett, tegyük vissza (mező, ticker) MultiIndexre
if not isinstance(prices.columns, pd.MultiIndex):
    prices.columns = pd.MultiIndex.from_product([[use_field], prices.columns])

price_wide = prices.copy()
price_wide.index = pd.to_datetime(price_wide.index).date
price_index = set(price_wide.index)

# Diagnosztika: mely tickerek vannak bent, de ezen a mezőn minden értékük NaN
present_cols = set(price_wide.columns.get_level_values(1))
nan_only = []
for t in sorted(present_cols):
    s = price_wide[(use_field, t)]
    if s.isna().all():
        nan_only.append(t)
if nan_only:
    pd.Series(nan_only, name=f"all_nan_in_{use_field}").to_csv(OUT_DIR / f"diagnostics_all_nan_in_{use_field}.csv", index=False)

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

# ==== 6) HOZAMOK KISZÁMOLÁSA ====
records = []
skipped_no_price = 0
skipped_no_entry = 0

for _, row in events.iterrows():
    tkr = clean_ticker(row["ticker"])  # itt is tisztítunk
    if not tkr:
        skipped_no_price += 1
        continue

    fdate = row["file_date"]
    entry_day = next_trading_day((pd.to_datetime(fdate) + pd.Timedelta(days=1)).date(), price_index)
    if entry_day is None:
        skipped_no_entry += 1
        continue

    has_any, field_for_ticker = ticker_has_any_prices(price_wide, tkr, PRICE_FIELDS)
    if not has_any:
        skipped_no_price += 1
        continue
    # ha a ticker más mezőn érhető el, azon számoljunk
    f_field = field_for_ticker or use_field

    # Entry árak
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
        "price_field": f_field,            # melyik mezőn sikerült (Open/Adj Close/Close)
        "entry_price": entry_px,
        "entry_price_spy": entry_spy,
    }

    for w in WINDOWS:
        exit_day = next_trading_day((pd.to_datetime(entry_day) + pd.Timedelta(days=w)).date(), price_index)
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

    records.append(out)

res = pd.DataFrame(records)

# ==== 7) EREDMÉNYEK, STATISZTIKÁK ====
def summarize(col):
    if col not in res.columns:
        return pd.Series({
            "N": 0, "mean": np.nan, "median": np.nan, "stdev": np.nan,
            "hit_%_>0": np.nan, "p25": np.nan, "p75": np.nan
        })
    s = res[col].dropna()
    return pd.Series({
        "N": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "stdev": s.std(),
        "hit_%_>0": (s > 0).mean()*100 if len(s) else np.nan,
        "p25": s.quantile(0.25) if len(s) else np.nan,
        "p75": s.quantile(0.75) if len(s) else np.nan
    })

if res.empty:
    # üres összefoglalók, hogy ne dobjon hibát
    pd.DataFrame().to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
    empty_summary = pd.DataFrame(index=["N","mean","median","stdev","hit_%_>0","p25","p75"])
    empty_summary.to_csv(OUT_DIR / "summary_raw_returns.csv")
    empty_summary.to_csv(OUT_DIR / "summary_abnormal_returns.csv")
    if 'failed_batch' in locals() and failed_batch:
        pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    print("Kész (de nem volt egyetlen érvényes esemény sem a megadott ticker/dátum halmazból).")
    print(f"Skip summary: no_entry={skipped_no_entry}, no_price={skipped_no_price}")
    raise SystemExit(0)

summary = pd.concat({f"ret_{w}d": summarize(f"ret_{w}d") for w in WINDOWS}, axis=1)
ab_summary = pd.concat({f"aret_{w}d": summarize(f"aret_{w}d") for w in WINDOWS}, axis=1)

# ==== 8) MENTÉS ====
res.to_csv(OUT_DIR / "schedule13_event_study.csv", index=False)
summary.to_csv(OUT_DIR / "summary_raw_returns.csv")
ab_summary.to_csv(OUT_DIR / "summary_abnormal_returns.csv")
if 'failed_batch' in locals() and failed_batch:
    pd.Series(failed_batch, name="failed_ticker").to_csv(OUT_DIR / "failed_tickers.csv", index=False)

print("Kész!")
print(f"Események száma: {len(res)}")
print(f"Skip summary: no_entry={skipped_no_entry}, no_price={skipped_no_price}")
print("Raw returns összefoglaló:")
print(summary.round(4))
print("\nAbnormális hozamok (ticker - SPY) összefoglaló:")
print(ab_summary.round(4))
