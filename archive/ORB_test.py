import os, time, requests, math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz

API_KEY = "OASYJ7NakveKTgn4YHArbYp0zkZMiZuv"
SYMBOL  = "AAPL"
# 5 perces gyertyák, egy konkrét napra (US/Eastern)
TRADING_DAY_ET = "2025-10-27"  # példa

def fetch_polygon_aggs(symbol, multiplier=5, timespan="minute", start="2025-09-09", end="2025-09-09"):
    """Polygon v2 aggs lekérés (pl. 5-min gyertyák egy napra)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"adjusted":"true","sort":"asc","limit":"50000","apiKey":API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("results") is None:
        raise RuntimeError(f"Nincs adat: {js}")
    df = pd.DataFrame(js["results"])
    # oszlopok: t(ms), o,h,l,c,v, vw ...
    df["ts_utc"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    # alakítsd US/Eastern-re és szűrd a piaci nyitvatartást (9:30–16:00)
    et = pytz.timezone("US/Eastern")
    df["ts_et"] = df["ts_utc"].dt.tz_convert(et)
    df = df.set_index("ts_et")
    # piacnyitás/ zárás
    start = df.index.normalize()[0] + pd.Timedelta(hours=9, minutes=30)
    end   = df.index.normalize()[0] + pd.Timedelta(hours=16, minutes=0)
    df = df.loc[(df.index>=start)&(df.index<=end)].copy()
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    return df[["open","high","low","close","volume","ts_utc"]]

df = fetch_polygon_aggs(SYMBOL, 5, "minute", TRADING_DAY_ET, TRADING_DAY_ET)

# Opening range: első 15 perc = az első 3 db 5p gyertya
orb_window = 3
if len(df) < orb_window+1:
    raise SystemExit("Túl kevés gyertya a napon.")

orb_high = df["high"].iloc[:orb_window].max()
orb_low  = df["low"].iloc[:orb_window].min()
rng = orb_high - orb_low

# Paraméterek
use_short = False       # csak long példában
entry_buffer = 0.00     # pici puffer a kitöréshez
stop_type = "range_frac" # "percent" vagy "range_frac"
stop_param = 0.5        # ha range_frac: a range 0.5x; ha percent: 0.005 = 0.5%
fee_bps = 5             # 0.05% round-trip (pl. 5 bps oldalanként ~ 0.05% összesen)
slippage_bps = 2        # 0.02% csúszás oldalanként

in_pos = False
entry_price = None
pnl = 0.0
trades = []

prices = df.iloc[orb_window:].copy()  # az OR után kezdünk kereskedni

def bps(x): return 1.0 + x/10000.0

for i in range(len(prices)-1):
    row = prices.iloc[i]
    nxt = prices.iloc[i+1]  # következő gyertya nyitóárán teljesítünk

    # szignál: kitörés
    long_signal = (row["high"] >= orb_high*(1+entry_buffer)) and not in_pos
    short_signal = (use_short and (row["low"] <= orb_low*(1-entry_buffer)) and not in_pos)

    if not in_pos and (long_signal or short_signal):
        side = "long" if long_signal else "short"
        # végrehajtás a következő bar open árán + slippage
        if side=="long":
            fill = nxt["open"]*bps(+slippage_bps)
        else:
            fill = nxt["open"]*bps(-slippage_bps)
        entry_price = fill
        in_pos = True
        trades.append({"time":nxt.name, "side":side, "price":entry_price, "type":"entry"})
        continue

    if in_pos:
        # stop
        if stop_type=="percent":
            stop_dist = stop_param*row["close"]
        else:
            stop_dist = rng*stop_param

        hit_stop = False
        if trades[-1]["side"]=="long":
            stop_level = entry_price - stop_dist
            if row["low"] <= stop_level:
                # kilépsz a következő nyitón - slippage
                exit_price = nxt["open"]*bps(-slippage_bps)
                trade_pnl = (exit_price - entry_price)/entry_price
                trade_pnl -= fee_bps/10000.0  # díj
                pnl += trade_pnl
                in_pos=False; entry_price=None
                trades.append({"time":nxt.name,"side":"long","price":exit_price,"type":"stop_exit","ret":trade_pnl})
                continue
        else:
            stop_level = entry_price + stop_dist
            if row["high"] >= stop_level:
                exit_price = nxt["open"]*bps(+slippage_bps)
                trade_pnl = (entry_price - exit_price)/entry_price
                trade_pnl -= fee_bps/10000.0
                pnl += trade_pnl
                in_pos=False; entry_price=None
                trades.append({"time":nxt.name,"side":"short","price":exit_price,"type":"stop_exit","ret":trade_pnl})
                continue

# nap végi zárás 16:00-kor
if in_pos:
    last_open = prices.iloc[-1]["open"]
    if trades[-1]["side"]=="long":
        exit_price = last_open*bps(-slippage_bps)
        trade_pnl = (exit_price - entry_price)/entry_price
    else:
        exit_price = last_open*bps(+slippage_bps)
        trade_pnl = (entry_price - exit_price)/entry_price
    trade_pnl -= fee_bps/10000.0
    pnl += trade_pnl
    trades.append({"time":prices.index[-1],"side":trades[-1]["side"],"price":exit_price,"type":"eod_exit","ret":trade_pnl})
    in_pos=False

trades_df = pd.DataFrame(trades)
print("\nORB eredmények –", SYMBOL, TRADING_DAY_ET)
print(trades_df)
if len(trades_df)>0 and "ret" in trades_df:
    print("\nÖsszhozam (nap):", round(100*trades_df["ret"].sum(), 3), "%")
else:
    print("\nNem volt kötés.")
