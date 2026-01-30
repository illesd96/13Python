import os, time, math, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# ========= PARAMÉTEREK =========
API_KEY = "OASYJ7NakveKTgn4YHArbYp0zkZMiZuv"
SYMBOL  = "AAPL"
MULTIPLIER = 5            # 5 perces gyertyák
TIMESPAN   = "minute"
ENTRY_BUFFER_LONG = 0.000   # 0.0 = azonnali törésnél, pl. 0.001 = +0.1%
ENTRY_BUFFER_SHORT = 0.000
USE_SHORT = True
STOP_TYPE_LONG = "range_frac"  # "range_frac" vagy "percent"
STOP_PARAM_LONG = 0.5          # range_frac: 0.5 * opening range; percent: pl. 0.005 = 0.5%
STOP_TYPE_SHORT  = "range_frac"
STOP_PARAM_SHORT = 0.5
FEE_BPS = 5               # round-trip díj ~0.05%
SLIPPAGE_BPS = 2          # oldalanként ~0.02%
START_CAPITAL = 10_000.0
# Dátumtartomány: ISO stringek (US/Eastern munkanapok közül ami elérhető)
DATE_FROM = "2025-10-01"
DATE_TO   = "2025-10-27"
# ===============================

et = pytz.timezone("US/Eastern")

def bps(x): 
    return 1.0 + x/10000.0

def fetch_polygon_day(symbol, day_str):
    """Visszaadja a megadott nap 5p gyertyáit (9:30-16:00, US/Eastern)."""
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
           f"{MULTIPLIER}/{TIMESPAN}/{day_str}/{day_str}")
    params = {"adjusted":"true","sort":"asc","limit":"50000","apiKey":API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("results") is None or len(js.get("results", [])) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(js["results"])
    df["ts_utc"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["ts_et"]  = df["ts_utc"].dt.tz_convert(et)
    df = df.set_index("ts_et")
    mkt_open = df.index.normalize()[0] + pd.Timedelta(hours=9, minutes=30)
    mkt_close= df.index.normalize()[0] + pd.Timedelta(hours=16, minutes=0)
    df = df.loc[(df.index>=mkt_open) & (df.index<=mkt_close)].copy()
    if df.empty:
        return df
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    return df[["open","high","low","close","volume","ts_utc"]]

def backtest_orb_one_day(df, day_str, symbol=SYMBOL):
    """Egy napos ORB backtest. Visszaad trades_df (sorok: entry/exit) és napi hozam (sum rets)."""
    results = []
    if df is None or df.empty:
        return pd.DataFrame(), 0.0, np.nan, np.nan

    # Opening range: első 15 perc = 3 db 5p gyertya
    orb_window = 3
    if len(df) < orb_window + 2:
        return pd.DataFrame(), 0.0, np.nan, np.nan

    orb_high = df["high"].iloc[:orb_window].max()
    orb_low  = df["low"].iloc[:orb_window].min()
    rng = orb_high - orb_low
    prices = df.iloc[orb_window:].copy()

    in_pos = False
    entry_price = None
    day_rets = []

    for i in range(len(prices)-1):
        row = prices.iloc[i]
        nxt = prices.iloc[i+1]  # végrehajtás a következő gyertya nyitón
        long_signal  = (row["high"] >= orb_high*(1 + ENTRY_BUFFER_LONG)) and not in_pos
        short_signal = (USE_SHORT and (row["low"]  <= orb_low  * (1 - ENTRY_BUFFER_SHORT)) and not in_pos)


        # ENTRY
        long_signal  = (row["high"] >= orb_high * (1 + ENTRY_BUFFER_LONG))  and not in_pos
        short_signal = (USE_SHORT and (row["low"]  <= orb_low  * (1 - ENTRY_BUFFER_SHORT)) and not in_pos)
        
        if not in_pos and (long_signal or short_signal):
            side = "long" if long_signal else "short"
            if side == "long":
                fill = nxt["open"] * bps(+SLIPPAGE_BPS)
            else:
                fill = nxt["open"] * bps(-SLIPPAGE_BPS)
            entry_price = fill
            in_pos = True
            results.append({"day":day_str, "time":nxt.name, "side":side, "price":entry_price, "type":"entry"})
            continue

        # STOP MENEDZSMENT
        if in_pos:
            if results[-1]["side"] == "long":
                # long stop
                if STOP_TYPE_LONG == "percent":
                    stop_dist = STOP_PARAM_LONG * row["close"]
                else:
                    stop_dist = rng * STOP_PARAM_LONG

                stop_level = entry_price - stop_dist
                if row["low"] <= stop_level:
                    exit_price = nxt["open"] * bps(-SLIPPAGE_BPS)
                    trade_ret = (exit_price - entry_price) / entry_price
                    trade_ret -= FEE_BPS/10000.0
                    day_rets.append(trade_ret)
                    results.append({"day":day_str,"time":nxt.name,"side":"long","price":exit_price,"type":"stop_exit","ret":trade_ret})
                    in_pos=False; entry_price=None
                    continue

            else:
                # short stop
                if STOP_TYPE_SHORT == "percent":
                    stop_dist = STOP_PARAM_SHORT * row["close"]
                else:
                    stop_dist = rng * STOP_PARAM_SHORT

                stop_level = entry_price + stop_dist
                if row["high"] >= stop_level:
                    exit_price = nxt["open"] * bps(+SLIPPAGE_BPS)
                    trade_ret = (entry_price - exit_price) / entry_price
                    trade_ret -= FEE_BPS/10000.0
                    day_rets.append(trade_ret)
                    results.append({"day":day_str,"time":nxt.name,"side":"short","price":exit_price,"type":"stop_exit","ret":trade_ret})
                    in_pos=False; entry_price=None
                    continue

    # EOD EXIT
    if in_pos:
        last_open = prices.iloc[-1]["open"]
        if results[-1]["side"] == "long":
            exit_price = last_open * bps(-SLIPPAGE_BPS)
            trade_ret = (exit_price - entry_price) / entry_price
        else:
            exit_price = last_open * bps(+SLIPPAGE_BPS)
            trade_ret = (entry_price - exit_price) / entry_price
        trade_ret -= FEE_BPS/10000.0
        day_rets.append(trade_ret)
        results.append({"day":day_str,"time":prices.index[-1],"side":results[-1]["side"],"price":exit_price,"type":"eod_exit","ret":trade_ret})

    trades_df = pd.DataFrame(results)
    day_ret = float(np.nansum(trades_df["ret"])) if "ret" in trades_df else 0.0
    return trades_df, day_ret, orb_high, orb_low

def metrics_from_daily(daily):
    """Sharpe (annualized), max drawdown, win rate, összhozam, equity curve."""
    eq = [START_CAPITAL]
    for r in daily["day_ret"].fillna(0.0).values:
        eq.append(eq[-1]*(1.0 + r))
    equity = pd.Series(eq[1:], index=daily.index, name="equity")

    # Max drawdown
    rollmax = equity.cummax()
    dd = (equity - rollmax) / rollmax
    max_dd = dd.min()

    # Sharpe (annualizált, 252 napi feltételezés)
    mu = daily["day_ret"].mean()
    sigma = daily["day_ret"].std(ddof=1)
    sharpe = (mu / sigma * np.sqrt(252)) if sigma and sigma > 0 else np.nan

    # Win rate
    wins = (daily["day_ret"] > 0).sum()
    count = daily["day_ret"].notna().sum()
    win_rate = wins / count * 100.0 if count > 0 else np.nan

    total_return = equity.iloc[-1] / START_CAPITAL - 1.0
    return {
        "sharpe": sharpe,
        "max_drawdown": float(max_dd) if not pd.isna(max_dd) else np.nan,
        "win_rate_pct": win_rate,
        "total_return_pct": total_return*100.0,
    }, equity, dd

def main():
    # dátumlista (munkanapokra lő, de ha nincs adat, kihagyja)
    dates = pd.bdate_range(DATE_FROM, DATE_TO, freq="C")  # business days
    all_trades = []
    daily_rows = []

    for d in dates:
        day_str = d.strftime("%Y-%m-%d")
        try:
            df = fetch_polygon_day(SYMBOL, day_str)
        except Exception as e:
            print(f"{day_str} hiba (API/kapcsolat): {e}")
            time.sleep(0.25)
            continue

        if df.empty:
            print(f"{day_str}: nincs adat (piac zárva/ünnepnap?).")
            continue

        trades_df, day_ret, orb_high, orb_low = backtest_orb_one_day(df, day_str, SYMBOL)
        if not trades_df.empty:
            all_trades.append(trades_df)
        daily_rows.append({"date": day_str, "day_ret": day_ret, "orb_high":orb_high, "orb_low":orb_low})
        # kíméletes az API-hoz
        time.sleep(0.15)

    if len(daily_rows) == 0:
        print("Nincs backtestelhető nap.")
        return

    daily = pd.DataFrame(daily_rows).set_index("date")
    if len(all_trades) > 0:
        trades = pd.concat(all_trades, ignore_index=True)
    else:
        trades = pd.DataFrame(columns=["day","time","side","price","type","ret"])

    # METRIKÁK
    m, equity, dd = metrics_from_daily(daily)
    print("\n=== ÖSSZEFOGLALÓ ===")
    print(f"Symbol: {SYMBOL}  Időszak: {DATE_FROM} → {DATE_TO}")
    print(f"Összhozam: {m['total_return_pct']:.2f}%")
    print(f"Sharpe: {m['sharpe']:.2f}   Max Drawdown: {m['max_drawdown']:.2%}")
    print(f"Win rate: {m['win_rate_pct']:.1f}%")
    print(f"Napi megfigyelések: {len(daily)}   Tranzakciós napok: {(daily['day_ret']!=0).sum()}")

    # MENTÉSEK
    out_prefix = f"ORB_{SYMBOL}_{DATE_FROM}_to_{DATE_TO}"
    daily.to_csv(f"{out_prefix}_daily.csv")           # napi hozam + OR szintek
    trades.to_csv(f"{out_prefix}_trades.csv", index=False)
    equity.to_csv(f"{out_prefix}_equity.csv", header=True)

    print(f"\nCSV mentve:\n  - {out_prefix}_daily.csv\n  - {out_prefix}_trades.csv\n  - {out_prefix}_equity.csv")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

out_prefix = f"ORB_{SYMBOL}_{DATE_FROM}_to_{DATE_TO}"

# CSV beolvasás
daily = pd.read_csv(f"{out_prefix}_daily.csv", index_col="date")
equity = pd.read_csv(f"{out_prefix}_equity.csv", index_col=0, names=["equity"])

# Típuskonverzió: equity-t biztosan float típusra alakítjuk
equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")

# Drawdown számítás
dd = (equity["equity"] / equity["equity"].cummax()) - 1

# Grafikon rajzolás
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(equity.index, equity["equity"], color="green", label="Equity")
ax1.set_ylabel("Tőke ($)")
ax1.set_title(f"Equity Curve – {SYMBOL} ({DATE_FROM} → {DATE_TO})")

# Drawdown másodlagos tengelyen
ax2 = ax1.twinx()
ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.2, label="Drawdown")
ax2.set_ylabel("Drawdown")
ax2.set_ylim(-1, 0)

ax1.legend(loc="upper left")
plt.tight_layout()
plt.show()
