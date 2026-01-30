#!/usr/bin/env python3
import argparse, subprocess, sys, json
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Paraméter-söprő a schedule13 backtesthez.")
    p.add_argument("--excel", type=str, default="schedule13_events.xlsx", help="Bemeneti Excel fájl")
    p.add_argument("--sheet", type=str, default="0", help="Lap neve vagy indexe")
    p.add_argument("--outdir", type=str, default="sweep_outputs", help="Söprés kimeneti gyökérmappa")
    p.add_argument("--script", type=str, default="schedule13_scraper.py", help="A backtest script neve")
    p.add_argument("--stop-type", type=str, default="atr", choices=["atr","pct","none"], help="Stop típusa")
    p.add_argument("--stop-k", type=float, default=2.0, help="ATR szorzó (atr), vagy százalék (pct)")
    p.add_argument("--price-field", type=str, default="Open", choices=["Open","Close","Adj Close"], help="Belépő ár mező")
    p.add_argument("--atr-period", type=int, default=14, help="ATR periódus")
    p.add_argument("--pre-days", type=int, default=30, help="file_date előtti napok letöltése")
    p.add_argument("--post-days", type=int, default=240, help="file_date utáni napok letöltése (ha nincs time exit)")
    p.add_argument("--no-time-exit", action="store_true", help="Kapcsold ki az időzárást")
    p.add_argument("--tickers-limit", type=int, default=0, help="Debug: csak az első N tickert futtasd")
    # A söprés rácsai (szabadon szerkeszthető)
    p.add_argument("--tp-grid", type=str, default="[0.05,0.10,0.15,0.0]", help="TP százalékok listája (JSON), pl. [0.05,0.10,0.0]")
    p.add_argument("--trail-grid", type=str, default="[0.0,3.0]", help="Trailing ATR-szorzók listája (JSON), pl. [0.0,3.0]")
    return p.parse_args()

def run_one(combo_outdir: Path, script: Path, excel, sheet, price_field, atr_period,
            stop_type, stop_k, tp_pct, trail_k, pre_days, post_days, no_time_exit, tickers_limit):
    combo_outdir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, str(script),
        "--excel", str(excel),
        "--sheet", str(sheet),
        "--price-field", price_field,
        "--atr-period", str(atr_period),
        "--stop-type", stop_type,
        "--stop-k", str(stop_k),
        "--tp-pct", str(tp_pct),
        "--trail-k", str(trail_k),
        "--outdir", str(combo_outdir),
        "--pre-days", str(pre_days),
        "--post-days", str(post_days),
    ]
    if no_time_exit:
        args.append("--no-time-exit")
    if tickers_limit and tickers_limit > 0:
        args += ["--tickers-limit", str(tickers_limit)]

    print(f"[RUN] tp_pct={tp_pct}  trail_k={trail_k}  -> {combo_outdir}")
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[WARN] Futási hiba (exit {res.returncode})\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    else:
        # rövid log
        print(res.stdout.strip())

def collect_results(root: Path):
    rows = []
    for combo_dir in sorted(root.glob("tp*_trail*")):
        trades = combo_dir / "schedule13_stop_trades.csv"
        summary = combo_dir / "schedule13_stop_summary.csv"
        # paraméterek visszafejtése a mappanévből
        name = combo_dir.name  # pl. tp0.10_trail3.00
        try:
            tp_str = name.split("_")[0].replace("tp","")    # "0.10"
            tr_str = name.split("_")[1].replace("trail","") # "3.00"
            tp = float(tp_str)
            tr = float(tr_str)
        except Exception:
            tp, tr = None, None

        n_trades = None
        winrate = None
        mean_pnl = None
        median_pnl = None
        exit_counts = {}

        if trades.exists():
            tdf = pd.read_csv(trades)
            n_trades = len(tdf)
            if n_trades:
                mean_pnl = tdf["pnl_pct"].mean()
                median_pnl = tdf["pnl_pct"].median()
                winrate = (tdf["pnl_pct"] > 0).mean() * 100
                exit_counts = tdf["exit_reason"].value_counts().to_dict()

        row = {
            "tp_pct": tp,
            "trail_k": tr,
            "n_trades": n_trades,
            "mean_pnl": mean_pnl,
            "median_pnl": median_pnl,
            "winrate_%": winrate,
            "exit_mix": json.dumps(exit_counts, ensure_ascii=False)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["tp_pct","trail_k"], na_position="last")

def main():
    a = parse_args()
    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=True)
    script = Path(a.script)

    tp_grid = json.loads(a.tp_grid)
    trail_grid = json.loads(a.trail_grid)

    # Futtatás rácson
    for tp in tp_grid:
        for tr in trail_grid:
            combo_dir = root / f"tp{tp:.2f}_trail{tr:.2f}"
            run_one(combo_dir, script, a.excel, a.sheet, a.price_field, a.atr_period,
                    a.stop_type, a.stop_k, tp, tr, a.pre_days, a.post_days, a.no_time_exit, a.tickers_limit)

    # Aggregált eredmények
    res = collect_results(root)
    out_csv = root / "sweep_results.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n[SÖPRÉS KÉSZ] Összesítés: {out_csv}")
    if not res.empty:
        print(res.to_string(index=False))

if __name__ == "__main__":
    main()
