#!/usr/bin/env python3
"""
Pattern Performance Analyzer — Backtest every TA setup against your universe.

This agent:
1. Pulls 2 years of daily data for all tracked tickers
2. Scans every candle for each pattern/setup trigger
3. Tracks forward returns (3, 5, 10, 20 days) after each signal
4. Computes win rates, avg returns, Sharpe per pattern
5. Ranks setups by performance and identifies which work best per ticker
6. Reports the results with clear ranking tables

Usage:
  python3 pattern_analyzer.py --full          # Full analysis (all tickers, 2yr)
  python3 pattern_analyzer.py --quick         # Quick scan (last 6mo, subset)
  python3 pattern_analyzer.py --ticker NVDA   # Single ticker deep dive
  python3 pattern_analyzer.py --patterns rsi,macd,bb  # Specific patterns only
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────

TICKERS = {
    "AI Semis": ["NVDA", "AMD", "AVGO", "MRVL"],
    "AI Infra": ["SMCI", "DELL", "VRT", "STRL"],
    "Cloud": ["MSFT", "GOOGL", "AMZN"],
    "SaaS CRM": ["CRM", "NOW"],
    "SaaS Data": ["SNOW", "DDOG", "MDB"],
    "SaaS Security": ["PANW", "CRWD", "NET", "ZS"],
    "SaaS Creative": ["SHOP", "ADBE", "INTU"],
    "Energy": ["CEG", "VST", "OKLO"],
}

WIDER_TICKERS = [
    "AAPL", "META", "TSLA", "PLTR", "SOUN", "IONQ", "RGTI", "QBTS",
    "ANET", "ESTC", "WDAY", "ZS", "MRNA", "COST", "WMT", "SPY", "QQQ",
    "IWM", "XLK", "SMH", "IBIT",
]

ALL_TICKERS = sorted(set(sum(TICKERS.values(), []) + WIDER_TICKERS))
FORWARD_DAYS = [3, 5, 10, 20]
MIN_SAMPLE_SIZE = 10

# Pattern categories for grouping
PATTERN_TYPES = {
    "rsi_oversold": "mean_reversion",
    "rsi_overbought": "reversal",
    "rsi_oversold_volume": "mean_reversion",
    "macd_bullish_cross": "momentum",
    "macd_bearish_cross": "reversal",
    "bb_lower_touch": "mean_reversion",
    "bb_upper_touch": "reversal",
    "bb_squeeze": "volatility",
    "higher_low": "trend",
    "higher_high": "trend",
    "volume_breakout": "breakout",
    "golden_cross": "trend",
    "death_cross": "reversal",
    "rsi_oversold_higher_low": "combined",
    "bb_squeeze_volume": "combined",
}

# ── Pattern Detectors ─────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[dict]:
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()
    signals = []

    rsi = ta.rsi(close, length=14)
    macd_data = ta.macd(close)
    bb = ta.bbands(close, length=20, std=2)
    sma20 = ta.sma(close, length=20)
    sma50 = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)
    atr = ta.atr(high, low, close, length=14)
    vol_ma20 = volume.rolling(20).mean()
    vol_ma50 = volume.rolling(50).mean()

    def _col(data, *keys):
        if data is None or not isinstance(data, pd.DataFrame):
            return None
        for c in data.columns:
            for k in keys:
                if k in c:
                    return data[c]
        return None

    bb_lower = _col(bb, 'BBL', 'lower')
    bb_upper = _col(bb, 'BBU', 'upper')
    bb_mid = _col(bb, 'BBM', 'mid')
    macd_line = _col(macd_data, 'MACD_')
    macd_signal = _col(macd_data, 'SIGNAL', 'signal')
    macd_hist = _col(macd_data, 'HIST', 'histogram')

    for i in range(60, len(df)):
        date = df.index[i]
        price = float(close.iloc[i])
        vol_i = float(volume.iloc[i])
        vm20 = float(vol_ma20.iloc[i]) if not pd.isna(vol_ma20.iloc[i]) else 0

        rsi_v = float(rsi.iloc[i]) if rsi is not None and not pd.isna(rsi.iloc[i]) else None

        # ── RSI Oversold ──
        if rsi_v is not None and rsi_v < 30:
            signals.append({"date": date, "price": price, "pattern": "rsi_oversold",
                "strength": round(30 - rsi_v, 1), "detail": f"RSI={rsi_v:.1f}"})
        elif rsi_v is not None and rsi_v > 70:
            signals.append({"date": date, "price": price, "pattern": "rsi_overbought",
                "strength": round(rsi_v - 70, 1), "detail": f"RSI={rsi_v:.1f}"})

        # ── RSI oversold + volume ──
        if rsi_v is not None and rsi_v < 35 and vm20 > 0 and (vol_i / vm20) > 1.5:
            signals.append({"date": date, "price": price, "pattern": "rsi_oversold_volume",
                "strength": round((35 - rsi_v) * (vol_i / vm20), 1),
                "detail": f"RSI={rsi_v:.1f} Vol={vol_i/vm20:.1f}x"})

        # ── MACD cross ──
        if macd_line is not None and macd_signal is not None and i > 0:
            ml_c = float(macd_line.iloc[i]); ml_p = float(macd_line.iloc[i-1])
            ms_c = float(macd_signal.iloc[i]); ms_p = float(macd_signal.iloc[i-1])
            if not any(pd.isna(x) for x in [ml_c, ml_p, ms_c, ms_p]):
                if ml_p <= ms_p and ml_c > ms_c:
                    signals.append({"date": date, "price": price, "pattern": "macd_bullish_cross",
                        "strength": round(ml_c - ms_c, 2), "detail": f"MACD={ml_c:.2f} Sig={ms_c:.2f}"})
                elif ml_p >= ms_p and ml_c < ms_c:
                    signals.append({"date": date, "price": price, "pattern": "macd_bearish_cross",
                        "strength": round(ms_c - ml_c, 2), "detail": f"MACD={ml_c:.2f} Sig={ms_c:.2f}"})

        # ── BB touches ──
        if bb_lower is not None and bb_upper is not None:
            bl = float(bb_lower.iloc[i]); bu = float(bb_upper.iloc[i]); bm = float(bb_mid.iloc[i])
            if not any(pd.isna(x) for x in [bl, bu, bm]):
                if price <= bl * 1.01:
                    signals.append({"date": date, "price": price, "pattern": "bb_lower_touch",
                        "strength": round((bl - price) / bl * 100, 2), "detail": f"Price={price:.2f} LB={bl:.2f}"})
                elif price >= bu * 0.99:
                    signals.append({"date": date, "price": price, "pattern": "bb_upper_touch",
                        "strength": round((price - bu) / bu * 100, 2), "detail": f"Price={price:.2f} UB={bu:.2f}"})

        # ── BB Squeeze ──
        if bb_lower is not None and bb_upper is not None and bb_mid is not None and i >= 10:
            try:
                bw = float(((bb_upper.iloc[i] - bb_lower.iloc[i]) / bb_mid.iloc[i]))
                bw10 = float(((bb_upper.iloc[i-10] - bb_lower.iloc[i-10]) / bb_mid.iloc[i-10]))
                if bw < bw10 * 0.85 and bw < 0.05:
                    signals.append({"date": date, "price": price, "pattern": "bb_squeeze",
                        "strength": round((1 - bw / bw10) * 100, 1), "detail": f"W={bw:.4f}->{bw10:.4f}"})
            except (IndexError, ValueError):
                pass

        # ── Higher Low / Higher High ──
        if i >= 25:
            low10 = close.iloc[i-10:i+1].min()
            low20 = close.iloc[i-20:i+1].min()
            high10 = close.iloc[i-10:i+1].max()
            high20 = close.iloc[i-20:i+1].max()
            if low10 > low20:
                signals.append({"date": date, "price": price, "pattern": "higher_low",
                    "strength": round((low10 - low20) / low20 * 100, 2), "detail": f"10dL={low10:.2f} 20dL={low20:.2f}"})
            if high10 > high20:
                signals.append({"date": date, "price": price, "pattern": "higher_high",
                    "strength": round((high10 - high20) / high20 * 100, 2), "detail": f"10dH={high10:.2f} 20dH={high20:.2f}"})

        # ── Volume Breakout ──
        if vm20 > 0 and sma20 is not None and not pd.isna(sma20.iloc[i]):
            vr = vol_i / vm20
            pvsma = (price / float(sma20.iloc[i]) - 1) * 100
            if vr > 2.0 and pvsma > 3:
                signals.append({"date": date, "price": price, "pattern": "volume_breakout",
                    "strength": round(vr * pvsma, 1), "detail": f"Vol={vr:.1f}x SMA+{pvsma:.1f}%"})

        # ── Golden/Death Cross ──
        if sma50 is not None and sma200 is not None and i > 0:
            s50_c = float(sma50.iloc[i]); s50_p = float(sma50.iloc[i-1])
            s200_c = float(sma200.iloc[i]); s200_p = float(sma200.iloc[i-1])
            if not any(pd.isna(x) for x in [s50_c, s50_p, s200_c, s200_p]):
                if s50_p <= s200_p and s50_c > s200_c:
                    signals.append({"date": date, "price": price, "pattern": "golden_cross",
                        "strength": round((s50_c - s200_c) / s200_c * 100, 2),
                        "detail": f"50SMA={s50_c:.2f} 200SMA={s200_c:.2f}"})
                elif s50_p >= s200_p and s50_c < s200_c:
                    signals.append({"date": date, "price": price, "pattern": "death_cross",
                        "strength": round((s200_c - s50_c) / s200_c * 100, 2),
                        "detail": f"50SMA={s50_c:.2f} 200SMA={s200_c:.2f}"})

        # ── Combined: RSI Oversold + Higher Low ──
        if rsi_v is not None and rsi_v < 35 and i >= 25:
            low10 = close.iloc[i-10:i+1].min()
            low20 = close.iloc[i-20:i+1].min()
            if low10 > low20:
                signals.append({"date": date, "price": price, "pattern": "rsi_oversold_higher_low",
                    "strength": round((35 - rsi_v) * (low10 / low20), 1),
                    "detail": f"RSI={rsi_v:.1f} HL={low10:.2f}>{low20:.2f}"})

        # ── BB Squeeze + Volume ──
        if bb_lower is not None and bb_upper is not None and bb_mid is not None and vm20 > 0 and i >= 10:
            try:
                bw_i = float(((bb_upper.iloc[i] - bb_lower.iloc[i]) / bb_mid.iloc[i]))
                bw_10 = float(((bb_upper.iloc[i-10] - bb_lower.iloc[i-10]) / bb_mid.iloc[i-10]))
                if bw_i < bw_10 * 0.8 and (vol_i / vm20) > 1.3:
                    signals.append({"date": date, "price": price, "pattern": "bb_squeeze_volume",
                        "strength": round((1 - bw_i / bw_10) * (vol_i / vm20), 1),
                        "detail": f"Sq={bw_i:.3f}->{bw_10:.3f} V={vol_i/vm20:.1f}x"})
            except (IndexError, ValueError):
                pass

    return signals


def compute_forward_returns(df: pd.DataFrame, signals: list[dict]) -> list[dict]:
    close = df['Close'].squeeze()
    date_index = {d.date(): i for i, d in enumerate(df.index)}
    results = []

    for sig in signals:
        sig_date = sig["date"]
        if isinstance(sig_date, pd.Timestamp):
            sig_date = sig_date.date()
        elif isinstance(sig_date, datetime):
            sig_date = sig_date.date()

        entry_idx = date_index.get(sig_date)
        if entry_idx is None:
            continue
        entry_price = float(close.iloc[entry_idx])
        if entry_price <= 0:
            continue

        sig["entry_price"] = entry_price
        sig["entry_index"] = entry_idx
        for fwd in FORWARD_DAYS:
            exit_idx = entry_idx + fwd
            if exit_idx < len(close):
                ret = (float(close.iloc[exit_idx]) - entry_price) / entry_price * 100
                sig[f"ret_{fwd}d"] = round(ret, 2)
                sig[f"win_{fwd}d"] = 1 if ret > 0 else 0
            else:
                sig[f"ret_{fwd}d"] = None
                sig[f"win_{fwd}d"] = None
        results.append(sig)
    return results


def analyze_patterns(tickers: list[str], period: str = "2y") -> dict:
    pattern_stats = defaultdict(lambda: {
        "signals": 0, "wins_5d": 0,
        "total_ret_5d": 0.0, "total_ret_sq_5d": 0.0,
        "best_5d": -999.0, "worst_5d": 999.0,
        "by_ticker": defaultdict(lambda: {"signals": 0, "wins": 0, "sum_ret": 0.0}),
        "by_type": defaultdict(int),
    })
    total_signals = 0
    total_processed = 0

    print(f"\n  📡 Scanning {len(tickers)} tickers ({period})...")
    print(f"  {'─' * 55}")

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker:<6s}", end="", flush=True)
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 100:
                print(f"  ⏭️  short")
                continue
            signals = detect_patterns(df)
            if not signals:
                print(f"  ⏭️  no patterns")
                continue
            results = compute_forward_returns(df, signals)
            total_signals += len(results)
            total_processed += 1

            for r in results:
                ps = pattern_stats[r["pattern"]]
                ps["signals"] += 1
                ps["by_type"][PATTERN_TYPES.get(r["pattern"], "unknown")] += 1

                bt = ps["by_ticker"][ticker]
                bt["signals"] += 1

                ret_5d = r.get("ret_5d")
                if ret_5d is not None:
                    ps["total_ret_5d"] += ret_5d
                    ps["total_ret_sq_5d"] += ret_5d ** 2
                    if ret_5d > 0:
                        ps["wins_5d"] += 1
                        bt["wins"] += 1
                    bt["sum_ret"] += ret_5d
                    ps["best_5d"] = max(ps["best_5d"], ret_5d)
                    ps["worst_5d"] = min(ps["worst_5d"], ret_5d)

            print(f"  ✅ {len(results)} sig")
        except Exception as e:
            print(f"  ❌ {e}")

    print(f"  {'─' * 55}")
    print(f"  ✅ {total_processed}/{len(tickers)} tkrs, {total_signals} total signals")

    final = {}
    for pat, s in sorted(pattern_stats.items(), key=lambda x: x[1]["signals"], reverse=True):
        if s["signals"] < MIN_SAMPLE_SIZE:
            continue
        wr = s["wins_5d"] / s["signals"] * 100
        avg_r = s["total_ret_5d"] / s["signals"]
        var = s["total_ret_sq_5d"] / s["signals"] - avg_r ** 2
        std = np.sqrt(var) if var > 0 else 0.001
        sh = avg_r / std if std > 0 else 0

        best_t = max(s["by_ticker"].items(),
            key=lambda x: x[1]["sum_ret"] / x[1]["signals"] if x[1]["signals"] > 0 else -999)

        final[pat] = {
            "signals": s["signals"],
            "win_rate_5d": round(wr, 1),
            "avg_ret_5d": round(avg_r, 2),
            "sharpe_5d": round(sh, 3),
            "best_5d": round(s["best_5d"], 2),
            "worst_5d": round(s["worst_5d"], 2),
            "types": dict(s["by_type"]),
            "best_ticker": best_t[0],
            "best_ticker_avg_ret": round(best_t[1]["sum_ret"] / best_t[1]["signals"], 2) if best_t[1]["signals"] > 0 else 0,
            "best_ticker_signals": best_t[1]["signals"],
        }
    return {
        "pattern_stats": final,
        "total_signals": total_signals,
        "total_tickers_processed": total_processed,
        "tickers_scanned": len(tickers),
    }


def print_report(results: dict):
    stats = results["pattern_stats"]
    by_wr = sorted(stats.items(), key=lambda x: x[1]["win_rate_5d"], reverse=True)
    by_sh = sorted(stats.items(), key=lambda x: x[1]["sharpe_5d"], reverse=True)

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  📊 PATTERN PERFORMANCE ANALYZER                          ║
║  {results['total_signals']} sig across {results['total_tickers_processed']} tkrs
╚════════════════════════════════════════════════════════════╝
""")

    # TOP by win rate
    print(f"  🏆 TOP SETUPS BY 5-DAY WIN RATE")
    print(f"  {'─' * 95}")
    print(f"  {'Rank':<6} {'Pattern':<30} {'Win%':<8} {'AvgRet':<8} {'Sharpe':<8} {'Sig':<6} {'Best Tkr':<10}")
    print(f"  {'─' * 95}")
    for rank, (name, ps) in enumerate(by_wr[:12], 1):
        bar = "█" * max(1, int(ps["win_rate_5d"] / 5))
        print(f"  {rank:<6} {name:<30} {ps['win_rate_5d']:<7.1f}% {ps['avg_ret_5d']:<7.2f}% "
              f"{ps['sharpe_5d']:<8.3f} {ps['signals']:<6} {ps['best_ticker']:<10} {bar}")

    # WORST
    print(f"\n  💀 WORST (avoid)")
    print(f"  {'─' * 55}")
    for name, ps in reversed(by_wr[-5:]):
        if ps["win_rate_5d"] >= 50:
            break
        print(f"  ❌ {name:<30s} {ps['win_rate_5d']:5.1f}% wr ({ps['signals']} sig, {ps['avg_ret_5d']:+.2f}%)")

    # By Sharpe
    print(f"\n  📈 TOP BY RISK-ADJ (Sharpe)")
    print(f"  {'─' * 60}")
    for name, ps in by_sh[:8]:
        ic = "📗" if ps["sharpe_5d"] > 0.1 else "📕"
        print(f"  {ic} {name:<30s} S={ps['sharpe_5d']:.3f}  W={ps['win_rate_5d']:.1f}%  R={ps['avg_ret_5d']:+.2f}%")

    # By category
    print(f"\n  🏷️  CATEGORIES")
    print(f"  {'─' * 50}")
    from collections import defaultdict as dd
    cats = dd(lambda: {"c": 0, "w": 0.0, "r": 0.0})
    for name, ps in stats.items():
        for t, v in ps.get("types", {}).items():
            cats[t]["c"] += v
            cats[t]["w"] += v * ps["win_rate_5d"] / 100
            cats[t]["r"] += v * ps["avg_ret_5d"]
    for t in sorted(cats.keys()):
        wr = cats[t]["w"] / cats[t]["c"] * 100 if cats[t]["c"] > 0 else 0
        avg = cats[t]["r"] / cats[t]["c"] if cats[t]["c"] > 0 else 0
        print(f"  {t:<18s} {cats[t]['c']:4d} sig | Win {wr:5.1f}% | Avg {avg:+.2f}%")

    # Rules of thumb
    print(f"\n  💡 QUICK RULES")
    print(f"  {'─' * 50}")
    for rank, (name, ps) in enumerate(by_wr[:3], 1):
        d = "BUY" if ps["avg_ret_5d"] > 0 else "SELL"
        print(f"  {rank}. {d}: {name} → win {ps['win_rate_5d']:.0f}%, avg {ps['avg_ret_5d']:+.2f}%")

    print()


def save_report(results: dict):
    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)

    jp = out_dir / "pattern_analysis.json"
    with open(jp, "w") as f:
        json.dump(json.loads(json.dumps(results, default=str)), f, indent=2)
    print(f"  💾 JSON: {jp}")

    stats = results.get("pattern_stats", {})
    by_wr = sorted(stats.items(), key=lambda x: x[1]["win_rate_5d"], reverse=True)

    md = f"""# Pattern Performance Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Tickers:** {results.get('tickers_scanned', 0)}
**Signals:** {results.get('total_signals', 0)}
**Tickers w/ Data:** {results.get('total_tickers_processed', 0)}

---

## Top by 5-Day Win Rate

| Rank | Pattern | Win% | AvgRet | Sharpe | Signals | Best On |
|------|---------|------|--------|--------|---------|---------|
"""
    for rank, (name, ps) in enumerate(by_wr[:15], 1):
        md += f"| {rank} | {name} | {ps['win_rate_5d']:.1f}% | {ps['avg_ret_5d']:+.2f}% | {ps['sharpe_5d']:.3f} | {ps['signals']} | {ps['best_ticker']} |\n"

    md += "\n## Worst\n\n| Pattern | Win% | Signals |\n|---------|------|---------|\n"
    for name, ps in reversed(by_wr[-8:]):
        if ps["win_rate_5d"] >= 50:
            continue
        md += f"| {name} | {ps['win_rate_5d']:.1f}% | {ps['signals']} |\n"

    from collections import defaultdict as dd
    cats = dd(lambda: {"c": 0, "w": 0.0, "r": 0.0})
    for name, ps in stats.items():
        for t, v in ps.get("types", {}).items():
            cats[t]["c"] += v
            cats[t]["w"] += v * ps["win_rate_5d"] / 100
            cats[t]["r"] += v * ps["avg_ret_5d"]

    md += "\n## Categories\n\n| Category | Signals | Win% | AvgRet |\n|----------|---------|------|--------|\n"
    for t in sorted(cats.keys()):
        wr = cats[t]["w"] / cats[t]["c"] * 100 if cats[t]["c"] > 0 else 0
        avg = cats[t]["r"] / cats[t]["c"] if cats[t]["c"] > 0 else 0
        md += f"| {t} | {cats[t]['c']} | {wr:.1f}% | {avg:+.2f}% |\n"

    md += "\n---\n*Pattern Performance Analyzer — Rocky*"
    mp = out_dir / "pattern_analysis_report.md"
    mp.write_text(md)
    print(f"  💾 MD: {mp}")


def main():
    parser = argparse.ArgumentParser(description="Pattern Performance Analyzer")
    parser.add_argument("--full", action="store_true", help="All tickers, 2yr")
    parser.add_argument("--quick", action="store_true", help="Quick scan (subset, 6mo)")
    parser.add_argument("--ticker", "-t", help="Single ticker")
    parser.add_argument("--period", "-p", default="2y", help="Data period")
    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  📊 Pattern Performance Analyzer                           ║
║  Backtest every TA setup → find what actually works        ║
╚════════════════════════════════════════════════════════════╝
""")

    if args.ticker:
        tickers = [args.ticker.upper()]
        period = args.period
    elif args.quick:
        tickers = ["SPY", "QQQ", "NVDA", "MSFT", "AAPL", "AMZN", "TSLA", "GOOGL"]
        period = "6mo"
    elif args.full:
        tickers = ALL_TICKERS
        period = args.period or "2y"
    else:
        tickers = sum([v for v in TICKERS.values()], [])
        tickers += ["SPY", "QQQ"]
        period = "1y"

    t0 = time.time()
    results = analyze_patterns(tickers, period)
    elapsed = time.time() - t0

    print_report(results)
    save_report(results)
    print(f"  ⏱️  {elapsed:.0f}s total\n")


if __name__ == "__main__":
    main()
