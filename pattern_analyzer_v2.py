#!/usr/bin/env python3
"""
Pattern Performance Analyzer v2 — Production-grade backtesting engine.

Backtests every TA setup against your stock universe with:
- Long/short separation (bullish vs bearish patterns)
- VIX regime filtering (high vol vs low vol)
- Market cap / sector buckets
- Walk-forward validation (train/test split)
- Multi-horizon analysis (3/5/10/20 day holds)
- Per-ticker and per-sector performance breakdowns

Usage:
  python3 pattern_analyzer_v2.py --full     # Full 46 ticker, 2yr analysis
  python3 pattern_analyzer_v2.py --quick    # Quick 8 ticker, 6mo scan
  python3 pattern_analyzer_v2.py --ticker NVDA --walk-forward  # With CV
  python3 pattern_analyzer_v2.py --json-only          # JSON output
"""

import argparse, json, math, os, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────
TICKERS = {
    "AI Semis":["NVDA","AMD","AVGO","MRVL"], "AI Infra":["SMCI","DELL","VRT","STRL"],
    "Cloud":["MSFT","GOOGL","AMZN"], "SaaS CRM":["CRM","NOW"],
    "SaaS Data":["SNOW","DDOG","MDB"], "SaaS Security":["PANW","CRWD","NET","ZS"],
    "SaaS Creative":["SHOP","ADBE","INTU"], "Energy":["CEG","VST","OKLO"],
}
WIDER = ["AAPL","META","TSLA","PLTR","SOUN","IONQ","RGTI","QBTS","ANET","ESTC",
         "WDAY","ZS","MRNA","COST","WMT","SPY","QQQ","IWM","XLK","SMH","IBIT"]
ALL_TICKERS = sorted(set(sum(TICKERS.values(),[]) + WIDER))

SECTOR_MAP = {}; [SECTOR_MAP.update({t:s}) for s,v in TICKERS.items() for t in v]

FWD_DAYS = [3,5,10,20]
MIN_SIG = 12
REGIME_WIN = 63

DIR = {
    "rsi_oversold":"long","rsi_oversold_volume":"long","macd_bullish_cross":"long",
    "bb_lower_touch":"long","bb_squeeze":"long","higher_low":"long","higher_high":"long",
    "volume_breakout":"long","golden_cross":"long","rsi_oversold_higher_low":"long",
    "bb_squeeze_volume":"long",
    "rsi_overbought":"short","macd_bearish_cross":"short","bb_upper_touch":"short","death_cross":"short",
}
PTYPE = {
    "rsi_oversold":"mean_reversion","rsi_overbought":"reversal","rsi_oversold_volume":"mean_reversion",
    "macd_bullish_cross":"momentum","macd_bearish_cross":"reversal","bb_lower_touch":"mean_reversion",
    "bb_upper_touch":"reversal","bb_squeeze":"volatility","higher_low":"trend","higher_high":"trend",
    "volume_breakout":"breakout","golden_cross":"trend","death_cross":"reversal",
    "rsi_oversold_higher_low":"combined","bb_squeeze_volume":"combined",
}
MCAP_LG = 200; MCAP_MID = 10
LARGE_CAPS = {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","COST","WMT","ADBE","CRM","NOW","SPY","QQQ","IWM","XLK","SMH","IBIT"}
MID_CAPS = {"PANW","AMD","SNOW","SHOP","MRVL","INTU","DDOG","NET","CRWD","MDB","ANET","WDAY","ZS","DELL","CEG","VST","PLTR"}

def get_vix(p="2y"):
    try:
        v=yf.download("^VIX",period=p,progress=False,auto_adjust=True)
        if not v.empty: return v["Close"].squeeze()
    except: pass
    return None

def classify_regime(date, vixd, w=REGIME_WIN):
    if vixd is None or vixd.empty: return "unknown"
    ei=vixd.index.searchsorted(date)
    si=max(0,ei-w); rv=vixd.iloc[si:ei]
    if rv.empty: return "unknown"
    cv=float(vixd.iloc[min(ei,len(vixd)-1)]) if ei<len(vixd) and ei>0 else float(rv.iloc[-1])
    if cv<14: return "low_vol"
    if cv<20: return "normal"
    if cv<30: return "elevated"
    return "high_vol"

def mcap_cat(t):
    if t in {"SPY","QQQ","IWM","XLK","SMH","IBIT"}: return "etf"
    if t in LARGE_CAPS: return "large"
    if t in MID_CAPS: return "mid"
    return "small"

# ── Pattern Detection ─────────────────────────────────────────────────────
def detect(df):
    c,h,l,v=df["Close"].squeeze(),df["High"].squeeze(),df["Low"].squeeze(),df["Volume"].squeeze()
    sigs=[]
    rsi=ta.rsi(c,14)
    md=ta.macd(c); bb=ta.bbands(c,20,2)
    s20=ta.sma(c,20); s50=ta.sma(c,50); s200=ta.sma(c,200)
    vm20=v.rolling(20).mean()
    def gc(d,*ks):
        if d is None or not isinstance(d,pd.DataFrame): return None
        for col in d.columns:
            for k in ks:
                if k.upper() in col.upper(): return d[col]
        return None
    bl=gc(bb,"BBL","lower"); bu=gc(bb,"BBU","upper"); bm=gc(bb,"BBM","mid")
    ml=gc(md,"MACD_"); ms=gc(md,"SIGNAL","signal")
    for i in range(60,len(df)):
        dt=df.index[i]; p=float(c.iloc[i]); vi=float(v.iloc[i])
        v20=float(vm20.iloc[i]) if not pd.isna(vm20.iloc[i]) else 0
        rv=float(rsi.iloc[i]) if rsi is not None and not pd.isna(rsi.iloc[i]) else None
        # RSI
        if rv is not None:
            if rv<30: sigs.append({"date":dt,"price":p,"pattern":"rsi_oversold","strength":round(30-rv,1),"detail":f"RSI={rv:.1f}"})
            elif rv>70: sigs.append({"date":dt,"price":p,"pattern":"rsi_overbought","strength":round(rv-70,1),"detail":f"RSI={rv:.1f}"})
        # RSI+Vol
        if rv is not None and rv<35 and v20>0 and (vi/v20)>1.5:
            sigs.append({"date":dt,"price":p,"pattern":"rsi_oversold_volume","strength":round((35-rv)*(vi/v20),1),"detail":f"RSI={rv:.1f} V={vi/v20:.1f}x"})
        # MACD
        if ml is not None and ms is not None and i>0:
            _mc=float(ml.iloc[i]);_mp=float(ml.iloc[i-1]);_sc=float(ms.iloc[i]);_sp=float(ms.iloc[i-1])
            if not any(pd.isna(x) for x in [_mc,_mp,_sc,_sp]):
                if _mp<=_sp and _mc>_sc: sigs.append({"date":dt,"price":p,"pattern":"macd_bullish_cross","strength":round(_mc-_sc,2),"detail":f"M={_mc:.2f} S={_sc:.2f}"})
                elif _mp>=_sp and _mc<_sc: sigs.append({"date":dt,"price":p,"pattern":"macd_bearish_cross","strength":round(_sc-_mc,2),"detail":f"M={_mc:.2f} S={_sc:.2f}"})
        # BB touch
        if bl is not None and bu is not None:
            _bl=float(bl.iloc[i]);_bu=float(bu.iloc[i])
            if not any(pd.isna(x) for x in [_bl,_bu]):
                if p<=_bl*1.01: sigs.append({"date":dt,"price":p,"pattern":"bb_lower_touch","strength":round((_bl-p)/_bl*100,2),"detail":f"P={p:.2f} LB={_bl:.2f}"})
                elif p>=_bu*0.99: sigs.append({"date":dt,"price":p,"pattern":"bb_upper_touch","strength":round((p-_bu)/_bu*100,2),"detail":f"P={p:.2f} UB={_bu:.2f}"})
        # BB squeeze
        if bl is not None and bu is not None and bm is not None and i>=10:
            try:
                bw=float(((bu.iloc[i]-bl.iloc[i])/bm.iloc[i]))
                bw10=float(((bu.iloc[i-10]-bl.iloc[i-10])/bm.iloc[i-10]))
                if bw<bw10*0.85 and bw<0.05: sigs.append({"date":dt,"price":p,"pattern":"bb_squeeze","strength":round((1-bw/bw10)*100,1),"detail":f"W={bw:.4f}->{bw10:.4f}"})
            except: pass
        # Higher low/high
        if i>=25:
            low10=float(c.iloc[i-10:i+1].min()); low20=float(c.iloc[i-20:i+1].min())
            high10=float(c.iloc[i-10:i+1].max()); high20=float(c.iloc[i-20:i+1].max())
            if low10>low20: sigs.append({"date":dt,"price":p,"pattern":"higher_low","strength":round((low10-low20)/low20*100,2),"detail":f"L10={low10:.2f} L20={low20:.2f}"})
            if high10>high20: sigs.append({"date":dt,"price":p,"pattern":"higher_high","strength":round((high10-high20)/high20*100,2),"detail":f"H10={high10:.2f} H20={high20:.2f}"})
        # Volume breakout
        if v20>0 and s20 is not None and not pd.isna(s20.iloc[i]):
            vr=vi/v20; pvs=(p/float(s20.iloc[i])-1)*100
            if vr>2.0 and pvs>3: sigs.append({"date":dt,"price":p,"pattern":"volume_breakout","strength":round(vr*pvs,1),"detail":f"V={vr:.1f}x S+{pvs:.1f}%"})
        # SMA cross
        if s50 is not None and s200 is not None and i>0:
            _5c=float(s50.iloc[i]);_5p=float(s50.iloc[i-1]);_2c=float(s200.iloc[i]);_2p=float(s200.iloc[i-1])
            if not any(pd.isna(x) for x in [_5c,_5p,_2c,_2p]):
                if _5p<=_2p and _5c>_2c: sigs.append({"date":dt,"price":p,"pattern":"golden_cross","strength":round((_5c-_2c)/_2c*100,2),"detail":f"50={_5c:.2f} 200={_2c:.2f}"})
                elif _5p>=_2p and _5c<_2c: sigs.append({"date":dt,"price":p,"pattern":"death_cross","strength":round((_2c-_5c)/_2c*100,2),"detail":f"50={_5c:.2f} 200={_2c:.2f}"})
        # RSI + HL
        if rv is not None and rv<35 and i>=25:
            low10=float(c.iloc[i-10:i+1].min()); low20=float(c.iloc[i-20:i+1].min())
            if low10>low20: sigs.append({"date":dt,"price":p,"pattern":"rsi_oversold_higher_low","strength":round((35-rv)*(low10/low20),1),"detail":f"RSI={rv:.1f} HL={low10:.2f}>{low20:.2f}"})
        # BB squeeze + vol
        if bl is not None and bu is not None and bm is not None and v20>0 and i>=10:
            try:
                _bw=float(((bu.iloc[i]-bl.iloc[i])/bm.iloc[i]))
                _bw10=float(((bu.iloc[i-10]-bl.iloc[i-10])/bm.iloc[i-10]))
                if _bw<_bw10*0.8 and (vi/v20)>1.3: sigs.append({"date":dt,"price":p,"pattern":"bb_squeeze_volume","strength":round((1-_bw/_bw10)*(vi/v20),1),"detail":f"SQ={_bw:.3f}->{_bw10:.3f} V={vi/v20:.1f}x"})
            except: pass
    return sigs

def compute_fwd(df, sigs, vixd=None):
    c=df["Close"].squeeze()
    di={d.date():i for i,d in enumerate(df.index)}
    out=[]
    for s in sigs:
        sd=s["date"]
        if isinstance(sd,pd.Timestamp): sd=sd.date()
        elif isinstance(sd,datetime): sd=sd.date()
        ei=di.get(sd)
        if ei is None: continue
        ep=float(c.iloc[ei])
        if ep<=0: continue
        s["entry_price"]=ep;s["entry_index"]=ei
        s["direction"]=DIR.get(s["pattern"],"long")
        s["vix_regime"]=classify_regime(pd.Timestamp(sd),vixd) if vixd is not None else "unknown"
        for f in FWD_DAYS:
            xi=ei+f
            if xi<len(c):
                rt=(float(c.iloc[xi])-ep)/ep*100
                s[f"r_{f}d"]=round(rt,2)
                dir_mul = 1 if DIR.get(s["pattern"],"long")=="long" else -1
                s[f"w_{f}d"]=1 if (rt * dir_mul) > 0 else 0
            else: s[f"r_{f}d"]=None;s[f"w_{f}d"]=None
        out.append(s)
    return out

# ── Walk-Forward ──────────────────────────────────────────────────────────
def walk_fwd(sigs, ns=4):
    if len(sigs)<50: return {"error":"too few","folds":[]}
    ss=sorted(sigs,key=lambda x:x["date"])
    fs=len(ss)//(ns+1); folds=[]
    for f in range(ns):
        te=(f+1)*fs; ts=te; tge=min(te+fs,len(ss))
        if tge-ts<MIN_SIG: break
        train=ss[:te]; test=ss[ts:tge]
        ts2=defaultdict(lambda:{"n":0,"w":0,"r":0.0})
        for s in train:
            p=s["pattern"]; ts2[p]["n"]+=1
            r5=s.get("r_5d")
            if r5 is not None: ts2[p]["w"]+=1 if r5>0 else 0; ts2[p]["r"]+=r5
        rkd=[]
        for p,st in ts2.items():
            if st["n"]>=MIN_SIG: rkd.append((p,st["w"]/st["n"]*100,st["r"]/st["n"],st["n"]))
        rkd.sort(key=lambda x:x[1],reverse=True)
        tm={"total":0,"wins":0,"sr":0.0}
        for p,_,_,_ in rkd[:3]:
            for s in test:
                if s["pattern"]==p:
                    tm["total"]+=1; r5=s.get("r_5d")
                    if r5 is not None: tm["wins"]+=1 if r5>0 else 0; tm["sr"]+=r5
        folds.append({"fold":f+1,"train":f"{train[0]['date']}->{train[-1]['date']}","test":f"{test[0]['date']}->{test[-1]['date']}","train_n":len(train),"test_n":len(test),"top":[(p,f"{wr:.1f}%") for p,wr,_,_ in rkd[:5]],"test_wr":round(tm["wins"]/tm["total"]*100,1) if tm["total"]>0 else 0,"test_ar":round(tm["sr"]/tm["total"],2) if tm["total"]>0 else 0,"test_traded":tm["total"]})
    tt=sum(f.get("test_traded",0) for f in folds)
    tw=sum(f["test_wr"]*f.get("test_traded",0)/100 for f in folds)
    return {"overall":{"n_folds":len(folds),"total_test":tt,"avg_wr":round(tw/tt*100,1) if tt>0 else 0,"avg_ret":round(sum(f["test_ar"]*f.get("test_traded",0) for f in folds)/tt,2) if tt>0 else 0},"folds":folds}

# ── Analysis Engine ───────────────────────────────────────────────────────
def analyze(tickers, p="2y", wf=False, jonly=False):
    ps=defaultdict(lambda:{"n":0,"w5":0,"r3":0.0,"r5":0.0,"r10":0.0,"r20":0.0,"r5sq":0.0,"b5":-999.0,"w5":999.0,"bt":defaultdict(lambda:{"n":0,"w":0,"r":0.0}),"bs":defaultdict(lambda:{"n":0,"w":0,"r":0.0}),"br":defaultdict(lambda:{"n":0,"w":0,"r":0.0}),"bm":defaultdict(lambda:{"n":0,"w":0,"r":0.0}),"by_t":defaultdict(int)})
    all_sigs=[]; tp=0; ts=0
    vd=get_vix(p)
    if not jonly: print(f"\n  📡 {len(tickers)} tkrs ({p})...\n  {'─'*55}")
    for i,t in enumerate(tickers):
        if not jonly: print(f"  [{i+1}/{len(tickers)}] {t:<6s}",end="",flush=True)
        try:
            df=yf.download(t,period=p,progress=False,auto_adjust=True)
            if df.empty or len(df)<100:
                if not jonly: print(f"  ⏭️ short"); continue
            sigs=detect(df)
            if not sigs:
                if not jonly: print(f"  ⏭️ none"); continue
            for s in sigs: s["ticker"]=t;s["sector"]=SECTOR_MAP.get(t,"Other");s["mcap"]=mcap_cat(t)
            res=compute_fwd(df,sigs,vd)
            all_sigs.extend(res);ts+=len(res);tp+=1
            for r in res:
                pn=r["pattern"];st=ps[pn];st["n"]+=1
                st["by_t"][PTYPE.get(pn,"?")]+=1
                bt=st["bt"][t];bt["n"]+=1
                bs=st["bs"][SECTOR_MAP.get(t,"Other")];bs["n"]+=1
                br=st["br"][r.get("vix_regime","?")];br["n"]+=1
                bm=st["bm"][r.get("mcap","?")];bm["n"]+=1
                for d in FWD_DAYS:
                    ret=r.get(f"r_{d}d")
                    if ret is not None:
                        st[f"r{d}"]+=ret
                        if d==5:
                            st["r5sq"]+=ret**2
                            if ret>0:
                                st["w5"]+=1;bt["w"]+=1;bs["w"]+=1;br["w"]+=1;bm["w"]+=1
                            bt["r"]+=ret;bs["r"]+=ret;br["r"]+=ret;bm["r"]+=ret
                            st["b5"]=max(st["b5"],ret);st["w5"]=min(st["w5"],ret)
            if not jonly: print(f"  ✅ {len(res):3d} sig")
        except Exception as e:
            if not jonly: print(f"  ❌ {str(e)[:50]}")
    final={}
    for pn,s in sorted(ps.items(),key=lambda x:x[1]["n"],reverse=True):
        if s["n"]<MIN_SIG: continue
        wr5=s["w5"]/s["n"]*100; av5=s["r5"]/s["n"]; av3=s["r3"]/s["n"]; av10=s["r10"]/s["n"]; av20=s["r20"]/s["n"]
        var=s["r5sq"]/s["n"]-av5**2; std=math.sqrt(var) if var>0 else 0.001; sh=av5/std if std>0 else 0
        bt=max(s["bt"].items(),key=lambda x:x[1]["r"]/x[1]["n"] if x[1]["n"]>0 else -999)
        bs=max(s["bs"].items(),key=lambda x:x[1]["r"]/x[1]["n"] if x[1]["n"]>0 else -999)
        br=max(s["br"].items(),key=lambda x:x[1]["r"]/x[1]["n"] if x[1]["n"]>0 else -999)
        bm=max(s["bm"].items(),key=lambda x:x[1]["r"]/x[1]["n"] if x[1]["n"]>0 else -999)
        final[pn]={"n":s["n"],"dir":DIR.get(pn,"long"),"type":list(s["by_t"].keys())[0] if s["by_t"] else "?",
            "wr5":round(wr5,1),"ar3":round(av3,2),"ar5":round(av5,2),"ar10":round(av10,2),"ar20":round(av20,2),
            "sh5":round(sh,3),"b5":round(s["b5"],2),"w5":round(s["w5"],2),
            "best_t":bt[0],"best_tr":round(bt[1]["r"]/bt[1]["n"],2) if bt[1]["n"]>0 else 0,"best_tn":bt[1]["n"],
            "best_s":bs[0],"best_sr":round(bs[1]["r"]/bs[1]["n"],2) if bs[1]["n"]>0 else 0,
            "best_reg":br[0],"best_reg_r":round(br[1]["r"]/br[1]["n"],2) if br[1]["n"]>0 else 0,
            "best_mc":bm[0],"best_mcr":round(bm[1]["r"]/bm[1]["n"],2) if bm[1]["n"]>0 else 0}
    wf_res=walk_fwd(all_sigs) if wf and len(all_sigs)>100 else None
    return {"stats":final,"total_n":ts,"total_tp":tp,"scanned":len(tickers),"wf":wf_res}

# ── Report ────────────────────────────────────────────────────────────────
def report(r):
    s=r["stats"]; bw=sorted(s.items(),key=lambda x:x[1]["wr5"],reverse=True)
    bsh=sorted(s.items(),key=lambda x:x[1]["sh5"],reverse=True); br=sorted(s.items(),key=lambda x:x[1]["ar5"],reverse=True)
    print(f'\n╔══════════════════════════════════════════════════════════════════════╗\n║  📊 PATTERN ANALYZER v2 — {r["total_n"]} sig / {r["total_tp"]} tkrs ║\n╚══════════════════════════════════════════════════════════════════════╝\n')
    print("  🏆 TOP 12 BY 5-DAY WIN RATE")
    print("  ─"+"─"*85)
    print(f"  {'Rank':<5} {'Pattern':<28} {'Dir':<6} {'Win%':<7} {'R+5d':<8} {'R+10d':<8} {'Shrp':<7} {'N':<5} {'Best':<8}")
    print("  ─"+"─"*85)
    for i,(n,p) in enumerate(bw[:12],1):
        bar="█"*max(1,int(p["wr5"]/5))
        d={"long":"📗","short":"📕"}.get(p["dir"],"⚪")
        print(f"  {i:<5} {n:<28} {d:<6} {p['wr5']:<6.1f}% {p['ar5']:<7.2f}% {p['ar10']:<7.2f}% {p['sh5']:<7.3f} {p['n']:<5} {p['best_t']:<8} {bar}")
    print(f"\n  💀 WORST (AVOID)")
    print(f"  {'─'*50}")
    for n,p in reversed(bw[-5:]):
        if p["wr5"]>=50: break
        print(f"  ❌ {n:<28s} {p['wr5']:5.1f}% wr ({p['n']} sig, {p['ar5']:+.2f}% avg)")
    print(f"\n  📈 TOP BY RISK-ADJ (Sharpe)")
    print(f"  {'─'*55}")
    for n,p in bsh[:8]:
        ic="📗" if p["sh5"]>0.1 else "📕"
        print(f"  {ic} {n:<28s} S={p['sh5']:.3f}  W={p['wr5']:.1f}%  R5={p['ar5']:+.2f}%  R20={p['ar20']:+.2f}%")
    print(f"\n  🏷️  PATTERN TYPES")
    print(f"  {'─'*50}")
    cats=defaultdict(lambda:{"n":0,"w":0.0,"r":0.0})
    for n,p in s.items():
        t=p.get("type","?"); cats[t]["n"]+=p["n"]; cats[t]["w"]+=p["n"]*p["wr5"]/100; cats[t]["r"]+=p["n"]*p["ar5"]
    for t in sorted(cats.keys()):
        wr=cats[t]["w"]/cats[t]["n"]*100 if cats[t]["n"]>0 else 0
        ar=cats[t]["r"]/cats[t]["n"] if cats[t]["n"]>0 else 0
        print(f"  {t:<22s} {cats[t]['n']:5d} sig | Win {wr:5.1f}% | Avg {ar:+.2f}%")
    # Long vs Short
    print(f"\n  📐 LONG vs SHORT")
    print(f"  {'─'*40}")
    for d_ in ["long","short"]:
        d_sigs=sum(p["n"] for n2,p in s.items() if DIR.get(n2,"long")==d_)
        d_wins=sum(p["n"]*p["wr5"]/100 for n2,p in s.items() if DIR.get(n2,"long")==d_)
        d_ret=sum(p["n"]*p["ar5"] for n2,p in s.items() if DIR.get(n2,"long")==d_)
        if d_sigs>0:
            print(f"  {'📈' if d_=='long' else '📉'} {d_:<8s} {d_sigs:5d} sig | Win {d_wins/d_sigs*100:5.1f}% | Avg {d_ret/d_sigs:+.2f}%")
    # VIX Regimes
    print(f"\n  🌩️  BEST PER VIX REGIME")
    print(f"  {'─'*50}")
    regs_seen=set()
    for n,p in bw:
        reg=p.get("best_reg","?")
        if reg not in regs_seen:
            regs_seen.add(reg); print(f"  {reg:<12s} {n:<28s} Win {p['wr5']:5.1f}% Ret {p['ar5']:+.2f}%")
    # MCap
    print(f"\n  📦 BEST PER MCAP")
    print(f"  {'─'*50}")
    mc_seen=set()
    for n,p in bw:
        mc=p.get("best_mc","?")
        if mc not in mc_seen:
            mc_seen.add(mc); print(f"  {mc:<10s} {n:<28s} Win {p['wr5']:5.1f}% Ret {p['ar5']:+.2f}%")
    # Sector
    print(f"\n  🏢 BEST PER SECTOR")
    print(f"  {'─'*50}")
    sec_seen=set()
    for n,p in bw:
        sc=p.get("best_s","?")
        if sc not in sec_seen:
            sec_seen.add(sc); print(f"  {sc:<20s} {n:<28s} Win {p['wr5']:5.1f}% Ret {p['ar5']:+.2f}%")
    # Walk-forward
    wf=r.get("wf")
    if wf and wf.get("overall"):
        ov=wf["overall"]
        print(f"\n  🔬 WALK-FORWARD VALIDATION")
        print(f"  {'─'*55}")
        print(f"  Folds: {ov['n_folds']}  |  Total test sigs: {ov['total_test']}")
        print(f"  Avg test win rate: {ov['avg_wr']:.1f}%  |  Avg test return: {ov['avg_ret']:+.2f}%")
        for f in wf.get("folds",[])[:3]:
            print(f"  Fold {f['fold']}: train={f['train']} → test={f['test']}: {f['test_wr']:.1f}% wr ({f['test_traded']} sig)")
    print(f"\n  💡 QUICK RULES")
    print(f"  {'─'*40}")
    for i,(n,p) in enumerate(bw[:3],1):
        d_="BUY" if p["ar5"]>0 else "SELL"
        print(f"  {i}. {d_}: {n} → win {p['wr5']:.0f}%, avg {p['ar5']:+.2f}% in 5d")
    print()

def save(r):
    od=Path(__file__).parent/"reports";od.mkdir(exist_ok=True)
    jp=od/"pattern_v2.json"
    with open(jp,"w") as f: json.dump(json.loads(json.dumps(r,default=str)),f,indent=2)
    print(f"  💾 JSON: {jp}")
    s=r.get("stats",{}); bw=sorted(s.items(),key=lambda x:x[1]["wr5"],reverse=True)
    md=f"# Pattern Performance v2\n**{datetime.now().strftime('%Y-%m-%d %H:%M')}** | {r.get('scanned',0)} tkrs | {r.get('total_n',0)} signals | {r.get('total_tp',0)} w/ data\n\n## Top by 5-Day Win Rate\n\n|Rank|Pattern|Dir|Win%|R+5d|R+10d|Sharpe|N|Best On|\n|-|-|-|-|-|-|-|-|-|\n"
    for i,(n,p) in enumerate(bw[:15],1):
        d={"long":"📗","short":"📕"}.get(p.get("dir","long"),"⚪")
        md+=f"|{i}|{n}|{d}|{p['wr5']:.1f}%|{p['ar5']:+.2f}%|{p['ar10']:+.2f}%|{p['sh5']:.3f}|{p['n']}|{p['best_t']}|\n"
    md+="\n## WORST\n\n|Pattern|Win%|N|\n|-|-|-|\n"
    for n,p in reversed(bw[-6:]):
        if p["wr5"]>=50: continue
        md+=f"|{n}|{p['wr5']:.1f}%|{p['n']}|\n"
    cats=defaultdict(lambda:{"n":0,"w":0.0,"r":0.0})
    for n,p in s.items():
        t=p.get("type","?"); cats[t]["n"]+=p["n"]; cats[t]["w"]+=p["n"]*p["wr5"]/100; cats[t]["r"]+=p["n"]*p["ar5"]
    md+="\n## Categories\n\n|Category|Signals|Win%|AvgRet|\n|-|-|-|-|\n"
    for t in sorted(cats.keys()):
        wr=cats[t]["w"]/cats[t]["n"]*100 if cats[t]["n"]>0 else 0
        ar=cats[t]["r"]/cats[t]["n"] if cats[t]["n"]>0 else 0
        md+=f"|{t}|{cats[t]['n']}|{wr:.1f}%|{ar:+.2f}%|\n"
    
    md+="\n---\n*Pattern Analyzer v2 — Rocky*"
    mp=od/"pattern_v2_report.md"
    mp.write_text(md)
    print(f"  💾 MD: {mp}")


def main():
    parser=argparse.ArgumentParser(description="Pattern Analyzer v2")
    parser.add_argument("--full",action="store_true",help="All tickers, 2yr")
    parser.add_argument("--quick",action="store_true",help="Quick scan")
    parser.add_argument("--ticker","-t",help="Single ticker")
    parser.add_argument("--period","-p",default="2y",help="Data period")
    parser.add_argument("--walk-forward","-wf",action="store_true",help="Walk-forward CV")
    parser.add_argument("--json-only",action="store_true",help="JSON output only")
    args=parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║  📊 Pattern Performance Analyzer v2                        ║
║  Multi-horizon backtest with regime & walk-forward         ║
╚════════════════════════════════════════════════════════════╝
""")
    if args.ticker: t0=[args.ticker.upper()];pr=args.period
    elif args.quick: t0=["SPY","QQQ","NVDA","MSFT","AAPL","AMZN","TSLA","GOOGL"];pr="6mo"
    elif args.full: t0=ALL_TICKERS;pr=args.period or "2y"
    else: t0=sum(TICKERS.values(),[])+["SPY","QQQ"];pr="1y"
    
    t1=time.time()
    res=analyze(t0,pr,args.walk_forward,args.json_only)
    et=time.time()-t1
    
    if not args.json_only: report(res)
    save(res)
    print(f"  ⏱️  {et:.0f}s\n")

if __name__=="__main__":
    main()