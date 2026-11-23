"""Independent re-implementation of the fill engine + ORB retest, compared to Codex output."""
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import nq_backtest as bt, session_timeframe_comparison as c, strategy_research_suite as s
NY="America/New_York"; TICK=.25; PV=2.0; SLIP=.25; COM=2.5

def my_engine(x, sig, atr, rr, flat=True, dd_limit=1000.0):
    o,h,l,cl = (x[k].to_numpy() for k in ("open","high","low","close"))
    t = x.index.tz_convert(NY); sig=sig.to_numpy(); atr=atr.to_numpy()
    out=[]; eq=pk=0.0; i=0; n=len(x)
    while i < n-1:
        if pk-eq >= dd_limit: break
        d=int(sig[i])
        if d==0 or not np.isfinite(atr[i]) or atr[i]<=0: i+=1; continue
        e=i+1; risk=max(atr[i],TICK)
        if (risk+2*SLIP)*PV+2*COM > dd_limit-(pk-eq): i+=1; continue
        ent=o[e]+d*SLIP; stp=ent-d*risk; tgt=ent+d*risk*rr
        cutoff = pd.Timestamp(f"{t[e].date()} 16:00", tz=NY)
        xi, px, why = n-1, cl[-1], "eod"
        for j in range(e,n):
            if flat and t[j] >= cutoff: xi,px,why=j,o[j],"session_end"; break
            if (l[j]<=stp) if d>0 else (h[j]>=stp):
                xi,px,why=j,(min(stp,o[j]) if d>0 else max(stp,o[j])),"stop"; break
            if (h[j]>=tgt) if d>0 else (l[j]<=tgt): xi,px,why=j,tgt,"target"; break
            if j<n-1 and sig[j]==-d: xi,px,why=j+1,o[j+1],"signal"; break
        px-=d*SLIP; pnl=(px-ent)*d*PV-2*COM
        out.append((x.index[e],x.index[xi],d,round(ent,4),round(px,4),why,round(pnl,4)))
        eq+=pnl; pk=max(pk,eq); i=xi+1
    return pd.DataFrame(out,columns=["entry_time","exit_time","direction","entry","exit","reason","pnl"])

def my_orb_retest(x5, rng_min=15, wait=30):
    """Range 9:30..9:30+rng; first close outside arms it; later bar wicks back to edge and closes outside -> signal."""
    sig=pd.Series(0,index=x5.index,dtype="int8"); t=x5.index.tz_convert(NY)
    for day,g in x5.groupby(t.date):
        gt=g.index.tz_convert(NY); m=gt.hour*60+gt.minute
        rng=g[(m>=570)&(m<570+rng_min)]
        if rng.empty: continue
        hi,lo=rng.high.max(),rng.low.min()
        cand=g[(m>=570+rng_min)&(m+5<960)]
        arm=0; armk=-1
        for k,(ts,r) in enumerate(cand.iterrows()):
            side = 1 if r.close>hi else -1 if r.close<lo else 0
            if arm==0:
                if side: arm,armk=side,k
                continue
            if (arm>0 and r.low<=hi and r.close>hi) or (arm<0 and r.high>=lo and r.close<lo):
                sig[ts]=arm; break
            if side and side!=arm: arm=0
            elif k-armk>wait//5: arm=0
    return sig

raw = bt.load_bars(c.MNQ_PATH, c.localize_session_config("new_york"))
x5 = bt.indicators(raw)
# 1) signal logic: independent ORB retest vs Codex ORB retest
cod,_ = c.orb_signal(raw, x5, "new_york", 5, 15, "retest")
mine = my_orb_retest(x5)
print("ORB15 retest signals  codex:", int((cod!=0).sum()), " mine:", int((mine!=0).sum()), " mismatched bars:", int((cod!=mine).sum()))
# 2) engine: run my engine on Codex's signals for every Oscar config, compare trade lists
cfg0 = c.localize_session_config("new_york"); worst=0; ncfg=0
for orb in (5,15):
  for var in ("breakout","two_close_fakeout_filter","retest"):
    for tf in (5,10,15):
      x = bt.indicators(c.resample_ohlcv(raw,tf)); sg,_ = c.orb_signal(raw,x,"new_york",tf,orb,var)
      for rr in (1.0,1.5,2.0,2.5,3.0):
        a = bt.backtest(x,sg,bt.Config(**{**cfg0.__dict__,"rr":rr}),None,None,True,NY,"16:00")
        b = my_engine(x,sg,x.atr,rr)
        ncfg+=1
        same = len(a)==len(b) and np.allclose(a.pnl.values,b.pnl.values) and (a.reason.values==b.reason.values).all()
        if not same:
            print("MISMATCH", orb,var,tf,rr,len(a),len(b)); worst+=1
print(f"engine cross-check: {ncfg-worst}/{ncfg} Oscar configs identical trade-by-trade")
# 3) November AVWAP + Lima in suite (CT index) on my engine
cfg_ct = bt.Config(session="new_york")
rawct = bt.load_bars(s.MNQ_PATH, cfg_ct)
for tf in (5,10):
    d = bt.indicators(s.resample_ohlcv(rawct,tf)); sg=s.make_avwap_signal(d,2.0)
    for rr in (1.0,2.0,3.0):
        a=bt.backtest(d,sg,bt.Config(**{**cfg_ct.__dict__,"rr":rr}),None,None,True,NY,"16:00"); b=my_engine(d,sg,d.atr,rr)
        print(f"November {tf}m RR{rr}: codex ${a.pnl.sum():.2f} ({len(a)}) vs mine ${b.pnl.sum():.2f} ({len(b)})")
