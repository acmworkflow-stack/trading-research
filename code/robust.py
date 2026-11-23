import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import functools
import nq_backtest as bt, session_timeframe_comparison as c, strategy_research_suite as s
NY="America/New_York"
def fix(raw):
    r=raw.copy(); ts=pd.Timestamp("2026-10-30 15:00",tz=NY).tz_convert(r.index.tz)
    if ts in r.index:
        prev=r.close[r.index<ts].iloc[-1]; r.loc[ts,"open"]=prev
        r.loc[ts,"high"]=max(prev,r.loc[ts,"close"])+5; r.loc[ts,"low"]=min(r.loc[ts,"low"],prev)
    return r
rawny=bt.load_bars(c.MNQ_PATH,c.localize_session_config("new_york"))
cfgct=bt.Config(session="new_york"); rawct=bt.load_bars(s.MNQ_PATH,cfgct)
_cache={}
def memo(key,f):
    if key not in _cache: _cache[key]=f()
    return _cache[key]

def oscar(orb,var,tf,rr):
    def run(raw,flip=None,**kw):
        x,sg=memo(("o",orb,var,tf,id(raw)),lambda:(lambda x:(x,c.orb_signal(raw,x,"new_york",tf,orb,var)[0]))(bt.indicators(c.resample_ohlcv(raw,tf))))
        if flip is not None: sg=sg*flip(sg)
        cfg=bt.Config(**{**c.localize_session_config("new_york").__dict__,"rr":rr,**kw})
        return bt.backtest(x,sg.astype("int8"),cfg,None,None,True,NY,"16:00")
    return run, rawny
def november(tf,sig,rr):
    def run(raw,flip=None,**kw):
        d,sg=memo(("n",tf,sig,id(raw)),lambda:(lambda d:(d,s.make_avwap_signal(d,sig)))(bt.indicators(s.resample_ohlcv(raw,tf))))
        if flip is not None: sg=sg*flip(sg)
        return bt.backtest(d,sg.astype("int8"),bt.Config(**{**cfgct.__dict__,"rr":rr,**kw}),None,None,True,NY,"16:00")
    return run, rawct
def papa(pts,rr):
    def run(raw,flip=None,**kw):
        d,(sg,rp)=memo(("p",pts,rr,id(raw)),lambda:(lambda d:(d,s.make_papa_signal(d,pts,rr)))(bt.indicators(raw)))
        if flip is not None: sg=sg*flip(sg)
        return s.backtest_papa(d,sg.astype("int8"),rp,pts,bt.Config(**{**cfgct.__dict__,"rr":rr,**kw}))
    return run, rawct
def lima(rr):
    def run(raw,flip=None,**kw):
        x,sg=memo(("l",id(raw)),lambda:(lambda x:(x,bt.make_signals(x,"mean_reversion_bb_rsi",cfgct)))(bt.indicators(raw)))
        if flip is not None: sg=sg*flip(sg)
        return bt.backtest(x,sg.astype("int8"),bt.Config(**{**cfgct.__dict__,"rr":rr,**kw}))
    return run, rawct
def ob(m,rr):
    def run(raw,flip=None,**kw):
        def mk():
            x=bt.indicators(raw); sg,st,atr,_=bt.make_fvg_orderblock_signals(x,m); v=bt.session_vwap(x,cfgct)
            return x,sg.where(bt.in_session(x.index,cfgct)&(((sg>0)&(x.close>v))|((sg<0)&(x.close<v))),0),st,atr
        x,sg,st,atr=memo(("ob",m,id(raw)),mk)
        if flip is not None: sg=sg*flip(sg)
        return bt.backtest(x,sg.astype("int8"),bt.Config(**{**cfgct.__dict__,"rr":rr,**kw}),atr,st)
    return run, rawct

cands={
 "Oscar 15m ORB retest | 5m bars | 3R":oscar(15,"retest",5,3.0),
 "Oscar 15m ORB breakout | 10m | 3R":oscar(15,"breakout",10,3.0),
 "Oscar 5m ORB 2-close | 10m | 1.5R":oscar(5,"two_close_fakeout_filter",10,1.5),
 "Oscar 15m ORB 2-close | 5m | 2R":oscar(15,"two_close_fakeout_filter",5,2.0),
 "November AVWAP 2σ | 5m | 2R":november(5,2.0,2.0),
 "Papa 30pt | 1R":papa(30.0,1.0),
 "Lima BB+RSI NY | 2R":lima(2.0),
 "15m FVG+OB+VWAP | 3.5R":ob(15,3.5),
}
NOKILL=dict(drawdown_dollars=1e9)
rows=[]; rng=np.random.default_rng(42)
split=pd.Timestamp("2026-09-26",tz=NY)
for name,(run,raw) in cands.items():
    fr=fix(raw); _keep=fr
    base=run(raw); fx=run(fr); s2=run(fr,slippage_ticks=2.0); nk=run(fr,**NOKILL)
    et=pd.to_datetime(nk.entry_time); h1=nk.pnl[et<split].sum(); h2=nk.pnl[et>=split].sum()
    n1=(et<split).sum(); n2=(et>=split).sum()
    obs=nk.pnl.sum(); perm=[]
    for _ in range(200):
        flip=lambda sg: pd.Series(rng.choice([-1,1],len(sg)),index=sg.index)
        perm.append(run(fr,flip=flip,**NOKILL).pnl.sum())
    perm=np.array(perm); p=(np.sum(perm>=obs)+1)/(len(perm)+1)
    eq=nk.pnl.cumsum(); dd=(eq.cummax().clip(lower=0)-eq).max()
    rows.append(dict(strategy=name,reported=round(base.pnl.sum(),0),bad_bar_fixed=round(fx.pnl.sum(),0),
        slip_2tick=round(s2.pnl.sum(),0),no_dd_stop=round(obs,0),trades=len(nk),max_dd_no_stop=round(dd,0),
        first_half=f"{h1:.0f} ({n1})",second_half=f"{h2:.0f} ({n2})",random_dir_p=round(p,3),
        rand_median=round(np.median(perm),0)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(sys.argv[2],index=False)
