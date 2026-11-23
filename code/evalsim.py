import pandas as pd, numpy as np
TARGET, MLL, DLL, MAXDAYS, PATHS = 1250.0, 1000.0, 600.0, 60, 4000
df=pd.read_pickle("trades.pkl")
df["tday"]=(pd.to_datetime(df.exit_time,utc=True).dt.tz_convert("America/New_York")+pd.Timedelta(hours=7)).dt.date  # CME day rolls at 17:00 ET
df["xt"]=pd.to_datetime(df.exit_time,utc=True)
ALLDAYS=pd.bdate_range("2026-08-21","2026-10-30").date
def day_lists(t):
    t=t.sort_values("xt"); g={d:x.pnl.to_numpy() for d,x in t.groupby("tday")}
    return [g.get(d,np.array([])) for d in ALLDAYS]
def sim(days,rng):
    passed=failed=0; dd_days=[]; dll_hits=0; tot=0
    for _ in range(PATHS):
        bal=peak_eod=0.0; out=None
        for k in range(MAXDAYS):
            day=days[rng.integers(len(days))]; dpl=0.0
            for p in day:
                bal+=p; dpl+=p
                if bal<=peak_eod-MLL: out="fail"; break
                if bal>=TARGET: out="pass"; break
                if dpl<=-DLL: dll_hits+=1; break
            if out: break
            peak_eod=max(peak_eod,bal)
        passed+=out=="pass"; failed+=out=="fail"
        if out=="pass": dd_days.append(k+1)
    return passed/PATHS, failed/PATHS, np.mean(dd_days) if dd_days else np.nan
rng=np.random.default_rng(1); rows=[]
for (tag,rr),t in df.groupby(["tag","rr"]):
    days=day_lists(t); p,f,dd=sim(days,rng)
    rows.append(dict(setup=tag,rr=rr,chosen=bool(t.chosen_rr.iloc[0]),trades=len(t),per_day=len(t)/len(ALLDAYS),win=(t.pnl>0).mean(),expR=t.R.mean(),net=t.pnl.sum(),
                     pass_p=p,fail_p=f,days_to_pass=dd))
R=pd.DataFrame(rows); R.to_csv("evalsim_setups.csv",index=False)
best=R.loc[R.groupby("setup").pass_p.idxmax()]
ch=R[R.chosen]
m=ch.merge(best[["setup","rr","pass_p","win"]],on="setup",suffixes=("","_best"))
print(m[["setup","per_day","rr","win","expR","pass_p","fail_p","days_to_pass","rr_best","win_best","pass_p_best"]].round(2).sort_values("pass_p",ascending=False).to_string(index=False))
from scipy.stats import spearmanr
print("\nacross all setup x RR configs: spearman(pass_p, win rate)=%.2f  spearman(pass_p, expected R)=%.2f"%(spearmanr(R.pass_p,R.win).statistic, spearmanr(R.pass_p,R.expR).statistic))
for name,tags in (("NEW 8 (both charts)",["GOLF","HOTEL","ALPHA","KILO","MIKE","TANGO","SIERRA","VICTOR"]),
                  ("OLD 10 (MNQ NY)",["LIMA 5m","MARCUS 15m","MARCUS 30m","NOVEMBER 5m","NOVEMBER 10m","NOVEMBER 15m","OSCAR BREAK","OSCAR FAKEOUT","OSCAR RETEST","PAPA"]),
                  ("ALL 18",list(R.setup.unique())),
                  ("TOP 5 by pass rate",list(ch.sort_values("pass_p",ascending=False).setup[:5]))):
    t=df[df.chosen_rr & df.tag.isin(tags)]
    p,f,dd=sim(day_lists(t),rng)
    worst=t.groupby("tday").pnl.sum()
    print(f"{name}: {len(t)/len(ALLDAYS):.1f} trades/day | win {(t.pnl>0).mean():.0%} | pass {p:.0%} fail {f:.0%} | avg days to pass {dd:.1f} | days below -$600: {(worst<=-600).sum()} of {len(ALLDAYS)} | max losing streak {t.sort_values('xt').pnl.lt(0).astype(int).groupby(t.sort_values('xt').pnl.ge(0).cumsum()).sum().max()}")
