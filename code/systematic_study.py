"""Three documented effects, tested with rules fixed before any results were seen.

    python code/systematic_study.py

Writes nothing except one CSV per Yahoo symbol in data/cache/daily/ (downloaded once with
yfinance, then reused; delete a file to refresh it). Results print to the console and
code/app.py shows the same tables, with equity curves, in its "Systematic" tab.

Test 1  Overnight vs intraday: SPY and QQQ since inception (dividend-adjusted opens and closes),
        NQ from the Databento series (3 years, cash session: 09:30 open, last bar before 16:00 as
        the close). Overnight = prior close -> open, intraday = open -> close, vs buy-and-hold.
        Costs: ETFs 1 bp per side (2 bp per day for either strategy); NQ 1 tick + $2.50 per side
        at micro size ($2/pt), charged against the contract's real (unadjusted) notional.
        Pass (per instrument): overnight beats intraday on annual return after costs, and
        overnight's annual return after costs is > 0 in both halves of the history.
        GC on Databento is run the same way as a cross-check only.
Test 2  Time-series momentum on 24 futures (Yahoo continuous, daily). At each month's last close:
        sign of the past 3 / 6 / 12-month return (month-end to month-end) x 10% / (60-day
        volatility, annualised); traded at the next day's open, 2 bp per side on the change in
        position; markets averaged equally among those with a position. Blend = equal average of
        the three lookbacks' daily returns. Run as-is and with flagged roll days' returns set to
        zero. Deflated Sharpe over the 4 trials per data version, as validate.deflated_sharpe.
        Pass (per data version): blend Sharpe > 0.3 in both halves (split at the midpoint date),
        deflated Sharpe >= 0.95, and more than half of the markets contribute positively.
        NQ and GC alone are re-run on Databento daily bars vs Yahoo as a cross-check.
Test 3  VIX regime on every setup trade from setup_report.py (study.SETUPS + ECHO with the 1m
        profile; 3 years Databento): high = the prior day's VIX close above its trailing 252-day
        median (median up to and including that close), low otherwise.
        Pass: one regime has pooled average R > 0 in both 2024 and 2025.

Roll flags (Yahoo futures are not back-adjusted): a day is flagged when |log(open / prior close)|
exceeds 4 x the standard deviation of daily log close-to-close returns over the 60 days ending the
day before, and the date falls in one of the market's contract months or the month before one.
Every decision uses data available at that time: volatility and the VIX median end at the
decision close, signals use month-end closes, trades fill at the next open.
"""
from __future__ import annotations
import sys, time
from math import sqrt
from pathlib import Path
import numpy as np, pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
DAILY = ROOT / "data" / "cache" / "daily"
NY = "America/New_York"

Q = [3, 6, 9, 12]
FUTURES = {  # Yahoo symbol: contract months
    "ES=F": Q, "NQ=F": Q, "YM=F": Q, "RTY=F": Q,
    "CL=F": list(range(1, 13)), "NG=F": list(range(1, 13)),
    "GC=F": [2, 4, 6, 8, 10, 12], "SI=F": [3, 5, 7, 9, 12], "HG=F": [3, 5, 7, 9, 12],
    "ZB=F": Q, "ZN=F": Q, "ZF=F": Q,
    "ZC=F": [3, 5, 7, 9, 12], "ZW=F": [3, 5, 7, 9, 12], "ZS=F": [1, 3, 5, 7, 8, 9, 11],
    "6E=F": Q, "6J=F": Q, "6B=F": Q, "6A=F": Q, "6C=F": Q,
    "KC=F": [3, 5, 7, 9, 12], "SB=F": [3, 5, 7, 10],
    "LE=F": [2, 4, 6, 8, 10, 12], "HE=F": [2, 4, 5, 6, 7, 8, 10, 12],
}
ETFS = ["SPY", "QQQ"]
VIX = "^VIX"
ETF_COST, TS_COST = 1e-4, 2e-4
VOL_TARGET, VOL_WIN, LOOKBACKS, GAP_K = 0.10, 60, (3, 6, 12), 4.0
MICRO = {"NQ": dict(tick=0.25, pv=2.0), "GC": dict(tick=0.10, pv=10.0)}
COMM = 2.50


# ----------------------------------------------------------------------------------------- data
def _path(t: str) -> Path:
    return DAILY / (t.replace("^", "").replace("=", "_") + ".csv")


def daily(ticker: str, log=print) -> pd.DataFrame:
    """Daily OHLC from the cache, downloading it once (bars dated today or later are left out,
    since they may still be forming)."""
    p = _path(ticker)
    if not p.exists():
        import yfinance as yf
        d = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False,
                        multi_level_index=False)
        if d is None or d.empty:
            raise RuntimeError(f"Yahoo returned no data for {ticker}")
        d = d.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d.index.name = "date"
        d = d[d.index < pd.Timestamp.now(tz=NY).tz_localize(None).normalize()]
        DAILY.mkdir(parents=True, exist_ok=True)
        d.to_csv(p)
        log(f"  downloaded {ticker}: {len(d)} days from {d.index[0]:%Y-%m-%d}")
    return pd.read_csv(p, index_col="date", parse_dates=True)


def clean_ohlc(d: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    d = d[~d.index.duplicated()].sort_index()
    ok = d[["open", "high", "low", "close"]].notna().all(axis=1) & (d[["open", "high", "low", "close"]] > 0).all(axis=1)
    return d[ok], int((~ok).sum())


def roll_flags(d: pd.DataFrame, months) -> pd.Series:
    r = np.log(d.close).diff()
    sig = r.rolling(VOL_WIN, min_periods=40).std().shift(1)
    gap = np.log(d.open / d.close.shift(1))
    m = np.asarray(d.index.month)
    near = np.isin(m, months) | np.isin(m % 12 + 1, months)
    return (gap.abs() > GAP_K * sig) & near


def returns(d: pd.DataFrame, flags: pd.Series | None = None) -> pd.DataFrame:
    f = pd.DataFrame({"r_cc": d.close.pct_change(), "r_on": d.open / d.close.shift(1) - 1,
                      "r_id": d.close / d.open - 1}, index=d.index).fillna(0.0)
    if flags is not None:
        f.loc[flags.fillna(False).to_numpy()] = 0.0
    return f


def databento_bars(mkt: str):
    """(cash-session daily O/C, CME-trading-day daily O/H/L/C), with the real price alongside the
    back-adjusted one, from the dashboard's continuous 1m series."""
    import app
    c = app.continuous_1m(mkt)
    c.index = c.index.tz_convert(NY)
    c["raw_close"] = c.close - c.adj
    c["raw_open"] = c.open - c.adj
    mins = c.index.hour * 60 + c.index.minute
    rth = c[(mins >= 570) & (mins < 960)]
    g = rth.groupby(rth.index.date)
    first = g.head(1); last = g.tail(1)
    first = first[(first.index.hour == 9) & (first.index.minute == 30)]
    cash = pd.DataFrame({"open": first.open.to_numpy(), "raw_open": first.raw_open.to_numpy()},
                        index=pd.to_datetime(first.index.date))
    cl = pd.DataFrame({"close": last.close.to_numpy(), "raw_close": last.raw_close.to_numpy()},
                      index=pd.to_datetime(last.index.date))
    cash = cash.join(cl, how="inner")
    tday = (c.index.tz_convert("America/Chicago") + pd.Timedelta(hours=7)).tz_localize(None).normalize()
    k = c.groupby(tday)
    day = pd.DataFrame({"open": k.open.first(), "high": k.high.max(), "low": k.low.min(), "close": k.close.last(),
                        "raw_open": k.raw_open.first(), "raw_close": k.raw_close.last()})
    day = day[pd.Series(day.index.dayofweek, index=day.index) < 5]
    return cash, day


def raw_returns(d: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from back-adjusted point moves over the real (unadjusted) price."""
    pc = d.raw_close.shift(1)
    return pd.DataFrame({"r_cc": (d.close - d.close.shift(1)) / pc, "r_on": (d.open - d.close.shift(1)) / pc,
                         "r_id": (d.close - d.open) / d.raw_open}, index=d.index).fillna(0.0)


# ------------------------------------------------------------------------------------ metrics
def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return dict(start=None, end=None, **{"total %": None, "annual %": None, "Sharpe": None, "max DD %": None})
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    ann = (eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 and eq.iloc[-1] > 0 else np.nan
    sd = r.std()
    return {"start": f"{r.index[0]:%Y-%m-%d}", "end": f"{r.index[-1]:%Y-%m-%d}",
            "total %": round(100 * (eq.iloc[-1] - 1), 1), "annual %": round(100 * ann, 2),
            "Sharpe": round(float(r.mean() / sd * sqrt(252)), 2) if sd > 0 else None,
            "max DD %": round(100 * float((eq / eq.cummax() - 1).min()), 1)}


def halves(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    r = r.dropna()
    mid = r.index[0] + (r.index[-1] - r.index[0]) / 2
    return r[r.index <= mid], r[r.index > mid]


def by_decade(r: pd.Series, **keys) -> list:
    r = r.dropna()
    out = []
    for dec, g in r.groupby((r.index.year // 10) * 10):
        p = perf(g)
        out.append(dict(keys, decade=f"{dec}s", **{k: p[k] for k in ("annual %", "Sharpe", "max DD %")}))
    return out


# --------------------------------------------------------------------------------------- test 1
def test1(log=print) -> dict:
    series, rows, dec, passes = {}, [], [], []
    inst = {}
    for t in ETFS:
        d, _ = clean_ohlc(daily(t, log))
        fac = d.adj_close / d.close
        o, c = d.open * fac, d.adj_close
        inst[t] = pd.DataFrame({"on": o / c.shift(1) - 1 - 2 * ETF_COST, "id": c / o - 1 - 2 * ETF_COST,
                                "bh": c / c.shift(1) - 1}).iloc[1:]
        inst[t].iloc[0, 2] -= ETF_COST                                   # buy-and-hold entry cost
    for mkt in ("NQ", "GC"):
        cash, _ = databento_bars(mkt)
        s = MICRO[mkt]
        rt = (2 * s["tick"] * s["pv"] + 2 * COMM) / s["pv"]              # round-trip cost in points
        pc = cash.raw_close.shift(1)
        inst[f"{mkt} (Databento)"] = pd.DataFrame({
            "on": (cash.open - cash.close.shift(1) - rt) / pc,
            "id": (cash.close - cash.open - rt) / cash.raw_open,
            "bh": (cash.close - cash.close.shift(1)) / pc}).iloc[1:]
    names = {"on": "overnight only", "id": "intraday only", "bh": "buy and hold"}
    for name, f in inst.items():
        series[name] = (1 + f).cumprod()
        res = {}
        for k in ("on", "id", "bh"):
            p = perf(f[k]); h1, h2 = halves(f[k])
            res[k] = p
            rows.append(dict(instrument=name, strategy=names[k], **p,
                             **{"1st half annual %": perf(h1)["annual %"], "2nd half annual %": perf(h2)["annual %"]}))
            dec += by_decade(f[k], instrument=name, strategy=names[k])
        h1, h2 = halves(f.on)
        ok = (res["on"]["annual %"] > res["id"]["annual %"]) and perf(h1)["annual %"] > 0 and perf(h2)["annual %"] > 0
        passes.append(dict(instrument=name, role="cross-check" if name.startswith("GC") else "test",
                           **{"overnight annual %": res["on"]["annual %"], "intraday annual %": res["id"]["annual %"],
                              "overnight 1st half %": perf(h1)["annual %"], "overnight 2nd half %": perf(h2)["annual %"]},
                           **{"meets rule": "YES" if ok else "no"}))
    return dict(t1=pd.DataFrame(rows), t1_decades=pd.DataFrame(dec), t1_pass=pd.DataFrame(passes), t1_equity=series)


# --------------------------------------------------------------------------------------- test 2
def tsmom_market(f: pd.DataFrame, L: int) -> tuple[pd.Series, pd.Series]:
    """(daily P&L as a fraction of capital at 10% vol target, active flag) for one market."""
    I = (1 + f.r_cc).cumprod()
    vol = f.r_cc.rolling(VOL_WIN, min_periods=VOL_WIN).std() * sqrt(252)
    me = f.index.to_series().groupby(f.index.to_period("M")).max()
    meI = pd.Series(I.loc[me.values].to_numpy(), index=me.index)
    full = pd.period_range(meI.index.min(), meI.index.max(), freq="M")
    meI = meI.reindex(full)
    past = (meI / meI.shift(L) - 1).reindex(me.index)
    w = np.sign(past.to_numpy()) * VOL_TARGET / vol.loc[me.values].to_numpy()
    Wd = pd.Series(np.nan, index=f.index)
    Wd.loc[me.values] = w
    Wd = Wd.ffill()
    W1, W2 = Wd.shift(1), Wd.shift(2)                 # decided at t-1 close (held intraday t) / at t-2
    w1, w2 = W1.fillna(0).to_numpy(), W2.fillna(0).to_numpy()
    pnl = np.where(w1 == w2, w1 * f.r_cc, w2 * f.r_on + w1 * f.r_id) - TS_COST * np.abs(w1 - w2)
    return pd.Series(pnl, index=f.index), W1.notna()


def tsmom_portfolio(frames: dict, L: int) -> tuple[pd.Series, pd.DataFrame]:
    """Equal average over markets with a position. Returns (daily return, per-market contribution)."""
    idx = sorted(set().union(*[f.index for f in frames.values()]))
    idx = pd.DatetimeIndex(idx)
    P, A = {}, {}
    for m, f in frames.items():
        p, a = tsmom_market(f, L)
        P[m] = p.reindex(idx).fillna(0.0)
        A[m] = a.reindex(idx).ffill().fillna(False).astype(bool)
    P, A = pd.DataFrame(P), pd.DataFrame(A)
    N = A.sum(axis=1)
    contrib = (P * A).div(N.where(N > 0), axis=0)
    live = N > 0
    return contrib.sum(axis=1)[live], contrib[live]


def test2(log=print) -> dict:
    import validate as v
    raw, flags, info = {}, {}, []
    for t, months in FUTURES.items():
        d, bad = clean_ohlc(daily(t, log))
        raw[t] = d
        flags[t] = roll_flags(d, months)
        info.append(dict(market=t, first=f"{d.index[0]:%Y-%m-%d}", days=len(d), **{"bad rows dropped": bad,
                         "roll gaps flagged": int(flags[t].sum())}))
    es = {}
    out = dict(t2_rolls=pd.DataFrame(info))
    rows, dec, contrib_rows, passes, equity = [], [], {}, [], {}
    for version in ("as-is", "roll days zeroed"):
        fr = {t: returns(raw[t], flags[t] if version != "as-is" else None) for t in FUTURES}
        es_r = fr["ES=F"].r_cc
        port, contrib = {}, {}
        for L in LOOKBACKS:
            port[f"{L}m"], contrib[f"{L}m"] = tsmom_portfolio(fr, L)
        common = port["3m"].index.intersection(port["6m"].index).intersection(port["12m"].index)
        port["blend"] = sum(port[f"{L}m"].reindex(common) for L in LOOKBACKS) / 3
        contrib["blend"] = sum(contrib[f"{L}m"].reindex(common) for L in LOOKBACKS) / 3
        M = np.column_stack([port[k].reindex(common).to_numpy() for k in ("3m", "6m", "12m", "blend")])
        all_sr = v.sharpe(M)
        for j, k in enumerate(("3m", "6m", "12m", "blend")):
            r = port[k]
            p = perf(r); h1, h2 = halves(r)
            dsr = v.deflated_sharpe(M[:, j], all_sr)
            corr = r.corr(es_r.reindex(r.index))
            rows.append(dict(data=version, lookback=k, **p, **{"Sharpe 1st half": perf(h1)["Sharpe"],
                             "Sharpe 2nd half": perf(h2)["Sharpe"], "corr with ES": round(float(corr), 2),
                             "deflated Sharpe": round(float(dsr), 3)}))
            dec += by_decade(r, data=version, lookback=k)
            equity[(version, k)] = (1 + r).cumprod()
        yrs = {k: (contrib[k].index[-1] - contrib[k].index[0]).days / 365.25 for k in contrib}
        contrib_rows[version] = pd.DataFrame({f"{k} %/yr": 100 * contrib[k].sum() / yrs[k] for k in contrib})
        b = rows[-1]
        n_pos = int((contrib_rows[version]["blend %/yr"] > 0).sum())
        ok = (b["Sharpe 1st half"] or -9) > 0.3 and (b["Sharpe 2nd half"] or -9) > 0.3 and b["deflated Sharpe"] >= 0.95 \
            and n_pos > len(FUTURES) / 2
        passes.append(dict(data=version, **{"blend Sharpe 1st half": b["Sharpe 1st half"],
                           "blend Sharpe 2nd half": b["Sharpe 2nd half"], "deflated Sharpe": b["deflated Sharpe"],
                           "markets contributing > 0": f"{n_pos}/{len(FUTURES)}"}, **{"meets rule": "YES" if ok else "no"}))
    ct = contrib_rows["roll days zeroed"].round(2).add_suffix(" (zeroed)").join(
        contrib_rows["as-is"].round(2)[["blend %/yr"]].rename(columns={"blend %/yr": "blend %/yr (as-is)"}))
    ct.insert(0, "roll gaps flagged", out["t2_rolls"].set_index("market")["roll gaps flagged"])
    out.update(t2=pd.DataFrame(rows), t2_decades=pd.DataFrame(dec), t2_contrib=ct.reset_index(names="market"),
               t2_pass=pd.DataFrame(passes), t2_equity=equity, t2_cross=crosscheck(raw, flags))
    return out


def crosscheck(raw: dict, flags: dict) -> pd.DataFrame:
    """NQ and GC alone: Databento daily vs Yahoo as-is vs Yahoo roll-zeroed, same dates."""
    rows = []
    for mkt, t in (("NQ", "NQ=F"), ("GC", "GC=F")):
        _, day = databento_bars(mkt)
        vers = {"Databento": raw_returns(day), "Yahoo as-is": returns(raw[t]),
                "Yahoo roll-zeroed": returns(raw[t], flags[t])}
        res = {}
        for name, f in vers.items():
            for L in LOOKBACKS:
                res[(name, f"{L}m")] = tsmom_portfolio({mkt: f}, L)[0]
        start = max(s.index[0] for s in res.values()); end = min(s.index[-1] for s in res.values())
        common = None
        for name in vers:
            parts = [res[(name, f"{L}m")] for L in LOOKBACKS]
            for p in parts:
                common = p.index if common is None else common.intersection(p.index)
        agree = vers["Databento"].r_cc.corr(vers["Yahoo as-is"].r_cc.reindex(vers["Databento"].index))
        for name in vers:
            for k in [f"{L}m" for L in LOOKBACKS] + ["blend"]:
                r = (sum(res[(name, f"{L}m")].reindex(common) for L in LOOKBACKS) / 3) if k == "blend" \
                    else res[(name, k)].reindex(common)
                r = r[(r.index >= start) & (r.index <= end)]
                p = perf(r)
                rows.append(dict(market=mkt, data=name, lookback=k, start=p["start"], end=p["end"],
                                 **{"annual %": p["annual %"], "Sharpe": p["Sharpe"], "max DD %": p["max DD %"],
                                    "daily corr Databento vs Yahoo": round(float(agree), 3)}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- test 3
def test3(log=print) -> dict:
    import app, setup_report as sr
    vix = daily(VIX, log).close.dropna()
    vix = vix[vix > 0]
    med = vix.rolling(252, min_periods=200).median()
    keys = [("databento", m, s) for m in ("MNQ", "MGC") for s in ("new_york", "asia")]
    for k in keys:
        app.pack_future(k, front=False)
    trades = []
    for tag, g, mkt, sess, fam, params, rr in app.study.SETUPS:
        a, b = sr.full_range(mkt)
        t = app.with_r(sr.setup_trades(("databento", mkt, sess), tag, a, b), app.base_cfg(mkt, sess))
        trades.append(t.assign(setup=tag))
    raw1, _ = sr.bars_1m("MNQ")
    t = app.with_r(sr.echo_trades(raw1), app.base_cfg("MNQ", "asia"))
    trades.append(t.assign(setup="ECHO"))
    T = pd.concat(trades, ignore_index=True)
    et = pd.to_datetime(T.entry_time).dt.tz_convert(NY)
    T["year"] = et.dt.year
    day = et.dt.tz_localize(None).dt.normalize()
    pos = vix.index.searchsorted(day.to_numpy(), side="left") - 1         # last VIX close before the entry date
    ok = pos >= 0
    vx = np.where(ok, vix.to_numpy()[np.clip(pos, 0, None)], np.nan)
    md = np.where(ok, med.to_numpy()[np.clip(pos, 0, None)], np.nan)
    T["regime"] = np.where(np.isnan(md), "n/a", np.where(vx > md, "high VIX", "low VIX"))
    T["vix"] = vx

    def agg(g):
        return {"trades": len(g), "win %": round(100 * float((g.pnl > 0).mean()), 1) if len(g) else None,
                "avg R": round(float(g.R.mean()), 3) if len(g) else None, "net $": round(float(g.pnl.sum()))}
    pooled, per = [], []
    for (reg, yr), g in T.groupby(["regime", "year"]):
        pooled.append(dict(regime=reg, year=int(yr), **agg(g)))
    for reg, g in T.groupby("regime"):
        pooled.append(dict(regime=reg, year="all", **agg(g)))
    for (s, reg, yr), g in T.groupby(["setup", "regime", "year"]):
        if yr in (2024, 2025, 2026):
            per.append(dict(setup=s, regime=reg, year=int(yr), **agg(g)))
    P = pd.DataFrame(pooled)
    passes = []
    for reg in ("high VIX", "low VIX"):
        r = {y: P[(P.regime == reg) & (P.year == y)]["avg R"] for y in (2024, 2025)}
        r = {y: (float(s.iloc[0]) if len(s) else np.nan) for y, s in r.items()}
        passes.append(dict(regime=reg, **{"avg R 2024": round(r[2024], 3), "avg R 2025": round(r[2025], 3)},
                           **{"meets rule": "YES" if r[2024] > 0 and r[2025] > 0 else "no"}))
    eq = {reg: g.sort_values("entry_time").set_index(pd.to_datetime(g.sort_values("entry_time").entry_time)).R.cumsum()
          for reg, g in T.groupby("regime") if reg != "n/a"}
    return dict(t3=P, t3_setups=pd.DataFrame(per), t3_pass=pd.DataFrame(passes), t3_equity=eq,
                t3_n=len(T), t3_setups_n=T.setup.nunique())


# ------------------------------------------------------------------------------------- running
VARIANTS = ("Test 1: 3 instruments (SPY, QQQ, NQ) x 2 strategies (overnight, intraday) vs buy-and-hold; "
            "GC Databento as an extra cross-check. Test 2: 4 trials (3, 6, 12 months, blend) x 2 data versions "
            "(as-is, roll days zeroed) = 8, deflated Sharpe over the 4 trials of each version; NQ/GC Databento "
            "cross-check separate. Test 3: 1 split (2 regimes) of the existing setup trades.")


def run_all(log=print) -> dict:
    out = {}
    t0 = time.time()
    out.update(test1(log)); log(f"  test 1 done ({time.time() - t0:.0f}s)")
    out.update(test2(log)); log(f"  test 2 done ({time.time() - t0:.0f}s)")
    out.update(test3(log)); log(f"  test 3 done ({time.time() - t0:.0f}s)")
    out["variants"] = VARIANTS
    return out


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    R = run_all()
    print(f"\nVariants tested. {VARIANTS}")
    print("\nTEST 1  overnight vs intraday (after costs)")
    print(R["t1"].to_string(index=False))
    print("\nTest 1 by decade")
    print(R["t1_decades"].to_string(index=False))
    print("\nTest 1 rule: overnight beats intraday on annual return, and overnight > 0 in both halves")
    print(R["t1_pass"].to_string(index=False))
    print("\nTEST 2  time-series momentum: data and roll gaps (|open gap| > 4 x 60-day vol near contract months)")
    print(R["t2_rolls"].to_string(index=False))
    print("\nTest 2 results (after 2 bp per side)")
    print(R["t2"].to_string(index=False))
    print("\nTest 2 by decade")
    print(R["t2_decades"].to_string(index=False))
    print("\nTest 2 contribution by market (% of capital per year)")
    print(R["t2_contrib"].to_string(index=False))
    print("\nTest 2 rule: blend Sharpe > 0.3 in both halves, deflated Sharpe >= 0.95, > half of markets contribute > 0")
    print(R["t2_pass"].to_string(index=False))
    print("\nTest 2 cross-check (not a pass test): NQ and GC alone, Databento vs Yahoo, same dates")
    print(R["t2_cross"].to_string(index=False))
    print(f"\nTEST 3  VIX regime on {R['t3_n']} setup trades ({R['t3_setups_n']} setups), pooled")
    print(R["t3"].to_string(index=False))
    print("\nTest 3 per setup, 2024-2026")
    print(R["t3_setups"].to_string(index=False))
    print("\nTest 3 rule: one regime has pooled average R > 0 in both 2024 and 2025")
    print(R["t3_pass"].to_string(index=False))


if __name__ == "__main__":
    main()
