"""Strategy validation harness: every strategy x NY/Asia session, with pass/fail on
initial results, walk-forward, PBO (CSCV), deflated Sharpe, cost stress and nearby settings,
plus a breakeven-at-1R comparison for every 3R configuration.

usage: python validate.py <outputs_dir_with_nq_backtest> <MGC|MNQ> <new_york|asia> <out_dir>
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from scipy.stats import norm
import nq_backtest as bt
import session_timeframe_comparison as cmp

INSTR = {"MGC": dict(file="mgc_yahoo_5m.csv", tick_size=0.10, point_value=10.0),
         "MNQ": dict(file="mnq_yahoo_5m.csv", tick_size=0.25, point_value=2.0)}
EXIT = {"new_york": "15:00", "asia": "02:00"}      # Chicago time, = 16:00 / 03:00 ET
PAPA = {"new_york": (14*60, "14:45"), "asia": (60, "01:45")}   # anchor minute CT, flat time CT
RRS = [1.0, 1.5, 2.0, 2.5, 3.0]
# drawdown-stop paths are not monotonic in cost, so every scenario must pass on its own
STRESS = [dict(slippage_ticks=1.5), dict(slippage_ticks=2.0), dict(commission_per_side=3.75),
          dict(slippage_ticks=2.0, commission_per_side=3.75)]
NOKILL = dict(drawdown_dollars=1e12)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Repair single-bar bad prints: open far from prior close but close back near it."""
    r = raw.copy()
    med = (r.high - r.low).rolling(200, min_periods=50).median().bfill()
    prev = r.close.shift(1)
    jump = (r.open - prev).abs()
    bad = (jump > 10 * med) & ((r.close - prev).abs() < jump / 3)
    r.loc[bad, "open"] = prev[bad]
    r.loc[bad, "high"] = np.maximum(r.loc[bad, "close"], prev[bad]) + med[bad]
    r.loc[bad, "high"] = np.minimum(r.loc[bad, "high"], raw.loc[bad, "high"])
    r.loc[bad, "low"] = np.maximum(raw.loc[bad, "low"], np.minimum(r.loc[bad, "close"], prev[bad]) - med[bad])
    return r, int(bad.sum())


def papa_signal(x, session, pct):
    """Fade a pct-of-price move from the last-hour anchor open; first close beyond it in 40 min."""
    anchor_min, _ = PAPA[session]
    sig = pd.Series(0, index=x.index, dtype="int8"); tgt = pd.Series(np.nan, index=x.index)
    mins = pd.Series(x.index.hour * 60 + x.index.minute, index=x.index)
    keys = cmp.session_keys(x.index, session)
    for key in keys.dropna().unique():
        day = x.index[keys.eq(key).to_numpy()]
        m = mins.loc[day].to_numpy()
        a = day[m == anchor_min]
        if not len(a): continue
        anchor = float(x.at[a[0], "open"]); pts = anchor * pct
        for ts in day[(m > anchor_min) & (m < anchor_min + 40)]:
            c = float(x.at[ts, "close"])
            d = 1 if c <= anchor - pts else -1 if c >= anchor + pts else 0
            if d: sig.at[ts], tgt.at[ts] = d, pts; break
    return sig, tgt


def romeo(x, cfg):
    vol = x.volume.astype(float)
    p = (vol * np.sign(x.close - x.open)).rolling(12, min_periods=8).sum() / vol.rolling(12, min_periods=8).sum().replace(0, np.nan)
    s = np.select([(p > .3) & (x.close > x.high.shift(1)), (p < -.3) & (x.close < x.low.shift(1))], [1, -1], 0)
    return pd.Series(s, index=x.index, dtype="int8").where(bt.in_session(x.index, cfg), 0)


def families(raw, cfg, session):
    """Yield (family, params, ordinal_param_names, builder). builder(rr) -> (x, signal, atr, exit_time)."""
    x5 = bt.indicators(raw)
    xs = {tf: (x5 if tf == 5 else bt.indicators(cmp.resample_ohlcv(raw, tf))) for tf in (5, 10, 15)}
    insess = bt.in_session(x5.index, cfg)
    vwap = bt.session_vwap(x5, cfg)
    ex = EXIT[session]
    def vw(sig): return sig.where(((sig > 0) & (x5.close > vwap)) | ((sig < 0) & (x5.close < vwap)), 0)
    out = []
    for bb, rl in itertools.product((1.5, 2.0, 2.5), (25.0, 30.0, 35.0)):
        sg = bt.make_signals(x5, "lima", cfg, bb, rl)
        out.append(("Lima BB+RSI", dict(bb_mult=bb, rsi=rl), ["bb_mult", "rsi"], lambda rr, sg=sg: (x5, sg, None, ex)))
    for name, key in (("RSI mean reversion", "mean_reversion_rsi"), ("Momentum", "momentum"),
                      ("EMA cloud", "ema_cloud"), ("BOS/CHOCH", "bos_choch")):
        sg = bt.make_signals(x5, key, cfg)
        out.append((name, {}, [], lambda rr, sg=sg: (x5, sg, None, ex)))
    for v in ("fvg", "ifvg", "fvg_vwap", "ifvg_vwap"):
        sg = bt.make_signals(x5, v, cfg)
        out.append(("FVG 5/15/30 alignment", dict(variant=v), [], lambda rr, sg=sg: (x5, sg, None, ex)))
    for tf in (5, 15, 30):
        sg0, _, atr = bt.make_fvg_ifvg_signals(x5, tf)
        for f in (0, 1):
            sg = sg0.where(insess, 0); sg = vw(sg) if f else sg
            out.append(("FVG/IFVG retest", dict(tf=tf, vwap=f), ["tf"], lambda rr, sg=sg, atr=atr: (x5, sg, atr, ex)))
    for tf in (15, 30):
        sg, _, atr = bt.make_fvg_structure_signals(x5, tf)
        out.append(("FVG + BOS/CHOCH", dict(tf=tf), ["tf"], lambda rr, sg=sg.where(insess, 0), atr=atr: (x5, sg, atr, ex)))
        sg0, _, atr, _ = bt.make_fvg_orderblock_signals(x5, tf)
        for f in (0, 1):
            sg = sg0.where(insess, 0); sg = vw(sg) if f else sg
            out.append(("FVG + order block", dict(tf=tf, vwap=f), ["tf"], lambda rr, sg=sg, atr=atr: (x5, sg, atr, ex)))
    for tf in (5, 10, 15):
        x = xs[tf]; x.attrs["bar_minutes"] = tf
        for sd in (1.5, 2.0, 2.5):
            sg = cmp.avwap_signal(x, session, sd)
            out.append(("November AVWAP", dict(tf=tf, sigma=sd), ["tf", "sigma"], lambda rr, x=x, sg=sg: (x, sg, None, ex)))
    for orb in (5, 15):
        for var in ("breakout", "two_close_fakeout_filter", "retest"):
            for tf in (5, 10, 15):
                sg, _ = cmp.orb_signal(raw, xs[tf], session, tf, orb, var)
                out.append(("Oscar ORB", dict(orb=orb, variant=var, tf=tf), ["orb", "tf"],
                            lambda rr, x=xs[tf], sg=sg: (x, sg, None, ex)))
    for pct in (0.0007, 0.0010, 0.0014):
        sg, tgt = papa_signal(x5, session, pct)
        out.append(("Papa last-hour fade", dict(target_pct=pct), ["target_pct"],
                    lambda rr, sg=sg, tgt=tgt: (x5, sg, tgt / rr, PAPA[session][1])))
    out.append(("Romeo volume proxy", {}, [], lambda rr, sg=romeo(x5, cfg): (x5, sg, None, ex)))
    return out


def run(build, rr, cfg, **kw):
    x, sg, atr, ex = build(rr)
    c = bt.Config(**{**cfg.__dict__, "rr": rr, **kw})
    return bt.backtest(x, sg.astype("int8"), c, atr, None, flat_at_session_end=True,
                       session_exit_tz=cfg.session_tz, session_exit_time=ex)


def stats(t):
    """Eval rules on an un-stopped run: the test is over at the first -$1,000 trailing closed-trade DD.
    (The engine's own DD stop skips trades that would breach, which parks equity just above -$1,000.)"""
    if not len(t):
        return dict(trades=0, net=0.0, win_rate=0.0, max_dd=0.0, busted=False, hit_1250_first=False)
    eq = t.pnl.cumsum().to_numpy(); dd = np.maximum.accumulate(np.maximum(eq, 0)) - eq
    breach = np.flatnonzero(dd >= 1000)
    n = breach[0] + 1 if len(breach) else len(t)
    pnl = t.pnl.iloc[:n]
    return dict(trades=n, net=float(pnl.sum()), win_rate=float((pnl > 0).mean()),
                max_dd=float(dd[:n].max()), busted=bool(len(breach)), hit_1250_first=bool((eq[:n] >= 1250).any()))


def daily(t, days, session):
    if not len(t): return pd.Series(0.0, index=days)
    k = [cmp.session_id(pd.Timestamp(e), session) or cmp.session_trade_key(pd.Timestamp(e), session) for e in t.entry_time]
    return pd.Series(t.pnl.to_numpy(), index=k).groupby(level=0).sum().reindex(days, fill_value=0.0)


def sharpe(m):  # per-day Sharpe of each column
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)


def pbo(M, blocks=8):
    """Probability of backtest overfitting via combinatorially symmetric cross-validation."""
    if M.shape[1] < 2: return np.nan
    parts = np.array_split(np.arange(M.shape[0]), blocks)
    lam = []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        ins = np.concatenate([parts[i] for i in combo])
        oos = np.concatenate([parts[i] for i in range(blocks) if i not in combo])
        best = int(np.argmax(sharpe(M[ins])))
        so = sharpe(M[oos])
        w = (np.sum(so < so[best]) + 0.5 * np.sum(so == so[best]) + 0.5) / (len(so) + 1)
        lam.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def deflated_sharpe(r, all_sr):
    """Bailey & Lopez de Prado DSR: P(true Sharpe > best-of-N expected under no skill)."""
    r = np.asarray(r, float); T = len(r); n = len(all_sr)
    if r.std(ddof=1) == 0 or T < 10: return 0.0
    sr = r.mean() / r.std(ddof=1)
    g = 0.5772156649
    sr0 = np.std(all_sr, ddof=1) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    sk = pd.Series(r).skew(); ku = pd.Series(r).kurt() + 3
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))


def walk_forward(M, folds=5):
    """Anchored: pick best config on folds 1..k, trade it unchanged on fold k+1."""
    parts = np.array_split(np.arange(M.shape[0]), folds)
    oos = 0.0
    for k in range(1, folds):
        ins = np.concatenate(parts[:k])
        oos += float(M[parts[k], int(np.argmax(M[ins].sum(axis=0)))].sum())
    return oos


def main():
    inst, session, out = sys.argv[2], sys.argv[3], Path(sys.argv[4])
    spec = INSTR[inst]
    cfg = bt.Config(tick_size=spec["tick_size"], point_value=spec["point_value"], session=session,
                    contracts=1, risk_fraction=0.0, slippage_ticks=1.0, commission_per_side=2.50,
                    drawdown_dollars=1000.0, target_dollars=1250.0)
    raw = bt.load_bars(Path(sys.argv[1]).parent / "data" / spec["file"], cfg)
    raw, nbad = clean(raw)
    days = cmp.session_keys(raw.index, session).dropna().unique()
    rows, cols, be_rows = [], [], []
    fams = families(raw, cfg, session)
    for fam, params, ordinal, build in fams:
        for rr in RRS:
            tn = run(build, rr, cfg, **NOKILL); tk = tn
            rows.append(dict(family=fam, **params, rr=rr, **stats(tk), net_no_ddstop=float(tn.pnl.sum())))
            cols.append(daily(tn, days, session).to_numpy())
            if rr == 3.0:
                bn = run(build, rr, cfg, breakeven_r=1.0, **NOKILL); bk = bn
                be_rows.append(dict(family=fam, **params, rr=3.0,
                    base_net=stats(tk)["net"], be_net=stats(bk)["net"],
                    base_net_no_ddstop=float(tn.pnl.sum()), be_net_no_ddstop=float(bn.pnl.sum()),
                    base_win=stats(tn)["win_rate"], be_win=stats(bn)["win_rate"],
                    base_busted=stats(tk)["busted"], be_busted=stats(bk)["busted"],
                    trades=len(tn),
                    round_trips=int(((tn.mfe_r >= 1) & (tn.reason == "stop")).sum()) if len(tn) else 0,
                    near_misses=int(((tn.mfe_r >= 2) & (tn.reason != "target")).sum()) if len(tn) else 0,
                    be_exits=int((bn.reason == "breakeven").sum()) if len(bn) else 0,
                    base_max_dd=stats(tn)["max_dd"], be_max_dd=stats(bn)["max_dd"]))
        print(fam, params, flush=True)
    grid = pd.DataFrame(rows); M_all = np.column_stack(cols)
    all_sr = sharpe(M_all)
    summary = []
    for fam, g in grid.groupby("family", sort=False):
        idx = g.index.to_numpy(); M = M_all[:, idx]
        b = g.net.idxmax(); best = grid.loc[b]
        ordinal = next(o for f, p, o, _ in fams if f == fam)
        pcols = [c for c in g.columns if c in ("bb_mult", "rsi", "variant", "tf", "vwap", "sigma", "orb", "target_pct", "va_pct")]
        # neighbours: same categorical params, one step on exactly one ordinal param (incl. RR)
        def step(c, v): vals = sorted(g[c].dropna().unique()); i = vals.index(v); return {vals[j] for j in (i-1, i+1) if 0 <= j < len(vals)}
        nb = []
        for c in ordinal + ["rr"]:
            for v in step(c, best[c]):
                m = g[(g[c] == v)]
                for o in pcols + ["rr"]:
                    if o != c and o in g and pd.notna(best[o]): m = m[m[o] == best[o]]
                nb += list(m.net)
        build = next(bu for f, p, o, bu in fams if f == fam and all(best.get(k) == v for k, v in p.items()))
        st = [stats(run(build, best.rr, cfg, **k, **NOKILL)) for k in STRESS]
        wf = walk_forward(M); p = pbo(M)
        dsr = deflated_sharpe(M_all[:, b], all_sr)
        initial = best.net > 0 and not best.busted and best.trades >= 20
        tests = dict(initial=initial, walk_forward=wf > 0, pbo=(p < 0.5) if np.isfinite(p) else False,
                     deflated_sharpe=dsr >= 0.95, cost_stress=all(t["net"] > 0 and not t["busted"] for t in st),
                     nearby=len(nb) >= 2 and np.mean(np.array(nb) > 0) >= 0.6)
        summary.append(dict(instrument=inst, session=session, family=fam,
            best_config=", ".join([f"{k}={best[k]:g}" if isinstance(best[k], float) else f"{k}={best[k]}" for k in pcols if pd.notna(best.get(k))] + [f"rr={best.rr:g}"]),
            configs=len(g), trades=int(best.trades), net=round(best.net, 2), win_rate=round(best.win_rate, 3),
            max_dd=round(best.max_dd, 2), hit_1250_before_fail=bool(best.hit_1250_first),
            **{f"{k}_pass": bool(v) for k, v in tests.items()},
            all_pass=all(tests.values()), walk_forward_oos=round(wf, 2), pbo=round(p, 3) if np.isfinite(p) else None,
            dsr=round(dsr, 3), stress_worst_net=round(min(t["net"] for t in st), 2),
            nearby_positive=f"{int(np.sum(np.array(nb) > 0))}/{len(nb)}"))
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{inst.lower()}_{session}"
    grid.assign(instrument=inst, session=session).to_csv(out / f"grid_{tag}.csv", index=False)
    pd.DataFrame(summary).to_csv(out / f"passfail_{tag}.csv", index=False)
    pd.DataFrame(be_rows).assign(instrument=inst, session=session).to_csv(out / f"breakeven_{tag}.csv", index=False)
    print(f"done {tag}: {len(grid)} configs, {nbad} bad bars repaired, {len(days)} session days")


if __name__ == "__main__":
    main()
