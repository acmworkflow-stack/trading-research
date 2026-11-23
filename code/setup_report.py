"""Setup reports on the Databento series. Prints tables; sections B and C also save them as CSV
in results/setup_report/ (plus the dashboard's own data/cache files, which it reuses).

    python code/setup_report.py [overlap] [setups] [vp]        (no argument = all three)

overlap  study.SETUPS on Databento vs the Yahoo 5m files over 2026-08-21 .. 2026-10-30:
         trades, win rate, net, share of matching entries, plus Databento signals built from
         the overlap bars alone (separates warm-up history from data differences)
setups   every study.SETUPS entry plus ECHO on 3 years: net, average R, trades/day, max DD,
         net and average R by year, prop-firm evaluation pass rate (25K account rules in
         code/evalsim.py; bootstrap and real order). Also written to results/setup_report/setups.csv
vp       vp.py's grid (3 families x 3 value areas x 5 R:R) on MNQ/MGC x NY/Asia, 3 years,
         with average R in 2024, 2025 and 2026. Also written to results/setup_report/vp_grid.csv

ECHO = vp.py "edge_reject", MNQ Asia, 60% value area, 3R, prior NY session's profile, with the
profile built from the 1-minute Databento bars (vp.levels on 1m bars) instead of 5-minute bars.
Signals, fills and exits are vp.py / validate.run unchanged. A 5m-profile ECHO row (vp.py as is)
is shown for comparison. Nothing is tuned here.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import app
from app import v, bt, study, NY
_argv = sys.argv
sys.argv = [_argv[0], str(app.CODE)]          # vp.py reads sys.argv[1] at import
import vp                                     # noqa: E402
sys.argv = _argv

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
OV_A, OV_B = pd.Timestamp("2026-08-21", tz=NY), pd.Timestamp("2026-10-31", tz=NY)
YEARS = (2023, 2024, 2025, 2026)
ECHO = dict(mkt="MNQ", session="asia", kind="edge_reject", va=0.6, rr=3.0)
OUT = app.ROOT / "results" / "setup_report"


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"saved {OUT / name}", flush=True)


# ------------------------------------------------------------------------------------- helpers
def setup_trades(key, tag, a, b):
    pack = app.PACKS[key].result()
    s = pack["setups"][tag]
    cfg = app.base_cfg(key[1], key[2])
    return app.run_many(key, cfg, dict(app.NOKILL), a, b, [(s["idx"], s["rr"], ())])[(s["idx"], s["rr"], ())]


def full_range(mkt):
    raw, _ = app.load("databento", mkt)
    return raw.index[0], raw.index[-1] + pd.Timedelta(minutes=5)


def match(t1, t2):
    k1 = set(zip(pd.to_datetime(t1.entry_time), t1.direction)); k2 = set(zip(pd.to_datetime(t2.entry_time), t2.direction))
    return len(k1 & k2) / max(len(k1), len(k2), 1)


def entry_year(t):
    return pd.to_datetime(t.entry_time).dt.tz_convert(NY).dt.year


# ---- evalsim.py rules (copied: that script reads trades.pkl when imported)
TARGET, MLL, DLL, MAXDAYS, PATHS = 1250.0, 1000.0, 600.0, 60, 4000


def day_lists(t, alldays):
    t = t.assign(tday=(pd.to_datetime(t.exit_time, utc=True).dt.tz_convert(NY) + pd.Timedelta(hours=7)).dt.date,
                 xt=pd.to_datetime(t.exit_time, utc=True)).sort_values("xt")
    g = {d: x.pnl.to_numpy() for d, x in t.groupby("tday")}
    return [g.get(d, np.array([])) for d in alldays]


def one_eval(seq):
    bal = peak_eod = 0.0
    for k, day in enumerate(seq):
        dpl = 0.0
        for p in day:
            bal += p; dpl += p
            if bal <= peak_eod - MLL: return "fail"
            if bal >= TARGET: return "pass"
            if dpl <= -DLL: break
        peak_eod = max(peak_eod, bal)
    return None


def eval_rates(t, a, b):
    """(bootstrap pass, fail) as evalsim.sim: 4000 paths of 60 days drawn with replacement from every
    business day; (real pass, fail): an eval started on every business day, days in real order."""
    alldays = pd.bdate_range(a.tz_convert(NY).date(), b.tz_convert(NY).date()).date
    days = day_lists(t, alldays)
    rng = np.random.default_rng(1)
    boot = [one_eval([days[rng.integers(len(days))] for _ in range(MAXDAYS)]) for _ in range(PATHS)]
    real = [one_eval(days[s:s + MAXDAYS]) for s in range(len(days) - MAXDAYS)]
    rate = lambda r, k: round(100 * float(np.mean([x == k for x in r])))
    return rate(boot, "pass"), rate(boot, "fail"), rate(real, "pass"), rate(real, "fail")


def metrics_row(tag, mkt, sess, rr, t, a, b):
    cfg = app.base_cfg(mkt, sess)
    t = app.with_r(t, cfg)
    s = app.summarize(tag, t, app.session_days("databento", mkt, sess, a, b))
    yr = entry_year(t)
    bp, bf, hp, hf = eval_rates(t, a, b)
    return dict(setup=tag, mkt=mkt, sess=app.SESS_LABEL[sess], rr=rr, trades=len(t), win=s["win_rate"],
                net=round(t.pnl.sum()), avg_R=s["avg_r"], per_day=s["trades_per_day"], max_dd=round(s["max_dd"]),
                **{f"net_{y}": round(t.pnl[yr == y].sum()) for y in YEARS},
                **{f"R_{y}": round(float(t.R[yr == y].mean()), 3) if (yr == y).any() else None for y in (2024, 2025, 2026)},
                pass_sim=bp, fail_sim=bf, pass_real=hp, fail_real=hf)


# ---------------------------------------------------------------------------------------- ECHO
def bars_1m(mkt):
    """The same rolled/adjusted series as the 5m cache, at 1 minute, Chicago tz, through validate.clean."""
    c = app.continuous_1m(mkt).drop(columns="symbol")
    c.index = c.index.tz_convert("America/Chicago")
    return v.clean(c)


def profile_on(index, prof_bars, mkt, va):
    """vp.levels on the given bars, then each 5m bar sees the latest profile whose session has closed."""
    lv = vp.levels(prof_bars, vp.BIN[mkt], va)
    return lv.reindex(index.union(lv.index)).ffill().reindex(index)


def echo_signal(raw5, raw1, cfg, use_1m=True):
    x5 = bt.indicators(raw5)
    lv = profile_on(x5.index, raw1, ECHO["mkt"], ECHO["va"]) if use_1m else vp.levels(raw5, vp.BIN[ECHO["mkt"]], ECHO["va"])
    return x5, vp.vp_signals(x5, lv, cfg, ECHO["session"], ECHO["kind"])


def echo_truncation_test(raw5, raw1, cfg, ncuts=25):
    """lookahead.py's test for the new 1m-profile construction: signals on bars already closed at a cut
    must not change when everything after the cut (5m and 1m) is removed."""
    _, full = echo_signal(raw5, raw1, cfg)
    rng = np.random.default_rng(0)
    bad = 0
    for cut in sorted(rng.integers(3000, len(raw5) - 10, ncuts)):
        r5 = raw5.iloc[:cut]
        r1 = raw1[raw1.index < r5.index[-1] + pd.Timedelta(minutes=5)]
        _, part = echo_signal(r5, r1, cfg)
        common = part.index[:-1]
        bad += bool((full.reindex(common).fillna(0).astype(int) != part.reindex(common).fillna(0).astype(int)).any())
    print(f"ECHO 1m-profile truncation test: {'LEAK' if bad else 'ok'} ({bad}/{ncuts} cuts differ)", flush=True)
    return bad == 0


def echo_trades(raw1=None):
    """raw1 = 1m bars for the 1m-profile ECHO; None = vp.py's own 5m profile."""
    mkt, sess = ECHO["mkt"], ECHO["session"]
    raw5, _ = app.load("databento", mkt)
    cfg = app.base_cfg(mkt, sess)
    x5, sg = echo_signal(raw5, raw1, cfg, use_1m=raw1 is not None)
    return v.run(lambda rr: (x5, sg, None, v.EXIT[sess]), ECHO["rr"], cfg, **app.NOKILL)


# ------------------------------------------------------------------------------------- reports
def report_overlap():
    for mkt in ("MNQ", "MGC"):
        y, _ = app.load("yahoo", mkt); d, _ = app.load("databento", mkt)
        d = d[(d.index >= OV_A) & (d.index < OV_B)]
        j = y.join(d.add_suffix("_d"), how="inner")
        day = j.index.tz_convert(NY).date
        off = j.close_d - j.close
        resid = (off - off.groupby(day).transform("median")).abs()
        only = d.index.difference(y.index)
        hrs = pd.Series(only.tz_convert(NY).hour)
        print(f"\n{mkt}: Yahoo {len(y)} bars, Databento {len(d)}, {len(j)} common; Databento-only bars {len(only)} "
              f"({int(hrs.isin([18, 19, 20, 21, 22, 23, 0, 1, 2]).sum())} in 18:00-03:00 ET); close offset by day "
              f"{off.groupby(day).median().min():+.2f} to {off.groupby(day).median().max():+.2f} (back-adjustment); "
              f"after offset median |diff| {resid.median():.2f}, closes > 2 ticks apart on "
              f"{100 * (resid > 2 * app.MARKETS[mkt]['tick_size']).mean():.0f}% of bars; 5m return corr "
              f"{j.close.pct_change().corr(j.close_d.pct_change()):.3f}; volume ratio {j.volume.sum() / j.volume_d.sum():.2f}")
    rows = []
    for mkt in ("MNQ", "MGC"):
        for sess in ("new_york", "asia"):
            todo = [s for s in study.SETUPS if (s[2], s[3]) == (mkt, sess)]
            if not todo:
                continue
            cfg = app.base_cfg(mkt, sess)
            raw, _ = app.load("databento", mkt)
            r = raw[(raw.index >= OV_A) & (raw.index < OV_B)]
            fams = v.families(r, cfg, sess)
            x5 = bt.indicators(r)
            ps, pt = v.papa_signal(x5, sess, 0.001); pt = pt.where(pt.isna(), 30.0)
            for tag, g, _, _, fam, params, rr in todo:
                y = setup_trades(("yahoo", mkt, sess), tag, OV_A, OV_B)
                d = setup_trades(("databento", mkt, sess), tag, OV_A, OV_B)
                b = (lambda q: (x5, ps, pt / q, v.PAPA[sess][1])) if fam == "PAPA30" else study.build(fams, fam, params)
                o = v.run(b, rr, cfg, **app.NOKILL)
                win = lambda t: round(100 * float((t.pnl > 0).mean()), 1) if len(t) else None
                rows.append(dict(setup=tag, mkt=mkt, sess=app.SESS_LABEL[sess], rr=rr, y_trades=len(y), d_trades=len(d),
                                 y_win=win(y), d_win=win(d), y_net=round(y.pnl.sum()), d_net=round(d.pnl.sum()),
                                 entry_match=round(100 * match(y, d)), d_hist_vs_overlap_only=round(100 * match(d, o)),
                                 y_vs_overlap_only=round(100 * match(y, o))))
    print("\n(A) overlap 2026-08-21 to 2026-10-30; entry_match = same (entry time, direction) / larger trade count")
    print(pd.DataFrame(rows).to_string(index=False))


def report_setups():
    rows = []
    for tag, g, mkt, sess, fam, params, rr in study.SETUPS:
        a, b = full_range(mkt)
        rows.append(metrics_row(tag, mkt, sess, rr, setup_trades(("databento", mkt, sess), tag, a, b), a, b))
    mkt, sess = ECHO["mkt"], ECHO["session"]
    raw5, _ = app.load("databento", mkt)
    raw1, nbad = bars_1m(mkt)
    print(f"MNQ 1m bars: {len(raw1):,}, {nbad} bad prints repaired", flush=True)
    echo_truncation_test(raw5, raw1, app.base_cfg(mkt, sess))
    a, b = full_range(mkt)
    t1 = echo_trades(raw1)
    t5 = echo_trades(None)
    rows.append(metrics_row("ECHO (1m profile)", mkt, sess, ECHO["rr"], t1, a, b))
    rows.append(metrics_row("ECHO (5m profile, vp.py)", mkt, sess, ECHO["rr"], t5, a, b))
    print("\n(B) Databento Oct 30 2023 - Oct 30 2026, own market/session/R:R, 1 contract, 1 tick slippage, $2.50/side")
    print("net_2023 = Nov-Dec 2023; R_2024/R_2025/R_2026 = average R of trades entered that year (2026 = Jan-Oct)")
    print(pd.DataFrame(rows).to_string(index=False))
    save(pd.DataFrame(rows), "setups.csv")
    e = rows[-2]
    verdict = "PASS" if (e["R_2024"] or -1) > 0 and (e["R_2025"] or -1) > 0 else "FAIL"
    print(f"\nECHO rule (average R > 0 in both 2024 and 2025): 2024 {e['R_2024']:+.3f}, 2025 {e['R_2025']:+.3f} -> {verdict}")


def report_vp():
    rows = []
    for mkt in ("MNQ", "MGC"):
        for sess in ("new_york", "asia"):
            raw, _ = app.load("databento", mkt)
            cfg = app.base_cfg(mkt, sess)
            fams = vp.vp_families(raw, cfg, sess, mkt)
            futs = {}
            for name, params, _, build in fams:
                x, sg, atr, ex = build(1.0)
                futs[(name, params["va_pct"])] = app.rpool().submit(
                    app.task_runs, x[app.COLS], sg, atr, ex, False, cfg, [(rr, dict(app.NOKILL)) for rr in v.RRS])
            for (name, va), f in futs.items():
                for rr, t in zip(v.RRS, f.result()):
                    t = app.with_r(t, cfg); yr = entry_year(t)
                    R = lambda y: round(float(t.R[yr == y].mean()), 3) if (yr == y).any() else None
                    rows.append(dict(mkt=mkt, sess=app.SESS_LABEL[sess], family=name, va=va, rr=rr, trades=len(t),
                                     net=round(t.pnl.sum()), avg_R=round(float(t.R.mean()), 3) if len(t) else None,
                                     n_2024=int((yr == 2024).sum()), R_2024=R(2024), n_2025=int((yr == 2025).sum()), R_2025=R(2025),
                                     n_2026=int((yr == 2026).sum()), R_2026=R(2026)))
            print("vp grid done", mkt, sess, flush=True)
    G = pd.DataFrame(rows)
    G["both_pos"] = (G.R_2024 > 0) & (G.R_2025 > 0)
    print("\n(C) vp.py grid, Databento 3 years, 5m profile as vp.py builds it, 1 contract, default costs")
    print(G.to_string(index=False))
    save(G, "vp_grid.csv")
    print(f"\n{int(G.both_pos.sum())} of {len(G)} configs have average R > 0 in both 2024 and 2025"
          f"; {int((G.net > 0).sum())} have net > 0 over 3 years")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["overlap", "setups", "vp"]
    keys = [("databento", m, s) for m in ("MNQ", "MGC") for s in ("new_york", "asia")]
    if "overlap" in parts:
        keys += [("yahoo", m, s) for m in ("MNQ", "MGC") for s in ("new_york", "asia")]
    if {"overlap", "setups"} & set(parts):
        for k in keys:
            app.pack_future(k, front=False)
        for k in keys:
            p = app.PACKS[k].result()
            print("signals", k, "cached" if p.get("cached") else f"built in {p['seconds']:.0f}s", flush=True)
    if "overlap" in parts:
        report_overlap()
    if "setups" in parts:
        report_setups()
    if "vp" in parts:
        report_vp()
