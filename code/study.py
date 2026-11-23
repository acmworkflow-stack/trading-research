"""Trades for every setup in the indicator, with confidence-score dimensions and weekly bias
measured on the 5m bar the signal closed on (known before entry).

usage: python study.py <outputs_dir> <out_parquet>
"""
from __future__ import annotations
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import numpy as np, pandas as pd
import nq_backtest as bt
import validate as v

NOKILL = dict(drawdown_dollars=1e12)
SPEC = {"MNQ": dict(tick_size=0.25, point_value=2.0), "MGC": dict(tick_size=0.10, point_value=10.0)}
# (tag, group, market, session, family, params, rr)
SETUPS = [
    ("LIMA 5m", "old", "MNQ", "new_york", "Lima BB+RSI", dict(bb_mult=2.0, rsi=30.0), 1.0),
    ("MARCUS 15m", "old", "MNQ", "new_york", "FVG + order block", dict(tf=15, vwap=1), 3.5),
    ("MARCUS 30m", "old", "MNQ", "new_york", "FVG + order block", dict(tf=30, vwap=1), 3.0),
    ("NOVEMBER 5m", "old", "MNQ", "new_york", "November AVWAP", dict(tf=5, sigma=2.0), 2.0),
    ("NOVEMBER 10m", "old", "MNQ", "new_york", "November AVWAP", dict(tf=10, sigma=2.0), 2.0),
    ("NOVEMBER 15m", "old", "MNQ", "new_york", "November AVWAP", dict(tf=15, sigma=2.0), 2.0),
    ("OSCAR BREAK", "old", "MNQ", "new_york", "Oscar ORB", dict(orb=15, variant="breakout", tf=10), 3.0),
    ("OSCAR FAKEOUT", "old", "MNQ", "new_york", "Oscar ORB", dict(orb=15, variant="two_close_fakeout_filter", tf=5), 2.0),
    ("OSCAR RETEST", "old", "MNQ", "new_york", "Oscar ORB", dict(orb=15, variant="retest", tf=5), 3.0),
    ("PAPA", "old", "MNQ", "new_york", "PAPA30", {}, 1.0),
    ("GOLF", "new", "MGC", "new_york", "Oscar ORB", dict(orb=15, variant="retest", tf=10), 3.0),
    ("HOTEL", "new", "MGC", "new_york", "Lima BB+RSI", dict(bb_mult=2.0, rsi=25.0), 3.0),
    ("ALPHA", "new", "MGC", "asia", "FVG + order block", dict(tf=30, vwap=0), 3.0),
    ("KILO", "new", "MGC", "asia", "Lima BB+RSI", dict(bb_mult=2.0, rsi=25.0), 2.0),
    ("MIKE", "new", "MNQ", "new_york", "FVG + order block", dict(tf=30, vwap=0), 2.5),
    ("TANGO", "new", "MNQ", "asia", "FVG + order block", dict(tf=15, vwap=0), 2.0),
    ("SIERRA", "new", "MNQ", "asia", "FVG + BOS/CHOCH", dict(tf=15), 1.0),
    ("VICTOR", "new", "MNQ", "asia", "FVG 5/15/30 alignment", dict(variant="fvg_vwap"), 2.5),
]
RRS = [1.0, 1.5, 2.0, 2.5, 3.0]


def features(raw: pd.DataFrame) -> pd.DataFrame:
    """Five a-priori score dimensions per 5m bar, each LONG-oriented in [-1, 1]."""
    x = bt.indicators(raw)
    c, h, l, o, vol = x.close, x.high, x.low, x.open, x.volume.astype(float)
    f = pd.DataFrame(index=x.index)
    # [1] structure: EMA20/50 alignment + sweep-and-reclaim of the prior 20-bar extreme
    ema = np.sign(x.ema20 - x.ema50)
    lo20, hi20 = l.shift(4).rolling(20).min(), h.shift(4).rolling(20).max()
    swept_lo = (l.rolling(3).min() < lo20) & (c > lo20)
    swept_hi = (h.rolling(3).max() > hi20) & (c < hi20)
    f["structure"] = (ema + swept_lo.astype(int) - swept_hi.astype(int)).clip(-1, 1)
    # [2] momentum: RSI change over 3 bars + ADX(14) direction when trending
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    rma = lambda s: pd.Series(s, index=x.index).ewm(alpha=1/14, adjust=False).mean()
    atr = rma(tr); pdi = 100 * rma(pdm) / atr; mdi = 100 * rma(mdm) / atr
    adx = rma(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    adx_dir = np.where(adx > 20, np.sign(pdi - mdi), 0.0)
    f["momentum"] = ((np.sign(x.rsi - x.rsi.shift(3)) + adx_dir) / 2).clip(-1, 1)
    # [3] volume: participation vs 20-bar average (direction-neutral, so the same for both sides)
    rel = vol / vol.rolling(20).mean()
    f["volume_abs"] = np.select([rel >= 1.3, rel <= 0.7], [1.0, -1.0], 0.0)
    # [4] volatility: TTM-style squeeze (BB20,2 inside KC20,1.5) released in the trade's direction
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    squeeze = ((mid + 2 * sd) < (mid + 1.5 * atr)) & ((mid - 2 * sd) > (mid - 1.5 * atr))
    released = squeeze.shift(1).rolling(6).max().astype(bool) & ~squeeze
    f["volatility"] = np.where(released, np.sign(c - mid), 0.0)
    f["squeeze_on"] = squeeze.astype(int)
    # [5] pattern: engulfing / hammer / close location on the signal candle
    body, rng = (c - o).abs(), (h - l).replace(0, np.nan)
    bull_eng = (c > o) & (o.shift() > c.shift()) & (c >= o.shift()) & (o <= c.shift())
    bear_eng = (c < o) & (o.shift() < c.shift()) & (c <= o.shift()) & (o >= c.shift())
    hammer = ((np.minimum(o, c) - l) >= 2 * body) & ((c - l) / rng >= 2/3)
    star = ((h - np.maximum(o, c)) >= 2 * body) & ((h - c) / rng >= 2/3)
    loc = (c - l) / rng
    f["pattern"] = np.select([bull_eng | hammer, bear_eng | star, loc >= 2/3, loc <= 1/3], [1.0, -1.0, 0.5, -0.5], 0.0)
    # weekly bias: last COMPLETED futures week (Sun evening -> Fri close) up or down
    wk = raw.resample("W-SAT", label="right", closed="right").agg({"open": "first", "close": "last"}).dropna()
    wk_dir = pd.Series(np.sign(wk.close - wk.open).to_numpy(), index=wk.index)
    f["weekly"] = pd.Series(wk_dir.to_numpy(), index=wk.index).reindex(f.index, method="ffill").fillna(0)
    return f


def build(fams, family, params):
    return next(b for f, p, o, b in fams if f == family and p == params)


def main():
    out = Path(sys.argv[2]); rows = []
    for mkt in ("MNQ", "MGC"):
        for sess in ("new_york", "asia"):
            todo = [s for s in SETUPS if s[2] == mkt and s[3] == sess]
            if not todo: continue
            cfg = bt.Config(**SPEC[mkt], session=sess, contracts=1, slippage_ticks=1.0, commission_per_side=2.5)
            raw, _ = v.clean(bt.load_bars(Path(sys.argv[1]).parent / "data" / f"{mkt.lower()}_yahoo_5m.csv", cfg))
            fams = v.families(raw, cfg, sess); feat = features(raw)
            x5 = bt.indicators(raw)
            papa_sig, papa_tgt = v.papa_signal(x5, sess, 0.001)
            papa_tgt = papa_tgt.where(papa_tgt.isna(), 30.0)
            for tag, group, _, _, fam, params, rr0 in todo:
                for rr in sorted(set(RRS + [rr0])):
                    b = (lambda r: (x5, papa_sig, papa_tgt / r, v.PAPA[sess][1])) if fam == "PAPA30" else build(fams, fam, params)
                    t = v.run(b, rr, cfg, **NOKILL)
                    if not len(t): continue
                    t = t.copy()
                    t["tag"], t["group"], t["market"], t["session"], t["rr"], t["chosen_rr"] = tag, group, mkt, sess, rr, rr == rr0
                    t["risk_1c"] = t.risk_dollars + 2 * cfg.tick_size * cfg.point_value + 5.0
                    t["R"] = t.pnl / t.risk_1c
                    pos = feat.index.searchsorted(pd.to_datetime(t.entry_time), side="left") - 1
                    fb = feat.iloc[pos].reset_index(drop=True)
                    d = t.direction.to_numpy()
                    for k in ("structure", "momentum", "volatility", "pattern", "weekly"):
                        t[k] = fb[k].to_numpy() * d
                    t["volume"] = fb["volume_abs"].to_numpy()
                    t["squeeze_on"] = fb["squeeze_on"].to_numpy()
                    t["day"] = [v.cmp.session_id(pd.Timestamp(e), sess) or v.cmp.session_trade_key(pd.Timestamp(e), sess) for e in t.entry_time]
                    rows.append(t)
            print(mkt, sess, "done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    dims = ["structure", "momentum", "volume", "volatility", "pattern"]
    df["score"] = 50 + 10 * df[dims].sum(axis=1)
    df["score_wk"] = 50 + (50 / 6) * df[dims + ["weekly"]].sum(axis=1)
    df.to_pickle(out)
    print("saved", len(df), "trades")


if __name__ == "__main__":
    main()
