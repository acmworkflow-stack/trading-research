"""Volume-profile strategies on the validate.py harness.

Profile = fixed range over the last completed New York cash session (09:30-16:00 ET), built from
5m bars by spreading each bar's volume evenly over its high-low range. POC = busiest price,
value area = the va_pct of volume grown out from the POC. NY trades use yesterday's profile,
Asia trades use the session that just closed.

  VP 80% rule      open outside value, two closes back inside  -> trade toward the far edge
  VP edge reject   wick through VAH/VAL from inside, close back inside -> fade toward POC
  VP acceptance    two closes outside VAH/VAL after being inside -> trade the breakout

usage: python vp.py <outputs_dir> <MGC|MNQ> <new_york|asia> <out_dir>
"""
from __future__ import annotations
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import nq_backtest as bt
import validate as v

NY = "America/New_York"
BIN = {"MNQ": 2.0, "MGC": 0.5}


def levels(raw: pd.DataFrame, bin_size: float, va_pct: float) -> pd.DataFrame:
    loc = raw.index.tz_convert(NY); m = loc.hour * 60 + loc.minute
    rth = raw[(m >= 570) & (m < 960)]
    out = []
    for d, s in rth.groupby(rth.index.tz_convert(NY).date):
        lo, hi = np.floor(s.low.min() / bin_size), np.ceil(s.high.max() / bin_size)
        edges = np.arange(lo, hi + 1) * bin_size
        hist = np.zeros(len(edges))
        for l, h, vol in zip(s.low.to_numpy(), s.high.to_numpy(), s.volume.to_numpy(float)):
            a, b = int(np.floor(l / bin_size) - lo), int(np.ceil(h / bin_size) - lo)
            hist[a:b + 1] += vol / (b - a + 1)
        if hist.sum() <= 0: continue
        p = int(hist.argmax()); i = j = p; tot = hist[p]
        while tot < va_pct * hist.sum() and (i > 0 or j < len(hist) - 1):
            up = hist[j + 1] if j < len(hist) - 1 else -1; dn = hist[i - 1] if i > 0 else -1
            if up >= dn: j += 1; tot += up
            else: i -= 1; tot += dn
        out.append(dict(ready=pd.Timestamp(f"{d} 16:00", tz=NY), poc=edges[p], vah=edges[j], val=edges[i]))
    lv = pd.DataFrame(out).set_index("ready").tz_convert(raw.index.tz)
    # each bar sees the most recent profile whose session has fully closed
    return lv.reindex(raw.index.union(lv.index)).ffill().reindex(raw.index)


def vp_signals(x: pd.DataFrame, lv: pd.DataFrame, cfg, session: str, kind: str) -> pd.Series:
    c, h, l, o = x.close, x.high, x.low, x.open
    vah, val = lv.vah, lv.val
    inside = (c > val) & (c < vah)
    active = bt.in_session(x.index, cfg)
    keys = v.cmp.session_keys(x.index, session)
    sig = pd.Series(0, index=x.index, dtype="int8")
    if kind == "edge_reject":
        long = (l <= val) & (c > val) & (o > val)
        short = (h >= vah) & (c < vah) & (o < vah)
        return pd.Series(np.select([long, short], [1, -1], 0), index=x.index, dtype="int8").where(active, 0)
    for key in keys.dropna().unique():
        idx = x.index[keys.eq(key).to_numpy()]
        if len(idx) < 3 or not np.isfinite(vah.loc[idx[0]]): continue
        cc, va_h, va_l = c.loc[idx].to_numpy(), vah.loc[idx].to_numpy(), val.loc[idx].to_numpy()
        if kind == "eighty":
            op = o.loc[idx[0]]
            side = -1 if op > va_h[0] else 1 if op < va_l[0] else 0   # opened above -> short back through
            if not side: continue
            run = 0
            for k in range(len(idx)):
                run = run + 1 if va_l[k] < cc[k] < va_h[k] else 0
                if run >= 2: sig.loc[idx[k]] = side; break
        else:  # acceptance outside value
            was_inside, up, dn, used = False, 0, 0, set()
            for k in range(len(idx)):
                if va_l[k] < cc[k] < va_h[k]: was_inside, up, dn = True, 0, 0; continue
                if not was_inside: continue
                up = up + 1 if cc[k] > va_h[k] else 0
                dn = dn + 1 if cc[k] < va_l[k] else 0
                d = 1 if up >= 2 else -1 if dn >= 2 else 0
                if d and d not in used: sig.loc[idx[k]] = d; used.add(d)
    return sig.where(active, 0)


def vp_families(raw, cfg, session, mkt):
    x5 = bt.indicators(raw); ex = v.EXIT[session]; out = []
    for va in (0.6, 0.7, 0.8):
        lv = levels(raw, BIN[mkt], va)
        for kind, name in (("eighty", "VP 80% rule"), ("edge_reject", "VP edge reject"), ("acceptance", "VP acceptance")):
            sg = vp_signals(x5, lv, cfg, session, kind)
            out.append((name, dict(va_pct=va), ["va_pct"], lambda rr, sg=sg: (x5, sg, None, ex)))
    return out


if __name__ == "__main__":
    mkt = sys.argv[2]
    v.families = lambda raw, cfg, session: vp_families(raw, cfg, session, mkt)
    v.main()
