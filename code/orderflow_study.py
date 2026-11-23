"""Order-flow confirmation study on Databento TBBO (NQ, GC), Jul 31 - Nov 1 2026.

    python code/orderflow_study.py [--cuts N]      (N truncation-test cuts per market, default 10)

Writes nothing except data/cache/tbbo_<mkt>_front.pkl.zst: the front-month trades processed into
1-minute bars (OHLC, volume, buy/sell aggressor volume) and a 1-minute x tick footprint (buy/sell
volume at each price). It is rebuilt when the TBBO files or the processing code change. All
signals below need nothing finer than that. Results print to the console; code/app.py shows the
same tables in its "Order flow" tab.

Data prep
  * outright contracts only; front month per CME trading day (17:00-17:00 CT) = most volume on the
    previous trading day, never rolling back to an earlier expiry (the dashboard's rule), with the
    same additive back-adjustment at the roll
  * side B = buy aggressor, A = sell aggressor, N = unspecified (Databento docs); N volume counts
    in total volume but in neither buy nor sell. Checked against the pre-trade BBO in TBBO.
Rules fixed before any results were seen (where the brief left a choice)
  * "top X%" = vs the 1-minute bars (or 1-minute x tick footprint cells) of the trailing 10
    sessions of the same type (NY vs NY, Asia vs Asia); long needs delta >= P(100-X), short needs
    delta <= P(X). Sessions with fewer than 10 earlier sessions are skipped for every row,
    controls included, so all rows share the same touches.
  * "within 10 minutes" = the entry bar is the touch bar or one of the next 9 one-minute bars.
  * zone, ATR and stop are fixed at the touch: zone = level +/- 0.25 x ATR(14) of 5m bars built
    from these 1m bars, using the last 5m bar closed before the touch minute.
  * touch = first 1m bar whose range enters the zone while the previous 1m bar was entirely
    outside it; from above = support (long), from below = resistance (short); at most 2 per level
    per session. FVG level = the gap midpoint; bullish gaps take long touches only, bearish short
    only; a gap is dropped after a 1m bar trades fully through it.
  * value area: vp.py's method (bins NQ 2.0 pts, GC 0.5 pts, grow from the POC one bin at a time)
    on the prior NY session's true traded volume; POC = bin centre, VAH/VAL = outer prices of the
    value-area bins.
  * C1 rolling 3-min delta = touch bar + 2 bars before it; the shift bar comes after the touch.
  * C2: the breakout bar is in the window; the close back inside is 1-5 bars after it.
  * C4: absorption bar k with k + 5 still in the window; entry at the close of bar k + 5.
  * exits use the engine's convention: 1 tick slippage on every exit (stop, target, session
    end), stop fills at the bar open if it gaps through. R = P&L / (stop distance + 1 tick
    slippage, at micro size, + $5 round-trip commission), so a clean stop-out is exactly -1R.
    Entries on a session's last bar are skipped (nothing left to trade).
Confirmation count (same touches, zones, stops, targets, costs and percentiles)
  * score = how many of C1-C4 fire within the touch's window (0-4); entry at the close of the bar
    where the last of them fires; score 0 enters on B2. Gross R = same levels and path with no
    slippage or commission, over the raw stop distance; net R as above.
  * rule fixed before results: at 2R, Spearman(score, R) > 0 on NQ and on GC, and score >= 2 has
    average R > 0 with >= 30 trades on each market.
  * the 11 combinations (pairs, triples, all four; all must fire, entry at the last) are
    exploratory only, not a pass test.
"""
from __future__ import annotations
import sys, io, re, os, time, hashlib, inspect, pickle, argparse, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from compression import zstd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
TBBO = ROOT / "data" / "databento_tbbo"
CACHE = ROOT / "data" / "cache"
NY, CT = "America/New_York", "America/Chicago"
MONTHS = "FGHJKMNQUVXZ"
SPEC = {"NQ": dict(tick=0.25, pv=2.0, micro="MNQ", vbin_ticks=8),     # vp.py bins: 2.0 pts
        "GC": dict(tick=0.10, pv=10.0, micro="MGC", vbin_ticks=5)}    # 0.5 pts
COMM = 2.50
ZONE_ATR, STOP_ATR, TRAP_ATR, PUSH_ATR = 0.25, 0.5, 0.1, 0.25
WINDOW, MAX_TOUCHES, TRAIL, VA_PCT = 10, 2, 10, 0.70
CONFS = ["C1", "C2", "C3", "C4", "B1", "B2"]
TESTED = ["C1", "C2", "C3", "C4"]
TARGETS = (1, 2, 3)
LEVEL_TYPES = ["prior_ny_hl", "prior_ny_va", "prior_ny_poc", "asia_hl", "opening_range", "fvg_5m"]


# ------------------------------------------------------------------------------------ data prep
def _expiry(code: str) -> int:
    d = int(code[1])
    return (2020 + d if d >= 3 else 2030 + d) * 12 + MONTHS.index(code[0])


def _files(mkt: str) -> list:
    pat = re.compile(rf"\.tbbo\.({mkt}[{MONTHS}]\d)\.csv\.zst$")
    return sorted(f for f in TBBO.rglob("*.csv.zst") if pat.search(f.name))


def process_file(path: str, mkt: str) -> dict | None:
    """One outright TBBO file -> 1m bars, footprint cells, volume per trading day, side check."""
    tick = SPEC[mkt]["tick"]
    df = pd.read_csv(io.BytesIO(zstd.decompress(Path(path).read_bytes())),
                     usecols=["ts_event", "action", "side", "price", "size", "sequence", "bid_px_00", "ask_px_00", "symbol"],
                     dtype={"side": str, "action": str, "symbol": str})
    df = df[df.action == "T"]
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df.ts_event, utc=True, format="ISO8601")
    df = df.sort_values(["ts", "sequence"], kind="stable")
    px = np.round(df.price.to_numpy() / tick).astype(np.int64)
    ask = np.round(df.ask_px_00.to_numpy() / tick); bid = np.round(df.bid_px_00.to_numpy() / tick)
    size = df["size"].to_numpy(np.int64)
    side = df.side.to_numpy()
    g = pd.DataFrame({"minute": df.ts.dt.floor("min").to_numpy(),
                      "tday": (df.ts.dt.tz_convert(CT) + pd.Timedelta(hours=7)).dt.tz_localize(None).dt.normalize().to_numpy(),
                      "px": px, "size": size,
                      "buy": np.where(side == "B", size, 0), "sell": np.where(side == "A", size, 0)})
    bars = g.groupby("minute").agg(tday=("tday", "first"), open=("px", "first"), high=("px", "max"), low=("px", "min"),
                                   close=("px", "last"), volume=("size", "sum"), buy=("buy", "sum"), sell=("sell", "sum"))
    cells = g.groupby(["minute", "px"], sort=True)[["buy", "sell", "size"]].sum().reset_index().rename(columns={"size": "volume"})
    chk = pd.DataFrame({"tday": g.tday, "B": side == "B", "A": side == "A", "N": ~np.isin(side, ["A", "B"]),
                        "at_ask": px >= ask, "at_bid": px <= bid, "size": size})
    sanity = chk.groupby("tday").apply(lambda c: pd.Series({
        "B_n": c.B.sum(), "B_at_ask": (c.B & c.at_ask).sum(), "B_at_bid": (c.B & c.at_bid & ~c.at_ask).sum(),
        "A_n": c.A.sum(), "A_at_bid": (c.A & c.at_bid).sum(), "A_at_ask": (c.A & c.at_ask & ~c.at_bid).sum(),
        "N_n": c.N.sum(), "B_size": c["size"][c.B].sum(), "A_size": c["size"][c.A].sum(), "N_size": c["size"][c.N].sum()}),
        include_groups=False)
    sym = df.symbol.iloc[0]
    return dict(symbol=sym, bars=bars.assign(symbol=sym), cells=cells.assign(symbol=sym),
                vol=g.groupby("tday")["size"].sum(), sanity=sanity.assign(symbol=sym))


def build_front(mkt: str, workers: int | None = None, log=print) -> dict:
    files = _files(mkt)
    if not files:
        raise FileNotFoundError(f"no {mkt} outright TBBO files under {TBBO}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers or max(2, min(10, (os.cpu_count() or 4) - 2))) as ex:
        res = [r for r in ex.map(process_file, [str(f) for f in files], [mkt] * len(files)) if r]
    log(f"  {mkt}: parsed {len(files)} outright files in {time.time() - t0:.0f}s")
    vol = pd.concat([pd.DataFrame({"tday": r["vol"].index, "symbol": r["symbol"], "v": r["vol"].to_numpy()}) for r in res])
    vol = vol.groupby(["tday", "symbol"]).v.sum().unstack(fill_value=0).sort_index()
    exp = {s: _expiry(s[len(mkt):]) for s in vol.columns}
    by_exp = {e: s for s, e in exp.items()}
    lead = vol.idxmax(axis=1).shift(1)
    lead.iloc[0] = vol.iloc[0].idxmax()
    lead = lead.map(exp).cummax().map(by_exp)
    bars = pd.concat([r["bars"] for r in res]).sort_index()
    cells = pd.concat([r["cells"] for r in res], ignore_index=True)
    sanity = pd.concat([r["sanity"] for r in res])
    # roll gaps (ticks) on the last day of the old contract, last common minute
    gap = pd.Series(0, index=lead.index, dtype=np.int64)
    rolls = []
    for i in np.flatnonzero(lead.ne(lead.shift()).to_numpy())[1:]:
        prev, old, new = lead.index[i - 1], lead.iloc[i - 1], lead.iloc[i]
        a = bars[(bars.tday == prev) & (bars.symbol == old)].close
        b = bars[(bars.tday == prev) & (bars.symbol == new)].close
        common = a.index.intersection(b.index)
        if len(common):
            gap.iloc[i] = int(b[common[-1]] - a[common[-1]])
        else:
            nb = bars[(bars.tday == lead.index[i]) & (bars.symbol == new)].close
            gap.iloc[i] = int(nb.iloc[0] - a.iloc[-1])
        rolls.append((str(lead.index[i].date()), old, new, gap.iloc[i] * SPEC[mkt]["tick"]))
    adj = (gap[::-1].cumsum()[::-1] - gap)
    keep = bars.symbol.to_numpy() == bars.tday.map(lead).to_numpy()
    fb = bars[keep].copy()
    fb = fb[~fb.index.duplicated()]
    shift = fb.tday.map(adj).to_numpy()
    for k in ("open", "high", "low", "close"):
        fb[k] = fb[k] + shift
    mins = fb.index
    fc = cells[cells.minute.isin(mins)]
    fc = fc[fc.symbol.to_numpy() == fb.symbol.reindex(fc.minute).to_numpy()].copy()
    fc["px"] = fc.px.to_numpy() + fb.tday.map(adj).reindex(fc.minute).to_numpy()
    sel = sanity[sanity.symbol.to_numpy() == pd.Series(sanity.index).map(lead).to_numpy()]
    return dict(bars=fb.drop(columns="symbol"), cells=fc[["minute", "px", "buy", "sell", "volume"]].reset_index(drop=True),
                lead=lead, rolls=rolls, sanity=sel.drop(columns="symbol").sum())


def cache_key(mkt: str) -> str:
    h = hashlib.sha256("".join(inspect.getsource(f) for f in (process_file, build_front, _expiry)).encode())
    for f in _files(mkt):
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def load_front(mkt: str, log=print) -> dict:
    p = CACHE / f"tbbo_{mkt.lower()}_front.pkl.zst"
    key = cache_key(mkt)
    if p.exists():
        try:
            d = pickle.loads(zstd.decompress(p.read_bytes()))     # our own file
            if d.get("key") == key:
                return d
        except Exception as exc:
            log(f"  ignoring unreadable {p.name}: {exc!r}")
    log(f"building {p.name} from the TBBO files (one time) ...")
    d = dict(build_front(mkt, log=log), key=key)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(zstd.compress(pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL)))
    tmp.replace(p)
    return d


# -------------------------------------------------------------------------------- preparation
def session_keys(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    """Session date per bar (NaT outside): NY 09:30-16:00 ET; Asia 19:00-03:00 ET keyed by its evening."""
    local = index.tz_convert(NY)
    minute = np.asarray(local.hour * 60 + local.minute)
    day = local.tz_localize(None).normalize()
    if session == "new_york":
        keep = (minute >= 570) & (minute < 960)
    else:
        morning = minute < 180
        keep = (minute >= 1140) | morning
        day = day - pd.to_timedelta(morning.astype(int), unit="D")
    out = day.to_numpy().copy()
    out[~keep] = np.datetime64("NaT", "ns")
    return out


def prepare(bars: pd.DataFrame, cells: pd.DataFrame, mkt: str) -> dict:
    b = bars.copy()
    b.index = b.index.tz_convert(NY)
    O, H, L, C = (b[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    D = (b.buy - b.sell).to_numpy(float)
    f = b[["open", "high", "low", "close"]].resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev = f.close.shift()
    tr = pd.concat([f.high - f.low, (f.high - prev).abs(), (f.low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()      # nq_backtest.indicators' ATR
    atr.index = atr.index + pd.Timedelta(minutes=5)                      # known once the 5m bar closes
    A = atr.reindex(b.index.union(atr.index)).ffill().reindex(b.index).to_numpy()
    pos = b.index.get_indexer(pd.DatetimeIndex(cells.minute).tz_convert(NY))
    order = np.argsort(pos, kind="stable")
    cpos = pos[order]
    sessions = {}
    for s in ("new_york", "asia"):
        k = session_keys(b.index, s)
        idx = np.flatnonzero(~np.isnat(k))
        brk = np.flatnonzero(k[idx][1:] != k[idx][:-1]) + 1
        sessions[s] = [(k[g[0]], int(g[0]), int(g[-1])) for g in np.split(idx, brk) if len(g)]
    return dict(mkt=mkt, idx=b.index, O=O, H=H, L=L, C=C, D=D, A=A,
                cpx=cells.px.to_numpy()[order], cbuy=cells.buy.to_numpy(float)[order],
                csell=cells.sell.to_numpy(float)[order], cvol=cells.volume.to_numpy(float)[order],
                off=np.searchsorted(cpos, np.arange(len(b) + 1)), sessions=sessions,
                mins=np.asarray(b.index.hour * 60 + b.index.minute))


def value_area(P, q0, q1):
    bt = SPEC[P["mkt"]]["vbin_ticks"]
    a, z = P["off"][q0], P["off"][q1 + 1]
    if z <= a:
        return None
    bins = np.floor_divide(P["cpx"][a:z].astype(np.int64), bt)
    lo = bins.min()
    hist = np.bincount(bins - lo, weights=P["cvol"][a:z])
    p = int(hist.argmax()); i = j = p; tot = hist[p]
    while tot < VA_PCT * hist.sum() and (i > 0 or j < len(hist) - 1):      # vp.levels' expansion
        up = hist[j + 1] if j < len(hist) - 1 else -1
        dn = hist[i - 1] if i > 0 else -1
        if up >= dn: j += 1; tot += up
        else: i -= 1; tot += dn
    return ((lo + j + 1) * bt - 1, (lo + i) * bt, (lo + p) * bt + (bt - 1) / 2)   # VAH, VAL, POC (ticks)


def session_levels(P, s, i):
    key, p0, p1 = P["sessions"][s][i]
    H, L, mins, idx = P["H"], P["L"], P["mins"], P["idx"]
    lv = []

    def add(t, price, allow="both", active=p0, gap=None):
        lv.append(dict(type=t, price=float(price), allow=allow, active=active, gap=gap,
                       id=f"{t}@{price:g}" + (f"#{idx[active]:%H%M}" if gap else "")))
    ny = P["sessions"]["new_york"]
    prior = [x for x in ny if (x[0] < key if s == "new_york" else x[0] <= key)][-1:]
    if prior:
        _, q0, q1 = prior[0]
        add("prior_ny_hl", H[q0:q1 + 1].max()); add("prior_ny_hl", L[q0:q1 + 1].min())
        va = value_area(P, q0, q1)
        if va:
            add("prior_ny_va", va[0]); add("prior_ny_va", va[1]); add("prior_ny_poc", va[2])
    if s == "new_york":
        asia = [x for x in P["sessions"]["asia"] if x[0] < key][-1:]
        if asia:
            _, q0, q1 = asia[0]
            add("asia_hl", H[q0:q1 + 1].max()); add("asia_hl", L[q0:q1 + 1].min())
        orb = np.flatnonzero(mins[p0:p1 + 1] < 585) + p0                 # 09:30-09:44
        after = np.flatnonzero(mins[p0:p1 + 1] >= 585)
        if len(orb) and len(after):
            act = int(after[0] + p0)
            add("opening_range", H[orb].max(), active=act); add("opening_range", L[orb].min(), active=act)
    # intraday 5m FVGs from this session's 1m bars
    t = idx[p0:p1 + 1]
    g = pd.DataFrame({"h": H[p0:p1 + 1], "l": L[p0:p1 + 1]}, index=t).resample("5min", label="left", closed="left").agg(
        {"h": "max", "l": "min"}).dropna()
    gt, gh, gl = g.index, g.h.to_numpy(), g.l.to_numpy()
    for k in range(2, len(g)):
        if gt[k] - gt[k - 2] != pd.Timedelta(minutes=10):
            continue
        ready = gt[k] + pd.Timedelta(minutes=5)
        act = p0 + int(np.searchsorted(t, ready))
        if act > p1:
            continue
        if gl[k] > gh[k - 2]:
            add("fvg_5m", (gl[k] + gh[k - 2]) / 2, "long", act, (gh[k - 2], gl[k]))
        elif gh[k] < gl[k - 2]:
            add("fvg_5m", (gh[k] + gl[k - 2]) / 2, "short", act, (gh[k], gl[k - 2]))
    return lv


def touches(P, lv, p1):
    H, L, A = P["H"], P["L"], P["A"]
    out, x = [], lv["price"]
    for pos in range(max(lv["active"], 1), p1 + 1):
        a, pa = A[pos], A[pos - 1]
        if np.isfinite(a) and np.isfinite(pa):
            zlo, zhi = x - ZONE_ATR * a, x + ZONE_ATR * a
            above, below = L[pos - 1] > x + ZONE_ATR * pa, H[pos - 1] < x - ZONE_ATR * pa
            if L[pos] <= zhi and H[pos] >= zlo and (above or below):
                d = 1 if above else -1
                if lv["allow"] == "both" or (lv["allow"] == "long") == (d > 0):
                    out.append((pos, d, zlo, zhi, a))
                    if len(out) == MAX_TOUCHES:
                        break
        if lv["gap"] and (L[pos] < lv["gap"][0] if lv["allow"] == "long" else H[pos] > lv["gap"][1]):
            break                                                       # traded fully through
    return out


def thresholds(P, prev):
    d = np.concatenate([P["D"][a:b + 1] for _, a, b in prev])
    cv = np.concatenate([P["cvol"][P["off"][a]:P["off"][b + 1]] for _, a, b in prev])
    return dict(p80=np.percentile(d, 80), p20=np.percentile(d, 20), p90=np.percentile(d, 90),
                p10=np.percentile(d, 10), c95=np.percentile(cv, 95) if len(cv) else np.inf)


def confirm(P, conf, t0, d, zlo, zhi, a, x, th, p1):
    C, D, H, L = P["C"], P["D"], P["H"], P["L"]
    last = min(t0 + WINDOW - 1, p1)
    long = d > 0
    if conf == "B1":
        return t0
    if conf == "B2":
        for k in range(t0, last + 1):
            if (C[k] > zhi) if long else (C[k] < zlo):
                return k
        return None
    if conf == "C1":
        d3 = D[max(t0 - 2, 0):t0 + 1].sum()
        if not ((d3 < 0) if long else (d3 > 0)):
            return None
        for k in range(t0 + 1, last + 1):
            if ((D[k] >= th["p80"]) if long else (D[k] <= th["p20"])) and zlo <= C[k] <= zhi:
                return k
        return None
    if conf == "C2":
        for k in range(t0, last + 1):
            past = (L[k] <= x - TRAP_ATR * a) if long else (H[k] >= x + TRAP_ATR * a)
            against = (D[k] <= th["p20"]) if long else (D[k] >= th["p80"])
            if past and against:
                for j in range(k + 1, min(k + 5, last) + 1):
                    if zlo <= C[j] <= zhi:
                        return j
        return None
    if conf == "C3":
        for k in range(t0, last + 1):
            if (D[k] >= th["p90"] and C[k] >= x + PUSH_ATR * a) if long else (D[k] <= th["p10"] and C[k] <= x - PUSH_ATR * a):
                return k
        return None
    if conf == "C4":
        off, cpx, cvol = P["off"], P["cpx"], P["cvol"]
        hit = P["csell"] if long else P["cbuy"]
        for k in range(t0, last - 5 + 1):
            for c in range(off[k], off[k + 1]):
                p = cpx[c]
                if zlo <= p <= zhi and cvol[c] >= th["c95"] and hit[c] >= 0.65 * cvol[c]:
                    held = (L[k + 1:k + 6].min() >= p - 2) if long else (H[k + 1:k + 6].max() <= p + 2)
                    if held:
                        return k + 5
        return None
    raise ValueError(conf)


def signals(P) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All touches and every confirmation's entry, each knowable at its bar's close."""
    T, E = [], []
    for s in ("new_york", "asia"):
        S = P["sessions"][s]
        for i in range(TRAIL, len(S)):
            key, p0, p1 = S[i]
            th = thresholds(P, S[i - TRAIL:i])
            for lv in session_levels(P, s, i):
                for (t0, d, zlo, zhi, a) in touches(P, lv, p1):
                    ks = {conf: confirm(P, conf, t0, d, zlo, zhi, a, lv["price"], th, p1) for conf in CONFS}
                    T.append(dict(session=s, skey=key, level_type=lv["type"], level=lv["id"], touch=t0, dir=d,
                                  zlo=zlo, zhi=zhi, atr=a, end=p1,
                                  **{f"k_{c}": (np.nan if ks[c] is None else ks[c]) for c in CONFS}))
                    for conf in CONFS:
                        k = ks[conf]
                        if k is None or k >= p1:
                            continue
                        entry = P["C"][k] + d                                  # + 1 tick slippage
                        stop = (zlo - STOP_ATR * a) if d > 0 else (zhi + STOP_ATR * a)
                        if (entry - stop) * d <= 0:
                            continue
                        E.append(dict(session=s, skey=key, level_type=lv["type"], level=lv["id"], conf=conf,
                                      touch=t0, entry_pos=k, end=p1, dir=d, entry=entry, stop=round(stop, 9)))
    return pd.DataFrame(T), pd.DataFrame(E)


def simulate(P, E: pd.DataFrame) -> pd.DataFrame:
    spec = SPEC[P["mkt"]]
    usd = spec["tick"] * spec["pv"]
    O, H, L, C = P["O"], P["H"], P["L"], P["C"]
    out = {f"{c}{k}": [] for k in TARGETS for c in ("R", "pnl", "gR")}
    for e, end, d, entry, stop in zip(E.entry_pos, E.end, E.dir, E.entry, E.stop):
        risk = (entry - stop) * d
        raw_entry = entry - d                                   # the bar close, before slippage
        raw_risk = (raw_entry - stop) * d
        for k in TARGETS:
            tgt = entry + d * k * risk
            raw = C[end]
            for j in range(e + 1, end + 1):
                if (L[j] <= stop) if d > 0 else (H[j] >= stop):
                    raw = min(stop, O[j]) if d > 0 else max(stop, O[j])
                    break
                if (H[j] >= tgt) if d > 0 else (L[j] <= tgt):
                    raw = tgt
                    break
            pnl = (raw - d - entry) * d * usd - 2 * COMM               # 1 tick exit slippage + commission
            out[f"pnl{k}"].append(pnl)
            out[f"R{k}"].append(pnl / ((risk + 1) * usd + 2 * COMM))
            # gross: same levels and path, no slippage or commission, over the raw stop distance
            out[f"gR{k}"].append((raw - raw_entry) * d / raw_risk if raw_risk > 0 else np.nan)
    return E.assign(**out)


# ------------------------------------------------------- confirmation count and combinations
def entries_at(P, T: pd.DataFrame, k) -> pd.DataFrame:
    """Entries for touches T at bar positions k (NaN = no entry), with the touch's zone and stop."""
    k = np.asarray(k, float)
    ok = np.isfinite(k)
    ok[ok] = k[ok] < T.end.to_numpy()[ok]                      # no entry on the session's last bar
    E = T[ok].copy()
    E["entry_pos"] = k[ok].astype(int)
    d = E.dir.to_numpy()
    E["entry"] = P["C"][E.entry_pos.to_numpy()] + d                 # + 1 tick slippage
    E["stop"] = np.where(d > 0, E.zlo - STOP_ATR * E.atr, E.zhi + STOP_ATR * E.atr).round(9)
    return E[(E.entry - E.stop) * d > 0]


def score_trades(P, T: pd.DataFrame) -> pd.DataFrame:
    """Score = how many of C1-C4 fire in the touch's window; entry at the last one's bar close;
    score 0 enters on B2 (price rejection)."""
    K = T[[f"k_{c}" for c in TESTED]].to_numpy(float)
    score = np.isfinite(K).sum(axis=1)
    last = np.where(score > 0, np.nanmax(np.where(np.isfinite(K), K, -1), axis=1), T.k_B2.to_numpy(float))
    return simulate(P, entries_at(P, T.assign(score=score), last))


COMBOS = [c for r in (2, 3, 4) for c in itertools.combinations(TESTED, r)]


def combo_trades(P, T: pd.DataFrame) -> pd.DataFrame:
    """Every pair/triple/all four of C1-C4: all must fire in the window; entry at the last one."""
    out = []
    for combo in COMBOS:
        K = T[[f"k_{c}" for c in combo]].to_numpy(float)
        k = np.where(np.isfinite(K).all(axis=1), K.max(axis=1), np.nan)
        out.append(simulate(P, entries_at(P, T, k)).assign(combo="+".join(combo)))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------------ checks
def truncation_test(bars, cells, mkt, ncuts=10, log=print) -> bool:
    """lookahead.py's test: touches and entries on bars closed before a cut must not change when
    every later bar (and footprint cell) is removed."""
    def keyset(P, T, E, before):
        idx = P["idx"]
        t = {(r.session, r.level, idx[r.touch], r.dir) for r in T.itertuples() if r.touch < before}
        e = {(r.session, r.level, r.conf, idx[r.touch], idx[r.entry_pos], r.dir, r.entry, r.stop)
             for r in E.itertuples() if r.entry_pos < before}
        return t, e
    P = prepare(bars, cells, mkt)
    T, E = signals(P)
    rng = np.random.default_rng(0)
    bad = 0
    cuts = sorted(rng.integers(len(bars) // 4, len(bars) - 10, ncuts))
    for cut in cuts:
        t = bars.index[cut]
        Q = prepare(bars[bars.index < t], cells[cells.minute < t], mkt)
        Tq, Eq = signals(Q)
        last = len(Q["idx"]) - 1                                # compare bars closed before the last one
        cutoff = Q["idx"][last]
        full = keyset(P, T, E, P["idx"].searchsorted(cutoff))
        part = keyset(Q, Tq, Eq, last)
        bad += full != part
    log(f"  {mkt} truncation test: {'LEAK' if bad else 'ok'} ({bad}/{len(cuts)} cuts differ; "
        f"{len(E)} entries, {len(T)} touches compared up to each cut)")
    return bad == 0


# ----------------------------------------------------------------------------------- reporting
def agg(g, touches):
    r = dict(touches=touches, trades=len(g))
    for k in TARGETS:
        r[f"win{k}%"] = round(100 * float((g[f"pnl{k}"] > 0).mean()), 1) if len(g) else None
    for k in TARGETS:
        r[f"avgR{k}"] = round(float(g[f"R{k}"].mean()), 3) if len(g) else None
    for k in TARGETS:
        r[f"net{k}$"] = round(float(g[f"pnl{k}"].sum()))
    return r


def tables(trades: pd.DataFrame, tch: pd.DataFrame) -> dict:
    months = sorted(trades.month.unique())

    def block(keys):
        rows = []
        tc = tch.groupby(keys).size() if keys else pd.Series({(): len(tch)})
        for gk, g in trades.groupby(keys + ["conf"], sort=False):
            gk = gk if isinstance(gk, tuple) else (gk,)
            n = tc.get(gk[:-1] if len(keys) > 1 else gk[0], 0)
            row = dict(zip(keys + ["conf"], gk), **agg(g, int(n)))
            for m in months:
                gm = g[g.month == m]
                row[f"R2 {m}"] = round(float(gm.R2.mean()), 3) if len(gm) else None
            rows.append(row)
        df = pd.DataFrame(rows)
        df["conf"] = pd.Categorical(df.conf, CONFS)
        return df.sort_values(keys + ["conf"]).reset_index(drop=True)
    detail = block(["market", "session", "level_type"])
    pooled = block(["market", "session"])
    overall = block(["market"])
    return dict(detail=detail, pooled=pooled, overall=overall, rule=rule_table(trades))


def rule_table(T: pd.DataFrame) -> pd.DataFrame:
    """Fixed in advance: at 2R, avg R > 0 on both NQ and GC, >= 30 trades each, and >= 0.1R better
    than B2 on the same touches' subset. Tested overall, per session, and per session x level type."""
    combos = [("all", "all")] + [(s, "all") for s in ("new_york", "asia")] + \
             [(s, lt) for s in ("new_york", "asia") for lt in LEVEL_TYPES if ((T.session == s) & (T.level_type == lt)).any()]
    rows = []
    for s, lt in combos:
        sub = T if s == "all" else T[T.session == s]
        sub = sub if lt == "all" else sub[sub.level_type == lt]
        for conf in TESTED:
            row, ok = dict(session=s, level_type=lt, conf=conf), True
            for m in ("NQ", "GC"):
                c = sub[(sub.market == m) & (sub.conf == conf)].R2
                b = sub[(sub.market == m) & (sub.conf == "B2")].R2
                rc, rb = (float(c.mean()) if len(c) else np.nan), (float(b.mean()) if len(b) else np.nan)
                row.update({f"{m} trades": len(c), f"{m} avgR2": round(rc, 3), f"{m} B2 avgR2": round(rb, 3)})
                ok &= len(c) >= 30 and rc > 0 and (rc - rb) >= 0.1
            row["meets rule"] = "YES" if ok else "no"
            rows.append(row)
    return pd.DataFrame(rows)


def score_tables(S: pd.DataFrame, T: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    months = sorted(S.month.unique())

    def block(keys):
        rows = []
        tc = T.groupby(keys + ["score"]).size()
        for gk, g in S.groupby(keys + ["score"]):
            r = dict(zip(keys + ["score"], gk), touches=int(tc.get(gk, 0)), trades=len(g))
            for k in TARGETS:
                r[f"win{k}%"] = round(100 * float((g[f"pnl{k}"] > 0).mean()), 1)
            for k in TARGETS:
                r[f"avgR{k} net"] = round(float(g[f"R{k}"].mean()), 3)
            for k in TARGETS:
                r[f"avgR{k} gross"] = round(float(g[f"gR{k}"].mean()), 3)
            for k in TARGETS:
                r[f"net{k}$"] = round(float(g[f"pnl{k}"].sum()))
            for m in months:
                gm = g[g.month == m]
                r[f"R2 {m}"] = round(float(gm.R2.mean()), 3) if len(gm) else None
            rows.append(r)
        return pd.DataFrame(rows)
    rule, ok_all = [], True
    for m in ("NQ", "GC"):
        g = S[S.market == m]
        rho, p = spearmanr(g.score, g.R2)
        hi = g[g.score >= 2]
        ok = bool(rho > 0) and len(hi) >= 30 and float(hi.R2.mean()) > 0
        ok_all &= ok
        rule.append(dict(market=m, **{f"avgR2 score {s}": (round(float(g.R2[g.score == s].mean()), 3) if (g.score == s).any() else None)
                                      for s in range(5)},
                         spearman_rho=round(float(rho), 3), p_value=round(float(p), 3),
                         **{"score>=2 trades": len(hi), "score>=2 avgR2": round(float(hi.R2.mean()), 3) if len(hi) else None},
                         market_ok="yes" if ok else "no"))
    rule = pd.DataFrame(rule)
    rule["meets rule"] = "YES" if ok_all else "no"
    return dict(score=block(["market"]), score_session=block(["market", "session"]), score_rule=rule)


def combo_table(K: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo in ["+".join(c) for c in COMBOS]:
        r = dict(combo=combo)
        for m in ("NQ", "GC"):
            g = K[(K.combo == combo) & (K.market == m)]
            r.update({f"{m} trades": len(g),
                      f"{m} avgR2 net": round(float(g.R2.mean()), 3) if len(g) else None,
                      f"{m} avgR2 gross": round(float(g.gR2.mean()), 3) if len(g) else None,
                      f"{m} win2%": round(100 * float((g.pnl2 > 0).mean()), 1) if len(g) else None})
        rows.append(r)
    return pd.DataFrame(rows)


def sanity_table(d: dict, mkt: str) -> dict:
    s = d["sanity"]
    return {"market": mkt, "buy-aggr trades": int(s.B_n), "buy at/above ask %": round(100 * s.B_at_ask / s.B_n, 2),
            "buy at/below bid %": round(100 * s.B_at_bid / s.B_n, 2), "sell-aggr trades": int(s.A_n),
            "sell at/below bid %": round(100 * s.A_at_bid / s.A_n, 2), "sell at/above ask %": round(100 * s.A_at_ask / s.A_n, 2),
            "no-side trades %": round(100 * s.N_n / (s.B_n + s.A_n + s.N_n), 3),
            "buy share of volume %": round(100 * s.B_size / (s.B_size + s.A_size + s.N_size), 1)}


def run_all(markets=("NQ", "GC"), cuts=0, log=print) -> dict:
    trades, tch, sanity, info, trunc, scores, combos = [], [], [], [], {}, [], []
    for mkt in markets:
        d = load_front(mkt, log=log)
        P = prepare(d["bars"], d["cells"], mkt)
        T, E = signals(P)
        T["score"] = np.isfinite(T[[f"k_{c}" for c in TESTED]].to_numpy(float)).sum(axis=1)
        E = simulate(P, E)
        S, K = score_trades(P, T), combo_trades(P, T)
        for X in (E, S, K):
            X["entry_time"] = P["idx"][X.entry_pos]
            X["month"] = X.entry_time.dt.strftime("%Y-%m")
        trades.append(E.assign(market=mkt)); tch.append(T.assign(market=mkt))
        scores.append(S.assign(market=mkt)); combos.append(K.assign(market=mkt))
        sanity.append(sanity_table(d, mkt))
        fronts = d["lead"].loc[d["bars"].tday.min():d["bars"].tday.max()]
        info.append(f"{mkt}: {len(d['bars']):,} 1m bars, {len(d['cells']):,} footprint cells, "
                    f"{d['bars'].index[0].tz_convert(NY):%Y-%m-%d} to {d['bars'].index[-1].tz_convert(NY):%Y-%m-%d}; "
                    f"front {', '.join(dict.fromkeys(fronts))}; rolls {d['rolls'] or 'none'}; "
                    f"sessions NY {len(P['sessions']['new_york'])}, Asia {len(P['sessions']['asia'])} "
                    f"(first {TRAIL} of each are history for the percentiles)")
        if cuts:
            trunc[mkt] = truncation_test(d["bars"], d["cells"], mkt, cuts, log=log)
    trades = pd.concat(trades, ignore_index=True)
    tch = pd.concat(tch, ignore_index=True)
    out = tables(trades, tch)
    out.update(score_tables(pd.concat(scores, ignore_index=True), tch))
    out["combos_table"] = combo_table(pd.concat(combos, ignore_index=True))
    out.update(sanity=pd.DataFrame(sanity), info=info, trunc=trunc, trades=trades, touches=tch,
               combos=len(out["rule"]),
               configs=len(markets) * 2 * len(TARGETS) * len(CONFS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, default=10)
    args = ap.parse_args()
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    R = run_all(cuts=args.cuts)
    print("\n".join(R["info"]))
    print("\nAggressor side vs the pre-trade BBO (front month, all trades):")
    print(R["sanity"].to_string(index=False))
    print("\n(1) market x session x level type x confirmation  (R2 by month = average R at 2R by entry month)")
    print(R["detail"].to_string(index=False))
    print("\n(2) pooled across level types")
    print(R["pooled"].to_string(index=False))
    print("\n(3) pooled across sessions and level types")
    print(R["overall"].to_string(index=False))
    print("\n(4) rule fixed in advance: at 2R, avg R > 0 on NQ and GC, >= 30 trades each, >= 0.1R better than B2")
    print(R["rule"].to_string(index=False))
    n_yes = (R["rule"]["meets rule"] == "YES").sum()
    print(f"\n{R['combos']} rule tests (4 confirmations x {R['combos'] // 4} session/level groupings); {n_yes} meet the rule. "
          f"Thresholds were fixed before running; nothing was tuned. Targets 1/2/3R are reported, the rule uses 2R only.")
    print("\n(5) confirmation count: score = how many of C1-C4 fire within the touch's 10-minute window; entry at the "
          "close of the bar where the last one fires; score 0 enters on B2. Pooled across sessions and level types.")
    print("    gross = same entry/stop/target levels, no slippage or commission; net = as in every other table")
    print(R["score"].to_string(index=False))
    print("\n(6) confirmation count by session")
    print(R["score_session"].to_string(index=False))
    print("\n(7) count rule fixed in advance: at 2R, Spearman(score, R) > 0 on NQ and on GC, and score >= 2 has "
          "average R > 0 with >= 30 trades on each market")
    print(R["score_rule"].to_string(index=False))
    print(f"\n(8) EXPLORATORY: NOT A PASS TEST. All {len(COMBOS)} combinations of C1-C4 (6 pairs, 4 triples, all four); "
          "every confirmation in the combination must fire in the window, entry at the last one's bar close. "
          "Pooled across sessions and level types.")
    print(R["combos_table"].to_string(index=False))


if __name__ == "__main__":
    main()
