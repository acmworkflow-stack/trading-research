"""Local strategy dashboard (Dash + Plotly).

    python code/app.py      then open http://127.0.0.1:8050

Results stay in memory. The only files this app writes, all in data/cache/:
  * <mkt>_databento_5m_<hash>.csv  one continuous 5m price series per market, built from the
    Databento 1m files; the hash covers those files and build_continuous
  * signals_<source>_<mkt>_<session>.pkl.zst  every family's signals for one market/session,
    reused while the price file and the signal code (engine modules + family_plan etc.) are unchanged
plus a PNG/CSV when a Save button is clicked (it goes to the browser's download folder).

Signals come from validate.families and study.SETUPS; trades come from the unchanged engine
(validate.run -> nq_backtest.backtest), so results match the earlier tests. A fresh signal build
takes 2-3 min per market/session on three years of bars (the engine's signal code loops bar by
bar), spread over worker processes.
"""
from __future__ import annotations
import sys, io, re, os, time, threading, itertools, pickle, hashlib, inspect
from compression import zstd
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import numpy as np, pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
if __name__ == "__main__":                         # study modules that `import app` share this instance
    sys.modules.setdefault("app", sys.modules[__name__])
_argv = sys.argv
sys.argv = [_argv[0], str(CODE)]          # validate.py / study.py read sys.argv[1] at import
import nq_backtest as bt                  # noqa: E402
import validate as v                      # noqa: E402
import study                              # noqa: E402
sys.argv = _argv
cmp = v.cmp

DB_DIR = ROOT / "data" / "databento_ohlcv_1m"
CACHE = ROOT / "data" / "cache"
MONTHS = "FGHJKMNQUVXZ"
MARKETS = {  # Yahoo only has the micros
    "NQ":  dict(tick_size=0.25, point_value=20.0, yahoo=None, asset="MNQ"),
    "MNQ": dict(tick_size=0.25, point_value=2.0, yahoo="mnq_yahoo_5m.csv", asset="MNQ"),
    "GC":  dict(tick_size=0.10, point_value=100.0, yahoo=None, asset="MGC"),
    "MGC": dict(tick_size=0.10, point_value=10.0, yahoo="mgc_yahoo_5m.csv", asset="MGC"),
}
SESS_LABEL = {"new_york": "NY", "asia": "Asia"}
NY = "America/New_York"
NOKILL = v.NOKILL
COLS = ["open", "high", "low", "close", "atr"]   # all that nq_backtest.backtest reads from x


# ------------------------------------------------------------ speed-up with identical output
def _session_keys_fast(index: pd.DatetimeIndex, session: str) -> pd.Series:
    """Vectorised session_timeframe_comparison.session_keys (same keys, same dtype; see
    check_session_keys). The original calls session_id once per bar."""
    local = index.tz_convert(cmp.NY_TZ)
    minute = np.asarray(local.hour * 60 + local.minute)
    day = local.tz_localize(None).normalize()
    if session == "new_york":
        keep = (minute >= 570) & (minute < 960)
    else:
        morning = minute < 180
        keep = (minute >= 1140) | morning
        day = day - pd.to_timedelta(morning.astype(int), unit="D")
    vals = np.array(day.date, dtype=object)
    vals[~keep] = None
    return pd.Series(vals, index=index, dtype=object)


_session_keys_orig = cmp.session_keys
cmp.session_keys = _session_keys_fast


def check_session_keys(index: pd.DatetimeIndex) -> None:
    for s in ("new_york", "asia"):
        a, b = _session_keys_orig(index, s), _session_keys_fast(index, s)
        assert a.index.equals(b.index) and list(a) == list(b), f"session_keys mismatch ({s})"


# ----------------------------------------------------------------------------------------- data
def _expiry(code: str) -> int:
    """'Z5' -> months since year 0. Year digit 3-9 = 2023-2029, 0-2 = 2030-2032."""
    d = int(code[1])
    return (2020 + d if d >= 3 else 2030 + d) * 12 + MONTHS.index(code[0])


def _db_files(mkt: str) -> list:
    pat = re.compile(rf"\.ohlcv-1m\.({mkt}[{MONTHS}]\d)\.csv\.zst$")
    return sorted(f for f in DB_DIR.rglob("*.csv.zst") if pat.search(f.name))


def _file_sig(f: Path) -> str:
    st = f.stat()
    return f"{f.name}|{st.st_size}|{st.st_mtime_ns}"


def cache_path(mkt: str) -> Path:
    """Name carries a hash of the Databento outright files and of build_continuous, so new data
    or a change to the roll logic builds a fresh cache."""
    h = hashlib.sha256((inspect.getsource(build_continuous) + inspect.getsource(continuous_1m)).encode())
    for f in _db_files(mkt):
        h.update(_file_sig(f).encode())
    return CACHE / f"{mkt.lower()}_databento_5m_{h.hexdigest()[:10]}.csv"


def build_continuous(mkt: str) -> pd.DataFrame:
    """5m bars (labelled at the bar's start, UTC) from continuous_1m."""
    c = continuous_1m(mkt)
    bars = c.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    bars.index.name = "timestamp"
    return bars


def continuous_1m(mkt: str) -> pd.DataFrame:
    """Volume-rolled, back-adjusted continuous 1m series from the per-contract Databento 1m files.

    * outright contracts only (calendar-spread files such as NQH5-NQM5 are dropped)
    * CME trading day = 17:00 CT to 17:00 CT; each day trades the contract that had the most
      volume on the PREVIOUS trading day (no lookahead); the roll never goes back to an earlier
      expiry
    * additive back-adjustment at each roll (difference of the two contracts' closes on the last
      common minute before the switch): point moves and $ P&L are exact and the latest contract's
      prices are real. Rolls happen at 17:00 CT, between sessions.
    * UTC timestamps; columns open, high, low, close, volume, symbol
    """
    files = _db_files(mkt)
    if not files:
        raise FileNotFoundError(f"no {mkt} outright files under {DB_DIR}")
    m = pd.concat([pd.read_csv(io.BytesIO(zstd.decompress(f.read_bytes())),
                               usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"])
                   for f in files], ignore_index=True)
    m = m[m.symbol.str.fullmatch(rf"{mkt}[{MONTHS}]\d")]
    m["ts"] = pd.to_datetime(m.ts_event, utc=True, format="ISO8601")
    m["day"] = (m.ts.dt.tz_convert("America/Chicago") + pd.Timedelta(hours=7)).dt.tz_localize(None).dt.normalize()
    vol = m.groupby(["day", "symbol"]).volume.sum().unstack(fill_value=0).sort_index()
    exp = {s: _expiry(s[len(mkt):]) for s in vol.columns}
    by_exp = {e: s for s, e in exp.items()}
    lead = vol.idxmax(axis=1).shift(1)
    lead.iloc[0] = vol.iloc[0].idxmax()
    lead = lead.map(exp).cummax().map(by_exp)
    gap = pd.Series(0.0, index=lead.index)
    by_day = dict(tuple(m.groupby("day")))
    for i in np.flatnonzero(lead.ne(lead.shift()).to_numpy())[1:]:
        prev, old, new = lead.index[i - 1], lead.iloc[i - 1], lead.iloc[i]
        d = by_day[prev]
        pa = d[d.symbol == old].set_index("ts").close
        pb = d[d.symbol == new].set_index("ts").close
        common = pa.index.intersection(pb.index)
        if len(common):
            gap.iloc[i] = pb[common[-1]] - pa[common[-1]]
        else:  # no overlapping minute: first new-contract print vs last old-contract print
            nd = by_day[lead.index[i]]
            gap.iloc[i] = nd[nd.symbol == new].close.iloc[0] - pa.iloc[-1]
        print(f"  {mkt} roll {old}->{new} on {lead.index[i].date()}  gap {gap.iloc[i]:+.2f}", flush=True)
    adj = gap[::-1].cumsum()[::-1] - gap          # sum of all later roll gaps
    c = m[m.symbol.to_numpy() == m.day.map(lead).to_numpy()].copy()
    a = c.day.map(adj).to_numpy()
    for k in ("open", "high", "low", "close"):
        c[k] = (c[k] + a).round(6)
    c["adj"] = a                                   # back-adjustment added; real price = price - adj
    c = c.set_index("ts").sort_index()
    c = c[~c.index.duplicated()]
    c.index.name = "timestamp"
    return c[["open", "high", "low", "close", "volume", "symbol", "adj"]]


def ensure_cache(mkt: str) -> Path:
    p = cache_path(mkt)
    if not p.exists():
        print(f"building {p.name} from Databento 1m files (one time) ...", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        for old in [*CACHE.glob(f"{mkt.lower()}_databento_5m_*.csv"), CACHE / f"{mkt.lower()}_databento_5m.csv"]:
            old.unlink(missing_ok=True)                # stale price cache for this market
        bars = build_continuous(mkt)
        tmp = p.with_suffix(".tmp")
        bars.to_csv(tmp)
        tmp.replace(p)
        print(f"  {len(bars):,} bars {bars.index[0]} -> {bars.index[-1]}", flush=True)
    return p


def base_cfg(mkt: str, session: str, slippage: float = 1.0, commission: float = 2.50) -> bt.Config:
    s = MARKETS[mkt]
    return bt.Config(tick_size=s["tick_size"], point_value=s["point_value"], session=session,
                     contracts=1, risk_fraction=0.0, slippage_ticks=float(slippage),
                     commission_per_side=float(commission), drawdown_dollars=1000.0, target_dollars=1250.0)


_loaded: dict = {}


def load(source: str, mkt: str) -> tuple[pd.DataFrame, int]:
    """Cleaned 5m bars (Chicago tz) via nq_backtest.load_bars + validate.clean, kept in memory."""
    if (source, mkt) not in _loaded:
        _loaded[(source, mkt)] = v.clean(bt.load_bars(price_path(source, mkt), base_cfg(mkt, "new_york")))
    return _loaded[(source, mkt)]


TBBO_DIR = ROOT / "data" / "databento_tbbo"


def no_data_msg(what: str, where: str) -> str:
    return (f"No data: this tab needs the {what} in {where} (licensed, not in the repository; see README, "
            f"\"Data\"). The GEX tab works without local data.")


def has_data(source: str, mkt: str) -> bool:
    if source == "databento":
        return bool(_db_files(mkt))
    f = MARKETS[mkt]["yahoo"]
    return f is not None and (ROOT / "data" / f).exists()


def price_path(source: str, mkt: str) -> Path:
    if source == "databento":
        return ensure_cache(mkt)
    f = MARKETS[mkt]["yahoo"]
    if f is None:
        raise ValueError(f"there is no Yahoo {mkt} file; pick MNQ/MGC or the Databento source")
    return ROOT / "data" / f


# ------------------------------------------------------------------------------- signal cache
SIGNAL_CODE = ["nq_backtest.py", "validate.py", "session_timeframe_comparison.py",
               "strategy_research_suite.py", "study.py"]


def signal_key(source: str, mkt: str, session: str) -> str:
    """Changes when the price file or any code that shapes the signals changes."""
    price = hashlib.sha256(price_path(source, mkt).read_bytes()).hexdigest()   # content, so a rebuilt-but-identical cache keeps its signals
    h = hashlib.sha256(f"{source}|{mkt}|{session}|{price}".encode())
    for f in SIGNAL_CODE:
        h.update((CODE / f).read_bytes())
    for fn in (family_plan, task_signal, build_pack, _frames, _session_keys_fast, base_cfg, load):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def signal_cache_path(source: str, mkt: str, session: str) -> Path:
    return CACHE / f"signals_{source}_{mkt.lower()}_{session}.pkl.zst"


def save_signals(key, pack, sig: str) -> None:
    """Signals and stop series only (as arrays); bars are rebuilt from the price cache on load."""
    ents = [dict({k: x for k, x in e.items() if k not in ("sg", "atr")}, sg=e["sg"].to_numpy(),
                 atr=None if e["atr"] is None else e["atr"].to_numpy()) for e in pack["entries"]]
    blob = zstd.compress(pickle.dumps(dict(key=sig, entries=ents, setups=pack["setups"], nbad=pack["nbad"],
                                           seconds=pack["seconds"]), protocol=pickle.HIGHEST_PROTOCOL))
    p = signal_cache_path(*key)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(blob)
    tmp.replace(p)


def load_signals(key, sig: str) -> dict | None:
    p = signal_cache_path(*key)
    if not p.exists():
        return None
    try:
        d = pickle.loads(zstd.decompress(p.read_bytes()))   # our own file, written by save_signals
    except Exception as exc:
        print(f"ignoring unreadable {p.name}: {exc!r}", flush=True)
        return None
    if d.get("key") != sig:
        return None
    _, xs = _frames(key[0], key[1])
    frames = {tf: x[COLS] for tf, x in xs.items()}
    entries = []
    for e in d["entries"]:
        idx = frames[e["frame"]].index
        if len(e["sg"]) != len(idx):
            return None
        entries.append(dict(e, sg=pd.Series(e["sg"], index=idx),
                            atr=None if e["atr"] is None else pd.Series(e["atr"], index=idx)))
    return dict(frames=frames, entries=entries, setups=d["setups"], nbad=d["nbad"], seconds=d["seconds"],
                cached=True)


# ------------------------------------------------------------------------------- signal plan
def family_plan(session: str) -> list:
    """validate.families, line for line, as data: (family, params, ordinal, spec, post, frame_tf).
    spec names one signal-function call, so the calls can run in parallel workers; post is the
    filter families() applies afterwards. check_parity() compares the result with validate.families
    itself on every launch, so a change there that isn't mirrored here shows up in the status line."""
    P = []
    for bb, rl in itertools.product((1.5, 2.0, 2.5), (25.0, 30.0, 35.0)):
        P.append(("Lima BB+RSI", dict(bb_mult=bb, rsi=rl), ["bb_mult", "rsi"], ("make_signals", "lima", bb, rl), None, 5))
    for name, key in (("RSI mean reversion", "mean_reversion_rsi"), ("Momentum", "momentum"),
                      ("EMA cloud", "ema_cloud"), ("BOS/CHOCH", "bos_choch")):
        P.append((name, {}, [], ("make_signals", key), None, 5))
    for var in ("fvg", "ifvg", "fvg_vwap", "ifvg_vwap"):
        P.append(("FVG 5/15/30 alignment", dict(variant=var), [], ("make_signals", var), None, 5))
    for tf in (5, 15, 30):
        for f in (0, 1):
            P.append(("FVG/IFVG retest", dict(tf=tf, vwap=f), ["tf"], ("fvg_ifvg", tf), "insess_vw" if f else "insess", 5))
    for tf in (15, 30):
        P.append(("FVG + BOS/CHOCH", dict(tf=tf), ["tf"], ("fvg_structure", tf), "insess", 5))
        for f in (0, 1):
            P.append(("FVG + order block", dict(tf=tf, vwap=f), ["tf"], ("fvg_ob", tf), "insess_vw" if f else "insess", 5))
    for tf in (5, 10, 15):
        for sd in (1.5, 2.0, 2.5):
            P.append(("November AVWAP", dict(tf=tf, sigma=sd), ["tf", "sigma"], ("avwap", tf, sd), None, tf))
    for orb in (5, 15):
        for var in ("breakout", "two_close_fakeout_filter", "retest"):
            for tf in (5, 10, 15):
                P.append(("Oscar ORB", dict(orb=orb, variant=var, tf=tf), ["orb", "tf"], ("orb", tf, orb, var), None, tf))
    for pct in (0.0007, 0.0010, 0.0014):
        P.append(("Papa last-hour fade", dict(target_pct=pct), ["target_pct"], ("papa", pct), "papa", 5))
    P.append(("Romeo volume proxy", {}, [], ("romeo",), None, 5))
    return P


SLOW_FIRST = {"fvg_ob": 0, "fvg_structure": 1, "fvg_ifvg": 2, "orb": 3, "avwap": 4, "make_signals": 5, "papa": 6, "romeo": 7}
_wframes: dict = {}


def _frames(source, mkt):
    """5/10/15m indicator frames exactly as validate.families builds them."""
    if (source, mkt) not in _wframes:
        import multiprocessing
        if multiprocessing.parent_process() is not None:   # worker: hold one market at a time
            _wframes.clear()
            for k in [k for k in _loaded if k != (source, mkt)]:
                del _loaded[k]
        raw, _ = load(source, mkt)
        x5 = bt.indicators(raw)
        _wframes[(source, mkt)] = (raw, {5: x5, 10: bt.indicators(cmp.resample_ohlcv(raw, 10)),
                                         15: bt.indicators(cmp.resample_ohlcv(raw, 15))})
    return _wframes[(source, mkt)]


def task_signal(source, mkt, session, spec):
    """One signal-function call from family_plan, with the arguments validate.families uses."""
    raw, xs = _frames(source, mkt)
    x5, cfg, kind = xs[5], base_cfg(mkt, session), spec[0]
    if kind == "make_signals":
        return bt.make_signals(x5, spec[1], cfg, *spec[2:]), None
    if kind == "fvg_ifvg":
        sg, _, atr = bt.make_fvg_ifvg_signals(x5, spec[1]); return sg, atr
    if kind == "fvg_structure":
        sg, _, atr = bt.make_fvg_structure_signals(x5, spec[1]); return sg, atr
    if kind == "fvg_ob":
        sg, _, atr, _ = bt.make_fvg_orderblock_signals(x5, spec[1]); return sg, atr
    if kind == "avwap":
        x = xs[spec[1]]; x.attrs["bar_minutes"] = spec[1]
        return cmp.avwap_signal(x, session, spec[2]), None
    if kind == "orb":
        sg, _ = cmp.orb_signal(raw, xs[spec[1]], session, spec[1], spec[2], spec[3]); return sg, None
    if kind == "papa":
        return v.papa_signal(x5, session, spec[1])
    if kind == "romeo":
        return v.romeo(x5, cfg), None
    raise ValueError(spec)


def build_pack(source: str, mkt: str, session: str, progress=None) -> dict:
    """Every validate.families builder plus study.SETUPS, materialised as (bars, signal, stop
    distance, exit time). Signals are causal (truncation-tested), so computing them once on all
    bars and slicing to a date range later equals a run that starts earlier, without warm-up loss."""
    t0 = time.time()
    raw, nbad = load(source, mkt)
    cfg = base_cfg(mkt, session)
    plan = family_plan(session)
    specs = list(dict.fromkeys([p[3] for p in plan] + [("papa", 0.001)]))
    specs.sort(key=lambda s: SLOW_FIRST[s[0]])
    futs = {pool().submit(task_signal, source, mkt, session, s): s for s in specs}
    _, xs = _frames(source, mkt)                   # main-process copy for the frames and filters
    x5 = xs[5]
    insess = bt.in_session(x5.index, cfg)
    vwap = bt.session_vwap(x5, cfg)
    res = {}
    for n, f in enumerate(as_completed(futs), 1):
        res[futs[f]] = f.result()
        if progress:
            progress(n, len(futs))
    ex = v.EXIT[session]
    entries = []
    for fam, params, ordinal, spec, post, tf in plan:
        sg, atr = res[spec]
        e = dict(family=fam, params=params, ordinal=ordinal, in_grid=True, frame=tf, atr=None, ex=ex, rr_div=False)
        if post in ("insess", "insess_vw"):
            sg = sg.where(insess, 0)
            if post == "insess_vw":
                sg = sg.where(((sg > 0) & (x5.close > vwap)) | ((sg < 0) & (x5.close < vwap)), 0)
            e["atr"] = atr
        elif post == "papa":
            e.update(atr=atr, ex=v.PAPA[session][1], rr_div=True)      # target pct / rr
        elif atr is not None:
            e["atr"] = atr
        e["sg"] = sg.astype("int8")
        entries.append(e)
    setups = {}
    for tag, group, smkt, ssess, fam, params, rr0 in study.SETUPS:
        if fam == "PAPA30":
            psig, ptgt = res[("papa", 0.001)]
            entries.append(dict(family="PAPA30", params={}, ordinal=[], in_grid=False, frame=5,
                                sg=psig.astype("int8"), atr=ptgt.where(ptgt.isna(), 30.0),
                                ex=v.PAPA[session][1], rr_div=True))
            i = len(entries) - 1
        else:
            i = next(k for k, e in enumerate(entries) if e["family"] == fam and e["params"] == params)
        setups[tag] = dict(idx=i, rr=rr0, group=group, market=smkt, session=ssess, family=fam, params=params)
    return dict(frames={tf: x[COLS] for tf, x in xs.items()}, entries=entries, setups=setups, nbad=nbad,
                seconds=time.time() - t0)


def task_pack(source: str, mkt: str, session: str) -> dict:
    """Reference path for check_parity: validate.families itself, its builders materialised."""
    t0 = time.time()
    raw, nbad = load(source, mkt)
    cfg = base_cfg(mkt, session)
    fams = v.families(raw, cfg, session)
    x5 = bt.indicators(raw)
    papa_sig, papa_tgt = v.papa_signal(x5, session, 0.001)
    papa_tgt = papa_tgt.where(papa_tgt.isna(), 30.0)
    frames: list = []

    def fid(x):
        for k, f in enumerate(frames):
            if f is x:
                return k
        frames.append(x)
        return len(frames) - 1

    def mat(b):
        x, sg, atr, ex = b(1.0)
        x2, sg2, atr2, ex2 = b(2.0)
        assert x2 is x and sg2 is sg and ex2 == ex
        rr_div = atr is not None and atr2 is not atr          # only the Papa targets scale with R:R
        if rr_div:
            assert np.allclose((atr2 * 2).to_numpy(), atr.to_numpy(), equal_nan=True)
        if atr is not None:
            assert len(atr) == len(x) and atr.index.equals(x.index)
        return dict(frame=fid(x), sg=sg.astype("int8"), atr=atr, ex=ex, rr_div=rr_div)

    entries = [dict(family=f, params=p, ordinal=o, in_grid=True, **mat(b)) for f, p, o, b in fams]
    setups = {}
    for tag, group, smkt, ssess, fam, params, rr0 in study.SETUPS:
        if fam == "PAPA30":
            entries.append(dict(family="PAPA30", params={}, ordinal=[], in_grid=False,
                                **mat(lambda r: (x5, papa_sig, papa_tgt / r, v.PAPA[session][1]))))
            i = len(entries) - 1
        else:
            i = next(k for k, e in enumerate(entries) if e["family"] == fam and e["params"] == params)
        setups[tag] = dict(idx=i, rr=rr0, group=group, market=smkt, session=ssess, family=fam, params=params)
    return dict(frames=[f[COLS] for f in frames], entries=entries, setups=setups, nbad=nbad,
                seconds=time.time() - t0)


def task_runs(x, sg, atr, ex, rr_div, cfg, runs) -> list:
    """validate.run for each (rr, kwargs) on one builder's materialised inputs."""
    out = []
    for rr, kw in runs:
        build = lambda r: (x, sg, (atr / r) if rr_div else atr, ex)
        out.append(v.run(build, rr, cfg, **kw))
    return out


# ----------------------------------------------------------------------------- main-process state
_pool: ProcessPoolExecutor | None = None
_rpool: ProcessPoolExecutor | None = None
_lock = threading.Lock()
PACKS: dict = {}                 # (source, mkt, session) -> Future[pack]
PACK_T0: dict = {}
TRADES: dict = {}                # run key -> trades DataFrame


PACK_PROG: dict = {}
PARITY = dict(state="pending", msg="")
_queue: list = []
_orch: threading.Thread | None = None
_cv = threading.Condition(_lock)


def pool() -> ProcessPoolExecutor:
    """Workers for signal builds (incl. the background prefetch)."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=max(2, min(12, (os.cpu_count() or 4) // 2)))
    return _pool


def rpool() -> ProcessPoolExecutor:
    """Separate workers for backtests, so charts and validation never queue behind a prefetch."""
    global _rpool
    if _rpool is None:
        _rpool = ProcessPoolExecutor(max_workers=max(2, min(12, (os.cpu_count() or 4) // 2 - 1)))
    return _rpool


def pack_future(key, front: bool = True) -> Future:
    """Packs are built one at a time (each one already uses every worker); the market/session
    on screen jumps the queue."""
    global _orch
    with _cv:
        if _orch is None:
            _orch = threading.Thread(target=_orchestrate, daemon=True)
            _orch.start()
        if key not in PACKS:
            load(key[0], key[1])                     # builds the price cache here, not in a worker
            PACKS[key] = Future()
            _queue.insert(0, key) if front else _queue.append(key)
            _cv.notify()
        elif front and key in _queue and _queue[0] != key:
            _queue.remove(key); _queue.insert(0, key)
        return PACKS[key]


def _orchestrate():
    while True:
        with _cv:
            while not _queue:
                _cv.wait()
            key = _queue.pop(0)
        PACK_T0[key] = time.time()
        try:
            sig = signal_key(*key)
            pack = load_signals(key, sig)
            if pack is None:
                pack = build_pack(*key, progress=lambda n, N, k=key: PACK_PROG.__setitem__(k, (n, N)))
                save_signals(key, pack, sig)
            PACKS[key].set_result(pack)
        except Exception as exc:
            PACKS[key].set_exception(exc)


def check_parity():
    """Every launch: family_plan's parallel build must equal validate.families itself (Yahoo MNQ,
    both sessions: every signal, stop series, exit time and R:R scaling)."""
    try:
        for sess in ("new_york", "asia"):
            ref = pool().submit(task_pack, "yahoo", "MNQ", sess)
            mine = pack_future(("yahoo", "MNQ", sess), front=False).result()
            ref = ref.result()
            assert len(ref["entries"]) == len(mine["entries"])
            for a, b in zip(ref["entries"], mine["entries"]):
                for k in ("family", "params", "ordinal", "in_grid", "ex", "rr_div"):
                    assert a[k] == b[k], (a["family"], a["params"], k)
                assert a["sg"].equals(b["sg"]), (a["family"], a["params"], "signal")
                assert (a["atr"] is None) == (b["atr"] is None)
                if a["atr"] is not None:
                    assert np.array_equal(a["atr"].to_numpy(), b["atr"].to_numpy(), equal_nan=True)
                assert ref["frames"][a["frame"]].equals(mine["frames"][b["frame"]])
        PARITY.update(state="ok", msg="parity with validate.families: OK")
    except Exception as exc:
        PARITY.update(state="fail", msg=f"⚠ parity with validate.families FAILED: {exc!r}")
    print(PARITY["msg"], flush=True)


def run_many(key, cfg, base_kw, a, b, jobs, progress=None) -> dict:
    """jobs: list of (entry_idx, rr, stress_items). Returns {job: trades}. Memoised in memory."""
    pack = PACKS[key].result()
    ck = (key, cfg.slippage_ticks, cfg.commission_per_side, tuple(sorted(base_kw.items())), a, b)
    out, todo = {}, []
    for j in dict.fromkeys(jobs):
        if (ck, j) in TRADES:
            out[j] = TRADES[(ck, j)]
        else:
            todo.append(j)
    futs = {}
    for j in todo:
        i, rr, stress = j
        e = pack["entries"][i]
        x = pack["frames"][e["frame"]]
        m = (x.index >= a) & (x.index < b)
        atr = e["atr"][m] if e["atr"] is not None else None
        futs[rpool().submit(task_runs, x[m], e["sg"][m], atr, e["ex"], e["rr_div"], cfg,
                           [(rr, {**base_kw, **dict(stress)})])] = j
    for n, f in enumerate(as_completed(futs), 1):
        j = futs[f]
        out[j] = TRADES[(ck, j)] = f.result()[0]
        if progress:
            progress(n, len(futs))
    if len(TRADES) > 5000:
        TRADES.clear()
    return out


def fmt_params(p: dict) -> str:
    return ", ".join(f"{k}={x:g}" if isinstance(x, float) else f"{k}={x}" for k, x in p.items())


def entry_label(e: dict) -> str:
    return f"{e['family']} · {fmt_params(e['params'])}" if e["params"] else e["family"]


def resolve(pack, sid: str, rr_ctrl: float, setup_override: bool):
    """Strategy id -> (display name, entry idx, rr)."""
    if sid.startswith("S:"):
        s = pack["setups"][sid[2:]]
        return sid[2:], s["idx"], (rr_ctrl if setup_override else s["rr"])
    i = int(sid[2:])
    return entry_label(pack["entries"][i]), i, rr_ctrl


def date_bounds(start, end):
    a = pd.Timestamp(start, tz=NY)
    b = pd.Timestamp(end, tz=NY) + pd.Timedelta(days=1)
    return a, b


def session_days(source, mkt, session, a, b):
    raw, _ = load(source, mkt)
    idx = raw.index[(raw.index >= a) & (raw.index < b)]
    return cmp.session_keys(idx, session).dropna().unique()


def with_r(t: pd.DataFrame, cfg) -> pd.DataFrame:
    """R per trade: P&L over the full 1-lot risk incl. both sides' slippage and commission
    (study.py's risk_1c at default costs)."""
    t = t.copy()
    risk = t.risk_dollars + 2 * cfg.slippage_ticks * cfg.tick_size * cfg.point_value * t.contracts \
        + 2 * cfg.commission_per_side * t.contracts
    t["R"] = t.pnl / risk
    return t


def summarize(name, t, ndays) -> dict:
    if not len(t):
        return dict(strategy=name, trades=0, win_rate=None, avg_r=None, net=0.0, max_dd=0.0,
                    trades_per_day=0.0, eval_end="no trades")
    eq = t.pnl.cumsum().to_numpy()
    dd = np.maximum.accumulate(np.maximum(eq, 0)) - eq
    breach = np.flatnonzero(dd >= 1000)
    end = "survives" if not len(breach) else pd.Timestamp(t.exit_time.iloc[breach[0]]).tz_convert(NY).strftime("%Y-%m-%d")
    return dict(strategy=name, trades=len(t), win_rate=round(100 * float((t.pnl > 0).mean()), 1),
                avg_r=round(float(t.R.mean()), 3), net=round(float(t.pnl.sum()), 2),
                max_dd=round(float(dd.max()), 2), trades_per_day=round(len(t) / max(len(ndays), 1), 2),
                eval_end=end)


# ------------------------------------------------------------------------------------ validation
VAL = dict(id=0, state="idle", msg="", rows=[], meta="")


def neighbours(entries, i, rr):
    """validate.py's 'nearby settings': one step on exactly one ordinal param (incl. R:R),
    everything else unchanged."""
    e = entries[i]
    same = [j for j, f in enumerate(entries) if f["family"] == e["family"]]
    out = []
    for c in e["ordinal"] + ["rr"]:
        if c == "rr":
            vals = sorted(set(v.RRS) | {rr})
            k = vals.index(rr)
            out += [(i, vals[n], ()) for n in (k - 1, k + 1) if 0 <= n < len(vals)]
            continue
        vals = sorted({entries[j]["params"][c] for j in same})
        k = vals.index(e["params"][c])
        for n in (k - 1, k + 1):
            if 0 <= n < len(vals):
                out += [(j, rr, ()) for j in same if entries[j]["params"][c] == vals[n]
                        and all(entries[j]["params"][o] == e["params"][o] for o in e["params"] if o != c)]
    return out


def run_validation(job_id, key, picks, slip, comm, be, start, end):
    """The six tests of validate.py on the current controls. picks: list of (name, idx, rr)."""
    def say(msg):
        if VAL["id"] == job_id:
            VAL["msg"] = msg
    try:
        source, mkt, session = key
        t0 = time.time()
        pack = PACKS[key].result()
        E = pack["entries"]
        cfg = base_cfg(mkt, session, slip, comm)
        base = dict(NOKILL, **({"breakeven_r": 1.0} if be else {}))
        a, b = date_bounds(start, end)
        days = session_days(source, mkt, session, a, b)
        grid = [(i, rr, ()) for i, e in enumerate(E) if e["in_grid"] for rr in v.RRS]
        T = run_many(key, cfg, base, a, b, grid, lambda n, N: say(f"grid: {n}/{N} backtests"))
        cols = {j: v.daily(T[j], days, session).to_numpy() for j in grid}
        all_sr = v.sharpe(np.column_stack([cols[j] for j in grid]))
        stress_items = [tuple(sorted(s.items())) for s in v.STRESS]

        def tests(name, i, rr, note):
            fam = E[i]["family"]
            fam_jobs = [j for j in grid if E[j[0]]["family"] == fam]
            me = (i, rr, ())
            nb = neighbours(E, i, rr)
            R = run_many(key, cfg, base, a, b, [me] + nb + [(i, rr, s) for s in stress_items])
            st = v.stats(R[me])
            col = cols[me] if me in cols else v.daily(R[me], days, session).to_numpy()
            dsr = v.deflated_sharpe(col, all_sr)
            if fam_jobs:
                M = np.column_stack([cols[j] for j in fam_jobs])
                wf, p = v.walk_forward(M), v.pbo(M)
            else:
                wf, p = np.nan, np.nan
            stress = [v.stats(R[(i, rr, s)]) for s in stress_items]
            nbn = np.array([v.stats(R[j])["net"] for j in nb])
            res = dict(initial=st["net"] > 0 and not st["busted"] and st["trades"] >= 20,
                       walk_forward=bool(np.isfinite(wf) and wf > 0),
                       pbo=bool(np.isfinite(p) and p < 0.5),
                       deflated_sharpe=dsr >= 0.95,
                       cost_stress=all(s["net"] > 0 and not s["busted"] for s in stress),
                       nearby=len(nbn) >= 2 and float(np.mean(nbn > 0)) >= 0.6)
            P = lambda ok: "PASS" if ok else "FAIL"
            return dict(strategy=name, config=(fmt_params(E[i]["params"]) + ", " if E[i]["params"] else "") + f"rr={rr:g}",
                        note=note, trades=st["trades"], net=round(st["net"], 2),
                        initial=P(res["initial"]) + (" (bust)" if st["busted"] else ""),
                        walk_forward=(P(res["walk_forward"]) + f" ${wf:,.0f}") if np.isfinite(wf) else "n/a",
                        pbo=(P(res["pbo"]) + f" {p:.2f}") if np.isfinite(p) else "n/a",
                        deflated_sharpe=P(res["deflated_sharpe"]) + f" {dsr:.2f}",
                        cost_stress=P(res["cost_stress"]) + f" worst ${min(s['net'] for s in stress):,.0f}",
                        nearby=P(res["nearby"]) + f" {int((nbn > 0).sum())}/{len(nbn)}",
                        all=P(all(res.values())))

        rows, done_best = [], set()
        for n, (name, i, rr) in enumerate(picks, 1):
            say(f"tests for {name} ({n}/{len(picks)})")
            rows.append(tests(name, i, rr, "selected"))
            fam = E[i]["family"]
            fam_jobs = [j for j in grid if E[j[0]]["family"] == fam]
            if fam_jobs and fam not in done_best:
                done_best.add(fam)
                nets = [v.stats(T[j])["net"] for j in fam_jobs]
                bi, brr, _ = fam_jobs[int(np.argmax(nets))]
                rows.append(tests(fam, bi, brr, f"grid winner ({len(fam_jobs)} configs)"))
        if VAL["id"] == job_id:
            VAL.update(state="done", rows=rows, msg="",
                       meta=(f"{len(grid)} configs in the grid ({sum(e['in_grid'] for e in E)} settings x "
                             f"{len(v.RRS)} R:R), {len(days)} session days, {source} {mkt} "
                             f"{SESS_LABEL[session]} {start} to {end}, slippage {slip:g} tick, "
                             f"commission ${comm:g}/side{', breakeven at 1R' if be else ''}. "
                             f"Stress scenarios use validate.py's fixed costs (1.5 / 2 ticks, $3.75). "
                             f"Took {time.time() - t0:.0f}s."))
    except Exception as exc:  # surface errors in the page
        if VAL["id"] == job_id:
            VAL.update(state="error", msg=f"validation failed: {exc!r}")


# ------------------------------------------------------------------------------------------- UI
# dark-mode steps of the reference categorical palette, fixed order
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
INK, INK2, MUTED, GRID, AXIS = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
SURFACE, PAGE = "#1a1a19", "#0d0d0d"
GOOD, BAD = "#0ca30c", "#d03b3b"
CSS = f"""
body {{ background:{PAGE}; color:{INK}; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; margin:0; }}
.wrap {{ display:flex; gap:16px; padding:16px; align-items:flex-start; flex-wrap:wrap; }}
.side {{ flex:0 0 300px; background:{SURFACE}; border-radius:8px; padding:14px; box-sizing:border-box; max-width:100%; }}
.main {{ flex:1 1 600px; min-width:0; }}
.card {{ background:{SURFACE}; border-radius:8px; padding:12px; margin-bottom:16px; }}
label.h {{ display:block; color:{MUTED}; font-size:12px; margin:12px 0 4px; text-transform:uppercase; letter-spacing:.04em; }}
.status {{ color:{INK2}; font-size:13px; margin-bottom:12px; min-height:18px; }}
button {{ background:#2c2c2a; color:{INK}; border:1px solid {AXIS}; border-radius:6px; padding:6px 12px; cursor:pointer; font-size:13px; margin:4px 6px 0 0; }}
button:hover {{ background:{AXIS}; }}
button.primary {{ background:#256abf; border-color:#256abf; }}
input[type=number] {{ background:#0d0d0d; color:{INK}; border:1px solid {AXIS}; border-radius:4px; padding:4px 6px; width:80px; }}
.note {{ color:{MUTED}; font-size:12px; margin-top:8px; line-height:1.4; }}
/* Dash 4 components (dropdown, radio, checklist, slider, date picker, number input) theme via these */
:root {{
  --Dash-Fill-Inverse-Strong: #0d0d0d;
  --Dash-Fill-Interactive-Strong: #3987e5;
  --Dash-Fill-Interactive-Weak: rgba(255,255,255,0.06);
  --Dash-Fill-Primary-Hover: rgba(255,255,255,0.06);
  --Dash-Fill-Primary-Active: rgba(255,255,255,0.12);
  --Dash-Fill-Disabled: rgba(255,255,255,0.08);
  --Dash-Text-Primary: {INK};
  --Dash-Text-Strong: #ffffff;
  --Dash-Text-Weak: {INK2};
  --Dash-Text-Disabled: {MUTED};
  --Dash-Stroke-Strong: #5a5955;
  --Dash-Stroke-Weak: {AXIS};
  --Dash-Shading-Strong: rgba(0,0,0,0.6);
  --Dash-Shading-Weak: rgba(0,0,0,0.4);
}}
"""

TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "#0d0d0d", "color": MUTED, "fontWeight": "600", "border": f"1px solid {GRID}"},
    style_cell={"backgroundColor": SURFACE, "color": INK, "border": f"1px solid {GRID}", "padding": "6px 8px",
                "fontFamily": "system-ui,'Segoe UI',sans-serif", "fontSize": 13, "textAlign": "right",
                "fontVariantNumeric": "tabular-nums", "whiteSpace": "normal", "height": "auto"},
)


HEAT = {"g2": "rgba(12,163,12,0.45)", "g1": "rgba(12,163,12,0.20)",
        "r1": "rgba(208,59,59,0.22)", "r2": "rgba(208,59,59,0.48)"}


def heat_rules(cols) -> list:
    """Cell shading, good to bad: average-R columns in bands (>= +0.15, +0.05..+0.15, -0.05..+0.05
    unshaded, -0.15..-0.05, <= -0.15); $ columns by sign; pass/fail columns by result."""
    rules = []
    for c in cols:
        q = "{" + c + "}"
        if c == "deflated Sharpe":
            rules.append({"if": {"filter_query": f"{q} >= 0.95", "column_id": c}, "backgroundColor": HEAT["g2"]})
            rules.append({"if": {"filter_query": f"{q} lt 0.95", "column_id": c}, "backgroundColor": HEAT["r1"]})
        elif "Sharpe" in c:
            for cond, col in ((f"{q} >= 0.5", "g2"), (f"{q} >= 0.2 && {q} < 0.5", "g1"),
                              (f"{q} <= -0.2 && {q} > -0.5", "r1"), (f"{q} <= -0.5", "r2")):
                rules.append({"if": {"filter_query": cond, "column_id": c}, "backgroundColor": HEAT[col]})
        elif "annual %" in c or "%/yr" in c or "half %" in c or c == "total %":
            rules += [{"if": {"filter_query": f"{q} gt 0", "column_id": c}, "backgroundColor": HEAT["g1"]},
                      {"if": {"filter_query": f"{q} lt 0", "column_id": c}, "backgroundColor": HEAT["r1"]}]
        if "avgR" in c or c.startswith("R2 ") or c.startswith("avg R"):
            for cond, col in ((f"{q} >= 0.15", "g2"), (f"{q} >= 0.05 && {q} < 0.15", "g1"),
                              (f"{q} <= -0.05 && {q} > -0.15", "r1"), (f"{q} <= -0.15", "r2")):
                rules.append({"if": {"filter_query": cond, "column_id": c}, "backgroundColor": HEAT[col]})
        elif c.endswith("$"):
            rules += [{"if": {"filter_query": f"{q} gt 0", "column_id": c}, "backgroundColor": HEAT["g1"]},
                      {"if": {"filter_query": f"{q} lt 0", "column_id": c}, "backgroundColor": HEAT["r1"]}]
        elif c in ("meets rule", "market_ok"):
            rules += [{"if": {"filter_query": f'{q} = "YES" || {q} = "yes"', "column_id": c},
                       "backgroundColor": HEAT["g2"], "fontWeight": "600"},
                      {"if": {"filter_query": f'{q} = "no"', "column_id": c}, "color": MUTED}]
    return rules


def _bar_layout(f, height, **kw):
    f.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height, barmode="group", bargap=0.25,
                    bargroupgap=0.08, font=dict(color=INK2, family="system-ui, Segoe UI, sans-serif", size=12),
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(color=INK2)), hovermode="closest", **kw)
    f.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    f.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    return f


def of_figs(R):
    """(score chart, combinations chart) from orderflow_study.run_all's tables."""
    import plotly.graph_objects as go
    sc, cb = R["score"], R["combos_table"]
    f1 = go.Figure()
    for m, color in (("NQ", SERIES[0]), ("GC", SERIES[1])):
        g = sc[sc.market == m].sort_values("score")
        f1.add_bar(x=[f"{s}" for s in g.score], y=g["avgR2 net"], name=m, marker_color=color,
                   marker_opacity=[1.0 if n >= 30 else 0.35 for n in g.trades],
                   text=[f"{y:+.2f}" for y in g["avgR2 net"]], textposition="outside", textfont=dict(color=INK2, size=11),
                   customdata=np.column_stack([g.trades, g["avgR2 gross"], g["win2%"]]),
                   hovertemplate=f"<b>{m}</b>, score %{{x}}<br>avg R at 2R: %{{y:+.3f}} net, %{{customdata[1]:+.3f}} gross"
                                 "<br>%{customdata[0]} trades, win %{customdata[2]}%<extra></extra>")
    f1.add_hline(y=0, line=dict(color=INK2, width=1))
    lo = float(min(sc["avgR2 net"].min(), 0)) - 0.12
    _bar_layout(f1, 380, margin=dict(l=70, r=20, t=50, b=80),
                xaxis_title="How many of C1–C4 fired in the window (0 = B2 price-rejection entry)",
                yaxis=dict(title="Average R per trade at 2R, net", range=[lo, 0.25], tickformat="+.2f"),
                annotations=[dict(text="Above 0 = profitable after costs. Faded bars: fewer than 30 trades.",
                                  x=0, y=-0.27, xref="paper", yref="paper", showarrow=False, xanchor="left",
                                  font=dict(color=MUTED, size=11))])
    cb = cb.assign(order=cb[["NQ avgR2 net", "GC avgR2 net"]].mean(axis=1)).sort_values("order")
    f2 = go.Figure()
    for m, color in (("NQ", SERIES[0]), ("GC", SERIES[1])):
        f2.add_bar(y=cb.combo, x=cb[f"{m} avgR2 net"], name=m, orientation="h", marker_color=color,
                   marker_opacity=[1.0 if n >= 30 else 0.35 for n in cb[f"{m} trades"]],
                   text=[f"{v:+.2f}" for v in cb[f"{m} avgR2 net"]], textposition="outside",
                   textfont=dict(color=INK2, size=11),
                   customdata=np.column_stack([cb[f"{m} trades"], cb[f"{m} avgR2 gross"], cb[f"{m} win2%"]]),
                   hovertemplate=f"<b>{m}</b> %{{y}}<br>avg R at 2R: %{{x:+.3f}} net, %{{customdata[1]:+.3f}} gross"
                                 "<br>%{customdata[0]} trades, win %{customdata[2]}%<extra></extra>")
    f2.add_vline(x=0, line=dict(color=INK2, width=1))
    lo = float(min(cb[["NQ avgR2 net", "GC avgR2 net"]].min().min(), 0)) - 0.12
    _bar_layout(f2, 520, margin=dict(l=110, r=30, t=50, b=70),
                xaxis=dict(title="Average R per trade at 2R, net", range=[lo, 0.3], tickformat="+.2f"),
                annotations=[dict(text="Exploratory, not a pass test. Sorted by the NQ/GC average. "
                                       "Faded bars: fewer than 30 trades.",
                                  x=0, y=-0.16, xref="paper", yref="paper", showarrow=False, xanchor="left",
                                  font=dict(color=MUTED, size=11))])
    return f1, f2


def empty_fig(msg=""):
    import plotly.graph_objects as go
    f = go.Figure()
    f.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=560, margin=dict(l=60, r=20, t=30, b=40),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    annotations=[dict(text=msg, showarrow=False, font=dict(color=INK2, size=14), x=.5, y=.5,
                                      xref="paper", yref="paper")] if msg else [])
    return f


def equity_fig(series):
    """series: list of (name, color, trades)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05,
                      subplot_titles=("Cumulative P&L, 1 contract ($)", "Drawdown from peak, closed trades ($)"))
    for name, color, t in series:
        if not len(t):
            continue
        x = pd.to_datetime(t.exit_time).dt.tz_convert(NY).dt.tz_localize(None)
        eq = t.pnl.cumsum().to_numpy()
        dd = -(np.maximum.accumulate(np.maximum(eq, 0)) - eq)
        Sc = go.Scattergl if len(t) > 4000 else go.Scatter
        cd = np.column_stack([t.pnl.to_numpy(), t.R.to_numpy()])
        f.add_trace(Sc(x=x, y=eq, mode="lines", name=name, legendgroup=name, line=dict(color=color, width=2),
                       customdata=cd, hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d %H:%M}} ET<br>"
                       "cum $%{y:,.0f}<br>trade $%{customdata[0]:,.0f} (%{customdata[1]:.2f}R)<extra></extra>"),
                    row=1, col=1)
        f.add_trace(Sc(x=x, y=dd, mode="lines", name=name, legendgroup=name, showlegend=False,
                       line=dict(color=color, width=2),
                       hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>drawdown $%{{y:,.0f}}<extra></extra>"),
                    row=2, col=1)
        breach = np.flatnonzero(-dd >= 1000)
        if len(breach):
            k = breach[0]
            f.add_trace(go.Scatter(x=[x.iloc[k]], y=[eq[k]], mode="markers", legendgroup=name, showlegend=False,
                                   marker=dict(symbol="x", size=10, color=color, line=dict(width=2, color=SURFACE)),
                                   hovertemplate=f"<b>{name}</b><br>eval ends here: -$1,000 trailing<br>"
                                   "%{x|%Y-%m-%d}<extra></extra>"), row=1, col=1)
        if len(series) <= 4:
            f.add_annotation(x=x.iloc[-1], y=eq[-1], text=f" {name}", showarrow=False, xanchor="left",
                             font=dict(color=INK2, size=11), row=1, col=1)
    f.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=1)
    f.add_hline(y=-1000, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1,
                annotation_text="-$1,000 eval limit", annotation_font=dict(color=MUTED, size=11),
                annotation_position="bottom left")
    f.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=620, hovermode="closest",
                    font=dict(color=INK2, family="system-ui, Segoe UI, sans-serif", size=12),
                    margin=dict(l=70, r=110, t=40, b=40),
                    legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK2)))
    f.update_annotations(font_color=INK2, selector=dict(text="Cumulative P&L, 1 contract ($)"))
    f.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), showspikes=True, spikecolor=MUTED,
                   spikethickness=1, spikedash="dot", spikemode="across")
    f.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), tickprefix="$",
                   tickformat=",.0f")
    return f


def make_app():
    from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
    app = Dash(__name__, title="Strategy Dashboard")
    app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")

    def lab(t):
        return html.Label(t, className="h")

    controls = html.Div(className="side", children=[
        html.Div("Strategy dashboard", style={"fontSize": 17, "fontWeight": 600}),
        lab("Data source"),
        dcc.RadioItems(id="source", value="databento", inline=True, inputStyle={"marginRight": 4},
                       labelStyle={"marginRight": 12},
                       options=[{"label": "Databento (Oct 2023 - Oct 2026)", "value": "databento"},
                                {"label": "Yahoo 5m (Aug - Oct 2026)", "value": "yahoo"}]),
        lab("Market"),
        dcc.RadioItems(id="market", value="MNQ", inline=True, options=list(MARKETS), inputStyle={"marginRight": 4},
                       labelStyle={"marginRight": 12}),
        lab("Session"),
        dcc.RadioItems(id="session", value="new_york", inline=True, inputStyle={"marginRight": 4},
                       labelStyle={"marginRight": 12},
                       options=[{"label": "NY 09:30-16:00", "value": "new_york"},
                                {"label": "Asia 19:00-03:00", "value": "asia"}]),
        lab("Strategies (up to 8)"),
        dcc.Dropdown(id="strats", multi=True, placeholder="pick setups or family configs", optionHeight=44,
                     maxHeight=420),
        lab("Reward : risk"),
        dcc.Slider(id="rr", min=1, max=4, step=0.5, value=2.0, marks={k: f"{k:g}" for k in (1, 1.5, 2, 2.5, 3, 3.5, 4)}),
        dcc.Checklist(id="flags", value=[], inputStyle={"marginRight": 6}, labelStyle={"display": "block", "marginTop": 4},
                      options=[{"label": "Breakeven stop at 1R", "value": "be"},
                               {"label": "Apply R:R to setups too (else each setup's own)", "value": "rr_setups"}]),
        lab("Date range (ET)"),
        dcc.DatePickerRange(id="dates", display_format="YYYY-MM-DD", minimum_nights=0),
        lab("Slippage (ticks/side)  ·  Commission ($/side)"),
        html.Div([dcc.Input(id="slip", type="number", value=1.0, min=0, step=0.25, debounce=True),
                  html.Span(" ", style={"display": "inline-block", "width": 12}),
                  dcc.Input(id="comm", type="number", value=2.50, min=0, step=0.25, debounce=True)]),
        html.Div(className="note", children=[
            "Signals close on a bar and fill next bar's open; stop 1 ATR(14); stop wins ties; "
            "flat at session end. 1 contract, no drawdown stop on the curve: the ✕ marks where a "
            "-$1,000 trailing closed-trade drawdown would end a prop-firm evaluation."]),
    ])

    main = html.Div(className="main", children=[
        html.Div(id="status", className="status"),
        html.Div(className="card", children=[
            dcc.Loading(dcc.Graph(id="equity", figure=empty_fig(), config={
                "displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "equity", "scale": 2}}),
                type="dot", color=MUTED),
            html.Div([html.Button("Save chart PNG", id="save-png"),
                      html.Button("Save summary CSV", id="save-sum"),
                      html.Button("Save trades CSV", id="save-trades")]),
        ]),
        html.Div(className="card", children=[
            dash_table.DataTable(id="summary", columns=[
                {"name": "Strategy", "id": "strategy"}, {"name": "Trades", "id": "trades"},
                {"name": "Win %", "id": "win_rate"}, {"name": "Avg R", "id": "avg_r"},
                {"name": "Net $", "id": "net", "type": "numeric", "format": {"specifier": "$,.0f"}},
                {"name": "Max DD $", "id": "max_dd", "type": "numeric", "format": {"specifier": "$,.0f"}},
                {"name": "Trades/day", "id": "trades_per_day"}, {"name": "-$1k eval ends", "id": "eval_end"}],
                data=[], style_cell_conditional=[{"if": {"column_id": "strategy"}, "textAlign": "left"}],
                style_data_conditional=[{"if": {"filter_query": "{net} < 0", "column_id": "net"}, "color": "#e66767"}],
                **TABLE_STYLE),
            html.Div(id="summary-note", className="note"),
        ]),
        html.Div(className="card", children=[
            html.Div([html.Button("Run the 6 validation tests", id="val-run", className="primary"),
                      html.Button("Save validation CSV", id="save-val")]),
            html.Div(id="val-status", className="status", style={"marginTop": 8}),
            dash_table.DataTable(id="val-table", data=[], columns=[
                {"name": n, "id": i} for n, i in [("Strategy", "strategy"), ("Config", "config"), ("Row", "note"),
                                                   ("Trades", "trades"), ("Net $", "net"), ("Initial", "initial"),
                                                   ("Walk-fwd OOS", "walk_forward"), ("PBO", "pbo"),
                                                   ("Defl. Sharpe", "deflated_sharpe"), ("Cost stress", "cost_stress"),
                                                   ("Nearby", "nearby"), ("All 6", "all")]],
                style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"} for c in ("strategy", "config", "note")],
                style_data_conditional=[s for c in ("initial", "walk_forward", "pbo", "deflated_sharpe", "cost_stress", "nearby", "all")
                                        for s in ({"if": {"filter_query": f'{{{c}}} contains "PASS"', "column_id": c}, "color": GOOD},
                                                  {"if": {"filter_query": f'{{{c}}} contains "FAIL"', "column_id": c}, "color": "#e66767"})],
                **TABLE_STYLE),
            html.Div(id="val-meta", className="note", children=[
                "Tests as in validate.py: initial (net > 0, never -$1,000, ≥ 20 trades), anchored 5-fold "
                "walk-forward, PBO (CSCV, 8 blocks, < 0.5), deflated Sharpe ≥ 0.95 over every grid trial, "
                "cost stress (all 4 scenarios survive), nearby settings (≥ 60% of one-step neighbours "
                "profitable). Each selected strategy is tested at its own setting; walk-forward and PBO test "
                "how its family's settings were chosen. The family's grid winner (validate.py's pick) is "
                "added for comparison. Runs the full grid on the current controls (300 backtests, "
                "a minute or two on 3 years)."]),
        ]),
        dcc.Interval(id="tick", interval=1000),
        dcc.Store(id="ready", data=""), dcc.Store(id="colors", data={}), dcc.Store(id="val-ver", data=-1),
        dcc.Download(id="dl"),
    ])
    def of_table(id_, page=None, filt=False):
        return dash_table.DataTable(
            id=id_, data=[], columns=[], page_size=page or 500, sort_action="native",
            filter_action="native" if filt else "none",
            style_filter={"backgroundColor": "#0d0d0d", "color": INK},
            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("market", "session", "level_type", "conf", "meets rule")],
            style_data_conditional=[{"if": {"filter_query": '{meets rule} = "YES"', "column_id": "meets rule"}, "color": GOOD},
                                    {"if": {"filter_query": '{meets rule} = "no"', "column_id": "meets rule"}, "color": "#e66767"}],
            **TABLE_STYLE)

    orderflow = html.Div(className="main", style={"padding": "0 16px 16px"}, children=[
        html.Div(id="of-status", className="status"),
        html.Div(className="note", style={"marginBottom": 12}, children=[
            "Cell colour, good to bad: average R ",
            html.Span(" ≥ +0.15 ", style={"background": HEAT["g2"], "color": INK}), " ",
            html.Span(" +0.05 to +0.15 ", style={"background": HEAT["g1"], "color": INK}),
            " · −0.05 to +0.05 unshaded · ",
            html.Span(" −0.15 to −0.05 ", style={"background": HEAT["r1"], "color": INK}), " ",
            html.Span(" ≤ −0.15 ", style={"background": HEAT["r2"], "color": INK}),
            "; $ columns green above 0, red below."]),
        html.Div(className="card", children=[
            html.Div("Rule, fixed in advance: at 2R, average R > 0 on both NQ and GC, ≥ 30 trades each, "
                     "and ≥ 0.1R better than B2 (price rejection) on the same subset", style={"fontWeight": 600}),
            of_table("of-rule"), html.Div(id="of-rule-note", className="note")]),
        html.Div(className="card", children=[html.Div("Pooled across sessions and level types", style={"fontWeight": 600}),
                                             of_table("of-overall")]),
        html.Div(className="card", children=[html.Div("Pooled across level types", style={"fontWeight": 600}),
                                             of_table("of-pooled")]),
        html.Div(className="card", children=[
            html.Div("Market × session × level type × confirmation (filter with the boxes under the headers)",
                     style={"fontWeight": 600}), of_table("of-detail", page=40, filt=True)]),
        html.Div(className="card", children=[
            html.Div("Confirmation count: score = how many of C1–C4 fire in the touch's 10-minute window; entry at the "
                     "close of the bar where the last one fires; score 0 enters on B2", style={"fontWeight": 600}),
            html.Div("Rule, fixed in advance: at 2R, Spearman(score, R) > 0 on NQ and on GC, and score ≥ 2 has "
                     "average R > 0 with ≥ 30 trades on each market", className="note"),
            of_table("of-score-rule"),
            dcc.Graph(id="of-score-fig", figure=empty_fig("waiting for the study ..."), config={"displaylogo": False}),
            html.Div("By score, pooled across sessions and level types (gross = no slippage or commission)",
                     className="note"), of_table("of-score"),
            html.Div("By score and session", className="note"), of_table("of-score-session")]),
        html.Div(className="card", children=[
            html.Div("Exploratory: not a pass test", style={"fontWeight": 600, "color": "#c98500"}),
            html.Div(id="of-combos-note", className="note"),
            dcc.Graph(id="of-combos-fig", figure=empty_fig("waiting for the study ..."), config={"displaylogo": False}),
            of_table("of-combos")]),
        html.Div(className="card", children=[html.Div("Data and aggressor-side check", style={"fontWeight": 600}),
                                             of_table("of-sanity"), html.Div(id="of-info", className="note")]),
        html.Div(className="note", children=[
            "C1 delta shift, C2 trapped traders, C3 aggressive push, C4 absorption; B1 plain touch and "
            "B2 price rejection are controls. Entry at the confirming 1m bar's close + 1 tick, stop = far zone "
            "edge ± 0.5 ATR(14, 5m), 1/2/3R targets side by side, stop wins ties, flat at session end, micro "
            "size (MNQ $2/pt, MGC $10/pt), $2.50/side. 'Top X%' is vs the trailing 10 sessions of the same "
            "type. avgR = average R per trade; R2 by month at 2R. Full rules: code/orderflow_study.py. "
            "Truncation test: run python code/orderflow_study.py."]),
    ])
    def card(title, *children, title_style=None):
        return html.Div(className="card", children=[html.Div(title, style={"fontWeight": 600, **(title_style or {})}), *children])

    graph = lambda id_, h=420: dcc.Graph(id=id_, figure=empty_fig("waiting ..."), config={"displaylogo": False},
                                         style={"height": h})
    systematic = html.Div(className="main", style={"padding": "0 16px 16px"}, children=[
        html.Div(id="sy-status", className="status"),
        html.Div(id="sy-variants", className="note", style={"marginBottom": 12}),
        *[card(title, *([graph(fig, 560 if fig == "sy-fig1" else 420)] +
                        [x for sub, key in subs for x in (html.Div(sub, className="note"), of_table(f"sy-{key}", page=60, filt=key == "t3_setups"))]))
          for title, subs, fig in SY_SECTIONS],
        html.Div(className="note", children=["Rules, data and costs: code/systematic_study.py docstring. Daily data is "
                                              "cached once in data/cache/daily/ (delete a file to refresh it)."]),
    ])
    gex_pane = html.Div(className="main", style={"padding": "0 16px 16px"}, children=[
        html.Div(className="card", children=[
            html.Div([
                dcc.RadioItems(id="gex-sym", value="QQQ", inline=True, options=["SPY", "QQQ", "^SPX"],
                               inputStyle={"marginRight": 4}, labelStyle={"marginRight": 14}),
                html.Span(" ", style={"display": "inline-block", "width": 24}),
                dcc.RadioItems(id="gex-view", value="all", inline=True, inputStyle={"marginRight": 4},
                               labelStyle={"marginRight": 14},
                               options=[{"label": v, "value": k} for k, v in GEX_VIEWS.items()]),
                html.Span(" ", style={"display": "inline-block", "width": 24}),
                dcc.RadioItems(id="gex-iv-type", value="otm", inline=True, inputStyle={"marginRight": 4},
                               labelStyle={"marginRight": 14},
                               options=[{"label": v, "value": k} for k, v in IV_TYPES.items()]),
                html.Button("Refresh", id="gex-refresh", className="primary", style={"marginLeft": 16})],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": 6}),
            html.Div(id="gex-status", className="status", style={"marginTop": 8}),
            html.Div(id="gex-notes", className="note")]),
        card("Levels (all three symbols, selected view) with NQ / ES equivalents", of_table("gex-levels")),
        card("Net GEX by strike", graph("gex-bars", 440)),
        card("Total GEX if spot moved (same IVs and times): the gamma flip is where it crosses zero", graph("gex-profile", 320)),
        card("Implied-volatility surface (pick calls, puts or OTM blend above; drag to rotate)", graph("gex-surface", 560)),
        card("Smile: nearest 3 expiries", graph("gex-smile", 380)),
        dcc.Store(id="gex-ver", data=-1),
    ])
    tabs = dcc.Tabs(id="tabs", value="strat", style={"margin": "0 16px"},
                    children=[dcc.Tab(label="Strategies", value="strat"), dcc.Tab(label="Order flow", value="of"),
                              dcc.Tab(label="Systematic", value="sys"), dcc.Tab(label="GEX", value="gex")])
    app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(style={"height": 12}), tabs,
                           html.Div(id="pane-strat", className="wrap", children=[controls, main]),
                           html.Div(id="pane-of", style={"display": "none"}, children=[orderflow, dcc.Store(id="of-ver", data=-1)]),
                           html.Div(id="pane-sys", style={"display": "none"}, children=[systematic, dcc.Store(id="sy-ver", data=-1)]),
                           html.Div(id="pane-gex", style={"display": "none"}, children=[gex_pane])])

    @app.callback(Output("tabs", "value"), Input("url", "search"))
    def tab_from_url(search):                      # e.g. http://127.0.0.1:8050/?tab=gex opens the GEX tab
        for t in ("of", "sys", "gex"):
            if search and f"tab={t}" in search:
                return t
        return no_update

    panes = ["strat", "of", "sys", "gex"]

    @app.callback(*[Output(f"pane-{p}", "style") for p in panes], Input("tabs", "value"))
    def switch(tab):
        return [{} if p == tab else {"display": "none"} for p in panes]

    # ---- Systematic tab
    sy_keys = [key for _, subs, _ in SY_SECTIONS for _, key in subs]

    @app.callback(Output("sy-status", "children"), Output("sy-variants", "children"),
                  Output("sy-fig1", "figure"), Output("sy-fig2", "figure"), Output("sy-fig3", "figure"),
                  *[Output(f"sy-{k}", p) for k in sy_keys for p in ("columns", "data", "style_data_conditional")],
                  Output("sy-ver", "data"), Input("tick", "n_intervals"), State("sy-ver", "data"))
    def sy_poll(_, ver):
        n_out = 5 + 3 * len(sy_keys) + 1
        if SY["state"] != "done":
            return [SY["msg"] or "systematic study not started"] + [no_update] * (n_out - 1)
        if ver == 1:
            return [no_update] * n_out
        R = SY["res"]
        out = ["Systematic study ready.", f"Variants tested. {R['variants']}", *sys_figs(R)]
        for k in sy_keys:
            cols, data = records(R[k])
            out += [cols, data, heat_rules([c["id"] for c in cols])]
        return out + [1]

    # ---- GEX tab
    @app.callback(Output("gex-status", "children", allow_duplicate=True), Input("gex-refresh", "n_clicks"),
                  prevent_initial_call=True)
    def gex_refresh(_):
        if GX["state"] == "running":
            return "already refreshing ..."
        threading.Thread(target=gex_job, daemon=True).start()
        return "refreshing (about 10-30 s) ..."

    @app.callback(Output("gex-status", "children"), Output("gex-ver", "data"),
                  Input("tick", "n_intervals"), State("gex-ver", "data"))
    def gex_poll(_, ver):
        if GX["state"] == "running":
            return GX["msg"] or "fetching option chains ...", no_update
        if GX["state"] == "error":
            return GX["msg"], no_update
        if GX["snap"] is None:
            return "no snapshot yet", no_update
        t = GX["snap"]["time"]
        msg = (f"Snapshot {t:%Y-%m-%d %H:%M} ET (data delayed). Risk-free {100 * GX['snap']['r']:.2f}% (^IRX). "
               f"{GX['saved']}")
        return msg, (GX["ver"] if ver != GX["ver"] else no_update)

    @app.callback(Output("gex-notes", "children"), Output("gex-levels", "columns"), Output("gex-levels", "data"),
                  Output("gex-levels", "style_data_conditional"), Output("gex-bars", "figure"),
                  Output("gex-profile", "figure"), Output("gex-surface", "figure"), Output("gex-smile", "figure"),
                  Input("gex-ver", "data"), Input("gex-sym", "value"), Input("gex-view", "value"),
                  Input("gex-iv-type", "value"))
    def gex_view(ver, sym, view, iv_type):
        import gex
        if GX["snap"] is None:
            e = empty_fig("waiting for the first snapshot ...")
            return gex.DEALER_NOTE + " " + gex.DELAY_NOTE, [], [], [], e, e, e, e
        cols, data = records(gex.levels_table(GX["snap"], view))
        rules = [{"if": {"filter_query": '{regime} = "positive gamma"', "column_id": "regime"}, "backgroundColor": HEAT["g1"]},
                 {"if": {"filter_query": '{regime} = "negative gamma"', "column_id": "regime"}, "backgroundColor": HEAT["r1"]},
                 {"if": {"filter_query": '{total GEX $bn per 1%} gt 0', "column_id": "total GEX $bn per 1%"}, "backgroundColor": HEAT["g1"]},
                 {"if": {"filter_query": '{total GEX $bn per 1%} lt 0', "column_id": "total GEX $bn per 1%"}, "backgroundColor": HEAT["r1"]}]
        return (gex.DEALER_NOTE + " " + gex.DELAY_NOTE, cols, data, rules, *gex_figs(GX["snap"], sym, view, iv_type))

    def records(df):
        df = df.copy()
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype(str)
        return [{"name": str(c), "id": str(c)} for c in df.columns], \
            df.astype(object).where(pd.notna(df), None).rename(columns=str).to_dict("records")

    of_tabs = {"of-rule": "rule", "of-overall": "overall", "of-pooled": "pooled", "of-detail": "detail",
               "of-sanity": "sanity", "of-score-rule": "score_rule", "of-score": "score",
               "of-score-session": "score_session", "of-combos": "combos_table"}
    n_text = 4

    @app.callback(Output("of-status", "children"), Output("of-rule-note", "children"), Output("of-info", "children"),
                  Output("of-combos-note", "children"), Output("of-score-fig", "figure"), Output("of-combos-fig", "figure"),
                  *[Output(i, p) for i in of_tabs for p in ("columns", "data", "style_data_conditional")],
                  Output("of-ver", "data"), Input("tick", "n_intervals"), State("of-ver", "data"))
    def of_poll(_, ver):
        n_out = n_text + 2 + 3 * len(of_tabs) + 1
        if OF["state"] != "done":
            return [OF["msg"] or "order-flow study not started"] + [no_update] * (n_out - 1)
        if ver == 1:
            return [no_update] * n_out
        R = OF["res"]
        n_yes = int((R["rule"]["meets rule"] == "YES").sum())
        out = [f"Order-flow study ready ({len(R['trades']):,} trades over {len(R['touches']):,} touches).",
               f"{R['combos']} rule tests (4 confirmations × {R['combos'] // 4} session/level groupings); "
               f"{n_yes} meet the rule. Nothing was tuned after seeing results.",
               " · ".join(R["info"]),
               f"All {len(R['combos_table'])} combinations of C1–C4 (6 pairs, 4 triples, all four). Every confirmation "
               f"in a combination must fire within the window; entry at the last one's bar close. Pooled across "
               f"sessions and level types; 2R target. Exploratory only: not a pass test."]
        out += list(of_figs(R))
        for name in of_tabs.values():
            cols, data = records(R[name])
            out += [cols, data, heat_rules([c["id"] for c in cols])]
        return out + [1]

    def current_key(source, mkt, session):
        return f"{source}|{mkt}|{session}"

    # ---- data range when source/market changes
    @app.callback(Output("dates", "min_date_allowed"), Output("dates", "max_date_allowed"),
                  Output("dates", "start_date"), Output("dates", "end_date"),
                  Input("source", "value"), Input("market", "value"))
    def set_dates(source, mkt):
        try:
            raw, _ = load(source, mkt)
        except Exception:
            return no_update, no_update, no_update, no_update
        lo, hi = raw.index[0].tz_convert(NY).date(), raw.index[-1].tz_convert(NY).date()
        return lo, hi, lo, hi

    # ---- signal status / readiness
    @app.callback(Output("status", "children"), Output("ready", "data"),
                  Input("tick", "n_intervals"), Input("source", "value"), Input("market", "value"),
                  Input("session", "value"), State("ready", "data"))
    def status(_, source, mkt, session, ready):
        key = (source, mkt, session)
        k = current_key(*key)
        if source == "databento" and not has_data(source, mkt):
            return no_data_msg(f"Databento {mkt} 1m files", "data/databento_ohlcv_1m/"), ("" if ready else no_update)
        if source == "yahoo" and MARKETS[mkt]["yahoo"] and not has_data(source, mkt):
            return no_data_msg(f"Yahoo {mkt} 5m file", f"data/{MARKETS[mkt]['yahoo']}"), ("" if ready else no_update)
        try:
            raw, nbad = load(source, mkt)
            fut = pack_future(key)
            pack_future((source, mkt, "asia" if session == "new_york" else "new_york"), front=False)
        except Exception as exc:
            return f"⚠ {exc}", ("" if ready else no_update)
        info = (f"{source.title()} {mkt}: {len(raw):,} 5m bars, {raw.index[0].tz_convert(NY):%Y-%m-%d} to "
                f"{raw.index[-1].tz_convert(NY):%Y-%m-%d}, {nbad} bad prints repaired.")
        par = f"  {PARITY['msg']}" if PARITY["state"] == "fail" else ""
        if not fut.done():
            if key not in PACK_T0:
                return f"{info}  Signals for {mkt} {SESS_LABEL[session]} queued ...{par}", ("" if ready else no_update)
            el = time.time() - PACK_T0[key]
            n, N = PACK_PROG.get(key, (0, 0))
            return (f"{info}  Building every family's signals for {mkt} {SESS_LABEL[session]} once per app "
                    f"session: {n}/{N or '...'} signal calls, {int(el // 60)}:{int(el % 60):02d} elapsed. "
                    f"Family configs appear when it's done.{par}"), ("" if ready else no_update)
        if fut.exception():
            return f"⚠ signal build failed: {fut.exception()!r}{par}", ("" if ready else no_update)
        p = fut.result()
        how = "loaded from the signal cache" if p.get("cached") else f"built in {p['seconds']:.0f}s and cached"
        return (f"{info}  Signals ready ({len(p['entries'])} builders, {how}). "
                f"{PARITY['msg']}"), (k if ready != k else no_update)

    # ---- strategy options and defaults
    @app.callback(Output("strats", "options"), Output("strats", "value"),
                  Input("ready", "data"), Input("market", "value"), Input("session", "value"),
                  State("strats", "value"))
    def options(ready, mkt, session, cur):
        asset = MARKETS[mkt]["asset"]
        setups = [s for s in study.SETUPS if s[2] == asset]
        setups.sort(key=lambda s: s[3] != session)
        opts = [{"label": f"{s[0]}  ·  {s[2]} {SESS_LABEL[s[3]]} setup, {s[6]:g}R", "value": f"S:{s[0]}"} for s in setups]
        if ready:
            pack = PACKS[tuple(ready.split("|"))].result()
            opts += [{"label": entry_label(e), "value": f"F:{i}"} for i, e in enumerate(pack["entries"]) if e["in_grid"]]
        valid = {o["value"] for o in opts}
        trig = ctx.triggered_id
        if trig in ("market", "session") or cur is None:
            value = [f"S:{s[0]}" for s in setups if s[3] == session][:3]
        else:
            value = [c for c in cur if c in valid] or no_update
        return opts, value

    # ---- stable colour slot per strategy (colour follows the strategy, not its rank)
    @app.callback(Output("colors", "data"), Input("strats", "value"), State("colors", "data"))
    def colors(sel, cur):
        sel = (sel or [])[:8]
        keep = {k: c for k, c in (cur or {}).items() if k in sel}
        for s in sel:
            if s not in keep:
                keep[s] = min(set(range(8)) - set(keep.values()))
        return {"order": sel, "slot": keep}

    def compute(ready, colors, rr, flags, start, end, slip, comm):
        key = tuple(ready.split("|"))
        source, mkt, session = key
        pack = PACKS[key].result()
        cfg = base_cfg(mkt, session, slip if slip is not None else 1.0, comm if comm is not None else 2.5)
        base = dict(NOKILL, **({"breakeven_r": 1.0} if "be" in (flags or []) else {}))
        a, b = date_bounds(start, end)
        picks = [(sid, *resolve(pack, sid, float(rr), "rr_setups" in (flags or []))) for sid in colors["order"]]
        R = run_many(key, cfg, base, a, b, [(i, r, ()) for _, _, i, r in picks])
        ndays = session_days(source, mkt, session, a, b)
        out = [(sid, name, i, r, with_r(R[(i, r, ())], cfg)) for sid, name, i, r in picks]
        return out, ndays, cfg

    @app.callback(Output("equity", "figure"), Output("summary", "data"), Output("summary-note", "children"),
                  Input("ready", "data"), Input("colors", "data"), Input("rr", "value"), Input("flags", "value"),
                  Input("dates", "start_date"), Input("dates", "end_date"), Input("slip", "value"),
                  Input("comm", "value"), State("source", "value"), State("market", "value"), State("session", "value"))
    def chart(ready, colors, rr, flags, start, end, slip, comm, source, mkt, session):
        if not ready or ready != current_key(source, mkt, session) or not start or not end:
            return empty_fig("waiting for signals ..."), [], ""
        if not colors or not colors.get("order"):
            return empty_fig("pick one or more strategies"), [], ""
        out, ndays, cfg = compute(ready, colors, rr, flags, start, end, slip, comm)
        series = [(name if sid.startswith("S:") else name, SERIES[colors["slot"][sid]], t) for sid, name, i, r, t in out]
        rows = []
        for sid, name, i, r, t in out:
            s = summarize(name, t, ndays)
            s["strategy"] = f"{name} ({r:g}R)"
            rows.append(s)
        note = (f"{len(ndays)} {SESS_LABEL[session]} session days. Avg R = P&L / (1 ATR stop + both sides' slippage "
                f"+ commission). Max DD is trailing from the closed-trade peak (floored at 0), as in validate.stats.")
        return equity_fig(series), rows, note

    # ---- saves (only on click)
    app.clientside_callback(
        """function(n){ if(!n) return window.dash_clientside.no_update;
            const g = document.querySelector('#equity .js-plotly-plot');
            if (g) Plotly.downloadImage(g, {format:'png', width:1600, height:900, filename:'equity'});
            return window.dash_clientside.no_update; }""",
        Output("save-png", "title"), Input("save-png", "n_clicks"), prevent_initial_call=True)

    @app.callback(Output("dl", "data"), Input("save-sum", "n_clicks"), Input("save-trades", "n_clicks"),
                  Input("save-val", "n_clicks"), State("summary", "data"), State("val-table", "data"),
                  State("ready", "data"), State("colors", "data"), State("rr", "value"), State("flags", "value"),
                  State("dates", "start_date"), State("dates", "end_date"), State("slip", "value"),
                  State("comm", "value"), prevent_initial_call=True)
    def save(_a, _b, _c, summary, val, ready, colors, rr, flags, start, end, slip, comm):
        tag = ready.replace("|", "_") if ready else "dashboard"
        if ctx.triggered_id == "save-sum" and summary:
            return dcc.send_data_frame(pd.DataFrame(summary).to_csv, f"summary_{tag}.csv", index=False)
        if ctx.triggered_id == "save-val" and val:
            return dcc.send_data_frame(pd.DataFrame(val).to_csv, f"validation_{tag}.csv", index=False)
        if ctx.triggered_id == "save-trades" and ready and colors and colors.get("order"):
            out, _, _ = compute(ready, colors, rr, flags, start, end, slip, comm)
            df = pd.concat([t.assign(strategy=name, rr=r) for sid, name, i, r, t in out], ignore_index=True)
            return dcc.send_data_frame(df.to_csv, f"trades_{tag}.csv", index=False)
        return no_update

    # ---- validation (background thread, polled)
    @app.callback(Output("val-status", "children", allow_duplicate=True), Input("val-run", "n_clicks"),
                  State("ready", "data"), State("colors", "data"), State("rr", "value"), State("flags", "value"),
                  State("dates", "start_date"), State("dates", "end_date"), State("slip", "value"),
                  State("comm", "value"), prevent_initial_call=True)
    def val_start(_, ready, colors, rr, flags, start, end, slip, comm):
        if not ready:
            return "signals are not ready yet"
        if not colors or not colors.get("order"):
            return "pick at least one strategy first"
        if VAL["state"] == "running":
            return "a validation run is already going"
        key = tuple(ready.split("|"))
        pack = PACKS[key].result()
        picks = [resolve(pack, sid, float(rr), "rr_setups" in (flags or [])) for sid in colors["order"]]
        VAL.update(id=VAL["id"] + 1, state="running", msg="starting ...", rows=[], meta="")
        threading.Thread(target=run_validation, daemon=True,
                         args=(VAL["id"], key, picks, slip if slip is not None else 1.0,
                               comm if comm is not None else 2.5, "be" in (flags or []), start, end)).start()
        return "starting ..."

    @app.callback(Output("val-status", "children"), Output("val-table", "data"), Output("val-meta", "children"),
                  Output("val-ver", "data"), Input("tick", "n_intervals"), State("val-ver", "data"),
                  prevent_initial_call=True)
    def val_poll(_, ver):
        if VAL["state"] == "running":
            return f"running: {VAL['msg']}", no_update, no_update, no_update
        if VAL["state"] == "error":
            return VAL["msg"], no_update, no_update, no_update
        if VAL["state"] == "done" and ver != VAL["id"]:
            return "", VAL["rows"], VAL["meta"], VAL["id"]
        return no_update, no_update, no_update, no_update

    return app


SY_SECTIONS = [
    ("Test 1: overnight vs intraday (after costs)",
     [("Rule: overnight beats intraday on annual return, and overnight > 0 in both halves of the history", "t1_pass"),
      ("Results", "t1"), ("By decade", "t1_decades")], "sy-fig1"),
    ("Test 2: time-series momentum, 24 futures (after 2 bp per side)",
     [("Rule: blend Sharpe > 0.3 in both halves, deflated Sharpe ≥ 0.95, more than half the markets contribute > 0", "t2_pass"),
      ("Results", "t2"), ("Contribution by market, % of capital per year", "t2_contrib"), ("By decade", "t2_decades"),
      ("Data and roll gaps flagged (|open gap| > 4 × 60-day vol near contract months)", "t2_rolls"),
      ("Cross-check, not a pass test: NQ and GC alone, Databento vs Yahoo, same dates", "t2_cross")], "sy-fig2"),
    ("Test 3: VIX regime on every setup trade (prior day's VIX vs its trailing 1-year median)",
     [("Rule: one regime has pooled average R > 0 in both 2024 and 2025", "t3_pass"),
      ("Pooled by regime and year", "t3"), ("Per setup, 2024-2026 (filter with the boxes under the headers)", "t3_setups")],
     "sy-fig3"),
]
GEX_VIEWS = {"all": "all expiries ≤ 60 days", "0dte": "0DTE only", "nearest": "nearest expiry only"}
IV_TYPES = {"C": "Calls", "P": "Puts", "otm": "OTM blend"}
IV_KEYS = {"C": "Calls: strike/spot below 1.0 = in the money, above 1.0 = out of the money.",
           "P": "Puts: strike/spot above 1.0 = in the money, below 1.0 = out of the money.",
           "otm": "OTM blend: puts are used below 1.0 and calls above 1.0, so every point is an out-of-the-money option."}
SY =dict(state="idle", msg="", res=None)
GX = dict(state="idle", msg="", snap=None, ver=0, saved="")


def sy_job():
    try:
        import systematic_study
        SY.update(state="running", msg="Systematic study running (about 1-2 minutes) ...")
        SY["res"] = systematic_study.run_all(log=lambda m: SY.update(msg=f"Systematic study: {m.strip()}"))
        SY["state"] = "done"
    except Exception as exc:
        SY.update(state="error", msg=f"⚠ systematic study failed: {exc!r}")


def gex_job():
    try:
        import gex
        GX.update(state="running", msg="Fetching option chains from Yahoo ...")
        snap = gex.snapshot(log=lambda m: GX.update(msg=f"Fetching option chains: {m.strip()}"))
        msgs = []
        gex.save_daily(snap, log=lambda m: msgs.append(m.strip()))
        GX.update(snap=snap, ver=GX["ver"] + 1, state="done", saved=" ".join(msgs))
    except Exception as exc:
        GX.update(state="error", msg=f"⚠ GEX fetch failed: {exc!r}")


def sys_figs(R):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    names = list(R["t1_equity"])
    f1 = make_subplots(rows=2, cols=2, subplot_titles=names, vertical_spacing=0.12, horizontal_spacing=0.07)
    for i, name in enumerate(names):
        eq = R["t1_equity"][name]
        for j, (k, lab) in enumerate((("on", "overnight only"), ("id", "intraday only"), ("bh", "buy and hold"))):
            f1.add_trace(go.Scatter(x=eq.index, y=eq[k], name=lab, legendgroup=lab, showlegend=i == 0,
                                    line=dict(color=SERIES[j], width=1.6),
                                    hovertemplate=f"{name} {lab}<br>%{{x|%Y-%m-%d}}: ×%{{y:.2f}}<extra></extra>"),
                         row=i // 2 + 1, col=i % 2 + 1)
    f1.update_yaxes(type="log")
    f2 = go.Figure()
    for j, k in enumerate(("3m", "6m", "12m", "blend")):
        eq = R["t2_equity"][("roll days zeroed", k)]
        f2.add_trace(go.Scatter(x=eq.index, y=eq, name=f"{k} (roll days zeroed)",
                                line=dict(color=SERIES[j], width=2.4 if k == "blend" else 1.4)))
    eq = R["t2_equity"][("as-is", "blend")]
    f2.add_trace(go.Scatter(x=eq.index, y=eq, name="blend (as-is)", line=dict(color=INK2, width=1.6, dash="dot")))
    f3 = go.Figure()
    for j, (reg, eq) in enumerate(sorted(R["t3_equity"].items())):
        f3.add_trace(go.Scatter(x=eq.index.tz_convert(NY).tz_localize(None), y=eq.to_numpy(), name=reg,
                                line=dict(color=SERIES[j], width=2),
                                hovertemplate=f"{reg}<br>%{{x|%Y-%m-%d}}: %{{y:.1f}}R<extra></extra>"))
    for f, h, yt in ((f1, 560, "growth of $1 (log)"), (f2, 420, "growth of $1"), (f3, 420, "cumulative R")):
        f.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=h, hovermode="closest",
                        font=dict(color=INK2, family="system-ui, Segoe UI, sans-serif", size=12),
                        legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=60, r=20, t=50, b=40))
        f.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED))
        f.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED))
        if f is not f1:
            f.update_yaxes(title=yt)
    f1.update_annotations(font_color=INK2)
    f3.add_hline(y=0, line=dict(color=AXIS, width=1))
    return f1, f2, f3


def gex_figs(snap, sym, view, iv_type="otm"):
    """(net GEX by strike, GEX profile, IV surface, smile) for one symbol and view; iv_type C / P / otm."""
    import plotly.graph_objects as go
    import gex
    s = snap["symbols"][sym]
    lv = s["views"].get(view)
    base = dict(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="closest",
                font=dict(color=INK2, family="system-ui, Segoe UI, sans-serif", size=12), margin=dict(l=70, r=20, t=40, b=50))
    if s.get("oi_coverage", 1) < 0.5:
        e1 = empty_fig(f"{sym}: Yahoo gives open interest for only {100 * s['oi_coverage']:.1f}% of contracts, so GEX is not computed")
        e2 = empty_fig("")
    elif lv is None:
        e1 = e2 = empty_fig(f"{sym}: no contracts for this view (e.g. no 0DTE expiry today)")
    else:
        by = lv["by_strike"]
        fut = lambda x: f" ({s['fut']} {x * s['ratio']:,.0f})"
        f1 = go.Figure(go.Bar(x=by.strike, y=by.net / 1e9, marker_color=np.where(by.net >= 0, "#0ca30c", "#d03b3b"),
                              customdata=np.column_stack([by.call / 1e9, by.put / 1e9]),
                              hovertemplate="strike %{x}<br>net %{y:.3f} $bn/1%<br>calls %{customdata[0]:.3f}, "
                                            "puts %{customdata[1]:.3f}<extra></extra>"))
        marks = [("spot", lv["spot"], INK, "dash"), ("call wall", lv["call_wall"], "#0ca30c", "dot"),
                 ("put wall", lv["put_wall"], "#e66767", "dot"), ("gamma flip", lv["flip"], "#c98500", "dashdot")]
        for k, (name, x, col, dash) in enumerate(marks):
            if x is not None:
                f1.add_vline(x=x, line=dict(color=col, width=1.5, dash=dash))
                f1.add_annotation(x=x, y=1.02 + 0.055 * k, yref="paper", text=f"{name} {x:,.1f}{fut(x)}", showarrow=False,
                                  font=dict(color=col, size=11), bgcolor=SURFACE)
        f1.update_layout(**{**base, "margin": dict(l=70, r=20, t=110, b=50)}, height=440, bargap=0.1, xaxis_title=f"{sym} strike",
                         yaxis_title="net GEX, $bn per 1% move", showlegend=False)
        e1 = f1
        p = lv["profile"]
        f2 = go.Figure(go.Scatter(x=p.spot, y=p.total_gex / 1e9, mode="lines", line=dict(color=SERIES[0], width=2),
                                  hovertemplate="spot %{x:,.2f}<br>total %{y:.3f} $bn/1%<extra></extra>"))
        f2.add_hline(y=0, line=dict(color=INK2, width=1))
        f2.add_vline(x=lv["spot"], line=dict(color=INK, width=1, dash="dash"))
        if lv["flip"] is not None:
            f2.add_vline(x=lv["flip"], line=dict(color="#c98500", width=1.5, dash="dashdot"))
        f2.update_layout(**base, height=320, xaxis_title=f"{sym} spot (±5%)", yaxis_title="total GEX, $bn per 1%",
                         showlegend=False)
        e2 = f2
    sf = s["surface"]                                   # OTM blend, used by the smile chart
    sv = gex.surface(s["chain"], s["spot"], iv_type)    # the type picked for the 3D surface
    if len(sv["dte"]):
        Z = 100 * sv["iv"]
        text = [[f"expiry {exp} ({d} days)<br>strike/spot {m:.2f}<br>IV {z:.1f}%<br>"
                 + (f"listed strike {lo:g}" if lo == hi else f"interpolated between strikes {lo:g} and {hi:g}")
                 for m, z, lo, hi in zip(sv["moneyness"], zr, lr, hr)]
                for exp, d, zr, lr, hr in zip(sv["expiry"], sv["dte"], Z, sv["lo"], sv["hi"])]
        z0, z1 = float(np.nanmin(Z)), float(np.nanmax(Z))
        d0, d1 = float(sv["dte"].min()), float(max(sv["dte"].max(), sv["dte"].min() + 1))
        f3 = go.Figure([
            go.Surface(x=sv["moneyness"], y=sv["dte"], z=Z, colorscale="Viridis", colorbar=dict(title="IV %"),
                       name=f"IV ({IV_TYPES[iv_type]})", showlegend=True,
                       customdata=np.array(text, dtype=object),   # 2D `text` gets flattened to strings by plotly 7
                       hovertemplate="%{customdata}<extra></extra>"),
            go.Surface(x=[[1, 1], [1, 1]], y=[[d0, d1], [d0, d1]], z=[[z0, z0], [z1, z1]], opacity=0.3,
                       colorscale=[[0, INK2], [1, INK2]], showscale=False, name="At the money (1.0)",
                       showlegend=True, hovertemplate="at the money: strike = spot<extra></extra>")])
        f3.add_annotation(text=IV_KEYS[iv_type], x=0, y=0, xref="paper", yref="paper", xanchor="left",
                          yanchor="bottom", showarrow=False, font=dict(color=INK2, size=12), bgcolor=SURFACE)
        f3.update_layout(**{**base, "margin": dict(l=0, r=0, t=60, b=0)}, height=560,
                         title=dict(text=f"{sym} implied vol: {IV_TYPES[iv_type]}"
                                         f"<br><sup>as of {snap['time']:%Y-%m-%d %H:%M} ET (Yahoo, delayed)</sup>",
                                    x=0.01, font=dict(color=INK)),
                         legend=dict(x=0.01, y=0.9, bgcolor=SURFACE),
                         scene=dict(xaxis_title="strike / spot", yaxis_title="days to expiry", zaxis_title="IV %",
                                    xaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED),
                                    yaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED),
                                    zaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED)))
    else:
        f3 = empty_fig(f"no usable implied volatilities for {IV_TYPES[iv_type].lower()}")
    if len(sf["dte"]):
        f4 = go.Figure()
        for j, (exp, g) in enumerate(sf["smiles"].items()):
            f4.add_trace(go.Scatter(x=g.strike, y=100 * g.iv, mode="lines+markers", name=exp,
                                    line=dict(color=SERIES[j], width=2), marker=dict(size=5)))
        f4.add_vline(x=s["spot"], line=dict(color=INK, width=1, dash="dash"))
        f4.update_layout(**base, height=380, xaxis_title=f"{sym} strike (puts below spot, calls above)",
                         yaxis_title="implied vol %", legend=dict(orientation="h", y=1.1, x=0))
    else:
        f4 = empty_fig("no usable implied volatilities")
    return e1, e2, f3, f4


OF = dict(state="idle", msg="", res=None)


def of_job():
    """Order-flow tab: code/orderflow_study.py on its TBBO cache (built on first use)."""
    try:
        import orderflow_study
        OF.update(state="running", msg="Order-flow study running (about a minute from the TBBO cache) ...")
        OF["res"] = orderflow_study.run_all(log=lambda m: OF.update(msg=f"Order-flow study: {m.strip()}"))
        OF["state"] = "done"
    except Exception as exc:
        OF.update(state="error", msg=f"⚠ order-flow study failed: {exc!r}")


def main():
    db = [m for m in MARKETS if has_data("databento", m)]   # markets without data are skipped, not fatal
    if len(db) < len(MARKETS):
        print(f"no Databento 1m files for {', '.join(m for m in MARKETS if m not in db)} under {DB_DIR}; "
              f"those markets are skipped (the GEX tab needs no local data)", flush=True)
    for mkt in db:                # build any missing price cache up front (a few seconds each)
        ensure_cache(mkt)
    if "MNQ" in db:
        pack_future(("databento", "MNQ", "new_york"))   # start the default market's signals right away
    if has_data("yahoo", "MNQ"):
        for sess in ("new_york", "asia"):                # small Yahoo packs for the parity check next
            pack_future(("yahoo", "MNQ", sess), front=False)
        threading.Thread(target=check_parity, daemon=True).start()
    for mkt in db:                                   # then everything else, behind what's on screen
        for sess in ("new_york", "asia"):
            pack_future(("databento", mkt, sess), front=False)
    if any(TBBO_DIR.rglob("*.zst")) or any(CACHE.glob("tbbo_*_front.pkl.zst")):
        threading.Thread(target=of_job, daemon=True).start()
    else:
        OF.update(state="nodata", msg=no_data_msg("Databento TBBO files", "data/databento_tbbo/"))
    if {"NQ", "GC"} <= set(db):
        threading.Thread(target=sy_job, daemon=True).start()
    else:
        SY.update(state="nodata", msg=no_data_msg("Databento NQ and GC 1m files", "data/databento_ohlcv_1m/"))
    threading.Thread(target=gex_job, daemon=True).start()
    app = make_app()
    print("open http://127.0.0.1:8050", flush=True)
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
