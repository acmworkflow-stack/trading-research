"""Gamma exposure (GEX) and implied-volatility surface for SPY, QQQ and ^SPX from Yahoo option chains.

    python code/gex.py            fetch now, print the levels, save today's snapshot if missing

Writes nothing except one compact snapshot per day, data/cache/gex/gex_YYYY-MM-DD.pkl.zst (ET date):
the levels plus every contract's open interest, implied volatility, bid and days to expiry, and each
symbol's spot, to build our own history (load_daily() / saved_surface() rebuild the IV surface from it).
A snapshot is skipped when today's file already exists, and is not saved while SPY or QQQ open
interest is mostly missing (Yahoo blanks chains while it resets them around midnight ET). code/app.py shows all of this in its "GEX" tab.

Method
  * chains: every expiry within 60 days, every strike within +/-15% of spot; strike, expiry, type,
    open interest, volume, implied volatility, bid/ask
  * Black-Scholes gamma per contract from Yahoo's implied volatility, risk-free rate = ^IRX (13-week
    T-bill), no dividend yield, expiry at 16:00 ET on the expiry date (at least 1 hour left)
  * GEX per contract = gamma x OI x 100 x spot^2 x 0.01 ($ of delta change per 1% move), calls
    positive and puts negative. ASSUMPTION: dealers are long the calls and short the puts customers
    trade (the usual convention); real dealer positioning is not observable from this data.
  * call wall = strike with the largest call GEX; put wall = strike with the largest (most negative)
    put GEX; gamma flip = where total GEX, recomputed for spot moved -5%..+5% in 0.25% steps (same
    IVs and times), crosses zero, the crossing nearest spot; positive gamma when total GEX > 0
  * levels in NQ / ES points: QQQ x (NQ=F / QQQ), SPY x (ES=F / SPY), SPX x (ES=F / ^GSPC), all at
    the snapshot's last prices
  * views: all expiries <= 60 days, 0DTE only, nearest expiry only
  * a symbol's GEX is not shown when fewer than 50% of its contracts carry open interest (Yahoo's
    ^SPX chain has implied volatility but almost no open interest); its vol surface still is
Yahoo data is delayed (typically 15+ minutes), and open interest is the exchanges' figure from the
prior session, so intraday changes in positioning are not visible.
"""
from __future__ import annotations
import sys, time, pickle
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from compression import zstd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
GEX_DIR = ROOT / "data" / "cache" / "gex"
NY = "America/New_York"
SYMBOLS = ["SPY", "QQQ", "^SPX"]
MAX_DTE, BAND, FLIP_BAND, FLIP_STEP = 60, 0.15, 0.05, 0.0025
MIN_OI_COVERAGE = 0.5      # below this share of contracts with open interest, GEX is not shown (Yahoo's ^SPX)
FUT = {"SPY": ("ES", "ES=F", "SPY"), "QQQ": ("NQ", "NQ=F", "QQQ"), "^SPX": ("ES", "ES=F", "^GSPC")}
VIEWS = {"all": "all expiries ≤ 60 days", "0dte": "0DTE only", "nearest": "nearest expiry only"}
DEALER_NOTE = ("Dealer-positioning assumption: dealers are long the calls and short the puts that customers "
               "trade, so call GEX counts positive and put GEX negative. Real dealer positions are not observable.")
DELAY_NOTE = ("Yahoo option quotes are delayed (typically 15+ minutes); open interest is the prior session's "
              "figure, so intraday positioning changes are not visible.")


def last_price(sym: str) -> float:
    import yfinance as yf
    t = yf.Ticker(sym)
    try:
        p = float(t.fast_info["last_price"])
        if np.isfinite(p) and p > 0:
            return p
    except Exception:
        pass
    return float(t.history(period="5d")["Close"].dropna().iloc[-1])


def fetch_chain(sym: str, now: pd.Timestamp) -> tuple[float, pd.DataFrame]:
    import yfinance as yf
    tk = yf.Ticker(sym)
    spot = last_price(sym)
    today = now.date()
    exps = [e for e in tk.options if 0 <= (date.fromisoformat(e) - today).days <= MAX_DTE]

    def one(e):
        oc = tk.option_chain(e)
        out = []
        for typ, df in (("C", oc.calls), ("P", oc.puts)):
            df = df[(df.strike >= spot * (1 - BAND)) & (df.strike <= spot * (1 + BAND))]
            out.append(pd.DataFrame({"expiry": e, "type": typ, "strike": df.strike.astype(float),
                                     "oi": df.openInterest.fillna(0).astype(float), "volume": df.volume.fillna(0).astype(float),
                                     "iv": df.impliedVolatility.astype(float), "bid": df.bid.astype(float),
                                     "ask": df.ask.astype(float)}))
        return pd.concat(out, ignore_index=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        parts = list(ex.map(one, exps))
    chain = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["expiry", "type", "strike", "oi", "volume", "iv", "bid", "ask"])
    exp_t = pd.to_datetime(chain.expiry).dt.tz_localize(NY) + pd.Timedelta(hours=16)
    chain["dte"] = (pd.to_datetime(chain.expiry) - pd.Timestamp(today)).dt.days
    chain["T"] = np.maximum((exp_t - now).dt.total_seconds(), 3600) / (365 * 24 * 3600)
    return spot, chain


def bs_gamma(S, K, T, sigma, r):
    S, K, T, sigma = (np.asarray(x, float) for x in (S, K, T, sigma))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * sigma * np.sqrt(T))


def usable(chain: pd.DataFrame) -> pd.DataFrame:
    return chain[(chain.iv > 0.01) & (chain.iv < 5) & (chain.oi > 0)]


def total_gex(c: pd.DataFrame, S: float, r: float) -> float:
    g = bs_gamma(S, c.strike, c["T"], c.iv, r)
    sign = np.where(c.type == "C", 1.0, -1.0)
    return float(np.nansum(sign * g * c.oi * 100 * S * S * 0.01))


def levels(chain: pd.DataFrame, S: float, r: float) -> dict | None:
    c = usable(chain)
    if c.empty:
        return None
    sign = np.where(c.type == "C", 1.0, -1.0)
    c = c.assign(gex=sign * bs_gamma(S, c.strike, c["T"], c.iv, r) * c.oi * 100 * S * S * 0.01)
    by = c.pivot_table(index="strike", columns="type", values="gex", aggfunc="sum", fill_value=0.0)
    by = by.reindex(columns=["C", "P"], fill_value=0.0).rename(columns={"C": "call", "P": "put"})
    by["net"] = by.call + by.put
    grid = S * (1 + np.arange(-FLIP_BAND, FLIP_BAND + FLIP_STEP / 2, FLIP_STEP))
    prof = np.array([total_gex(c, s, r) for s in grid])
    flips = []
    for i in range(len(grid) - 1):
        if prof[i] == 0 or np.sign(prof[i]) != np.sign(prof[i + 1]):
            flips.append(grid[i] + (grid[i + 1] - grid[i]) * prof[i] / (prof[i] - prof[i + 1]) if prof[i] != prof[i + 1] else grid[i])
    flip = min(flips, key=lambda x: abs(x - S)) if flips else None
    total = float(c.gex.sum())
    return dict(spot=S, total=total, regime="positive gamma" if total > 0 else "negative gamma",
                call_wall=float(by.call.idxmax()) if (by.call > 0).any() else None,
                put_wall=float(by.put.idxmin()) if (by.put < 0).any() else None,
                flip=None if flip is None else float(flip), by_strike=by.reset_index(),
                profile=pd.DataFrame({"spot": grid, "total_gex": prof}), contracts=len(c),
                expiries=sorted(c.expiry.unique()))


def view_chain(chain: pd.DataFrame, view: str) -> pd.DataFrame:
    if view == "0dte":
        return chain[chain.dte == 0]
    if view == "nearest":
        return chain[chain.dte == chain.dte.min()] if len(chain) else chain
    return chain


def surface(chain: pd.DataFrame, S: float, kind: str = "otm") -> dict:
    """Implied vol on a moneyness grid, one row per expiry (expiries are never interpolated).
    kind: "otm" (puts below spot, calls above), "C" (calls only) or "P" (puts only).
    IV between listed strikes is linearly interpolated; lo/hi are the listed strikes bracketing each
    grid point (lo == hi where a listed strike sits exactly on the grid point)."""
    c = chain[(chain.iv > 0.01) & (chain.iv < 5) & (chain.bid > 0)]
    if kind == "otm":
        c = c[((c.type == "C") & (c.strike >= S)) | ((c.type == "P") & (c.strike < S))]
    else:
        c = c[c.type == kind]
    c = c.copy()
    c["m"] = c.strike / S
    grid = np.round(np.arange(1 - BAND, 1 + BAND + 1e-9, 0.01), 3)
    rows = []
    for (exp, dte), g in c.groupby(["expiry", "dte"]):
        g = g.sort_values("m")
        if len(g) < 5:
            continue
        m, k = g.m.to_numpy(), g.strike.to_numpy()
        z = np.interp(grid, m, g.iv, left=np.nan, right=np.nan)
        i = np.clip(np.searchsorted(m, grid), 0, len(m) - 1)
        exact = np.isclose(m[i], grid, rtol=0, atol=1e-9)
        lo = np.where(exact, k[i], k[np.maximum(i - 1, 0)])
        rows.append((int(dte), exp, z, lo, k[i]))
    rows.sort(key=lambda r: r[0])
    smiles = {exp: g.sort_values("strike")[["strike", "iv", "type"]] for exp, g in
              sorted(c.groupby("expiry"), key=lambda kv: kv[0])[:3]}
    col = lambda j: np.array([r[j] for r in rows]) if rows else np.empty((0, len(grid)))
    return dict(moneyness=grid, dte=np.array([r[0] for r in rows]), expiry=[r[1] for r in rows],
                iv=col(2), lo=col(3), hi=col(4), smiles=smiles)


def snapshot(log=print) -> dict:
    now = pd.Timestamp.now(tz=NY)
    r = last_price("^IRX") / 100.0
    px = {s: last_price(s) for s in ("NQ=F", "ES=F", "QQQ", "SPY", "^GSPC")}
    out = dict(time=now, r=r, prices=px, symbols={})
    for sym in SYMBOLS:
        t0 = time.time()
        spot, chain = fetch_chain(sym, now)
        fut, fsym, base = FUT[sym]
        ratio = px[fsym] / px[base]                     # NQ=F/QQQ, ES=F/SPY, ES=F/^GSPC
        views = {v: levels(view_chain(chain, v), spot, r) for v in VIEWS}
        cov = float((chain.oi > 0).mean()) if len(chain) else 0.0
        out["symbols"][sym] = dict(spot=spot, chain=chain, views=views, fut=fut, ratio=ratio,
                                   surface=surface(chain, spot), oi_coverage=cov)
        log(f"  {sym}: spot {spot:,.2f}, {chain.expiry.nunique()} expiries, {len(chain):,} contracts "
            f"({time.time() - t0:.0f}s)")
    return out


def levels_table(snap: dict, view: str = "all") -> pd.DataFrame:
    rows = []
    for sym, s in snap["symbols"].items():
        lv = s["views"].get(view)
        conv = lambda x: None if x is None else round(x * s["ratio"], 2)
        base = dict(symbol=sym, view=VIEWS[view], **{"OI coverage %": round(100 * s.get("oi_coverage", 1.0), 1)})
        if s.get("oi_coverage", 1.0) < MIN_OI_COVERAGE:
            rows.append(dict(base, spot=round(s["spot"], 2), **{"total GEX $bn per 1%": None},
                             regime=f"not computed: Yahoo gives open interest for only {100 * s['oi_coverage']:.1f}% of contracts"))
            continue
        if lv is None:
            rows.append(dict(base, **{"total GEX $bn per 1%": None, "regime": "no contracts"}))
            continue
        rows.append(dict(base, spot=round(lv["spot"], 2), **{"total GEX $bn per 1%": round(lv["total"] / 1e9, 3)},
                         regime=lv["regime"], **{"call wall": lv["call_wall"], "put wall": lv["put_wall"],
                         "gamma flip": None if lv["flip"] is None else round(lv["flip"], 2),
                         "futures": s["fut"], "ratio": round(s["ratio"], 4),
                         "spot (fut)": conv(lv["spot"]), "call wall (fut)": conv(lv["call_wall"]),
                         "put wall (fut)": conv(lv["put_wall"]), "gamma flip (fut)": conv(lv["flip"])},
                         expiries=len(lv["expiries"]), contracts=lv["contracts"]))
    return pd.DataFrame(rows)


def save_daily(snap: dict, log=print) -> Path | None:
    p = GEX_DIR / f"gex_{snap['time']:%Y-%m-%d}.pkl.zst"
    if p.exists():
        log(f"  today's snapshot already exists ({p.name}); not saved again")
        return None
    thin = [s for s in ("SPY", "QQQ") if snap["symbols"][s].get("oi_coverage", 0) < MIN_OI_COVERAGE]
    if thin:   # Yahoo blanks open interest and quotes while it resets chains around midnight ET
        log(f"  not saved: open interest missing for most {', '.join(thin)} contracts (Yahoo resetting?); "
            f"a later fetch today will save")
        return None
    chains = pd.concat([s["chain"].assign(symbol=sym)[["symbol", "expiry", "type", "strike", "oi", "iv", "bid", "dte"]]
                        for sym, s in snap["symbols"].items()], ignore_index=True)
    chains["oi"] = chains.oi.astype("int32"); chains["iv"] = chains.iv.astype("float32")
    chains["bid"] = chains.bid.astype("float32"); chains["dte"] = chains.dte.astype("int16")
    lv = pd.concat([levels_table(snap, v) for v in VIEWS], ignore_index=True)
    blob = dict(time=snap["time"], r=snap["r"], prices=snap["prices"], levels=lv, chains=chains,
                spots={sym: s["spot"] for sym, s in snap["symbols"].items()})
    GEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(zstd.compress(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)))
    tmp.replace(p)
    log(f"  saved {p.name} ({p.stat().st_size / 1e3:.0f} KB)")
    return p


def load_daily(p: Path) -> dict:
    """A saved snapshot. Files saved before bid / dte / spots were added get dte rebuilt from the
    snapshot date (same rule as fetch_chain), bid as NaN and spots as {} (surface() then has no
    contracts to use, since it needs bid > 0 and the spot of the same pull)."""
    blob = pickle.loads(zstd.decompress(Path(p).read_bytes()))
    c = blob["chains"]
    if "dte" not in c:
        c["dte"] = (pd.to_datetime(c.expiry) - pd.Timestamp(blob["time"].date())).dt.days
    if "bid" not in c:
        c["bid"] = np.nan
    blob.setdefault("spots", {})
    return blob


def saved_surface(blob: dict, sym: str, kind: str = "otm") -> dict | None:
    """surface() rebuilt from a saved snapshot; None when the file has no spot for sym (older files)."""
    if sym not in blob["spots"]:
        return None
    c = blob["chains"]
    return surface(c[c.symbol == sym], blob["spots"][sym], kind)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    snap = snapshot()
    print(f"\nsnapshot {snap['time']:%Y-%m-%d %H:%M} ET, risk-free {100 * snap['r']:.2f}% (^IRX)")
    for v in VIEWS:
        print(f"\n{VIEWS[v]}")
        print(levels_table(snap, v).to_string(index=False))
    print("\n" + DEALER_NOTE + "\n" + DELAY_NOTE)
    save_daily(snap)


if __name__ == "__main__":
    main()
