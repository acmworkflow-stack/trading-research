"""Run the November/Oscar/Papa/Quebec/Romeo/Sulu research batch on local bars.

Signals use completed candles and are filled at the next bar open. Existing
backtest engine supplies one-contract MNQ fills, commissions, slippage and the
evaluation drawdown stop. True one-minute bars, order-book flow and options IV
are not present in the local Yahoo OHLCV files and are not fabricated here.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import nq_backtest as bt  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
MNQ_PATH = DATA_DIR / "mnq_yahoo_5m.csv"
MES_PATH = DATA_DIR / "mes_yahoo_5m.csv"
RR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
NY_TZ = "America/New_York"


def ny_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(NY_TZ)


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 5:
        return df.copy()
    rule = f"{minutes}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df:
        agg["volume"] = "sum"
    result = df.resample(rule, label="left", closed="left", origin="start_day").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return result


def daily_sharpe(trades: pd.DataFrame, full_index: pd.DatetimeIndex) -> float:
    if trades.empty:
        return np.nan
    local = pd.to_datetime(trades.exit_time).dt.tz_convert(NY_TZ)
    daily = trades.assign(day=local.dt.normalize()).groupby("day").pnl.sum()
    local_bars = ny_index(full_index)
    mins = local_bars.hour * 60 + local_bars.minute
    session_days = local_bars[(mins >= 570) & (mins < 960)].normalize().unique()
    days = pd.Index(session_days).sort_values()
    daily = daily.reindex(days, fill_value=0.0)
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    return float(daily.mean() / std * np.sqrt(252)) if std > 0 else np.nan


def metrics(trades: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    pnl = trades.pnl.astype(float) if len(trades) else pd.Series(dtype=float)
    equity = pnl.cumsum()
    dd = float((equity.cummax().clip(lower=0) - equity).max()) if len(pnl) else 0.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    local_bars = ny_index(index)
    mins = local_bars.hour * 60 + local_bars.minute
    total_days = max(len(pd.Index(local_bars[(mins >= 570) & (mins < 960)].normalize().unique())), 1)
    active_days = pd.to_datetime(trades.entry_time).dt.tz_convert(NY_TZ).dt.normalize().nunique() if len(trades) else 0
    return {
        "trades": len(trades),
        "win_rate": float(pnl.gt(0).mean()) if len(pnl) else 0.0,
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else np.nan,
        "max_closed_drawdown": dd,
        "daily_sharpe": daily_sharpe(trades, index),
        "trades_per_calendar_day": len(trades) / total_days,
        "trades_per_active_day": len(trades) / active_days if active_days else 0.0,
        "avg_bars_held": float(trades.bars_held.mean()) if len(trades) else 0.0,
        "contracts_max": int(trades.contracts.max()) if len(trades) else 0,
    }


def summarize_run(rows: list[dict], trade_frames: list[pd.DataFrame], *,
                  strategy: str, family: str, params: dict, df: pd.DataFrame,
                  signal: pd.Series, cfg: bt.Config,
                  atr_series: pd.Series | None = None,
                  setup: pd.Series | None = None,
                  flat_at_session_end: bool = True) -> None:
    trades = bt.backtest(df, signal, cfg, atr_series, setup,
                         flat_at_session_end=flat_at_session_end,
                         session_exit_tz=NY_TZ, session_exit_time="16:00")
    row = {
        "family": family, "strategy": strategy, **params,
        "rr": cfg.rr, "status": "tested", **metrics(trades, df.index),
        "instrument": "MNQ (1 Micro contract)",
        "commission_per_side": cfg.commission_per_side,
        "slippage_ticks_per_side": cfg.slippage_ticks,
        "drawdown_limit": cfg.drawdown_dollars,
        "profit_target": cfg.target_dollars,
    }
    rows.append(row)
    if len(trades):
        t = trades.copy()
        for key, val in {"family": family, "strategy": strategy, **params, "rr": cfg.rr}.items():
            t[key] = val
        trade_frames.append(t)


def anchored_vwap_bands(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """NY-open anchored VWAP and volume-weighted standard deviation, known at bar close."""
    local = ny_index(df.index)
    minute = local.hour * 60 + local.minute
    active = (minute >= 9 * 60 + 30) & (minute < 16 * 60)
    day = pd.Series(local.date, index=df.index)
    volume = pd.to_numeric(df.get("volume", pd.Series(1.0, index=df.index)), errors="coerce").fillna(0.0)
    volume = volume.where(active, 0.0).clip(lower=0.0)
    typical = (df.high + df.low + df.close) / 3.0
    weighted = typical * volume
    weighted_sq = typical.pow(2) * volume
    cum_vol = volume.groupby(day).cumsum().replace(0.0, np.nan)
    vwap = weighted.groupby(day).cumsum() / cum_vol
    second = weighted_sq.groupby(day).cumsum() / cum_vol
    sd = np.sqrt((second - vwap.pow(2)).clip(lower=0.0))
    vwap = vwap.where(active)
    sd = sd.where(active)
    return vwap, sd, pd.Series(active, index=df.index)


def make_avwap_signal(df: pd.DataFrame, sigma: float) -> pd.Series:
    vwap, sd, active = anchored_vwap_bands(df)
    local = ny_index(df.index)
    minute = local.hour * 60 + local.minute
    upper, lower = vwap + sigma * sd, vwap - sigma * sd
    # Fade only after a completed close returns inside an outer band.
    long = (df.close.shift(1) < lower.shift(1)) & (df.close >= lower) & (df.close < vwap)
    short = (df.close.shift(1) > upper.shift(1)) & (df.close <= upper) & (df.close > vwap)
    entry_allowed = active & (minute < 15 * 60 + 55)
    return pd.Series(np.select([long & entry_allowed, short & entry_allowed], [1, -1], default=0), index=df.index, dtype="int8")


def make_orb_signal(df: pd.DataFrame, range_minutes: int, variant: str) -> tuple[pd.Series, pd.Series]:
    """NY opening-range breakout; one trade signal per day, generated on closed bars."""
    signal = pd.Series(0, index=df.index, dtype="int8")
    setup = pd.Series("", index=df.index, dtype=object)
    local = ny_index(df.index)
    local_dates = pd.Series(local.date, index=df.index)
    times = local.hour * 60 + local.minute
    tod = pd.Series(times, index=df.index)
    for day in local_dates.unique():
        idx = df.index[local_dates.to_numpy() == day]
        dmin = tod.loc[idx]
        opening = (dmin >= 570) & (dmin < 570 + range_minutes)
        after = (dmin >= 570 + range_minutes) & (dmin < 955)
        if not opening.any() or not after.any():
            continue
        orb_high = float(df.loc[idx[opening.to_numpy()], "high"].max())
        orb_low = float(df.loc[idx[opening.to_numpy()], "low"].min())
        if not np.isfinite(orb_high + orb_low) or orb_high <= orb_low:
            continue
        candidate = idx[after.to_numpy()]
        direction = 0
        consecutive = 0
        pending_dir = 0
        pending_bar = -1
        for j, ts in enumerate(candidate):
            close, high, low = float(df.at[ts, "close"]), float(df.at[ts, "high"]), float(df.at[ts, "low"])
            outside = 1 if close > orb_high else -1 if close < orb_low else 0
            if variant == "breakout":
                if outside:
                    direction = outside
            elif variant == "two_close_fakeout_filter":
                if outside:
                    if pending_dir == outside:
                        consecutive += 1
                    else:
                        pending_dir, consecutive = outside, 1
                if consecutive >= 2:
                    direction = outside
                if not outside:
                    pending_dir, consecutive = 0, 0
            elif variant == "retest":
                if pending_dir == 0 and outside:
                    pending_dir, pending_bar = outside, j
                elif pending_dir and j > pending_bar:
                    touched = (low <= orb_high and close > orb_high) if pending_dir > 0 else (high >= orb_low and close < orb_low)
                    if touched:
                        direction = pending_dir
                    elif j - pending_bar > 6 or (outside and outside != pending_dir):
                        pending_dir = 0
            if direction:
                signal.at[ts] = direction
                setup.at[ts] = f"ORB{range_minutes}_{variant}"
                break
    return signal, setup


def make_quebec_signal(mnq: pd.DataFrame, mes: pd.DataFrame, threshold: float) -> pd.Series:
    """Long/short MNQ on relative standardized returns; MES only informs the signal."""
    common = mnq.index.intersection(mes.index)
    r_nq = np.log(mnq.close.loc[common]).diff()
    r_es = np.log(mes.close.loc[common]).diff()
    def zscore(s: pd.Series) -> pd.Series:
        mean, sd = s.rolling(48, min_periods=24).mean(), s.rolling(48, min_periods=24).std(ddof=0)
        return (s - mean) / sd.replace(0.0, np.nan)
    rel = zscore(r_nq) - zscore(r_es)
    sig = pd.Series(0, index=common, dtype="int8")
    sig.loc[(rel >= threshold) & (rel.shift(1) < threshold)] = 1
    sig.loc[(rel <= -threshold) & (rel.shift(1) > -threshold)] = -1
    local = ny_index(common)
    allowed = (local.hour * 60 + local.minute >= 570) & (local.hour * 60 + local.minute < 955)
    return sig.where(allowed, 0).reindex(mnq.index).fillna(0).astype("int8")


def make_romeo_proxy(df: pd.DataFrame) -> pd.Series:
    """OHLCV-only directional-volume proxy; this is not exchange bid/ask order flow."""
    vol = pd.to_numeric(df.get("volume", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    signed = vol * np.sign(df.close - df.open)
    pressure = signed.rolling(12, min_periods=8).sum() / vol.rolling(12, min_periods=8).sum().replace(0.0, np.nan)
    long = (pressure > 0.30) & (df.close > df.high.shift(1))
    short = (pressure < -0.30) & (df.close < df.low.shift(1))
    local = ny_index(df.index)
    ny = (local.hour * 60 + local.minute >= 570) & (local.hour * 60 + local.minute < 955)
    return pd.Series(np.select([long & ny, short & ny], [1, -1], default=0), index=df.index, dtype="int8")


def make_papa_signal(df: pd.DataFrame, target_points: float, rr: float) -> tuple[pd.Series, pd.Series]:
    """Fade 20/30/40-point extensions from 15:00 ET anchor; stop distance = target/RR."""
    signal = pd.Series(0, index=df.index, dtype="int8")
    risk_points = pd.Series(np.nan, index=df.index, dtype=float)
    local = ny_index(df.index)
    day = pd.Series(local.date, index=df.index)
    minute = local.hour * 60 + local.minute
    for d in day.unique():
        idx = df.index[day.to_numpy() == d]
        mins = pd.Series(minute, index=df.index).loc[idx]
        anchor_rows = idx[mins.to_numpy() == 900]
        if not len(anchor_rows):
            continue
        anchor = float(df.at[anchor_rows[0], "open"])
        window = idx[(mins.to_numpy() > 900) & (mins.to_numpy() < 940)]
        for ts in window:
            close = float(df.at[ts, "close"])
            if close <= anchor - target_points:
                signal.at[ts] = 1
            elif close >= anchor + target_points:
                signal.at[ts] = -1
            if signal.at[ts]:
                risk_points.at[ts] = target_points / rr
                break
    return signal, risk_points


def backtest_papa(df: pd.DataFrame, signal: pd.Series, risk_points: pd.Series,
                  target_points: float, cfg: bt.Config) -> pd.DataFrame:
    """Fixed-contract test with a hard 15:45 ET time exit and conservative OHLC fills."""
    rows = []
    pnl_total = peak = 0.0
    slip = cfg.slippage_ticks * cfg.tick_size
    local = ny_index(df.index)
    i = 0
    while i < len(df) - 1:
        dd = peak - pnl_total
        if dd >= cfg.drawdown_dollars:
            break
        direction = int(signal.iloc[i])
        r = risk_points.iloc[i]
        if direction == 0 or not np.isfinite(r) or r <= 0:
            i += 1
            continue
        entry_i = i + 1
        entry = float(df.open.iloc[entry_i]) + direction * slip
        planned = (r + 2 * slip) * cfg.point_value + 2 * cfg.commission_per_side
        if planned > cfg.drawdown_dollars - dd:
            i += 1
            continue
        stop = entry - direction * r
        target = entry + direction * target_points
        reason, exit_i, exit_price = "end_of_data", len(df) - 1, float(df.close.iloc[-1])
        for j in range(entry_i, len(df)):
            lt = local[j]
            if lt.date() != local[entry_i].date() or lt.hour * 60 + lt.minute >= 945:
                reason, exit_i, exit_price = "time", j, float(df.open.iloc[j])
                break
            hi, lo = float(df.high.iloc[j]), float(df.low.iloc[j])
            stop_hit = lo <= stop if direction > 0 else hi >= stop
            target_hit = hi >= target if direction > 0 else lo <= target
            if stop_hit:
                op = float(df.open.iloc[j])
                exit_price = min(stop, op) if direction > 0 else max(stop, op)
                reason, exit_i = "stop", j
                break
            if target_hit:
                reason, exit_i, exit_price = "target", j, target
                break
        exit_price -= direction * slip
        gross = (exit_price - entry) * direction * cfg.point_value
        fees = 2 * cfg.commission_per_side
        pnl = gross - fees
        rows.append({"entry_time": df.index[entry_i], "exit_time": df.index[exit_i],
                     "direction": direction, "entry": entry, "exit": exit_price,
                     "reason": reason, "contracts": 1, "gross_pnl": gross,
                     "fees": fees, "pnl": pnl, "risk_dollars": r * cfg.point_value,
                     "rr_target": cfg.rr, "bars_held": exit_i - entry_i,
                     "setup_type": f"Papa target {target_points:.0f}pt"})
        pnl_total += pnl
        peak = max(peak, pnl_total)
        i = max(exit_i, entry_i) + 1
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "direction", "entry", "exit", "reason",
                                      "contracts", "gross_pnl", "fees", "pnl", "risk_dollars",
                                      "rr_target", "bars_held", "setup_type"])


def sulu_historical_range(df: pd.DataFrame) -> pd.DataFrame:
    """Historical-volatility next-day range forecast; not implied volatility or IV walls."""
    local = ny_index(df.index)
    minute = local.hour * 60 + local.minute
    active = (minute >= 570) & (minute < 960)
    daily = df.loc[active].copy()
    daily["ny_day"] = local[active].date
    byday = daily.groupby("ny_day").agg(open=("open", "first"), high=("high", "max"),
                                          low=("low", "min"), close=("close", "last"))
    byday["ret"] = np.log(byday.close / byday.close.shift(1))
    byday["hv20"] = byday.ret.rolling(20, min_periods=10).std(ddof=1) * np.sqrt(252)
    byday["prior_close"] = byday.close.shift(1)
    # Use only information through the prior session to forecast the current day.
    byday["forecast_1sigma_pts"] = byday.prior_close * byday.hv20.shift(1) / np.sqrt(252)
    byday["forecast_2sigma_pts"] = 2 * byday.forecast_1sigma_pts
    for k in (1, 2):
        band = byday[f"forecast_{k}sigma_pts"]
        byday[f"upper_{k}sigma"] = byday.prior_close + band
        byday[f"lower_{k}sigma"] = byday.prior_close - band
        byday[f"full_range_covered_{k}sigma"] = (byday.high <= byday[f"upper_{k}sigma"]) & (byday.low >= byday[f"lower_{k}sigma"])
    byday["realized_up_pts"] = byday.high - byday.prior_close
    byday["realized_down_pts"] = byday.prior_close - byday.low
    byday["forecast_method"] = "20-session historical close-to-close volatility; no options data"
    return byday.reset_index()


def main() -> None:
    if not MNQ_PATH.exists() or not MES_PATH.exists():
        raise FileNotFoundError(f"Expected local data at {MNQ_PATH} and {MES_PATH}")
    cfg_base = bt.Config(point_value=2.0, contracts=1, risk_fraction=0.0,
                         slippage_ticks=1.0, commission_per_side=2.50,
                         drawdown_dollars=1000.0, target_dollars=1250.0,
                         session="new_york", ny_start="08:30", ny_end="15:00")
    raw_nq = bt.load_bars(MNQ_PATH, cfg_base)
    raw_es = bt.load_bars(MES_PATH, cfg_base)
    start, end = raw_nq.index.min(), raw_nq.index.max()
    rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    # November: NY-open anchored VWAP fade with weighted sigma bands.
    # Local source has 5m bars; 10m is resampled. 1m cannot be inferred.
    for tf in (5, 10):
        df = bt.indicators(resample_ohlcv(raw_nq, tf))
        for sigma in (1.0, 1.5, 2.0, 2.5, 3.0):
            sig = make_avwap_signal(df, sigma)
            for rr in RR_GRID:
                cfg = bt.Config(**{**cfg_base.__dict__, "rr": rr})
                summarize_run(rows, trade_frames, strategy=f"November AVWAP {sigma:g}σ",
                              family="November", params={"bar_minutes": tf, "band_sigma": sigma,
                              "entry_rule": "close re-enters band; NY-open anchor"},
                              df=df, signal=sig, cfg=cfg)

    # Oscar: 5m opening range and 15m opening range, executed on the 5m tape.
    df5 = bt.indicators(raw_nq)
    for orb in (5, 15):
        for variant in ("breakout", "two_close_fakeout_filter", "retest"):
            sig, setup = make_orb_signal(df5, orb, variant)
            for rr in RR_GRID:
                cfg = bt.Config(**{**cfg_base.__dict__, "rr": rr})
                summarize_run(rows, trade_frames, strategy=f"Oscar ORB {orb}m {variant}",
                              family="Oscar", params={"opening_range_minutes": orb,
                              "variant": variant, "execution_bar_minutes": 5},
                              df=df5, signal=sig, cfg=cfg, setup=setup)

    # Papa: fade 20/30/40 MNQ-point extensions from the 15:00 ET cash-close window anchor.
    for target_pts in (20.0, 30.0, 40.0):
        for rr in RR_GRID:
            sig, fixed_risk = make_papa_signal(df5, target_pts, rr)
            cfg = bt.Config(**{**cfg_base.__dict__, "rr": rr})
            trades = backtest_papa(df5, sig, fixed_risk, target_pts, cfg)
            rows.append({"family": "Papa", "strategy": f"Papa 3pm reversion {target_pts:.0f}pt",
                         "target_points": target_pts, "rr": rr, "status": "tested",
                         **metrics(trades, df5.index), "instrument": "MNQ (1 Micro contract)",
                         "commission_per_side": cfg.commission_per_side,
                         "slippage_ticks_per_side": cfg.slippage_ticks,
                         "drawdown_limit": cfg.drawdown_dollars, "profit_target": cfg.target_dollars,
                         "time_exit": "15:45 America/New_York"})
            if len(trades):
                t = trades.copy()
                t["family"], t["strategy"], t["target_points"], t["rr"] = "Papa", f"Papa 3pm reversion {target_pts:.0f}pt", target_pts, rr
                trade_frames.append(t)

    # Quebec: aligned MNQ/MES relative return z-scores; trades only MNQ, no hedge.
    df_es = bt.indicators(raw_es)
    for threshold in (1.0, 2.0):
        sig = make_quebec_signal(df5, df_es, threshold)
        for rr in RR_GRID:
            cfg = bt.Config(**{**cfg_base.__dict__, "rr": rr})
            summarize_run(rows, trade_frames, strategy=f"Quebec MNQ-MES relative z {threshold:+g}",
                          family="Quebec", params={"relative_z_threshold": threshold,
                          "signal_only_instrument": "MNQ", "MES_hedge": False,
                          "rolling_lookback_bars": 48, "session": "New York"},
                          df=df5, signal=sig, cfg=cfg)

    # Romeo: OHLCV directional-volume proxy only; true bid/ask delta is unavailable.
    romeo = make_romeo_proxy(df5)
    for rr in RR_GRID:
        cfg = bt.Config(**{**cfg_base.__dict__, "rr": rr})
        summarize_run(rows, trade_frames, strategy="Romeo directional-volume proxy",
                      family="Romeo", params={"flow_source": "sign(close-open)*volume",
                      "window_bars": 12, "threshold": 0.30, "not_true_orderflow": True},
                      df=df5, signal=romeo, cfg=cfg)

    # Sulu: range forecast diagnostic only. RR has no meaning without a directional entry.
    hv = sulu_historical_range(raw_nq)
    hv.to_csv(OUT_DIR / "sulu_historical_vol_range_forecast.csv", index=False)
    for sigma in (1, 2):
        col = f"full_range_covered_{sigma}sigma"
        rows.append({"family": "Sulu", "strategy": f"Historical volatility range {sigma}σ",
                     "status": "range forecast only; not a trade/RR backtest",
                     "rr": np.nan, "trades": 0, "win_rate": np.nan, "net_pnl": np.nan,
                     "profit_factor": np.nan, "max_closed_drawdown": np.nan,
                     "daily_sharpe": np.nan, "trades_per_calendar_day": np.nan,
                     "trades_per_active_day": np.nan, "avg_bars_held": np.nan,
                     "contracts_max": np.nan, "full_range_coverage": float(hv[col].mean()) if hv[col].notna().any() else np.nan,
                     "forecast_days": int(hv[col].notna().sum()),
                     "data_limitation": "No options implied volatility or open interest; historical-vol proxy only"})

    # Explicit not-run row for inaccessible 1-minute bars.
    rows.append({"family": "November", "strategy": "Anchored VWAP 1m",
                 "bar_minutes": 1, "status": "not run: 1m source bars unavailable",
                 "data_limitation": "Cannot reconstruct 1m OHLC from 5m data; no upsampling used"})

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "november_oscar_papa_quebec_romeo_sulu_summary.csv"
    trades_path = OUT_DIR / "november_oscar_papa_quebec_romeo_sulu_trades.csv"
    report_path = OUT_DIR / "november_oscar_papa_quebec_sulu_report.png"
    summary.to_csv(summary_path, index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)

    make_report(summary, hv, report_path, start, end)
    tested = summary[summary.status.eq("tested")]
    print(f"Data window: {ny_index(pd.DatetimeIndex([start]))[0]} through {ny_index(pd.DatetimeIndex([end]))[0]} (5m source)")
    print(f"Ran {len(tested):,} strategy/RR configurations; RR={RR_GRID}; MNQ one contract; 1 tick slippage and $2.50/side commission.")
    print("Top result per family by net P&L (in-sample, not a selection recommendation):")
    for family, sub in tested.groupby("family", sort=False):
        best = sub.loc[sub.net_pnl.astype(float).idxmax()]
        print(f"{family}: {best.strategy} RR={best.rr:g} trades={best.trades} net=${best.net_pnl:.2f} "
              f"win={best.win_rate:.1%} DD=${best.max_closed_drawdown:.2f} Sharpe={best.daily_sharpe:.2f}")
    print("Sulu historical-vol full-day 1σ/2σ coverage:",
          f"{hv.full_range_covered_1sigma.mean():.1%}/{hv.full_range_covered_2sigma.mean():.1%}")
    print("Not testable from these files: true 1m AVWAP, real bid/ask order flow, options IV/open-interest walls.")
    print(f"Saved: {summary_path}")
    print(f"Saved: {trades_path}")
    print(f"Saved: {report_path}")


def make_report(summary: pd.DataFrame, hv: pd.DataFrame, path: Path,
                start: pd.Timestamp, end: pd.Timestamp) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    tested = summary[summary.status.eq("tested")].copy()
    tested["rr"] = pd.to_numeric(tested.rr, errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MNQ Strategy Research | First-Pass Comparison", fontsize=19, fontweight="bold")
    fig.text(.5, .94, f"Local Yahoo 5m data | {ny_index(pd.DatetimeIndex([start]))[0].date()} to {ny_index(pd.DatetimeIndex([end]))[0].date()} | 1 MNQ Micro | $2.50/side + 1 tick/side | RR 1–3 | NY flat 4pm ET",
             ha="center", fontsize=10, color="#59636e")
    colors = {"November":"#3b6f89", "Oscar":"#b07d3c", "Papa":"#8368a8", "Quebec":"#4f8b79", "Romeo":"#ae6260"}

    ax = axes[0, 0]
    for fam, sub in tested.groupby("family", sort=False):
        g = sub.groupby("rr").net_pnl.mean()
        ax.plot(g.index, g.values, marker="o", label=fam, color=colors.get(fam, "#777"))
    ax.axhline(0, color="#555", lw=.8)
    ax.set(title="Average net P&L by RR", xlabel="Reward:risk", ylabel="Mean net P&L per configuration ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    fam_stats = tested.groupby("family").agg(net=("net_pnl", "mean"), sharpe=("daily_sharpe", "mean"), trades=("trades", "mean"))
    fam_stats = fam_stats.reindex([f for f in ["November", "Oscar", "Papa", "Quebec", "Romeo"] if f in fam_stats.index])
    bars = ax.bar(fam_stats.index, fam_stats.sharpe, color=[colors.get(f, "#777") for f in fam_stats.index])
    ax.axhline(0, color="#555", lw=.8)
    ax.set(title="Mean daily dollar Sharpe (rough; short sample)", ylabel="Annualized Sharpe")
    ax.tick_params(axis="x", rotation=20)
    for b, tr in zip(bars, fam_stats.trades):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{tr:.0f} trades/config", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    best = tested.sort_values("daily_sharpe", ascending=False).groupby(["family", "strategy", "bar_minutes", "variant", "band_sigma", "target_points", "relative_z_threshold"], dropna=False).head(1)
    best = best.sort_values("daily_sharpe", ascending=False).head(12)
    labels = [f"{r.family}: {r.strategy.replace(r.family+' ', '')} | {r.rr:g}R" for r in best.itertuples()]
    ax.barh(range(len(best)), best.daily_sharpe, color=[colors.get(f, "#777") for f in best.family])
    ax.set_yticks(range(len(best)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#555", lw=.8)
    ax.set(title="Highest-Sharpe configurations (exploratory)", xlabel="Daily dollar Sharpe")

    ax = axes[1, 1]
    valid = hv.dropna(subset=["forecast_1sigma_pts"])
    if len(valid):
        ax.plot(pd.to_datetime(valid.ny_day), valid.forecast_1sigma_pts, label="Forecast 1σ range (pts)", color="#3b6f89")
        actual = np.maximum(valid.realized_up_pts, valid.realized_down_pts)
        ax.plot(pd.to_datetime(valid.ny_day), actual, label="Actual max excursion (pts)", color="#c27c57", alpha=.75)
        ax.legend(frameon=False, fontsize=8)
    ax.set(title="Sulu proxy: prior-session historical-vol forecast", ylabel="MNQ points", xlabel="Session date")
    ax.tick_params(axis="x", rotation=25)

    fig.text(.5, .015,
             "Limits: 1m, true bid/ask order flow and options IV/open-interest unavailable. Sulu is a historical-vol range diagnostic, not IV walls. In-sample results are not proof of edge.",
             ha="center", fontsize=9, color="#59636e")
    fig.tight_layout(rect=[0, .045, 1, .92])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
