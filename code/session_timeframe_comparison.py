"""Matched MNQ session/timeframe comparison for selected November/Oscar/Papa tests."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import nq_backtest as bt  # noqa: E402
from strategy_research_suite import (  # noqa: E402
    MNQ_PATH, RR_GRID, NY_TZ, resample_ohlcv,
)

OUT = ROOT / "results"
TIMEFRAMES = (5, 10, 15)
SESSIONS = ("new_york", "asia")
CONTRACT_NAME = "MNQ micro (1 contract)"
SLIPPAGE_TICKS = 1.0
COMMISSION_SIDE = 2.50
DD_LIMIT = 1000.0
PROFIT_TARGET = 1250.0


def localize_session_config(session: str) -> bt.Config:
    return bt.Config(
        point_value=2.0, contracts=1, risk_fraction=0.0, rr=1.0,
        slippage_ticks=SLIPPAGE_TICKS, commission_per_side=COMMISSION_SIDE,
        drawdown_dollars=DD_LIMIT, target_dollars=PROFIT_TARGET,
        session=session, session_tz=NY_TZ,
        ny_start="09:30", ny_end="16:00",
        asia_start="19:00", asia_end="03:00",
    )


def session_id(ts: pd.Timestamp, session: str) -> object:
    t = ts.tz_convert(NY_TZ)
    minute = t.hour * 60 + t.minute
    if session == "new_york":
        return t.date() if 570 <= minute < 960 else None
    if minute >= 1140:
        return t.date()
    if minute < 180:
        return t.date() - timedelta(days=1)
    return None


def session_keys(index: pd.DatetimeIndex, session: str) -> pd.Series:
    return pd.Series([session_id(ts, session) for ts in index], index=index, dtype=object)


def session_clock(session: str, key: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = "09:30" if session == "new_york" else "19:00"
    end = "16:00" if session == "new_york" else "03:00"
    start_dt = pd.Timestamp(f"{key} {start}", tz=NY_TZ)
    end_day = key if session == "new_york" else key + timedelta(days=1)
    end_dt = pd.Timestamp(f"{end_day} {end}", tz=NY_TZ)
    return start_dt, end_dt


def session_vwap_bands(df: pd.DataFrame, session: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    keys = session_keys(df.index, session)
    active = keys.notna()
    volume = pd.to_numeric(df.get("volume", pd.Series(1.0, index=df.index)), errors="coerce").fillna(0).clip(lower=0)
    volume = volume.where(active, 0.0)
    typical = (df.high + df.low + df.close) / 3.0
    key_group = keys.fillna("outside")
    cum_vol = volume.groupby(key_group).cumsum().replace(0, np.nan)
    vwap = (typical * volume).groupby(key_group).cumsum() / cum_vol
    second = (typical.pow(2) * volume).groupby(key_group).cumsum() / cum_vol
    sd = (second - vwap.pow(2)).clip(lower=0).pow(0.5)
    return vwap.where(active), sd.where(active), active


def avwap_signal(df: pd.DataFrame, session: str, sigma: float = 2.0) -> pd.Series:
    vwap, sd, active = session_vwap_bands(df, session)
    upper, lower = vwap + sigma * sd, vwap - sigma * sd
    long = (df.close.shift(1) < lower.shift(1)) & (df.close >= lower) & (df.close < vwap)
    short = (df.close.shift(1) > upper.shift(1)) & (df.close <= upper) & (df.close > vwap)
    keys = session_keys(df.index, session)
    allowed = pd.Series(False, index=df.index)
    for key in keys.dropna().unique():
        _, end_dt = session_clock(session, key)
        local = df.index.tz_convert(NY_TZ)
        # Bar-start labels: require next-open fill to occur before the session ends.
        allowed.loc[keys.eq(key)] = local[keys.eq(key).to_numpy()] + pd.Timedelta(minutes=int(df.attrs.get("bar_minutes", 5))) < end_dt
    return pd.Series(np.select([long & active & allowed, short & active & allowed], [1, -1], 0), index=df.index, dtype="int8")


def orb_signal(source5: pd.DataFrame, execution: pd.DataFrame, session: str,
               bar_minutes: int, range_minutes: int, variant: str) -> tuple[pd.Series, pd.Series]:
    signal = pd.Series(0, index=execution.index, dtype="int8")
    setup = pd.Series("", index=execution.index, dtype=object)
    src_keys = session_keys(source5.index, session)
    exec_keys = session_keys(execution.index, session)
    local_src = source5.index.tz_convert(NY_TZ)
    local_exec = execution.index.tz_convert(NY_TZ)
    range_len = range_minutes

    for key in exec_keys.dropna().unique():
        start_dt, end_dt = session_clock(session, key)
        range_end = start_dt + pd.Timedelta(minutes=range_len)
        src_mask = src_keys.eq(key) & (source5.index < range_end.tz_convert(source5.index.tz))
        if not src_mask.any():
            continue
        orb_high = float(source5.loc[src_mask, "high"].max())
        orb_low = float(source5.loc[src_mask, "low"].min())
        if not np.isfinite(orb_high + orb_low) or orb_high <= orb_low:
            continue
        row_mask = exec_keys.eq(key)
        day_rows = execution.index[row_mask.to_numpy()]
        candidate = [ts for ts in day_rows
                     if ts.tz_convert(NY_TZ) >= range_end
                     and ts.tz_convert(NY_TZ) + pd.Timedelta(minutes=bar_minutes) < end_dt]
        direction = pending_dir = consecutive = 0
        pending_bar = -1
        for j, ts in enumerate(candidate):
            close = float(execution.at[ts, "close"])
            high = float(execution.at[ts, "high"])
            low = float(execution.at[ts, "low"])
            outside = 1 if close > orb_high else -1 if close < orb_low else 0
            if variant == "retest":
                if pending_dir == 0 and outside:
                    pending_dir, pending_bar = outside, j
                elif pending_dir and j > pending_bar:
                    touched = (low <= orb_high and close > orb_high) if pending_dir > 0 else (high >= orb_low and close < orb_low)
                    if touched:
                        direction = pending_dir
                    elif outside and outside != pending_dir:
                        pending_dir = 0
                    elif j - pending_bar > max(1, 30 // bar_minutes):
                        pending_dir = 0
            elif variant == "two_close_fakeout_filter":
                if outside:
                    if outside == pending_dir:
                        consecutive += 1
                    else:
                        pending_dir, consecutive = outside, 1
                    if consecutive >= 2:
                        direction = outside
                else:
                    pending_dir, consecutive = 0, 0
            elif outside:
                direction = outside
            if direction:
                signal.at[ts] = direction
                setup.at[ts] = f"ORB15_{variant}"
                break
    return signal, setup


def papa_signal(df: pd.DataFrame, session: str, bar_minutes: int,
                target_points: float = 30.0) -> tuple[pd.Series, pd.Series]:
    signal = pd.Series(0, index=df.index, dtype="int8")
    stop_risk = pd.Series(np.nan, index=df.index, dtype=float)
    keys = session_keys(df.index, session)
    local = df.index.tz_convert(NY_TZ)
    for key in keys.dropna().unique():
        start_dt, end_dt = session_clock(session, key)
        if session == "new_york":
            anchor_dt = pd.Timestamp(f"{key} 15:00", tz=NY_TZ)
        else:
            anchor_dt = end_dt - pd.Timedelta(hours=1)
        anchor_loc = anchor_dt.tz_convert(df.index.tz)
        anchor_rows = df.index[(keys.eq(key).to_numpy()) & (df.index == anchor_loc)]
        if not len(anchor_rows):
            continue
        anchor = float(df.at[anchor_rows[0], "open"])
        window_end = anchor_dt + pd.Timedelta(minutes=45)
        latest_bar_start = window_end - pd.Timedelta(minutes=bar_minutes)
        rows = df.index[(keys.eq(key).to_numpy()) & (local > anchor_dt) & (local <= latest_bar_start)]
        for ts in rows:
            close = float(df.at[ts, "close"])
            if close <= anchor - target_points:
                signal.at[ts] = 1
            elif close >= anchor + target_points:
                signal.at[ts] = -1
            if signal.at[ts]:
                stop_risk.at[ts] = target_points  # Prior best configuration was 1R.
                break
    return signal, stop_risk


def papa_backtest(execution: pd.DataFrame, signal: pd.Series, risk_points: pd.Series,
                  raw5: pd.DataFrame, session: str, cfg: bt.Config) -> pd.DataFrame:
    """Fill on the requested interval; resolve stops/targets on 5m bars to honor :45 cutoff."""
    rows = []
    realized = peak = 0.0
    slip = cfg.slippage_ticks * cfg.tick_size
    local5 = raw5.index.tz_convert(NY_TZ)
    i = 0
    while i < len(execution) - 1:
        dd = peak - realized if cfg.drawdown_mode == "trailing" else -realized
        if dd >= cfg.drawdown_dollars:
            break
        direction = int(signal.iloc[i])
        risk = risk_points.iloc[i]
        if direction == 0 or not np.isfinite(risk) or risk <= 0:
            i += 1
            continue
        entry_i = i + 1
        entry_ts = execution.index[entry_i]
        entry_local = entry_ts.tz_convert(NY_TZ)
        cutoff_local = (pd.Timestamp(f"{entry_local.date()} 15:45", tz=NY_TZ)
                        if session == "new_york"
                        else pd.Timestamp(f"{entry_local.date()} 02:45", tz=NY_TZ))
        # For Asia, early-morning entries belong to the evening session already in progress.
        if session == "asia" and entry_local.hour >= 19:
            cutoff_local += pd.Timedelta(days=1)
        if entry_local >= cutoff_local:
            i += 1
            continue
        entry = float(execution.open.iloc[entry_i]) + direction * slip
        planned = (risk + 2 * slip) * cfg.point_value + 2 * cfg.commission_per_side
        if planned > cfg.drawdown_dollars - dd:
            i += 1
            continue
        stop = entry - direction * risk
        target = entry + direction * 30.0
        raw_start = raw5.index.searchsorted(entry_ts, side="left")
        raw_end = raw5.index.searchsorted(cutoff_local.tz_convert(raw5.index.tz), side="left")
        exit_ts = cutoff_local.tz_convert(raw5.index.tz)
        cutoff_pos = raw5.index.get_indexer([exit_ts])[0]
        exit_price = float(raw5.open.iloc[cutoff_pos]) if cutoff_pos >= 0 else float(raw5.close.iloc[max(raw_end - 1, raw_start)])
        reason = "time"
        used_raw_pos = max(raw_end - 1, raw_start)
        for j in range(raw_start, raw_end):
            hi, lo = float(raw5.high.iloc[j]), float(raw5.low.iloc[j])
            stop_hit = lo <= stop if direction > 0 else hi >= stop
            target_hit = hi >= target if direction > 0 else lo <= target
            if stop_hit:
                op = float(raw5.open.iloc[j])
                exit_price = min(stop, op) if direction > 0 else max(stop, op)
                reason, used_raw_pos = "stop", j
                break
            if target_hit:
                exit_price, reason, used_raw_pos = target, "target", j
                break
        exit_ts = raw5.index[used_raw_pos] if reason != "time" else exit_ts
        exit_price -= direction * slip
        gross = (exit_price - entry) * direction * cfg.point_value
        fees = 2 * cfg.commission_per_side
        pnl = gross - fees
        rows.append({"entry_time": entry_ts, "exit_time": exit_ts, "direction": direction,
                     "entry": entry, "exit": exit_price, "reason": reason,
                     "contracts": 1, "gross_pnl": gross, "fees": fees, "pnl": pnl,
                     "risk_dollars": risk * cfg.point_value, "rr_target": cfg.rr,
                     "bars_held": max(0, used_raw_pos - raw_start), "setup_type": "Papa 30pt reversion"})
        realized += pnl
        peak = max(peak, realized)
        i = max(entry_i, execution.index.searchsorted(exit_ts, side="right"))
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "direction", "entry", "exit", "reason",
                                      "contracts", "gross_pnl", "fees", "pnl", "risk_dollars",
                                      "rr_target", "bars_held", "setup_type"])


def session_trade_key(ts: pd.Timestamp, session: str) -> object:
    key = session_id(ts, session)
    if key is not None:
        return key
    local = ts.tz_convert(NY_TZ)
    minute = local.hour * 60 + local.minute
    if session == "new_york" and minute == 960:
        return local.date()
    if session == "asia" and minute == 180:
        return local.date() - timedelta(days=1)
    return local.date()


def summarize(trades: pd.DataFrame, df: pd.DataFrame, session: str) -> dict:
    pnl = trades.pnl.astype(float) if len(trades) else pd.Series(dtype=float)
    equity = pnl.cumsum()
    dd = float((equity.cummax().clip(lower=0) - equity).max()) if len(pnl) else 0.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    keys = session_keys(df.index, session).dropna().unique()
    exit_days = [session_trade_key(pd.Timestamp(t), session) for t in trades.exit_time] if len(trades) else []
    daily_pnl = pd.Series(pnl.to_numpy(), index=pd.Index(exit_days)).groupby(level=0).sum() if len(pnl) else pd.Series(dtype=float)
    daily_pnl = daily_pnl.reindex(keys, fill_value=0.0)
    daily_std = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0
    sharpe = float(daily_pnl.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan
    active = len(set(session_trade_key(pd.Timestamp(t), session) for t in trades.entry_time)) if len(trades) else 0
    return {
        "trades": int(len(trades)), "win_rate": float(pnl.gt(0).mean()) if len(pnl) else 0.0,
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else np.nan,
        "max_closed_drawdown": dd, "daily_dollar_sharpe": sharpe,
        "trades_per_session_day": len(trades) / max(len(keys), 1),
        "active_session_days": active,
        "trades_per_active_day": len(trades) / active if active else 0.0,
        "contract_max": int(trades.contracts.max()) if len(trades) else 0,
    }


def main() -> None:
    if not MNQ_PATH.exists():
        raise FileNotFoundError(MNQ_PATH)
    raw = bt.load_bars(MNQ_PATH, localize_session_config("new_york"))
    rows, all_trades = [], []
    start, end = raw.index.min(), raw.index.max()
    session = "new_york"
    cfg = localize_session_config(session)
    # Oscar comparisons isolate OR width and fakeout confirmation. Use 2R on all four.
    oscar_tests = ((5, 5, "breakout", 2.0), (5, 5, "two_close_fakeout_filter", 2.0),
                   (15, 15, "breakout", 2.0), (15, 15, "two_close_fakeout_filter", 2.0),
                   # Preserve the previously strongest specification as a direct audit row.
                   (5, 15, "retest", 3.0))
    for tf, range_minutes, variant, rr in oscar_tests:
        x = bt.indicators(resample_ohlcv(raw, tf))
        signal, setup = orb_signal(raw, x, session, tf, range_minutes, variant)
        use_cfg = bt.Config(**{**cfg.__dict__, "rr": rr})
        trades = bt.backtest(x, signal, use_cfg, None, setup,
                             flat_at_session_end=True, session_exit_tz=NY_TZ,
                             session_exit_time="16:00")
        params = {"strategy": f"Oscar {range_minutes}m ORB {variant}", "session": session,
                  "session_timezone": NY_TZ, "bar_minutes": tf,
                  "opening_range_minutes": range_minutes, "rr": rr}
        rows.append({**params, **summarize(trades, x, session),
                     "instrument": CONTRACT_NAME, "slippage_ticks_per_side": SLIPPAGE_TICKS,
                     "commission_per_side": COMMISSION_SIDE, "drawdown_limit": DD_LIMIT,
                     "profit_target": PROFIT_TARGET, "session_hours": "09:30-16:00 ET"})
        if len(trades):
            tt = trades.copy()
            for k, v in params.items(): tt[k] = v
            all_trades.append(tt)

    # November: NY-open anchored VWAP, fixed 2 standard deviations and 2R.
    for tf in (5, 10, 15):
        x = bt.indicators(resample_ohlcv(raw, tf))
        x.attrs["bar_minutes"] = tf
        signal = avwap_signal(x, session, 2.0)
        use_cfg = bt.Config(**{**cfg.__dict__, "rr": 2.0})
        trades = bt.backtest(x, signal, use_cfg, None, None,
                             flat_at_session_end=True, session_exit_tz=NY_TZ,
                             session_exit_time="16:00")
        params = {"strategy": "November anchored VWAP 2σ", "session": session,
                  "session_timezone": NY_TZ, "bar_minutes": tf, "rr": 2.0,
                  "band_sigma": 2.0}
        rows.append({**params, **summarize(trades, x, session),
                     "instrument": CONTRACT_NAME, "slippage_ticks_per_side": SLIPPAGE_TICKS,
                     "commission_per_side": COMMISSION_SIDE, "drawdown_limit": DD_LIMIT,
                     "profit_target": PROFIT_TARGET, "session_hours": "09:30-16:00 ET"})
        if len(trades):
            tt = trades.copy()
            for k, v in params.items(): tt[k] = v
            all_trades.append(tt)

    # Papa: same 30-point target, 1R and 15:00-15:45 ET window.
    for tf in (5, 10):
        x = bt.indicators(resample_ohlcv(raw, tf))
        signal, stop_risk = papa_signal(x, session, tf, 30.0)
        use_cfg = bt.Config(**{**cfg.__dict__, "rr": 1.0})
        trades = papa_backtest(x, signal, stop_risk, raw, session, use_cfg)
        params = {"strategy": "Papa 30pt reversion", "session": session,
                  "session_timezone": NY_TZ, "bar_minutes": tf, "rr": 1.0,
                  "target_points": 30.0, "window": "15:00-15:45 ET"}
        rows.append({**params, **summarize(trades, x, session),
                     "instrument": CONTRACT_NAME, "slippage_ticks_per_side": SLIPPAGE_TICKS,
                     "commission_per_side": COMMISSION_SIDE, "drawdown_limit": DD_LIMIT,
                     "profit_target": PROFIT_TARGET, "session_hours": "09:30-16:00 ET"})
        if len(trades):
            tt = trades.copy()
            for k, v in params.items(): tt[k] = v
            all_trades.append(tt)

    summary = pd.DataFrame(rows)
    summary["data_start"] = str(start)
    summary["data_end"] = str(end)
    summary_path = OUT / "selected_strategies_new_york_summary.csv"
    trades_path = OUT / "selected_strategies_new_york_trades.csv"
    report_path = OUT / "selected_strategies_new_york_report.png"
    summary.to_csv(summary_path, index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False) if all_trades else pd.DataFrame().to_csv(trades_path, index=False)
    make_report(summary, report_path, start, end)
    print(f"Local MNQ Yahoo data: {start} through {end} ({len(raw):,} original 5m bars)")
    print("Compared 10 New York configurations: Oscar 5m and 15m ORB raw/fakeout, plus prior-best 15m retest; November 5/10/15m; Papa 5/10m.")
    print("Fixed parameters: Oscar raw/fakeout 2R; prior-best Oscar retest 3R; November AVWAP 2σ/2R; Papa 30 points/1R.")
    print("Costs: 1 tick slippage/side, $2.50 commission/side; one MNQ Micro; $1,000 closed-trade DD stop.")
    print("All tests use New York session 09:30-16:00 ET; Papa entries are limited to 15:00-15:45 and flatten at 15:45 ET.")
    print("Results by strategy/timeframe:")
    for strategy, sub in summary.groupby("strategy", sort=False):
        best = sub.loc[sub.net_pnl.idxmax()]
        print(f"{strategy} | best {int(best.bar_minutes)}m | RR={best.rr:g} | trades={int(best.trades)} | "
              f"net=${best.net_pnl:.2f} | win={best.win_rate:.1%} | closed DD=${best.max_closed_drawdown:.2f} | "
              f"Sharpe={best.daily_dollar_sharpe:.2f}")
    print(f"Saved {summary_path}")
    print(f"Saved {trades_path}")
    print(f"Saved {report_path}")


def make_report(summary: pd.DataFrame, path: Path,
                start: pd.Timestamp, end: pd.Timestamp) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Selected MNQ Strategies | New York Session", fontsize=19, fontweight="bold")
    fig.text(.5, .94, f"Local Yahoo 5m data {start.date()}–{end.date()} | 1 MNQ Micro | 1 tick/side + $2.50/side | prior RR held fixed",
             ha="center", fontsize=10, color="#59636e")
    label_map = {
        "Oscar 5m ORB breakout": "Oscar 5m break",
        "Oscar 5m ORB two_close_fakeout_filter": "Oscar 5m fakeout",
        "Oscar 15m ORB breakout": "Oscar 15m break",
        "Oscar 15m ORB two_close_fakeout_filter": "Oscar 15m fakeout",
        "Oscar 15m ORB retest": "Oscar 15m retest",
        "November anchored VWAP 2σ": "November AVWAP 2σ",
        "Papa 30pt reversion": "Papa 30pt",
    }
    labels = []
    for row in summary.itertuples():
        label = label_map.get(row.strategy, row.strategy)
        labels.append(f"{label} {int(row.bar_minutes)}m" if not label.endswith(f"{int(row.bar_minutes)}m") else label)
    colors = ["#3b6f89" if str(s).startswith("Oscar") else "#b07d3c" if str(s).startswith("November") else "#8368a8" for s in summary.strategy]
    x = np.arange(len(summary))
    for ax, metric, title, ylabel in [
        (axes[0, 0], "net_pnl", "Net P&L", "Net P&L ($)"),
        (axes[0, 1], "max_closed_drawdown", "Closed-trade drawdown", "Drawdown ($)"),
        (axes[1, 0], "daily_dollar_sharpe", "Daily dollar Sharpe (rough)", "Annualized Sharpe"),
        (axes[1, 1], "win_rate", "Win rate", "Win rate (%)"),
    ]:
        ax.bar(x, summary[metric].astype(float), color=colors, edgecolor="white", linewidth=.6)
        ax.axhline(0, color="#555", lw=.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
        if metric == "net_pnl" or metric == "max_closed_drawdown":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        if metric == "win_rate":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.text(.5, .015, "NY = 09:30–16:00 ET | Papa setup window = 15:00–15:45 ET | Oscar raw/fakeout use 2R | Closed-trade DD excludes intrabar swings; in-sample results.",
             ha="center", fontsize=9, color="#59636e")
    fig.tight_layout(rect=[0, .045, 1, .92])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
