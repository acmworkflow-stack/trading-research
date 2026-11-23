"""NQ strategy research backtester. Signals are formed on closed bars and filled next bar."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Config:
    tick_size: float = 0.25
    point_value: float = 2.0        # MNQ Micro: $2 per point; NQ E-mini is $20 per point
    slippage_ticks: float = 1.0     # per side; raise for volatile/illiquid periods
    commission_per_side: float = 2.50  # editable all-in broker + exchange/clearing estimate
    commission_min_per_side: float = 0.0  # per-order minimum, used for share commissions
    target_dollars: float = 1250.0
    drawdown_dollars: float = 1000.0
    drawdown_mode: str = "trailing"  # peak-to-valley loss limit, measured on closed trades
    contracts: int = 1              # Fixed quantity unless risk_fraction enables dynamic sizing
    risk_fraction: float = 0.0      # Fraction of drawdown allowance risked per trade; 0 means fixed quantity
    stop_atr: float = 1.0
    rr: float = 2.0
    breakeven_r: float = 0.0        # move stop to entry once price reaches this many R; 0 = off
    session_tz: str = "America/Chicago"
    session: str = "all"            # all, asia, new_york
    # Asia 18:00-02:00 CT; New York 08:30-15:00 CT. Edit to match your definition.
    asia_start: str = "18:00"
    asia_end: str = "02:00"
    ny_start: str = "08:30"
    ny_end: str = "15:00"


def load_bars(path: str | Path, cfg: Config, start: str | None = None,
              end: str | None = None, data_timezone: str | None = None) -> pd.DataFrame:
    """CSV columns: timestamp, open, high, low, close; optional volume. Timestamps must be sortable."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime":"timestamp"})
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    # Naive timestamps are interpreted as Chicago time. Prefer timezone-aware timestamps in data exports.
    ts = pd.to_datetime(df.pop("timestamp"), errors="raise")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(data_timezone or cfg.session_tz, ambiguous="NaT", nonexistent="NaT")
        ts = ts.dt.tz_convert(cfg.session_tz)
    else:
        ts = ts.dt.tz_convert(cfg.session_tz)
    df.index = ts
    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.isna()]
    if start:
        start_ts = pd.Timestamp(start, tz=cfg.session_tz)
        df = df.loc[df.index >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz=cfg.session_tz)
        df = df.loc[df.index < end_ts]
    if df.empty:
        raise ValueError(f"No bars remain for requested date range start={start!r}, end={end!r}. "
                         "Supply a CSV containing that historical interval.")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found; clean the input before backtesting.")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    if (df.high < df.low).any() or (df.close <= 0).any():
        raise ValueError("Invalid OHLC values found.")
    return df


def download_yahoo(symbol: str = "MNQ=F", period: str = "60d", interval: str = "5m",
                   out: str | Path = "mnq_yahoo.csv", start: str | None = None,
                   end: str | None = None, prepost: bool = True) -> Path:
    """Download a recent Yahoo Finance continuous futures series as backtest-ready OHLCV CSV."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance in this Python environment: python -m pip install yfinance") from exc
    date_args = {"start":start, "end":end} if start or end else {"period":period}
    data = yf.download(symbol, interval=interval, prepost=prepost, auto_adjust=False,
                       progress=False, multi_level_index=False, **date_args)
    if data is None or data.empty:
        raise RuntimeError(f"Yahoo returned no bars for {symbol} in the requested window. Yahoo intraday history is limited to the last 60 days; use an older local/exported 5-minute CSV for earlier dates.")
    data = data.rename_axis("timestamp").reset_index()
    data.columns = [str(c).lower() for c in data.columns]
    data = data.rename(columns={"datetime":"timestamp"})
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in data.columns]
    path = Path(out)
    data[cols].to_csv(path, index=False)
    return path


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x.close.ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = x.close.ewm(span=50, adjust=False, min_periods=50).mean()
    delta = x.close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    prev = x.close.shift(1)
    tr = pd.concat([x.high - x.low, (x.high - prev).abs(), (x.low - prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    mid = x.close.rolling(20, min_periods=20).mean()
    sd = x.close.rolling(20, min_periods=20).std(ddof=0)
    x["bb_mid"], x["bb_upper"], x["bb_lower"] = mid, mid + 2*sd, mid - 2*sd
    # Session VWAP resets each Chicago calendar date. Uses bar typical price and volume.
    vol = x["volume"] if "volume" in x else pd.Series(1.0, index=x.index)
    day = pd.Series(x.index.date, index=x.index)
    tp = (x.high + x.low + x.close) / 3
    x["vwap"] = (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)
    # Closed higher-timeframe FVG state. Resample labels are at bar end; shift one full HTF bar
    # before forward fill so an unfinished/current higher-timeframe candle cannot leak in.
    for minutes in (5, 15, 30):
        rule = f"{minutes}min"
        h = x[["open", "high", "low", "close"]].resample(rule, label="right", closed="right").agg(
            {"open":"first", "high":"max", "low":"min", "close":"last"}).dropna()
        bull = h.low > h.high.shift(2)
        bear = h.high < h.low.shift(2)
        # Signed most recent gap state, available only after that HTF candle closes.
        state = pd.Series(np.where(bull, 1, np.where(bear, -1, np.nan)), index=h.index).ffill()
        state = state.shift(1)
        x[f"fvg{minutes}"] = state.reindex(x.index, method="ffill")
    return x


def in_session(index: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    mins = index.hour * 60 + index.minute
    def window(start: str, end: str) -> pd.Series:
        a = int(start[:2])*60 + int(start[3:]); b = int(end[:2])*60 + int(end[3:])
        return (mins >= a) | (mins < b) if a > b else (mins >= a) & (mins < b)
    if cfg.session == "asia": return window(cfg.asia_start, cfg.asia_end)
    if cfg.session == "new_york": return window(cfg.ny_start, cfg.ny_end)
    return pd.Series(True, index=index)


def session_vwap(x: pd.DataFrame, cfg: Config) -> pd.Series:
    """VWAP resets at the start of the selected analysis session, using only its bars."""
    active = in_session(x.index, cfg)
    if cfg.session == "asia":
        # Asia starts at 18:00 CT and continues through 02:00 CT next calendar day.
        days = pd.Series(x.index.date, index=x.index)
        before_end = (x.index.hour * 60 + x.index.minute) < 120
        days.loc[before_end] = days.loc[before_end].map(lambda d: d - pd.Timedelta(days=1))
    else:
        days = pd.Series(x.index.date, index=x.index)
    volume = x["volume"].astype(float) if "volume" in x else pd.Series(1.0, index=x.index)
    volume = volume.where(active, 0.0)
    typical = (x.high + x.low + x.close) / 3
    cumulative_volume = volume.groupby(days).cumsum().replace(0, np.nan)
    cumulative_value = (typical * volume).groupby(days).cumsum()
    return cumulative_value / cumulative_volume


def make_signals(x: pd.DataFrame, strategy: str, cfg: Config,
                 bb_mult: float = 2.0, rsi_level: float = 30.0) -> pd.Series:
    c, o = x.close, x.open
    if strategy == "bos_choch":
        structure_signal, _ = make_bos_choch_signals(x)
        return structure_signal.where(in_session(x.index, cfg), 0)
    if strategy == "mean_reversion_rsi":
        mean = c.rolling(20, min_periods=20).mean()
        long = (c < mean) & (x.rsi <= 30)
        short = (c > mean) & (x.rsi >= 70)
    elif strategy in ("mean_reversion_bb_rsi", "lima"):
        bb_mid = c.rolling(20, min_periods=20).mean()
        bb_sd = c.rolling(20, min_periods=20).std(ddof=0)
        bb_upper = bb_mid + bb_mult * bb_sd
        bb_lower = bb_mid - bb_mult * bb_sd
        long = (c <= bb_lower) & (x.rsi <= rsi_level)
        short = (c >= bb_upper) & (x.rsi >= 100.0 - rsi_level)
    elif strategy == "momentum":
        long, short = (c > c.shift(20)) & (c > x.ema20), (c < c.shift(20)) & (c < x.ema20)
    elif strategy == "ema_cloud":
        long, short = x.ema20 > x.ema50, x.ema20 < x.ema50
    elif strategy in ("fvg", "ifvg", "fvg_vwap", "ifvg_vwap"):
        cols = ["fvg5", "fvg15", "fvg30"]
        aligned_up = (x[cols] == 1).all(axis=1)
        aligned_down = (x[cols] == -1).all(axis=1)
        inverse = strategy.startswith("ifvg")
        long, short = ((aligned_down, aligned_up) if inverse else (aligned_up, aligned_down))
        if strategy.endswith("vwap"):
            # Select by price relative to session VWAP, with direction preserved.
            long, short = long & (c > x.vwap), short & (c < x.vwap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    sig = pd.Series(np.select([long, short], [1, -1], default=0), index=x.index, dtype="int8")
    # Strategies decide at close; entry is deferred by the engine to the following bar's open.
    return sig.where(in_session(x.index, cfg), 0)


def make_fvg_ifvg_signals(x: pd.DataFrame, minutes: int,
                          min_gap_points: float = 0.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build completed-bar FVG/IFVG retest signals and timeframe ATR from 5m OHLC."""
    rule = f"{minutes}min"
    # Yahoo intraday timestamps denote bar starts. Right labels represent completed bar ends.
    h = x[["open", "high", "low", "close"]].resample(rule, label="right", closed="left").agg(
        {"open":"first", "high":"max", "low":"min", "close":"last"}).dropna()
    prev_close = h.close.shift(1)
    tr = pd.concat([h.high-h.low, (h.high-prev_close).abs(), (h.low-prev_close).abs()], axis=1).max(axis=1)
    h_atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    h_sig = pd.Series(0, index=h.index, dtype="int8")
    h_setup = pd.Series("", index=h.index, dtype=object)
    zones: list[dict] = []

    for k in range(len(h)):
        hi, lo, close = float(h.high.iloc[k]), float(h.low.iloc[k]), float(h.close.iloc[k])
        # Update zones created on earlier completed bars. Most recent zone takes precedence.
        for zone in reversed(zones):
            if zone["used"]:
                continue
            overlap = hi >= zone["bottom"] and lo <= zone["top"]
            if not zone["inverted"]:
                broken = close < zone["bottom"] if zone["direction"] > 0 else close > zone["top"]
                if broken:
                    zone["inverted"] = True
                    zone["invert_bar"] = k
                    continue
                if overlap:
                    h_sig.iloc[k] = zone["direction"]
                    h_setup.iloc[k] = "FVG"
                    zone["used"] = True
                    break
            elif k > zone["invert_bar"] and overlap:
                h_sig.iloc[k] = -zone["direction"]
                h_setup.iloc[k] = "IFVG"
                zone["used"] = True
                break

        # Three-candle imbalance, created only when the third candle is complete.
        if k >= 2:
            old_high, old_low = float(h.high.iloc[k-2]), float(h.low.iloc[k-2])
            if lo - old_high >= min_gap_points:
                if lo > old_high:
                    zones.append({"bottom":old_high, "top":lo, "direction":1,
                                  "inverted":False, "invert_bar":-1, "used":False})
            if old_low - hi >= min_gap_points:
                if hi < old_low:
                    zones.append({"bottom":hi, "top":old_low, "direction":-1,
                                  "inverted":False, "invert_bar":-1, "used":False})

    # Map a completed HTF candle's signal/ATR onto its final constituent 5m bar.
    # The execution engine then enters at the next 5m bar's open, after that HTF close.
    base_sig = pd.Series(0, index=x.index, dtype="int8")
    base_setup = pd.Series("", index=x.index, dtype=object)
    base_atr = pd.Series(np.nan, index=x.index, dtype=float)
    for k, end_time in enumerate(h.index):
        pos = x.index.searchsorted(end_time, side="left") - 1
        if 0 <= pos < len(x):
            base_sig.iloc[pos] = h_sig.iloc[k]
            base_setup.iloc[pos] = h_setup.iloc[k]
            base_atr.iloc[pos] = h_atr.iloc[k]
    return base_sig, base_setup, base_atr


def make_bos_choch_signals(x: pd.DataFrame, left: int = 2, right: int = 2) -> tuple[pd.Series, pd.Series]:
    """Use two-bar fractal swings, available only after both right-side bars close."""
    signal = pd.Series(0, index=x.index, dtype="int8")
    setup = pd.Series("", index=x.index, dtype=object)
    last_high = last_low = None
    high_broken = low_broken = True
    bias = 0
    for k in range(left + right, len(x)):
        pivot = k - right
        start = pivot - left
        end = pivot + right + 1
        highs = x.high.iloc[start:end]
        lows = x.low.iloc[start:end]
        if x.high.iloc[pivot] > highs.iloc[:left].max() and x.high.iloc[pivot] >= highs.iloc[left+1:].max():
            last_high = float(x.high.iloc[pivot]); high_broken = False
        if x.low.iloc[pivot] < lows.iloc[:left].min() and x.low.iloc[pivot] <= lows.iloc[left+1:].min():
            last_low = float(x.low.iloc[pivot]); low_broken = False

        close = float(x.close.iloc[k])
        if last_high is not None and not high_broken and close > last_high:
            signal.iloc[k] = 1
            setup.iloc[k] = "CHOCH" if bias < 0 else "BOS"
            bias = 1; high_broken = True
        elif last_low is not None and not low_broken and close < last_low:
            signal.iloc[k] = -1
            setup.iloc[k] = "CHOCH" if bias > 0 else "BOS"
            bias = -1; low_broken = True
    return signal, setup


def make_fvg_structure_signals(x: pd.DataFrame, minutes: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """FVG/IFVG retests allowed only in the direction of confirmed BOS/CHOCH bias."""
    fvg_signal, fvg_setup, atr_series = make_fvg_ifvg_signals(x, minutes)
    h = x[["open", "high", "low", "close"]].resample(
        f"{minutes}min", label="right", closed="left").agg(
        {"open":"first", "high":"max", "low":"min", "close":"last"}).dropna()
    structural_events, structural_types = make_bos_choch_signals(h)
    bias_by_htf = pd.Series(0, index=h.index, dtype="int8")
    type_by_htf = pd.Series("", index=h.index, dtype=object)
    bias, latest_type = 0, ""
    for k in range(len(h)):
        if structural_events.iloc[k] != 0:
            bias = int(structural_events.iloc[k])
            latest_type = str(structural_types.iloc[k])
        bias_by_htf.iloc[k] = bias
        type_by_htf.iloc[k] = latest_type

    base_bias = pd.Series(0, index=x.index, dtype="int8")
    base_type = pd.Series("", index=x.index, dtype=object)
    for k, end_time in enumerate(h.index):
        pos = x.index.searchsorted(end_time, side="left") - 1
        if 0 <= pos < len(x):
            base_bias.iloc[pos] = bias_by_htf.iloc[k]
            base_type.iloc[pos] = type_by_htf.iloc[k]
    aligned = (fvg_signal != 0) & (fvg_signal == base_bias)
    filtered_signal = fvg_signal.where(aligned, 0)
    filtered_setup = pd.Series("", index=x.index, dtype=object)
    filtered_setup.loc[aligned] = fvg_setup.loc[aligned] + "+" + base_type.loc[aligned]
    return filtered_signal, filtered_setup, atr_series


def make_fvg_orderblock_signals(x: pd.DataFrame, minutes: int,
                                min_gap_points: float = 0.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Filter FVG/IFVG retests by a causally confirmed order block and its direction.

    An order block is the last opposite-colored completed HTF candle before a confirmed
    two-left/two-right swing break. The zone only becomes active on the break candle close.
    A signal must match its direction and the signal candle must overlap the zone.
    """
    fvg_signal, fvg_setup, atr_series = make_fvg_ifvg_signals(x, minutes, min_gap_points)
    rule = f"{minutes}min"
    h = x[["open", "high", "low", "close"]].resample(
        rule, label="right", closed="left").agg(
        {"open":"first", "high":"max", "low":"min", "close":"last"}).dropna()
    events, event_types = make_bos_choch_signals(h)
    zone_top = pd.Series(np.nan, index=h.index, dtype=float)
    zone_bottom = pd.Series(np.nan, index=h.index, dtype=float)
    zone_direction = pd.Series(0, index=h.index, dtype="int8")
    active_top = active_bottom = np.nan
    active_direction = 0
    for k in range(len(h)):
        direction = int(events.iloc[k])
        if direction:
            # Search backward for the latest completed candle against the break direction.
            for j in range(k - 1, -1, -1):
                bearish = h.close.iloc[j] < h.open.iloc[j]
                bullish = h.close.iloc[j] > h.open.iloc[j]
                if (direction > 0 and bearish) or (direction < 0 and bullish):
                    active_bottom = float(h.low.iloc[j])
                    active_top = float(h.high.iloc[j])
                    active_direction = direction
                    break
        zone_top.iloc[k] = active_top
        zone_bottom.iloc[k] = active_bottom
        zone_direction.iloc[k] = active_direction

    base_top = pd.Series(np.nan, index=x.index, dtype=float)
    base_bottom = pd.Series(np.nan, index=x.index, dtype=float)
    base_direction = pd.Series(0, index=x.index, dtype="int8")
    base_overlap = pd.Series(False, index=x.index, dtype=bool)
    for k, end_time in enumerate(h.index):
        pos = x.index.searchsorted(end_time, side="left") - 1
        if 0 <= pos < len(x):
            base_top.iloc[pos] = zone_top.iloc[k]
            base_bottom.iloc[pos] = zone_bottom.iloc[k]
            base_direction.iloc[pos] = zone_direction.iloc[k]
            base_overlap.iloc[pos] = (np.isfinite(zone_top.iloc[k]) and
                                      float(h.high.iloc[k]) >= zone_bottom.iloc[k] and
                                      float(h.low.iloc[k]) <= zone_top.iloc[k])
    valid = ((fvg_signal != 0) & (fvg_signal == base_direction) & base_overlap)
    signal = fvg_signal.where(valid, 0)
    setup = pd.Series("", index=x.index, dtype=object)
    setup.loc[valid] = fvg_setup.loc[valid] + "+ORDERBLOCK"
    bias = base_direction
    return signal, setup, atr_series, bias


def backtest(x: pd.DataFrame, signal: pd.Series, cfg: Config,
             atr_series: pd.Series | None = None, setup_series: pd.Series | None = None,
             flat_at_session_end: bool = False,
             session_exit_tz: str = "America/New_York",
             session_exit_time: str = "16:00") -> pd.DataFrame:
    """One position at a time. Stop-first when both stop and target touch in one OHLC bar."""
    rows = []
    idx = x.index
    slip = cfg.slippage_ticks * cfg.tick_size
    realized_pnl = 0.0
    peak_equity = 0.0
    risk_atr = atr_series if atr_series is not None else x.atr
    i = 0
    while i < len(x)-1:
        current_drawdown = (peak_equity - realized_pnl) if cfg.drawdown_mode == "trailing" else -realized_pnl
        if current_drawdown >= cfg.drawdown_dollars:
            break  # evaluation failed: stop taking new trades after the loss limit
        direction = int(signal.iloc[i])
        if direction == 0 or not np.isfinite(risk_atr.iloc[i]) or risk_atr.iloc[i] <= 0:
            i += 1; continue
        entry_i = i + 1
        entry_raw = float(x.open.iloc[entry_i])
        entry = entry_raw + direction * slip
        risk_pts = max(cfg.stop_atr * float(risk_atr.iloc[i]), cfg.tick_size)
        remaining_loss_budget = (cfg.drawdown_dollars - current_drawdown
                                 if cfg.drawdown_mode == "trailing"
                                 else cfg.drawdown_dollars + realized_pnl)
        if cfg.risk_fraction > 0:
            trade_risk_budget = min(cfg.drawdown_dollars * cfg.risk_fraction, remaining_loss_budget)
            per_unit_estimate = (risk_pts + 2 * slip) * cfg.point_value + 2 * cfg.commission_per_side
            trade_contracts = int(np.floor(trade_risk_budget / per_unit_estimate))
        else:
            trade_contracts = cfg.contracts
        def total_planned_risk(quantity: int) -> float:
            commissions = 2 * max(cfg.commission_per_side * quantity, cfg.commission_min_per_side)
            return (risk_pts + 2 * slip) * cfg.point_value * quantity + commissions
        planned_risk = total_planned_risk(trade_contracts)
        while trade_contracts > 0 and planned_risk > min(remaining_loss_budget,
                cfg.drawdown_dollars * cfg.risk_fraction if cfg.risk_fraction > 0 else remaining_loss_budget):
            trade_contracts -= 1
            planned_risk = total_planned_risk(trade_contracts)
        if trade_contracts < 1 or planned_risk > remaining_loss_budget:
            i += 1
            continue  # skip if one contract exceeds per-trade or remaining evaluation risk budget
        stop = entry - direction * risk_pts
        target = entry + direction * risk_pts * cfg.rr
        exit_i, exit_price, reason = len(x)-1, float(x.close.iloc[-1]), "end_of_data"
        be_moved, mfe = False, 0.0
        for j in range(entry_i, len(x)):
            if flat_at_session_end:
                ts = idx[j]
                local_ts = ts.tz_convert(session_exit_tz) if ts.tzinfo else ts.tz_localize(session_exit_tz)
                local_entry = idx[entry_i].tz_convert(session_exit_tz) if idx[entry_i].tzinfo else idx[entry_i].tz_localize(session_exit_tz)
                exit_h, exit_m = map(int, session_exit_time.split(":"))
                session_start = cfg.asia_start if cfg.session == "asia" else cfg.ny_start
                start_h, start_m = map(int, session_start.split(":"))
                start_min = start_h * 60 + start_m
                end_min = exit_h * 60 + exit_m
                entry_min = local_entry.hour * 60 + local_entry.minute
                overnight = start_min > end_min
                exit_date = local_entry.date()
                if overnight and entry_min >= start_min:
                    exit_date += pd.Timedelta(days=1)
                exit_at = pd.Timestamp(f"{exit_date} {session_exit_time}", tz=session_exit_tz)
                if local_ts >= exit_at:
                    exit_i, exit_price, reason = j, float(x.open.iloc[j]), "session_end"
                    break
            hi, lo = float(x.high.iloc[j]), float(x.low.iloc[j])
            mfe = max(mfe, (hi - entry) if direction > 0 else (entry - lo))
            stopped = lo <= stop if direction > 0 else hi >= stop
            targeted = hi >= target if direction > 0 else lo <= target
            if stopped:  # conservative ordering if both are touched in the same bar
                # A stop-market order can slip through a gap; use the worse opening price.
                gap_fill = min(stop, float(x.open.iloc[j])) if direction > 0 else max(stop, float(x.open.iloc[j]))
                exit_i, exit_price, reason = j, gap_fill, "breakeven" if be_moved else "stop"
                break
            if targeted:
                exit_i, exit_price, reason = j, target, "target"
                break
            # Exit on opposite closed-bar signal; fill at next bar open.
            if j < len(x)-1 and int(signal.iloc[j]) == -direction:
                exit_i, exit_price, reason = j+1, float(x.open.iloc[j+1]), "signal"
                break
            # Intrabar order is unknown, so a breakeven stop only protects from the next bar on.
            if cfg.breakeven_r > 0 and not be_moved and mfe >= cfg.breakeven_r * risk_pts:
                stop, be_moved = entry, True
        # Adverse slippage applied on both sides. Commissions are round-turn per contract.
        if reason == "stop": exit_price -= direction * slip
        else: exit_price -= direction * slip
        gross = (exit_price-entry) * direction * cfg.point_value * trade_contracts
        fees = 2 * max(cfg.commission_per_side * trade_contracts, cfg.commission_min_per_side)
        rows.append({"entry_time":idx[entry_i], "exit_time":idx[exit_i], "direction":direction,
                     "entry":entry, "exit":exit_price, "reason":reason,
                     "contracts":trade_contracts,
                     "gross_pnl":gross, "fees":fees, "pnl":gross-fees,
                     "risk_dollars":risk_pts*cfg.point_value*trade_contracts,
                     "rr_target":cfg.rr, "bars_held":exit_i-entry_i,
                     "mfe_r":mfe / risk_pts,
                     "setup_type":setup_series.iloc[i] if setup_series is not None else ""})
        realized_pnl += gross - fees
        peak_equity = max(peak_equity, realized_pnl)
        i = max(exit_i, entry_i) + 1
    columns = ["entry_time", "exit_time", "direction", "entry", "exit", "reason", "contracts",
               "gross_pnl", "fees", "pnl", "risk_dollars", "rr_target", "bars_held", "mfe_r", "setup_type"]
    return pd.DataFrame(rows, columns=columns)


def monte_carlo(trades: pd.DataFrame, cfg: Config, paths: int = 10000, seed: int = 7) -> dict:
    """Bootstrap trade outcomes for path-order risk; null test randomizes signs around zero edge."""
    pnl = trades.pnl.to_numpy(dtype=float)
    if len(pnl) == 0: return {"trades":0, "pass_probability":np.nan, "null_pass_probability":np.nan}
    rng = np.random.default_rng(seed)
    n = len(pnl)
    sample = rng.choice(pnl, size=(paths, n), replace=True)
    def pass_rate(samples: np.ndarray) -> float:
        equity = np.cumsum(samples, axis=1)
        peak = np.maximum.accumulate(np.maximum(equity, 0), axis=1)
        dd = peak - equity
        target_hit = (equity >= cfg.target_dollars).any(axis=1)
        if cfg.drawdown_mode == "trailing": fail = (dd >= cfg.drawdown_dollars).any(axis=1)
        else: fail = (equity <= -cfg.drawdown_dollars).any(axis=1)
        first_target = np.where(target_hit, (equity >= cfg.target_dollars).argmax(axis=1), n+1)
        first_fail = np.where(fail, (dd >= cfg.drawdown_dollars).argmax(axis=1) if cfg.drawdown_mode == "trailing" else (equity <= -cfg.drawdown_dollars).argmax(axis=1), n+1)
        return float(np.mean(first_target < first_fail))
    observed = pass_rate(sample)
    # Sign-flip null: preserve observed trade magnitudes, remove directional edge.
    null = rng.choice(np.array([-1.0, 1.0]), size=(paths,n)) * np.abs(rng.choice(pnl, size=(paths,n), replace=True))
    return {"trades":n, "pass_probability":observed, "null_pass_probability":pass_rate(null)}


def save_dashboard(trades: pd.DataFrame, sweep: list[tuple[float, float, float]],
                   mc: dict, cfg: Config, strategy: str, session: str, instrument: str,
                   out_path: str | Path) -> Path:
    """Save a readable four-panel PNG summary next to the trades CSV."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    out_path = Path(out_path)
    chart_path = out_path.with_name(f"{out_path.stem}_dashboard.png")
    pnl = trades.pnl.astype(float) if len(trades) else pd.Series(dtype=float)
    equity = pnl.cumsum()
    drawdown = equity.cummax().clip(lower=0) - equity if len(equity) else equity
    win_rate = float(pnl.gt(0).mean()) if len(pnl) else 0.0
    mean_day = (trades.assign(day=pd.to_datetime(trades.entry_time).dt.date)
                .groupby("day").size().mean()) if len(trades) else 0.0
    net = float(pnl.sum()) if len(pnl) else 0.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"{strategy.replace('_', ' ').title()} | {session.replace('_', ' ').title()} | {instrument} backtest",
                 fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    if len(equity):
        ax.plot(np.arange(1, len(equity)+1), equity, color="#1769aa", linewidth=1.8)
        ax.axhline(cfg.target_dollars, color="#20854e", linestyle="--", label=f"Target +${cfg.target_dollars:,.0f}")
        ax.axhline(-cfg.drawdown_dollars, color="#c43c39", linestyle="--", label=f"Start balance limit -${cfg.drawdown_dollars:,.0f}")
        ax.fill_between(np.arange(1, len(equity)+1), equity, 0, where=equity < 0,
                        color="#d9534f", alpha=.12)
    ax.set_title("Cumulative net P&L by trade")
    ax.set_xlabel("Trade number"); ax.set_ylabel("Net P&L ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if len(drawdown):
        ax.fill_between(np.arange(1, len(drawdown)+1), 0, drawdown, color="#d9534f", alpha=.65)
        ax.plot(np.arange(1, len(drawdown)+1), drawdown, color="#9c2f2c", linewidth=1)
    ax.set_title("Peak-to-valley drawdown")
    ax.set_xlabel("Trade number"); ax.set_ylabel("Drawdown ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax.grid(alpha=.25)

    ax = axes[1, 0]
    rr, net_by_rr, _ = zip(*sweep) if sweep else ([], [], [])
    if sweep:
        colors = ["#20854e" if v >= 0 else "#c43c39" for v in net_by_rr]
        ax.bar(rr, net_by_rr, width=.36, color=colors, alpha=.85)
        ax.axhline(0, color="black", linewidth=.8)
    ax.set_title("Reward:risk sweep (in-sample)")
    ax.set_xlabel("Target R multiple"); ax.set_ylabel("Net P&L ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax.grid(axis="y", alpha=.25)

    ax = axes[1, 1]
    ax.axis("off")
    summary = (f"TRADES     {len(trades):,}\n"
               f"NET P&L     ${net:,.2f}\n"
               f"WIN RATE    {win_rate:.1%}\n"
               f"TRADES/DAY  {mean_day:.2f}\n"
               f"MC PASS     {mc.get('pass_probability', np.nan):.1%}\n"
               f"ZERO-EDGE   {mc.get('null_pass_probability', np.nan):.1%}\n\n"
               f"Costs: {cfg.slippage_ticks:g} tick(s) slippage per side; "
               f"${cfg.commission_per_side:.2f} commission per side\n"
               f"Drawdown mode: {cfg.drawdown_mode} from starting balance")
    ax.text(.03, .95, "RESULTS AT A GLANCE", fontsize=15, fontweight="bold", va="top")
    ax.text(.03, .82, summary, fontsize=13, va="top", linespacing=1.65,
            family="DejaVu Sans Mono")
    fig.savefig(chart_path, dpi=160, facecolor="white")
    plt.close(fig)
    return chart_path


def save_comparison_report(results: list[dict], strategy: str, instrument: str,
                           out_path: str | Path, cfg: Config) -> Path:
    """Save one visual report comparing both sessions and their RR sweeps."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    out_path = Path(out_path)
    report_path = out_path.with_name(f"{out_path.stem}_report.png")
    colors = {"asia": "#d27b30", "new_york": "#1769aa"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    label = "Lima" if strategy == "lima" else strategy.replace("_", " ").title()
    fig.suptitle(f"{label} | {instrument} | Asia vs New York", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    for result in results:
        trades = result["trades"]
        if len(trades):
            curve = trades.pnl.cumsum()
            ax.plot(np.arange(1, len(curve)+1), curve,
                    label=result["session"].replace("_", " ").title(),
                    color=colors.get(result["session"], "#555555"), linewidth=1.8)
    ax.axhline(cfg.target_dollars, color="#20854e", linestyle="--", label=f"Target +${cfg.target_dollars:,.0f}")
    ax.set_title("Cumulative net P&L by trade"); ax.set_xlabel("Trade number"); ax.set_ylabel("P&L ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for result in results:
        trades = result["trades"]
        if len(trades):
            curve = trades.pnl.cumsum()
            dd = curve.cummax().clip(lower=0) - curve
            ax.plot(np.arange(1, len(dd)+1), dd,
                    label=result["session"].replace("_", " ").title(),
                    color=colors.get(result["session"], "#555555"), linewidth=1.8)
    ax.axhline(cfg.drawdown_dollars, color="#bd4141", linestyle="--", label=f"Limit ${cfg.drawdown_dollars:,.0f}")
    ax.set_title("Closed-trade peak-to-valley drawdown"); ax.set_xlabel("Trade number"); ax.set_ylabel("Drawdown ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for result in results:
        sweep = result["sweep"]
        if sweep:
            ax.plot([v[0] for v in sweep], [v[1] for v in sweep], marker="o",
                    label=result["session"].replace("_", " ").title(),
                    color=colors.get(result["session"], "#555555"))
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_title("In-sample RR sweep"); ax.set_xlabel("Target R multiple"); ax.set_ylabel("Net P&L ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[1, 1]; ax.axis("off")
    lines = ["SESSION       NET P&L       TRADES   WIN RATE   MC PASS"]
    for result in results:
        trades = result["trades"]; n = len(trades)
        net = float(trades.pnl.sum()) if n else 0.0
        win = float(trades.pnl.gt(0).mean()) if n else 0.0
        mc_pass = result["mc"].get("pass_probability", np.nan)
        lines.append(f"{result['session'].upper():12s} ${net:>9,.2f}   {n:>5d}    {win:>6.1%}    {mc_pass:>6.1%}")
    lines.extend(["", f"Costs: {cfg.slippage_ticks:g} tick(s) slippage/side; ${cfg.commission_per_side:.2f} commission/side",
                  f"One {instrument} contract; trailing drawdown on closed-trade equity"])
    ax.text(.02, .96, "RESULTS AT A GLANCE", fontsize=14, fontweight="bold", va="top")
    ax.text(.02, .82, "\n".join(lines), fontsize=10.5, va="top", family="DejaVu Sans Mono", linespacing=1.7)
    fig.savefig(report_path, dpi=160, facecolor="white")
    plt.close(fig)
    return report_path


def save_fvg_matrix_report(results: list[dict], instrument: str, out_path: str | Path,
                           cfg: Config, vwap_filter: bool = False,
                           structure_filter: bool = False) -> Path:
    """Compact annotated heatmap report for FVG/IFVG timeframe, session, and mode comparisons."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    out_path = Path(out_path)
    report_path = out_path.with_name(f"{out_path.stem}_report.png")
    row_labels = [f"{tf}m  {session.replace('_', ' ').title()}"
                  for tf in (5, 15, 30) for session in ("asia", "new_york")]
    modes = ["baseline", "structure", "orderblock", "structure_orderblock"]
    mode_labels = ["Base", "BOS/CHOCH", "Order block", "Both"]
    metrics = {name: np.full((6, 4), np.nan) for name in
               ("net", "drawdown", "trades", "win", "mc", "null")}
    for r in results:
        i = (5, 15, 30).index(int(r["timeframe"])) * 2 + (0 if r["session"] == "asia" else 1)
        j = modes.index(r.get("mode", "baseline"))
        t = r["trades"]
        eq = t.pnl.cumsum() if len(t) else pd.Series(dtype=float)
        metrics["net"][i, j] = float(t.pnl.sum()) if len(t) else 0.0
        metrics["drawdown"][i, j] = float((eq.cummax().clip(lower=0)-eq).max()) if len(t) else 0.0
        metrics["trades"][i, j] = len(t)
        metrics["win"][i, j] = float(t.pnl.gt(0).mean()) if len(t) else 0.0
        metrics["mc"][i, j] = float(r["mc"].get("pass_probability", np.nan))
        metrics["null"][i, j] = float(r["mc"].get("null_pass_probability", np.nan))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    vwap_label = " + session VWAP filter" if vwap_filter else ""
    fig.suptitle(f"FVG + IFVG Confluence Comparison | {instrument}{vwap_label}\n"
                 "Rows = timeframe/session   •   Columns = filters",
                 fontsize=17, fontweight="bold")
    specs = [
        ("net", "Net P&L ($)", "RdYlGn", "${:,.0f}"),
        ("drawdown", "Peak-to-valley drawdown ($)", "YlOrRd", "${:,.0f}"),
        ("trades", "Trades", "Blues", "{:.0f}"),
        ("win", "Win rate", "Purples", "{:.1%}"),
        ("mc", "Monte Carlo target pass", "Greens", "{:.1%}"),
        ("null", "Zero-edge pass probability", "Greys", "{:.1%}"),
    ]
    for ax, (key, title, cmap, fmt) in zip(axes.flat, specs):
        values = metrics[key]
        if key == "net":
            vmax = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1.0)
            image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
        else:
            image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=9)
        ax.set_xticks(np.arange(4), mode_labels, fontsize=8)
        ax.set_yticks(np.arange(6), row_labels, fontsize=9)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(6):
            for j in range(4):
                value = values[i, j]
                label = "—" if not np.isfinite(value) else fmt.format(value)
                ax.text(j, i, label, ha="center", va="center", fontsize=9, fontweight="bold", color="#202020")
        fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
    fig.savefig(report_path, dpi=180, facecolor="white")
    plt.close(fig)
    return report_path


def save_strategy_comparison_report(results: list[dict], out_path: str | Path,
                                    cfg: Config) -> Path:
    """One-page scorecard for selected strategy variants."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    out_path = Path(out_path)
    report_path = out_path.with_name(f"{out_path.stem}_report.png")
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    columns = ["Strategy", "Contract", "Avg/max qty", "RR", "Trades", "Win rate", "Net P&L",
               "Max DD", "Trades/day", "MC pass", "Zero-edge"]
    cells = []
    for r in results:
        t = r["trades"]
        net = float(t.pnl.sum()) if len(t) else 0.0
        eq = t.pnl.cumsum() if len(t) else pd.Series(dtype=float)
        dd = float((eq.cummax().clip(lower=0)-eq).max()) if len(t) else 0.0
        wins = float(t.pnl.gt(0).mean()) if len(t) else 0.0
        days = t.assign(day=pd.to_datetime(t.entry_time).dt.date).groupby("day").size() if len(t) else pd.Series(dtype=float)
        qty = f"{t.contracts.mean():.1f}/{t.contracts.max()}" if len(t) and "contracts" in t else "0/0"
        mc_value = r["mc"].get("pass_probability", np.nan)
        null_value = r["mc"].get("null_pass_probability", np.nan)
        cells.append([r["short"], r["contract"], qty, f"{r['rr']:.1f}", f"{len(t)}", f"{wins:.1%}",
                      f"${net:,.0f}", f"${dd:,.0f}", f"{days.mean():.1f}" if len(days) else "0.0",
                      f"{mc_value:.1%}" if np.isfinite(mc_value) else "—",
                      f"{null_value:.1%}" if np.isfinite(null_value) else "—"])
    table = ax.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center",
                     colWidths=[.22, .10, .09, .055, .06, .08, .085, .075, .09, .075, .07])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#24364b"); cell.set_text_props(color="white", weight="bold")
        elif col == 6:
            cell.set_facecolor("#d9efe2" if results[row-1]["trades"].pnl.sum() >= 0 else "#f6dddd")
    ax.set_title("Strategy scorecard — risk-sized position", fontsize=15, weight="bold", pad=12)

    ax1 = fig.add_subplot(grid[1, 0])
    names = [r["short"] for r in results]
    net_values = [float(r["trades"].pnl.sum()) if len(r["trades"]) else 0.0 for r in results]
    bar_colors = ["#32865b" if v >= 0 else "#bd4141" for v in net_values]
    ax1.barh(names, net_values, color=bar_colors)
    ax1.axvline(0, color="#333333", linewidth=.8)
    ax1.set_title("Net P&L by strategy"); ax1.set_xlabel("Net P&L ($)")
    ax1.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax1.grid(axis="x", alpha=.2)
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(grid[1, 1])
    dd_values = []
    for r in results:
        t = r["trades"]
        curve = t.pnl.cumsum() if len(t) else pd.Series(dtype=float)
        dd_values.append(float((curve.cummax().clip(lower=0)-curve).max()) if len(t) else 0.0)
    ax2.barh(names, dd_values, color=["#bd4141" if v > cfg.drawdown_dollars else "#d59a37" for v in dd_values])
    ax2.axvline(cfg.drawdown_dollars, color="#bd4141", linestyle="--", label=f"Limit ${cfg.drawdown_dollars:,.0f}")
    ax2.set_title("Closed-trade drawdown"); ax2.set_xlabel("Drawdown ($)")
    ax2.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}")); ax2.grid(axis="x", alpha=.2); ax2.legend()

    fig.suptitle("NQ Strategy Comparison | New York session", fontsize=19, weight="bold")
    risk_label = f"Risk cap: {cfg.risk_fraction:.0%} of $1,000 drawdown allowance per trade. " if cfg.risk_fraction > 0 else "Fixed quantity. "
    fig.text(.01, -.005, "Same CSV period. " + risk_label + "MNQ data scaled by point value for NQ Mini rows. "
             f"Slippage: {cfg.slippage_ticks:g} tick/side; commission: ${cfg.commission_per_side:.2f}/side; "
             f"drawdown limit: ${cfg.drawdown_dollars:,.0f}. MC figures are resampled trade-sequence estimates.", fontsize=8)
    fig.savefig(report_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return report_path


def save_strategy_grid_report(summary: pd.DataFrame, out_path: str | Path,
                              title: str = "New York Strategy Parameter Grid | MNQ Micro / NQ Mini",
                              footer: str | None = None) -> Path:
    """One-page overview of P&L, drawdown, and Monte Carlo pass across the parameter grid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    out_path = Path(out_path)
    report_path = out_path.with_name(f"{out_path.stem}_report.png")
    specs = list(summary[["strategy", "short"]].drop_duplicates().itertuples(index=False, name=None))
    report_title = title
    metrics = [("net_pnl", "Net P&L ($)", "RdYlGn", "${:,.0f}"),
               ("max_drawdown", "Closed-trade drawdown ($)", "YlOrRd", "${:,.0f}"),
               ("mc_pass", "Monte Carlo target pass", "Greens", "{:.0%}")]
    fig, axes = plt.subplots(len(specs), len(metrics), figsize=(18, 3.1*len(specs)), constrained_layout=True)
    risks = sorted(summary.risk_fraction.unique())
    rrs = sorted(summary.rr.unique())
    for row, (strategy, short) in enumerate(specs):
        sub = summary[summary.strategy == strategy]
        for col, (key, metric_title, cmap, fmt) in enumerate(metrics):
            ax = axes[row, col]
            matrix = (sub.pivot(index="risk_fraction", columns="rr", values=key)
                      .reindex(index=risks, columns=rrs).to_numpy(dtype=float))
            if key == "net_pnl":
                limit = max(float(np.nanmax(np.abs(matrix))), 1.0)
                image = ax.imshow(matrix, cmap=cmap,
                                  norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
            else:
                image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)
            ax.set_xticks(np.arange(len(rrs)), [f"{v:g}" for v in rrs])
            ax.set_yticks(np.arange(len(risks)), [f"{v:.0%}" for v in risks])
            if row == 0: ax.set_title(metric_title, fontsize=12, weight="bold")
            if col == 0: ax.set_ylabel(f"{short}\nRisk per trade", fontsize=10, weight="bold")
            else: ax.set_ylabel("Risk per trade")
            ax.set_xlabel("Reward:risk target")
            ax.set_xticks(np.arange(-.5, len(rrs), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(risks), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            for i in range(len(risks)):
                for j in range(len(rrs)):
                    val = matrix[i, j]
                    ax.text(j, i, fmt.format(val) if np.isfinite(val) else "—",
                            ha="center", va="center", fontsize=7.5, color="#202020")
            fig.colorbar(image, ax=ax, fraction=.045, pad=.025)
    fig.suptitle(f"{report_title}\nRows: risk cap as % of $1,000 drawdown | Columns: reward:risk", fontsize=17, weight="bold")
    fig.text(.01, -.002, footer or ("Net and drawdown include slippage and commission. MC pass is a bootstrap of realized trade P&L; "
             "position sizing is not recomputed inside each simulated path. Evaluate out of sample."), fontsize=8)
    fig.savefig(report_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return report_path


def run_cross_instrument_grid(args: argparse.Namespace) -> None:
    """Run selected NY strategies across Micro index and metals futures with contract-specific sizing."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    markets = [
        {"asset":"MES", "symbol":"MES=F", "type":"Micro E-mini S&P 500", "point_value":5.0, "market_multiplier":5.0, "tick_size":0.25},
        {"asset":"MYM", "symbol":"MYM=F", "type":"Micro E-mini Dow", "point_value":0.50, "market_multiplier":0.50, "tick_size":1.0},
        {"asset":"MGC", "symbol":"MGC=F", "type":"Micro Gold", "point_value":10.0, "market_multiplier":10.0, "tick_size":0.10},
        {"asset":"MNQ", "symbol":"MNQ=F", "type":"Micro E-mini Nasdaq-100", "point_value":2.0, "market_multiplier":2.0, "tick_size":0.25},
    ]
    for market in markets:
        market["commission_per_side"] = 2.50
        market["commission_min_per_side"] = 0.0
        market["slippage_ticks"] = 1.0
        market["file"] = download_yahoo(market["symbol"], args.period, "5m",
                         data_dir / f"{market['asset'].lower()}_yahoo_5m.csv", prepost=True)
    strategy_defs = [
        {"key":"lima", "name":"Lima: Bollinger + RSI", "rr_reference":1.0},
        {"key":"ob15", "name":"15m FVG/IFVG + OB + BOS/CHOCH", "rr_reference":3.5},
        {"key":"ob30", "name":"30m FVG/IFVG + OB + BOS/CHOCH", "rr_reference":3.0},
    ]
    risks = [0.10, 0.20, 0.30, 0.40, 0.50]
    rrs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    rows: list[dict] = []
    raw_bars: dict[str, pd.DataFrame] = {}
    for market in markets:
        load_cfg = Config(session="new_york", tick_size=market["tick_size"],
                          point_value=market["point_value"],
                          commission_per_side=market["commission_per_side"],
                          commission_min_per_side=market["commission_min_per_side"],
                          slippage_ticks=market["slippage_ticks"])
        raw_bars[market["asset"]] = load_bars(market["file"], load_cfg, data_timezone="America/New_York")
    common_start = max(frame.index.min() for frame in raw_bars.values())
    common_end = min(frame.index.max() for frame in raw_bars.values())
    for market in markets:
        load_cfg = Config(session="new_york", tick_size=market["tick_size"],
                          point_value=market["point_value"],
                          commission_per_side=market["commission_per_side"],
                          commission_min_per_side=market["commission_min_per_side"],
                          slippage_ticks=market["slippage_ticks"])
        common_bars = raw_bars[market["asset"]].loc[
            (raw_bars[market["asset"]].index >= common_start) &
            (raw_bars[market["asset"]].index <= common_end)]
        x = indicators(common_bars)
        strategy_signals: dict[str, tuple[pd.Series, pd.Series | None, pd.Series | None]] = {}
        strategy_signals["lima"] = (make_signals(x, "mean_reversion_bb_rsi", load_cfg), None, None)
        for key, minutes in (("ob15", 15), ("ob30", 30)):
            sig, setup, atr_tf, _ = make_fvg_orderblock_signals(x, minutes)
            vwap = session_vwap(x, load_cfg)
            active = in_session(x.index, load_cfg)
            aligned = ((sig > 0) & (x.close > vwap)) | ((sig < 0) & (x.close < vwap))
            strategy_signals[key] = (sig.where(active & aligned, 0), setup.where(active & aligned, ""), atr_tf)

        for strategy in strategy_defs:
            raw_signal, raw_setup, atr_tf = strategy_signals[strategy["key"]]
            for risk_fraction in risks:
                for rr in rrs:
                    cfg = Config(session="new_york", tick_size=market["tick_size"],
                                 point_value=market["point_value"], rr=rr, contracts=1,
                                 risk_fraction=risk_fraction,
                                 slippage_ticks=market["slippage_ticks"],
                                 commission_per_side=market["commission_per_side"],
                                 commission_min_per_side=market["commission_min_per_side"])
                    trades = backtest(x, raw_signal, cfg, atr_tf, raw_setup)
                    equity = trades.pnl.cumsum() if len(trades) else pd.Series(dtype=float)
                    dd = float((equity.cummax().clip(lower=0)-equity).max()) if len(trades) else 0.0
                    by_day = (trades.assign(day=pd.to_datetime(trades.entry_time).dt.date).groupby("day").size()
                              if len(trades) else pd.Series(dtype=float))
                    mc = monte_carlo(trades, cfg)
                    rows.append({
                        "asset":market["asset"], "symbol":market["symbol"], "instrument_type":market["type"],
                        "strategy":f"{market['asset']} | {strategy['name']}",
                        "short":f"{market['asset']} — {strategy['key'].upper()}",
                        "reference_rr":strategy["rr_reference"],
                        "risk_fraction":risk_fraction,
                        "risk_cap_dollars":risk_fraction*cfg.drawdown_dollars,
                        "rr":rr, "trades":len(trades),
                        "avg_quantity":float(trades.contracts.mean()) if len(trades) else 0.0,
                        "max_quantity":int(trades.contracts.max()) if len(trades) else 0,
                        "avg_market_notional_usd":float((trades.entry * trades.contracts * market["market_multiplier"]).mean()) if len(trades) else 0.0,
                        "max_market_notional_usd":float((trades.entry * trades.contracts * market["market_multiplier"]).max()) if len(trades) else 0.0,
                        "win_rate":float(trades.pnl.gt(0).mean()) if len(trades) else 0.0,
                        "net_pnl":float(trades.pnl.sum()) if len(trades) else 0.0,
                        "max_drawdown":dd,
                        "trades_per_day":float(by_day.mean()) if len(by_day) else 0.0,
                        "mc_pass":float(mc.get("pass_probability", np.nan)),
                        "null_pass":float(mc.get("null_pass_probability", np.nan)),
                        "commission_per_side":market["commission_per_side"],
                        "commission_min_per_side":market["commission_min_per_side"],
                        "slippage_ticks_per_side":market["slippage_ticks"],
                        "bars":len(x), "data_start":x.index.min(), "data_end":x.index.max(),
                    })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False)
    report = save_strategy_grid_report(
        summary, args.out, title="MES / MYM / MGC / MNQ Micro Futures Grid",
        footer=("5m Yahoo continuous futures; all trades restricted to New York hours. "
                "One tick slippage/side and $2.50 commission/side per contract. Contract multipliers and tick sizes follow CME specs. "
                "MC resamples realized P&L; sizing is not recomputed. Yahoo continuous series may contain roll discontinuities; validate out of sample."))
    for market in markets:
        print(f"Downloaded {market['symbol']} 5m data ({len(pd.read_csv(market['file'])):,} bars) to {market['file']}")
    print(f"Aligned test window: {common_start} through {common_end}")
    print("Best net-P&L in-sample configuration per instrument/strategy:")
    for _, sub in summary.groupby(["asset", "strategy"], sort=False):
        best = sub.loc[sub.net_pnl.idxmax()]
        print(f"{best['strategy']}: risk={best.risk_fraction:.0%} (${best.risk_cap_dollars:.0f}) "
              f"RR={best.rr:.1f} trades={int(best.trades)} qty_avg/max={best.avg_quantity:.1f}/{int(best.max_quantity)} "
              f"win={best.win_rate:.1%} net=${best.net_pnl:.2f} DD=${best.max_drawdown:.2f} "
              f"MC_pass={best.mc_pass:.1%}")
    print(f"Saved {len(summary)}-configuration summary to {args.out}")
    print(f"Saved report to {report}")


STRATEGIES = ["lima", "mean_reversion_rsi", "mean_reversion_bb_rsi", "momentum", "ema_cloud", "bos_choch", "fvg", "ifvg", "fvg_vwap", "ifvg_vwap"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", help="OHLCV CSV with timestamp,open,high,low,close[,volume]")
    ap.add_argument("--download-yahoo", action="store_true", help="download recent Yahoo MNQ/NQ bars to CSV first")
    ap.add_argument("--symbol", default="MNQ=F", help="Yahoo futures symbol (default MNQ=F, Micro E-mini)")
    ap.add_argument("--period", default="60d", help="Yahoo lookback, e.g. 30d or 60d")
    ap.add_argument("--start", help="inclusive date YYYY-MM-DD; requires bars for this range")
    ap.add_argument("--end", help="exclusive date YYYY-MM-DD")
    ap.add_argument("--data-timezone", default="America/Chicago", help="timezone of naive CSV timestamps; FirstRateData files use America/New_York")
    ap.add_argument("--interval", default="5m", help="Yahoo bar size, e.g. 1m, 5m, 15m")
    ap.add_argument("--instrument", choices=["MNQ", "NQ"], default="MNQ")
    ap.add_argument("--strategy", choices=STRATEGIES, default="ema_cloud")
    ap.add_argument("--session", choices=["all","asia","new_york"], default="all")
    ap.add_argument("--compare-sessions", action="store_true", help="compare Asia and New York in one report")
    ap.add_argument("--fvg-matrix", action="store_true", help="run combined FVG/IFVG at 5m, 15m, and 30m in both sessions (six tests)")
    ap.add_argument("--fvg-structure-matrix", action="store_true", help="compare six FVG/IFVG tests against the same tests filtered by BOS/CHOCH bias")
    ap.add_argument("--fvg-orderblock-matrix", action="store_true", help="compare baseline, BOS/CHOCH, order-block, and combined FVG/IFVG confluence across six timeframe/session pairs")
    ap.add_argument("--strategy-scorecard", action="store_true", help="compare the five selected NY strategies in one CSV and one-page report")
    ap.add_argument("--strategy-grid", action="store_true", help="sweep five selected NY strategies over risk fractions .10-.50 and RR 1.0-3.5")
    ap.add_argument("--cross-instrument-grid", action="store_true", help="download MES, MYM, MGC, and MNQ 5m futures data and compare selected strategies")
    ap.add_argument("--vwap-filter", action="store_true", help="require longs above, shorts below the selected session VWAP")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--commission-per-side", type=float, default=2.50)
    ap.add_argument("--risk-fraction", type=float, default=0.10, help="dynamic scorecard risk cap per trade as a fraction of the $1,000 drawdown limit")
    ap.add_argument("--out", default="trades.csv")
    args = ap.parse_args()
    if args.strategy == "lima" and args.session == "all" and not args.compare_sessions:
        args.session = "new_york"
    if args.download_yahoo:
        default_data_name = f"{args.instrument.lower()}_yahoo.csv"
        args.csv = str(download_yahoo(args.symbol, args.period, args.interval,
                                      args.csv or default_data_name, args.start, args.end))
        print(f"Downloaded recent Yahoo data to {args.csv}")
    if args.cross_instrument_grid:
        run_cross_instrument_grid(args)
        return
    if not args.csv:
        ap.error("Provide a CSV path or use --download-yahoo")
    point_value = 2.0 if args.instrument == "MNQ" else 20.0
    base_cfg = Config(point_value=point_value, session=args.session, rr=args.rr,
                      slippage_ticks=args.slippage_ticks, commission_per_side=args.commission_per_side,
                      contracts=1)
    x = indicators(load_bars(args.csv, base_cfg, args.start, args.end, args.data_timezone))
    if args.strategy_scorecard:
        specs = [
            {"name":"RSI mean reversion, no BB", "short":"RSI no BB — NQ", "contract":"NQ Mini", "rr":2.0, "strategy":"mean_reversion_rsi", "instrument":"NQ"},
            {"name":"Lima: Bollinger + RSI", "short":"Lima BB+RSI — MNQ", "contract":"MNQ Micro", "rr":2.0, "strategy":"mean_reversion_bb_rsi", "instrument":"MNQ"},
        ]
        results = []
        combined = []
        for spec in specs:
            cfg = Config(point_value=20.0 if spec["instrument"] == "NQ" else 2.0,
                         session="new_york", rr=spec["rr"], contracts=1,
                         risk_fraction=args.risk_fraction,
                         slippage_ticks=args.slippage_ticks,
                         commission_per_side=args.commission_per_side)
            sig = make_signals(x, spec["strategy"], cfg)
            trades = backtest(x, sig, cfg)
            results.append({**spec, "trades":trades, "mc":monte_carlo(trades, cfg)})
        for minutes, kind in ((15, "FVG+IFVG VWAP"), (15, "Order block + BOS/CHOCH"), (30, "Order block + BOS/CHOCH")):
            cfg = Config(point_value=2.0, session="new_york", rr=args.rr, contracts=1,
                         risk_fraction=args.risk_fraction,
                         slippage_ticks=args.slippage_ticks,
                         commission_per_side=args.commission_per_side)
            if kind.startswith("FVG"):
                sig, setup, atr_tf = make_fvg_ifvg_signals(x, minutes)
            else:
                sig, setup, atr_tf, _ = make_fvg_orderblock_signals(x, minutes)
            vwap = session_vwap(x, cfg)
            aligned = ((sig > 0) & (x.close > vwap)) | ((sig < 0) & (x.close < vwap))
            sig = sig.where(aligned & in_session(x.index, cfg), 0)
            setup = setup.where(sig != 0, "")
            trades = backtest(x, sig, cfg, atr_tf, setup)
            short = (f"{minutes}m FVG/IFVG+VWAP — MNQ" if kind.startswith("FVG")
                     else f"{minutes}m FVG+OB/BOS — MNQ")
            results.append({"name":f"{minutes}m {kind}", "short":short,
                            "contract":"MNQ Micro", "rr":args.rr,
                            "trades":trades, "mc":monte_carlo(trades, cfg)})
        for r in results:
            trades = r["trades"]
            net = float(trades.pnl.sum()) if len(trades) else 0.0
            dd_curve = trades.pnl.cumsum() if len(trades) else pd.Series(dtype=float)
            dd = float((dd_curve.cummax().clip(lower=0)-dd_curve).max()) if len(trades) else 0.0
            win = float(trades.pnl.gt(0).mean()) if len(trades) else 0.0
            print(f"{r['name']} | {r['contract']} | RR={r['rr']:.1f} | trades={len(trades)} | "
                  f"net=${net:.2f} | win={win:.1%} | max_closed_DD=${dd:.2f} | "
                  f"qty_avg/max={trades.contracts.mean() if len(trades) else 0:.1f}/{trades.contracts.max() if len(trades) else 0} | "
                  f"MC_pass={r['mc'].get('pass_probability', np.nan):.1%} | "
                  f"zero_edge={r['mc'].get('null_pass_probability', np.nan):.1%}")
            tagged = trades.copy()
            tagged.insert(0, "strategy", r["name"])
            tagged.insert(0, "contract", r["contract"])
            tagged.insert(0, "rr", r["rr"])
            combined.append(tagged)
        pd.concat(combined, ignore_index=True).to_csv(args.out, index=False)
        report = save_strategy_comparison_report(results, args.out,
                  Config(slippage_ticks=args.slippage_ticks, commission_per_side=args.commission_per_side,
                         risk_fraction=args.risk_fraction))
        print(f"Saved combined trades to {args.out}")
        print(f"Saved one-page strategy report to {report}")
        return
    if args.strategy_grid:
        strategy_specs = [
            {"strategy":"RSI mean reversion, no BB", "short":"RSI no BB | NQ Mini", "contract":"NQ Mini", "instrument":"NQ", "rr_base":2.0, "kind":"rsi"},
            {"strategy":"Lima: Bollinger + RSI", "short":"Lima | MNQ Micro", "contract":"MNQ Micro", "instrument":"MNQ", "rr_base":2.0, "kind":"lima"},
            {"strategy":"15m FVG/IFVG + VWAP", "short":"15m FVG/IFVG | MNQ", "contract":"MNQ Micro", "instrument":"MNQ", "rr_base":3.5, "kind":"fvg15"},
            {"strategy":"15m FVG/IFVG + OB + BOS/CHOCH", "short":"15m FVG+OB+BOS | MNQ", "contract":"MNQ Micro", "instrument":"MNQ", "rr_base":3.5, "kind":"ob15"},
            {"strategy":"30m FVG/IFVG + OB + BOS/CHOCH", "short":"30m FVG+OB+BOS | MNQ", "contract":"MNQ Micro", "instrument":"MNQ", "rr_base":3.5, "kind":"ob30"},
        ]
        risks = [0.10, 0.20, 0.30, 0.40, 0.50]
        rrs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        signals = {}
        indicator_cfg = Config(session="new_york")
        for spec in strategy_specs:
            if spec["kind"] in ("rsi", "lima"):
                strat_name = "mean_reversion_rsi" if spec["kind"] == "rsi" else "mean_reversion_bb_rsi"
                signals[spec["kind"]] = (make_signals(x, strat_name, indicator_cfg), None, None)
            else:
                minutes = 15 if spec["kind"] in ("fvg15", "ob15") else 30
                if spec["kind"] == "fvg15":
                    raw_sig, raw_setup, atr_tf = make_fvg_ifvg_signals(x, minutes)
                else:
                    raw_sig, raw_setup, atr_tf, _ = make_fvg_orderblock_signals(x, minutes)
                vwap_cfg = Config(session="new_york")
                vwap = session_vwap(x, vwap_cfg)
                allowed = in_session(x.index, vwap_cfg)
                aligned = ((raw_sig > 0) & (x.close > vwap)) | ((raw_sig < 0) & (x.close < vwap))
                signals[spec["kind"]] = (raw_sig.where(allowed & aligned, 0), raw_setup.where(allowed & aligned, ""), atr_tf)

        summary_rows = []
        for spec in strategy_specs:
            raw_signal, raw_setup, atr_tf = signals[spec["kind"]]
            for risk_fraction in risks:
                for rr in rrs:
                    cfg = Config(point_value=20.0 if spec["instrument"] == "NQ" else 2.0,
                                 session="new_york", rr=rr, contracts=1,
                                 risk_fraction=risk_fraction, slippage_ticks=args.slippage_ticks,
                                 commission_per_side=args.commission_per_side)
                    trades = backtest(x, raw_signal, cfg, atr_tf, raw_setup)
                    equity = trades.pnl.cumsum() if len(trades) else pd.Series(dtype=float)
                    drawdown = float((equity.cummax().clip(lower=0)-equity).max()) if len(trades) else 0.0
                    mc = monte_carlo(trades, cfg)
                    by_day = (trades.assign(day=pd.to_datetime(trades.entry_time).dt.date).groupby("day").size()
                              if len(trades) else pd.Series(dtype=float))
                    summary_rows.append({
                        "strategy":spec["strategy"], "short":spec["short"], "contract":spec["contract"],
                        "risk_fraction":risk_fraction, "risk_cap_dollars":risk_fraction*cfg.drawdown_dollars,
                        "rr":rr, "trades":len(trades),
                        "avg_contracts":float(trades.contracts.mean()) if len(trades) else 0.0,
                        "max_contracts":int(trades.contracts.max()) if len(trades) else 0,
                        "win_rate":float(trades.pnl.gt(0).mean()) if len(trades) else 0.0,
                        "net_pnl":float(trades.pnl.sum()) if len(trades) else 0.0,
                        "max_drawdown":drawdown, "trades_per_day":float(by_day.mean()) if len(by_day) else 0.0,
                        "mc_pass":float(mc.get("pass_probability", np.nan)),
                        "null_pass":float(mc.get("null_pass_probability", np.nan)),
                    })
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(args.out, index=False)
        report = save_strategy_grid_report(summary, args.out)
        print("Top configuration per strategy by net P&L (in-sample):")
        for spec in strategy_specs:
            sub = summary[summary.strategy == spec["strategy"]]
            best = sub.loc[sub.net_pnl.idxmax()]
            print(f"{spec['strategy']}: risk={best.risk_fraction:.0%} (${best.risk_cap_dollars:.0f}/trade) "
                  f"RR={best.rr:.1f} trades={int(best.trades)} net=${best.net_pnl:.2f} "
                  f"win={best.win_rate:.1%} DD=${best.max_drawdown:.2f} MC_pass={best.mc_pass:.1%}")
        print(f"Saved 150-configuration summary to {args.out}")
        print(f"Saved one-page parameter grid report to {report}")
        return
    if args.fvg_matrix:
        results = []
        combined = []
        for minutes in (5, 15, 30):
            signal, setup, atr_timeframe = make_fvg_ifvg_signals(x, minutes)
            for session in ("asia", "new_york"):
                cfg = Config(**{**base_cfg.__dict__, "session":session})
                allowed = in_session(x.index, cfg)
                session_signal = signal.where(allowed, 0)
                session_setup = setup.where(allowed, "")
                if args.vwap_filter:
                    vwap = session_vwap(x, cfg)
                    aligned = ((session_signal > 0) & (x.close > vwap)) | ((session_signal < 0) & (x.close < vwap))
                    session_signal = session_signal.where(aligned, 0)
                    session_setup = session_setup.where(aligned, "")
                trades = backtest(x, session_signal, cfg, atr_timeframe, session_setup)
                dd = ((trades.pnl.cumsum().cummax().clip(lower=0)-trades.pnl.cumsum()).max()
                      if len(trades) else 0.0)
                mc = monte_carlo(trades, cfg)
                label = f"{minutes}m {session.replace('_',' ').title()}"
                print(f"instrument={args.instrument} contracts=1 setup=FVG+IFVG{' + VWAP' if args.vwap_filter else ''} timeframe={minutes}m "
                      f"session={session} trades={len(trades)} net=${trades.pnl.sum() if len(trades) else 0:.2f} "
                      f"win_rate={trades.pnl.gt(0).mean() if len(trades) else 0:.1%} "
                      f"max_closed_trade_drawdown=${dd:.2f} MC_pass={mc.get('pass_probability',np.nan):.1%} "
                      f"null_pass={mc.get('null_pass_probability',np.nan):.1%}")
                tagged = trades.copy()
                tagged.insert(0, "timeframe", f"{minutes}m")
                tagged.insert(0, "session", session)
                combined.append(tagged)
                results.append({"label":label, "timeframe":minutes, "session":session,
                                "trades":trades, "mc":mc})
        pd.concat(combined, ignore_index=True).to_csv(args.out, index=False)
        report = save_fvg_matrix_report(results, args.instrument, args.out, base_cfg, args.vwap_filter)
        print(f"Saved six-test trade details to {args.out}")
        print(f"Saved one six-test image report to {report}")
        return
    if args.fvg_structure_matrix:
        results = []
        combined = []
        for minutes in (5, 15, 30):
            base_signal, base_setup, atr_timeframe = make_fvg_ifvg_signals(x, minutes)
            confluence_signal, confluence_setup, _ = make_fvg_structure_signals(x, minutes)
            for session in ("asia", "new_york"):
                cfg = Config(**{**base_cfg.__dict__, "session":session})
                allowed = in_session(x.index, cfg)
                vwap = session_vwap(x, cfg) if args.vwap_filter else None
                for mode, raw_signal, raw_setup in (
                    ("baseline", base_signal, base_setup),
                    ("structure", confluence_signal, confluence_setup),
                ):
                    signal = raw_signal.where(allowed, 0)
                    setup = raw_setup.where(allowed, "")
                    if vwap is not None:
                        vwap_aligned = ((signal > 0) & (x.close > vwap)) | ((signal < 0) & (x.close < vwap))
                        signal = signal.where(vwap_aligned, 0)
                        setup = setup.where(vwap_aligned, "")
                    trades = backtest(x, signal, cfg, atr_timeframe, setup)
                    mc = monte_carlo(trades, cfg)
                    dd = ((trades.pnl.cumsum().cummax().clip(lower=0)-trades.pnl.cumsum()).max()
                          if len(trades) else 0.0)
                    label = f"{minutes}m {session.replace('_',' ').title()} {'+ BOS/CHOCH' if mode == 'structure' else 'baseline'}"
                    print(f"{label}: trades={len(trades)} net=${trades.pnl.sum() if len(trades) else 0:.2f} "
                          f"win={trades.pnl.gt(0).mean() if len(trades) else 0:.1%} drawdown=${dd:.2f}")
                    tagged = trades.copy()
                    tagged.insert(0, "mode", mode)
                    tagged.insert(0, "timeframe", f"{minutes}m")
                    tagged.insert(0, "session", session)
                    combined.append(tagged)
                    results.append({"label":label, "timeframe":minutes, "session":session,
                                    "mode":mode, "trades":trades, "mc":mc})
        pd.concat(combined, ignore_index=True).to_csv(args.out, index=False)
        report = save_fvg_matrix_report(results, args.instrument, args.out, base_cfg,
                                        args.vwap_filter, structure_filter=True)
        print(f"Saved 12 paired test results to {args.out}")
        print(f"Saved one comparison image to {report}")
        return
    if args.fvg_orderblock_matrix:
        results = []
        combined = []
        modes = ("baseline", "structure", "orderblock", "structure_orderblock")
        for minutes in (5, 15, 30):
            base_signal, base_setup, atr_timeframe = make_fvg_ifvg_signals(x, minutes)
            structure_signal, structure_setup, _ = make_fvg_structure_signals(x, minutes)
            ob_signal, ob_setup, _, ob_bias = make_fvg_orderblock_signals(x, minutes)
            both_signal = ob_signal.where(ob_signal == structure_signal, 0)
            both_setup = ob_setup.where(both_signal != 0, "")
            mode_data = {
                "baseline": (base_signal, base_setup),
                "structure": (structure_signal, structure_setup),
                "orderblock": (ob_signal, ob_setup),
                "structure_orderblock": (both_signal, both_setup),
            }
            for session in ("asia", "new_york"):
                cfg = Config(**{**base_cfg.__dict__, "session":session})
                allowed = in_session(x.index, cfg)
                vwap = session_vwap(x, cfg) if args.vwap_filter else None
                for mode in modes:
                    raw_signal, raw_setup = mode_data[mode]
                    signal = raw_signal.where(allowed, 0)
                    setup = raw_setup.where(allowed, "")
                    if vwap is not None:
                        aligned = ((signal > 0) & (x.close > vwap)) | ((signal < 0) & (x.close < vwap))
                        signal = signal.where(aligned, 0)
                        setup = setup.where(aligned, "")
                    trades = backtest(x, signal, cfg, atr_timeframe, setup)
                    mc = monte_carlo(trades, cfg)
                    dd = ((trades.pnl.cumsum().cummax().clip(lower=0)-trades.pnl.cumsum()).max()
                          if len(trades) else 0.0)
                    label = f"{minutes}m {session.replace('_',' ').title()} {mode.replace('_',' + ')}"
                    print(f"{label}: trades={len(trades)} net=${trades.pnl.sum() if len(trades) else 0:.2f} "
                          f"win={trades.pnl.gt(0).mean() if len(trades) else 0:.1%} drawdown=${dd:.2f} "
                          f"MC_pass={mc.get('pass_probability', np.nan):.1%} null_pass={mc.get('null_pass_probability', np.nan):.1%}")
                    tagged = trades.copy()
                    tagged.insert(0, "mode", mode)
                    tagged.insert(0, "timeframe", f"{minutes}m")
                    tagged.insert(0, "session", session)
                    combined.append(tagged)
                    results.append({"label":label, "timeframe":minutes, "session":session,
                                    "mode":mode, "orderblock":mode in ("orderblock", "structure_orderblock"),
                                    "trades":trades, "mc":mc})
        pd.concat(combined, ignore_index=True).to_csv(args.out, index=False)
        report = save_fvg_matrix_report(results, args.instrument, args.out, base_cfg,
                                        args.vwap_filter, structure_filter=True)
        print(f"Saved 24 paired results to {args.out}")
        print(f"Saved one comparison image to {report}")
        return
    if args.strategy == "bos_choch":
        raw_signal, raw_setup = make_bos_choch_signals(x)
    else:
        raw_signal = make_signals(x, args.strategy, base_cfg)
        raw_setup = pd.Series("", index=x.index, dtype=object)
    sessions = ["asia", "new_york"] if args.compare_sessions else [args.session]
    comparison = []
    combined_trades = []
    for session in sessions:
        cfg = Config(**{**base_cfg.__dict__, "session":session})
        allowed = in_session(x.index, cfg)
        session_signal = raw_signal.where(allowed, 0)
        session_setup = raw_setup.where(allowed, "")
        trades = backtest(x, session_signal, cfg, setup_series=session_setup)
        out_path = Path(args.out)
        if args.compare_sessions:
            tagged = trades.copy()
            tagged.insert(0, "session", session)
            combined_trades.append(tagged)
        else:
            trades.to_csv(out_path, index=False)
        net = trades.pnl.sum() if len(trades) else 0.0
        equity = trades.pnl.cumsum() if len(trades) else pd.Series(dtype=float)
        dd = (equity.cummax().clip(lower=0) - equity).max() if len(trades) else 0.0
        by_day = trades.assign(day=pd.to_datetime(trades.entry_time).dt.date).groupby("day").size()
        win_rate = trades.pnl.gt(0).mean() if len(trades) else 0.0
        trades_per_day = by_day.mean() if len(by_day) else 0.0
        print(f"instrument={args.instrument} contracts=1 strategy={args.strategy} session={session} "
              f"trades={len(trades)} net=${net:.2f} win_rate={win_rate:.1%} "
              f"max_peak_to_valley_drawdown=${dd:.2f} trades/day={trades_per_day:.2f}")
        mc = monte_carlo(trades, cfg)
        print("Monte Carlo (10,000 paths):", mc)
        sweep = []
        for rr in np.arange(0.5, 5.01, 0.5):
            candidate = Config(**{**cfg.__dict__, "rr":float(rr)})
            candidate_trades = backtest(x, session_signal, candidate, setup_series=session_setup)
            sweep.append((float(rr), float(candidate_trades.pnl.sum()) if len(candidate_trades) else 0.0,
                          float(candidate_trades.pnl.gt(0).mean()) if len(candidate_trades) else 0.0))
        best = max(sweep, key=lambda row: row[1])
        print(f"Best in-sample net RR={best[0]:.1f} (net ${best[1]:.2f}, win rate {best[2]:.1%}); validate out-of-sample.")
        if args.compare_sessions:
            comparison.append({"session":session, "trades":trades, "sweep":sweep, "mc":mc})
        else:
            chart_path = save_dashboard(trades, sweep, mc, cfg, args.strategy, session, args.instrument, out_path)
            print(f"Saved trades to {out_path}")
            print(f"Saved visual dashboard to {chart_path}")
    if args.compare_sessions:
        print("\nSESSION COMPARISON — same data, one contract, same costs and risk rules")
        for result in comparison:
            trades = result["trades"]
            net = trades.pnl.sum() if len(trades) else 0.0
            win = trades.pnl.gt(0).mean() if len(trades) else 0.0
            print(f"{result['session']:10s} net=${net:,.2f} | trades={len(trades):4d} | win={win:.1%}")
        pd.concat(combined_trades, ignore_index=True).to_csv(args.out, index=False)
        report_path = save_comparison_report(comparison, args.strategy, args.instrument, args.out, base_cfg)
        print(f"Saved combined trades to {args.out}")
        print(f"Saved single session comparison report to {report_path}")


if __name__ == "__main__":
    main()
