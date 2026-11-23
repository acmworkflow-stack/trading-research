# TradingView indicator changes (Nov 2026)

File: tradingview/market_structure_fvg_playbook_v2.pine (Pine v6).

## Setups on the shared trade engine
| Nickname | Market / session | Strategy | R |
|---|---|---|---|
| GOLF | Gold NY | 15m ORB retest, 10m candles | 3 |
| HOTEL | Gold NY | BB(20, 2σ) + RSI 25/75, 5m | 3 |
| ALPHA | Gold Asia | 30m FVG/IFVG + order block | 3 |
| KILO | Gold Asia | BB(20, 2σ) + RSI 25/75, 5m | 2 |
| MIKE | Nasdaq NY | 30m FVG/IFVG + order block | 2.5 |
| TANGO | Nasdaq Asia | 15m FVG/IFVG + order block | 2 |
| SIERRA | Nasdaq Asia | 15m FVG/IFVG + BOS/CHoCH bias | 1 |
| VICTOR | Nasdaq Asia | FVG 5/15/30 alignment + Chicago-day VWAP, 5m | 2.5 |
| ECHO | Nasdaq Asia | Prior-NY volume profile VAH/VAL rejection, 60% VA | 3 |
| LIMA 5m | Nasdaq NY | BB(20, 2σ) + RSI 30/70 | 1 |
| MARCUS 15m / 30m | Nasdaq NY | FVG/IFVG + order block + NY VWAP | 3.5 / 3 |
Old basket: Oscar breakout / fakeout / retest only. Papa and November removed 2026-11-17 (archive/papa_november_archive.pine).

Rules match validate.py: next-open entry +1 tick, 1 ATR stop, stop-first, opposite-signal exit, flat 16:00 ET (NY) / 03:00 ET (Asia). Verified by replaying the Pine rules in Python on the backtest signals (identical except where Yahoo bars are missing).

## Display / alerts
- One dark running-P&L table (old setups, new setups, totals); lookback + "all history" toggle first in Settings; no drawdown or target stops.
- Dark bias table (5m, 15m, 1H, 4H, D, W): last closed candle vs EMA20/EMA50.
- Every entry: permanent marker, full levels for the latest trade, alert() message "TAG LONG/SHORT @ entry | SL | TP | contracts | flat by | score". Set one alert per chart with condition "Any alert() function call".
- Contract count from risk budget (default $75 = $600 daily limit ÷ 8).

## Fixes made along the way
Basket runtime error on gold charts removed; Marcus core matches the backtest; November first-bar bug fixed (now archived); entry-bar stop/target counted; old entries no longer lose their markers.
