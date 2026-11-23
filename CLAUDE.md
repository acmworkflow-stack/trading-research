# Project notes for Claude

Research on NQ/MNQ and GC/MGC futures, NY session (09:30-16:00 ET) and Asia session (19:00-03:00 ET). Evaluation-account rules used throughout: +$1,250 target, $1,000 end-of-day trailing drawdown, optional $600 daily loss limit.

Don't say "this is not financial advice"; the owner uses the information at his own discretion.

## Code layout
- `code/nq_backtest.py` is the engine. Signals form on closed bars and fill at the next bar's open + 1 tick slippage, $2.50 commission per side, stop = 1 ATR(14) of the signal timeframe, stop wins when stop and target touch in one bar. `Config.breakeven_r` moves the stop to entry after N R (off by default). Trades include `mfe_r`.
- `code/validate.py` runs 13 strategy families x NY/Asia with 6 pass/fail tests: initial (net>0, never -$1,000, >=20 trades), walk-forward (anchored 5 folds), PBO (CSCV, 8 blocks, <0.5), deflated Sharpe (>=0.95 over all trials), cost stress (1.5 tick, 2 tick, $3.75 comm, both; all must survive), nearby settings (>=60% of one-step neighbours profitable). The eval rule: a test is over at the first -$1,000 trailing closed-trade drawdown (stats() in validate.py). Usage: `python code/validate.py code <MGC|MNQ> <new_york|asia> <out_dir>`.
- `code/vp.py`: volume-profile families on the same harness. `code/study.py` + `code/evalsim.py`: confidence score, weekly bias and prop-firm evaluation simulation.
- `code/orderflow_study.py [--cuts N]`: TBBO order-flow confirmation study (C1 delta shift, C2 trapped traders, C3 aggressive push, C4 absorption vs controls B1 touch / B2 rejection) at prior-NY H/L, VA/POC, Asia H/L, opening range and 5m FVG zones, NQ+GC x NY/Asia, 1/2/3R. Caches front-month trades as 1m bars + 1m x tick footprint in `data/cache/tbbo_<mkt>_front.pkl.zst`; all rule choices are in its docstring; truncation-tested. Same tables in the dashboard's "Order flow" tab. First run (Nov 19 2026): 2 of 52 rule tests passed (NY opening range C1, C2), on 41-76 trades per market. Confirmation-count score (0-4, entry at the last confirmation) failed its fixed rule: Spearman +0.035 NQ / -0.013 GC, score>=2 avg R -0.16 / -0.24; the 11-combination table is exploratory only.
- `code/systematic_study.py`: overnight vs intraday (SPY/QQQ since inception, NQ/GC Databento), time-series momentum on 24 Yahoo futures (3/6/12m + blend, 10% vol target, as-is and roll-gap-zeroed), VIX regime on the setup trades. Yahoo daily data cached once in `data/cache/daily/`. Dashboard "Systematic" tab. First run (Nov 20 2026): Test 1 passes on SPY, QQQ, NQ; Test 2 fails (as-is ruined by a Yahoo 6J=F unit change on 2001-12-18; zeroed blend Sharpe 0.64 / 0.22 by half, DSR 0.925); Test 3 fails (both regimes negative in 2024 and 2025).
- `code/gex.py`: SPY/QQQ/^SPX option chains (<= 60 DTE, +/-15%), Black-Scholes GEX, walls, gamma flip, NQ/ES conversion, IV surface; one snapshot per ET day in `data/cache/gex/`. Yahoo's ^SPX chain has almost no open interest, so SPX GEX is suppressed (OI coverage < 50%). Dashboard "GEX" tab with Refresh. `continuous_1m` in app.py now also returns `adj` (real price = price - adj).
- Data loader expects `timestamp,open,high,low,close,volume`; timezone-aware timestamps. `validate.clean()` repairs single-bar bad prints.

## Standards (keep these)
- No lookahead: any new signal must pass the truncation test in `code/lookahead.py` (signals unchanged when future bars are removed).
- Compare variants in R and in eval pass probability, not just net $. Expected R drives eval pass rates far more than win rate.
- Few, stated hypotheses; report how many configs were tried; treat in-sample grid winners as unproven. Say plainly when evidence is thin.
- Every TradingView change is checked by replaying the Pine rules in Python against the backtest signals before shipping.

## TradingView indicator (`tradingview/market_structure_fvg_playbook_v2.pine`)
Pine v6. One trade engine (UDT `NewModel`) runs: GOLF, HOTEL, ALPHA, KILO (gold), MIKE, TANGO, SIERRA, VICTOR, ECHO (Nasdaq), plus LIMA 5m and MARCUS 15/30. Old basket keeps only Oscar breakout / fakeout / retest (Papa and November archived in `tradingview/archive`). One dark running-P&L table (top right, lookback setting at top of Settings, no drawdown stops), dark bias table (5m/15m/1H/4H/D/W, bottom left), score 0-100 and contract count in every entry label and alert(), $75 default risk budget. Stays under TradingView's 40 request.* call limit (about 26 used).

## Data
- `data/*_yahoo_5m.csv`: Aug 21 - Oct 30 2026, MNQ/MGC/MES/MYM/AAPL.
- `data/databento_ohlcv_1m`: OHLCV-1m Oct 2023 - Nov 2026, GC, MGC, MNQ, NQ, ISO 8601 UTC, zstd CSV, split by instrument (includes contract months and possibly calendar spreads; build a volume-rolled continuous series before use).
- `data/databento_tbbo`: TBBO (trades + best bid/ask, aggressor side) Jul 31 - Nov 1 2026, GC and NQ, split by day and instrument. Good for delta/CVD/footprint/absorption/true volume profile; no book depth (would need mbp-10 or mbo).

## Research history
See `docs/` (audit, validation, indicator changes, score/eval/volume-profile research).
