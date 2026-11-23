from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
ROOT_DIR = SCRIPT_DIR.parent
import nq_backtest as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else ROOT_DIR / 'data' / 'mnq_yahoo_5m.csv'
OUT_CSV = SCRIPT_DIR / 'mnq_parameter_sensitivity_yfinance.csv'
OUT_PNG = SCRIPT_DIR / 'mnq_parameter_sensitivity_yfinance.png'
START, END = '2026-10-02', '2026-10-31'
RISK_FRACTION = 0.30
SLIPPAGE_TICKS = 1.0
COMMISSION_SIDE = 2.50

base_cfg = bt.Config(point_value=2.0, session='new_york', rr=3.0, contracts=1,
                     risk_fraction=RISK_FRACTION, slippage_ticks=SLIPPAGE_TICKS,
                     commission_per_side=COMMISSION_SIDE)
raw = bt.load_bars(DATA, base_cfg, START, END)
x = bt.indicators(raw)
ny = bt.in_session(x.index, base_cfg)
vwap = bt.session_vwap(x, base_cfg)

rows = []

def summarize(label, family, params, cfg, signal, atr=None, setup=None, seed=7):
    trades = bt.backtest(x, signal, cfg, atr, setup)
    pnl = trades.pnl.astype(float) if len(trades) else pd.Series(dtype=float)
    equity = pnl.cumsum()
    dd = float((equity.cummax().clip(lower=0) - equity).max()) if len(pnl) else 0.0
    days = (pd.to_datetime(trades.entry_time).dt.date.nunique() if len(trades) else 0)
    mc = bt.monte_carlo(trades, cfg, paths=10000, seed=seed)
    row = {
        'family': family, 'configuration': label, **params,
        'trades': len(trades), 'win_rate': float(pnl.gt(0).mean()) if len(pnl) else 0.0,
        'net_pnl': float(pnl.sum()) if len(pnl) else 0.0,
        'max_closed_drawdown': dd,
        'avg_contracts': float(trades.contracts.mean()) if len(trades) else 0.0,
        'max_contracts': int(trades.contracts.max()) if len(trades) else 0,
        'active_days': days,
        'trades_per_active_day': len(trades) / days if days else 0.0,
        'eval_pass_probability': mc.get('pass_probability', np.nan),
        'null_pass_probability': mc.get('null_pass_probability', np.nan),
        'data_source': DATA.name, 'start_date': START, 'end_date_exclusive': END,
        'session': 'New York (08:30–15:00 America/Chicago)', 'risk_fraction': RISK_FRACTION,
        'rr': cfg.rr, 'slippage_ticks_per_side': SLIPPAGE_TICKS,
        'commission_per_contract_per_side': COMMISSION_SIDE,
        'instrument': 'MNQ (Micro E-mini Nasdaq-100)', 'drawdown_limit': cfg.drawdown_dollars,
        'profit_target': cfg.target_dollars,
    }
    rows.append(row)
    return trades

# Lima: sweep 20-period BB standard-deviation width against symmetric RSI thresholds.
# The baseline is BB 2.0 sigma + RSI 30/70; all runs use 3R as the user's latest comparison.
lima_cfg = bt.Config(**{**base_cfg.__dict__, 'rr': 3.0})
for bb_mult in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
    for rsi_level in [20, 25, 30, 35, 40]:
        sig = bt.make_signals(x, 'mean_reversion_bb_rsi', lima_cfg,
                              bb_mult=bb_mult, rsi_level=rsi_level)
        name = f'Lima BB {bb_mult:g}σ | RSI {rsi_level}/{100-rsi_level}'
        summarize(name, 'Lima', {'bb_period': 20, 'bb_sigma': bb_mult,
                                  'rsi_period': 14, 'rsi_long': rsi_level,
                                  'rsi_short': 100-rsi_level}, lima_cfg, sig,
                  seed=1000 + int(bb_mult * 100) + rsi_level)

# Marcus 15: minimum qualifying gap width in MNQ price points. Keep OB/BOS/CHOCH,
# NY VWAP, ATR stop, 3.5R, session, sizing and costs fixed. Compare FVG vs IFVG vs both.
marcus_cfg = bt.Config(**{**base_cfg.__dict__, 'rr': 3.5})
for min_gap in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    signal, setup, atr_htf, _ = bt.make_fvg_orderblock_signals(x, 15, min_gap_points=min_gap)
    aligned = ((signal > 0) & (x.close > vwap)) | ((signal < 0) & (x.close < vwap))
    eligible = signal.where(ny & aligned, 0)
    eligible_setup = setup.where(ny & aligned, '')
    for kind in ['Combined', 'FVG only', 'IFVG only']:
        if kind == 'FVG only':
            keep = eligible_setup.str.startswith('FVG')
        elif kind == 'IFVG only':
            keep = eligible_setup.str.startswith('IFVG')
        else:
            keep = eligible != 0
        filtered = eligible.where(keep, 0)
        filtered_setup = eligible_setup.where(keep, '')
        label = f'Marcus 15 {kind} | min gap {min_gap:g} pt'
        summarize(label, 'Marcus 15', {'timeframe': '15m', 'gap_type': kind,
                                       'min_gap_points': min_gap,
                                       'min_gap_ticks': min_gap / 0.25,
                                       'ob_bos_choch': True, 'vwap_filter': True},
                  marcus_cfg, filtered, atr_htf, filtered_setup,
                  seed=2000 + int(min_gap * 100) + len(kind))

summary = pd.DataFrame(rows)
summary.to_csv(OUT_CSV, index=False)

# Compact report: Lima heatmap and Marcus gap-width curve/table.
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.15], height_ratios=[1.0, .78],
                      left=.065, right=.985, top=.88, bottom=.15, wspace=.28, hspace=.35)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
fig.suptitle('MNQ parameter sensitivity | Yahoo Finance 5-minute data | Oct 2–30, 2026',
             fontsize=17, weight='bold')

lima = summary[summary.family == 'Lima']
mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
levels = [20, 25, 30, 35, 40]
mat = np.zeros((len(levels), len(mults)))
for iy, lvl in enumerate(levels):
    for ix, mult in enumerate(mults):
        row = lima[(lima.bb_sigma == mult) & (lima.rsi_long == lvl)].iloc[0]
        mat[iy, ix] = row.net_pnl
lim = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1.0)
ax1.imshow(mat, cmap='RdYlGn', vmin=-lim, vmax=lim, aspect='auto')
ax1.set_xticks(range(len(mults)), [f'{m:g}' for m in mults])
ax1.set_yticks(range(len(levels)), [f'{v}/{100-v}' for v in levels])
ax1.set_xlabel('Bollinger width (standard deviations)')
ax1.set_ylabel('RSI long/short threshold')
ax1.set_title('Lima | net P&L by BB width and RSI')
for iy, lvl in enumerate(levels):
    for ix, mult in enumerate(mults):
        row = lima[(lima.bb_sigma == mult) & (lima.rsi_long == lvl)].iloc[0]
        ax1.text(ix, iy, f"${row.net_pnl:,.0f}\n{int(row.trades)} trades\n{row.win_rate:.0%} WR",
                 ha='center', va='center', fontsize=7, color='black')

marcus = summary[summary.family == 'Marcus 15']
colors = {'Combined':'#315a8a', 'FVG only':'#2a9d8f', 'IFVG only':'#e76f51'}
for kind in ['Combined', 'FVG only', 'IFVG only']:
    sub = marcus[marcus.gap_type == kind].sort_values('min_gap_points')
    ax2.plot(sub.min_gap_points, sub.net_pnl, marker='o', linewidth=2,
             label=kind, color=colors[kind])
ax2.axhline(0, color='#444', linewidth=1)
ax2.set_xticks([0, .25, .5, 1, 2, 4, 8], ['0 (any)', '.25', '.5', '1', '2', '4', '8'], rotation=12)
ax2.set_xlabel('Minimum FVG/IFVG width (MNQ points)')
ax2.set_ylabel('Net P&L ($), costs included')
ax2.set_title('Marcus 15 | BOS/CHoCH + OB + NY VWAP | 3.5R')
ax2.grid(alpha=.22)
ax2.legend(loc='best', ncol=3, fontsize=8)

ax3.axis('off')
col_labels = ['Min gap', 'FVG only', 'IFVG only', 'Combined']
table_rows = []
for gap in [0.0, .25, .5, 1.0, 2.0, 4.0, 8.0]:
    rowvals = [f'{gap:g} pt']
    for kind in ['FVG only', 'IFVG only', 'Combined']:
        r = marcus[(marcus.min_gap_points == gap) & (marcus.gap_type == kind)].iloc[0]
        rowvals.append(f"${r.net_pnl:,.0f} | {int(r.trades)}t | {r.win_rate:.0%}")
    table_rows.append(rowvals)
tab = ax3.table(cellText=table_rows, colLabels=col_labels, cellLoc='center', loc='center')
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.35)
for (r, c), cell in tab.get_celld().items():
    if r == 0:
        cell.set_facecolor('#e9eef4'); cell.set_text_props(weight='bold')
    elif c == 0:
        cell.set_facecolor('#f4f5f7')
ax3.set_title('Marcus detail | net P&L | trades | win rate', fontsize=10, pad=4)

fig.text(.065, .035,
         'Yahoo Finance local MNQ CSV only. Oct 2–30, 2026; New York session; 30% risk cap (USD 300 max risk budget per trade), USD 1,000 trailing drawdown, USD 1,250 profit target; 1 tick slippage/side and USD 2.50 commission/contract/side. 1 MNQ point = 4 ticks. Stops prioritized when stop and target touch in one bar. Results are in-sample; gaps are minimum zone width filters.',
         fontsize=8, color='#555', wrap=True)
fig.savefig(OUT_PNG, dpi=170, bbox_inches='tight')
print(f'Loaded {len(x):,} MNQ bars from {x.index.min()} through {x.index.max()}')
print(f'Analysis window: {raw.index.min()} through {raw.index.max()}')
print(f'Saved {len(summary)} configurations to {OUT_CSV}')
print(f'Saved report to {OUT_PNG}')
print('\nTop Lima configurations by net P&L (in-sample):')
print(lima.sort_values('net_pnl', ascending=False)[['configuration','trades','win_rate','net_pnl','max_closed_drawdown','eval_pass_probability']].head(8).to_string(index=False))
print('\nMarcus 15 by gap width and signal type:')
print(marcus[['configuration','trades','win_rate','net_pnl','max_closed_drawdown','eval_pass_probability']].to_string(index=False))
