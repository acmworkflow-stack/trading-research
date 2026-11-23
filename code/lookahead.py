"""Truncation test: a signal at bar t must not change when future bars are removed."""
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import nq_backtest as bt, strategy_research_suite as s, session_timeframe_comparison as c

NY = "America/New_York"
cfg_ct = bt.Config(session="new_york")
raw_ct = bt.load_bars(s.MNQ_PATH, cfg_ct)                     # suite / grid use CT index
raw_ny = bt.load_bars(s.MNQ_PATH, c.localize_session_config("new_york"))
es_ct = bt.load_bars(s.MES_PATH, cfg_ct)

gens = {
 "lima_bb_rsi":      lambda r: bt.make_signals(bt.indicators(r), "mean_reversion_bb_rsi", cfg_ct),
 "rsi_mr":           lambda r: bt.make_signals(bt.indicators(r), "mean_reversion_rsi", cfg_ct),
 "ema_cloud":        lambda r: bt.make_signals(bt.indicators(r), "ema_cloud", cfg_ct),
 "fvg_matrix_15":    lambda r: bt.make_signals(bt.indicators(r), "fvg", cfg_ct),
 "bos_choch":        lambda r: bt.make_bos_choch_signals(r)[0],
 "fvg_ifvg_15":      lambda r: bt.make_fvg_ifvg_signals(r, 15)[0],
 "fvg_ob_15":        lambda r: bt.make_fvg_orderblock_signals(r, 15)[0],
 "fvg_ob_30":        lambda r: bt.make_fvg_orderblock_signals(r, 30)[0],
 "fvg_struct_15":    lambda r: bt.make_fvg_structure_signals(r, 15)[0],
 "ob15+vwap":        lambda r: (lambda x, sg: sg.where(((sg>0)&(x.close>bt.session_vwap(x,cfg_ct)))|((sg<0)&(x.close<bt.session_vwap(x,cfg_ct))),0))(bt.indicators(r), bt.make_fvg_orderblock_signals(r,15)[0]),
 "november_avwap_2s":lambda r: s.make_avwap_signal(bt.indicators(r), 2.0),
 "oscar_suite_15_retest": lambda r: s.make_orb_signal(bt.indicators(r), 15, "retest")[0],
 "papa_30":          lambda r: s.make_papa_signal(r, 30.0, 1.0)[0],
 "quebec_2":         lambda r: s.make_quebec_signal(r, es_ct.loc[:r.index[-1]], 2.0),
 "romeo":            lambda r: s.make_romeo_proxy(r),
}
gens_ny = {
 "oscar_cmp_15_retest_5m": lambda r: c.orb_signal(r, bt.indicators(r), "new_york", 5, 15, "retest")[0],
 "oscar_cmp_5_fake_10m":   lambda r: (lambda x: c.orb_signal(r, x, "new_york", 10, 5, "two_close_fakeout_filter")[0])(bt.indicators(c.resample_ohlcv(r,10))),
 "november_cmp_10m":       lambda r: (lambda x: (x.attrs.__setitem__("bar_minutes",10), c.avwap_signal(x,"new_york",2.0))[1])(bt.indicators(c.resample_ohlcv(r,10))),
}
rng = np.random.default_rng(0)
def test(name, f, raw, ncuts=25):
    full = f(raw)
    cuts = sorted(rng.integers(3000, len(raw)-10, ncuts))
    bad = 0; first = None
    for cut in cuts:
        r = raw.iloc[:cut]
        part = f(r)
        # compare only bars whose candle is fully complete within the truncated data
        common = part.index[:-1]
        diff = (full.reindex(common).fillna(0).astype(int) != part.reindex(common).fillna(0).astype(int))
        if diff.any():
            bad += 1; first = first or (cut, diff[diff].index[0])
    print(f"{name:26s} {'LEAK' if bad else 'ok  '} ({bad}/{len(cuts)} cuts differ){'  first: '+str(first) if first else ''}")
for k,f in gens.items(): test(k,f,raw_ct)
for k,f in gens_ny.items(): test(k,f,raw_ny)
