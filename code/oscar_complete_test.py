"""Complete Oscar NY ORB matrix: OR size x entry filter x execution bar x RR."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import nq_backtest as bt  # noqa: E402
import session_timeframe_comparison as cmp  # noqa: E402

OUT = ROOT / "results"
RR_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
EXEC_TFS = (5, 10, 15)
ORB_SIZES = (5, 15)
VARIANTS = ("breakout", "two_close_fakeout_filter", "retest")
VARIANT_LABELS = {
    "breakout": "Breakout",
    "two_close_fakeout_filter": "Two-close fakeout filter",
    "retest": "Retest",
}


def main() -> None:
    cfg_base = cmp.localize_session_config("new_york")
    raw = bt.load_bars(cmp.MNQ_PATH, cfg_base)
    start, end = raw.index.min(), raw.index.max()
    rows, all_trades = [], []
    prepared = {tf: bt.indicators(cmp.resample_ohlcv(raw, tf)) for tf in EXEC_TFS}

    for orb_minutes in ORB_SIZES:
        for variant in VARIANTS:
            for tf in EXEC_TFS:
                x = prepared[tf]
                signal, setup = cmp.orb_signal(raw, x, "new_york", tf, orb_minutes, variant)
                for rr in RR_GRID:
                    cfg = bt.Config(**{**cfg_base.__dict__, "rr": rr})
                    trades = bt.backtest(x, signal, cfg, None, setup,
                                         flat_at_session_end=True,
                                         session_exit_tz=cmp.NY_TZ,
                                         session_exit_time="16:00")
                    model = f"Oscar ORB {orb_minutes}m {variant}"
                    stat = cmp.summarize(trades, x, "new_york")
                    row = {
                        "model": model,
                        "opening_range_minutes": orb_minutes,
                        "entry_variant": variant,
                        "execution_bar_minutes": tf,
                        "rr": rr,
                        **stat,
                        "session": "New York 09:30-16:00 ET",
                        "retest_wait_minutes": 30 if variant == "retest" else np.nan,
                        "instrument": "MNQ Micro, one contract",
                        "slippage_ticks_per_side": cfg.slippage_ticks,
                        "commission_per_side": cfg.commission_per_side,
                        "closed_drawdown_limit": cfg.drawdown_dollars,
                        "profit_target": cfg.target_dollars,
                        "data_start": str(start),
                        "data_end": str(end),
                    }
                    rows.append(row)
                    if len(trades):
                        tagged = trades.copy()
                        for k, v in {"model": model, "opening_range_minutes": orb_minutes,
                                     "entry_variant": variant, "execution_bar_minutes": tf,
                                     "rr": rr}.items():
                            tagged[k] = v
                        all_trades.append(tagged)

    summary = pd.DataFrame(rows)
    summary_path = OUT / "oscar_complete_test_summary.csv"
    trades_path = OUT / "oscar_complete_test_trades.csv"
    report_path = OUT / "oscar_complete_test_report.png"
    summary.to_csv(summary_path, index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False) if all_trades else pd.DataFrame().to_csv(trades_path, index=False)
    make_report(summary, report_path, start, end)

    print(f"MNQ local Yahoo data: {start} through {end} ({len(raw):,} 5m source bars).")
    print(f"Completed {len(summary)} configurations: 2 OR sizes x 3 entry rules x 3 execution bars x 5 RR values.")
    print("Entry rules: first close outside range; 2 consecutive closes outside; breakout then retest-and-close back outside.")
    print("Retest window is fixed at 30 elapsed minutes across execution intervals. Entries fill next bar open; NY flat at 16:00 ET.")
    print("Costs: one MNQ Micro, 1 tick slippage per side, $2.50 commission per side; $1,000 closed-trade DD stop.")
    print("Best in-sample configuration by ORB/entry variant:")
    for (orb, variant), sub in summary.groupby(["opening_range_minutes", "entry_variant"], sort=False):
        best = sub.loc[sub.net_pnl.idxmax()]
        print(f"{orb}m {VARIANT_LABELS[variant]}: exec={int(best.execution_bar_minutes)}m RR={best.rr:g}, "
              f"trades={best.trades}, win={best.win_rate:.1%}, net=${best.net_pnl:.2f}, "
              f"closed_DD=${best.max_closed_drawdown:.2f}, Sharpe={best.daily_dollar_sharpe:.2f}")
    # This is the previously reported winner, included as a direct reconciliation check.
    prior = summary[(summary.opening_range_minutes == 15) & (summary.entry_variant == "retest") &
                    (summary.execution_bar_minutes == 5) & (summary.rr == 3.0)]
    if len(prior):
        p = prior.iloc[0]
        print(f"Prior winner row (15m OR, 5m execution, retest, 3R): ${p.net_pnl:.2f} net, "
              f"{int(p.trades)} trades, {p.win_rate:.1%} wins, ${p.max_closed_drawdown:.2f} closed DD.")
    print(f"Saved {summary_path}")
    print(f"Saved {trades_path}")
    print(f"Saved {report_path}")


def make_report(summary: pd.DataFrame, path: Path,
                start: pd.Timestamp, end: pd.Timestamp) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5))
    fig.suptitle("Oscar ORB Complete Test | New York", fontsize=19, fontweight="bold")
    fig.text(.5, .94, f"MNQ 5m source {start.date()}–{end.date()} | one micro | 1 tick/side + $2.50/side | 30-minute retest expiry",
             ha="center", fontsize=10, color="#59636e")
    row_labels = [f"{orb}m {VARIANT_LABELS[variant]}" for orb in ORB_SIZES for variant in VARIANTS]
    for ax, tf in zip(axes, EXEC_TFS):
        part = summary[summary.execution_bar_minutes == tf]
        matrix = np.full((len(row_labels), len(RR_GRID)), np.nan)
        counts = np.zeros_like(matrix)
        for ri, (orb, variant) in enumerate((o, v) for o in ORB_SIZES for v in VARIANTS):
            sub = part[(part.opening_range_minutes == orb) & (part.entry_variant == variant)]
            for ci, rr in enumerate(RR_GRID):
                match = sub[sub.rr == rr]
                if len(match):
                    matrix[ri, ci] = float(match.iloc[0].net_pnl)
                    counts[ri, ci] = int(match.iloc[0].trades)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1000, vmax=1500, aspect="auto")
        ax.set_title(f"{tf}-minute execution")
        ax.set_xticks(range(len(RR_GRID)), [f"{r:g}R" for r in RR_GRID])
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
        ax.set_xlabel("Reward:risk")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"${matrix[i,j]:,.0f}\n{int(counts[i,j])}t",
                            ha="center", va="center", fontsize=7,
                            color="black" if abs(matrix[i,j]) < 800 else "white")
    fig.text(.5, .015, "NY 09:30–16:00 ET | signals on completed bars, fills at next bar open | closed-trade DD only; in-sample grid results are exploratory.",
             ha="center", fontsize=9, color="#59636e")
    fig.subplots_adjust(left=.10, right=.92, bottom=.14, top=.88, wspace=.48)
    cax = fig.add_axes([.94, .20, .014, .62])
    fig.colorbar(im, cax=cax, label="Net P&L ($)")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
