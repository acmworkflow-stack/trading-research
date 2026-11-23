"""Draw the validation charts from the CSVs in this folder.

usage: python report.py      (run inside the validation_2026-11-11 folder)
writes: pass_fail_report.png, breakeven_report.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["text.parse_math"] = False  # dollar signs are money, not math
HERE = Path(__file__).resolve().parent
TESTS = ["initial_pass", "walk_forward_pass", "pbo_pass", "deflated_sharpe_pass", "cost_stress_pass", "nearby_pass"]
LABELS = ["Initial", "Walk-fwd", "PBO", "Defl. Sharpe", "Cost stress", "Nearby"]
PANELS = [("MGC", "new_york"), ("MGC", "asia"), ("MNQ", "new_york"), ("MNQ", "asia")]
NAMES = {"MGC": "Gold (MGC)", "MNQ": "Nasdaq (MNQ)", "new_york": "New York", "asia": "Asia"}

# After a fresh validate.py run, merge its per-market files into the two summary CSVs.
if list(HERE.glob("passfail_*.csv")):
    pf = pd.concat([pd.read_csv(f) for f in sorted(HERE.glob("passfail_*.csv"))])
    pf.insert(3, "tests_passed", pf[TESTS].sum(axis=1).astype(str) + "/6")
    for c in TESTS + ["all_pass"]: pf[c] = pf[c].map({True: "PASS", False: "FAIL"})
    pf.to_csv(HERE / "pass_fail_summary.csv", index=False)
    pd.concat([pd.read_csv(f) for f in sorted(HERE.glob("breakeven_m*_*.csv"))]).to_csv(HERE / "breakeven_1R_vs_3R.csv", index=False)
    pd.concat([pd.read_csv(f) for f in sorted(HERE.glob("grid_*.csv"))]).to_csv(HERE / "all_configs.csv", index=False)
pf = pd.read_csv(HERE / "pass_fail_summary.csv")
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
for ax, (inst, sess) in zip(axes.flat, PANELS):
    d = pf[(pf.instrument == inst) & (pf.session == sess)].copy()
    d["score"] = d[TESTS].eq("PASS").sum(axis=1)
    d = d.sort_values(["score", "net"], ascending=False)
    grid = d[TESTS].eq("PASS").to_numpy(int)
    ax.imshow(grid, cmap="RdYlGn", vmin=-0.3, vmax=1.3, aspect="auto")
    for i, (_, r) in enumerate(d.iterrows()):
        for j in range(len(TESTS)):
            ax.text(j, i, "PASS" if grid[i, j] else "FAIL", ha="center", va="center", fontsize=8, weight="bold")
        ax.text(len(TESTS) - 0.4, i, f"{r.score}/6   ${r.net:,.0f}   {r.best_config}", va="center", fontsize=8)
    ax.set_xticks(range(len(TESTS)), LABELS, fontsize=9)
    ax.set_yticks(range(len(d)), d.family, fontsize=9)
    ax.set_xlim(-0.5, len(TESTS) + 3.2)
    ax.set_title(f"{NAMES[inst]} | {NAMES[sess]}", fontsize=13, weight="bold")
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Strategy validation: pass/fail by test (best settings per strategy)", fontsize=18, weight="bold")
fig.text(.5, .01, "1 contract, 1 tick + $2.50/side, flat at session end, test over at -$1,000 trailing closed-trade DD. "
         "Score, eval net P&L and chosen settings shown right of each row.", ha="center", fontsize=10, color="#555")
fig.tight_layout(rect=[0, .02, 1, .96])
fig.savefig(HERE / "pass_fail_report.png", dpi=150, facecolor="white")
plt.close(fig)

be = pd.read_csv(HERE / "breakeven_1R_vs_3R.csv")
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
for ax, (inst, sess) in zip(axes.flat, PANELS):
    d = be[(be.instrument == inst) & (be.session == sess)]
    d = d.loc[d.groupby("family").base_net.idxmax()].sort_values("base_net")  # best 3R setting per family
    y = np.arange(len(d))
    ax.barh(y + .2, d.base_net, .4, color="#3b6f89", label="3R, fixed stop")
    ax.barh(y - .2, d.be_net, .4, color="#d59a37", label="3R, stop to breakeven at 1R")
    for k, r in enumerate(d.itertuples()):
        x = max(r.base_net, r.be_net, 0) + 40
        ax.text(x, k, f"DD ${r.base_max_dd:,.0f}→${r.be_max_dd:,.0f}  round-trips {r.round_trips}", va="center", fontsize=7.5)
    ax.axvline(0, color="#333", lw=.8)
    ax.set_yticks(y, d.family, fontsize=9)
    ax.set_xlabel("Eval net P&L ($)")
    ax.set_title(f"{NAMES[inst]} | {NAMES[sess]}", fontsize=13, weight="bold")
    ax.margins(x=.25)
    ax.grid(axis="x", alpha=.2)
fig.suptitle("Breakeven at 1R vs fixed stop, 3R target (best 3R setting per strategy)", fontsize=18, weight="bold")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, .955), fontsize=10)
fig.text(.5, .01, "Round-trips = trades that reached +1R and then hit the full stop with the fixed-stop version.",
         ha="center", fontsize=10, color="#555")
fig.tight_layout(rect=[0, .02, 1, .93])
fig.savefig(HERE / "breakeven_report.png", dpi=150, facecolor="white")
plt.close(fig)
print("Saved pass_fail_report.png and breakeven_report.png in", HERE)
