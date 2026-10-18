"""Figures for the Part 1 report, built from the saved result CSVs (no GPU).

  python report/make_report_figures.py     # run from part1/
writes report/fig_grid.png and report/fig_mechanism.png (300 dpi)
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
V, T = "outputs/part1_loraplus", "outputs/part1_loraplus_textvqa"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated all-pairs)
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 1: (eta_A, lambda) grids, val accuracy, both datasets -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [4, 3]})
for ax, (path, title) in zip(axes, [(V, "VQAv2 (exact match, val)"), (T, "TextVQA (VQA accuracy, val)")]):
    g = pd.read_csv(f"{path}/grid.csv")
    piv = g.pivot(index="lr_A", columns="ratio", values="val_acc")
    vals = piv.values
    vmax = np.nanmax(vals); vmin = vmax - 10          # colour the top 10 points; collapsed cells clip to lightest
    ax.imshow(vals, cmap=BLUES, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax.set_xlabel(r"$\lambda=\eta_B/\eta_A$"); ax.set_ylabel(r"$\eta_A$"); ax.set_title(title, loc="left")
    mid = (vmin + vmax) / 2
    for (i, j), v in np.ndenumerate(vals):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white" if v > mid else INK)
    i, j = np.unravel_index(np.nanargmax(vals), vals.shape)   # best cell
    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.6))
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
plt.tight_layout(w_pad=2)
plt.savefig(f"{HERE}/fig_grid.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- Figure 2: (a) delta balance per tower, (b) TextVQA tower ablation ---------------------------------------------
d = pd.read_csv(f"{V}/connector_deltas_summary.csv")
methods = [("LoRA +conn", r"LoRA ($\lambda$=1)"), ("LoRA+ global +conn", r"LoRA+ global ($\lambda$=4)"),
           ("LoRA+ width-scaled +conn", r"LoRA+ width-scaled")]
towers = ["text", "vision", "connector"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [3.3, 1.7]})
w = 0.25
for k, (m, label) in enumerate(methods):
    y = [d[(d.method == m) & (d.tower == t)].log10_ratio_A_over_B.item() for t in towers]
    x = np.arange(len(towers)) + (k - 1) * (w + 0.05)
    a.bar(x, y, width=w, color=SERIES[k], label=label, edgecolor="white", linewidth=1)
    for xi, yi in zip(x, y):
        a.text(xi, yi + (0.03 if yi >= 0 else -0.03), f"{yi:+.2f}", ha="center", va="bottom" if yi >= 0 else "top",
               fontsize=6, color=INK2)
a.axhline(0, color=INK2, lw=0.8)
a.set_xticks(range(len(towers)), ["text (576)", "vision (768)", "connector (12288)"])
a.set_ylabel(r"$\log_{10}(\delta_A/\delta_B)$"); a.set_ylim(-0.95, 0.6)
a.set_title(r"(a) Per-step update balance (0 = balanced), VQAv2, 3 seeds", loc="left")
a.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(length=0)

tw = pd.read_csv(f"{T}/tower_table.csv").set_index("method")
rows = [("lambda=1 (neither)", r"$\lambda$=1 (neither)"), ("lambda text only", r"$\lambda$=16 text only"),
        ("lambda vision only", r"$\lambda$=16 vision only"), ("lambda both", r"$\lambda$=16 both")]
yv = [tw.loc[k, "test_acc"] for k, _ in rows]; ye = [tw.loc[k, "test_acc_std"] for k, _ in rows]
b.barh(range(len(rows)), yv, xerr=ye, color=SERIES[0], height=0.6, edgecolor="white",
       error_kw=dict(ecolor=INK2, lw=0.9, capsize=2.5))
for i, (v, e) in enumerate(zip(yv, ye)):
    b.text(v + e + 0.3, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
b.set_yticks(range(len(rows)), [l for _, l in rows]); b.invert_yaxis()
b.set_xlim(35, 48.5); b.set_xlabel("TextVQA test accuracy (%)")
b.set_title(r"(b) $\lambda$ per tower, $\eta_A$=1e-4, 3 seeds", loc="left")
b.xaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True); b.tick_params(length=0)
plt.tight_layout(w_pad=2)
plt.savefig(f"{HERE}/fig_mechanism.png", dpi=300, bbox_inches="tight")
plt.close()
print("wrote", f"{HERE}/fig_grid.png", f"{HERE}/fig_mechanism.png")
