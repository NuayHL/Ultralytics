"""
Pareto figure for the USAA soft-label fairness study (YOLO12-s, AI-TODv2).
Two panels share the x-axis AP_vt (very-tiny AP, higher = better).
  (a) y = ECE  (lower = better)  -> "is it well calibrated?"
  (b) y = MCE  (lower = better)  -> "is the worst bin trustworthy?"
The Pareto frontier (max AP_vt, min error) is drawn; USAA sits ON it.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,   # editable / embeddable fonts for the paper
    "ps.fonttype": 42,
})

# --- data: soft-label family + ours (+ TAL baseline as reference start) ---
# name: (AP_vt, ECE, MCE, group)
data = {
    "TAL (baseline)": (8.9,  5.25, 27.90, "ref"),
    "NWD-soft":       (7.5,  0.79,  5.97, "blunt"),
    "KLD-soft":       (10.6, 1.02,  4.50, "blunt"),
    "DotD-soft":      (9.5,  1.42,  7.42, "blunt"),
    "SimD-soft":      (9.6,  9.19, 47.27, "unstable"),
    "WD-soft":        (7.6, 12.79, 51.78, "unstable"),
    "RFLA(WD)-soft":  (12.3,22.82, 61.44, "unstable"),
    "RFLA(KLD)-soft": (14.7, 3.31, 21.10, "rfla"),
    "USAA (ours)":    (13.4, 2.40, 10.89, "ours"),
}

colors = {
    "ref":      "#9aa0a6",
    "blunt":    "#1f77b4",   # good calibration, blunt (low AP_vt)
    "unstable": "#c23b22",   # calibration blows up
    "rfla":     "#e08214",   # the only real AP_vt competitor
    "ours":     "#b2182b",   # USAA
}
labels_group = {
    "blunt":    "smooth soft label (blunt)",
    "unstable": "native metric soft label (unstable)",
    "rfla":     "RFLA / HieAssign (sharp, mis-calibrated)",
    "ours":     "USAA (ours)",
    "ref":      "hard-label baseline",
}

def pareto_front(points):
    """points: list of (ap_vt, err). Return non-dominated set (max ap_vt, min err), sorted by ap_vt."""
    front = []
    for p in points:
        dominated = any((q[0] >= p[0] and q[1] <= p[1] and q != p) for q in points)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda t: t[0])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
metrics = [("ECE", 1, "(a)  sharpness vs. calibration error"),
           ("MCE", 2, "(b)  sharpness vs. worst-bin error")]

for ax, (mname, idx, subtitle) in zip(axes, metrics):
    pts = [(v[0], v[idx]) for v in data.values()]
    front = pareto_front(pts)

    # shade the dominated region (up-left of the frontier) very lightly
    ax.plot([f[0] for f in front], [f[1] for f in front],
            "--", color="#444444", lw=1.1, zorder=1, alpha=0.7)

    seen = set()
    for name, v in data.items():
        x, y, g = v[0], v[idx], v[3]
        lbl = labels_group[g] if g not in seen else None
        seen.add(g)
        marker = "*" if g == "ours" else ("D" if g == "rfla" else ("s" if g == "ref" else "o"))
        size = 320 if g == "ours" else (120 if g == "rfla" else 90)
        edge = "black" if g in ("ours", "rfla") else "none"
        ax.scatter(x, y, s=size, c=colors[g], marker=marker,
                   edgecolors=edge, linewidths=1.0, zorder=3, label=lbl)

    # annotate
    _s = 1.0 if mname == "ECE" else 2.2   # vertical scale for the two panels
    off = {
        "USAA (ours)":    (0.25, 0.55*_s),
        "RFLA(KLD)-soft": (0.25, 0.55*_s),
        "KLD-soft":       (0.15, -1.5*_s),
        "DotD-soft":      (0.25, 0.5*_s),
        "NWD-soft":       (-2.75, 0.4*_s),
        "SimD-soft":      (0.25, 0.0),
        "WD-soft":        (0.25, 0.0),
        "RFLA(WD)-soft":  (-2.9, -1.9*_s),
        "TAL (baseline)": (-2.5, 0.5*_s),
    }
    for name, v in data.items():
        dx, dy = off.get(name, (0.2, 0.4))
        fw = "bold" if name == "USAA (ours)" else "normal"
        ax.annotate(name, (v[0], v[idx]), (v[0]+dx, v[idx]+dy),
                    fontsize=8.4, fontweight=fw)

    ax.set_xlabel(r"$\mathrm{AP}_{vt}$  (very-tiny, higher $\rightarrow$ better)")
    ax.set_ylabel(f"{mname}  (lower $\\rightarrow$ better)")
    ax.set_title(subtitle, fontsize=10.5, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax.set_xlim(6.3, 16.0)
    # arrow marking the "good" corner (bottom-right)
    y0, y1 = ax.get_ylim()
    ax.annotate("better", xy=(15.7, y0 + (y1-y0)*0.02),
                xytext=(14.4, y0 + (y1-y0)*0.14),
                fontsize=8, color="#2a7f2a",
                arrowprops=dict(arrowstyle="->", color="#2a7f2a", lw=1.2))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           fontsize=8.2, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
name = "./pareto_softlabel"
fig.savefig(f"{name}.pdf", bbox_inches="tight")
fig.savefig(f"{name}.png", dpi=170, bbox_inches="tight")
print("saved")
print("frontier (ECE):", pareto_front([(v[0], v[1]) for v in data.values()]))
print("frontier (MCE):", pareto_front([(v[0], v[2]) for v in data.values()]))