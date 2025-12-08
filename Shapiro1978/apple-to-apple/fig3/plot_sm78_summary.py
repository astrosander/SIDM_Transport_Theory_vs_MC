#!/usr/bin/env python3
"""
Plot SM78 MC summary with ratio diagnostics.

Reads tables of:
x_center gbar_MC_norm gbar_MC_raw gbar_paper gbar_err_paper

Generates:
1) baseline overlay + ratio
2) loss-cone sweep overlay + ratio
3) e1-scale sweep overlay + ratio
4) clones compare overlay + ratio
5) flux vs snapshot overlay + ratio
6) noloss occupancy (uses MC_raw as N(E) from debug-occupancy-norm)

No argparse.
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt




import matplotlib as mpl


mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.labelsize": 18,           # axis label text
    "xtick.labelsize": 16,          # x-tick labels
    "ytick.labelsize": 16,          # y-tick labels
})

LOGDIR = Path("logs")
OUTDIR = Path("figs")

FILES = {
    "baseline": [
        LOGDIR / "gbar_sm78_baseline_lc1.out",
        LOGDIR / "gbar_sm78_lc1_with_clones.out",
        LOGDIR / "gbar_sm78_lc1.out",
        LOGDIR / "gbar_sm78_snapshot_lc1.out",
    ],
    "lc_sweep": {
        "LC=0.5": LOGDIR / "gbar_sm78_lc0p5.out",
        "LC=1.0": LOGDIR / "gbar_sm78_lc1.out",
        "LC=2.0": LOGDIR / "gbar_sm78_lc2.out",
    },
    "e1_sweep": {
        "e1=0.8": LOGDIR / "gbar_sm78_e1scale0p8_lc1.out",
        "e1=1.0": LOGDIR / "gbar_sm78_e1scale1p0_lc1.out",
        "e1=1.2": LOGDIR / "gbar_sm78_e1scale1p2_lc1.out",
    },
    "clones": {
        "with clones": LOGDIR / "gbar_sm78_lc1_with_clones.out",
        "parent only": LOGDIR / "gbar_sm78_lc1_parent_only.out",
    },
    "flux": {
        "snapshot-based": LOGDIR / "gbar_sm78_snapshot_lc1.out",
        "flux-based": LOGDIR / "gbar_sm78_flux_lc1.out",
        "flux-based (alt)": LOGDIR / "gbar_sm78_lc1_flux_based.out",
    },
    "bw_noloss": [
        LOGDIR / "bw_noloss_hiacc.out",#"bw_noloss.out",
        LOGDIR / "bw_noloss_e1scale1p2.out",
    ],
}


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def parse_gbar_table(path: Path):
    lines = path.read_text().splitlines()
    data = []
    header_found = False

    for line in lines:
        if re.match(r"^\s*x_center\s+gbar_MC_norm", line):
            header_found = True
            continue

        if header_found:
            if not line.strip():
                break
            if line.lstrip().startswith("chi^2"):
                break

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                x = float(parts[0])
                mc_norm = float(parts[1])
                mc_raw = float(parts[2])
                paper = float(parts[3])
                err = float(parts[4])
            except ValueError:
                continue

            data.append((x, mc_norm, mc_raw, paper, err))

    if not data:
        return None

    arr = np.array(data, dtype=float)
    return {
        "x": arr[:, 0],
        "mc_norm": arr[:, 1],
        "mc_raw": arr[:, 2],
        "paper": arr[:, 3],
        "paper_err": arr[:, 4],
        "path": str(path),
    }


def ratio_safe(num, den):
    out = np.full_like(num, np.nan, dtype=float)
    mask = (den != 0) & np.isfinite(num) & np.isfinite(den)
    out[mask] = num[mask] / den[mask]
    return out


def plot_overlay_and_ratio(d_list, labels, title, fname):
    """
    d_list: list of parsed dicts
    labels: list of labels (same length)
    """
    if not d_list:
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Use first dataset for paper points
    d0 = d_list[0]
    x = d0["x"]
    paper = d0["paper"]
    perr = d0["paper_err"]

    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.05)

    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    # Paper
    ax.errorbar(x, paper, yerr=perr, fmt="o", label="SM78 paper")

    # Curves
    for lab, d in zip(labels, d_list):
        ax.plot(d["x"], d["mc_norm"], marker="o", linestyle="-", label=lab)

        r = ratio_safe(d["mc_norm"], d["paper"])
        axr.plot(d["x"], r, marker="o", linestyle="-")

    ax.set_xscale("log")
    ax.set_ylabel("ḡ(x) (normalized)")
    ax.set_title(title)
    ax.legend()

    axr.set_xscale("log")
    axr.set_xlabel("x")
    axr.set_ylabel("MC/paper")

    # A reference line at 1
    axr.axhline(1.0, linestyle="--")

    # Make ratio y-limits a bit robust
    all_r = []
    for d in d_list:
        rr = ratio_safe(d["mc_norm"], d["paper"])
        all_r.append(rr[np.isfinite(rr)])
    if all_r:
        vals = np.concatenate(all_r)
        if vals.size:
            lo = np.nanpercentile(vals, 5)
            hi = np.nanpercentile(vals, 95)
            if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
                axr.set_ylim(max(0.1, 0.5 * lo), 1.5 * hi)

    plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=200)
    plt.close(fig)

    print(f"Saved {fname}")


def plot_baseline():
    path = first_existing(FILES["baseline"])
    if not path:
        print("No baseline LC=1 file found.")
        return
    d = parse_gbar_table(path)
    if not d:
        print("Could not parse baseline.")
        return
    print("Выфц", path)
    plot_overlay_and_ratio([d], ["MC (LC=1)"], "Baseline ḡ(x): MC vs SM78", "fig1_baseline_overlay_ratio.png")
    print(f"  source: {path}")


def plot_lc_sweep():
    ds, labs = [], []
    for lab, path in FILES["lc_sweep"].items():
        if path.exists():
            d = parse_gbar_table(path)
            if d:
                ds.append(d); labs.append(lab)

    if not ds:
        print("No LC sweep files found.")
        return

    plot_overlay_and_ratio(ds, labs, "Loss-cone strength sweep", "fig2_lc_sweep_overlay_ratio.png")


def plot_e1_sweep():
    ds, labs = [], []
    for lab, path in FILES["e1_sweep"].items():
        if path.exists():
            d = parse_gbar_table(path)
            if d:
                ds.append(d); labs.append(lab)

    if not ds:
        print("No e1 sweep files found.")
        return

    plot_overlay_and_ratio(ds, labs, "e1-scale sweep (LC=1)", "fig3_e1_sweep_overlay_ratio.png")


def plot_clones():
    ds, labs = [], []
    for lab, path in FILES["clones"].items():
        if path.exists():
            d = parse_gbar_table(path)
            if d:
                ds.append(d); labs.append(lab)

    if not ds:
        print("No clones compare files found.")
        return

    plot_overlay_and_ratio(ds, labs, "Clones on vs parent-only (LC=1)", "fig4_clones_overlay_ratio.png")


def plot_flux():
    snap = FILES["flux"]["snapshot-based"]
    flux = FILES["flux"]["flux-based"]
    alt = FILES["flux"]["flux-based (alt)"]

    ds, labs = [], []
    if snap.exists():
        d = parse_gbar_table(snap)
        if d:
            ds.append(d); labs.append("snapshot-based")

    if flux.exists():
        d = parse_gbar_table(flux)
        if d:
            ds.append(d); labs.append("flux-based")
    elif alt.exists():
        d = parse_gbar_table(alt)
        if d:
            ds.append(d); labs.append("flux-based")

    if not ds:
        print("No flux/snapshot files found.")
        return

    plot_overlay_and_ratio(ds, labs, "Snapshot-based vs flux-based ḡ(x)", "fig5_flux_overlay_ratio.png")


def plot_bw_noloss():
    path = first_existing(FILES["bw_noloss"])
    if not path:
        print("No noloss file found.")
        return

    d = parse_gbar_table(path)
    if not d:
        print("Could not parse noloss table.")
        return

    print("lol", path)
    x = d["x"]
    n_raw = d["mc_raw"]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x, n_raw,marker="o", linestyle="-", color="black", label="MC occupancy $N(E)$")

    # Reference x^-2 line normalized at first x>=1 point
    mask = (x >= 1.0) & (n_raw > 0) &(x <= 50.0) 
    if mask.any():
        i0 = np.where(mask)[0][0]
        x0, y0 = x[i0], n_raw[i0]
        xline = np.array([x[mask].min(), x[mask].max()])
        yline = y0 * (xline / x0) ** (-2.25)*3
        plt.plot(xline, yline, linestyle="-", color="red", label="$x^{-9/4}$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$x$")
    plt.ylabel("$N(E)$")
    plt.title("No-loss-cone")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig6_noloss_occupancy.png", dpi=300)
    plt.close()

    print("Saved fig6_noloss_occupancy.png")
    print(f"  source: {path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    plot_baseline()
    plot_lc_sweep()
    plot_e1_sweep()
    plot_clones()
    plot_flux()
    plot_bw_noloss()

    print("\nDone. New figures in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
