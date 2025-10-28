# Re-run with corrected BW II arrays and full pipeline.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sci(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return None
    if "(" in s and ")" in s:
        base, expo = s.split("(")
        expo = expo.strip(" )")
        return float(base.strip()) * (10.0 ** float(expo))
    return float(s)

# ---- Table 2 rows ----
rows = [
    ("2.25 (-1)", 1.00, 0.04, None, None),
    ("3.03 (-1)", 1.07, 0.06, None, None),
    ("4.95 (-1)", 1.13, 0.04, None, None),
    ("1.04",       1.60, 0.16, None, None),
    ("1.26",       1.34, 0.18, None, None),
    ("1.62",       1.37, 0.14, "1.2 (-1)", "1.2 (-1)"),
    ("2.35",       1.55, 0.15, "9.3 (-2)", "7.4 (-2)"),
    ("5.00",       2.11, 0.34, "1.87 (-1)", "0.32 (-1)"),
    ("7.20",       2.22, 0.44, "1.24 (-1)", "0.38 (-1)"),
    ("8.94",       2.20, 0.35, "1.43 (-1)", "0.27 (-1)"),
    ("1.21 (1)",   2.41, 0.22, "8.84 (-2)", "1.42 (-2)"),
    ("1.97 (1)",   3.00, 0.17, "4.53 (-2)", "0.50 (-2)"),
    ("4.16 (1)",   3.50, 0.43, "7.92 (-3)", "1.89 (-3)"),
    ("5.03 (1)",   3.79, 0.30, "6.43 (-3)", "0.82 (-3)"),
    ("6.46 (1)",   3.61, 0.30, "4.52 (-3)", "0.92 (-3)"),
    ("9.36 (1)",   3.66, 0.18, "2.05 (-3)", "0.35 (-3)"),
    ("1.98 (2)",   4.03, 0.44, "4.62 (-4)", "0.22 (-4)"),
    ("2.87 (2)",   3.98, 0.99, "1.65 (-4)", "0.33 (-4)"),
    ("3.56 (2)",   3.31, 0.68, "9.38 (-5)", "2.08 (-5)"),
    ("4.80 (2)",   2.92, 0.53, "5.14 (-5)", "0.38 (-5)"),
    ("7.84 (2)",   2.35, 0.18, "1.55 (-5)", "0.19 (-5)"),
    ("1.65 (3)",   1.57, 0.54, "1.48 (-5)", "1.08 (-5)"),
    ("2.00 (3)",   0.85, 0.12, "7.18 (-7)", "1.81 (-7)"),
    ("2.57 (3)",   0.74, 1.05, "2.74 (-7)", "1.05 (-7)"),
    ("3.73 (3)",   0.20, 0.14, "5.69 (-8)", "1.07 (-8)"),
]
x_vals   = np.array([sci(r[0]) for r in rows], dtype=float)
gbar     = np.array([r[1] for r in rows], dtype=float)
gbar_err = np.array([r[2] for r in rows], dtype=float)
Fstar    = np.array([sci(r[3]) if r[3] is not None else np.nan for r in rows], dtype=float)
Fstar_err= np.array([sci(r[4]) if r[4] is not None else np.nan for r in rows], dtype=float)
FE_x     = Fstar * x_vals
FE_x_err = Fstar_err * x_vals

# ---- BW II Table 1 -> g0(X) ----
ln1p = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], dtype=float)

g0_tab = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], dtype=float)

X_g0 = np.exp(ln1p) - 1.0
mask = (g0_tab > 0) & (X_g0 > 0)
lx, lg = np.log10(X_g0[mask]), np.log10(g0_tab[mask])
slope = np.diff(lg)/np.diff(lx)

def interp_loglog(xq):
    xq = np.asarray(xq)
    lt = np.log10(xq)
    y = np.empty_like(lt)
    left = lt < lx[0]
    right = lt > lx[-1]
    mid = ~(left | right)
    y[left]  = lg[0]  + slope[0]   * (lt[left]  - lx[0])
    y[right] = lg[-1] + slope[-1]  * (lt[right] - lx[-1])
    idx = np.searchsorted(lx, lt[mid]) - 1
    idx = np.clip(idx, 0, len(slope)-1)
    y[mid] = lg[idx] + slope[idx] * (lt[mid] - lx[idx])
    return 10**y

X_line = np.logspace(-1, 4, 600)
g0_line = interp_loglog(X_line)
X_last = X_g0[mask][-1]
tail = (X_line >= X_last) & (X_line <= 1e4)
if tail.any():
    g_start = g0_line[tail][0]
    g0_line[tail] = np.maximum(g_start * (1 - (X_line[tail] - X_last)/(1e4 - X_last)), 1e-16)

# ---- Save CSVs ----
pd.DataFrame({"x": x_vals, "gbar": gbar, "gbar_err": gbar_err}).to_csv("/mnt/data/figure3_gbar.csv", index=False)
pd.DataFrame({"x": x_vals, "Fstar": Fstar, "Fstar_err": Fstar_err, "FE_x": FE_x, "FE_x_err": FE_x_err}).to_csv("/mnt/data/figure3_FEx.csv", index=False)
pd.DataFrame({"x": X_line, "g0": g0_line}).to_csv("/mnt/data/figure3_g0_curve.csv", index=False)

# ---- Plot ----
plt.figure(figsize=(6,5), dpi=140)
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-1, 1e4)
plt.ylim(1e-3, 1e2)

plt.plot(X_line, g0_line, linewidth=2, label=r"$g_0$ (BW II)")

mask_g = ~np.isnan(gbar)
plt.errorbar(x_vals[mask_g], gbar[mask_g], yerr=gbar_err[mask_g],
             fmt="o", markersize=4, capsize=2, label=r"$\bar{g}$")

mask_f = ~np.isnan(FE_x)
rel_err = np.zeros_like(FE_x)
rel_err[mask_f] = (FE_x_err[mask_f] / np.maximum(FE_x[mask_f], 1e-99))
large = mask_f & (rel_err >= 0.6)
small = mask_f & ~large

plt.errorbar(x_vals[small], FE_x[small], yerr=FE_x_err[small],
             fmt="s", markersize=4, capsize=2, label=r"$F_E^*\cdot X$")
plt.errorbar(x_vals[large], FE_x[large], yerr=FE_x_err[large],
             fmt="s", mfc="none", markersize=5, capsize=2, label="large-error points")

plt.annotate(r"$x_{\rm crit}=10$", xy=(10, 1e-1), xytext=(12, 2e-1),
             arrowprops=dict(arrowstyle="->", lw=1))

plt.xlabel(r"$X \equiv (-E/v_0^2)$")
plt.ylabel(r"$\bar g(E),\  F_E^*\cdot X$")
plt.title("Reproduction of Fig. 3 (canonical case: $x_{\\rm crit}=10,\\ x_D=10^4$)")
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig("figure3_repro.png")
