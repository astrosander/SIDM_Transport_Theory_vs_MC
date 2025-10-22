# Reproduce g0(x) from Bahcall & Wolf (1977, BW II) Table 1 (M1 = M2) and plot it vs X.
# Uses ln(1 + E/M) -> X = exp(ln(1+E/M)) - 1; XD ≈ 1e4.
# Smooths with simple log–log interpolation to emulate the solid curve in Shapiro & Marchant (1978) Fig. 3.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- BW II Table 1 (M1 = M2) -----
ln1p_E_over_M = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], dtype=float)

g0_table = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], dtype=float)

# Compute X from ln(1 + E/M)
X_data = np.exp(ln1p_E_over_M) - 1.0

# Fine grid for a smooth line (≈ Fig. 3 axes: 1e-1 to 1e4)
X_min, X_max = 1e-1, 1e4
X_fine = np.logspace(-1, 4, 2000)  # Increased points for smoother curve

# Log–log interpolation (no SciPy needed)
mask = (g0_table > 0) & (X_data > 0)
lx, lg = np.log10(X_data[mask]), np.log10(g0_table[mask])
m = np.diff(lg) / np.diff(lx)

def interp_loglog(xq, lx=lx, lg=lg, m=m):
    lt = np.log10(xq)
    y = np.empty_like(lt)
    left = lt < lx[0]
    right = lt > lx[-1]
    mid = ~(left | right)
    y[left]  = lg[0]  + m[0]   * (lt[left]  - lx[0])
    y[right] = lg[-1] + m[-1]  * (lt[right] - lx[-1])
    idx = np.searchsorted(lx, lt[mid]) - 1
    idx = np.clip(idx, 0, len(m)-1)
    y[mid] = lg[idx] + m[idx] * (lt[mid] - lx[idx])
    return 10**y

g0_smooth = interp_loglog(X_fine)

# Stop interpolation at the last valid data point to avoid solid line over the last point
X_last = X_data[mask][-1]
# Only keep interpolation up to the last data point
valid_mask = X_fine <= X_last
X_fine = X_fine[valid_mask]
g0_smooth = g0_smooth[valid_mask]

# Save data and figure
pd.DataFrame({"X_data": X_data, "g0_table": g0_table}).to_csv("/mnt/data/g0_bwii_table1_points.csv", index=False)
pd.DataFrame({"X": X_fine, "g0_interp": g0_smooth}).to_csv("/mnt/data/g0_bwii_curve.csv", index=False)

plt.figure(figsize=(6,4), dpi=140)
plt.loglog(X_fine, g0_smooth, lw=2, label=r"$g_0$ from BW II (interp.)", color="blue")
plt.scatter(X_data[g0_table>0], g0_table[g0_table>0], s=18, label="BW II Table 1 points", color="blue")
plt.xlim(X_min, X_max)
plt.ylim(1e-1, 1e2)
plt.xlabel(r"$X \equiv (-E/v_0^2)$")
plt.ylabel(r"$g_0(X)$")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("g0_vs_X_bwii.png")
plt.show()