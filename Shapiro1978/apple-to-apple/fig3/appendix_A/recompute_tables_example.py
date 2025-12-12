# recompute_tables_example.py
import numpy as np

from appendixA_bw_tables import compute_tables, save_tables_npz, quick_checks

# -------------------------
# 1) Provide your bound, isotropized DF g0_bw(x) for x>0
#    MUST accept numpy arrays. Replace this with your real function.
# -------------------------
def g0_bw(x):
    x = np.asarray(x, dtype=float)
    # placeholder example ONLY (monotone decaying)
    return np.exp(-x)

# -------------------------
# 2) Choose grids (use your project’s X_TABLE and J_GRID)
# -------------------------
X_TABLE = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])   # example
J_GRID  = np.array([1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0])

# -------------------------
# 3) Set parameters (replace with yours)
# -------------------------
P_star = 1.0     # <-- your P_*
x_D    = 10.0    # <-- your cutoff

# Quadrature orders (increase if you need more accuracy)
n_theta = 48
n_x     = 48

# Optional: explore isotropization prefactor ambiguity by changing g0_scale
g0_scale = 1.0
unbound_scale = 1.0

# -------------------------
# 4) Compute and save
# -------------------------
tables = compute_tables(
    X_TABLE,
    J_GRID,
    g0_bw=g0_bw,
    P_star=P_star,
    x_D=x_D,
    g0_scale=g0_scale,
    unbound_scale=unbound_scale,
    n_theta=n_theta,
    n_x=n_x,
)

print("Quick checks:", quick_checks(tables, X_TABLE, J_GRID))
print(tables)
save_tables_npz("bw_appendixA_tables.npz", tables, X_TABLE, J_GRID)
print("Saved: bw_appendixA_tables.npz")
