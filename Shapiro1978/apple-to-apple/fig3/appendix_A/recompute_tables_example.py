import numpy as np
from appendixA_bw_tables import compute_tables, quick_checks, make_gbar_from_table2, print_table1_like

P_star = 0.005
x_D = 1.0e4

g0_bw = make_gbar_from_table2(extrapolate="slope", below_behavior="flat")

# Optional: print Table-1-like block
print_table1_like(g0_bw=g0_bw, P_star=P_star, x_D=x_D)

# Compute your full grid tables
X_TABLE = np.logspace(np.log10(0.05), np.log10(1e4), 80)
J_GRID  = np.array([1.0, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02])

tables = compute_tables(X_TABLE, J_GRID, g0_bw=g0_bw, P_star=P_star, x_D=x_D)
print("Quick checks:", quick_checks(tables, X_TABLE, J_GRID))
