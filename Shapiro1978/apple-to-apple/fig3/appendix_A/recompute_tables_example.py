# recompute_tables_example.py
import numpy as np

from appendixA_bw_tables import (
    compute_tables,
    quick_checks,
    make_gbar_from_table2,
    print_table1_like,
)

P_star = 0.005
x_D = 1.0e4

# If you want to compare to the paper's Table 1, use their steady-state gbar(x) from Table 2:
g0_bw = make_gbar_from_table2()

# Table 1 x and j points (from what you pasted)
X = np.array([3.36e-1, 3.31, 3.27e1, 3.23e2, 3.18e3], dtype=float)
J = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=float)

tables = compute_tables(
    X, J,
    g0_bw=g0_bw,
    P_star=P_star,
    x_D=x_D,
    n_theta=64,
    n_x=64,
)

print("Quick checks:", quick_checks(tables, X, J))
print_table1_like(X, J, tables)
