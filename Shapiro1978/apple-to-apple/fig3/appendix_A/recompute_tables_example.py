import numpy as np
from appendixA_bw_tables import AppendixA, make_gbar_from_table2, print_table1_like

P_star = 0.005
x_D = 1.0e4

# Table 1 convention: unbound_scale=1.0 (∫_{-∞}^0 e^{x'}dx'=1)
unbound_scale = 1.0
j_circular_eps = 1e-10

g0_bw = make_gbar_from_table2(below_behavior="flat", above_behavior="zero")

# Optional: print Table-1-like block (same 5x5 as SM78 Table 1)
print_table1_like(
    g0_bw=g0_bw,
    P_star=P_star,
    x_D=x_D,
    unbound_scale=unbound_scale,
    j_circular_eps=j_circular_eps,
    n_theta=96,
    n_x=96,
)

# Example: Table 1 rows
xs = [0.336, 3.31, 32.7, 323.0, 3.18e3]
js = [1.0, 0.401, 0.161, 0.065, 0.026]

calc = AppendixA(
    P_star=P_star,
    x_D=x_D,
    g0_bw=g0_bw,
    n_theta=96,
    n_x=96,
    unbound_scale=unbound_scale,
    j_circular_eps=j_circular_eps,
)

print("\nDetailed Table 1 comparison:")
print("   x        j        -eps1*       eps2*        j1*         j2*       zeta*2")
for x in xs:
    for j in js:
        if abs(j - 1.0) < 1e-6:
            c = calc.coeffs_star_circular_limit(x)
        else:
            c = calc.coeffs_star(x, j)
        print(f"{x:8.3g}  {j:7.3f}  {-c['eps1_star']: .3e}  {c['eps2_star']: .3e}"
              f"  {c['j1_star']: .3e}  {c['j2_star']: .3e}  {c['zeta_star2']: .3e}")
    print()
