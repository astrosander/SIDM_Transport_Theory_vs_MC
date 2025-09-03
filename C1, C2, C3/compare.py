# Minimal, single-figure plot comparing three curves vs A1:
# 1) C1_big (log/reciprocal form in A1)
# 2) C1_big (A,B,R form with Ab=A=v^2/2+w^2, Bb=B=v^2/2, R=-ln(1+2a))
# 3) a^2 * I(a) from numerical integration of x/(1 + a(1-x))^2 over x in [-1,1],
#    where a = (1/2) * A1^2
#
# Notes:
# - We set w=1 so A1=v/w => v=A1. Adjust w if desired.
# - If your intended definition of R differs, change how R is computed below.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parameterization
w = 1.0
A1 = np.linspace(0.05, 6.0, 400)  # avoid A1=0 so that a>0 and denominators are safe
a = 0.5 * A1**2                   # a = (1/2) A1^2

# 1) Closed-form in terms of A1 directly
b = a / (1.0 + a)  # b = ( (A1^2)/2 ) / ( 1 + (A1^2)/2 )
C1_big_form2 = np.log(1.0 - b) - np.log(1.0 + b) + 1.0/(1.0 - b) - 1.0/(1.0 + b)

# 2) (A,B,R) form with Ab=A, Bb=B, R chosen to match the log structure
v = A1 * w
Ab = 0.5 * v**2 + w**2
Bb = 0.5 * v**2
R = -np.log(1.0 + 2.0*a)  # inferred from the closed form; update if your R differs

C01 = Ab/(Bb**2) * (1/(Ab-Bb)-1/(Ab+Bb))+1/(Bb**2)*np.log((Ab-Bb)/(Ab+Bb))

R = np.log(np.clip((Ab - Bb) / (Ab + Bb), 1e-300, 1.0))
C1_big_form1 = 1.0       + (Ab * (Ab*Ab - Bb*Bb) / (Bb**3)) * R
#(Ab*Ab - Bb*Bb)/2 * (C01)

# (Ab / Bb) + ((Ab*Ab - Bb*Bb) / (2.0*Bb*Bb)) * R

# 3) Numerical integral: I(a) = ∫_{-1}^{1} x / (1 + a(1 - x))^2 dx
# We compare the same quantity as the two C1_big expressions, which is a^2 * I(a).
def I_numeric(a_val):
    f = lambda x: x**2 / (1.0 + a_val*(1.0 - x))**2
    val, _ = quad(f, -1.0, 1.0, limit=200)
    return val

I_vals = np.array([I_numeric(av) for av in a])
num_C1_from_I = a**2 * I_vals

# --- Single figure with three curves ---
plt.figure()
plt.plot(A1, C1_big_form2, label=r"$C1_{\mathrm{big}}$ (log/reciprocal in $A_1$)")
plt.plot(A1, C1_big_form1, '--', label=r"$C1_{\mathrm{big}}$ (A,B,R form)")
plt.plot(A1, num_C1_from_I, ':', label=r"$a^2\,I(a)$ (numerical integration)")
plt.xlabel(r"$A_1$")
plt.ylabel("Value")
plt.title(r"Comparison: $C1_{\mathrm{big}}$ (two forms) vs $a^2 I(a)$")
plt.grid(True)
plt.legend()
plt.show()
