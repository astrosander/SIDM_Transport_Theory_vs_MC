import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

w = 1.0
A1 = np.linspace(0.05, 6.0, 400)  # let A1 = r = v/w
r = A1
v = r * w
Ab = 0.5 * v**2 + w**2
Bb = 0.5 * v**2

# C1_big_form2 = np.log(1.0 - b) - np.log(1.0 + b) + 1.0/(1.0 - b) - 1.0/(1.0 + b)


# R = -np.log(1.0 + 2.0*a)

# C01 = Ab/(Bb**2) * (1/(Ab-Bb)-1/(Ab+Bb))+1/(Bb**2)*np.log((Ab-Bb)/(Ab+Bb)) * (Bb*Bb - Ab*Ab)/2

C01 = 2/(Bb**2-Ab**2)

# C01 = 1/(Bb**2) * np.log((Ab-Bb)/(Ab+Bb)) - 2*Ab/Bb/(Bb**2-Ab**2)
# C01 = C01 * (Bb**2-Ab**2) / 2

# C01 = Ab/Bb+(Ab**2-Bb**2)/(2*Bb**2)*np.log((Ab-Bb)/(Ab+Bb))
#2/(Bb**2-Ab**2)

# C01 = 1/(Bb**2)*(np.log(1.0 - b) - np.log(1.0 + b))-Ab/Bb*2/(Bb**2-Ab**2
# b = 
# C01 = np.log((Ab - Bb)/(Ab + Bb)) + 2*Ab*Bb/(Ab**2-Bb**2)


# R = np.log(np.clip((Ab - Bb) / (Ab + Bb), 1e-300, 1.0))
# C1_big_form1 = 1.0       + (Ab * (Ab*Ab - Bb*Bb) / (Bb**3)) * R

#(Ab*Ab - Bb*Bb)/2 * (C01)
# (Ab / Bb) + ((Ab*Ab - Bb*Bb) / (2.0*Bb*Bb)) * R

def I_numeric(r_val):
    # f = lambda x: 1 / (1.0 + a_val**2*(1.0 - x)/2)**2
    f = lambda x: -1.0 / (1.0 + 0.5 * (r_val**2) * (1.0 - x))**2
    val, _ = quad(f, -1.0, 1.0, limit=200)
    return val

I_vals = np.array([I_numeric(rv) for rv in r])
num_C1_from_I = I_vals

plt.figure()
# plt.plot(A1, C1_big_form2, label=r"$C1_{\mathrm{big}}$ (log/reciprocal in $A_1$)")
plt.plot(A1, C01, '--', label=r"$C1_{\mathrm{big}}$ (A,B,R form)")
plt.plot(A1, num_C1_from_I, ':', label=r"$a^2\,I(a)$ (numerical integration)")
plt.xlabel(r"$A_1$")
plt.ylabel("Value")
plt.title(r"Comparison: $C1_{\mathrm{big}}$ (two forms) vs $a^2 I(a)$")
plt.grid(True)
plt.legend()
plt.show()
