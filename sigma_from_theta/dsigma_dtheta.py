import numpy as np
import matplotlib.pyplot as plt

# Parameters (set your values here)
sigma0 = 1.0
w = 1.0
v = 1.0


# Function for dsigma/dcos(theta)
def dsigma_dcos(theta_val):
    return (sigma0 * w**4) / (2 * (w**2 + v**2 * np.sin(theta_val / 2)**2)**2)

# Theta range (0 to pi)
theta_vals = np.linspace(0, np.pi, 500)

plt.figure(figsize=(8,6))

for x in np.logspace(0, 1, num=10, base=10):
    v = x / 4.5
    dsigma_vals = dsigma_dcos(theta_vals)
    plt.plot(theta_vals, dsigma_vals, label=rf'$\frac{{v}}{{\omega}} = $ {v/w:.1f}')

plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$\frac{d\sigma}{d\cos\theta}$')
plt.title('Differential Cross Section vs Scattering Angle')
plt.legend()
plt.grid(True)
plt.savefig("dsigma_dtheta.png", dpi=160)
plt.show()