import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

sigma0 = 1.0
w = 1.0
v = 1.0

def sigma(theta, v):
    def dsigma_dcos(theta_val):
        return -(sigma0 * w**4) / (2 * (w**2 + v**2 * np.sin(theta_val / 2)**2)**2)

    def integrand(theta_val):
        return -dsigma_dcos(theta_val) * np.sin(theta_val)

    result, _ = quad(integrand, 0, theta)
    return result

def dsigma_dcos(theta_val):
    return (sigma0 * w**4) / (2 * (w**2 + v**2 * np.sin(theta_val / 2)**2)**2)

theta_vals = np.linspace(0, np.pi, 500)

plt.figure(figsize=(8,6))

for x in np.concatenate(([0], np.logspace(0, 1, num=10, base=10))):
    v = x / 2

    dsigma_vals = np.linspace(0, 0, 500)
    for i in range(500):
        dsigma_vals[i] = sigma(theta_vals[i], v)

    plt.plot(theta_vals, dsigma_vals, label=rf'$\frac{{v}}{{\omega}} = $ {v/w:.1f}')

plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'Normalized $\sigma(\theta)$')
plt.title('Cross Section vs Scattering Angle')
plt.legend()
plt.grid(True)
plt.savefig("sigma_theta.png", dpi=160)
# plt.show()

plt.figure(figsize=(8,6))

for x in np.concatenate(([0], np.logspace(0, 1, num=10, base=10))):
    v = x / 2

    dsigma_vals = np.linspace(0, 0, 500)
    for i in range(500):
        dsigma_vals[i] = sigma(theta_vals[i], v)*v

    plt.plot(theta_vals, dsigma_vals, label=rf'$\frac{{v}}{{\omega}} = $ {v/w:.1f}')

plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$v\cdot\sigma(v,\theta)$')
plt.title(r'Collision probability $v\cdot\sigma(v,\theta)$ vs Scattering Angle')
plt.legend()
plt.grid(True)
plt.savefig("sigma_thetaV.png", dpi=160)
# plt.show()