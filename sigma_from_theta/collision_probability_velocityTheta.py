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

# theta_vals = np.linspace(0, np.pi, 10)
# velocity_vals = np.linspace(0, 1000, 500)

theta_vals = np.concatenate(([0], np.logspace(-3, np.log10(np.pi), 7)))
velocity_vals = np.logspace(0, 3, 500)


plt.figure(figsize=(8,6))


for theta in theta_vals:
    dsigma_vals = np.linspace(0, 0, 500)
    for i in range(len(velocity_vals)):
        dsigma_vals[i] = sigma(theta, velocity_vals[i])*velocity_vals[i]

    plt.loglog(velocity_vals, dsigma_vals, label=rf'$\theta =  {(theta/3.1415926*180):.1f}^\circ$ ')

plt.xlabel(r'$v$')
plt.ylabel(r'$v\cdot\sigma(V,\theta)$')
plt.title(r'$v\cdot\sigma(v,\theta)$ vs $v$')
plt.legend()
plt.grid(True)
plt.savefig("collision_probability_velocityTheta.png", dpi=160)
plt.close()


theta_vals = np.linspace(0, np.pi, 7)
velocity_vals = np.linspace(0, 3, 500)


for theta in theta_vals:
    dsigma_vals = np.linspace(0, 0, 500)
    for i in range(len(velocity_vals)):
        dsigma_vals[i] = sigma(theta, velocity_vals[i])

    plt.plot(velocity_vals, dsigma_vals, label=rf'$\theta =  {int(theta/3.1415926*180)}^\circ$ ')

plt.xlabel(r'$v$')
plt.ylabel(r'Normalized $\sigma(\theta)$')
plt.title(r'$\sigma(v)$')
plt.legend()
plt.grid(True)
plt.savefig("collision_probability_velocity.png", dpi=160)
# plt.show()
