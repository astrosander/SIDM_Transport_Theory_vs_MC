import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt

sigma0 = 2.0e-28
w = 1.0e3
theta0=0

def sigma(theta, v):
    def dsigma_dcos(theta_val):
        return (sigma0 * w**4) / (2 * (w**2 + v**2 * np.sin(theta_val / 2)**2)**2)

    def integrand(theta_val):
        return -dsigma_dcos(theta_val) * np.sin(theta_val)

    result, _ = quad(integrand, theta0, theta)
    return result

pi = 3.1415926

plt.figure(figsize=(8,5))
for velocity in range(1, 20000000, 2000000):
    sigma_stack = []
    sigma_x = []

    theta_step = 5000

    for theta_idx in range(0, 5000, 1):
        theta = (theta_idx / 5000) * 360 
        sigma_stack.append(sigma(theta=theta,v=velocity))
        sigma_x.append(theta)

    plt.plot(sigma_x, sigma_stack, 'o', markersize=0.5, label = f"$v={velocity}$")  

plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("sigma_from_theta.png", dpi=160)
plt.show()
