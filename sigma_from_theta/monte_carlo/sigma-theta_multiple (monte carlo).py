import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

sigma0 = 2.0e-28
w = 1.0e3
theta0 = 0

def sigma(theta, v):
    def dsigma_dcos(theta_val):
        return (sigma0 * w**4) / (2 * (w**2 + v**2 * np.sin(theta_val / 2)**2)**2)

    def integrand(theta_val):
        return -dsigma_dcos(theta_val) * np.sin(theta_val)

    result, _ = quad(integrand, theta0, theta)
    return result

plt.figure(figsize=(8, 5))

for velocity in range(0, 20000000, 2000000):
    sigma_stack = []
    sigma_x = []

    # дискретизация по theta
    for theta_idx in range(0, 5000, 1):
        theta = (theta_idx / 5000) * 360 
        sigma_stack.append(sigma(theta=theta, v=velocity))
        sigma_x.append(theta)

    sigma_x = np.array(sigma_x)
    sigma_stack = np.array(sigma_stack)

    # 100 бинов
    bins = np.linspace(sigma_x.min(), sigma_x.max(), 20)
    bin_indices = np.digitize(sigma_x, bins)

    bin_means = [sigma_stack[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
                 for i in range(1, len(bins))]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # фильтрация NaN, чтобы линия была непрерывной
    mask = ~np.isnan(bin_means)
    plt.plot(bin_centers[mask], np.array(bin_means)[mask], '-', linewidth=1.5, label=f"$v={velocity}$")

plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("sigma_from_theta_with_bins.png", dpi=160)
plt.show()
