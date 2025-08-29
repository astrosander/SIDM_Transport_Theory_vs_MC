import random
import numpy as np
import math
import matplotlib.pyplot as plt

random.seed(1)

v_bath = []
num_bath = 1000_000
velocity_vals = np.linspace(0, 1, 500)
number_encounters = 10_000
m_particle = 1
m_bath = 1

for i in range(num_bath):
	v_bath.append((2*random.random()-1,2*random.random()-1,2*random.random()-1))

relative_magnitudes = []

plt.figure(figsize=(8,6))

for particle_velocity in velocity_vals:
	mean_relative_magnitude = 0
	for i in range(number_encounters):
		vx, vy, vz = v_bath[random.randint(0, num_bath-1)]
		
		mean_relative_magnitude += math.sqrt((particle_velocity-vx)**2+vy**2+vz**2)


	mean_relative_magnitude = mean_relative_magnitude / number_encounters

	relative_magnitudes.append(mean_relative_magnitude)


plt.plot(velocity_vals, relative_magnitudes)

plt.xlabel(r'$v_\text{particle}$')
plt.ylabel(r'$v_\text{relative}$')
plt.title(r'Relative velocity vs particle velocity')
plt.legend()
plt.grid(True)
plt.savefig("relative velocity vs particle velocity.png", dpi=160)
plt.show()