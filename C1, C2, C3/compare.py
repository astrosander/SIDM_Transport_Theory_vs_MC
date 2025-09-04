import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

r_vals = np.linspace(0.05, 60.0, 400)

def I0_numeric(r):
    f = lambda x: 1.0 / (1.0 + 0.5 * r*r * (1.0 - x))**2
    val, _ = quad(f, -1.0, 1.0, limit=200)
    return val

def I1_numeric(r):
    f = lambda x: x / (1.0 + 0.5 * r*r * (1.0 - x))**2
    val, _ = quad(f, -1.0, 1.0, limit=200)
    return val


def I2_numeric(r):
    f = lambda x: x*x / (1.0 + 0.5*r*r*(1.0 - x))**2
    val, _ = quad(f, -1.0, 1.0, limit=200)
    return val

def C1_numeric(r):
    I0 = I0_numeric(r)
    I1 = I1_numeric(r)
    return I1 / I0

def C2_numeric(r):
    I0 = I0_numeric(r)
    I2 = I2_numeric(r)
    return I2 / I0

def I0_analytic(r):
    A = 1.0 + 0.5*r*r
    B = 0.5*r*r

    return 2 / (A**2-B**2)

def C1_analytic(r):
    A = 1.0 + 0.5*r*r
    B = 0.5*r*r

    R = (A - B) / (A + B)
    return (A/B) + ((A*A - B*B)/(2.0*B*B)) * np.log(R)

def C2_analytic(r):
    A = 1.0 + 0.5*r*r
    B = 0.5*r*r
        
    R = (A - B) / (A + B)
    I2 = (A*A/B**3)*(1/(A-B)-1/(A+B)) + (2*A/B**3)*np.log(R) + 2/B**2
    return I2/I0_analytic(r)

    # return (2/(B**2)+2*A/(B**3)*np.log(R)-2/(B**2-A**2)) / I0_analytic(r)

# C1_num = np.array([I0_numeric(r) for r in r_vals])
# C1_an  = I0_analytic(r_vals)

# C1_num = np.array([C1_numeric(r) for r in r_vals])
# C1_an  = C1_analytic(r_vals)

C1_num = np.array([C2_numeric(r) for r in r_vals])
C1_an  = C2_analytic(r_vals)

plt.plot(r_vals, C1_num, 'r-',  label='numeric C1 = I1/I0')
plt.plot(r_vals, C1_an,  'b--', label='analytic C1')
plt.xlabel(r'$v/w = r$')
plt.ylabel(r'$C_1$')
plt.legend()
plt.tight_layout()
plt.show()
