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

def I3_numeric(r):
    f = lambda x: x*x*x / (1.0 + 0.5*r*r*(1.0 - x))**2
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

def C3_numeric(r):
    I0 = I0_numeric(r)
    I3 = I3_numeric(r)
    return I3 / I0

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


def C3_analytic(r):
    A = 1.0 + 0.5*r*r
    B = 0.5*r*r
    
    R = (A - B) / (A + B)
    I3 = 6*A/(B**3)+2*A/(B*(A**2-B**2))+3*A**2/(B**4)*np.log(R)

    return I3/I0_analytic(r)

    # return (A*(3.0*A*A - 2.0*B*B) / (B**3)) + (3.0*A*A*(A*A - B*B) / (2.0*B**4)) * np.log(R)

C0_num = np.array([I0_numeric(r) for r in r_vals])
C0_an  = I0_analytic(r_vals)

C1_num = np.array([C1_numeric(r) for r in r_vals])
C1_an  = C1_analytic(r_vals)

C2_num = np.array([C2_numeric(r) for r in r_vals])
C2_an  = C2_analytic(r_vals)

C3_num = np.array([C3_numeric(r) for r in r_vals])
C3_an  = C3_analytic(r_vals)

plt.plot(r_vals, C0_num, color='red',    linestyle='-',  label='numeric C0')
plt.plot(r_vals, C0_an,  color='green',  linestyle='--', label='analytic C0')

plt.plot(r_vals, C1_num, color='blue',   linestyle='-',  label='numeric C1')
plt.plot(r_vals, C1_an,  color='magenta',linestyle='--', label='analytic C1')

plt.plot(r_vals, C2_num, color='purple',   linestyle='-',  label='numeric C2')
plt.plot(r_vals, C2_an,  color='pink', linestyle='--', label='analytic C2')

plt.plot(r_vals, C3_num, color='black',  linestyle='-',  label='numeric C3')
plt.plot(r_vals, C3_an,  color='orange', linestyle='--', label='analytic C3')

plt.xlabel(r'$v/w = r$')
plt.ylabel(r'$C_1$')
plt.legend()
plt.tight_layout()
plt.show()
