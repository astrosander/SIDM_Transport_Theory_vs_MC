import math, random, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

def jmax(E, M):
    """
    Maximum of angular momentum corresponds to a star in circular orbit
    """
    
    return M / math.sqrt(-2 * E) # between (3) and (4)

def jmin(E, M, rD):
    """
    Minimum angular momentum for which a stellar orbit of energy E remains entirely outside the disruption radius
    """

    return math.sqrt(2 * (-E + M/rD)) * rD # (4)

def period(E, M):
    """
    Orbital period (cf. LS eq. [30])
    """

    return 2 * math.pi * M / ((-2 * E)**1.5) # P(E) between (9) and (10)

def coeffs(E, J, M, rD):
    raise NotImplementedError

class MonteCarloStepper:
    def __init__(self, M, rD, coeffs_fn=None):
        self.M = M
        self.rD = rD
        self.coeffs_fn = coeffs_fn if coeffs_fn else coeffs
        self.orb_sum = 0.0
    def choose_n(self, E, J, e1, e2, j1, j2, Jmax, Jmin):# Section III.c. Step-Adjustment Algorithm
        a = (0.15 * abs(E) / max(e2, 1e-30))**2         # (29a)
        b = (0.10 * Jmax / max(j2, 1e-30))**2           # (29b)
        c = max(0.40 * (1.0075 * Jmax - J), 0.0)        # (29c)
        c = (c / max(j2, 1e-30))**2 if c > 0 else float("inf") # (29c)
        d = max(abs(0.25 * (J - Jmin)), 0.10 * Jmin)    # (29d)
        d = (d / max(j2, 1e-30))**2                     # (29d)
        n = max(min(a, b, c, d), 1e-12)                 # find as (29.a-d) minimum
        return n
    def step(self, E, J, t):
        Jmax = jmax(E, self.M)
        Jmin = jmin(E, self.M, self.rD)
        e1, e2, j1, j2, l2 = self.coeffs_fn(E, J, self.M, self.rD)
        n = self.choose_n(E, J, e1, e2, j1, j2, Jmax, Jmin)     # Section III.c. Step-Adjustment Algorithm
        p = period(E, self.M)       # orbital period
        rho = max(-1.0, min(1.0, l2 / max(e2 * j2, 1e-30)))     # Cross-correlation coefficient 
        
        """
        We then pick two random numbers y1 and y2 chosen from distributions characterized by means ⟨y1⟩ = ⟨y2⟩ = 0, dispersions ⟨y1²⟩ = ⟨y2²⟩ = 1, and a cross-correlation ⟨y1 y2⟩ = ξ²/ε₂j₂. Appendix B describes how we construct such a correlation.
        """

        y1 = random.gauss(0, 1)
        z = random.gauss(0, 1)
        y2 = rho * y1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
        dE = n * e1 + math.sqrt(n) * e2 * y1    # (27a)
        use_iso = math.sqrt(n) * j2 > J/4.0 and J <= Jmin and J < 0.4 * Jmax    # Switch to isotropic 2D J-walk (eq. 28) 
        if use_iso:
            """
            implying that the angular momentum step is large compared to J we replace equation (27b) by the expression (28):
            """

            y3 = random.gauss(0, 1)
            y4 = random.gauss(0, 1)
            dJ = math.hypot(J + math.sqrt(n) * y3 * j2, math.sqrt(n) * y4 * j2) - J # (28)
        else:
            dJ = n * j1 + math.sqrt(n) * j2 * y2    # (27b)
        t += n * p # (27c): time increment for the given star 
        self.orb_sum += n
        E += dE
        J = max(0.0, min(J + dJ, jmax(E, self.M)))
        hit = False
        consumed = False
        if self.orb_sum >= 1.0 - 1e-12:
            """
            2.c. Boundary conditions. iii
            """

            k = math.floor(self.orb_sum + 1e-12)
            self.orb_sum -= k
            hit = True                 # “At pericenter” whenever total n sums to an integer number of orbits
        if hit and J <= jmin(E, self.M, self.rD):
            """
            Stars are consumed by the black hole if and only if they lie within the loss cone, when they are at pericenter.
            """
            
            consumed = True
        return E, J, t, consumed, n

def coeffs_simple(E, J, M, rD):
    Jm = jmax(E, M)
    u = max(1e-6, -E)
    e1 = 0.0
    j1 = 0.0
    e2 = 0.02 * u
    j2 = 0.02 * Jm * math.sqrt(max(1e-6, 1 - (J/Jm)**2))
    l2 = 0.2 * e2 * j2
    return e1, e2, j1, j2, l2

def main():
    random.seed(42)
    M = 1.0
    rD = 0.1
    E = -0.2
    J = 0.8
    t = 0.0
    stepper = MonteCarloStepper(M, rD, coeffs_simple)

    T = [t]
    Es = [E]
    Js = [J]
    Ns = []
    Jmins = [jmin(E, M, rD)]
    Jmaxs = [jmax(E, M)]
    
    for _ in range(4000):
        E, J, t, consumed, n = stepper.step(E, J, t)
        T.append(t)
        Es.append(E)
        Js.append(J)
        Ns.append(n)
        Jmins.append(jmin(E, M, rD))
        Jmaxs.append(jmax(E, M))
        if consumed:
            break

    plt.figure()
    plt.plot(T, Es, color="blue")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E$")
    plt.title(r"Energy vs Time")
    plt.tight_layout()
    plt.savefig("fig/energy_vs_time.png", dpi=150)

    plt.figure()
    plt.plot(T, Js, color="green", label=r"$J(t)$")
    plt.plot(T, Jmins, linestyle="--", color="red", label=r"$J_{\min}(t)$")
    plt.plot(T, Jmaxs, linestyle=":", color="blue", label=r"$J_{\max}(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$J$")
    plt.title(r"Angular Momentum vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/J_vs_time.png", dpi=150)
    
    plt.figure()
    plt.scatter(Es, Js, s=15, color='blue', edgecolors='none')
    plt.xlabel(r"$E$")
    plt.ylabel(r"$J$")
    plt.title(r"Phase Space: $J$ vs $E$")
    plt.tight_layout()
    plt.savefig("fig/phase_J_vs_E.png", dpi=150)
    
    plt.figure()
    plt.hist(Ns, bins=40, color="darkblue")
    plt.xlabel(r"$n$")
    plt.ylabel("Count")
    plt.title(r"Adaptive Step Sizes $n$")
    plt.tight_layout()
    plt.savefig("fig/n_hist.png", dpi=150)

if __name__ == "__main__":
    main()