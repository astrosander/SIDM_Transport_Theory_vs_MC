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

# -------- SIDM Transport Kernel + Moment Projection ----------
# Maxwellian bath (isotropic, dispersion s_bath)
def maxwell_3d_pdf(u, s):
    c = (2.0*np.pi*s**2)**(-1.5)
    return c*np.exp(-(u*u)/(2.0*s*s))

# Example momentum-transfer cross sections
def sigma_T_const(s, sigma0, s0=None):
    return sigma0

def sigma_T_yukawa(s, sigma0, s0):
    return sigma0/(1.0+(s/s0)**4)

# Gauss-Legendre in mu and simple radial grid in u
def _quad_mu(nmu=64):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return x.astype(float), w.astype(float)

def _radial_grid(s_bath, umax_factor=8.0, Nu=120):
    umax = umax_factor*s_bath
    u = np.linspace(0.0, umax, Nu)
    du = u[1]-u[0] if Nu>1 else 1.0
    return u, du

# Scalar SIDM coefficients for an individual speed v=|v|
def sidm_scalars(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), nmu=64, Nu=120):
    if v <= 0.0:
        return 0.0, 0.0, 0.0
    mu, wmu = _quad_mu(nmu)
    u, du = _radial_grid(s_bath, Nu=Nu)

    alpha = m_a/(m_t+m_a)          # per-collision velocity change prefactor
    a2 = alpha**2

    # weights for bath average: 2π ∫ u^2 du ∫ dμ  [ ... ] maxwell(u)
    shell = 2.0*np.pi*u*u*maxwell_3d_pdf(u, s_bath)*du  # shape (Nu,)

    # broadcast to (Nu, nmu)
    U = u[:,None]
    MU = mu[None,:]
    V = float(v)

    s = np.sqrt(np.maximum(V*V + U*U - 2.0*V*U*MU, 0.0))  # relative speed |v-u|
    rate = n_a * sigmaT(s, *sigma_args) * s               # collisions per time

    # components along v-hat
    s_par = V - U*MU
    s2 = s*s

    # Drift D[Δv_parallel] = alpha ∫ rate * (u·vhat - v) = -alpha ∫ rate * s_par
    D_par = alpha*np.sum(shell[:,None]*wmu[None,:]*rate*(U*MU - V))

    # Diffusion scalars from tensor formula B = a2 ∫ rate [ 1/2 q q^T + (s^2/12) I ]
    # vhat^T B vhat = a2 ∫ rate [ 1/2 s_par^2 + s^2/12 ]
    D_par2 = a2*np.sum(shell[:,None]*wmu[None,:]*rate*(0.5*s_par*s_par + (s2/12.0)))

    # Sum over the two perpendicular components:
    # Tr(B) = a2 ∫ rate [ 1/2 s^2 + (s^2/12)*3 ] = a2 ∫ rate * (3/4) s^2
    trB = a2*np.sum(shell[:,None]*wmu[None,:]*rate*(0.75*s2))
    D_perp2 = trB - D_par2  # this is the sum over the two ⊥ components

    return float(D_par), float(D_par2), float(D_perp2)

# Vector/tensor FP coefficients A(v), B(v) from scalars
def sidm_A_B(v_vec, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), **kw):
    v = float(np.linalg.norm(v_vec))
    if v == 0.0:
        return np.zeros(3), np.zeros((3,3)), (0.0,0.0,0.0)
    vhat = np.asarray(v_vec, float)/v
    Dpar, Dpar2, Dperp2 = sidm_scalars(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, **kw)
    # A = D[Δv_parallel] vhat
    A = Dpar * vhat
    # B = Dpar2 vhat vhat^T + (Dperp2/2) (I - vhat vhat^T)
    P = np.outer(vhat, vhat)
    I = np.eye(3)
    B = Dpar2*P + 0.5*Dperp2*(I - P)
    return A, B, (Dpar, Dpar2, Dperp2)

# Sample a Maxwellian test DF (mean u_vec, dispersion s_t)
def sample_test_maxwell(N, u_vec, s_t, rng):
    return u_vec + s_t*rng.normal(size=(N,3))

# Project to moments for a Maxwellian test population
def sidm_moment_rates_Maxwell(n_t, u_vec, s_t, n_a, m_t, m_a, s_bath,
                               sigmaT, sigma_args=(), Nmc=4000, seed=0,
                               **kw):
    rng = np.random.default_rng(seed)
    V = sample_test_maxwell(Nmc, np.asarray(u_vec, float), s_t, rng)
    A_sum = np.zeros(3)
    E_rhs_sum = 0.0
    vv_sum = np.zeros((3,3))
    vv_rhs_sum = np.zeros((3,3))
    u_vec = np.asarray(u_vec, float)

    for v in V:
        A, B, _ = sidm_A_B(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, **kw)
        A_sum += A
        E_rhs_sum += A.dot(v) + 0.5*np.trace(B)  # energy RHS (per particle)
        vv_sum += np.outer(v, v)
        vv_rhs_sum += np.outer(A, v) + np.outer(v, A) + B

    # Averages over test DF:
    A_bar = A_sum/Nmc
    vv_bar = vv_sum/Nmc
    vv_rhs = vv_rhs_sum/Nmc

    # Moments and their rates:
    # mass: dn/dt=0
    dn_dt = 0.0

    # momentum: n du/dt = n * <A>
    du_dt = A_bar

    # energy: d/dt( 1/2 <v^2> ) = < A·v + 1/2 tr B >
    v2_bar = np.trace(vv_bar)
    dEkin_dt = E_rhs_sum/Nmc

    # split to bulk + random: <v^2> = |u|^2 + 3 sigma^2
    sigma2 = max(0.0, (v2_bar - np.dot(u_vec,u_vec))/3.0)
    dsigma2_dt = (2.0/3.0)*(dEkin_dt - u_vec.dot(du_dt))

    # pressure tensor P_ij = <(v_i-u_i)(v_j-u_j)> = <v_i v_j> - u_i u_j
    P = vv_bar - np.outer(u_vec,u_vec)
    d_vv_dt = vv_rhs
    dP_dt = d_vv_dt - (np.outer(du_dt, u_vec) + np.outer(u_vec, du_dt))

    return {
        "dn_dt": dn_dt,
        "du_dt": du_dt,
        "dsigma2_dt": dsigma2_dt,
        "P": P,
        "dP_dt": dP_dt,
        "sigma2": sigma2
    }

# Convenience: sweep coefficients vs |v| (for diagnostics)
def sweep_coeffs(v_grid, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), **kw):
    out = {"v": np.asarray(v_grid, float), "Dpar": [], "Dpar2": [], "Dperp2": []}
    for v in out["v"]:
        D1,D2,Dp = sidm_scalars(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, **kw)
        out["Dpar"].append(D1); out["Dpar2"].append(D2); out["Dperp2"].append(Dp)
    for k in ("Dpar","Dpar2","Dperp2"):
        out[k] = np.asarray(out[k], float)
    return out

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

def demo_sidm_transport():
    """Demonstrate SIDM transport kernel with moment projection"""
    print("=== SIDM Transport Kernel Demo ===")
    
    # SIDM parameters
    m_t, m_a = 1.0, 1.0  # test and bath masses
    n_a = 1.0e-2         # bath density
    s_bath = 1.0         # bath dispersion
    sigma0, s0 = 1.0, 2.0  # cross section parameters
    
    # Test population parameters
    n_t = 1.0
    u_vec = np.array([0.5, 0.0, 0.0])  # test bulk velocity
    s_t = 0.8                          # test dispersion
    
    print(f"Bath: n_a={n_a}, s_bath={s_bath}")
    print(f"Test: u_vec={u_vec}, s_t={s_t}")
    
    # Compute moment rates for Yukawa cross section
    rates = sidm_moment_rates_Maxwell(n_t, u_vec, s_t, n_a, m_t, m_a, s_bath,
                                      sigma_T_yukawa, (sigma0, s0),
                                      Nmc=6000, seed=7, nmu=64, Nu=120)
    
    print("\nMoment rates:")
    print(f"du/dt = {rates['du_dt']}")
    print(f"dsigma^2/dt = {rates['dsigma2_dt']}")
    print(f"Tr(dP/dt) = {np.trace(rates['dP_dt'])}")
    print(f"Pressure tensor P = \n{rates['P']}")
    
    # Sweep coefficients vs speed for diagnostics
    vgrid = np.logspace(-1, 2, 50)
    coeffs = sweep_coeffs(vgrid, n_a, m_t, m_a, s_bath, sigma_T_yukawa, (sigma0, s0))
    
    # Plot transport coefficients
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.loglog(coeffs["v"], np.abs(coeffs["Dpar"]), 'b-', label='|D_parallel|')
    plt.xlabel('$v$')
    plt.ylabel('$|D[v_{\\parallel}]|$')
    plt.title('Drift Coefficient')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.loglog(coeffs["v"], coeffs["Dpar2"], 'r-', label='D_parallel^2')
    plt.xlabel('$v$')
    plt.ylabel('$D[(v_{\\parallel})^2]$')
    plt.title('Parallel Diffusion')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.loglog(coeffs["v"], coeffs["Dperp2"], 'g-', label='D_perp^2')
    plt.xlabel('$v$')
    plt.ylabel('$D[(v_{\\perp})^2]$')
    plt.title('Perpendicular Diffusion')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fig/sidm_transport_coeffs.png", dpi=150)
    print("\nTransport coefficients plotted to fig/sidm_transport_coeffs.png")

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
    # Run SIDM transport demo
    demo_sidm_transport()
    
    # Run original orbital evolution
    main()