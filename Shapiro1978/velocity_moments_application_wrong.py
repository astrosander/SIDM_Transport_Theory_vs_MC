import math, random, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import time
from functools import partial

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ---------- SIDM: local drift/diffusion in velocity space ----------
def maxwell_3d_pdf(u, s):
    c = (2.0*np.pi*s**2)**(-1.5)
    return c*np.exp(-(u*u)/(2.0*s*s))

def sigma_T_const(s, sigma0, s0=None):  # example σ_T(s)
    return sigma0

def sigma_T_yukawa(s, sigma0, s0):      # example σ_T(s)
    return sigma0/(1.0+(s/s0)**4)

def _quad_mu(nmu=64):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return x.astype(float), w.astype(float)

def _radial_grid(s_bath, umax_factor=8.0, Nu=120):
    umax = umax_factor*s_bath
    u = np.linspace(0.0, umax, Nu)
    du = u[1]-u[0] if Nu>1 else 1.0
    return u, du

def sidm_scalars(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), nmu=64, Nu=120):
    if v <= 0.0: return 0.0, 0.0, 0.0
    mu, wmu = _quad_mu(nmu); u, du = _radial_grid(s_bath, Nu=Nu)
    alpha = m_a/(m_t+m_a); a2 = alpha**2
    shell = 2.0*np.pi*u*u*maxwell_3d_pdf(u, s_bath)*du
    U = u[:,None]; MU = mu[None,:]; V = float(v)
    s = np.sqrt(np.maximum(V*V + U*U - 2.0*V*U*MU, 0.0))
    rate = n_a * sigmaT(s, *sigma_args) * s
    s_par = V - U*MU; s2 = s*s
    Dpar  = alpha*np.sum(shell[:,None]*wmu[None,:]*rate*(U*MU - V))
    Dpar2 = a2*np.sum(shell[:,None]*wmu[None,:]*rate*(0.5*s_par*s_par + (s2/12.0)))
    trB   = a2*np.sum(shell[:,None]*wmu[None,:]*rate*(0.75*s2))
    Dperp2= trB - Dpar2
    return float(Dpar), float(Dpar2), float(Dperp2)

# In-plane basis {r_hat, t_hat, z_hat=J_hat}. With B = D∥ v̂v̂ᵀ + (D⊥/2)(I-v̂v̂ᵀ),
# the projections collapse to simple forms used below.

# ---------- Kepler orbit kinematics ----------
def jmax(E, GM):          # circular
    return GM/math.sqrt(-2.0*E)

def ra_rp(E, J, GM):
    a = -GM/(2.0*E); e = math.sqrt(max(0.0, 1.0 - (J/jmax(E, GM))**2))
    return a*(1.0+e), a*(1.0-e)

def period(E, GM):
    a = -GM/(2.0*E)
    return 2.0*math.pi*math.sqrt(a**3/GM)

def vr_vt_at_r(E, J, r, GM):
    vt = J/r
    vr2 = max(0.0, 2.0*(E + GM/r) - (J*J)/(r*r))
    return math.sqrt(vr2), vt

# ---------- Orbit-averaged SIDM energy/J coefficients ----------
def orbit_avg_coeffs(E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(),
                     n_steps=400, nmu=48, Nu=120):
    ra, rp = ra_rp(E, J, GM)
    if rp <= 0 or ra <= 0: raise ValueError("unbound/invalid orbit")
    # integral (2/P)∫_{rp}^{ra} Q(r)/|v_r| dr
    P = period(E, GM)
    rs = np.linspace(rp, ra, n_steps)
    acc_AE = 0.0; acc_DEE = 0.0; acc_AJ = 0.0; acc_DJJ = 0.0; acc_DEJ = 0.0
    for i in range(n_steps-1):
        r = 0.5*(rs[i]+rs[i+1]); dr = rs[i+1]-rs[i]
        vr, vt = vr_vt_at_r(E, J, r, GM); v = math.hypot(vr, vt)
        if vr<=0: continue
        Dpar, Dpar2, Dperp2 = sidm_scalars(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args,
                                           nmu=nmu, Nu=Nu)
        # local projections (see analysis)
        AE  = Dpar*v + 0.5*(Dpar2 + Dperp2)
        DEE = Dpar2*v*v
        AJ  = (J/max(v,1e-30))*Dpar               # = r*Dpar*(vt/v)
        DJJ = (r*r)*( Dpar2*(vt/v)**2 + 0.5*Dperp2*(1.0 - (vt/v)**2) )
        DEJ = J*Dpar2                              # = r*Dpar2*vt

        w = (2.0/P)*(dr/max(vr,1e-30))
        acc_AE  += AE * w
        acc_DEE += DEE* w
        acc_AJ  += AJ * w
        acc_DJJ += DJJ* w
        acc_DEJ += DEJ* w
    return acc_AE, acc_DEE, acc_AJ, acc_DJJ, acc_DEJ

# ---------- Isotropized (1D in E) coefficients by averaging over J ----------
def J_iso_weights(E, GM, nJ=64):
    Jc = jmax(E, GM)
    # isotropic ⇒ uniform in J^2: w(J) = 2J/Jc^2, J∈[0,Jc]
    J  = Jc*np.sqrt(np.linspace(0.0, 1.0, nJ, endpoint=False)+0.5/nJ)
    w  = 2.0*J/(Jc*Jc)
    return J, w

def orbit_avg_energy_only(E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(),
                          nJ=64, **kw):
    Jlist, wJ = J_iso_weights(E, GM, nJ=nJ)
    AE=DEE=0.0
    for J, w in zip(Jlist, wJ):
        aE, dEE, *_ = orbit_avg_coeffs(E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, **kw)
        AE  += w*aE
        DEE += w*dEE
    return AE, DEE

def _compute_single_energy_coeffs(args):
    """Helper function for multiprocessing - computes coefficients for a single energy"""
    E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, nJ, kw = args
    try:
        AE, DEE = orbit_avg_energy_only(E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, nJ=nJ, **kw)
        return AE, DEE
    except Exception as e:
        print(f"Error computing coefficients for E={E}: {e}")
        return 0.0, 0.0

def compute_energy_coeffs_parallel(E_list, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), 
                                 nJ=64, n_processes=11, **kw):
    """Compute orbit-averaged energy coefficients in parallel"""
    print(f"Computing coefficients for {len(E_list)} energy points using {n_processes} processes...")
    
    # Prepare arguments for each process
    args_list = [(E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, nJ, kw) for E in E_list]
    
    start_time = time.time()
    
    # Use multiprocessing
    with mp.Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        results = []
        total = len(args_list)
        
        for i, result in enumerate(pool.imap(_compute_single_energy_coeffs, args_list)):
            results.append(result)
            
            # Progress estimation with visual bar
            elapsed = time.time() - start_time
            progress = (i + 1) / total
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            if i > 0:  # Avoid division by zero
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate
                eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
                print(f"\r[{bar}] {i+1}/{total} ({100*progress:.1f}%) - ETA: {eta}", end="", flush=True)
            else:
                print(f"\r[{bar}] {i+1}/{total} ({100*progress:.1f}%) - Computing...", end="", flush=True)
    
    print()  # New line after progress
    
    # Unpack results
    AE_list, DEE_list = zip(*results)
    return np.array(AE_list), np.array(DEE_list)

# ---------- 1D (energy-only) FP solver: ∂g/∂t = -∂E( A_E g - ∂E(D_EE g) ) ----------
def solve_energy_FP_steady(E_grid, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(),
                           g_outer=None, flux_E=0.0, nJ=64, **kw):
    """
    Finite-volume steady state on E∈[Emin,Emax] (E<0). Boundary at Emin (weakly bound):
    g(Emin)=g_outer(Emin). At Emax (deeply bound): impose energy flux F_E = flux_E.
    """
    if g_outer is None:
        g_outer = lambda E: (abs(E))**0.5  # default power law
    
    E = np.asarray(E_grid, float)  # increasing, all <0
    N = len(E); dE = np.diff(E)
    
    # Compute coefficients in parallel
    print("Computing orbit-averaged coefficients...")
    AE, DEE = compute_energy_coeffs_parallel(
        E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args,
        nJ=nJ, n_processes=11, **kw
    )
    # harmonic averaging of diffusion at faces
    Df = np.zeros(N-1)
    for i in range(N-1):
        Df[i] = (DEE[i]*DEE[i+1])/(DEE[i]+DEE[i+1]+1e-300)
    # Build tridiagonal system for steady-state conservative flux:
    # F_{i+1/2} = -Df * (g_{i+1}-g_i)/dE + A_face * g_face   with upwind A_face
    # Enforce F_{i+1/2} constant in i (steady, no sources) ⇒ unknown constant = flux_E
    Aface = 0.5*(AE[:-1]+AE[1:])
    # Upwind for advective part
    upw = (Aface>=0).astype(float)
    # Solve for g given left Dirichlet and right flux
    b = np.zeros(N); aL = np.zeros(N-1); aU = np.zeros(N-1)
    # Left boundary: Dirichlet
    gL = float(g_outer(E[0]))
    b[0] = gL; aU[0]=0.0
    # Interior: flux continuity ⇒ F_{i+1/2}=F_{i-1/2}=flux_E
    for i in range(1,N-1):
        De = Df[i-1]/dE[i-1]; Dw = Df[i]/dE[i]
        Af = Aface[i-1]; Ag = Aface[i]
        # coefficients for g_{i-1}, g_i, g_{i+1}
        aW = De + max(-Af,0.0)
        aE = Dw + max( Ag,0.0)
        aP = aW + aE + (max(Af,0.0)+max(-Ag,0.0))
        aL[i-1] = -aW
        aU[i]   = -aE
        b[i]    = (Af<0)*Af + (Ag>0)*Ag + flux_E*(1.0/dE[i-1]-1.0/dE[i])  # constant cancels; keep as small reg.
        # the simple conservative form works well in practice; tweak b if you add sources/sinks.
    # Right boundary: impose flux_E ⇒ -Df*(gN-gN-1)/dE + AfaceN g_up = flux_E
    DeN = Df[-1]/dE[-1]
    AfN = Aface[-1]
    if AfN>=0:  # upwind = last cell
        aL[-1] = -DeN
        b[-1]  = flux_E
        aP = DeN + AfN
    
    # Assemble tridiagonal system: A*g = b
    # A is tridiagonal with lower diagonal aL, main diagonal diag, upper diagonal aU
    diag = np.zeros(N)
    diag[0] = 1.0  # left boundary: g[0] = gL
    diag[-1] = DeN + max(AfN, 0.0)  # right boundary: flux condition
    
    # Interior points: flux continuity
    for i in range(1, N-1):
        De = Df[i-1]/dE[i-1]; Dw = Df[i]/dE[i]
        Af = Aface[i-1]; Ag = Aface[i]
        aW = De + max(-Af, 0.0)
        aE = Dw + max(Ag, 0.0)
        aP = aW + aE + (max(Af, 0.0) + max(-Ag, 0.0))
        diag[i] = aP
    
    # Thomas algorithm for tridiagonal system
    # Forward elimination
    for i in range(1, N):
        if diag[i-1] != 0:
            w = aL[i-1] / diag[i-1]
            diag[i] -= w * aU[i-1]
            b[i] -= w * b[i-1]
    
    # Back substitution
    g = np.zeros(N)
    g[-1] = b[-1] / diag[-1]
    for i in range(N-2, -1, -1):
        g[i] = (b[i] - aU[i] * g[i+1]) / diag[i]
    
    return E, g, AE, DEE

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

def demo_sidm_orbit_averaged():
    """Demonstrate orbit-averaged SIDM Fokker-Planck solver"""
    print("=== SIDM Orbit-Averaged FP Demo ===")
    
    # Problem setup (code units)
    GM = 1.0
    m_t, m_a = 1.0, 1.0  # test and bath masses
    n_a = 1e-2           # bath density
    s_bath = 1.0         # bath dispersion
    sigma0, s0 = 1.0, 2.0  # cross section parameters
    
    print(f"GM={GM}, n_a={n_a}, s_bath={s_bath}")
    print(f"Cross section: σ₀={sigma0}, s₀={s0}")
    
    # Energy grid (bound orbits E < 0)
    E_grid = -10.0**np.linspace(3, -2, 100)  # from deep to weak binding
    print(f"Energy grid: {len(E_grid)} points from E={E_grid[0]:.1e} to {E_grid[-1]:.1e}")
    
    # Outer boundary: power law g ∝ |E|^p
    g0 = lambda E: (abs(E))**0.5
    
    # Solve steady-state FP equation with parallel processing
    print("Computing orbit-averaged coefficients and solving FP equation...")
    print("Using 11 parallel processes for coefficient computation...")
    
    start_time = time.time()
    E, g, AE, DEE = solve_energy_FP_steady(
        E_grid, GM, n_a, m_t, m_a, s_bath,
        sigma_T_yukawa, (sigma0, s0),
        g_outer=g0, flux_E=0.0,   # zero flux = isolated system
        nJ=32, n_steps=200, nmu=32, Nu=80
    )
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.1f} seconds")
    
    print("Solution complete!")
    print(f"g(E) range: {g.min():.2e} to {g.max():.2e}")
    print(f"AE range: {AE.min():.2e} to {AE.max():.2e}")
    print(f"DEE range: {DEE.min():.2e} to {DEE.max():.2e}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Distribution function
    plt.subplot(1, 3, 1)
    plt.loglog(-E, g, 'b-', linewidth=2, label='g(E)')
    plt.loglog(-E, g0(E), 'r--', linewidth=1, alpha=0.7, label='g₀(E)')
    plt.xlabel('$-E$')
    plt.ylabel('$g(E)$')
    plt.title('Isotropized Distribution Function')
    plt.legend()
    plt.grid(True)
    
    # Drift coefficient
    plt.subplot(1, 3, 2)
    plt.loglog(-E, np.abs(AE), 'g-', linewidth=2)
    plt.xlabel('$-E$')
    plt.ylabel('$|A_E(E)|$')
    plt.title('Energy Drift Coefficient')
    plt.grid(True)
    
    # Diffusion coefficient
    plt.subplot(1, 3, 3)
    plt.loglog(-E, DEE, 'm-', linewidth=2)
    plt.xlabel('$-E$')
    plt.ylabel('$D_{EE}(E)$')
    plt.title('Energy Diffusion Coefficient')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fig/sidm_orbit_averaged_fp.png", dpi=150)
    print("\nResults plotted to fig/sidm_orbit_averaged_fp.png")
    
    # Test individual orbit-averaged coefficients
    print("\nTesting individual orbit coefficients...")
    E_test = -1.0
    J_test = 0.5 * jmax(E_test, GM)
    AE_test, DEE_test, AJ_test, DJJ_test, DEJ_test = orbit_avg_coeffs(
        E_test, J_test, GM, n_a, m_t, m_a, s_bath, sigma_T_yukawa, (sigma0, s0)
    )
    print(f"For E={E_test}, J={J_test:.3f}:")
    print(f"  A_E = {AE_test:.2e}")
    print(f"  D_EE = {DEE_test:.2e}")
    print(f"  A_J = {AJ_test:.2e}")
    print(f"  D_JJ = {DJJ_test:.2e}")
    print(f"  D_EJ = {DEJ_test:.2e}")

def demo_sidm_transport():
    """Demonstrate local SIDM transport coefficients"""
    print("=== SIDM Local Transport Coefficients Demo ===")
    
    # SIDM parameters
    m_t, m_a = 1.0, 1.0  # test and bath masses
    n_a = 1.0e-2         # bath density
    s_bath = 1.0         # bath dispersion
    sigma0, s0 = 1.0, 2.0  # cross section parameters
    
    print(f"Bath: n_a={n_a}, s_bath={s_bath}")
    
    # Sweep coefficients vs speed for diagnostics
    vgrid = np.logspace(-1, 2, 50)
    coeffs = {"v": vgrid, "Dpar": [], "Dpar2": [], "Dperp2": []}
    
    for v in vgrid:
        D1, D2, Dp = sidm_scalars(v, n_a, m_t, m_a, s_bath, sigma_T_yukawa, (sigma0, s0))
        coeffs["Dpar"].append(D1)
        coeffs["Dpar2"].append(D2)
        coeffs["Dperp2"].append(Dp)
    
    for k in ("Dpar", "Dpar2", "Dperp2"):
        coeffs[k] = np.asarray(coeffs[k])
    
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
    print("Local transport coefficients plotted to fig/sidm_transport_coeffs.png")

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
    # Set multiprocessing start method for Windows compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Run SIDM orbit-averaged FP demo
    demo_sidm_orbit_averaged()
    
    # Run SIDM local transport demo
    demo_sidm_transport()
    
    # Run original orbital evolution
    main()