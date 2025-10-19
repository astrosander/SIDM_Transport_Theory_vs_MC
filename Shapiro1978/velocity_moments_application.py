import numpy as np, math
import multiprocessing as mp
import time
from functools import partial

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# ---------------- SIDM local FP coefficients (unchanged from earlier) ----------------
def maxwell_3d_pdf(u, s):
    c = (2.0*np.pi*s**2)**(-1.5)
    return c*np.exp(-(u*u)/(2.0*s*s))

def sigma_T_const(s, sigma0, s0=None):
    return sigma0

def sigma_T_yukawa(s, sigma0, s0):
    return sigma0/(1.0+(s/s0)**4)

def _quad_mu(nmu=64):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return x.astype(float), w.astype(float)

def _radial_grid(s_bath, umax_factor=8.0, Nu=160):
    umax = umax_factor*s_bath
    u = np.linspace(0.0, umax, Nu)
    du = u[1]-u[0] if Nu>1 else 1.0
    return u, du

def sidm_scalars(v, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), nmu=64, Nu=160):
    v = float(v)
    if v <= 0.0:  # handle extremely small v smoothly (limit exists and is finite)
        v = 1e-12
    mu, wmu = _quad_mu(nmu); u, du = _radial_grid(s_bath, Nu=Nu)
    alpha = m_a/(m_t+m_a); a2 = alpha**2
    shell = 2.0*np.pi*u*u*maxwell_3d_pdf(u, s_bath)*du
    U = u[:,None]; MU = mu[None,:]; V = v
    s = np.sqrt(np.maximum(V*V + U*U - 2.0*V*U*MU, 0.0))
    rate = n_a * sigmaT(s, *sigma_args) * s
    s_par = V - U*MU; s2 = s*s
    Dpar  = alpha*np.sum(shell[:,None]*wmu[None,:]*rate*(U*MU - V))
    Dpar2 = a2*np.sum(shell[:,None]*wmu[None,:]*rate*(0.5*s_par*s_par + (s2/12.0)))
    trB   = a2*np.sum(shell[:,None]*wmu[None,:]*rate*(0.75*s2))
    Dperp2= trB - Dpar2
    # numerical guardrails
    Dpar2  = max(Dpar2, 0.0)
    Dperp2 = max(Dperp2, 0.0)
    return float(Dpar), float(Dpar2), float(Dperp2)

# ---------------- Kepler tools ----------------
def jmax(E, GM):  # circular
    return GM/math.sqrt(-2.0*E)

def period(E, GM):
    a = -GM/(2.0*E)
    return 2.0*math.pi*math.sqrt(a**3/GM)

def elements_from_EJ(E, J, GM):
    Jc = jmax(E, GM)
    a  = -GM/(2.0*E)
    e  = math.sqrt(max(0.0, 1.0 - (J/Jc)**2))
    return a, e, Jc

# ---------------- orbit-average in true anomaly θ (smooth at turning points) ----------------
def orbit_avg_coeffs(E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(),
                     ntheta=512, nmu=48, Nu=160):
    a, e, Jc = elements_from_EJ(E, J, GM)
    P = period(E, GM)
    J = float(J)
    th = np.linspace(0.0, 2.0*np.pi, ntheta+1)
    thc = 0.5*(th[:-1]+th[1:]); dth = th[1]-th[0]
    r  = a*(1.0 - e*e)/(1.0 + e*np.cos(thc))
    vt = J/r
    # energy: v^2 = 2(E + GM/r)
    v2 = np.maximum(2.0*(E + GM/r), 0.0)
    v  = np.sqrt(v2)
    # FP scalars at each location
    Dpar  = np.zeros_like(v)
    Dpar2 = np.zeros_like(v)
    Dperp2= np.zeros_like(v)
    for k in range(len(v)):
        s1,s2,s3 = sidm_scalars(v[k], n_a, m_t, m_a, s_bath, sigmaT, sigma_args,
                                nmu=nmu, Nu=Nu)
        Dpar[k], Dpar2[k], Dperp2[k] = s1,s2,s3

    # local projections
    AE_loc  = Dpar*v + 0.5*(Dpar2 + Dperp2)
    DEE_loc = Dpar2*v2
    AJ_loc  = (J/np.maximum(v,1e-30))*Dpar
    DJJ_loc = (r*r)*( Dpar2*(vt/np.maximum(v,1e-30))**2 + 0.5*Dperp2*(1.0 - (vt/np.maximum(v,1e-30))**2) )
    DEJ_loc = J*Dpar2

    # average: <Q> = (1/P)∮ Q dt = (1/P)∑ Q * (r^2/J) dθ
    weight = (r*r/J) * dth
    invP   = 1.0/P
    AE  = invP*np.sum(AE_loc*weight)
    DEE = invP*np.sum(DEE_loc*weight)
    AJ  = invP*np.sum(AJ_loc*weight)
    DJJ = invP*np.sum(DJJ_loc*weight)
    DEJ = invP*np.sum(DEJ_loc*weight)
    return AE, DEE, AJ, DJJ, DEJ

# ---------------- isotropized (energy-only) coefficients ----------------
def J_iso_weights(E, GM, nJ=96):
    Jc = jmax(E, GM)
    x  = np.linspace(0.0, 1.0, nJ, endpoint=False)+0.5/nJ
    J  = Jc*np.sqrt(x)   # uniform in J^2
    w  = 2.0*J/(Jc*Jc)   # PDF(J) dJ
    return J, w

def orbit_avg_energy_only(E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(),
                          nJ=96, **kw):
    Jlist, wJ = J_iso_weights(E, GM, nJ=nJ)
    AE=DEE=0.0
    for J, w in zip(Jlist, wJ):
        aE, dEE, *_ = orbit_avg_coeffs(E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, **kw)
        AE  += w*aE
        DEE += w*dEE
    return AE, max(DEE, 1e-30)

def _compute_single_energy_coeffs(args):
    """Helper function for multiprocessing - computes coefficients for a single energy"""
    E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, nJ, kw = args
    try:
        AE, DEE = orbit_avg_energy_only(E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, nJ=nJ, **kw)
        return AE, DEE
    except Exception as e:
        print(f"Error computing coefficients for E={E}: {e}")
        return 0.0, 1e-30

def compute_energy_coeffs_parallel(E_list, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(), 
                                 nJ=96, n_processes=11, **kw):
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

# ---------------- steady 1D FP via integrating factor (robust & monotone) ----------------
def solve_energy_FP_steady(E_grid, GM, n_a, m_t, m_a, s_bath,
                           sigmaT, sigma_args=(),
                           g_outer=None, flux_E=0.0, nJ=96, **kw):
    if g_outer is None:
        g_outer = lambda E: (abs(E))**0.5  # default power law
    
    E = np.asarray(E_grid, float)
    if not (np.all(E<0.0) and np.all(np.diff(E)>0.0)):
        raise ValueError("E_grid must be strictly increasing and all < 0.")
    N  = len(E)
    
    # Compute coefficients in parallel
    print("Computing orbit-averaged coefficients...")
    AE, DEE = compute_energy_coeffs_parallel(
        E, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args,
        nJ=nJ, n_processes=11, **kw
    )
    # g' - (AE/DEE) g = -flux_E/DEE
    P = -(AE/DEE)
    Q = -flux_E/DEE
    # integrating factor μ(E) = exp( ∫ P dE )
    # do cumulative trapezoid (increasing E)
    dE  = np.diff(E)
    Pmid= 0.5*(P[1:]+P[:-1])
    logmu = np.concatenate(([0.0], np.cumsum(Pmid*dE)))
    mu    = np.exp(logmu - logmu[0])  # normalize
    # solution: g(E) = (1/μ)[ g0 + ∫ μ Q dE ]
    g0 = float(g_outer(E[0]))
    Qmid = 0.5*(Q[1:]+Q[:-1])
    mu_mid = 0.5*(mu[1:]+mu[:-1])
    int_term = np.concatenate(([0.0], np.cumsum(mu_mid*Qmid*dE)))
    g = (g0 + int_term)/np.maximum(mu, 1e-300)
    return E, g, AE, DEE

# ---------------- demo / diagnostics ----------------
if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("=== SIDM Orbit-Averaged FP Demo (Copy Version) ===")
    
    GM = 1.0
    m_t = m_a = 1.0
    n_a = 1.0e-2
    s_bath = 1.0
    sigma0, s0 = 1.0, 2.0

    print(f"GM={GM}, n_a={n_a}, s_bath={s_bath}")
    print(f"Cross section: σ₀={sigma0}, s₀={s0}")

    # strictly increasing energy grid: from deep to weak binding (all <0)
    # Create geometric sequence from -1000 to -0.01 (increasing)
    E_grid = -np.geomspace(1e3, 1e-2, 22*3)  # increasing: -1000 ... -0.01
    print(f"Energy grid: {len(E_grid)} points from E={E_grid[0]:.1e} to {E_grid[-1]:.1e}")

    g0 = lambda E: (-E)**0.5  # outer boundary "BW-like" seed; swap as needed

    # Solve with parallel processing and timing
    print("Using 11 parallel processes for coefficient computation...")
    start_time = time.time()
    
    E, g, AE, DEE = solve_energy_FP_steady(
        E_grid, GM, n_a, m_t, m_a, s_bath,
        sigma_T_yukawa, (sigma0, s0),
        g_outer=g0, flux_E=0.0,
        nJ=96, ntheta=512, nmu=48, Nu=160
    )
    
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.1f} seconds")
    print("Solution complete!")
    print(f"g(E) range: {g.min():.2e} to {g.max():.2e}")
    print(f"AE range: {AE.min():.2e} to {AE.max():.2e}")
    print(f"DEE range: {DEE.min():.2e} to {DEE.max():.2e}")

    import matplotlib.pyplot as plt
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
    
    x = -E
    fig, ax = plt.subplots(1,3, figsize=(14,4.2))
    ax[0].loglog(x, g, lw=2.5, color='C0', label=r'$g(E)$')
    ax[0].loglog(x, g0(E), ls='--', color='C3', label=r'$g_0(E)$')
    ax[0].set_xlabel(r'$-E$'); ax[0].set_ylabel(r'$g(E)$'); ax[0].grid(True, ls=':', alpha=0.4); ax[0].legend()
    ax[1].loglog(x, np.abs(AE), lw=2.5, color='C2'); ax[1].set_xlabel(r'$-E$'); ax[1].set_ylabel(r'$|A_E(E)|$'); ax[1].grid(True, ls=':', alpha=0.4)
    ax[2].loglog(x, DEE, lw=2.5, color='m'); ax[2].set_xlabel(r'$-E$'); ax[2].set_ylabel(r'$D_{EE}(E)$'); ax[2].grid(True, ls=':', alpha=0.4)
    plt.tight_layout(); plt.savefig("fig/sidm_energy_fp_diagnostics.pdf", dpi=160)
