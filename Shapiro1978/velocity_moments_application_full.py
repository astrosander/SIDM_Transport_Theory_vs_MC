import numpy as np, math
import multiprocessing as mp
import time
from functools import partial
import matplotlib.pyplot as plt

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
    # Avoid NaN in rate calculation
    s = np.maximum(s, 1e-12)
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
    return GM/np.sqrt(-2.0*E)

def period(E, GM):
    a = -GM/(2.0*E)
    return 2.0*np.pi*np.sqrt(a**3/GM)

def elements_from_EJ(E, J, GM):
    Jc = jmax(E, GM)
    a  = -GM/(2.0*E)
    e  = np.sqrt(np.maximum(0.0, 1.0 - (J/Jc)**2))
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
    # Avoid division by zero
    r = np.maximum(r, 1e-12)
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
    # Avoid division by zero
    J_safe = np.maximum(J, 1e-12)
    weight = (r*r/J_safe) * dth
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

# ---------------- Loss-cone & boundary helpers -----------------------------
def J_lc_of_E(E, GM, r_lc=None):
    """ Loss-cone: pericenter r_p(E,J)=r_lc ⇒ J_lc(E)=sqrt(2GM r_lc) * sqrt(1 + Er_lc/GM)
        For a small r_lc (tidal/disruption radius), the common approximation is:
        J_lc ≈ sqrt(2 GM r_lc). If r_lc is None, return 0 (no loss-cone).
    """
    if r_lc is None: return 0.0
    return math.sqrt(2.0*GM*r_lc)

def build_masks_losscone(E_grid, J_grid, GM, r_lc=None):
    NE,NJ = len(E_grid), len(J_grid)
    Jc = jmax(E_grid[:,None], GM)
    Jlc = np.array([J_lc_of_E(E, GM, r_lc) for E in E_grid])[:,None]
    valid = (J_grid[None,:] <= Jc) & (J_grid[None,:] >= Jlc)
    return valid, Jlc.squeeze(), Jc.squeeze()

# ---------------- Coefficients on a 2-D grid (E,J) -------------------------
def build_coefficients(E_grid, J_grid, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args=(),
                       ntheta=384, nmu=48, Nu=140):
    NE, NJ = len(E_grid), len(J_grid)
    A_E  = np.zeros((NE,NJ)); D_EE = np.zeros((NE,NJ))
    A_J  = np.zeros((NE,NJ)); D_JJ = np.zeros((NE,NJ))
    D_EJ = np.zeros((NE,NJ))
    Jc = jmax(E_grid[:,None], GM)  # shape (NE,1)
    mask = (J_grid[None,:] <= Jc)    # valid domain (bound orbits)
    
    print(f"Computing 2D coefficients for {NE}×{NJ} grid using 11 processes...")
    
    # Prepare arguments for parallel computation
    args_list = []
    for i,E in enumerate(E_grid):
        Jc_i = jmax(E, GM)
        for j,J in enumerate(J_grid):
            if J > Jc_i: continue
            args_list.append((E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, ntheta, nmu, Nu))
    
    # Compute in parallel with progress tracking
    start_time = time.time()
    with mp.Pool(processes=11) as pool:
        results = []
        total = len(args_list)
        
        for i, result in enumerate(pool.imap(_compute_single_orbit_coeffs, args_list)):
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
    
    # Unpack results back to grid
    idx = 0
    for i,E in enumerate(E_grid):
        Jc_i = jmax(E, GM)
        for j,J in enumerate(J_grid):
            if J > Jc_i: continue
            AE, DEE, AJ, DJJ, DEJ = results[idx]
            A_E[i,j], D_EE[i,j] = AE, DEE
            A_J[i,j], D_JJ[i,j] = AJ, DJJ
            D_EJ[i,j] = DEJ
            idx += 1
    
    # tiny floors
    D_EE = np.maximum(D_EE, 1e-30); D_JJ = np.maximum(D_JJ, 1e-30)
    return A_E, D_EE, A_J, D_JJ, D_EJ, mask

def _compute_single_orbit_coeffs(args):
    """Helper function for multiprocessing - computes coefficients for a single (E,J)"""
    E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args, ntheta, nmu, Nu = args
    try:
        AE, DEE, AJ, DJJ, DEJ = orbit_avg_coeffs(E, J, GM, n_a, m_t, m_a, s_bath, sigmaT, sigma_args,
                                                  ntheta=ntheta, nmu=nmu, Nu=Nu)
        return AE, DEE, AJ, DJJ, DEJ
    except Exception as e:
        print(f"Error computing coefficients for E={E}, J={J}: {e}")
        return 0.0, 1e-30, 0.0, 1e-30, 0.0

# ---------------- 1-D line solvers (Crank–Nicolson) -----------------------
def tridiag_solve(a,b,c,d):
    """Solve tridiagonal system with subdiag a[1..n-1], diag b[0..n-1], superdiag c[0..n-2]."""
    n = len(b); ac,bc,cc,dc = map(np.array, (a,b,c,d))
    for i in range(1,n):
        m = ac[i]/bc[i-1]
        bc[i] -= m*cc[i-1]
        dc[i] -= m*dc[i-1]
    x = np.zeros(n)
    x[-1] = dc[-1]/bc[-1]
    for i in range(n-2,-1,-1):
        x[i] = (dc[i]-cc[i]*x[i+1])/bc[i]
    return x

def cn_line_E(f_line, AE_line, DEE_line, dE, dt, bc_left_dir_value, bc_right_flux=0.0):
    """ Crank–Nicolson step along E for a single J, with left Dirichlet and right no-flux. """
    N = len(f_line)
    # faces
    Df = np.zeros(N-1)
    for i in range(N-1):
        Df[i] = (DEE_line[i]*DEE_line[i+1])/(DEE_line[i]+DEE_line[i+1]+1e-300)
    Aface = 0.5*(AE_line[:-1]+AE_line[1:])
    # Assemble CN: (I - dt/2 L) f^{*} = (I + dt/2 L) f^n + RHS_explicit
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
    # Left boundary Dirichlet
    b[0] = 1.0; d[0] = bc_left_dir_value
    # Interior
    for i in range(1,N-1):
        De = Df[i-1]/dE[i-1]; Dw = Df[i]/dE[i]
        Af = Aface[i-1]; Ag = Aface[i]
        LW = De + max(-Af,0.0); LE = Dw + max( Ag,0.0)
        LP = LW + LE + (max(Af,0.0)+max(-Ag,0.0))
        a[i] = -0.5*dt*LW
        b[i] = 1.0 + 0.5*dt*LP
        c[i] = -0.5*dt*LE
        # RHS
        d[i] = (1.0 - 0.5*dt*LP)*f_line[i] + 0.5*dt*(LW*f_line[i-1] + LE*f_line[i+1])
    # Right boundary: zero E-flux ⇒ -Df*(fN-fN-1)/dE + A_face*N g_up = bc_right_flux
    i = N-1
    DeN = Df[-1]/dE[-1]; AfN = Aface[-1]
    a[i] = -0.5*dt*DeN
    b[i] = 1.0 + 0.5*dt*DeN
    c[i] = 0.0
    d[i] = f_line[i] + dt*bc_right_flux
    return tridiag_solve(a, b, c, d)

def cn_line_J(f_line, AJ_line, DJJ_line, dJ, dt, bc_left_dir_zero=True, bc_right_noflux=True):
    """ Crank–Nicolson step along J for a single E, with J_lc Dirichlet(0) and J_c no-flux. """
    N = len(f_line)
    Df = np.zeros(N-1)
    for j in range(N-1):
        Df[j] = (DJJ_line[j]*DJJ_line[j+1])/(DJJ_line[j]+DJJ_line[j+1]+1e-300)
    Aface = 0.5*(AJ_line[:-1]+AJ_line[1:])
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
    # Left boundary (loss cone): f=0
    b[0]=1.0; d[0]=0.0
    for j in range(1,N-1):
        Dw = Df[j-1]/dJ[j-1]; De = Df[j]/dJ[j]
        Af = Aface[j-1]; Ag = Aface[j]
        LW = Dw + max(-Af,0.0); LE = De + max( Ag,0.0)
        LP = LW + LE + (max(Af,0.0)+max(-Ag,0.0))
        a[j] = -0.5*dt*LW
        b[j] = 1.0 + 0.5*dt*LP
        c[j] = -0.5*dt*LE
        d[j] = (1.0 - 0.5*dt*LP)*f_line[j] + 0.5*dt*(LW*f_line[j-1] + LE*f_line[j+1])
    # Right boundary (J_c): no-flux
    j = N-1
    DeN = Df[-1]/dJ[-1]; AfN = Aface[-1]
    a[j] = -0.5*dt*DeN
    b[j] = 1.0 + 0.5*dt*DeN
    c[j] = 0.0
    d[j] = f_line[j]
    return tridiag_solve(a,b,c,d)

# ---------------- ADI evolution with cross-terms explicit ------------------
def evolve_FP_ADI(E_grid, J_grid, GM, A_E, D_EE, A_J, D_JJ, D_EJ, mask,
                  f_init, g_outer, dt, nsteps, r_lc=None, report_every=10):
    E = np.asarray(E_grid, float); J = np.asarray(J_grid, float)
    NE,NJ = len(E), len(J)
    dE = np.diff(E); dJ = np.diff(J)

    f = f_init.copy()
    # Precompute index ranges per E for J-sweeps (between J_lc and J_c)
    valid, Jlc, Jc = build_masks_losscone(E, J, GM, r_lc=r_lc)
    # Left energy boundary value from g_outer
    gL = np.array([g_outer(E[0]) for _ in range(NJ)])

    for n in range(1,nsteps+1):
        # ---- cross terms explicit: C = -∂E(-D_EJ ∂J f) - ∂J(-D_EJ ∂E f)
        # central differences where valid, else zero
        C = np.zeros_like(f)
        # ∂J f
        dfdJ = np.zeros_like(f)
        if NJ > 2:
            dfdJ[:,1:-1] = (f[:,2:] - f[:,:-2])/(J[2:]-J[:-2])
        # ∂E f
        dfdE = np.zeros_like(f)
        if NE > 2:
            dfdE[1:-1,:] = (f[2:,:] - f[:-2,:])/(E[2:]-E[:-2])[:,None]
        # first term: -∂E(-D_EJ ∂J f)
        if NE > 2:
            T1 = -( ( (D_EJ[2:,:]*dfdJ[2:,:] - D_EJ[:-2,:]*dfdJ[:-2,:]) / (E[2:]-E[:-2])[:,None] ) )
            C[1:-1,:] += T1
        # second term: -∂J(-D_EJ ∂E f)
        if NJ > 2:
            T2 = -( ( (D_EJ[:,2:]*dfdE[:,2:] - D_EJ[:,:-2]*dfdE[:,:-2]) / (J[2:]-J[:-2])[None,:] ) )
            C[:,1:-1] += T2

        # ---- Half step: E-implicit (J-explicit)
        f_star = f.copy()
        for j in range(NJ):
            # if entire column invalid, skip
            valid_i = mask[:,j]
            if not np.any(valid_i): continue
            # contiguous valid segment in E (all E are valid when J <= min Jc across E segment)
            i0, i1 = np.where(valid_i)[0][[0,-1]]
            # build RHS: f + dt*( L_J f + C )
            # L_J f = -∂J( -D_JJ ∂J f + A_J f ) — do it explicit with conservative fluxes
            LJf = np.zeros(i1-i0+1)
            # fluxes along J at fixed E_i
            if 1 < NJ:
                # central differences on the full column, but apply only to segment
                i_slice = slice(i0, i1+1)
                # build per-row explicit J operator using conservative faces
                for ii in range(i0, i1+1):
                    # find the local J valid range for this E row
                    valid_j = mask[ii,:]
                    if not np.any(valid_j): continue
                    j0, j1r = np.where(valid_j)[0][[0,-1]]
                    # if our column index is outside valid row segment, skip
                    if j < j0 or j > j1r: continue
                    # finite-volume divergence at cell (ii,j)
                    # compute neighbor indices with boundary conditions
                    # left face
                    if j == j0:
                        flux_L = 0.0  # absorbing f=0 ⇒ use f=0 inside cn_line_J; here explicit zero suffices
                    else:
                        DfL = (D_JJ[ii,j-1]*D_JJ[ii,j])/(D_JJ[ii,j-1]+D_JJ[ii,j]+1e-300)
                        dJloc = J[j]-J[j-1]
                        gradL = (f[ii,j] - f[ii,j-1])/dJloc
                        AfaceL= 0.5*(A_J[ii,j-1]+A_J[ii,j])
                        up = f[ii,j-1] if AfaceL>0 else f[ii,j]
                        flux_L = -DfL*gradL + AfaceL*up
                    # right face
                    if j == j1r:
                        flux_R = 0.0  # no-flux at J_c
                    else:
                        DfR = (D_JJ[ii,j]*D_JJ[ii,j+1])/(D_JJ[ii,j]+D_JJ[ii,j+1]+1e-300)
                        dJloc = J[j+1]-J[j]
                        gradR = (f[ii,j+1] - f[ii,j])/dJloc
                        AfaceR= 0.5*(A_J[ii,j]+A_J[ii,j+1])
                        up = f[ii,j] if AfaceR>0 else f[ii,j+1]
                        flux_R = -DfR*gradR + AfaceR*up
                    LJf[ii-i0] = -(flux_R - flux_L)/(J[min(j+1,NJ-1)]-J[max(j-1,0)])
            rhs = f[i0:i1+1, j] + dt*(LJf + C[i0:i1+1, j])
            # CN in E with left Dirichlet and right no-flux
            f_star[i0:i1+1, j] = cn_line_E(
                f[i0:i1+1, j], A_E[i0:i1+1, j], D_EE[i0:i1+1, j],
                dE[i0:i1], dt,
                bc_left_dir_value=g_outer(E[0]),
                bc_right_flux=0.0
            )

        # ---- Second half: J-implicit (E-explicit)
        f_new = f_star.copy()
        for i in range(NE):
            valid_j = mask[i,:]
            if not np.any(valid_j): continue
            j0, j1 = np.where(valid_j)[0][[0,-1]]
            # RHS includes explicit E operator + C
            # E operator explicit (conservative)
            LEf = np.zeros(j1-j0+1)
            if 1 < NE:
                for jj in range(j0, j1+1):
                    # divergence along E at fixed J_jj
                    # left face (Dirichlet at Emin)
                    if i == 0:
                        flux_L = (g_outer(E[0]) - f_star[i,jj]) * 0.0  # value not used; boundary enforced in solver
                    else:
                        DfL = (D_EE[i-1,jj]*D_EE[i,jj])/(D_EE[i-1,jj]+D_EE[i,jj]+1e-300)
                        dEloc = E[i]-E[i-1]
                        gradL = (f_star[i,jj] - f_star[i-1,jj])/dEloc
                        AfaceL= 0.5*(A_E[i-1,jj]+A_E[i,jj])
                        up = f_star[i-1,jj] if AfaceL>0 else f_star[i,jj]
                        flux_L = -DfL*gradL + AfaceL*up
                    # right face (no-flux at inner energy boundary)
                    if i == NE-1:
                        flux_R = 0.0
                    else:
                        DfR = (D_EE[i,jj]*D_EE[i+1,jj])/(D_EE[i,jj]+D_EE[i+1,jj]+1e-300)
                        dEloc = E[i+1]-E[i]
                        gradR = (f_star[i+1,jj] - f_star[i,jj])/dEloc
                        AfaceR= 0.5*(A_E[i,jj]+A_E[i+1,jj])
                        up = f_star[i,jj] if AfaceR>0 else f_star[i+1,jj]
                        flux_R = -DfR*gradR + AfaceR*up
                    LEf[jj-j0] = -(flux_R - flux_L)/(E[min(i+1,NE-1)]-E[max(i-1,0)])
            rhs = f_star[i, j0:j1+1] + dt*(LEf + C[i, j0:j1+1])
            f_new[i, j0:j1+1] = cn_line_J(
                f_star[i, j0:j1+1],
                A_J[i, j0:j1+1], D_JJ[i, j0:j1+1],
                dJ[j0:j1], dt,
                bc_left_dir_zero=True, bc_right_noflux=True
            )

        f = f_new
        if (n%report_every)==0:
            print(f"step {n}/{nsteps}")

    return f

# ---------------- Example driver --------------------------------------------
def demo_2D_SIDM():
    # Physical setup (code units)
    GM = 1.0
    m_t = m_a = 1.0
    n_a = 1.0e-2
    s_bath = 1.0
    sigma0, s0 = 1.0, 2.0     # Yukawa-like σ_T(s)

    # Grids (E increasing, all negative; J from 0 to Jc(E≈0))
    NE, NJ = 36, 36  # Smaller grid for demo
    E_grid = -np.geomspace(5e2, 1e-2, NE)[::-1]  # [-5e2, ..., -1e-2] increasing
    Jmax_global = jmax(E_grid[-1], GM)         # near-zero binding ⇒ largest J_c
    J_grid = np.linspace(0.0, Jmax_global, NJ)

    # Coefficients (precompute once; this is the expensive step)
    print("Building orbit-averaged coefficients …")
    A_E, D_EE, A_J, D_JJ, D_EJ, mask = build_coefficients(
        E_grid, J_grid, GM, n_a, m_t, m_a, s_bath,
        sigma_T_yukawa, (sigma0, s0),
        ntheta=256, nmu=32, Nu=120
    )

    # Loss cone (set r_lc to enable; None disables)
    r_lc = None
    valid, Jlc, Jc = build_masks_losscone(E_grid, J_grid, GM, r_lc=r_lc)
    mask &= valid

    # Initial condition: isotropic in J, BW-like in E
    g0 = lambda E: (-E)**0.5
    f_init = np.zeros((NE,NJ))
    for i,E in enumerate(E_grid):
        Jc_i = jmax(E, GM)
        jmask = J_grid <= Jc_i
        f_init[i, jmask] = g0(E)  # independent of J to start

    # Time step & run
    dt   = 0.05   # in code time units; increase cautiously
    steps= 30    # Fewer steps for demo
    print("Evolving 2D FP …")
    f = evolve_FP_ADI(E_grid, J_grid, GM,
                      A_E, D_EE, A_J, D_JJ, D_EJ, mask,
                      f_init, g_outer=g0, dt=dt, nsteps=steps,
                      r_lc=r_lc, report_every=10)

    # Diagnostics & plots
    # Isotropized g(E) = ∫ f(E,J) (2J/Jc^2) dJ
    gE = np.zeros(NE)
    for i,E in enumerate(E_grid):
        Jc_i = jmax(E, GM)
        jmask = J_grid <= Jc_i
        w = 2.0*J_grid[jmask]/(Jc_i*Jc_i)
        gE[i] = np.trapz(f[i,jmask]*w, J_grid[jmask])

    # Plot
    X,Y = np.meshgrid(-E_grid, J_grid/Jmax_global, indexing='ij')
    plt.figure(figsize=(12,4.2))
    plt.subplot(1,3,1)
    plt.plot(-E_grid, gE, lw=2.2)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$-E$'); plt.ylabel(r'$g(E)$'); plt.title('Isotropized DF')
    plt.grid(ls=':', alpha=0.4)

    plt.subplot(1,3,2)
    plt.imshow(f.T, origin='lower',
               extent=[-E_grid[0], -E_grid[-1], 0, 1],
               aspect='auto', cmap='viridis')
    plt.xlabel(r'$-E$'); plt.ylabel(r'$J/J_{\rm max}(-E\!\to\!0)$')
    plt.title(r'$f(E,J)$'); plt.colorbar(shrink=0.78)

    plt.subplot(1,3,3)
    i_mid = NE//2
    Jc_mid = jmax(E_grid[i_mid], GM)
    jmask = J_grid<=Jc_mid
    R = (J_grid[jmask]/Jc_mid)**2
    plt.plot(R, f[i_mid,jmask], lw=2.2)
    plt.xlabel(r'$R=J^2/J_c^2(E)$'); plt.ylabel(r'$f$'); plt.title('Mid-E J-profile')
    plt.grid(ls=':', alpha=0.4)
    plt.tight_layout(); plt.savefig("sidm_fp_2D.png", dpi=180)
    print("Saved: sidm_fp_2D.png")

# ---------------- demo / diagnostics ----------------
if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("=== SIDM 2D Orbit-Averaged FP Demo ===")
    demo_2D_SIDM()
