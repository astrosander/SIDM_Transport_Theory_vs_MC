#!/usr/bin/env python3
# Stable MC for isotropized ḡ(x) — canonical case (x_D=1e4, P*=0.005)
# - Constant-population SMC each step (exactly N=--pop stars, equal weights)
# - Time deposition: add w*DT to current log-x bin; divide by total time
# - No per-block renormalization to g0; optional final normalize to ḡ(0.225)=1
# - One LUT from BWII g0; errors from block averages (no distortions)

import math, argparse, numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import time
import concurrent.futures as cf
import multiprocessing as mp

# Import scipy - handle potential circular import issues in multiprocessing workers
# On Windows with multiprocessing, scipy can have circular import issues
# We'll try to import it, and if it fails, Background.__init__ will retry
try:
    from scipy.integrate import cumulative_trapezoid
except (ImportError, RuntimeError, RecursionError) as e:
    # If import fails (e.g., in a worker process with circular imports),
    # set to None and let Background.__init__ retry the import
    cumulative_trapezoid = None
    # Only print warning if we're in the main process (not a worker)
    if __name__ == "__main__":
        import traceback
        if 'circular' in str(e).lower() or 'recursion' in str(e).lower():
            # This is expected in some multiprocessing scenarios, don't warn
            pass
        else:
            print(f"Warning: scipy import issue (will retry in Background): {e}", file=sys.stderr)

# ---------- Optional Numba JIT ----------
try:
    import numba as nb
    HAVE_NUMBA = True
except ImportError:
    nb = None
    HAVE_NUMBA = False

def nb_njit(*a, **k):
    """Fallback decorator if numba is not available."""
    def deco(f):
        return f
    return deco

# If numba is present and user doesn't disable it, use real njit
# (will be overridden in main() if --no-jit is set)
njit = nb.njit if HAVE_NUMBA else nb_njit
fastmath = dict(fastmath=True, nogil=True, cache=True) if HAVE_NUMBA else {}

# ----------------- Global RNG -----------------
rng = None

# ----------------- Canonical parameters -----------------
X_D     = 1.0e4
P_STAR  = 5.0e-3
X_BOUND = 0.20   # canonical outer reservoir (x_b ≈ 0.2)

# Fig. 3 bin centers from the paper (same list used for FE* runs)
X_F3 = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=float)

def _edges_from_centers(c):
    e = np.empty(c.size + 1, dtype=c.dtype)
    for i in range(1, c.size):
        e[i] = math.sqrt(c[i-1]*c[i])
    e[0]  = c[0]**2 / e[1]
    e[-1] = c[-1]**2 / e[-2]
    return e

X_EDGES = _edges_from_centers(X_F3)
DLNX    = np.log(X_EDGES[1:] / X_EDGES[:-1])
XCEN    = X_F3.copy()
XBINS   = X_EDGES.copy()  # kept for backward compatibility in bidx()

# ----------------- BWII g0(x) -----------------
_ln1p = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], float)
_g0tab = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], float)

_Xg0  = np.exp(_ln1p) - 1.0
mask  = (_g0tab > 0) & (_Xg0 > 0)
_lx   = np.log10(_Xg0[mask])
_lg   = np.log10(_g0tab[mask])
_slope = np.diff(_lg) / np.diff(_lx)

def g0_interp(x):
    """Log–log interpolation of BWII g0 with gentle taper near X_D."""
    x = np.asarray(x)
    lx = np.log10(np.clip(x, 1e-12, X_D))
    y  = np.empty_like(lx)

    left  = lx < _lx[0]
    right = lx > _lx[-1]
    mid   = ~(left | right)

    # left extrapolation
    y[left]  = _lg[0]  + _slope[0]  * (lx[left]  - _lx[0])
    # right extrapolation
    y[right] = _lg[-1] + _slope[-1] * (lx[right] - _lx[-1])

    # interior interpolation
    idx = np.clip(np.searchsorted(_lx, lx[mid]) - 1, 0, len(_slope)-1)
    y[mid] = _lg[idx] + _slope[idx] * (lx[mid] - _lx[idx])

    out = 10.0**y

    # gentle taper to zero near X_D
    x_last = _Xg0[mask][-1]
    tmask  = (x >= x_last) & (x <= X_D)
    if tmask.any():
        g_start = out[tmask][0]
        out[tmask] = np.maximum(g_start * (1 - (x[tmask]-x_last)/(X_D-x_last)), 1e-16)
    return out

# ----------------- Kepler helpers for capture phase check -----------------
@njit(**fastmath)
def j_min_exact(x):
    xd = 1.0e4  # X_D constant for numba
    val = 2.0*x/xd - (x*x)/(xd*xd)
    if val < 0.0:
        val = 0.0
    if val > 1.0:
        val = 1.0
    return math.sqrt(val)

@njit(**fastmath)
def Ptilde(x):  # ∝ x^{-3/2}, only for pericenter counting
    result = x**(-1.5)
    if result < 1e-3:
        return 1e-3
    return result

# ----------------- θ-quadrature (12-pt) -----------------
def th_nodes(n=12):
    t, w = np.polynomial.legendre.leggauss(n)
    th = 0.25*np.pi*(t + 1.0)
    wt = 0.25*np.pi*w
    s, c = np.sin(th), np.cos(th)
    return th, wt, s, c

TH, WT, S, C = th_nodes(12)

# ----------------- Appendix A fast kernels -----------------
def _z123(x, j, xp, xap, xp2):
    inv_x1 = 1.0/max(xp2, xap) - 1.0/max(xp, 1e-12)
    inv_x2 = 1.0/min(xp2, xap) - 1.0/max(xp, 1e-12)
    inv_x3 = 1.0/max(x,  1e-12) - 1.0/max(xp, 1e-12)
    x1 = 1.0/max(inv_x1, 1e-16)
    x2 = 1.0/max(inv_x2, 1e-16)
    x3 = 1.0/max(inv_x3, 1e-16)
    z1 = 1.0 + (xp/x1)*(S*S)
    z2 = 1.0 - (x2/x1)*(S*S)
    z3 = 1.0 - (x3/x1)*(S*S)
    return x1, x2, x3, z1, z2, z3

@njit(**fastmath)
def pericenter_xp(x, j):
    e = math.sqrt(max(0.0, 1.0 - j*j))
    denom = 1.0 - e
    if denom < 1e-12:
        denom = 1e-12
    return 2.0*x / denom

@njit(**fastmath)
def apocenter_xap(x, j):
    e = math.sqrt(max(0.0, 1.0 - j*j))
    return 2.0*x / (1.0 + e)

def I1(xp2):  return 0.25*np.pi/(xp2**1.5)
def I4(xp2):  return 0.25*np.pi/(xp2**0.5)

def I2(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((xp2*x3)/(x*x*xp*x2))
    integ= z1*np.sqrt(np.clip(z2,0,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I3(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((xp2*x2*x3)/(x*x*x1*xp*xp))
    integ= (C*C)*z1/np.sqrt(np.clip(z2,1e-20,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I6(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((xp2*x2*x3)/(x*x*(x1**3)))
    integ= (C**4)*z1/np.sqrt(np.clip(z2,1e-20,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I7(x,j,xp):
    x1 = 1.0/(1.0/max(x,1e-12) - 1.0/max(xp,1e-12))
    z1 = 1.0 + (xp/x1)*(S*S)
    pref= math.sqrt(1.0/(x*(xp**3)))
    return pref*np.sum(z1*WT)

def I9(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((xp2*x2*x3)/(x*(xp**3)*(x1**2)))
    integ= (C*C)*(z1**3)/np.sqrt(np.clip(z2,1e-20,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I10(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt(((xp2**3)*(x3**3))/(x*(xp**3)*(x2**3)))
    integ= (z1**3)*(z2**1.5)/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)

def I11(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt(((xp2**3)*(x2**3))/(x*(xp**3)*(x1**3)))
    integ= (C**4)*(z1**3)/np.sqrt(np.clip(z2,1e-20,None))/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)

def I12(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((4*(xp2**2)*(x3**3))/((x**2)*(xp**4)*x1*x2))
    integ= (S*S)*(C*C)*(z1**2)*np.sqrt(np.clip(z2,0,None))/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)

def I13(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((4*(xp2**3)*(x3**5))/((x**4)*(xp**4)*(x1**4)*(x2**3)))
    integ= (C*C)*(S*S)*(z1**3)*(z2**1.5)/np.power(np.clip(z3,1e-20,None),2.5)
    return pref*np.sum(integ*WT)

def I14(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt((4*(xp2**3)*(x3**5))/((x**4)*(xp**4)*(x1**6)*x2))
    integ= (C**4)*(S*S)*(z1**3)/np.sqrt(np.clip(z2,1e-20,None))/np.power(np.clip(z3,1e-20,None),2.5)
    return pref*np.sum(integ*WT)

def I15(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt(((xp2**3)*(x3**3))/((x**4)*(xp**4)*(x2**3)))
    integ= (z1)*(z2**1.5)/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)

def I16(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3 = _z123(x,j,xp,xap,xp2)
    pref = math.sqrt(((xp2**3)*x2*(x3**3))/((x**4)*(xp**2)*(x1**4)))
    integ= (C**4)*z1/np.sqrt(np.clip(z2,1e-20,None))/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)

# ----------------- smoothing helper -----------------
def smooth_log(xs, ys, k=7):
    """Smooth ys(x) in log-x space with Gaussian kernel, return positive array."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    lx = np.log(xs)
    ly = np.log(np.clip(ys, 1e-20, None))
    ker = np.exp(-0.5 * ((np.arange(-k, k+1) / 2.0)**2))
    ker /= ker.sum()
    ly_s = np.convolve(ly, ker, mode='same')
    return np.exp(ly_s)

# ----------------- background + moments -----------------
class Background:
    def __init__(self, xs, gs):
        # Lazy import of scipy to avoid circular import issues in multiprocessing workers
        global cumulative_trapezoid
        if cumulative_trapezoid is None:
            try:
                from scipy.integrate import cumulative_trapezoid as _ct
                cumulative_trapezoid = _ct
            except (ImportError, RuntimeError, RecursionError) as e:
                raise RuntimeError(
                    f"Failed to import scipy.integrate.cumulative_trapezoid: {e}\n"
                    "This is required for Background class. "
                    "If using multiprocessing, try setting start method to 'spawn'."
                ) from e
        
        # Build an internal fine log grid and precompute moments
        self.xg = np.logspace(-3, 4, 400)
        lx   = np.log(self.xg)
        lx_s = np.log(xs)
        glog = np.log(np.clip(np.interp(lx, lx_s, np.log(np.clip(gs,1e-16,None))), 1e-16, None))
        # mild smoothing
        k = 7
        ker = np.exp(-0.5*((np.arange(-k,k+1)/2.0)**2)); ker /= ker.sum()
        glog = np.convolve(glog, ker, mode='same')
        self.gg = np.exp(glog)

        self.Mm32 = cumulative_trapezoid(self.gg * self.xg**(-1.5), self.xg, initial=0.0)
        self.Mm12 = cumulative_trapezoid(self.gg * self.xg**(-0.5), self.xg, initial=0.0)

    def g(self, x):
        x = np.asarray(x)
        idx = np.clip(np.searchsorted(self.xg, x) - 1, 0, len(self.xg)-2)
        x0,x1 = self.xg[idx], self.xg[idx+1]
        g0,g1 = self.gg[idx], self.gg[idx+1]
        t = (x - x0) / np.maximum(x1-x0, 1e-30)
        return np.maximum(g0*(1-t) + g1*t, 1e-16)

    def moment(self, x, which):
        M = self.Mm32 if which=='m32' else self.Mm12
        x = np.asarray(x)
        idx = np.clip(np.searchsorted(self.xg, x) - 1, 0, len(self.xg)-2)
        x0,x1 = self.xg[idx], self.xg[idx+1]
        m0,m1 = M[idx], M[idx+1]
        t = (x - x0) / np.maximum(x1-x0, 1e-30)
        return m0*(1-t) + m1*t

# tiny x′-quadrature
T12, W12 = np.polynomial.legendre.leggauss(12)

def _int_region(fun_tuple, x, j, xp, xap, bg: Background):
    val = 0.0
    # (1) x'<x
    if fun_tuple[0] is I1:
        val += np.pi*0.25 * bg.moment(x, 'm32')
    elif fun_tuple[0] is I4:
        val += np.pi*0.25 * bg.moment(x, 'm12')
    else:
        lo,hi = 1e-4,x
        if hi>lo*(1+1e-10):
            llo,lhi = math.log(lo),math.log(hi)
            lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
            xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
            vals= np.array([fun_tuple[0](x,j,xp,xap,xp2)*bg.g(xp2) for xp2 in xs])
            val+= float(np.sum(vals*jac*W12))
    # (2) x≤x'<x_ap
    lo,hi = x, min(apocenter_xap(x,j), X_D)
    if hi>lo*(1+1e-10):
        llo,lhi = math.log(lo),math.log(hi)
        lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
        xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
        vals= np.array([fun_tuple[1](x,j,xp,xap,xp2)*bg.g(xp2) for xp2 in xs])
        val+= float(np.sum(vals*jac*W12))
    # (3) x_ap≤x'≤x_p
    lo,hi = max(apocenter_xap(x,j), x*(1+1e-12)), min(pericenter_xp(x,j), X_D)
    if hi>lo*(1+1e-10):
        llo,lhi = math.log(lo),math.log(hi)
        lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
        xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
        vals= np.array([fun_tuple[2](x,j,xp,xap,xp2)*bg.g(xp2) for xp2 in xs])
        val+= float(np.sum(vals*jac*W12))
    return val

def orbital_perturbations(x, j, bg: Background):
    xp, xap = pericenter_xp(x,j), apocenter_xap(x,j)

    e1 = 3.0*np.sqrt(2*np.pi)*P_STAR * _int_region(
        (lambda *a: I1(a[-1]), I2, I3), x,j,xp,xap,bg
    )
    e2 = 4.0*np.sqrt(2*np.pi)*P_STAR * _int_region(
        (lambda *a: I4(a[-1]), I6, I6), x,j,xp,xap,bg
    )
    e2 = max(e2, 0.0)

    # j-diffusion, three regimes
    if j > 0.95:
        j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
            (lambda *a: 2*I7(x,j,xp),
             lambda *a: 6*I12(x,j,xp,xap,a[-1])-9*I9(x,j,xp,xap,a[-1])-I10(x,j,xp,xap,a[-1]),
             lambda *a: 6*I12(x,j,xp,xap,a[-1])-9*I9(x,j,xp,xap,a[-1])-I11(x,j,xp,xap,a[-1])),
            x,j,xp,xap,bg
        )
        j2 = max(e2/(2*x), 0.0)
        z2 = min(e2*j2, e2*j2+1e-22)
        return e1, e2, j1, j2, max(z2,0.0)

    if j < 0.05:
        j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
            (lambda *a: 2*I7(x,j,xp),
             lambda *a: 6*I12(x,j,xp,xap,a[-1])-9*I9(x,j,xp,xap,a[-1])-I10(x,j,xp,xap,a[-1]),
             lambda *a: 6*I12(x,j,xp,xap,a[-1])-9*I9(x,j,xp,xap,a[-1])-I11(x,j,xp,xap,a[-1])),
            x,j,xp,xap,bg
        )
        j2 = math.sqrt(max(2*j*j1, 0.0))
        z2 = min(e2*j2, e2*j2+1e-22)
        return e1, e2, j1, j2, max(z2,0.0)

    # generic j
    j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
        (lambda *a: 2*I7(x,j,xp),
         lambda *a: 3*I12(x,j,xp,xap,a[-1])+4*I10(x,j,xp,xap,a[-1])-3*I13(x,j,xp,xap,a[-1]),
         lambda *a: 3*I12(x,j,xp,xap,a[-1])+4*I11(x,j,xp,xap,a[-1])-3*I14(x,j,xp,xap,a[-1])),
        x,j,xp,xap,bg
    )
    j2 = np.sqrt(2*np.pi)*x*P_STAR * _int_region(
        (lambda *a: 4*I7(x,j,xp),
         lambda *a: 3*I12(x,j,xp,xap,a[-1])+4*I10(x,j,xp,xap,a[-1])-3*I13(x,j,xp,xap,a[-1]),
         lambda *a: 3*I12(x,j,xp,xap,a[-1])+4*I11(x,j,xp,xap,a[-1])-3*I14(x,j,xp,xap,a[-1])),
        x,j,xp,xap,bg
    )
    j2 = max(j2, 0.0)
    z2 = 2*np.sqrt(2*np.pi)*(j**2)*P_STAR * _int_region(
        (lambda *a: I1(a[-1]), I15, I16), x,j,xp,xap,bg
    )
    z2 = max(min(z2, e2*j2+1e-22), 0.0)
    return e1, e2, j1, j2, z2

# ----------------- coefficient LUT -----------------
def _compute_LUT_chunk(ix_start, ix_end, Xg, Jg, bg: "Background"):
    """Worker function: compute LUT rows ix_start:ix_end for all j."""
    nx_chunk = ix_end - ix_start
    nj = Jg.size

    E1_c = np.zeros((nx_chunk, nj))
    E2_c = np.zeros_like(E1_c)
    J1_c = np.zeros_like(E1_c)
    J2_c = np.zeros_like(E1_c)
    Z2_c = np.zeros_like(E1_c)

    for local_ix, ix in enumerate(range(ix_start, ix_end)):
        x = Xg[ix]
        for ij, j in enumerate(Jg):
            e1, e2, j1, j2, z2 = orbital_perturbations(x, j, bg)
            E1_c[local_ix, ij] = e1
            E2_c[local_ix, ij] = e2
            J1_c[local_ix, ij] = j1
            J2_c[local_ix, ij] = j2
            Z2_c[local_ix, ij] = z2

    return ix_start, ix_end, E1_c, E2_c, J1_c, J2_c, Z2_c

def precompute_LUT(nx, nj, bg: Background, n_procs: int = 1):
    Xg = np.logspace(-3, 4, nx)
    Jg = np.linspace(0.0, 1.0, nj)

    E1 = np.zeros((nx, nj))
    E2 = np.zeros_like(E1)
    J1 = np.zeros_like(E1)
    J2 = np.zeros_like(E1)
    Z2 = np.zeros_like(E1)

    print(f"Precomputing LUT {nx}x{nj} ...", file=sys.stderr, flush=True)

    if n_procs <= 1:
        # Original serial version
        for ix, x in enumerate(Xg):
            for ij, j in enumerate(Jg):
                e1, e2, j1, j2, z2 = orbital_perturbations(x, j, bg)
                E1[ix, ij] = e1
                E2[ix, ij] = e2
                J1[ix, ij] = j1
                J2[ix, ij] = j2
                Z2[ix, ij] = z2
    else:
        # Parallel over x using ProcessPoolExecutor
        n_procs = min(n_procs, mp.cpu_count())
        # Build contiguous chunks in x
        base = nx // n_procs
        rem = nx % n_procs
        chunks = []
        start = 0
        for p in range(n_procs):
            size = base + (1 if p < rem else 0)
            if size <= 0:
                continue
            end = start + size
            chunks.append((start, end))
            start = end

        total_expected = len(chunks)
        total_streams_completed = 0
        start_time = time.time()

        with cf.ProcessPoolExecutor(max_workers=n_procs) as ex:
            futures = [
                ex.submit(_compute_LUT_chunk, s, e, Xg, Jg, bg)
                for (s, e) in chunks
            ]
            for fut in cf.as_completed(futures):
                ix_start, ix_end, E1_c, E2_c, J1_c, J2_c, Z2_c = fut.result()
                E1[ix_start:ix_end, :] = E1_c
                E2[ix_start:ix_end, :] = E2_c
                J1[ix_start:ix_end, :] = J1_c
                J2[ix_start:ix_end, :] = J2_c
                Z2[ix_start:ix_end, :] = Z2_c

                # progress accounting
                total_streams_completed += 1
                elapsed = time.time() - start_time
                rate = total_streams_completed / max(elapsed, 1e-6)
                eta = (total_expected - total_streams_completed) / max(rate, 1e-6)

                # "rep/streams" labels are mostly cosmetic here, but match your format
                rep = 0
                replicates = 1
                completed = total_streams_completed
                n_streams = total_expected
                print(
                    f"\rProgress: rep {rep+1}/{replicates}, {completed}/{n_streams} streams | "
                    f"Total: {total_streams_completed}/{total_expected} "
                    f"({100*total_streams_completed/total_expected:5.1f}%) | "
                    f"Rate: {rate:5.1f} streams/s | ETA: {eta:5.0f}s",
                    end="", flush=True, file=sys.stderr
                )

        print("", file=sys.stderr)  # newline after progress bar

    return dict(x=Xg, j=Jg, e1=E1, e2=E2, j1=J1, j2=J2, z2=Z2)

def _interp2(bx, by, gx, gy, A):
    ix = np.clip(np.searchsorted(gx, bx) - 1, 0, len(gx)-2)
    iy = np.clip(np.searchsorted(gy, by) - 1, 0, len(gy)-2)
    x0,x1 = gx[ix], gx[ix+1]
    y0,y1 = gy[iy], gy[iy+1]
    tx = (bx - x0)/max(x1-x0,1e-30)
    ty = (by - y0)/max(y1-y0,1e-30)
    a00 = A[ix,iy];   a10 = A[ix+1,iy]
    a01 = A[ix,iy+1]; a11 = A[ix+1,iy+1]
    return (1-tx)*(1-ty)*a00 + tx*(1-ty)*a10 + (1-tx)*ty*a01 + tx*ty*a11

def coeff_from_LUT(x, j, L):
    e1 = _interp2(x,j,L['x'],L['j'],L['e1'])
    e2 = _interp2(x,j,L['x'],L['j'],L['e2'])
    j1 = _interp2(x,j,L['x'],L['j'],L['j1'])
    j2 = _interp2(x,j,L['x'],L['j'],L['j2'])
    z2 = _interp2(x,j,L['x'],L['j'],L['z2'])

    e2 = max(e2, 0.0); j2 = max(j2, 0.0)
    z2 = max(min(z2, e2*j2+1e-22), 0.0)

    # asymptotes
    if j>0.95 and e2>0:
        j2 = max(e2/(2*x),0.0)
        z2 = min(e2*j2, e2*j2+1e-22)
    if j<0.05 and j1>0:
        j2 = math.sqrt(max(2*j*j1,0.0))
    z2 = max(min(z2, e2*j2+1e-22),0.0)
    return e1,e2,j1,j2,z2

# ----------------- FP consistency diagnostic -----------------
def check_fp_consistency(LUT, bg: Background = None):
    """
    Check if g0 is a stationary solution of the FP operator defined by LUT.
    
    The MC implements: dx/dt = -e1 + sqrt(e2) * noise
    So the FP equation in x-space is:
        ∂g/∂t = ∂/∂x[e1 * g] + (1/2)∂²/∂x²[e2 * g]
    
    For stationarity: L[g0] = ∂/∂x[e1 * g0] + (1/2)∂²/∂x²[e2 * g0] = 0
    
    This function computes j-averaged coefficients and checks the residual.
    
    Returns:
        dict with keys: x, D_E (j-averaged drift), D_EE (j-averaged diffusion),
        g0, dg0_dx, d2g0_dx2, residual, relative_residual
    """
    x_grid = LUT['x']
    j_grid = LUT['j']
    nx = len(x_grid)
    
    # j-averaged coefficients (weighted by phase-space measure)
    # For isotropic distribution, p(j) = 2j, so we weight by 2j
    j_weights = 2.0 * j_grid  # p(j) = 2j for isotropic
    j_weights = j_weights / j_weights.sum()  # normalize
    
    D_E = np.zeros(nx)
    D_EE = np.zeros(nx)
    
    for ix in range(nx):
        # Weighted average over j
        D_E[ix] = np.sum(LUT['e1'][ix, :] * j_weights)
        D_EE[ix] = np.sum(LUT['e2'][ix, :] * j_weights)
    
    # Evaluate g0 on the LUT x-grid
    g0_vals = g0_interp(x_grid)
    g0_vals = np.maximum(g0_vals, 1e-16)
    
    # Compute derivatives in x-space (not ln x)
    dg0_dx = np.gradient(g0_vals, x_grid)
    d2g0_dx2 = np.gradient(dg0_dx, x_grid)
    
    # Also compute derivatives of coefficients
    dD_E_dx = np.gradient(D_E, x_grid)
    dD_EE_dx = np.gradient(D_EE, x_grid)
    d2D_EE_dx2 = np.gradient(dD_EE_dx, x_grid)
    
    # FP operator: L[g] = ∂/∂x[e1 * g] + (1/2)∂²/∂x²[e2 * g]
    # Expanding: e1 * dg/dx + g * de1/dx + (1/2)[e2 * d²g/dx² + 2*(de2/dx)*(dg/dx) + g * d²e2/dx²]
    # For stationarity: L[g0] = 0
    Lg0 = (D_E * dg0_dx + 
           g0_vals * dD_E_dx + 
           0.5 * (D_EE * d2g0_dx2 + 
                  2.0 * dD_EE_dx * dg0_dx + 
                  g0_vals * d2D_EE_dx2))
    
    # Relative residual (normalized by typical scale of each term)
    drift_term = D_E * dg0_dx
    drift_coeff_term = g0_vals * dD_E_dx
    diff_term = 0.5 * D_EE * d2g0_dx2
    diff_mixed_term = dD_EE_dx * dg0_dx
    diff_coeff_term = 0.5 * g0_vals * d2D_EE_dx2
    
    scale = (np.abs(drift_term) + np.abs(drift_coeff_term) + 
             np.abs(diff_term) + np.abs(diff_mixed_term) + 
             np.abs(diff_coeff_term) + 1e-30)
    relative_residual = Lg0 / scale
    
    return dict(
        x=x_grid,
        D_E=D_E,
        D_EE=D_EE,
        g0=g0_vals,
        dg0_dx=dg0_dx,
        d2g0_dx2=d2g0_dx2,
        dD_E_dx=dD_E_dx,
        dD_EE_dx=dD_EE_dx,
        residual=Lg0,
        relative_residual=relative_residual,
        drift_term=drift_term,
        drift_coeff_term=drift_coeff_term,
        diff_term=diff_term,
        diff_mixed_term=diff_mixed_term,
        diff_coeff_term=diff_coeff_term
    )

# ----------------- MC machinery -----------------
@dataclass
class Star:
    x: float
    j: float
    phase: float = 0.0

def sample_initial(n):
    xs = np.full(n, X_BOUND, dtype=float)
    js = np.sqrt(rng.random(n))  # p(j)=2j
    return [Star(float(xs[i]), float(js[i]), 0.0) for i in range(n)]

def sample_initial_from_g0(n):
    """Sample initial x from g0(x) distribution between X_BOUND and X_D."""
    # Define a fine grid in x for sampling
    xs = np.logspace(math.log10(X_BOUND), math.log10(X_D), 2000)
    g = g0_interp(xs)
    # We want p(x) ∝ g(x) per dln x
    # Since our bins are equal in ln x, g(x) itself is ∝ probability per dln x
    pdf = g
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    # Inverse-CDF sampling
    us = rng.random(n)
    x_samples = np.interp(us, cdf, xs)
    j_samples = np.sqrt(rng.random(n))  # isotropic

    return [Star(float(x_samples[i]), float(j_samples[i]), 0.0) for i in range(n)]

def bidx(x):
    return int(np.clip(np.searchsorted(X_EDGES, x, side='right')-1, 0, len(XCEN)-1))

# ----- Optional importance floors (log-x) -----
def floor_idx_for(x, X_FLOORS):
    if X_FLOORS is None or len(X_FLOORS) == 0:
        return -1
    k = -1
    for i, xf in enumerate(X_FLOORS):
        if x >= xf:
            k = i
    return k

def maybe_split(obj, X_FLOORS, n_clones):
    if X_FLOORS is None or n_clones <= 0:
        return []
    k_now = floor_idx_for(obj['x'], X_FLOORS)
    new_clones = []
    while obj['floor_idx'] < k_now:
        obj['floor_idx'] += 1
        new_w = obj['w'] / (1.0 + n_clones)
        obj['w'] = new_w
        for _ in range(n_clones):
            new_clones.append(dict(
                x=obj['x'], j=obj['j'], w=new_w,
                phase=obj['phase'], floor_idx=obj['floor_idx']
            ))
    return new_clones

# Numba-parallel star evolution (array-based for parallelization)
if HAVE_NUMBA:
    from numba import prange
    from numba import types
    from numba.typed import Dict
    
    @njit(parallel=True, **fastmath)
    def evolve_stars_parallel(x_arr, j_arr, phase_arr, x_prev_arr,
                              e1_grid, e2_grid, j1_grid, j2_grid, z2_grid,
                              x_grid, j_grid, DT,
                              disable_capture, flip_energy_sign, inner_sink_frac, no_reservoir,
                              seed_offset):
        """
        Evolve all stars in parallel using numba prange.
        
        Parameters:
        - x_arr, j_arr, phase_arr: current star states (modified in-place)
        - x_prev_arr: previous x positions (for deposition)
        - e1_grid, etc.: LUT coefficient grids
        - x_grid, j_grid: LUT coordinate grids
        - seed_offset: offset for thread-safe RNG
        """
        n = len(x_arr)
        xd = 1.0e4  # X_D
        xbound = 0.2  # X_BOUND
        
        for i in prange(n):
            # Thread-local RNG seed (thread-safe LCG)
            rng_seed = (seed_offset + i) * 1103515245 + 12345
            
            x = x_arr[i]
            j = j_arr[i]
            phase = phase_arr[i]
            # x_prev_arr[i] is already set to pre-step position before calling this function
            
            # Bilinear interpolation of coefficients from LUT
            # Find grid indices
            ix = 0
            for idx in range(len(x_grid) - 1):
                if x >= x_grid[idx] and x < x_grid[idx + 1]:
                    ix = idx
                    break
            if x >= x_grid[-1]:
                ix = len(x_grid) - 2
            
            ij = 0
            for idx in range(len(j_grid) - 1):
                if j >= j_grid[idx] and j < j_grid[idx + 1]:
                    ij = idx
                    break
            if j >= j_grid[-1]:
                ij = len(j_grid) - 2
            
            # Bilinear interpolation
            x0, x1 = x_grid[ix], x_grid[ix + 1]
            j0, j1 = j_grid[ij], j_grid[ij + 1]
            tx = (x - x0) / max(x1 - x0, 1e-30)
            ty = (j - j0) / max(j1 - j0, 1e-30)
            tx = max(0.0, min(1.0, tx))
            ty = max(0.0, min(1.0, ty))
            
            e1 = (1-tx)*(1-ty)*e1_grid[ix,ij] + tx*(1-ty)*e1_grid[ix+1,ij] + (1-tx)*ty*e1_grid[ix,ij+1] + tx*ty*e1_grid[ix+1,ij+1]
            e2 = (1-tx)*(1-ty)*e2_grid[ix,ij] + tx*(1-ty)*e2_grid[ix+1,ij] + (1-tx)*ty*e2_grid[ix,ij+1] + tx*ty*e2_grid[ix+1,ij+1]
            j1 = (1-tx)*(1-ty)*j1_grid[ix,ij] + tx*(1-ty)*j1_grid[ix+1,ij] + (1-tx)*ty*j1_grid[ix,ij+1] + tx*ty*j1_grid[ix+1,ij+1]
            j2 = (1-tx)*(1-ty)*j2_grid[ix,ij] + tx*(1-ty)*j2_grid[ix+1,ij] + (1-tx)*ty*j2_grid[ix,ij+1] + tx*ty*j2_grid[ix+1,ij+1]
            z2 = (1-tx)*(1-ty)*z2_grid[ix,ij] + tx*(1-ty)*z2_grid[ix+1,ij] + (1-tx)*ty*z2_grid[ix,ij+1] + tx*ty*z2_grid[ix+1,ij+1]
            
            e2 = max(e2, 0.0)
            j2 = max(j2, 0.0)
            z2 = max(min(z2, e2*j2+1e-22), 0.0)
            
            # Correlated Gaussian kick
            muE, muJ = e1*DT, j1*DT
            sE2, sJ2, sEZ = e2*DT, j2*DT, z2*DT
            a = math.sqrt(max(sE2, 0.0) + 1e-30)
            c0 = math.sqrt(max(sJ2, 0.0) + 1e-30)
            rho_val = sEZ / max(a*c0, 1e-30)
            if rho_val > 0.999:
                rho_val = 0.999
            if rho_val < -0.999:
                rho_val = -0.999
            
            # Generate normal random numbers using LCG + Box-Muller
            # First normal
            rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
            u1 = rng_seed / 2147483648.0
            rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
            u2 = rng_seed / 2147483648.0
            z1 = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2.0 * math.pi * u2)
            # Second normal
            rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
            u1 = rng_seed / 2147483648.0
            rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
            u2 = rng_seed / 2147483648.0
            z2n = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2.0 * math.pi * u2)
            
            dE = muE + a*z1
            dJ = muJ + rho_val*c0*z1 + c0*math.sqrt(max(1-rho_val*rho_val, 0.0))*z2n
            
            # Bounded step
            max_dx = 0.15 * max(x, 1e-6)
            jmin_val = math.sqrt(max(0.0, min(1.0, 2.0*x/xd - (x*x)/(xd*xd))))
            max_dj = min(0.10, 0.40*max(0.0, 1.0 - j), max(0.25*abs(j - jmin_val), 0.10*max(jmin_val, 1e-8)))
            
            # Energy update
            dE_clipped = max(-max_dx, min(max_dx, dE))
            if flip_energy_sign:
                x = max(1e-4, min(xd, x + dE_clipped))
            else:
                x = max(1e-4, min(xd, x - dE_clipped))
            
            dj = max(-max_dj, min(max_dj, dJ))
            j = max(0.0, min(1.0, j + dj))
            
            # Inner sink boundary
            if inner_sink_frac > 0.0:
                sink_threshold = (1.0 - inner_sink_frac) * xd
                if x >= sink_threshold:
                    if no_reservoir:
                        x = min(x, xd * 0.999)
                    else:
                        x = xbound
                        rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
                        j = math.sqrt(rng_seed / 2147483648.0)
                        phase = 0.0
                    x_arr[i] = x
                    j_arr[i] = j
                    phase_arr[i] = phase
                    continue
            
            # Reservoir boundary
            if not no_reservoir and x < xbound:
                x = xbound
                rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
                j = math.sqrt(rng_seed / 2147483648.0)
            
            # Capture at pericenter crossings
            if not disable_capture:
                prev_phase_int = int(phase)
                ptilde_val = max(1e-3, x**(-1.5))
                phase += DT / max(ptilde_val, 1e-6)
                if int(phase) > prev_phase_int and j <= jmin_val:
                    if no_reservoir:
                        j = max(jmin_val * 1.01, 1e-6)
                        phase = 0.0
                    else:
                        x = xbound
                        rng_seed = (rng_seed * 1103515245 + 12345) & 0x7fffffff
                        j = math.sqrt(rng_seed / 2147483648.0)
                        phase = 0.0
            
            x_arr[i] = x
            j_arr[i] = j
            phase_arr[i] = phase
else:
    # Fallback if numba not available
    def evolve_stars_parallel(*args, **kwargs):
        raise RuntimeError("numba is required for parallel star evolution")

def mc_step(s: Star, L, DT, disable_capture=False, flip_energy_sign=False, inner_sink_frac=0.0, no_reservoir=False):
    e1,e2,j1,j2,z2 = coeff_from_LUT(s.x, s.j, L)

    # correlated Gaussian kick
    muE, muJ = e1*DT, j1*DT
    sE2, sJ2, sEZ = e2*DT, j2*DT, z2*DT
    a  = math.sqrt(max(sE2, 0.0) + 1e-30)
    c0 = math.sqrt(max(sJ2, 0.0) + 1e-30)
    rho = np.clip(sEZ/(a*c0), -0.999, 0.999)
    z1,z2n = rng.standard_normal(), rng.standard_normal()
    dE = muE + a*z1
    dJ = muJ + rho*c0*z1 + c0*math.sqrt(max(1-rho*rho,0.0))*z2n

    # bounded step (Shapiro 29a–c style, conservative)
    max_dx = 0.15*max(s.x, 1e-6)
    jmin   = j_min_exact(s.x)
    max_dj = min(0.10,
                 0.40*max(0.0, 1.0 - s.j),
                 max(0.25*abs(s.j - jmin), 0.10*max(jmin,1e-8)))

    # Energy update: flip sign if requested (diagnostic for sign convention)
    if flip_energy_sign:
        s.x = float(np.clip(s.x + np.clip(dE, -max_dx, max_dx), 1e-4, X_D))
    else:
        s.x = float(np.clip(s.x - np.clip(dE, -max_dx, max_dx), 1e-4, X_D))
    dj  = np.clip(dJ, -max_dj, max_dj)
    s.j = float(np.clip(s.j + dj, 0.0, 1.0))

    # Inner sink boundary: treat high-x orbits as "lost" and re-inject from reservoir
    if inner_sink_frac > 0.0:
        sink_threshold = (1.0 - inner_sink_frac) * X_D
        if s.x >= sink_threshold:
            if no_reservoir:
                # If no reservoir, just clip at X_D instead of re-injecting
                s.x = float(min(s.x, X_D * 0.999))
            else:
                s.x = X_BOUND
                s.j = float(np.sqrt(rng.random()))
                s.phase = 0.0
            return

    # reservoir boundary (only if reservoir is enabled)
    if not no_reservoir and s.x < X_BOUND:
        s.x = X_BOUND
        s.j = float(np.sqrt(rng.random()))

    # capture at pericenter crossings (diagnostic; re-inject immediately)
    if not disable_capture:
        captured = False
        prev = int(s.phase)
        s.phase += DT / max(Ptilde(s.x),1e-6)
        if int(s.phase) > prev and (s.j <= j_min_exact(s.x)):
            captured = True
        if captured:
            if no_reservoir:
                # If no reservoir, just clip j to minimum and continue (for testing)
                s.j = float(max(j_min_exact(s.x) * 1.01, 1e-6))
                s.phase = 0.0
            else:
                s.x = X_BOUND
                s.j = float(np.sqrt(rng.random()))
                s.phase = 0.0
    return

def run_once(steps, DT, pop, LUT, blocks=8, progress_every=0.05,
             disable_capture=False, X_FLOORS=None, n_clones=0,
             per_dlnx=True, burn_frac=0.0, init_g0=False,
             flip_energy_sign=False, inner_sink_frac=0.0, no_reservoir=False):

    if init_g0:
        stars = sample_initial_from_g0(pop)
    else:
        stars = sample_initial(pop)

    use_clones = (X_FLOORS is not None and n_clones > 0)
    
    # Determine if we should use parallel evolution
    # Only use if numba is available and we're not using clones
    # Note: --no-jit won't affect already-decorated functions, but parallel evolution
    # will still work (just slower if JIT was disabled at import time)
    use_parallel = (HAVE_NUMBA and not use_clones)
    
    clones  = []
    if use_clones:
        parents = [dict(x=s.x, j=s.j, w=1.0, phase=s.phase,
                        floor_idx=-1, x_prev=s.x) for s in stars]
        stars = None
    else:
        parents = None
        if use_parallel:
            # Convert to arrays once, before the loop
            x_arr = np.array([s.x for s in stars], dtype=np.float64)
            j_arr = np.array([s.j for s in stars], dtype=np.float64)
            phase_arr = np.array([s.phase for s in stars], dtype=np.float64)
            x_prev_arr = x_arr.copy()  # Initialize with current positions
            weights = np.ones(pop, dtype=np.float64)
            # Extract LUT grids once
            e1_grid = LUT['e1']
            e2_grid = LUT['e2']
            j1_grid = LUT['j1']
            j2_grid = LUT['j2']
            z2_grid = LUT['z2']
            x_grid = LUT['x']
            j_grid = LUT['j']
            # We don't need Star objects in parallel mode
            stars = None
        else:
            # Serial mode: keep Star objects
            x_prev = [s.x for s in stars]
            weights = [1.0]*len(stars)

    gtime_blocks = [np.zeros_like(XCEN) for _ in range(blocks)]
    t0_blocks    = np.zeros(blocks, float)
    block_edges  = np.linspace(0, steps, blocks+1, dtype=int)
    last_print   = 0.0

    for k in range(steps):
        if use_clones:
            active = parents + [c for c in clones if c.get('active', True)]
            stars  = [Star(x=obj['x'], j=obj['j'], phase=obj['phase']) for obj in active]

        # evolve
        if use_clones:
            # parents
            for p in parents:
                p['x_prev'] = p['x']
                s = Star(x=p['x'], j=p['j'], phase=p['phase'])
                mc_step(s, LUT, DT, disable_capture=disable_capture,
                        flip_energy_sign=flip_energy_sign, inner_sink_frac=inner_sink_frac,
                        no_reservoir=no_reservoir)
                p['x'], p['j'], p['phase'] = s.x, s.j, s.phase
                if not no_reservoir and p['x'] < X_BOUND:
                    p['x'] = X_BOUND
                    p['j'] = float(np.sqrt(rng.random()))
                new_clones = maybe_split(p, X_FLOORS, n_clones)
                for nc in new_clones:
                    nc['x_prev'] = nc['x']
                clones.extend(new_clones)
            # clones
            for c in clones:
                if not c.get('active', True):
                    continue
                c['x_prev'] = c['x']
                s = Star(x=c['x'], j=c['j'], phase=c['phase'])
                mc_step(s, LUT, DT, disable_capture=disable_capture,
                        flip_energy_sign=flip_energy_sign, inner_sink_frac=inner_sink_frac,
                        no_reservoir=no_reservoir)
                c['x'], c['j'], c['phase'] = s.x, s.j, s.phase
                if not no_reservoir and c['x'] < X_BOUND:
                    c['x'] = X_BOUND
                    c['j'] = float(np.sqrt(rng.random()))
                if c['floor_idx'] >= 0 and c['x'] < X_FLOORS[c['floor_idx']]:
                    c['active'] = False
                else:
                    new_clones = maybe_split(c, X_FLOORS, n_clones)
                    for nc in new_clones:
                        nc['x_prev'] = nc['x']
                    clones.extend(new_clones)
        else:
            if use_parallel:
                # Parallel evolution: x_prev_arr is set inside evolve_stars_parallel
                # to the pre-step position, so we need to save it first
                x_prev_arr[:] = x_arr[:]  # Save current positions before evolution
                
                # Evolve all stars in parallel
                seed_offset = int(rng.integers(0, 2**31))
                evolve_stars_parallel(x_arr, j_arr, phase_arr, x_prev_arr,
                                     e1_grid, e2_grid, j1_grid, j2_grid, z2_grid,
                                     x_grid, j_grid, DT,
                                     disable_capture, flip_energy_sign, inner_sink_frac, no_reservoir,
                                     seed_offset)
            else:
                # Serial evolution (fallback): use original Star objects
                x_prev_saved = x_prev.copy()
                for i, s in enumerate(stars):
                    mc_step(s, LUT, DT, disable_capture=disable_capture,
                            flip_energy_sign=flip_energy_sign, inner_sink_frac=inner_sink_frac,
                            no_reservoir=no_reservoir)
                    x_prev[i] = s.x  # Update x_prev for next iteration (though we use x_prev_saved for deposition)

        # deposit along path in log-x
        blk = int(np.searchsorted(block_edges, k, side='right') - 1)
        total_w = 0.0

        if use_clones:
            active = parents + [c for c in clones if c.get('active', True)]
            for obj in active:
                x0 = max(obj.get('x_prev', obj['x']), X_BOUND)
                x1 = max(obj['x'], X_BOUND)
                w  = obj['w']
                dt0 = DT
                dlog = abs(math.log(max(x1,1e-12)/max(x0,1e-12)))
                if dlog <= 0.25:
                    xm = math.sqrt(x0*x1)
                    xm = max(xm, X_BOUND*(1+1e-12))
                    k_idx = bidx(xm)
                    if 0 <= k_idx < len(XCEN):
                        gtime_blocks[blk][k_idx] += w*dt0
                else:
                    nsub  = min(8, 1+int(dlog/0.25))
                    ratio = max(x1,1e-12)/max(x0,1e-12)
                    for u in range(nsub):
                        xm = x0*(ratio**((u+0.5)/nsub))
                        xm = max(xm, X_BOUND*(1+1e-12))
                        k_idx = bidx(xm)
                        if 0 <= k_idx < len(XCEN):
                            gtime_blocks[blk][k_idx] += w*(dt0/nsub)
                total_w += w
        else:
            if use_parallel:
                # Parallel mode: use arrays
                x_min = X_BOUND if not no_reservoir else 1e-4
                for i in range(pop):
                    x0 = max(x_prev_arr[i], x_min)
                    x1 = max(x_arr[i], x_min)
                    w  = weights[i]
                    dt0 = DT
                    dlog = abs(math.log(max(x1, 1e-12) / max(x0, 1e-12)))
                    if dlog <= 0.25:
                        xm = math.sqrt(x0 * x1)
                        xm = max(xm, x_min * (1.0 + 1e-12))
                        k_idx = bidx(xm)
                        if 0 <= k_idx < len(XCEN):
                            gtime_blocks[blk][k_idx] += w * dt0
                    else:
                        nsub = min(8, 1 + int(dlog / 0.25))
                        ratio = max(x1, 1e-12) / max(x0, 1e-12)
                        for u in range(nsub):
                            xm = x0 * (ratio ** ((u + 0.5) / nsub))
                            xm = max(xm, x_min * (1.0 + 1e-12))
                            k_idx = bidx(xm)
                            if 0 <= k_idx < len(XCEN):
                                gtime_blocks[blk][k_idx] += w * (dt0 / nsub)
                    total_w += w
            else:
                # Serial mode: use Star objects
                for i, s in enumerate(stars):
                    x0 = max(x_prev_saved[i], X_BOUND)
                    x1 = max(s.x, X_BOUND)
                    w  = weights[i]
                    dt0 = DT
                    dlog = abs(math.log(max(x1,1e-12)/max(x0,1e-12)))
                    if dlog <= 0.25:
                        xm = math.sqrt(x0*x1)
                        xm = max(xm, X_BOUND*(1+1e-12))
                        k_idx = bidx(xm)
                        if 0 <= k_idx < len(XCEN):
                            gtime_blocks[blk][k_idx] += w*dt0
                    else:
                        nsub  = min(8, 1+int(dlog/0.25))
                        ratio = max(x1,1e-12)/max(x0,1e-12)
                        for u in range(nsub):
                            xm = x0*(ratio**((u+0.5)/nsub))
                            xm = max(xm, X_BOUND*(1+1e-12))
                            k_idx = bidx(xm)
                            if 0 <= k_idx < len(XCEN):
                                gtime_blocks[blk][k_idx] += w*(dt0/nsub)
                    total_w += w

        t0_blocks[blk] += total_w*DT

        # progress
        p = (k+1)/steps
        if p - last_print >= progress_every:
            print(f"Progress: {100*p:5.1f}% ({k+1}/{steps})", flush=True)
            last_print = p

    # per-block spectra
    g_list = []
    bin_widths = X_EDGES[1:] - X_EDGES[:-1]
    dlnx = DLNX

    # Calculate burn-in: skip early blocks when averaging
    burn_blocks = int(burn_frac * blocks)
    burn_blocks = max(0, min(burn_blocks, blocks - 1))  # ensure at least 1 block remains

    for b in range(burn_blocks, blocks):
        tmp = gtime_blocks[b] / max(t0_blocks[b], 1e-30)
        if per_dlnx:
            g_b = np.divide(tmp, dlnx, out=np.zeros_like(tmp), where=(dlnx>0))
        else:
            g_b = np.divide(tmp, bin_widths, out=np.zeros_like(tmp), where=(bin_widths>0))
        g_list.append(g_b)

    if len(g_list) == 0:
        # Fallback: if all blocks were burned, use the last block
        tmp = gtime_blocks[-1] / max(t0_blocks[-1], 1e-30)
        if per_dlnx:
            g_b = np.divide(tmp, dlnx, out=np.zeros_like(tmp), where=(dlnx>0))
        else:
            g_b = np.divide(tmp, bin_widths, out=np.zeros_like(tmp), where=(bin_widths>0))
        g_list.append(g_b)

    g_arr  = np.stack(g_list, axis=0)
    g_mean = np.mean(g_arr, axis=0)
    nb     = g_arr.shape[0]
    g_var  = np.var(g_arr, axis=0, ddof=1) if nb>1 else np.zeros_like(g_mean)
    g_err  = np.sqrt(np.maximum(g_var/nb, 0.0))
    
    if burn_blocks > 0:
        print(f"  (Averaged over blocks {burn_blocks} to {blocks-1}, discarding {burn_blocks} early blocks)")
    
    return g_mean, g_err

# ----------------- safer background update -----------------
def build_updated_background_from_measure(bg0: Background,
                                          g_mean, g_err,
                                          snr_max=1.0,
                                          x_min=0.23,
                                          x_max=50.0):
    """
    Build a *gentle* updated background:

    - Work with ratio R = g_meas / g0.
    - Smooth log R.
    - Clamp 0.5 <= R <= 2 so diffusion never collapses.
    - Outside [x_min,x_max] or bad-SNR bins, keep R=1.
    """
    g_mean = np.asarray(g_mean)
    g_err  = np.asarray(g_err)
    x      = XCEN

    g0c = g0_interp(x)
    R_raw = np.ones_like(g_mean)

    good = (g_mean > 0) & (g0c > 0) & (g_err > 0)
    good &= (g_err/g_mean < snr_max)
    good &= (x >= x_min) & (x <= x_max)

    if np.count_nonzero(good) < 3:
        print("Background update: not enough good bins; keeping original g0.")
        # rebuild background purely from g0
        xbg = np.logspace(-3, 4, 400)
        return Background(xs=xbg, gs=g0_interp(xbg))

    R_raw[good] = g_mean[good] / np.maximum(g0c[good], 1e-16)

    # smooth log R across *all* bins (good and not, but R=1 where not good)
    R_raw = np.clip(R_raw, 0.2, 5.0)
    R_s   = smooth_log(x, R_raw, k=3)
    R_s   = np.clip(R_s, 0.5, 2.0)

    # interpolate smoothed R onto fine grid and build new background
    xbg = np.logspace(-3, 4, 400)
    lx_bg = np.log(xbg)
    lx_c  = np.log(x)
    lR_s  = np.log(R_s)
    lR_bg = np.interp(lx_bg, lx_c, lR_s)
    R_bg  = np.clip(np.exp(lR_bg), 0.5, 2.0)

    g0bg = g0_interp(xbg)
    gbg  = np.maximum(g0bg * R_bg, 1e-16)

    print("Background update: using ratio-smoothed background (0.5–2 × g0).")
    return Background(xs=xbg, gs=gbg)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=None,
                    help="if None, dt = 6/steps (one 6 t0 iteration)")
    ap.add_argument("--lut-x", type=int, default=40)
    ap.add_argument("--lut-j", type=int, default=28)
    ap.add_argument("--pop",   type=int, default=256)
    ap.add_argument("--blocks",type=int, default=8)
    ap.add_argument("--normalize-225", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--disable-capture", action="store_true",
                    help="disable capture during measurement (tail diagnostics); keep OFF for central curve")
    ap.add_argument("--floors", type=int, default=0,
                    help="number of log-x floors for importance scheme (0=off)")
    ap.add_argument("--clones", type=int, default=0,
                    help="number of clones per split (0=off, set to 2 to enable)")
    ap.add_argument("--xmax-floor", type=float, default=80.0,
                    help="top of the floor ladder")
    ap.add_argument("--bin-locked-floors", action="store_true",
                    help="use Fig. 3 bin centers as floor rungs")
    ap.add_argument("--no-per-dlnx", dest="per_dlnx", action="store_false", default=True,
                    help="use per-Δx normalization instead of per-Δln x (diagnostic mode). Default: per-Δln x.")
    ap.add_argument("--background-update", action="store_true",
                    help="enable 2-pass background update (gentle ratio-smoothing)")
    ap.add_argument("--burn-frac", type=float, default=0.5,
                    help="fraction of blocks to discard as burn-in when averaging ḡ (default: 0.5)")
    ap.add_argument("--init-g0", action="store_true",
                    help="initialize stars from g0(x) instead of the outer reservoir")
    ap.add_argument("--flip-energy-sign", action="store_true",
                    help="flip sign of energy drift (diagnostic: test if e1 convention is reversed)")
    ap.add_argument("--inner-sink-frac", type=float, default=0.0,
                    help="fraction of X_D to use as inner sink threshold (0=off, 0.1 means sink at 0.9*X_D)")
    ap.add_argument("--save-ratio", action="store_true",
                    help="save R(x) = g_MC(x) / g0(x) diagnostic to gbar_ratio.csv")
    ap.add_argument("--no-reservoir", action="store_true",
                    help="do not enforce outer reservoir at x = X_BOUND (for g0 stationarity tests)")
    ap.add_argument("--procc", type=int, default=1,
                    help="number of worker processes for LUT precomputation (1=serial)")
    ap.add_argument("--no-jit", dest="use_jit", action="store_false", default=True,
                    help="disable numba JIT even if available")
    ap.add_argument("--seed", type=int, default=7,
                    help="random seed for reproducibility")
    ap.add_argument("--check-fp-consistency", action="store_true",
                    help="check if g0 is a stationary solution of the FP operator (diagnostic)")
    args = ap.parse_args()

    # Set multiprocessing start method for Windows compatibility (before using ProcessPoolExecutor)
    if args.procc > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass

    global rng, njit, fastmath
    rng = np.random.default_rng(args.seed)

    # Update JIT settings based on --no-jit flag
    use_jit = getattr(args, "use_jit", True) and HAVE_NUMBA
    njit = nb.njit if use_jit else nb_njit
    fastmath = dict(fastmath=True, nogil=True, cache=True) if use_jit else {}
    if use_jit and HAVE_NUMBA:
        # Set numba to use all available CPU threads for parallel execution
        import os
        num_threads = int(os.environ.get('NUMBA_NUM_THREADS', mp.cpu_count()))
        nb.set_num_threads(num_threads)
        print(f"Using numba JIT compilation with {num_threads} threads for parallel execution")
    elif HAVE_NUMBA:
        print(f"Numba available but JIT disabled (--no-jit)")
    else:
        print(f"Numba not available, running without JIT")

    DT = (6.0/args.steps) if args.dt is None else float(args.dt)

    # initial background from BWII g0
    xgrid = np.logspace(-3, 4, 400)
    bg0   = Background(xs=xgrid, gs=g0_interp(xgrid))
    LUT   = precompute_LUT(args.lut_x, args.lut_j, bg0, n_procs=args.procc)

    # FP consistency check (diagnostic)
    if args.check_fp_consistency:
        print("\n" + "="*70)
        print("FP Consistency Check: Is g0 a stationary solution?")
        print("="*70)
        fp_diag = check_fp_consistency(LUT, bg0)
        
        x_fp = fp_diag['x']
        residual = fp_diag['residual']
        rel_residual = fp_diag['relative_residual']
        D_E = fp_diag['D_E']
        D_EE = fp_diag['D_EE']
        g0_fp = fp_diag['g0']
        
        # Find regions where residual is significant
        abs_rel_res = np.abs(rel_residual)
        significant = abs_rel_res > 0.1  # >10% relative error
        
        print(f"\nResidual statistics:")
        print(f"  Mean |relative residual|: {np.mean(abs_rel_res):.4f}")
        print(f"  Median |relative residual|: {np.median(abs_rel_res):.4f}")
        print(f"  Max |relative residual|: {np.max(abs_rel_res):.4f} (at x = {x_fp[np.argmax(abs_rel_res)]:.2f})")
        print(f"  Bins with |relative residual| > 0.1: {np.sum(significant)} / {len(x_fp)}")
        
        if np.sum(significant) > 0:
            print(f"\n  Regions with significant residual:")
            for ix in np.where(significant)[0][:10]:  # Show first 10
                print(f"    x = {x_fp[ix]:8.2f}: |rel_res| = {abs_rel_res[ix]:.4f}, "
                      f"D_E = {D_E[ix]:.4e}, D_EE = {D_EE[ix]:.4e}, g0 = {g0_fp[ix]:.4e}")
        
        # Show breakdown of terms for a few representative points
        print(f"\n  Breakdown of FP operator terms (showing largest residuals):")
        top_ix = np.argsort(abs_rel_res)[-5:][::-1]  # Top 5 worst
        for ix in top_ix:
            print(f"    x = {x_fp[ix]:8.2f}:")
            print(f"      L[g0] = {residual[ix]:.4e} (rel = {abs_rel_res[ix]:.4f})")
            print(f"        drift: e1*dg/dx = {fp_diag['drift_term'][ix]:.4e}")
            print(f"        drift_coeff: g*de1/dx = {fp_diag['drift_coeff_term'][ix]:.4e}")
            print(f"        diff: 0.5*e2*d²g/dx² = {fp_diag['diff_term'][ix]:.4e}")
            print(f"        diff_mixed: de2/dx*dg/dx = {fp_diag['diff_mixed_term'][ix]:.4e}")
            print(f"        diff_coeff: 0.5*g*d²e2/dx² = {fp_diag['diff_coeff_term'][ix]:.4e}")
        
        # Save diagnostic CSV with all terms
        arr_fp = np.column_stack([
            x_fp, D_E, D_EE, g0_fp, fp_diag['dg0_dx'], fp_diag['d2g0_dx2'],
            fp_diag['dD_E_dx'], fp_diag['dD_EE_dx'],
            residual, rel_residual,
            fp_diag['drift_term'], fp_diag['drift_coeff_term'],
            fp_diag['diff_term'], fp_diag['diff_mixed_term'], fp_diag['diff_coeff_term']
        ])
        np.savetxt("fp_consistency.csv", arr_fp, delimiter=",",
                   header="x,D_E,D_EE,g0,dg0_dx,d2g0_dx2,dD_E_dx,dD_EE_dx,residual,relative_residual,"
                          "drift_term,drift_coeff_term,diff_term,diff_mixed_term,diff_coeff_term", 
                   comments="")
        print(f"\n  Saved FP consistency diagnostic to fp_consistency.csv")
        print("="*70 + "\n")

    # build floors
    X_FLOORS = None
    if args.floors > 0 and args.clones > 0:
        if args.bin_locked_floors:
            valid_centers = X_F3[(X_F3 >= X_BOUND) & (X_F3 <= args.xmax_floor)]
            if len(valid_centers) > 0:
                floors_list = [valid_centers[0]]
                for i in range(len(valid_centers)-1):
                    mid = math.sqrt(valid_centers[i]*valid_centers[i+1])
                    if mid > floors_list[-1]*1.01:
                        floors_list.append(mid)
                    floors_list.append(valid_centers[i+1])
                X_FLOORS = np.array(floors_list)
                if len(X_FLOORS) > args.floors > 0:
                    idx = np.linspace(0, len(X_FLOORS)-1, args.floors, dtype=int)
                    X_FLOORS = X_FLOORS[idx]
                print(f"Using {len(X_FLOORS)} bin-locked floors (from Fig. 3 centers), clones={args.clones}")
            else:
                X_FLOORS = np.geomspace(X_BOUND*(1+1e-9), args.xmax_floor, args.floors)
                print(f"Using {args.floors} log-x floors from {X_BOUND:.3f} to {args.xmax_floor:.1f}, clones={args.clones}")
        else:
            X_FLOORS = np.geomspace(X_BOUND*(1+1e-9), args.xmax_floor, args.floors)
            print(f"Using {args.floors} log-x floors from {X_BOUND:.3f} to {args.xmax_floor:.1f}, clones={args.clones}")

    use_per_dlnx = getattr(args, 'per_dlnx', True)
    if use_per_dlnx:
        print("Using per-Δln x normalization (paper-like)")
    else:
        print("Using per-Δx normalization (diagnostic)")

    # ----- Pass 1: baseline measurement -----
    print("Pass 1: Measuring with BWII g0 background...")
    if args.init_g0:
        print("  Initializing stars from g0(x) distribution")
    else:
        print("  Initializing stars at outer reservoir (x = X_BOUND)")
    if args.flip_energy_sign:
        print("  WARNING: Energy drift sign is FLIPPED (diagnostic mode)")
    if args.inner_sink_frac > 0.0:
        sink_thresh = (1.0 - args.inner_sink_frac) * X_D
        print(f"  Inner sink active: stars at x >= {sink_thresh:.1f} re-inject to reservoir")
    if args.no_reservoir:
        print("  WARNING: Outer reservoir is DISABLED (for g0 stationarity tests)")
    g_mean, g_err = run_once(args.steps, DT, args.pop, LUT,
                             blocks=args.blocks,
                             disable_capture=args.disable_capture,
                             X_FLOORS=X_FLOORS, n_clones=args.clones,
                             per_dlnx=use_per_dlnx,
                             burn_frac=args.burn_frac,
                             init_g0=args.init_g0,
                             flip_energy_sign=args.flip_energy_sign,
                             inner_sink_frac=args.inner_sink_frac,
                             no_reservoir=args.no_reservoir)

    # ----- Optional background update -----
        # Optional: background update (paper's 2-pass style)
    if args.background_update:
        print("Updating background from measured ḡ...")

        # Original BWII background at the Fig. 3 bin centers
        g0_cen = g0_interp(XCEN)
        g0_cen = np.maximum(g0_cen, 1e-16)

        # Signal-to-noise for the Pass-1 measurement
        snr = np.zeros_like(g_mean)
        good = (g_mean > 0) & (g_err > 0)
        snr[good] = g_mean[good] / g_err[good]

        # Only trust bins with reasonable S/N and in a central-x window
        # (you can tweak x_min/x_max or the SNR cut if you like)
        x_min, x_max = 0.23, 50.0
        use = good & (snr >= 1.0) & (XCEN >= x_min) & (XCEN <= x_max)

        # Raw ratio R_raw = ḡ^(1) / g0, defaulting to 1 outside "use"
        R_raw = np.ones_like(g_mean)
        R_raw[use] = g_mean[use] / g0_cen[use]

        # Smooth the ratio in log-space, not the absolute ḡ
        R_smooth = smooth_log(XCEN, np.maximum(R_raw, 1e-3), k=3)

        # Clamp to a modest range so we don't over-react to noise
        R_smooth = np.clip(R_smooth, 0.5, 2.0)

        # Build new background: g_bg(x) = R(x) * g0(x)
        xbg = np.logspace(-3, 4, 400)
        lx_bg = np.log(xbg)
        lx_cen = np.log(XCEN)

        # Interpolate ln R to the fine grid
        lnR_cen = np.log(R_smooth)
        lnR_bg = np.interp(lx_bg, lx_cen, lnR_cen)
        R_bg = np.exp(lnR_bg)

        # Final updated background
        gbg = g0_interp(xbg) * R_bg

        # Rebuild LUT on updated background
        bg1 = Background(xs=xbg, gs=gbg)
        print("Rebuilding LUT on updated background...")
        LUT = precompute_LUT(args.lut_x, args.lut_j, bg1, n_procs=args.procc)

        # Pass 2: measure again with updated background
        print("Pass 2: Measuring with updated background...")
        g_mean, g_err = run_once(
            args.steps, DT, args.pop, LUT, blocks=args.blocks,
                                 disable_capture=args.disable_capture,
                                 X_FLOORS=X_FLOORS, n_clones=args.clones,
            per_dlnx=use_per_dlnx,
            burn_frac=args.burn_frac,
            init_g0=args.init_g0,
            flip_energy_sign=args.flip_energy_sign,
            inner_sink_frac=args.inner_sink_frac,
            no_reservoir=args.no_reservoir
        )


    # final normalization at x≈0.225 if requested
    if args.normalize_225:
        idx_225 = int(np.argmin(np.abs(XCEN - 0.225)))
        s = g_mean[idx_225] if g_mean[idx_225] > 0 else 1.0
        g_mean /= s; g_err /= s

    # Diagnostic: compute R(x) = g_MC(x) / g0(x) (always compute, optionally save)
    g0_vals = g0_interp(XCEN)
    g0_vals = np.maximum(g0_vals, 1e-16)
    R = g_mean / g0_vals
    R_finite = R[np.isfinite(R) & (R > 0)]
    
    if len(R_finite) > 0:
        print(f"\nDiagnostic: g_MC / g0 ratio statistics:")
        print(f"  Mean ratio: {np.mean(R_finite):.4f}")
        print(f"  Median ratio: {np.median(R_finite):.4f}")
        print(f"  Min ratio: {np.min(R_finite):.4f} (at x = {XCEN[np.argmin(R_finite)]:.2f})")
        print(f"  Max ratio: {np.max(R_finite):.4f} (at x = {XCEN[np.argmax(R_finite)]:.2f})")
        print(f"  Ratio range: {np.max(R_finite)/np.min(R_finite):.2f}x")
        if args.no_reservoir and args.disable_capture and args.init_g0:
            print(f"  (For g0 stationarity test: ratio should be ~constant if operator is self-consistent)")

    # Check for missing bins (zero deposition)
    missing_bins = (g_mean == 0.0) | (g_err == 0.0)
    if np.any(missing_bins):
        n_missing = np.sum(missing_bins)
        print(f"\nWarning: {n_missing} / {len(XCEN)} bins have zero deposition (g_mean=0 or g_err=0):")
        for idx in np.where(missing_bins)[0][:10]:  # Show first 10
            x_cen = XCEN[idx]
            # Find bin edges
            if idx == 0:
                x_left = X_EDGES[0]
                x_right = X_EDGES[1]
            elif idx == len(XCEN) - 1:
                x_left = X_EDGES[-2]
                x_right = X_EDGES[-1]
            else:
                x_left = X_EDGES[idx]
                x_right = X_EDGES[idx + 1]
            print(f"  x = {x_cen:.2f} (bin [{x_left:.3f}, {x_right:.3f}]): "
                  f"g_mean = {g_mean[idx]:.4e}, g_err = {g_err[idx]:.4e}")
        if n_missing > 10:
            print(f"  ... and {n_missing - 10} more")
        print(f"  These bins may indicate binning issues or regions with no star visits.")
        print(f"  Check bidx() logic and deposition path integration for these bins.")
    else:
        print(f"\nAll {len(XCEN)} bins have non-zero deposition (good).")
    
    # Additional diagnostic: check bin edge coverage
    print(f"\nBin coverage diagnostic:")
    print(f"  Number of bins: {len(XCEN)}")
    print(f"  Bin centers range: [{XCEN[0]:.3f}, {XCEN[-1]:.1f}]")
    print(f"  Bin edges range: [{X_EDGES[0]:.3f}, {X_EDGES[-1]:.1f}]")
    # Check for any suspicious gaps
    edge_gaps = np.diff(X_EDGES)
    large_gaps = edge_gaps > 2.0 * np.median(edge_gaps)
    if np.any(large_gaps):
        print(f"  Warning: Found {np.sum(large_gaps)} bins with unusually large width:")
        for idx in np.where(large_gaps)[0]:
            print(f"    Bin {idx}: width = {edge_gaps[idx]:.3f} (center = {XCEN[idx]:.2f})")

    # write CSV
    arr = np.column_stack([XCEN, g_mean, g_err])
    np.savetxt("gbar_table.csv", arr, delimiter=",",
               header="x,gbar,gbar_err", comments="")
    print("Wrote gbar_table.csv")

    # Save ratio diagnostic if requested
    if args.save_ratio:
        # Renormalize R so R at x=0.225 is 1 (for comparison with paper)
        idx_225 = int(np.argmin(np.abs(XCEN - 0.225)))
        if R[idx_225] > 0:
            R_norm = R / R[idx_225]
        else:
            R_norm = R.copy()
        arr_ratio = np.column_stack([XCEN, R, R_norm, g0_vals])
        np.savetxt("gbar_ratio.csv", arr_ratio, delimiter=",",
                   header="x,R_raw,R_norm,g0", comments="")
        print("Wrote gbar_ratio.csv (R = g_MC / g0, normalized at x=0.225)")

    # plot
    if args.out:
        xcurve = np.logspace(-1, 4, 600)
        g0c    = g0_interp(xcurve)
        plt.figure(figsize=(6.2,6), dpi=140)
        plt.xscale("log"); plt.yscale("log")
        plt.xlim(1e-1, 1e4)

        positive = g_mean > 0
        if np.any(positive):
            ymin = max(np.min(g_mean[positive])*0.6, 1e-4)
            ymax = max(np.max(g_mean)*2, 1.0)
        else:
            ymin, ymax = 1e-4, 10.0
        plt.ylim(ymin, ymax)

        plt.plot(xcurve, g0c, lw=2, label=r"$g_0$ (BW II)")
        plt.errorbar(XCEN, g_mean, yerr=g_err, fmt="o", ms=4, capsize=2,
                     label=r"$\bar g$")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\bar g(x)$")
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out)
        print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
