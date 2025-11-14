#!/usr/bin/env python3
# Stable MC for isotropized ḡ(x) — canonical case (x_D=1e4, P*=0.005)
# - Constant-population SMC each step (exactly N=--pop stars, equal weights)
# - Time deposition: add w*DT to current log-x bin; divide by total time
# - No per-block renormalization to g0; optional final normalize to ḡ(0.225)=1
# - One LUT from BWII g0; errors from block averages (no distortions)

import math, argparse, numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import cumulative_trapezoid

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
def j_min_exact(x):
    val = 2.0*x/X_D - (x*x)/(X_D*X_D)
    return math.sqrt(max(0.0, min(1.0, val)))

def Ptilde(x):  # ∝ x^{-3/2}, only for pericenter counting
    return max(1e-3, x**(-1.5))

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

def pericenter_xp(x, j):
    e = math.sqrt(max(0.0, 1.0 - j*j))
    return 2.0*x / max(1e-12, 1.0 - e)

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
def precompute_LUT(nx, nj, bg: Background):
    Xg = np.logspace(-3, 4, nx)
    Jg = np.linspace(0.0, 1.0, nj)
    E1 = np.zeros((nx, nj)); E2 = np.zeros_like(E1)
    J1 = np.zeros_like(E1);   J2 = np.zeros_like(E1)
    Z2 = np.zeros_like(E1)
    print(f"Precomputing LUT {nx}x{nj} ...")
    for ix,x in enumerate(Xg):
        for ij,j in enumerate(Jg):
            e1,e2,j1,j2,z2 = orbital_perturbations(x,j,bg)
            E1[ix,ij]=e1; E2[ix,ij]=e2
            J1[ix,ij]=j1; J2[ix,ij]=j2
            Z2[ix,ij]=z2
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

def mc_step(s: Star, L, DT, disable_capture=False):
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

    s.x = float(np.clip(s.x - np.clip(dE, -max_dx, max_dx), 1e-4, X_D))
    dj  = np.clip(dJ, -max_dj, max_dj)
    s.j = float(np.clip(s.j + dj, 0.0, 1.0))

    # reservoir boundary
    if s.x < X_BOUND:
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
            s.x = X_BOUND
            s.j = float(np.sqrt(rng.random()))
            s.phase = 0.0
    return

def run_once(steps, DT, pop, LUT, blocks=8, progress_every=0.05,
             disable_capture=False, X_FLOORS=None, n_clones=0,
             per_dlnx=True):

    stars = sample_initial(pop)

    use_clones = (X_FLOORS is not None and n_clones > 0)
    clones  = []
    if use_clones:
        parents = [dict(x=s.x, j=s.j, w=1.0, phase=s.phase,
                        floor_idx=-1, x_prev=s.x) for s in stars]
        stars = None
    else:
        parents = None
        x_prev = [s.x for s in stars]

    gtime_blocks = [np.zeros_like(XCEN) for _ in range(blocks)]
    t0_blocks    = np.zeros(blocks, float)
    block_edges  = np.linspace(0, steps, blocks+1, dtype=int)
    last_print   = 0.0

    for k in range(steps):
        if use_clones:
            active = parents + [c for c in clones if c.get('active', True)]
            stars  = [Star(x=obj['x'], j=obj['j'], phase=obj['phase']) for obj in active]
            weights = [obj['w'] for obj in active]
        else:
            weights = [1.0]*len(stars)

        # evolve
        if use_clones:
            # parents
            for p in parents:
                p['x_prev'] = p['x']
                s = Star(x=p['x'], j=p['j'], phase=p['phase'])
                mc_step(s, LUT, DT, disable_capture=disable_capture)
                p['x'], p['j'], p['phase'] = s.x, s.j, s.phase
                if p['x'] < X_BOUND:
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
                mc_step(s, LUT, DT, disable_capture=disable_capture)
                c['x'], c['j'], c['phase'] = s.x, s.j, s.phase
                if c['x'] < X_BOUND:
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
            x_prev_saved = x_prev.copy()
            for i,s in enumerate(stars):
                mc_step(s, LUT, DT, disable_capture=disable_capture)
                x_prev[i] = s.x

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
            for i,s in enumerate(stars):
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

    for b in range(blocks):
        tmp = gtime_blocks[b] / max(t0_blocks[b], 1e-30)
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
    ap.add_argument("--seed", type=int, default=7,
                    help="random seed for reproducibility")
    args = ap.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    DT = (6.0/args.steps) if args.dt is None else float(args.dt)

    # initial background from BWII g0
    xgrid = np.logspace(-3, 4, 400)
    bg0   = Background(xs=xgrid, gs=g0_interp(xgrid))
    LUT   = precompute_LUT(args.lut_x, args.lut_j, bg0)

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
    g_mean, g_err = run_once(args.steps, DT, args.pop, LUT,
                             blocks=args.blocks,
                             disable_capture=args.disable_capture,
                             X_FLOORS=X_FLOORS, n_clones=args.clones,
                             per_dlnx=use_per_dlnx)

    # ----- Optional background update -----
    if args.background_update:
        print("Updating background from measured ḡ (gentle ratio method)...")
        bg1 = build_updated_background_from_measure(bg0, g_mean, g_err,
                                                    snr_max=1.0,
                                                    x_min=0.23,
                                                    x_max=50.0)
        print("Rebuilding LUT on updated background...")
        LUT = precompute_LUT(args.lut_x, args.lut_j, bg1)

        print("Pass 2: Measuring with updated background...")
        g_mean, g_err = run_once(args.steps, DT, args.pop, LUT,
                                 blocks=args.blocks,
                                 disable_capture=args.disable_capture,
                                 X_FLOORS=X_FLOORS, n_clones=args.clones,
                                 per_dlnx=use_per_dlnx)

    # final normalization at x≈0.225 if requested
    if args.normalize_225:
        idx_225 = int(np.argmin(np.abs(XCEN - 0.225)))
        s = g_mean[idx_225] if g_mean[idx_225] > 0 else 1.0
        g_mean /= s; g_err /= s

    # write CSV
    arr = np.column_stack([XCEN, g_mean, g_err])
    np.savetxt("gbar_table.csv", arr, delimiter=",",
               header="x,gbar,gbar_err", comments="")
    print("Wrote gbar_table.csv")

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
