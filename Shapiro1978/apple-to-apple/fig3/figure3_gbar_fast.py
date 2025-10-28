#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Monte-Carlo for the isotropized distribution function ḡ(x)
(canonical case x_D=1e4, P*=0.005) with creation–annihilation,
exact loss-cone boundary (checked at pericenter), and a precomputed
coefficient LUT for speed. Two iterations are performed:
  Iter0 field-star background = BWII g0(x)
  Iter1 field-star background = smoothed measured ḡ(x) from Iter0
Outputs:
  - figure_gbar.png  (g0 vs ḡ with 1σ bars)
  - gbar_table.csv   (x, ḡ, σ)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import cumulative_trapezoid

# ------------------------ parameters -------------------------
SEED      = 7
rng       = np.random.default_rng(SEED)

# Canonical inputs
X_D    = 1.0e4        # outer energy cutoff
P_STAR = 5.0e-3       # coupling
TAU_IT = 6.0          # one iteration spans 6 t0

# Runtime profile
FAST_DEMO = True      # True: ~1.2e3 steps; False: 1e4 steps (historical)
N_STEPS   = 1200 if FAST_DEMO else 10_000
DT        = TAU_IT / N_STEPS

# Creation–annihilation & population controls
TARGET_MEAN     = 150
MAX_POP         = 450
CLONE_FACTOR    = 5          # per inward boundary crossing (canonical run)
MAX_CLONE_DEPTH = 3          # clones do not clone beyond this
U_BOUNDS = 10.0**(-np.arange(0,6))       # u_i = 10^{-i}
X_BOUNDS = U_BOUNDS**(-4.0/5.0)          # x_i = 10^{0.8 i}

# Bins for ḡ(x)
XBINS = np.logspace(-1, 4, 60)
XCEN  = np.sqrt(XBINS[:-1] * XBINS[1:])
DLNX  = np.log(XBINS[1:] / XBINS[:-1])

# ---------------------- BWII g0(x) curve ----------------------
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
_Xg0 = np.exp(_ln1p) - 1.0
_mask = (_g0tab > 0) & (_Xg0 > 0)
_lx, _lg = np.log10(_Xg0[_mask]), np.log10(_g0tab[_mask])
_slope   = np.diff(_lg) / np.diff(_lx)

def _loglin_interp(x, xk, yk):
    """Simple log-x linear interpolant (x>0)."""
    x = np.asarray(x)
    lx = np.log10(np.clip(x, 1e-12, None))
    idx = np.clip(np.searchsorted(xk, lx) - 1, 0, len(xk)-2)
    t = (lx - xk[idx]) / np.maximum(xk[idx+1] - xk[idx], 1e-30)
    return yk[idx] * (1 - t) + yk[idx+1] * t

def g0_interp(x):
    x = np.asarray(x)
    lt = np.log10(np.clip(x, 1e-3, X_D))
    y  = np.empty_like(lt)
    left  = lt < _lx[0]
    right = lt > _lx[-1]
    mid   = ~(left | right)
    y[left]  = _lg[0]  + _slope[0]  * (lt[left]  - _lx[0])
    y[right] = _lg[-1] + _slope[-1] * (lt[right] - _lx[-1])
    idx      = np.clip(np.searchsorted(_lx, lt[mid]) - 1, 0, len(_slope)-1)
    y[mid]   = _lg[idx] + _slope[idx] * (lt[mid] - _lx[idx])
    out = 10.0**y
    # gentle taper near X_D
    x_last = _Xg0[_mask][-1]
    tmask  = (x >= x_last) & (x <= X_D)
    if tmask.any():
        g_start = out[tmask][0]
        out[tmask] = np.maximum(g_start * (1 - (x[tmask]-x_last)/(X_D-x_last)), 1e-16)
    return out

# ---------------- Kepler helpers & loss-cone ------------------
def pericenter_xp(x, j):
    e = math.sqrt(max(0.0, 1.0 - j*j))
    return 2.0*x / max(1e-12, 1.0 - e)

def apocenter_xap(x, j):
    e = math.sqrt(max(0.0, 1.0 - j*j))
    return 2.0*x / (1.0 + e)

def j_min_exact(x):
    val = 2.0*x/X_D - (x*x)/(X_D*X_D)
    return math.sqrt(max(0.0, min(1.0, val)))

# dimensionless orbital period proxy (normalized so P~1 at x~1)
def Ptilde(x):  # proportional to x^{-3/2}
    return max(1e-3, x**(-1.5))

# --------------- θ-quadrature (small, fast) -------------------
def th_nodes(n=12):
    t, w = np.polynomial.legendre.leggauss(n)
    th = 0.25*np.pi*(t + 1.0)
    wt = 0.25*np.pi*w
    s, c = np.sin(th), np.cos(th)
    return th, wt, s, c
TH, WT, S, C = th_nodes(12)

# ---------------- Appendix A kernels (fast) -------------------
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

def I1(x):  return 0.25*np.pi/(x**1.5)
def I4(x):  return 0.25*np.pi/(x**0.5)

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

# ------------- field-star background & moments ----------------
class Background:
    """Holds g_bg(x), its log-interpolant, and two cumulative moments for x'<x."""
    def __init__(self, xs, gs):
        # store on a clean log grid spanning [1e-3, 1e4]
        self.xg = np.logspace(-3, 4, 400)
        # smooth gs by simple log-domain moving average
        lx = np.log(self.xg); lx_in = np.log(np.clip(xs,1e-12,None))
        glog = np.log(np.clip(np.interp(lx, lx_in, np.log(np.clip(gs,1e-12,None))), 1e-12, None))
        k = 7
        ker = np.exp(-0.5*((np.arange(-k,k+1)/2.0)**2))
        ker /= ker.sum()
        glog_s = np.convolve(glog, ker, mode='same')
        self.gg = np.exp(glog_s)
        # cumulative moments ∫ g x'^(-3/2) dx', ∫ g x'^(-1/2) dx'
        self.Mm32 = cumulative_trapezoid(self.gg * self.xg**(-1.5), self.xg, initial=0.0)
        self.Mm12 = cumulative_trapezoid(self.gg * self.xg**(-0.5), self.xg, initial=0.0)

    def g(self, x):
        # log-x linear interpolation on (self.xg, self.gg)
        x = np.asarray(x)
        idx = np.clip(np.searchsorted(self.xg, x)-1, 0, len(self.xg)-2)
        x0, x1 = self.xg[idx], self.xg[idx+1]
        g0, g1 = self.gg[idx], self.gg[idx+1]
        t = (x - x0)/np.maximum(x1-x0, 1e-30)
        return np.maximum(g0*(1-t) + g1*t, 1e-16)

    def moment(self, x, which):
        # piecewise linear in x (robust)
        M = self.Mm32 if which=='m32' else self.Mm12
        x = np.asarray(x)
        idx = np.clip(np.searchsorted(self.xg, x)-1, 0, len(self.xg)-2)
        x0, x1 = self.xg[idx], self.xg[idx+1]
        m0, m1 = M[idx], M[idx+1]
        t = (x - x0)/np.maximum(x1-x0, 1e-30)
        return m0*(1-t) + m1*t

# -------- integrate over x' using tiny x'-quadrature -----------
T12, W12 = np.polynomial.legendre.leggauss(12)

def _int_region(fun_tuple, x, j, xp, xap, bg: Background):
    val = 0.0
    # (1) x' < x : use closed-form kernels + background moments where possible
    if fun_tuple[0] is I1:
        val += np.pi*0.25 * bg.moment(x, 'm32')
    elif fun_tuple[0] is I4:
        val += np.pi*0.25 * bg.moment(x, 'm12')
    else:
        lo, hi = 1e-4, x
        if hi > lo*(1+1e-10):
            llo, lhi = math.log(lo), math.log(hi)
            lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
            xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
            vals = np.array([fun_tuple[0](x,j,xp,xap,xp2) * bg.g(xp2) for xp2 in xs])
            val += float(np.sum(vals * jac * W12))
    # (2) x ≤ x' < x_ap
    lo, hi = x, min(xap, X_D)
    if hi > lo*(1+1e-10):
        llo, lhi = math.log(lo), math.log(hi)
        lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
        xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
        vals = np.array([fun_tuple[1](x,j,xp,xap,xp2) * bg.g(xp2) for xp2 in xs])
        val += float(np.sum(vals * jac * W12))
    # (3) x_ap ≤ x' ≤ x_p
    lo, hi = max(xap, x*(1+1e-12)), min(xp, X_D)
    if hi > lo*(1+1e-10):
        llo, lhi = math.log(lo), math.log(hi)
        lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
        xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
        vals = np.array([fun_tuple[2](x,j,xp,xap,xp2) * bg.g(xp2) for xp2 in xs])
        val += float(np.sum(vals * jac * W12))
    return val

def orbital_perturbations(x, j, bg: Background):
    xp, xap = pericenter_xp(x, j), apocenter_xap(x, j)
    e1 = 3.0*np.sqrt(2*np.pi)*P_STAR * _int_region(
        (lambda *args: I1(args[-1]), I2, I3), x, j, xp, xap, bg)
    e2 = 4.0*np.sqrt(2*np.pi)*P_STAR * _int_region(
        (lambda *args: I4(args[-1]), I6, I6), x, j, xp, xap, bg)

    if j > 0.95:
        j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
            (lambda *args: 2*I7(x,j,xp),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I10(x,j,xp,xap,args[-1]),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I11(x,j,xp,xap,args[-1])), x,j,xp,xap,bg)
        j2 = max(e2/(2*x), 0.0)
        z2 = max(min(e2*j2, e2*j2+1e-22), 0.0)
        return e1, max(e2,0.0), j1, j2, z2

    if j < 0.05:
        j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
            (lambda *args: 2*I7(x,j,xp),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I10(x,j,xp,xap,args[-1]),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I11(x,j,xp,xap,args[-1])), x,j,xp,xap,bg)
        j2 = math.sqrt(max(2*j*j1, 0.0))
        z2 = min(max(e2,0.0)*j2, max(e2,0.0)*j2+1e-22)
        return e1, max(e2,0.0), j1, j2, max(z2,0.0)

    j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
        (lambda *args: 2*I7(x,j,xp),
         lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I10(x,j,xp,xap,args[-1]),
         lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I11(x,j,xp,xap,args[-1])), x,j,xp,xap,bg)

    j2 = np.sqrt(2*np.pi)*x*P_STAR * _int_region(
        (lambda *args: 4*I7(x,j,xp),
         lambda *args: 3*I12(x,j,xp,xap,args[-1])+4*I10(x,j,xp,xap,args[-1])-3*I13(x,j,xp,xap,args[-1]),
         lambda *args: 3*I12(x,j,xp,xap,args[-1])+4*I11(x,j,xp,xap,args[-1])-3*I14(x,j,xp,xap,args[-1])), x,j,xp,xap,bg)

    z2 = 2*np.sqrt(2*np.pi)*(j**2)*P_STAR * _int_region(
        (lambda *args: I1(args[-1]), I15, I16), x, j, xp, xap, bg)

    e2 = max(e2, 0.0); j2 = max(j2, 0.0); z2 = max(min(z2, e2*j2+1e-22), 0.0)
    return e1, e2, j1, j2, z2

# ---------------- coefficient LUT over (x,j) ------------------
X_COARSE = np.logspace(-3, 4, 48)   # 48 x-nodes
J_COARSE = np.linspace(0.0, 1.0, 36) # 36 j-nodes

def precompute_coeff_LUT(bg: Background):
    E1 = np.zeros((len(X_COARSE), len(J_COARSE)))
    E2 = np.zeros_like(E1); J1 = np.zeros_like(E1)
    J2 = np.zeros_like(E1); Z2 = np.zeros_like(E1)
    print(f"Precomputing coefficient LUT on {len(X_COARSE)}x{len(J_COARSE)} grid ...")
    for ix, x in enumerate(X_COARSE):
        for ij, j in enumerate(J_COARSE):
            e1,e2,j1,j2,z2 = orbital_perturbations(x, j, bg)
            E1[ix,ij]=e1; E2[ix,ij]=e2; J1[ix,ij]=j1; J2[ix,ij]=j2; Z2[ix,ij]=z2
    return dict(x=X_COARSE, j=J_COARSE, e1=E1, e2=E2, j1=J1, j2=J2, z2=Z2)

def _interp2(bx, by, gridx, gridy, A):
    ix = np.clip(np.searchsorted(gridx, bx) - 1, 0, len(gridx)-2)
    iy = np.clip(np.searchsorted(gridy, by) - 1, 0, len(gridy)-2)
    x0,x1 = gridx[ix], gridx[ix+1]
    y0,y1 = gridy[iy], gridy[iy+1]
    tx = (bx - x0)/max(x1-x0, 1e-30)
    ty = (by - y0)/max(y1-y0, 1e-30)
    a00 = A[ix,  iy]; a10 = A[ix+1,iy]
    a01 = A[ix,  iy+1]; a11 = A[ix+1,iy+1]
    return (1-tx)*(1-ty)*a00 + tx*(1-ty)*a10 + (1-tx)*ty*a01 + tx*ty*a11

def coeff_from_LUT(x, j, LUT):
    e1 = _interp2(x,j,LUT['x'],LUT['j'],LUT['e1'])
    e2 = _interp2(x,j,LUT['x'],LUT['j'],LUT['e2'])
    j1 = _interp2(x,j,LUT['x'],LUT['j'],LUT['j1'])
    j2 = _interp2(x,j,LUT['x'],LUT['j'],LUT['j2'])
    z2 = _interp2(x,j,LUT['x'],LUT['j'],LUT['z2'])
    # asymptotic guards
    if j > 0.95 and e2>0: j2 = max(e2/(2*x), 0.0); z2 = min(e2*j2, e2*j2+1e-22)
    if j < 0.05 and j1>0: j2 = math.sqrt(max(2*j*j1, 0.0))
    z2 = max(min(z2, max(e2,0.0)*max(j2,0.0) + 1e-22), 0.0)
    return e1, max(e2,0.0), j1, max(j2,0.0), z2

# -------------------- Monte-Carlo engine ----------------------
@dataclass
class Star:
    x: float
    j: float
    w: float
    floor_bin: int = 0
    sticky: bool   = False
    phase: float   = 0.0  # counts number of orbits (dimensionless)

def sample_initial(n):
    # ln x ~ U(ln 1e-2, ln X_D); accept/reject on g0
    lx = rng.uniform(np.log(1e-2), np.log(X_D), size=2*n)
    x  = np.exp(lx)
    u  = rng.random(2*n) * (g0_interp(np.array([5.0]))[0]*1.1)
    xs = x[u < g0_interp(x)][:n]
    if xs.size < n: return sample_initial(n)
    js = np.sqrt(rng.random(n))  # p(j)=2j isotropic
    return [Star(float(xs[i]), float(js[i]), 1.0) for i in range(n)]

def enforce_sticky(s: Star):
    bx = X_BOUNDS[min(s.floor_bin, len(X_BOUNDS)-1)]
    if s.sticky and s.x < bx: s.x = bx

def clone_inward(s: Star):
    if CLONE_FACTOR <= 1 or s.floor_bin >= MAX_CLONE_DEPTH: return []
    k = CLONE_FACTOR; w = s.w / k
    return [Star(s.x,
                 float(np.clip(s.j + 1e-3*(rng.random()-0.5), 0.0, 1.0)),
                 w, min(s.floor_bin+1, len(X_BOUNDS)-1), True, s.phase)
            for _ in range(k)]

def resample_down(stars, target=TARGET_MEAN):
    w = np.array([s.w for s in stars], float)
    W = w / max(w.sum(), 1e-30)
    # systematic resampling
    u0 = rng.random()/target
    idx = np.searchsorted(np.cumsum(W), u0 + np.arange(target)/target)
    return [Star(stars[i].x, stars[i].j, 1.0, stars[i].floor_bin, stars[i].sticky, stars[i].phase) for i in idx]

def mc_step(s: Star, LUT, capture_on=True):
    # get coefficients from LUT
    e1,e2,j1,j2,z2 = coeff_from_LUT(s.x, s.j, LUT)
    # correlated kicks (Gaussian)
    muE, muJ = e1*DT, j1*DT
    sE2, sJ2, sEZ = e2*DT, j2*DT, z2*DT
    a = math.sqrt(max(sE2,0.0) + 1e-30)
    b = 0.0 if a==0 else np.clip(sEZ, -a*math.sqrt(max(sJ2,0.0)+1e-30),
                                       a*math.sqrt(max(sJ2,0.0)+1e-30))/a
    c = math.sqrt(max(sJ2 - b*b, 0.0) + 1e-30)
    z1, z2n = rng.standard_normal(), rng.standard_normal()
    dx_try = muE + a*z1
    dj_try = muJ + b*z1 + c*z2n
    # step-size rules (29a–d style)
    max_dx = 0.15*max(s.x, 1e-6)
    max_dj = min(0.10, 0.40*max(0.0, 1.0075 - s.j))
    jmin   = j_min_exact(s.x)
    max_dj = min(max_dj, max(0.25*abs(s.j - jmin), 0.10*max(jmin,1e-8)))
    s.x = float(np.clip(s.x + np.clip(dx_try, -max_dx, max_dx), 1e-4, X_D))
    s.j = float(np.clip(s.j + np.clip(dj_try, -max_dj, max_dj), 0.0, 1.0))
    enforce_sticky(s)
    # pericenter-check capture (boundary iii)
    captured = False
    if capture_on:
        phase_prev = s.phase
        s.phase += DT / Ptilde(s.x)
        reached_pericenter = int(s.phase) > int(phase_prev)
        if reached_pericenter and (s.j <= j_min_exact(s.x)):
            captured = True
    return captured

def run_iteration(LUT, capture_on=True):
    stars = sample_initial(20)
    sub_edges = np.linspace(0, N_STEPS, 7, dtype=int)

    g_by = [np.zeros_like(XCEN) for _ in range(6)]
    w_by = [np.zeros_like(XCEN) for _ in range(6)]

    def bidx(x):
        return int(np.clip(np.searchsorted(XBINS, x) - 1, 0, len(XCEN)-1))

    for step in range(N_STEPS):
        new = []
        if len(stars) > MAX_POP:
            stars = resample_down(stars, TARGET_MEAN)

        for s in stars:
            x_prev = s.x
            sub = np.searchsorted(sub_edges, step, side='right') - 1

            if mc_step(s, LUT, capture_on=capture_on):
                # replacement from outer reservoir
                xs = float(np.exp(rng.uniform(np.log(1e-2), np.log(X_D))))
                new.append(Star(xs, float(np.sqrt(rng.random())), s.w, 0, False, 0.0))
                continue

            # record residence for ḡ
            idx = bidx(s.x)
            g_by[sub][idx] += s.w
            w_by[sub][idx] += s.w

            # creation–annihilation on crossing inward boundary
            prev_idx = np.searchsorted(X_BOUNDS, x_prev) - 1
            now_idx  = np.searchsorted(X_BOUNDS, s.x) - 1
            if (now_idx > prev_idx) and (now_idx < len(X_BOUNDS)-1):
                s.floor_bin = now_idx
                new += clone_inward(s)

            new.append(s)

        while len(new) < TARGET_MEAN:
            new.append(sample_initial(1)[0])
        stars = new

    # average last 5 sub-intervals and compute errors like paper
    g_stack = []
    for sub in range(1, 6):
        g_sub = np.divide(g_by[sub], np.maximum(w_by[sub], 1e-30)) / DLNX
        # normalize vs g0 on x∈[1,10]
        mask = (XCEN >= 1.0) & (XCEN <= 10.0)
        scale = np.median(g0_interp(XCEN[mask]) / np.maximum(g_sub[mask], 1e-12))
        g_stack.append(g_sub * scale)
    g_stack = np.array(g_stack)
    g_mean  = np.mean(g_stack, axis=0)
    g_err   = np.sqrt(np.sum((g_stack - g_mean)**2, axis=0) / 4.0)
    return XCEN, g_mean, g_err

# ------------------------------ main --------------------------
def main():
    # Iteration 0: background = BWII g0
    bg0 = Background(xs=np.logspace(-3,4,200), gs=g0_interp(np.logspace(-3,4,200)))
    LUT0 = precompute_coeff_LUT(bg0)
    print("Running canonical iteration with background = g0(x) ...")
    x0, gbar0, gerr0 = run_iteration(LUT0, capture_on=True)

    # Build new background from measured ḡ (smoothed) and iterate once more
    print("Building updated background from measured ḡ and recomputing LUT ...")
    bg1 = Background(xs=x0, gs=np.maximum(gbar0, 1e-12))
    LUT1 = precompute_coeff_LUT(bg1)
    print("Running canonical iteration with updated background ...")
    x1, gbar1, gerr1 = run_iteration(LUT1, capture_on=True)

    # Save CSV (final)
    arr = np.column_stack([x1, gbar1, gerr1])
    np.savetxt("gbar_table.csv", arr, delimiter=",", header="x,gbar,gbar_err", comments="")
    print("Wrote gbar_table.csv")

    # Plot g0 and final ḡ
    xcurve = np.logspace(-1, 4, 600)
    g0c    = g0_interp(xcurve)

    plt.figure(figsize=(6,6), dpi=140)
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(1e-1, 1e4); plt.ylim(1e-3, 1e2)
    plt.plot(xcurve, g0c, linewidth=2, label=r"$g_0$ (BW II)")
    plt.errorbar(x1, gbar1, yerr=gerr1, fmt="o", markersize=4, capsize=2, label=r"$\bar g$ (canonical)")
    plt.annotate(r"$x_{\rm crit}\approx 10$", xy=(10, 1e-1), xytext=(13, 2e-1),
                 arrowprops=dict(arrowstyle="->", lw=1))
    plt.xlabel(r"$x \equiv (-E/v_0^2)$")
    plt.ylabel(r"$\bar g(x)$")
    plt.title(f"Isotropized distribution — canonical case (FAST_DEMO={'ON' if FAST_DEMO else 'OFF'})")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig("figure_gbar.png")
    print("Wrote figure_gbar.png")

if __name__ == "__main__":
    main()
