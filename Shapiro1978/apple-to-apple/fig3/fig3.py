#!/usr/bin/env python3
import math, numpy as np, matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import cumulative_trapezoid
import time

# ---------- Canonical parameters ----------
X_D     = 1.0e4      # outer energy cut-off
P_STAR  = 5.0e-3     # coupling
TAU_ITER= 6.0        # one iteration spans 6 t0
SEED    = 7
FAST_DEMO = True     # True: ~1.2e3 steps; False: 1e4 steps
N_STEPS = 150*2 if FAST_DEMO else 10_000
DT      = TAU_ITER / N_STEPS

# population control (creation–annihilation)
N_INIT       = 20
TARGET_MEAN  = 150
MAX_POP      = 450            # cap population to prevent avalanche
U_BOUNDS     = 10.0**(-np.arange(0,6))        # u_i = 10^{-i}
X_BOUNDS     = U_BOUNDS**(-4.0/5.0)           # x_i = 10^{0.8 i}

# ---- toggles for different runs ----
NO_LOSS_CONE = False          # True for "without loss cone" run
CLONE_FACTOR = 0 if NO_LOSS_CONE else 5   # 0 stops clones in no-loss-cone run
MAX_CLONE_DEPTH = 3          # don't allow clones to clone more than 3 times

# plotting / binning
XBINS = np.logspace(-1, 4, 60)
XCEN  = np.sqrt(XBINS[:-1]*XBINS[1:])
DLNX  = np.log(XBINS[1:]/XBINS[:-1])

rng = np.random.default_rng(SEED)

# ---------- BWII g0(x): single-mass table -> smooth curve ----------
ln1p = np.array([0.00,0.37,0.74,1.11,1.47,1.84,2.21,2.58,2.95,3.32,
                 3.68,4.05,4.42,4.79,5.16,5.53,5.89,6.26,6.63,7.00,
                 7.37,7.74,8.11,8.47,8.84,9.21], float)
g0t  = np.array([1.00,1.30,1.55,1.79,2.03,2.27,2.53,2.82,3.13,3.48,
                 3.88,4.32,4.83,5.43,6.12,6.94,7.93,9.11,10.55,12.29,
                 14.36,16.66,18.80,19.71,15.70,0.00], float)
Xtab = np.exp(ln1p)-1.0
m    = (g0t>0)&(Xtab>0)
lx, lg = np.log10(Xtab[m]), np.log10(g0t[m])
sl     = np.diff(lg)/np.diff(lx)
def g0_interp(x):
    x = np.asarray(x); lt=np.log10(np.clip(x,1e-3,X_D))
    y=np.empty_like(lt)
    left  = lt<lx[0]; right=lt>lx[-1]; mid=~(left|right)
    y[left]  = lg[0]  + sl[0]*(lt[left]-lx[0])
    y[right] = lg[-1] + sl[-1]*(lt[right]-lx[-1])
    idx=np.clip(np.searchsorted(lx, lt[mid])-1, 0, len(sl)-1)
    y[mid]=lg[idx]+sl[idx]*(lt[mid]-lx[idx])
    out=10**y
    x_last=Xtab[m][-1]; t=(x>=x_last)&(x<=X_D)
    if t.any():
        g0s=out[t][0]
        out = np.asarray(out)  # Ensure out is always an array
        out[t]=np.maximum(g0s*(1-(x[t]-x_last)/(X_D-x_last)),1e-16)
    return out

# ------- coefficient LUT over (x,j) -------
X_COARSE = np.logspace(-3, 4, 64)        # 64 x-nodes
J_COARSE = np.linspace(0.0, 1.0, 48)     # 48 j-nodes

def precompute_coeff_LUT():
    print("Precomputing coefficient lookup table...")
    E1 = np.zeros((len(X_COARSE), len(J_COARSE)))
    E2 = np.zeros_like(E1)
    J1 = np.zeros_like(E1)
    J2 = np.zeros_like(E1)
    Z2 = np.zeros_like(E1)
    for ix, x in enumerate(X_COARSE):
        for ij, j in enumerate(J_COARSE):
            e1,e2,j1,j2,z2 = orbital_perturbations(x, j)  # your (fast) Appendix-A routine
            E1[ix,ij]=e1; E2[ix,ij]=e2; J1[ix,ij]=j1; J2[ix,ij]=j2; Z2[ix,ij]=z2
    print("Lookup table precomputed!")
    return dict(x=X_COARSE, j=J_COARSE, e1=E1, e2=E2, j1=J1, j2=J2, z2=Z2)

# LUT will be initialized after all functions are defined

def interp2(bx, by, gridx, gridy, A):
    # bilinear interpolation on A[ix,iy]
    ix = np.clip(np.searchsorted(gridx, bx)-1, 0, len(gridx)-2)
    iy = np.clip(np.searchsorted(gridy, by)-1, 0, len(gridy)-2)
    x0,x1 = gridx[ix], gridx[ix+1]
    y0,y1 = gridy[iy], gridy[iy+1]
    tx = (bx - x0)/max(x1-x0, 1e-30)
    ty = (by - y0)/max(y1-y0, 1e-30)
    a00 = A[ix,  iy]
    a10 = A[ix+1,iy]
    a01 = A[ix,  iy+1]
    a11 = A[ix+1,iy+1]
    return (1-tx)*(1-ty)*a00 + tx*(1-ty)*a10 + (1-tx)*ty*a01 + tx*ty*a11

def coeff_from_LUT(x, j, lut):
    e1 = interp2(x,j,lut['x'],lut['j'],lut['e1'])
    e2 = interp2(x,j,lut['x'],lut['j'],lut['e2'])
    j1 = interp2(x,j,lut['x'],lut['j'],lut['j1'])
    j2 = interp2(x,j,lut['x'],lut['j'],lut['j2'])
    z2 = interp2(x,j,lut['x'],lut['j'],lut['z2'])
    # guards (A12–A13):
    if j > 0.95 and e2>0: j2 = max(e2/(2*x), 0.0); z2 = min(e2*j2, e2*j2+1e-22)
    if j < 0.05 and j1>0: j2 = math.sqrt(max(2*j*j1, 0.0))
    z2 = max(min(z2, e2*j2+1e-22), 0.0)
    return e1, max(e2,0.0), j1, max(j2,0.0), z2

# ---------- θ-quadrature (small) ----------
def th_nodes(n=12):
    t,w=np.polynomial.legendre.leggauss(n)
    th=0.25*np.pi*(t+1.0); wt=0.25*np.pi*w
    s=np.sin(th); c=np.cos(th)
    return th, wt, s, c
TH, WT, S, C = th_nodes(12)   # <<-- small rule (fast & accurate enough)

# ---------- Kepler helpers & loss-cone ----------
def pericenter_xp(x,j):
    e=np.sqrt(max(0.0,1.0-j*j)); return 2.0*x/max(1e-12,1.0-e)
def apocenter_xap(x,j):
    e=np.sqrt(max(0.0,1.0-j*j)); return 2.0*x/(1.0+e)
def j_min_exact(x):
    val=2.0*x/X_D - (x*x)/(X_D*X_D)
    return math.sqrt(max(0.0, min(1.0,val)))

# ---------- FAST kernels (A6) using the tiny θ-rule ----------
def _z123(x,j,xp,xap,xp2):
    inv_x1 = 1.0/max(xp2,xap) - 1.0/max(xp,1e-12)
    inv_x2 = 1.0/min(xp2,xap) - 1.0/max(xp,1e-12)
    inv_x3 = 1.0/max(x, 1e-12) - 1.0/max(xp,1e-12)
    x1=1.0/max(inv_x1,1e-16); x2=1.0/max(inv_x2,1e-16); x3=1.0/max(inv_x3,1e-16)
    z1 = 1.0 + (xp/x1)*(S*S); z2 = 1.0 - (x2/x1)*(S*S); z3 = 1.0 - (x3/x1)*(S*S)
    return x1,x2,x3,z1,z2,z3

def I1(x): return 0.25*np.pi/(x**1.5)
def I4(x): return 0.25*np.pi/(x**0.5)

def I2(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((xp2*x3)/(x*x*xp*x2))
    integ=z1*np.sqrt(np.clip(z2,0,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I3(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((xp2*x2*x3)/(x*x*x1*xp*xp))
    integ=(C*C)*z1/np.sqrt(np.clip(z2,1e-20,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I6(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((xp2*x2*x3)/(x*x*(x1**3)))
    integ=(C**4)*z1/np.sqrt(np.clip(z2,1e-20,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)

def I7(x,j,xp):
    x1 = 1.0/(1.0/max(x,1e-12) - 1.0/max(xp,1e-12))
    z1 = 1.0 + (xp/x1)*(S*S)
    pref=math.sqrt(1.0/(x*(xp**3)))
    return pref*np.sum(z1*WT)

def I9(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((xp2*x2*x3)/(x*(xp**3)*(x1**2)))
    integ=(C*C)*(z1**3)/np.sqrt(np.clip(z2,1e-20,None))/np.sqrt(np.clip(z3,1e-20,None))
    return pref*np.sum(integ*WT)
def I10(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt(((xp2**3)*(x3**3))/(x*(xp**3)*(x2**3)))
    integ=(z1**3)*(z2**1.5)/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)
def I11(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt(((xp2**3)*(x2**3))/(x*(xp**3)*(x1**3)))
    integ=(C**4)*(z1**3)/np.sqrt(np.clip(z2,1e-20,None))/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)
def I12(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((4*(xp2**2)*(x3**3))/((x**2)*(xp**4)*x1*x2))
    integ=(S*S)*(C*C)*(z1**2)*np.sqrt(np.clip(z2,0,None))/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)
def I13(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((4*(xp2**3)*(x3**5))/((x**4)*(xp**4)*(x1**4)*(x2**3)))
    integ=(C*C)*(S*S)*(z1**3)*(z2**1.5)/np.power(np.clip(z3,1e-20,None),2.5)
    return pref*np.sum(integ*WT)
def I14(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt((4*(xp2**3)*(x3**5))/((x**4)*(xp**4)*(x1**6)*x2))
    integ=(C**4)*(S*S)*(z1**3)/np.sqrt(np.clip(z2,1e-20,None))/np.power(np.clip(z3,1e-20,None),2.5)
    return pref*np.sum(integ*WT)
def I15(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt(((xp2**3)*(x3**3))/((x**4)*(xp**4)*(x2**3)))
    integ=(z1)*(z2**1.5)/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)
def I16(x,j,xp,xap,xp2):
    x1,x2,x3,z1,z2,z3=_z123(x,j,xp,xap,xp2)
    pref=math.sqrt(((xp2**3)*x2*(x3**3))/((x**4)*(xp**2)*(x1**4)))
    integ=(C**4)*z1/np.sqrt(np.clip(z2,1e-20,None))/np.power(np.clip(z3,1e-20,None),1.5)
    return pref*np.sum(integ*WT)

# ---------- precompute cumulative moments for x'<x part ----------
# log-grid for g0 and its moments (reused in every call)
XG = np.logspace(-3, 4, 400)
G0 = g0_interp(XG)
M_m32 = cumulative_trapezoid(G0*XG**(-1.5), XG, initial=0.0)  # ∫ g x'^(-3/2) dx'
M_m12 = cumulative_trapezoid(G0*XG**(-0.5), XG, initial=0.0)  # ∫ g x'^(-1/2) dx'

def moment_interp(x, M, power):
    # piecewise-linear in log x for robustness
    x = np.asarray(x)
    idx = np.clip(np.searchsorted(XG, x)-1, 0, len(XG)-2)
    x0, x1 = XG[idx], XG[idx+1]
    m0, m1 = M[idx],  M[idx+1]
    t = (x - x0)/np.maximum(x1-x0, 1e-30)
    return m0 + t*(m1-m0)

# ---------- integrate over x' with tiny x'-quadrature (12 nodes) ----------
T12, W12 = np.polynomial.legendre.leggauss(12)
def _int_region(fun_tuple, x, j, xp, xap, gx):
    # combine 3 regions using 12 x'-nodes in log-space where needed
    val = 0.0
    # (1) x'<x  ---> use cumulative moments (closed form) when possible
    if fun_tuple[0] is I1:   # epsilon1*
        val += np.sum(gx)*0.0  # placeholder; closed below
        val += np.pi*0.25*moment_interp(x, M_m32, -1.5)
    elif fun_tuple[0] is I4: # epsilon2*
        val += np.pi*0.25*moment_interp(x, M_m12, -0.5)
    else:
        # rare: if a different kernel shows up in region 1 (it shouldn't)
        lo = 1e-4; hi = x
        if hi>lo*(1+1e-10):
            llo, lhi = math.log(lo), math.log(hi)
            lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
            xs = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
            vals = np.array([fun_tuple[0](x,j,xp,xap,xp2)*g0_interp(xp2) for xp2 in xs])
            val += float(np.sum(vals*jac*W12))
    # (2) x ≤ x' < x_ap
    lo, hi = x, min(xap, X_D)
    if hi>lo*(1+1e-10):
        llo, lhi = math.log(lo), math.log(hi)
        lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
        xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
        vals = np.array([fun_tuple[1](x,j,xp,xap,xp2)*g0_interp(xp2) for xp2 in xs])
        val += float(np.sum(vals*jac*W12))
    # (3) x_ap ≤ x' ≤ x_p
    lo, hi = max(xap, x*(1+1e-12)), min(xp, X_D)
    if hi>lo*(1+1e-10):
        llo, lhi = math.log(lo), math.log(hi)
        lxs = 0.5*(lhi-llo)*T12 + 0.5*(lhi+llo)
        xs  = np.exp(lxs); jac = 0.5*(lhi-llo)*xs
        vals = np.array([fun_tuple[2](x,j,xp,xap,xp2)*g0_interp(xp2) for xp2 in xs])
        val += float(np.sum(vals*jac*W12))
    return val

def orbital_perturbations(x, j):
    xp, xap = pericenter_xp(x,j), apocenter_xap(x,j)

    # epsilon1* and epsilon2* (fast; region-1 uses closed moments)
    e1 = 3.0*np.sqrt(2*np.pi)*P_STAR * _int_region(
        (lambda *args: I1(args[-1]),  # region 1 uses I1(x')
         I2, I3), x, j, xp, xap, G0)

    e2 = 4.0*np.sqrt(2*np.pi)*P_STAR * _int_region(
        (lambda *args: I4(args[-1]),
         I6, I6), x, j, xp, xap, G0)

    # j1*, j2*, zeta*: use asymptotics to avoid heavy work in corners
    if j > 0.95:
        # compute only e2, then enforce A12 relations
        j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
            (lambda *args: 2*I7(x,j,xp),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I10(x,j,xp,xap,args[-1]),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I11(x,j,xp,xap,args[-1])),
            x, j, xp, xap, G0)
        j2 = max(e2/(2*x), 0.0)
        z2 = max(min(e2*j2, e2*j2+1e-22), 0.0)
        return e1, max(e2,0.0), j1, j2, z2

    if j < 0.05:
        # compute j1 cheaply, then j2 from A13
        j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
            (lambda *args: 2*I7(x,j,xp),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I10(x,j,xp,xap,args[-1]),
             lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I11(x,j,xp,xap,args[-1])),
            x, j, xp, xap, G0)
        j2 = math.sqrt(max(2*j*j1, 0.0))
        # cross-correlation small here
        z2 = min(e2*j2, e2*j2+1e-22)
        return e1, max(e2,0.0), j1, j2, max(z2,0.0)

    # mid-j: do the tiny-quadrature version of all terms
    j1 = np.sqrt(2*np.pi)*(x*P_STAR/max(j,1e-10)) * _int_region(
        (lambda *args: 2*I7(x,j,xp),
         lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I10(x,j,xp,xap,args[-1]),
         lambda *args: 6*I12(x,j,xp,xap,args[-1])-9*I9(x,j,xp,xap,args[-1])-I11(x,j,xp,xap,args[-1])),
        x, j, xp, xap, G0)

    j2 = np.sqrt(2*np.pi)*x*P_STAR * _int_region(
        (lambda *args: 4*I7(x,j,xp),
         lambda *args: 3*I12(x,j,xp,xap,args[-1])+4*I10(x,j,xp,xap,args[-1])-3*I13(x,j,xp,xap,args[-1]),
         lambda *args: 3*I12(x,j,xp,xap,args[-1])+4*I11(x,j,xp,xap,args[-1])-3*I14(x,j,xp,xap,args[-1])),
        x, j, xp, xap, G0)

    z2 = 2*np.sqrt(2*np.pi)*(j**2)*P_STAR * _int_region(
        (lambda *args: I1(args[-1]),
         I15, I16), x, j, xp, xap, G0)

    e2 = max(e2,0.0); j2=max(j2,0.0); z2=max(min(z2, e2*j2+1e-22),0.0)
    return e1, e2, j1, j2, z2

# ---------- creation–annihilation MC ----------
@dataclass
class Star:
    x: float; j: float; w: float; floor_bin: int=0; sticky: bool=False

def sample_initial(n):
    # ln x ~ U(ln 1e-2, ln X_D); accept/reject on g0
    lx = rng.uniform(np.log(1e-2), np.log(X_D), size=2*n)
    x  = np.exp(lx)
    u  = rng.random(2*n)*(g0_interp(np.array([5.0]))[0]*1.1)
    xs = x[u<g0_interp(x)][:n]
    if xs.size<n: return sample_initial(n)
    js = np.sqrt(rng.random(n))  # p(j)=2j
    return [Star(float(xs[i]), float(js[i]), 1.0) for i in range(n)]

def enforce_sticky(s):
    bx = X_BOUNDS[min(s.floor_bin, len(X_BOUNDS)-1)]
    if s.sticky and s.x < bx: s.x = bx

def clone_inward(s):
    if CLONE_FACTOR <= 1 or s.floor_bin >= MAX_CLONE_DEPTH:  # prevent avalanche
        return []
    k = CLONE_FACTOR
    w = s.w / k
    return [Star(s.x,
                 float(np.clip(s.j + 1e-3*(rng.random()-0.5), 0.0, 1.0)),
                 w, min(s.floor_bin+1, len(X_BOUNDS)-1), True)
            for _ in range(k)]

def resample_down(stars, target=TARGET_MEAN):
    """Systematic resampling to control population size"""
    w = np.array([s.w for s in stars], float)
    W = w / w.sum()
    idx = np.searchsorted(np.cumsum(W), (np.arange(target)+rng.random())/target)
    new = [Star(stars[i].x, stars[i].j, 1.0, stars[i].floor_bin, stars[i].sticky) for i in idx]
    return new

def step_star(s, capture_on=True, lut=None):
    if lut is None:
        e1,e2,j1,j2,z2 = orbital_perturbations(s.x, s.j)  # fallback to original
    else:
        e1,e2,j1,j2,z2 = coeff_from_LUT(s.x, s.j, lut)  # use LUT
    # correlated kicks
    muE, muJ = e1*DT, j1*DT
    sE2, sJ2, sEZ = e2*DT, j2*DT, z2*DT
    a = math.sqrt(max(sE2,0.0)+1e-30)
    b = 0.0 if a==0 else np.clip(sEZ, -a*math.sqrt(max(sJ2,0.0)+1e-30),
                                       a*math.sqrt(max(sJ2,0.0)+1e-30))/a
    c = math.sqrt(max(sJ2 - b*b, 0.0)+1e-30)
    z1, z2n = rng.standard_normal(), rng.standard_normal()
    dx_try = muE + a*z1
    dj_try = muJ + b*z1 + c*z2n
    # step-size rules (29a–d style)
    max_dx = 0.15*max(s.x, 1e-6)
    max_dj = min(0.10, 0.40*max(0.0, 1.0075 - s.j))
    jmin   = j_min_exact(s.x)
    max_dj = min(max_dj, max(0.25*abs(s.j-jmin), 0.10*max(jmin,1e-8)))
    s.x = float(np.clip(s.x + np.clip(dx_try,-max_dx,max_dx), 1e-4, X_D))
    s.j = float(np.clip(s.j + np.clip(dj_try,-max_dj,max_dj), 0.0, 1.0))
    enforce_sticky(s)
    captured = (s.j <= j_min_exact(s.x)) if capture_on else False
    return captured

# Removed process_star_batch - using sequential processing instead

def run_iteration(capture_on=True, lut=None):
    stars = sample_initial(N_INIT)
    sub_edges = np.linspace(0, N_STEPS, 7, dtype=int)
    g_by, w_by, F_by = [np.zeros_like(XCEN) for _ in range(6)], \
                       [np.zeros_like(XCEN) for _ in range(6)], \
                       [np.zeros_like(XCEN) for _ in range(6)]

    def bidx(x): return int(np.clip(np.searchsorted(XBINS,x)-1,0,len(XCEN)-1))

    start_time = time.time()
    print("Using sequential processing (no multiprocessing)")
    
    for k in range(N_STEPS):
        elapsed = time.time() - start_time
        if k > 0:
            avg_time_per_step = elapsed / k
            remaining_steps = N_STEPS - k
            eta_seconds = remaining_steps * avg_time_per_step
            eta_minutes = eta_seconds / 60
            print(f"Step {k}/{N_STEPS} ({k/N_STEPS*100:.1f}%) - {len(stars)} stars - ETA: {eta_minutes:.1f} min")
        else:
            print(f"Step {k}/{N_STEPS} ({k/N_STEPS*100:.1f}%) - {len(stars)} stars - Starting...")
        new=[]
        
        # Population control: resample if too many stars
        if len(stars) > MAX_POP:
            stars = resample_down(stars, TARGET_MEAN)
        
        if len(stars) > 2*TARGET_MEAN:
            stars.sort(key=lambda s: s.w)
            merged=[]; i=0
            while i < len(stars)-1:
                a,b=stars[i],stars[i+1]
                if a.w<0.05 and b.w<0.05:
                    w=a.w+b.w; x=(a.x*a.w+b.x*b.w)/w; j=(a.j*a.w+b.j*b.w)/w
                    merged.append(Star(x,j,w,max(a.floor_bin,b.floor_bin), a.sticky or b.sticky)); i+=2
                else: merged.append(stars[i]); i+=1
            if i==len(stars)-1: merged.append(stars[-1])
            stars=merged

        # Process stars sequentially
        sub=np.searchsorted(sub_edges,k,side='right')-1
        for s in stars:
            # Use LUT for fast coefficient lookup
            e1,e2,j1,j2,z2 = coeff_from_LUT(s.x, s.j, lut) if lut else orbital_perturbations(s.x, s.j)
            
            # correlated kicks
            muE, muJ = e1*DT, j1*DT
            sE2, sJ2, sEZ = e2*DT, j2*DT, z2*DT
            a = math.sqrt(max(sE2,0.0)+1e-30)
            b = 0.0 if a==0 else np.clip(sEZ, -a*math.sqrt(max(sJ2,0.0)+1e-30),
                                               a*math.sqrt(max(sJ2,0.0)+1e-30))/a
            c = math.sqrt(max(sJ2 - b*b, 0.0)+1e-30)
            z1, z2n = rng.standard_normal(), rng.standard_normal()
            dx_try = muE + a*z1
            dj_try = muJ + b*z1 + c*z2n
            
            # step-size rules (29a–d style)
            max_dx = 0.15*max(s.x, 1e-6)
            max_dj = min(0.10, 0.40*max(0.0, 1.0075 - s.j))
            jmin   = j_min_exact(s.x)
            max_dj = min(max_dj, max(0.25*abs(s.j-jmin), 0.10*max(jmin,1e-8)))
            new_x = float(np.clip(s.x + np.clip(dx_try,-max_dx,max_dx), 1e-4, X_D))
            new_j = float(np.clip(s.j + np.clip(dj_try,-max_dj,max_dj), 0.0, 1.0))
            
            # Create new star with updated values
            new_s = Star(new_x, new_j, s.w, s.floor_bin, s.sticky)
            enforce_sticky(new_s)
            captured = (new_s.j <= j_min_exact(new_s.x)) if capture_on else False
            
            if captured:
                idx=bidx(s.x)
                F_by[sub][idx]+= new_s.w/(DLNX[idx]*TAU_ITER)
                xs=float(np.exp(rng.uniform(np.log(1e-2), np.log(X_D))))
                new.append(Star(xs, float(np.sqrt(rng.random())), new_s.w))
                continue
                
            idx=bidx(new_s.x)
            g_by[sub][idx]+=new_s.w; w_by[sub][idx]+=new_s.w
            prev_idx=np.searchsorted(X_BOUNDS, s.x)-1
            now_idx =np.searchsorted(X_BOUNDS, new_s.x)-1
            if (not NO_LOSS_CONE) and now_idx>prev_idx and now_idx<len(X_BOUNDS)-1:
                new_s.floor_bin=now_idx; new += clone_inward(new_s)
            new.append(new_s)
        
        while len(new)<TARGET_MEAN:
            new.append(sample_initial(1)[0])
        stars=new

    # build outputs from last five sub-intervals (2..6)
    g_stack=[]
    for sub in range(1,6):
        g_sub = np.divide(g_by[sub], np.maximum(w_by[sub],1e-30))/DLNX
        mask = (XCEN>=1.0)&(XCEN<=10.0)
        scale=np.median(g0_interp(XCEN[mask])/np.maximum(g_sub[mask],1e-12))
        g_stack.append(g_sub*scale)
    g_stack=np.array(g_stack)
    g_mean=np.mean(g_stack,axis=0)
    g_err =np.sqrt(np.sum((g_stack-g_mean)**2,axis=0)/4.0)

    F_stack=np.array([F_by[sub] for sub in range(1,6)])
    F_mean =np.mean(F_stack,axis=0)
    F_err  =np.sqrt(np.sum((F_stack-F_mean)**2,axis=0)/4.0)
    
    total_time = time.time() - start_time
    print(f"Completed in {total_time/60:.1f} minutes")
    return XCEN, g_mean, g_err, F_mean, F_err

if __name__ == "__main__":
    # Initialize lookup table after all functions are defined
    LUT = precompute_coeff_LUT()
    
    # ---------- run: (a) no loss cone; (b) canonical with loss cone ----------
    print("Running simulation without loss cone...")
    # Set parameters for no-loss-cone run
    NO_LOSS_CONE = True
    CLONE_FACTOR = 0  # disable cloning
    x0,g0m,g0e,_,_   = run_iteration(capture_on=False, lut=LUT)   # should match BW within errors
    
    print("Running simulation with loss cone...")
    # Set parameters for canonical run
    NO_LOSS_CONE = False
    CLONE_FACTOR = 5  # enable cloning
    x ,g ,ge ,FE,FEe = run_iteration(capture_on=True, lut=LUT)

    # ---------- plot + save ----------
    curve_x=np.logspace(-1,4,600); curve_g0=g0_interp(curve_x)
    np.savetxt("fast_mc_gbar.csv", np.column_stack([x,g,ge]), delimiter=",",
               header="x,gbar,gbar_err", comments="")
    np.savetxt("fast_mc_FEx.csv", np.column_stack([x,FE,FE*x,FEe*x]), delimiter=",",
               header="x,FE_star,FE_star_times_x,FE_star_times_x_err", comments="")
    np.savetxt("fast_mc_g0_curve.csv", np.column_stack([curve_x,curve_g0]), delimiter=",",
               header="x,g0", comments="")

    plt.figure(figsize=(6,6),dpi=140)
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(1e-1,1e4); plt.ylim(1e-3,1e5)
    plt.plot(curve_x,curve_g0,linewidth=2,label=r"$g_0$ (BW II)")
    plt.errorbar(x,g,yerr=ge,fmt="o",markersize=4,capsize=2,label=r"$\bar g$")
    m=FE>0
    plt.errorbar(x[m],(FE*x)[m],yerr=(FEe*x)[m],fmt="s",markersize=4,capsize=2,label=r"$F_E^*\cdot x$")
    plt.annotate(r"$x_{\rm crit}\approx 10$", xy=(10,1e-1), xytext=(12,2e-1),
                 arrowprops=dict(arrowstyle="->",lw=1))
    plt.xlabel(r"$x\equiv(-E/v_0^2)$"); plt.ylabel(r"$\bar g,\ F_E^*x$")
    plt.title(f"Fast MC (FAST_DEMO={'ON' if FAST_DEMO else 'OFF'}) — canonical case")
    plt.legend(loc="lower left",fontsize=9); plt.tight_layout(); plt.savefig("fast_figure3.png")
    print("Simulation completed successfully!")
    print("Wrote: fast_figure3.png, fast_mc_gbar.csv, fast_mc_FEx.csv, fast_mc_g0_curve.csv")
