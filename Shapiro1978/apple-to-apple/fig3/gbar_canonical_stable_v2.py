#!/usr/bin/env python3
# Stable MC for isotropized ḡ(x) — canonical case (x_D=1e4, P*=0.005)
# - Constant-population SMC each step (exactly N=--pop stars, equal weights)
# - Time deposition: add w*DT to current log-x bin; divide by total time
# - No per-block renormalization to g0; optional final normalize to ḡ(0.225)=1
# - One LUT from BWII g0; optional 2-pass background update; errors from block averages

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import cumulative_trapezoid

# Global RNG (initialized in main() with user-specified seed)
rng = None

# Canonical parameters
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
        e[i] = math.sqrt(c[i - 1] * c[i])
    e[0] = c[0] ** 2 / e[1]
    e[-1] = c[-1] ** 2 / e[-2]
    return e


X_EDGES = _edges_from_centers(X_F3)
DLNX = np.log(X_EDGES[1:] / X_EDGES[:-1])
XCEN = X_F3.copy()
XBINS = X_EDGES.copy()   # for backward compatibility in bidx()

# ---------- BWII g0(x) ----------
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
mask = (_g0tab > 0) & (_Xg0 > 0)
_lx, _lg = np.log10(_Xg0[mask]), np.log10(_g0tab[mask])
_slope = np.diff(_lg) / np.diff(_lx)


def g0_interp(x):
    x = np.asarray(x)
    lx = np.log10(np.clip(x, 1e-12, X_D))
    y = np.empty_like(lx)

    left = lx < _lx[0]
    right = lx > _lx[-1]
    mid = ~(left | right)

    # left tail: linear in log10 x
    y[left] = _lg[0] + _slope[0] * (lx[left] - _lx[0])
    # right tail: linear in log10 x
    y[right] = _lg[-1] + _slope[-1] * (lx[right] - _lx[-1])

    idx = np.clip(np.searchsorted(_lx, lx[mid]) - 1, 0, len(_slope) - 1)
    y[mid] = _lg[idx] + _slope[idx] * (lx[mid] - _lx[idx])
    out = 10.0 ** y

    # gentle taper to zero near X_D
    x_last = _Xg0[mask][-1]
    tmask = (x >= x_last) & (x <= X_D)
    if tmask.any():
        g_start = out[tmask][0]
        out[tmask] = np.maximum(
            g_start * (1 - (x[tmask] - x_last) / (X_D - x_last)),
            1e-16
        )
    return out


# ---------- Kepler helpers ----------
def j_min_exact(x):
    val = 2.0 * x / X_D - (x * x) / (X_D * X_D)
    return math.sqrt(max(0.0, min(1.0, val)))


def Ptilde(x):  # ∝ x^{-3/2}, only for pericenter counting
    return max(1e-3, x ** (-1.5))


# ---------- θ-quadrature (12-pt) ----------
def th_nodes(n=12):
    t, w = np.polynomial.legendre.leggauss(n)
    th = 0.25 * np.pi * (t + 1.0)
    wt = 0.25 * np.pi * w
    s, c = np.sin(th), np.cos(th)
    return th, wt, s, c


TH, WT, S, C = th_nodes(12)


# ---------- Appendix A fast kernels ----------
def _z123(x, j, xp, xap, xp2):
    inv_x1 = 1.0 / max(xp2, xap) - 1.0 / max(xp, 1e-12)
    inv_x2 = 1.0 / min(xp2, xap) - 1.0 / max(xp, 1e-12)
    inv_x3 = 1.0 / max(x, 1e-12) - 1.0 / max(xp, 1e-12)
    x1 = 1.0 / max(inv_x1, 1e-16)
    x2 = 1.0 / max(inv_x2, 1e-16)
    x3 = 1.0 / max(inv_x3, 1e-16)
    z1 = 1.0 + (xp / x1) * (S * S)
    z2 = 1.0 - (x2 / x1) * (S * S)
    z3 = 1.0 - (x3 / x1) * (S * S)
    return x1, x2, x3, z1, z2, z3


def pericenter_xp(x, j):
    e = math.sqrt(max(0.0, 1.0 - j * j))
    return 2.0 * x / max(1e-12, 1.0 - e)


def apocenter_xap(x, j):
    e = math.sqrt(max(0.0, 1.0 - j * j))
    return 2.0 * x / (1.0 + e)


def I1(xp2):
    return 0.25 * np.pi / (xp2 ** 1.5)


def I4(xp2):
    return 0.25 * np.pi / (xp2 ** 0.5)


def I2(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((xp2 * x3) / (x * x * xp * x2))
    integ = z1 * np.sqrt(np.clip(z2, 0, None)) / np.sqrt(np.clip(z3, 1e-20, None))
    return pref * np.sum(integ * WT)


def I3(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((xp2 * x2 * x3) / (x * x * x1 * xp * xp))
    integ = (C * C) * z1 / np.sqrt(np.clip(z2, 1e-20, None)) / np.sqrt(np.clip(z3, 1e-20, None))
    return pref * np.sum(integ * WT)


def I6(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((xp2 * x2 * x3) / (x * x * (x1 ** 3)))
    integ = (C ** 4) * z1 / np.sqrt(np.clip(z2, 1e-20, None)) / np.sqrt(np.clip(z3, 1e-20, None))
    return pref * np.sum(integ * WT)


def I7(x, j, xp):
    x1 = 1.0 / (1.0 / max(x, 1e-12) - 1.0 / max(xp, 1e-12))
    z1 = 1.0 + (xp / x1) * (S * S)
    pref = math.sqrt(1.0 / (x * (xp ** 3)))
    return pref * np.sum(z1 * WT)


def I9(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((xp2 * x2 * x3) / (x * (xp ** 3) * (x1 ** 2)))
    integ = (C * C) * (z1 ** 3) / np.sqrt(np.clip(z2, 1e-20, None)) / np.sqrt(np.clip(z3, 1e-20, None))
    return pref * np.sum(integ * WT)


def I10(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt(((xp2 ** 3) * (x3 ** 3)) / (x * (xp ** 3) * (x2 ** 3)))
    integ = (z1 ** 3) * (z2 ** 1.5) / np.power(np.clip(z3, 1e-20, None), 1.5)
    return pref * np.sum(integ * WT)


def I11(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt(((xp2 ** 3) * (x2 ** 3)) / (x * (xp ** 3) * (x1 ** 3)))
    integ = (C ** 4) * (z1 ** 3) / np.sqrt(np.clip(z2, 1e-20, None)) / np.power(np.clip(z3, 1e-20, None), 1.5)
    return pref * np.sum(integ * WT)


def I12(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((4 * (xp2 ** 2) * (x3 ** 3)) / ((x ** 2) * (xp ** 4) * x1 * x2))
    integ = (S * S) * (C * C) * (z1 ** 2) * np.sqrt(np.clip(z2, 0, None)) / np.power(np.clip(z3, 1e-20, None), 1.5)
    return pref * np.sum(integ * WT)


def I13(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((4 * (xp2 ** 3) * (x3 ** 5)) / ((x ** 4) * (xp ** 4) * (x1 ** 4) * (x2 ** 3)))
    integ = (C * C) * (S * S) * (z1 ** 3) * (z2 ** 1.5) / np.power(np.clip(z3, 1e-20, None), 2.5)
    return pref * np.sum(integ * WT)


def I14(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt((4 * (xp2 ** 3) * (x3 ** 5)) / ((x ** 4) * (xp ** 4) * (x1 ** 6) * x2))
    integ = (C ** 4) * (S * S) * (z1 ** 3) / np.sqrt(np.clip(z2, 1e-20, None)) / np.power(np.clip(z3, 1e-20, None), 2.5)
    return pref * np.sum(integ * WT)


def I15(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt(((xp2 ** 3) * (x3 ** 3)) / ((x ** 4) * (xp ** 4) * (x2 ** 3)))
    integ = z1 * (z2 ** 1.5) / np.power(np.clip(z3, 1e-20, None), 1.5)
    return pref * np.sum(integ * WT)


def I16(x, j, xp, xap, xp2):
    x1, x2, x3, z1, z2, z3 = _z123(x, j, xp, xap, xp2)
    pref = math.sqrt(((xp2 ** 3) * x2 * (x3 ** 3)) / ((x ** 4) * (xp ** 2) * (x1 ** 4)))
    integ = (C ** 4) * z1 / np.sqrt(np.clip(z2, 1e-20, None)) / np.power(np.clip(z3, 1e-20, None), 1.5)
    return pref * np.sum(integ * WT)


# ---------- smoothing helper for background update ----------
def smooth_log(xs, ys, k=7):
    """
    Smooth log(ys) vs log(xs) with a Gaussian kernel of half-width k.
    xs must be increasing; ys>0.
    """
    xs = np.asarray(xs).flatten()
    ys = np.asarray(ys).flatten()
    if len(xs) != len(ys):
        raise ValueError(f"smooth_log: xs and ys must have same length, got {len(xs)} and {len(ys)}")
    if len(xs) == 0:
        return np.array([])
    lx = np.log(xs)
    ly = np.log(np.clip(ys, 1e-30, None))
    offs = np.arange(-k, k + 1, dtype=float)
    ker = np.exp(-0.5 * (offs / 2.0) ** 2)
    ker /= ker.sum()
    ly_s = np.convolve(ly, ker, mode="same")
    result = np.exp(ly_s)
    # Ensure output has same length as input
    if len(result) != len(xs):
        # Fallback: return original if convolution fails
        return ys
    return result


# ---------- background + moments ----------
class Background:
    def __init__(self, xs, gs):
        # build fine log grid
        self.xg = np.logspace(-3, 4, 400)
        lx = np.log(self.xg)
        glog = np.log(
            np.clip(
                np.interp(lx, np.log(xs), np.log(np.clip(gs, 1e-16, None))),
                1e-16,
                None,
            )
        )
        # mild smoothing
        k = 7
        offs = np.arange(-k, k + 1, dtype=float)
        ker = np.exp(-0.5 * (offs / 2.0) ** 2)
        ker /= ker.sum()
        glog = np.convolve(glog, ker, mode="same")
        self.gg = np.exp(glog)

        # precompute moments ∫ g x^{-3/2} dx, ∫ g x^{-1/2} dx
        self.Mm32 = cumulative_trapezoid(self.gg * self.xg ** (-1.5), self.xg, initial=0.0)
        self.Mm12 = cumulative_trapezoid(self.gg * self.xg ** (-0.5), self.xg, initial=0.0)

    def g(self, x):
        """Log-linear interpolation of g(x) on self.xg to avoid undershoot."""
        x = np.asarray(x)
        x_clip = np.clip(x, self.xg[0], self.xg[-1])
        idx = np.clip(np.searchsorted(self.xg, x_clip) - 1, 0, len(self.xg) - 2)
        x0, x1 = self.xg[idx], self.xg[idx + 1]
        lx0, lx1 = np.log(x0), np.log(x1)
        lx = np.log(x_clip)
        t = (lx - lx0) / np.maximum(lx1 - lx0, 1e-30)
        g0, g1 = self.gg[idx], self.gg[idx + 1]
        lg = np.log(np.maximum(g0, 1e-30)) * (1 - t) + np.log(np.maximum(g1, 1e-30)) * t
        out = np.exp(lg)
        return np.maximum(out, 1e-16)

    def moment(self, x, which):
        M = self.Mm32 if which == "m32" else self.Mm12
        x = np.asarray(x)
        x_clip = np.clip(x, self.xg[0], self.xg[-1])
        idx = np.clip(np.searchsorted(self.xg, x_clip) - 1, 0, len(self.xg) - 2)
        x0, x1 = self.xg[idx], self.xg[idx + 1]
        m0, m1 = M[idx], M[idx + 1]
        t = (x_clip - x0) / np.maximum(x1 - x0, 1e-30)
        return m0 * (1 - t) + m1 * t


# tiny x′-quadrature
T12, W12 = np.polynomial.legendre.leggauss(12)


def _int_region(fun_tuple, x, j, xp, xap, bg: Background):
    """
    3 regions in x':
      (1) x' < x
      (2) x <= x' < x_ap
      (3) x_ap <= x' <= x_p
    fun_tuple = (f1, f2, f3) for the three regions (kernels).
    For f1 = I1 or I4, region (1) uses analytic moments.
    """
    val = 0.0

    # (1) x' < x
    if fun_tuple[0] is I1:
        val += np.pi * 0.25 * bg.moment(x, "m32")
    elif fun_tuple[0] is I4:
        val += np.pi * 0.25 * bg.moment(x, "m12")
    else:
        lo, hi = 1e-4, x
        if hi > lo * (1 + 1e-10):
            llo, lhi = math.log(lo), math.log(hi)
            lxs = 0.5 * (lhi - llo) * T12 + 0.5 * (lhi + llo)
            xs = np.exp(lxs)
            jac = 0.5 * (lhi - llo) * xs
            vals = np.array(
                [fun_tuple[0](x, j, xp, xap, xp2) * bg.g(xp2) for xp2 in xs]
            )
            val += float(np.sum(vals * jac * W12))

    # (2) x <= x' < x_ap
    lo, hi = x, min(apocenter_xap(x, j), X_D)
    if hi > lo * (1 + 1e-10):
        llo, lhi = math.log(lo), math.log(hi)
        lxs = 0.5 * (lhi - llo) * T12 + 0.5 * (lhi + llo)
        xs = np.exp(lxs)
        jac = 0.5 * (lhi - llo) * xs
        vals = np.array(
            [fun_tuple[1](x, j, xp, xap, xp2) * bg.g(xp2) for xp2 in xs]
        )
        val += float(np.sum(vals * jac * W12))

    # (3) x_ap <= x' <= x_p
    lo, hi = max(apocenter_xap(x, j), x * (1 + 1e-12)), min(pericenter_xp(x, j), X_D)
    if hi > lo * (1 + 1e-10):
        llo, lhi = math.log(lo), math.log(hi)
        lxs = 0.5 * (lhi - llo) * T12 + 0.5 * (lhi + llo)
        xs = np.exp(lxs)
        jac = 0.5 * (lhi - llo) * xs
        vals = np.array(
            [fun_tuple[2](x, j, xp, xap, xp2) * bg.g(xp2) for xp2 in xs]
        )
        val += float(np.sum(vals * jac * W12))

    return val


def orbital_perturbations(x, j, bg: Background):
    xp, xap = pericenter_xp(x, j), apocenter_xap(x, j)

    # E diffusion and drift
    e1 = 3.0 * np.sqrt(2 * np.pi) * P_STAR * _int_region((I1, I2, I3), x, j, xp, xap, bg)
    e2 = 4.0 * np.sqrt(2 * np.pi) * P_STAR * _int_region((I4, I6, I6), x, j, xp, xap, bg)
    e2 = max(e2, 0.0)

    # J diffusion/drift regimes
    if j > 0.95:
        j1 = np.sqrt(2 * np.pi) * (x * P_STAR / max(j, 1e-10)) * _int_region(
            (
                lambda *a: 2 * I7(x, j, xp),
                lambda *a: 6 * I12(x, j, xp, xap, a[-1]) - 9 * I9(x, j, xp, xap, a[-1]) - I10(x, j, xp, xap, a[-1]),
                lambda *a: 6 * I12(x, j, xp, xap, a[-1]) - 9 * I9(x, j, xp, xap, a[-1]) - I11(x, j, xp, xap, a[-1]),
            ),
            x,
            j,
            xp,
            xap,
            bg,
        )
        j2 = max(e2 / (2 * x), 0.0)
        z2 = min(e2 * j2, e2 * j2 + 1e-22)
        return e1, e2, j1, j2, max(z2, 0.0)

    if j < 0.05:
        j1 = np.sqrt(2 * np.pi) * (x * P_STAR / max(j, 1e-10)) * _int_region(
            (
                lambda *a: 2 * I7(x, j, xp),
                lambda *a: 6 * I12(x, j, xp, xap, a[-1]) - 9 * I9(x, j, xp, xap, a[-1]) - I10(x, j, xp, xap, a[-1]),
                lambda *a: 6 * I12(x, j, xp, xap, a[-1]) - 9 * I9(x, j, xp, xap, a[-1]) - I11(x, j, xp, xap, a[-1]),
            ),
            x,
            j,
            xp,
            xap,
            bg,
        )
        j2 = math.sqrt(max(2 * j * j1, 0.0))
        z2 = min(e2 * j2, e2 * j2 + 1e-22)
        return e1, e2, j1, j2, max(z2, 0.0)

    j1 = np.sqrt(2 * np.pi) * (x * P_STAR / max(j, 1e-10)) * _int_region(
        (
            lambda *a: 2 * I7(x, j, xp),
            lambda *a: 3 * I12(x, j, xp, xap, a[-1]) + 4 * I10(x, j, xp, xap, a[-1]) - 3 * I13(x, j, xp, xap, a[-1]),
            lambda *a: 3 * I12(x, j, xp, xap, a[-1]) + 4 * I11(x, j, xp, xap, a[-1]) - 3 * I14(x, j, xp, xap, a[-1]),
        ),
        x,
        j,
        xp,
        xap,
        bg,
    )

    j2 = np.sqrt(2 * np.pi) * x * P_STAR * _int_region(
        (
            lambda *a: 4 * I7(x, j, xp),
            lambda *a: 3 * I12(x, j, xp, xap, a[-1]) + 4 * I10(x, j, xp, xap, a[-1]) - 3 * I13(x, j, xp, xap, a[-1]),
            lambda *a: 3 * I12(x, j, xp, xap, a[-1]) + 4 * I11(x, j, xp, xap, a[-1]) - 3 * I14(x, j, xp, xap, a[-1]),
        ),
        x,
        j,
        xp,
        xap,
        bg,
    )
    j2 = max(j2, 0.0)

    z2 = 2 * np.sqrt(2 * np.pi) * (j ** 2) * P_STAR * _int_region(
        (lambda *a: I1(a[-1]), I15, I16),
        x,
        j,
        xp,
        xap,
        bg,
    )
    z2 = max(min(z2, e2 * j2 + 1e-22), 0.0)
    return e1, e2, j1, j2, z2


# ---------- coefficient LUT ----------
def precompute_LUT(nx, nj, bg: Background):
    Xg = np.logspace(-3, 4, nx)
    Jg = np.linspace(0.0, 1.0, nj)

    E1 = np.zeros((nx, nj))
    E2 = np.zeros_like(E1)
    J1 = np.zeros_like(E1)
    J2 = np.zeros_like(E1)
    Z2 = np.zeros_like(E1)

    print(f"Precomputing LUT {nx}x{nj} ...")
    for ix, x in enumerate(Xg):
        for ij, j in enumerate(Jg):
            e1, e2, j1, j2, z2 = orbital_perturbations(x, j, bg)
            E1[ix, ij] = e1
            E2[ix, ij] = e2
            J1[ix, ij] = j1
            J2[ix, ij] = j2
            Z2[ix, ij] = z2

    return dict(x=Xg, j=Jg, e1=E1, e2=E2, j1=J1, j2=J2, z2=Z2)


def _interp2(bx, by, gx, gy, A):
    ix = np.clip(np.searchsorted(gx, bx) - 1, 0, len(gx) - 2)
    iy = np.clip(np.searchsorted(gy, by) - 1, 0, len(gy) - 2)
    x0, x1 = gx[ix], gx[ix + 1]
    y0, y1 = gy[iy], gy[iy + 1]
    tx = (bx - x0) / max(x1 - x0, 1e-30)
    ty = (by - y0) / max(y1 - y0, 1e-30)
    a00 = A[ix, iy]
    a10 = A[ix + 1, iy]
    a01 = A[ix, iy + 1]
    a11 = A[ix + 1, iy + 1]
    return (
        (1 - tx) * (1 - ty) * a00
        + tx * (1 - ty) * a10
        + (1 - tx) * ty * a01
        + tx * ty * a11
    )


def coeff_from_LUT(x, j, L):
    e1 = _interp2(x, j, L["x"], L["j"], L["e1"])
    e2 = _interp2(x, j, L["x"], L["j"], L["e2"])
    j1 = _interp2(x, j, L["x"], L["j"], L["j1"])
    j2 = _interp2(x, j, L["x"], L["j"], L["j2"])
    z2 = _interp2(x, j, L["x"], L["j"], L["z2"])

    e2 = max(e2, 0.0)
    j2 = max(j2, 0.0)
    z2 = max(min(z2, e2 * j2 + 1e-22), 0.0)

    # asymptotes
    if j > 0.95 and e2 > 0:
        j2 = max(e2 / (2 * x), 0.0)
        z2 = min(e2 * j2, e2 * j2 + 1e-22)

    if j < 0.05 and j1 > 0:
        j2 = math.sqrt(max(2 * j * j1, 0.0))

    z2 = max(min(z2, e2 * j2 + 1e-22), 0.0)
    return e1, e2, j1, j2, z2


# ---------- MC ----------
@dataclass
class Star:
    x: float
    j: float
    phase: float = 0.0


def sample_initial(n):
    # Start from the outer reservoir (canonical) and isotropic in j
    xs = np.full(n, X_BOUND, dtype=float)
    js = np.sqrt(rng.random(n))  # p(j)=2j
    return [Star(float(xs[i]), float(js[i]), 0.0) for i in range(n)]


def bidx(x):
    # Find bin index for x using Fig. 3 edges
    return int(np.clip(np.searchsorted(X_EDGES, x, side="right") - 1, 0, len(XCEN) - 1))


# ---------- Optional importance scheme (log-x floors) ----------
def floor_idx_for(x, X_FLOORS):
    """Return deepest floor index k with x >= X_FLOORS[k], or -1 if x < X_FLOORS[0]"""
    if X_FLOORS is None or len(X_FLOORS) == 0:
        return -1
    k = -1
    for i, xf in enumerate(X_FLOORS):
        if x >= xf:
            k = i
    return k


def maybe_split(obj, X_FLOORS, n_clones):
    """Split object when crossing into a deeper floor (one-time per floor)"""
    if X_FLOORS is None or n_clones <= 0:
        return []
    k_now = floor_idx_for(obj["x"], X_FLOORS)
    new_clones = []
    while obj["floor_idx"] < k_now:
        obj["floor_idx"] += 1
        new_w = obj["w"] / (1.0 + n_clones)
        obj["w"] = new_w
        for _ in range(n_clones):
            new_clones.append(
                dict(
                    x=obj["x"],
                    j=obj["j"],
                    w=new_w,
                    phase=obj["phase"],
                    floor_idx=obj["floor_idx"],
                )
            )
    return new_clones


def mc_step(s: Star, L, DT, disable_capture=False):
    e1, e2, j1, j2, z2 = coeff_from_LUT(s.x, s.j, L)

    # correlated Gaussian kick
    muE, muJ = e1 * DT, j1 * DT
    sE2, sJ2, sEZ = e2 * DT, j2 * DT, z2 * DT

    a = math.sqrt(max(sE2, 0.0) + 1e-30)
    c0 = math.sqrt(max(sJ2, 0.0) + 1e-30)
    rho = np.clip(sEZ / (a * c0), -0.999, 0.999) if (a > 0 and c0 > 0) else 0.0

    z1, z2n = rng.standard_normal(), rng.standard_normal()
    dE = muE + a * z1
    dJ = muJ + rho * c0 * z1 + c0 * math.sqrt(max(1 - rho * rho, 0.0)) * z2n

    # bounded step (conservative)
    max_dx = 0.15 * max(s.x, 1e-6)
    jmin = j_min_exact(s.x)
    max_dj = min(
        0.10,
        0.40 * max(0.0, 1.0 - s.j),
        max(0.25 * abs(s.j - jmin), 0.10 * max(jmin, 1e-8)),
    )

    s.x = float(np.clip(s.x - np.clip(dE, -max_dx, max_dx), 1e-4, X_D))
    dj = np.clip(dJ, -max_dj, max_dj)
    s.j = float(np.clip(s.j + dj, 0.0, 1.0))

    # enforce reservoir boundary
    if s.x < X_BOUND:
        s.x = X_BOUND
        s.j = float(np.sqrt(rng.random()))

    # capture at pericenter crossings (diagnostic; we replace immediately)
    if not disable_capture:
        prev = int(s.phase)
        s.phase += DT / max(Ptilde(s.x), 1e-6)
        if int(s.phase) > prev and (s.j <= j_min_exact(s.x)):
            # re-inject from reservoir
            s.x = X_BOUND
            s.j = float(np.sqrt(rng.random()))
            s.phase = 0.0

    return False  # kept for compatibility


def run_once(
    steps,
    DT,
    pop,
    LUT,
    blocks=8,
    progress_every=0.05,
    disable_capture=False,
    X_FLOORS=None,
    n_clones=0,
    per_dlnx=True,
):
    # constant-population SMC with equal weights => total weight = pop
    stars = sample_initial(pop)

    # Optional: importance scheme with clones
    use_clones = (X_FLOORS is not None and n_clones > 0)
    clones = []  # list of dicts: {x,j,w,phase,floor_idx}

    if use_clones:
        parents = [
            dict(x=s.x, j=s.j, w=1.0, phase=s.phase, floor_idx=-1, x_prev=s.x)
            for s in stars
        ]
        stars = None
    else:
        parents = None
        x_prev = [s.x for s in stars]

    gtime_blocks = [np.zeros_like(XCEN) for _ in range(blocks)]
    t0_blocks = np.zeros(blocks, float)
    block_edges = np.linspace(0, steps, blocks + 1, dtype=int)
    last_print = 0.0

    for k in range(steps):
        if use_clones:
            active = parents + [c for c in clones if c.get("active", True)]
            stars = [Star(x=obj["x"], j=obj["j"], phase=obj["phase"]) for obj in active]
            weights = [obj["w"] for obj in active]
            x_prev_list = [obj.get("x_prev", obj["x"]) for obj in active]
        else:
            weights = [1.0] * len(stars)

        # advance all objects
        if use_clones:
            for p in parents:
                p["x_prev"] = p["x"]
                s = Star(x=p["x"], j=p["j"], phase=p["phase"])
                mc_step(s, LUT, DT, disable_capture=disable_capture)
                p["x"], p["j"], p["phase"] = s.x, s.j, s.phase
                if p["x"] < X_BOUND:
                    p["x"] = X_BOUND
                    p["j"] = float(np.sqrt(rng.random()))
                new_clones = maybe_split(p, X_FLOORS, n_clones)
                for nc in new_clones:
                    nc["x_prev"] = nc["x"]
                clones.extend(new_clones)

            for c in clones:
                if not c.get("active", True):
                    continue
                c["x_prev"] = c["x"]
                s = Star(x=c["x"], j=c["j"], phase=c["phase"])
                mc_step(s, LUT, DT, disable_capture=disable_capture)
                c["x"], c["j"], c["phase"] = s.x, s.j, s.phase
                if c["x"] < X_BOUND:
                    c["x"] = X_BOUND
                    c["j"] = float(np.sqrt(rng.random()))
                if c["floor_idx"] >= 0 and c["x"] < X_FLOORS[c["floor_idx"]]:
                    c["active"] = False
                else:
                    new_clones = maybe_split(c, X_FLOORS, n_clones)
                    for nc in new_clones:
                        nc["x_prev"] = nc["x"]
                    clones.extend(new_clones)
        else:
            x_prev_saved = x_prev.copy()
            for i, s in enumerate(stars):
                mc_step(s, LUT, DT, disable_capture=disable_capture)
                x_prev[i] = s.x

        # deposit time along the path (boundary-safe)
        blk = int(np.searchsorted(block_edges, k, side="right") - 1)
        total_w = 0.0

        if use_clones:
            active = parents + [c for c in clones if c.get("active", True)]
            for obj in active:
                x0 = max(obj.get("x_prev", obj["x"]), X_BOUND)
                x1 = max(obj["x"], X_BOUND)
                w = obj["w"]
                dt0 = DT

                dlog = abs(math.log(max(x1, 1e-12) / max(x0, 1e-12)))
                if dlog <= 0.25:
                    xm = math.sqrt(x0 * x1)
                    xm = max(xm, X_BOUND * (1.0 + 1e-12))
                    k_idx = bidx(xm)
                    if 0 <= k_idx < len(XCEN):
                        gtime_blocks[blk][k_idx] += w * dt0
                else:
                    nsub = min(8, 1 + int(dlog / 0.25))
                    ratio = max(x1, 1e-12) / max(x0, 1e-12)
                    for u in range(nsub):
                        xm = x0 * (ratio ** ((u + 0.5) / nsub))
                        xm = max(xm, X_BOUND * (1.0 + 1e-12))
                        k_idx = bidx(xm)
                        if 0 <= k_idx < len(XCEN):
                            gtime_blocks[blk][k_idx] += w * (dt0 / nsub)

                total_w += w
        else:
            for i, s in enumerate(stars):
                x0 = max(x_prev_saved[i], X_BOUND)
                x1 = max(s.x, X_BOUND)
                w = weights[i]
                dt0 = DT

                dlog = abs(math.log(max(x1, 1e-12) / max(x0, 1e-12)))
                if dlog <= 0.25:
                    xm = math.sqrt(x0 * x1)
                    xm = max(xm, X_BOUND * (1.0 + 1e-12))
                    k_idx = bidx(xm)
                    if 0 <= k_idx < len(XCEN):
                        gtime_blocks[blk][k_idx] += w * dt0
                else:
                    nsub = min(8, 1 + int(dlog / 0.25))
                    ratio = max(x1, 1e-12) / max(x0, 1e-12)
                    for u in range(nsub):
                        xm = x0 * (ratio ** ((u + 0.5) / nsub))
                        xm = max(xm, X_BOUND * (1.0 + 1e-12))
                        k_idx = bidx(xm)
                        if 0 <= k_idx < len(XCEN):
                            gtime_blocks[blk][k_idx] += w * (dt0 / nsub)

                total_w += w

        t0_blocks[blk] += total_w * DT

        # progress
        p = (k + 1) / steps
        if p - last_print >= progress_every:
            print(f"Progress: {100 * p:5.1f}% ({k + 1}/{steps})", flush=True)
            last_print = p

    # per-block spectra (no per-block renorm)
    g_list = []
    bin_widths = X_EDGES[1:] - X_EDGES[:-1]
    dlnx = DLNX

    for b in range(blocks):
        tmp = gtime_blocks[b] / max(t0_blocks[b], 1e-30)
        if per_dlnx:
            g_b = np.divide(tmp, dlnx, out=np.zeros_like(tmp), where=(dlnx > 0))
        else:
            g_b = np.divide(tmp, bin_widths, out=np.zeros_like(tmp), where=(bin_widths > 0))
        g_list.append(g_b)

    g_arr = np.stack(g_list, axis=0)
    g_mean = np.mean(g_arr, axis=0)
    nb = g_arr.shape[0]
    g_var = np.var(g_arr, axis=0, ddof=1) if nb > 1 else np.zeros_like(g_mean)
    g_err = np.sqrt(np.maximum(g_var / nb, 0.0))
    return g_mean, g_err


# ---------- background-update helper ----------
def build_bg_from_measure(
    xcen,
    g_mean,
    g_err,
    xmin_upd=0.25,
    xmax_upd=50.0,
    snr_max=0.6,
    alpha=0.45,
    floor_rel=0.30,
    ceil_rel=3.0,
):
    """
    Build a safe updated background g(x) for the LUT, using a measured ḡ:
      - Anchor at x=0.225 to g0(x) BEFORE smoothing
      - Use only bins with decent SNR and within [xmin_upd, xmax_upd]
      - Smooth in log-space
      - Blend with g0 in log-space with weight alpha
      - Clamp to [floor_rel*g0, ceil_rel*g0]
    """
    xcen = np.asarray(xcen)
    g_mean = np.asarray(g_mean)
    g_err = np.asarray(g_err)

    xbg = np.logspace(-3, 4, 400)
    g0b = g0_interp(xbg)

    # anchor to g0 at 0.225 before any smoothing
    idx_225 = int(np.argmin(np.abs(xcen - 0.225)))
    g0_225 = g0_interp(np.array([0.225]))[0]
    scale = g0_225 / max(g_mean[idx_225], 1e-16)
    g1 = g_mean * scale
    e1 = g_err * scale

    # choose reliable bins
    ok = (xcen >= xmin_upd) & (xcen <= xmax_upd) & (g1 > 0)
    ok &= (e1 / np.maximum(g1, 1e-16)) <= snr_max

    if not np.any(ok):
        print("Background-update: no reliable bins; using g0 only.")
        return Background(xs=xbg, gs=g0b)

    # smooth in log-space over the reliable window
    xs_in = xcen[ok]
    gs_in = g1[ok]
    gs_sm = smooth_log(xs_in, np.maximum(gs_in, 1e-16), k=7)

    # Ensure arrays have the same length
    xs_in = np.asarray(xs_in).flatten()
    gs_sm = np.asarray(gs_sm).flatten()
    if len(xs_in) != len(gs_sm):
        # If lengths don't match, use unsmoothed values
        gs_sm = np.maximum(gs_in, 1e-16)
        xs_in = xcen[ok]
        gs_sm = np.asarray(gs_sm).flatten()
        xs_in = np.asarray(xs_in).flatten()

    lx_in = np.log(xs_in)
    lg_in = np.log(np.maximum(gs_sm, 1e-16))

    # Final safety check
    if len(lx_in) != len(lg_in):
        print(f"Warning: length mismatch lx_in={len(lx_in)}, lg_in={len(lg_in)}")
        min_len = min(len(lx_in), len(lg_in))
        lx_in = lx_in[:min_len]
        lg_in = lg_in[:min_len]

    inwin = (xbg >= xmin_upd) & (xbg <= xmax_upd)
    lg_up = np.interp(np.log(xbg[inwin]), lx_in, lg_in)

    g_upd = g0b.copy()
    g_upd[inwin] = np.exp(lg_up)

    # log-space blend with g0 and clamp
    lg0 = np.log(np.maximum(g0b, 1e-16))
    lgu = np.log(np.maximum(g_upd, 1e-16))
    lgm = (1 - alpha) * lg0 + alpha * lgu
    g_mix = np.exp(lgm)

    g_mix = np.clip(g_mix, floor_rel * g0b, ceil_rel * g0b)

    return Background(xs=xbg, gs=g_mix)


# optional LUT probe (diagnostic)
def _lut_probe(L, label="LUT"):
    xs = [0.3, 1.0, 3.0]
    print(f"{label} probe:")
    for xv in xs:
        ix = max(0, min(len(L["x"]) - 2, np.searchsorted(L["x"], xv) - 1))
        e2_med = float(np.median(L["e2"][ix, :]))
        j2_med = float(np.median(L["j2"][ix, :]))
        print(f"  x≈{xv:4.1f}: median E2={e2_med:.3e}, J2={j2_med:.3e}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="if None, dt = 6/steps (one 6 t0 iteration)",
    )
    ap.add_argument("--lut-x", type=int, default=40)
    ap.add_argument("--lut-j", type=int, default=28)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--normalize-225", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument(
        "--disable-capture",
        action="store_true",
        help="disable capture during measurement (for tail diagnostics only; keep OFF for paper-like central curve)",
    )
    ap.add_argument(
        "--floors",
        type=int,
        default=0,
        help="number of log-x floors for importance scheme (0=off)",
    )
    ap.add_argument(
        "--clones",
        type=int,
        default=0,
        help="number of clones per split (0=off, set to 2 to enable)",
    )
    ap.add_argument(
        "--xmax-floor",
        type=float,
        default=80.0,
        help="top of the floor ladder",
    )
    ap.add_argument(
        "--bin-locked-floors",
        action="store_true",
        help="use Fig. 3 bin centers as floor rungs (more stable)",
    )
    ap.add_argument(
        "--no-per-dlnx",
        dest="per_dlnx",
        action="store_false",
        default=True,
        help="use per-Δx normalization instead of per-Δln x (diagnostic mode). "
             "Default is per-Δln x (paper-like).",
    )
    ap.add_argument(
        "--background-update",
        action="store_true",
        help="enable 2-pass background update (paper's method)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed for reproducibility",
    )
    args = ap.parse_args()

    # Initialize global RNG
    global rng
    rng = np.random.default_rng(args.seed)

    DT = (6.0 / args.steps) if args.dt is None else float(args.dt)

    # Baseline background from BWII g0
    xgrid = np.logspace(-3, 4, 400)
    bg0 = Background(xs=xgrid, gs=g0_interp(xgrid))
    LUT0 = precompute_LUT(args.lut_x, args.lut_j, bg0)
    _lut_probe(LUT0, label="LUT0 (g0)")

    # Floors for importance sampling (if enabled)
    X_FLOORS = None
    if args.floors > 0 and args.clones > 0:
        if args.bin_locked_floors:
            valid_centers = X_F3[(X_F3 >= X_BOUND) & (X_F3 <= args.xmax_floor)]
            if len(valid_centers) > 0:
                floors_list = [valid_centers[0]]
                for i in range(len(valid_centers) - 1):
                    mid = math.sqrt(valid_centers[i] * valid_centers[i + 1])
                    if mid > floors_list[-1] * 1.01:
                        floors_list.append(mid)
                    floors_list.append(valid_centers[i + 1])
                X_FLOORS = np.array(floors_list)
                if args.floors > 0 and len(X_FLOORS) > args.floors:
                    idx = np.linspace(0, len(X_FLOORS) - 1, args.floors, dtype=int)
                    X_FLOORS = X_FLOORS[idx]
                print(
                    f"Using {len(X_FLOORS)} bin-locked floors (from Fig. 3 centers), clones={args.clones}"
                )
            else:
                X_FLOORS = np.geomspace(X_BOUND * (1 + 1e-9), args.xmax_floor, args.floors)
                print(
                    f"Using {args.floors} log-x floors from {X_BOUND:.3f} to {args.xmax_floor:.1f}, clones={args.clones}"
                )
        else:
            X_FLOORS = np.geomspace(X_BOUND * (1 + 1e-9), args.xmax_floor, args.floors)
            print(
                f"Using {args.floors} log-x floors from {X_BOUND:.3f} to {args.xmax_floor:.1f}, clones={args.clones}"
            )

    use_per_dlnx = getattr(args, "per_dlnx", True)
    if use_per_dlnx:
        print("Using per-Δln x normalization (paper-like)")
    else:
        print("Using per-Δx normalization (diagnostic mode)")

    # ----- Pass 1: measure with g0 background -----
    print("Pass 1: Measuring with g0 background...")
    g_mean, g_err = run_once(
        args.steps,
        DT,
        args.pop,
        LUT0,
        blocks=args.blocks,
        disable_capture=args.disable_capture,
        X_FLOORS=X_FLOORS,
        n_clones=args.clones,
        per_dlnx=use_per_dlnx,
    )

    # ----- Optional 2-pass background update -----
    if args.background_update:
        print("Updating background from measured ḡ (anchored/SNR-gated/log-blended)...")
        bg1 = build_bg_from_measure(
            XCEN,
            g_mean,
            g_err,
            xmin_upd=0.25,
            xmax_upd=50.0,
            snr_max=0.6,
            alpha=0.45,
            floor_rel=0.30,
            ceil_rel=3.0,
        )
        print("Rebuilding LUT on updated background...")
        LUT1 = precompute_LUT(args.lut_x, args.lut_j, bg1)
        _lut_probe(LUT1, label="LUT1 (updated ḡ)")

        print("Pass 2: Measuring with updated background...")
        g_mean, g_err = run_once(
            args.steps,
            DT,
            args.pop,
            LUT1,
            blocks=args.blocks,
            disable_capture=args.disable_capture,
            X_FLOORS=X_FLOORS,
            n_clones=args.clones,
            per_dlnx=use_per_dlnx,
        )

    # ----- Final normalization, output, plot -----
    if args.normalize_225:
        idx_225 = int(np.argmin(np.abs(XCEN - 0.225)))
        s = g_mean[idx_225] if g_mean[idx_225] > 0 else 1.0
        g_mean /= s
        g_err /= s

    arr = np.column_stack([XCEN, g_mean, g_err])
    np.savetxt("gbar_table.csv", arr, delimiter=",", header="x,gbar,gbar_err", comments="")
    print("Wrote gbar_table.csv")

    if args.out:
        xcurve = np.logspace(-1, 4, 600)
        g0c = g0_interp(xcurve)
        plt.figure(figsize=(6.2, 6), dpi=140)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1e-1, 1e4)
        positive = g_mean[g_mean > 0]
        if positive.size > 0:
            ymin = max(np.min(positive) * 0.6, 1e-4)
        else:
            ymin = 1e-4
        ymax = max(np.max(g_mean) * 2 if positive.size > 0 else 1.0, 1e0)
        plt.ylim(ymin, ymax)
        plt.plot(xcurve, g0c, lw=2, label=r"$g_0$ (BW II)")
        plt.errorbar(XCEN, g_mean, yerr=g_err, fmt="o", ms=4, capsize=2, label=r"$\bar g$")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\bar g(x)$")
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
