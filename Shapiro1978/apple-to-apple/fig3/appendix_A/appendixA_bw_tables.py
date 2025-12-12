# appendixA_bw_tables.py
"""
Appendix A tables from Shapiro & Marchant (1978) for a given isotropized field-star DF.

This version fixes rare-but-real OverflowError crashes by computing all prefactors
in log-space (so no intermediate like x1**4 can overflow in Python).

It also includes:
  - make_gbar_from_table2(): builds an interpolated gbar(x) from Table 2 values
  - print_table1_like(): prints -eps1*, eps2*, j1*, j2*, zeta*2 for comparison

Conventions match the earlier file you were using:
  eps1_star    = epsilon_1 / v0^2
  eps2_star2   = epsilon_2^2 / v0^4
  j1_star      = j_1 / Jmax(E)
  j2_star2     = j_2^2 / Jmax(E)^2
  zeta_star2   = zeta^2 / (v0^2 Jmax(E))   (paper's zeta^{*2})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

Array = np.ndarray


# -------------------------
# Numerical helpers
# -------------------------

_TINY = 1e-300
_LOG_EXP_MAX = 700.0   # exp(700) ~ 1e304  (safe under float max ~ 1e308)
_LOG_EXP_MIN = -745.0  # exp(-745) underflows to 0


def _logpos(x: float) -> float:
    return float(np.log(max(float(x), _TINY)))


def _exp_clipped(logx: float) -> float:
    return float(np.exp(np.clip(logx, _LOG_EXP_MIN, _LOG_EXP_MAX)))


def _leggauss(n: int) -> Tuple[Array, Array]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x.astype(float), w.astype(float)


@dataclass(frozen=True)
class ThetaQuad:
    """Gauss–Legendre quadrature on theta in [0, pi/2], with cached trig."""
    n: int = 48

    def __post_init__(self):
        if self.n < 8:
            raise ValueError("n_theta should be >= 8")

    @property
    def theta(self) -> Array:
        t, _ = _leggauss(self.n)
        return 0.25 * np.pi * (t + 1.0)

    @property
    def w(self) -> Array:
        _, w = _leggauss(self.n)
        return 0.25 * np.pi * w

    @property
    def sin2(self) -> Array:
        s = np.sin(self.theta)
        return s * s

    @property
    def cos2(self) -> Array:
        c = np.cos(self.theta)
        return c * c

    @property
    def cos4(self) -> Array:
        c2 = self.cos2
        return c2 * c2


@dataclass(frozen=True)
class GLQuad:
    """Gauss–Legendre quadrature on [-1,1]."""
    n: int = 48

    def __post_init__(self):
        if self.n < 8:
            raise ValueError("n_x should be >= 8")

    @property
    def x(self) -> Array:
        x, _ = _leggauss(self.n)
        return x

    @property
    def w(self) -> Array:
        _, w = _leggauss(self.n)
        return w

    def map(self, a: float, b: float) -> Tuple[Array, Array, float]:
        """Map nodes/weights from [-1,1] to [a,b]. Returns (x_mapped, w, halfwidth)."""
        if b <= a:
            return np.empty(0), np.empty(0), 0.0
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xm = mid + half * self.x
        return xm, self.w, half


# -------------------------
# Orbit geometry
# -------------------------

def x_p_x_ap(x: float, j: float) -> Tuple[float, float]:
    """
    r_{ap,p} = (r_a/(2x)) [1 ± sqrt(1-j^2)]
    x_{ap,p} = r_a / r_{ap,p} = 2x / [1 ± sqrt(1-j^2)]
    """
    if x <= 0:
        raise ValueError("Require x>0")
    if not (0.0 < j <= 1.0):
        raise ValueError("Require 0<j<=1")
    s = np.sqrt(max(0.0, 1.0 - j * j))
    denom_p = max(_TINY, 1.0 - s)
    denom_ap = 1.0 + s
    return 2.0 * x / denom_p, 2.0 * x / denom_ap


def _x1_x2_x3(x: float, xprime: float, x_p: float, x_ap: float) -> Tuple[float, float, float]:
    """
    Eq. (A7a–c) with max/min depending on xprime vs x_ap:
      x1^{-1} = max(x', x_ap)^{-1} - x_p^{-1}
      x2^{-1} = min(x', x_ap)^{-1} - x_p^{-1}
      x3^{-1} = x^{-1} - x_p^{-1}
    """
    xmax = x_ap if xprime < x_ap else xprime
    xmin = xprime if xprime < x_ap else x_ap

    inv_x1 = (1.0 / xmax) - (1.0 / x_p)
    inv_x2 = (1.0 / xmin) - (1.0 / x_p)
    inv_x3 = (1.0 / x) - (1.0 / x_p)

    inv_x1 = max(inv_x1, _TINY)
    inv_x2 = max(inv_x2, _TINY)
    inv_x3 = max(inv_x3, _TINY)

    return 1.0 / inv_x1, 1.0 / inv_x2, 1.0 / inv_x3


def _x1_for_I7(x_p: float, x_ap: float) -> float:
    inv_x1 = (1.0 / x_ap) - (1.0 / x_p)
    inv_x1 = max(inv_x1, _TINY)
    return 1.0 / inv_x1


# -------------------------
# Main calculator
# -------------------------

@dataclass
class AppendixA:
    P_star: float
    x_D: float
    g0_bw: Callable[[Array], Array]
    n_theta: int = 48
    n_x: int = 48

    def __post_init__(self):
        self.theta = ThetaQuad(self.n_theta)
        self.glx = GLQuad(self.n_x)

    # ---- isotropized field-star DF ----

    def bar_g(self, xprime: Array) -> Array:
        """
        Isotropized DF \bar{g}(x'):
          - x'<=0 : exp(x')
          - 0<x'<=x_D : g0_bw(x')
          - x'>x_D : 0
        """
        xprime = np.asarray(xprime, dtype=float)
        out = np.zeros_like(xprime)
        m_unb = xprime <= 0.0
        out[m_unb] = np.exp(xprime[m_unb])
        m_bnd = (xprime > 0.0) & (xprime <= self.x_D)
        if np.any(m_bnd):
            out[m_bnd] = self.g0_bw(xprime[m_bnd])
        return out

    def _int_bar_g_minusinf_to_x(self, x: float) -> float:
        """
        For x>0:
          ∫_{-∞}^0 exp(x') dx' = 1
          so ∫_{-∞}^x \bar{g} = 1 + ∫_0^{min(x,x_D)} g0_bw(x') dx'
        """
        if x <= 0:
            return float(np.exp(x))
        xb = min(x, self.x_D)
        if xb <= 0:
            return 1.0
        xp, w, half = self.glx.map(0.0, xb)
        return 1.0 + half * float(np.sum(w * self.g0_bw(xp)))

    # ---- analytic I1/I4 ----

    @staticmethod
    def I1(x: float) -> float:
        return 0.25 * np.pi * x ** (-1.5)

    @staticmethod
    def I4(x: float) -> float:
        return 0.25 * np.pi * x ** (-0.5)

    # ---- I7 (depends only on x,j) ----

    def I7(self, x: float, x_p: float, x_ap: float) -> float:
        th = self.theta
        x1 = _x1_for_I7(x_p=x_p, x_ap=x_ap)

        z1 = 1.0 + (x_p / x1) * th.sin2

        # pref = sqrt(1/(x * x_p^6)) in log-space
        logpref = -0.5 * (_logpos(x) + 6.0 * _logpos(x_p))
        pref = _exp_clipped(logpref)

        return pref * float(np.sum(th.w * (z1 ** 3)))

    # ---- full I-set (A6) for x' >= x ----

    def Iset(self, x: float, j: float, xprime: float, x_p: float, x_ap: float) -> Dict[str, float]:
        th = self.theta
        x1, x2, x3 = _x1_x2_x3(x=x, xprime=xprime, x_p=x_p, x_ap=x_ap)

        sin2 = th.sin2
        cos2 = th.cos2
        cos4 = th.cos4

        z1 = 1.0 + (x_p / x1) * sin2
        z2 = 1.0 - (x2 / x1) * sin2
        z3 = 1.0 - (x3 / x1) * sin2

        z2 = np.maximum(z2, _TINY)
        z3 = np.maximum(z3, _TINY)

        w = th.w

        lx = _logpos(x)
        lxp = _logpos(xprime)
        lx1 = _logpos(x1)
        lx2 = _logpos(x2)
        lx3 = _logpos(x3)
        lxpv = _logpos(x_p)

        # helper: pref = exp(0.5 * (sum(log num) - sum(log den)))
        def pref_from_logs(num_log: float, den_log: float) -> float:
            return _exp_clipped(0.5 * (num_log - den_log))

        # I2: sqrt( (x' x3) / (x^2 x_p^2 x2) )
        pref2 = pref_from_logs(lxp + lx3, 2 * lx + 2 * lxpv + lx2)
        I2 = pref2 * float(np.sum(w * (z1 * np.sqrt(z2) / np.sqrt(z3))))

        # I3: sqrt( (x' x2 x3) / (x^2 x1^2 x_p^2) )
        pref3 = pref_from_logs(lxp + lx2 + lx3, 2 * lx + 2 * lx1 + 2 * lxpv)
        I3 = pref3 * float(np.sum(w * (cos2 * z1 / np.sqrt(z2 * z3))))

        # I5: sqrt( (x'^3 x3) / (x^2 x2^3) )
        pref5 = pref_from_logs(3 * lxp + lx3, 2 * lx + 3 * lx2)
        I5 = pref5 * float(np.sum(w * ((z2 ** 1.5) / np.sqrt(z3))))

        # I6: sqrt( (x'^3 x2 x3) / (x^2 x1^4) )
        pref6 = pref_from_logs(3 * lxp + lx2 + lx3, 2 * lx + 4 * lx1)
        I6 = pref6 * float(np.sum(w * (cos4 / np.sqrt(z2 * z3))))

        # I8: sqrt( (x' x3) / (x^2 x_p^6 x2) )
        pref8 = pref_from_logs(lxp + lx3, 2 * lx + 6 * lxpv + lx2)
        I8 = pref8 * float(np.sum(w * ((z1 ** 3) * np.sqrt(z2) / np.sqrt(z3))))

        # I9: sqrt( (x' x2 x3) / (x^2 x_p^6 x1^2) )
        pref9 = pref_from_logs(lxp + lx2 + lx3, 2 * lx + 6 * lxpv + 2 * lx1)
        I9 = pref9 * float(np.sum(w * (cos2 * (z1 ** 3) / np.sqrt(z2 * z3))))

        # I10: sqrt( (x'^3 x3^3) / (x^4 x_p^6 x2^3) )
        pref10 = pref_from_logs(3 * lxp + 3 * lx3, 4 * lx + 6 * lxpv + 3 * lx2)
        I10 = pref10 * float(np.sum(w * ((z1 ** 3) * (z2 ** 1.5) / (z3 ** 1.5))))

        # I11: sqrt( (x'^3 x2 x3^3) / (x^4 x_p^6 x1^4) )
        pref11 = pref_from_logs(3 * lxp + lx2 + 3 * lx3, 4 * lx + 6 * lxpv + 4 * lx1)
        I11 = pref11 * float(
            np.sum(w * (cos4 * (z1 ** 3) / (np.sqrt(z2) * (z3 ** 1.5))))
        )

        # I12: sqrt( (4 x' x3^3) / (x^2 x_p^4 x1^4 x2) )
        pref12 = pref_from_logs(np.log(4.0) + lxp + 3 * lx3, 2 * lx + 4 * lxpv + 4 * lx1 + lx2)
        I12 = pref12 * float(
            np.sum(w * ((sin2 * cos2) * (z1 ** 2) * np.sqrt(z2) / (z3 ** 1.5)))
        )

        # I13: sqrt( (4 x'^3 x3^5) / (x^4 x_p^4 x1^4 x2^3) )
        pref13 = pref_from_logs(np.log(4.0) + 3 * lxp + 5 * lx3, 4 * lx + 4 * lxpv + 4 * lx1 + 3 * lx2)
        I13 = pref13 * float(
            np.sum(w * ((cos2 * sin2) * (z1 ** 2) * (z2 ** 1.5) / (z3 ** 2.5)))
        )

        # I14: sqrt( (4 x'^3 x3^5) / (x^4 x_p^4 x1^6 x2) )
        pref14 = pref_from_logs(np.log(4.0) + 3 * lxp + 5 * lx3, 4 * lx + 4 * lxpv + 6 * lx1 + lx2)
        I14 = pref14 * float(
            np.sum(w * ((cos4 * sin2) * (z1 ** 2) * np.sqrt(z2) / (z3 ** 2.5)))
        )

        # I15: sqrt( (x'^3 x3^3) / (x^4 x_p^2 x2^3) )
        pref15 = pref_from_logs(3 * lxp + 3 * lx3, 4 * lx + 2 * lxpv + 3 * lx2)
        I15 = pref15 * float(np.sum(w * (z1 * (z2 ** 1.5) / (z3 ** 1.5))))

        # I16: sqrt( (x'^3 x2 x3^3) / (x^4 x_p^2 x1^4) )
        pref16 = pref_from_logs(3 * lxp + lx2 + 3 * lx3, 4 * lx + 2 * lxpv + 4 * lx1)
        I16 = pref16 * float(np.sum(w * (cos4 * z1 / (np.sqrt(z2) * (z3 ** 1.5)))))

        return {
            "I2": I2,
            "I3": I3,
            "I5": I5,
            "I6": I6,
            "I8": I8,
            "I9": I9,
            "I10": I10,
            "I11": I11,
            "I12": I12,
            "I13": I13,
            "I14": I14,
            "I15": I15,
            "I16": I16,
        }

    # -------------------------
    # Main coefficient function
    # -------------------------

    def coeffs_star(self, x: float, j: float) -> Dict[str, float]:
        """Compute (eps1*, eps2*^2, j1*, j2*^2, zeta*^2) at (x,j)."""
        x = float(x)
        j = float(j)
        if x <= 0:
            raise ValueError("This is for bound test stars: require x>0")
        if not (0.0 < j <= 1.0):
            raise ValueError("Require 0<j<=1")

        x_p, x_ap = x_p_x_ap(x, j)
        x_p_eff = min(x_p, self.x_D)  # because \bar{g}(x'>x_D)=0

        # Region 1: x' < x
        I1 = self.I1(x)
        I4 = self.I4(x)
        I7 = self.I7(x=x, x_p=x_p, x_ap=x_ap)

        G1 = self._int_bar_g_minusinf_to_x(x)

        # Accumulators for the three regions
        A_eps1 = I1 * G1
        A_eps2 = I4 * G1
        A_j1 = (2.0 * I7) * G1
        A_j2 = (4.0 * I7) * G1
        A_zeta = I1 * G1

        B_eps1 = B_eps2 = B_j1 = B_j2 = B_zeta = 0.0
        C_eps1 = C_eps2 = C_j1 = C_j2 = C_zeta = 0.0

        # Region 2: x <= x' < x_ap
        if x_ap > x and x_p_eff > x:
            b2 = min(x_ap, x_p_eff)
            if b2 > x:
                xnodes, w, half = self.glx.map(x, b2)
                gvals = self.g0_bw(xnodes)
                for xpr, wi, gi in zip(xnodes, w, gvals):
                    Is = self.Iset(x=x, j=j, xprime=float(xpr), x_p=x_p, x_ap=x_ap)
                    B_eps1 += wi * gi * (-Is["I2"])
                    B_eps2 += wi * gi * (Is["I5"])
                    B_j1 += wi * gi * (6.0 * Is["I12"] - 9.0 * Is["I8"] - Is["I10"])
                    B_j2 += wi * gi * (3.0 * Is["I12"] + 4.0 * Is["I10"] - 3.0 * Is["I13"])
                    B_zeta += wi * gi * (Is["I15"])
                B_eps1 *= half
                B_eps2 *= half
                B_j1 *= half
                B_j2 *= half
                B_zeta *= half

        # Region 3: x_ap <= x' <= x_p
        if x_p_eff > x_ap:
            a3 = max(x_ap, x)
            b3 = x_p_eff
            if b3 > a3:
                xnodes, w, half = self.glx.map(a3, b3)
                gvals = self.g0_bw(xnodes)
                for xpr, wi, gi in zip(xnodes, w, gvals):
                    Is = self.Iset(x=x, j=j, xprime=float(xpr), x_p=x_p, x_ap=x_ap)
                    C_eps1 += wi * gi * (-Is["I3"])
                    C_eps2 += wi * gi * (Is["I6"])
                    C_j1 += wi * gi * (6.0 * Is["I12"] - 9.0 * Is["I9"] - Is["I11"])
                    C_j2 += wi * gi * (3.0 * Is["I12"] + 4.0 * Is["I11"] - 3.0 * Is["I14"])
                    C_zeta += wi * gi * (Is["I16"])
                C_eps1 *= half
                C_eps2 *= half
                C_j1 *= half
                C_j2 *= half
                C_zeta *= half

        sqrt2pi = np.sqrt(2.0 * np.pi)

        eps1_star = 3.0 * sqrt2pi * self.P_star * (A_eps1 + B_eps1 + C_eps1)
        eps2_star2 = 4.0 * sqrt2pi * self.P_star * (A_eps2 + B_eps2 + C_eps2)
        j1_star = sqrt2pi * (x / j) * self.P_star * (A_j1 + B_j1 + C_j1)
        j2_star2 = sqrt2pi * x * self.P_star * (A_j2 + B_j2 + C_j2)
        zeta_star2 = 2.0 * sqrt2pi * j * self.P_star * (A_zeta + B_zeta + C_zeta)

        eps2_star2 = max(0.0, float(eps2_star2))
        j2_star2 = max(0.0, float(j2_star2))

        return {
            "eps1_star": float(eps1_star),
            "eps2_star2": float(eps2_star2),
            "eps2_star": float(np.sqrt(eps2_star2)),
            "j1_star": float(j1_star),
            "j2_star2": float(j2_star2),
            "j2_star": float(np.sqrt(j2_star2)),
            "zeta_star2": float(zeta_star2),
            "x_p": float(x_p),
            "x_ap": float(x_ap),
        }


# -------------------------
# Table builder
# -------------------------

def compute_tables(
    X_TABLE: Array,
    J_GRID: Array,
    *,
    g0_bw: Callable[[Array], Array],
    P_star: float,
    x_D: float,
    n_theta: int = 48,
    n_x: int = 48,
) -> Dict[str, Array]:
    """
    Compute coefficient tables on (X_TABLE, J_GRID).

    Returns dict of arrays with shape (len(X_TABLE), len(J_GRID)):
      eps1_star, eps2_star, j1_star, j2_star, zeta_star2
    plus eps2_star2, j2_star2, x_p, x_ap.
    """
    X_TABLE = np.asarray(X_TABLE, dtype=float)
    J_GRID = np.asarray(J_GRID, dtype=float)

    if np.any(X_TABLE <= 0):
        raise ValueError("X_TABLE must contain only x>0")
    if np.any(J_GRID <= 0) or np.any(J_GRID > 1.0):
        raise ValueError("J_GRID must be in (0,1]")

    calc = AppendixA(P_star=P_star, x_D=x_D, g0_bw=g0_bw, n_theta=n_theta, n_x=n_x)

    nx = X_TABLE.size
    nj = J_GRID.size

    out: Dict[str, Array] = {
        "eps1_star": np.empty((nx, nj)),
        "eps2_star": np.empty((nx, nj)),
        "eps2_star2": np.empty((nx, nj)),
        "j1_star": np.empty((nx, nj)),
        "j2_star": np.empty((nx, nj)),
        "j2_star2": np.empty((nx, nj)),
        "zeta_star2": np.empty((nx, nj)),
        "x_p": np.empty((nx, nj)),
        "x_ap": np.empty((nx, nj)),
    }

    for i, x in enumerate(X_TABLE):
        for k, j in enumerate(J_GRID):
            c = calc.coeffs_star(float(x), float(j))
            for key in out:
                out[key][i, k] = c[key]

    return out


def quick_checks(tables: Dict[str, Array], X_TABLE: Array, J_GRID: Array) -> Dict[str, float]:
    """
    A couple of quick “shape” checks:
      - As j->1: eps2*/j2* -> 2x
      - As j->0: j2*^2 / (2 j j1*) -> 1
    """
    X_TABLE = np.asarray(X_TABLE, dtype=float)
    J_GRID = np.asarray(J_GRID, dtype=float)

    k1 = int(np.argmax(J_GRID))
    ratio1 = np.median(tables["eps2_star"][:, k1] / (tables["j2_star"][:, k1] + _TINY) / (2.0 * X_TABLE))

    k0 = int(np.argmin(J_GRID))
    ratio0 = np.median(tables["j2_star2"][:, k0] / (2.0 * J_GRID[k0] * tables["j1_star"][:, k0] + _TINY))

    return {"A12a_like_median": float(ratio1), "A13_like_median": float(ratio0)}


# -------------------------
# Helpers to reproduce Table 1 using your pasted Table 2 values
# -------------------------

def make_gbar_from_table2() -> Callable[[Array], Array]:
    """
    Builds a smooth gbar(x) from the Table 2 central values you pasted.
    Uses log-log interpolation (log g vs log x) with mild extrapolation.
    """
    x = np.array([
        2.25e-1, 3.03e-1, 4.95e-1, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
        1.21e1, 1.97e1, 4.16e1, 5.03e1, 6.46e1, 9.36e1, 1.98e2, 2.87e2,
        3.56e2, 4.80e2, 7.84e2, 1.65e3, 2.00e3, 2.57e3, 3.73e3
    ], dtype=float)

    g = np.array([
        1.00, 1.07, 1.13, 1.60, 1.34, 1.37, 1.55, 2.11, 2.22, 2.20,
        2.41, 3.00, 3.50, 3.79, 3.61, 3.66, 4.03, 3.98, 3.31, 2.92,
        2.35, 1.57, 0.85, 0.74, 0.20
    ], dtype=float)

    lx = np.log(x)
    lg = np.log(np.maximum(g, 1e-12))

    # end slopes for extrapolation in log-log
    slope_lo = (lg[1] - lg[0]) / (lx[1] - lx[0])
    slope_hi = (lg[-1] - lg[-2]) / (lx[-1] - lx[-2])

    def gbar(xx: Array) -> Array:
        xx = np.asarray(xx, dtype=float)
        out = np.empty_like(xx)

        m_lo = xx <= x[0]
        m_hi = xx >= x[-1]
        m_mid = (~m_lo) & (~m_hi)

        if np.any(m_mid):
            out[m_mid] = np.exp(np.interp(np.log(xx[m_mid]), lx, lg))

        if np.any(m_lo):
            # extrapolate downward
            ll = np.log(np.maximum(xx[m_lo], 1e-300))
            out[m_lo] = np.exp(lg[0] + slope_lo * (ll - lx[0]))

        if np.any(m_hi):
            # extrapolate upward (still positive, usually decaying by here)
            ll = np.log(xx[m_hi])
            out[m_hi] = np.exp(lg[-1] + slope_hi * (ll - lx[-1]))

        return out

    return gbar


def print_table1_like(X: Array, J: Array, tables: Dict[str, Array]) -> None:
    """
    Print a Table-1-like block:
      x, j, -eps1*, eps2*, j1*, j2*, zeta*2
    """
    X = np.asarray(X, dtype=float)
    J = np.asarray(J, dtype=float)

    print("   x        j        -eps1*       eps2*        j1*         j2*       zeta*2")
    for i, x in enumerate(X):
        for k, j in enumerate(J):
            meps1 = -tables["eps1_star"][i, k]
            eps2  =  tables["eps2_star"][i, k]
            j1    =  tables["j1_star"][i, k]
            j2    =  tables["j2_star"][i, k]
            z2    =  tables["zeta_star2"][i, k]
            print(f"{x:8.3g}  {j:7.3f}  {meps1:11.3e}  {eps2:11.3e}  {j1:11.3e}  {j2:11.3e}  {z2:11.3e}")
        print()
