"""
appendixA_bw_tables.py

Numerical implementation of Shapiro & Marchant (1978) Appendix A (eqs. A5–A7).

Computes the dimensionless orbit-averaged diffusion coefficients:

  eps1_star   = epsilon_1 / v0^2
  eps2_star2  = epsilon_2^2 / v0^4   (and eps2_star = sqrt(eps2_star2))
  j1_star     = j_1 / Jmax(E)
  j2_star2    = j_2^2 / Jmax(E)^2    (and j2_star = sqrt(j2_star2))
  zeta_star2  = zeta^2 / (v0^2 Jmax(E))

Given an isotropized field-star DF gbar(x'):

  - x' <= 0   : gbar = exp(x')   (unbound Maxwellian tail in SM78)
  - 0 < x' <= x_D : gbar = g0_bw(x')
  - x' > x_D : gbar = 0

Key numerical stability features:
--------------------------------
1) Log-space prefactors for any expression containing x1^4, x_p^6, etc.
   This removes OverflowError even when x1 ~ 1e300.

2) Optional regularization for exact circular orbit j=1:
      j_eff = min(j, 1 - j_circular_eps)
   Default j_circular_eps=1e-10.

3) eps1_star sign:
   SM78 Table 1 prints (-eps1*). eps1_star itself is typically negative.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np

Array = np.ndarray

_LOG_MAX = float(np.log(np.finfo(float).max))      # ~709.78
_LOG_MIN_SUB = -745.0                             # safe underflow bound


# -------------------------
# Quadrature helpers
# -------------------------

def _leggauss(n: int) -> Tuple[Array, Array]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x.astype(float), w.astype(float)


@dataclass(frozen=True)
class ThetaQuad:
    """Gauss–Legendre quadrature on theta in [0, pi/2], cached trig."""
    n: int = 48

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
    """Gauss–Legendre quadrature on [-1,1] with mapping to [a,b]."""
    n: int = 48

    @property
    def x(self) -> Array:
        x, _ = _leggauss(self.n)
        return x

    @property
    def w(self) -> Array:
        _, w = _leggauss(self.n)
        return w

    def map(self, a: float, b: float) -> Tuple[Array, Array, float]:
        if b <= a:
            return np.empty(0), np.empty(0), 0.0
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xm = mid + half * self.x
        return xm, self.w, half


def _exp_clip(logx):
    return np.exp(np.clip(logx, _LOG_MIN_SUB, _LOG_MAX))


def _sqrt_ratio_from_logs(log_num, log_den):
    """Return sqrt(exp(log_num - log_den)) with clipping."""
    return _exp_clip(0.5 * (log_num - log_den))


# -------------------------
# Orbit geometry (SM78 eq. 7)
# -------------------------

def x_p_x_ap(x: float, j: float) -> Tuple[float, float]:
    """
    x_{ap,p} = 2x / [1 ± sqrt(1-j^2)].
    - j->0: x_p -> +inf, x_ap -> x
    - j=1:  x_p = x_ap = 2x
    """
    if x <= 0:
        raise ValueError("Require x>0")
    if not (0.0 < j <= 1.0):
        raise ValueError("Require 0<j<=1")
    s = np.sqrt(max(0.0, 1.0 - j * j))
    denom_p = max(1e-300, 1.0 - s)
    denom_ap = 1.0 + s
    return 2.0 * x / denom_p, 2.0 * x / denom_ap


def _x1_x2_x3(x: float, xprime: float, x_p: float, x_ap: float) -> Tuple[float, float, float]:
    """
    SM78 eq. (A7a–c).
    xmax = max(x', x_ap), xmin = min(x', x_ap)
    1/x1 = 1/xmax - 1/xp
    1/x2 = 1/xmin - 1/xp
    1/x3 = 1/x    - 1/xp
    """
    xmax = x_ap if xprime < x_ap else xprime
    xmin = xprime if xprime < x_ap else x_ap

    inv_x1 = (1.0 / xmax) - (1.0 / x_p)
    inv_x2 = (1.0 / xmin) - (1.0 / x_p)
    inv_x3 = (1.0 / x) - (1.0 / x_p)

    inv_x1 = max(inv_x1, 1e-300)
    inv_x2 = max(inv_x2, 1e-300)
    inv_x3 = max(inv_x3, 1e-300)

    return 1.0 / inv_x1, 1.0 / inv_x2, 1.0 / inv_x3


def _x1_for_I7(x_p: float, x_ap: float) -> float:
    """For x'<x, SM78 takes max(x',x_ap)=x_ap in x1."""
    inv_x1 = (1.0 / x_ap) - (1.0 / x_p)
    inv_x1 = max(inv_x1, 1e-300)
    return 1.0 / inv_x1


# -------------------------
# Appendix A calculator
# -------------------------

@dataclass
class AppendixA:
    P_star: float
    x_D: float
    g0_bw: Callable[[Array], Array]
    n_theta: int = 48
    n_x: int = 48
    j_circular_eps: float = 1e-10

    def __post_init__(self):
        self.theta = ThetaQuad(self.n_theta)
        self.glx = GLQuad(self.n_x)

    # ---- isotropized field DF ----
    def bar_g(self, xprime: Array) -> Array:
        xprime = np.asarray(xprime, dtype=float)
        out = np.zeros_like(xprime)
        m_unb = xprime <= 0.0
        out[m_unb] = np.exp(xprime[m_unb])
        m_bnd = (xprime > 0.0) & (xprime <= self.x_D)
        if np.any(m_bnd):
            out[m_bnd] = self.g0_bw(xprime[m_bnd])
        return out

    def _int_bar_g_minusinf_to_x(self, x: float) -> float:
        """Integral ∫_{-∞}^{x} gbar(x') dx' for x>0."""
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

    # ---- I7 ----
    def I7(self, x: float, x_p: float, x_ap: float) -> float:
        th = self.theta
        x1 = _x1_for_I7(x_p, x_ap)
        z1 = 1.0 + (x_p / x1) * th.sin2

        # pref = sqrt(1 / (x * x_p^6)) in log-space
        log_den = np.log(x) + 6.0 * np.log(x_p)
        pref = float(_sqrt_ratio_from_logs(0.0, log_den))
        return pref * float(np.sum(th.w * (z1 ** 3)))

    # ---- Iset (I2..I16) ----
    def Iset(self, x: float, xprime: float, x_p: float, x_ap: float) -> Dict[str, float]:
        th = self.theta
        x1, x2, x3 = _x1_x2_x3(x, xprime, x_p, x_ap)

        sin2 = th.sin2
        cos2 = th.cos2
        cos4 = th.cos4
        w = th.w

        z1 = 1.0 + (x_p / x1) * sin2
        z2 = 1.0 - (x2 / x1) * sin2
        z3 = 1.0 - (x3 / x1) * sin2
        z2 = np.maximum(z2, 1e-300)
        z3 = np.maximum(z3, 1e-300)

        # logs
        lx = np.log(x)
        lxp = np.log(xprime)
        lx1 = np.log(x1)
        lx2 = np.log(x2)
        lx3 = np.log(x3)
        lxp_orb = np.log(x_p)

        pref2  = float(_sqrt_ratio_from_logs(lxp + lx3, 2*lx + 2*lxp_orb + lx2))
        pref3  = float(_sqrt_ratio_from_logs(lxp + lx2 + lx3, 2*lx + 2*lx1 + 2*lxp_orb))
        pref5  = float(_sqrt_ratio_from_logs(3*lxp + lx3, 2*lx + 3*lx2))
        pref6  = float(_sqrt_ratio_from_logs(3*lxp + lx2 + lx3, 2*lx + 4*lx1))
        pref8  = float(_sqrt_ratio_from_logs(lxp + lx3, 2*lx + 6*lxp_orb + lx2))
        pref9  = float(_sqrt_ratio_from_logs(lxp + lx2 + lx3, 2*lx + 6*lxp_orb + 2*lx1))
        pref10 = float(_sqrt_ratio_from_logs(3*lxp + 3*lx3, 4*lx + 6*lxp_orb + 3*lx2))
        pref11 = float(_sqrt_ratio_from_logs(3*lxp + lx2 + 3*lx3, 4*lx + 6*lxp_orb + 4*lx1))
        pref12 = float(_sqrt_ratio_from_logs(np.log(4.0) + lxp + 3*lx3, 2*lx + 4*lxp_orb + 4*lx1 + lx2))
        pref13 = float(_sqrt_ratio_from_logs(np.log(4.0) + 3*lxp + 5*lx3, 4*lx + 4*lxp_orb + 4*lx1 + 3*lx2))
        pref14 = float(_sqrt_ratio_from_logs(np.log(4.0) + 3*lxp + 5*lx3, 4*lx + 4*lxp_orb + 6*lx1 + lx2))
        pref15 = float(_sqrt_ratio_from_logs(3*lxp + 3*lx3, 4*lx + 2*lxp_orb + 3*lx2))
        pref16 = float(_sqrt_ratio_from_logs(3*lxp + lx2 + 3*lx3, 4*lx + 2*lxp_orb + 4*lx1))

        I2  = pref2  * float(np.sum(w * (z1 * np.sqrt(z2) / np.sqrt(z3))))
        I3  = pref3  * float(np.sum(w * (cos2 * z1 / np.sqrt(z2 * z3))))
        I5  = pref5  * float(np.sum(w * ((z2 ** 1.5) / np.sqrt(z3))))
        I6  = pref6  * float(np.sum(w * (cos4 / np.sqrt(z2 * z3))))
        I8  = pref8  * float(np.sum(w * ((z1 ** 3) * np.sqrt(z2) / np.sqrt(z3))))
        I9  = pref9  * float(np.sum(w * (cos2 * (z1 ** 3) / np.sqrt(z2 * z3))))
        I10 = pref10 * float(np.sum(w * ((z1 ** 3) * (z2 ** 1.5) / (z3 ** 1.5))))
        I11 = pref11 * float(np.sum(w * (cos4 * (z1 ** 3) / (np.sqrt(z2) * (z3 ** 1.5)))))
        I12 = pref12 * float(np.sum(w * ((sin2 * cos2) * (z1 ** 2) * np.sqrt(z2) / (z3 ** 1.5))))
        I13 = pref13 * float(np.sum(w * ((cos2 * sin2) * (z1 ** 2) * (z2 ** 1.5) / (z3 ** 2.5))))
        I14 = pref14 * float(np.sum(w * ((cos4 * sin2) * (z1 ** 2) * np.sqrt(z2) / (z3 ** 2.5))))
        I15 = pref15 * float(np.sum(w * (z1 * (z2 ** 1.5) / (z3 ** 1.5))))
        I16 = pref16 * float(np.sum(w * (cos4 * z1 / (np.sqrt(z2) * (z3 ** 1.5)))))

        return dict(I2=I2, I3=I3, I5=I5, I6=I6, I8=I8, I9=I9, I10=I10, I11=I11,
                    I12=I12, I13=I13, I14=I14, I15=I15, I16=I16)

    # ---- main coefficients ----
    def coeffs_star(self, x: float, j: float) -> Dict[str, float]:
        x = float(x)
        j = float(j)
        if x <= 0:
            raise ValueError("Require x>0")
        if not (0.0 < j <= 1.0):
            raise ValueError("Require 0<j<=1")

        # regularize exact j=1
        if self.j_circular_eps and j >= 1.0:
            j_eff = 1.0 - self.j_circular_eps
        else:
            j_eff = j

        x_p, x_ap = x_p_x_ap(x, j_eff)
        x_p_eff = min(x_p, self.x_D)

        # Region 1
        I1 = self.I1(x)
        I4 = self.I4(x)
        I7 = self.I7(x, x_p, x_ap)
        G1 = self._int_bar_g_minusinf_to_x(x)

        A_eps1 = I1 * G1
        A_eps2 = I4 * G1
        A_j1   = (2.0 * I7) * G1
        A_j2   = (4.0 * I7) * G1
        A_zeta = I1 * G1

        # Region 2
        B_eps1 = B_eps2 = B_j1 = B_j2 = B_zeta = 0.0
        if x_ap > x and x_p_eff > x:
            b2 = min(x_ap, x_p_eff)
            if b2 > x:
                xnodes, w, half = self.glx.map(x, b2)
                gvals = self.g0_bw(xnodes)
                for xpr, wi, gi in zip(xnodes, w, gvals):
                    Is = self.Iset(x, float(xpr), x_p, x_ap)
                    B_eps1 += wi * gi * (-Is["I2"])
                    B_eps2 += wi * gi * ( Is["I5"])
                    B_j1   += wi * gi * (6*Is["I12"] - 9*Is["I8"] - Is["I10"])
                    B_j2   += wi * gi * (3*Is["I12"] + 4*Is["I10"] - 3*Is["I13"])
                    B_zeta += wi * gi * ( Is["I15"])
                B_eps1 *= half; B_eps2 *= half; B_j1 *= half; B_j2 *= half; B_zeta *= half

        # Region 3
        C_eps1 = C_eps2 = C_j1 = C_j2 = C_zeta = 0.0
        if x_p_eff > x_ap:
            a3 = max(x_ap, x)
            b3 = x_p_eff
            if b3 > a3:
                xnodes, w, half = self.glx.map(a3, b3)
                gvals = self.g0_bw(xnodes)
                for xpr, wi, gi in zip(xnodes, w, gvals):
                    Is = self.Iset(x, float(xpr), x_p, x_ap)
                    C_eps1 += wi * gi * (-Is["I3"])
                    C_eps2 += wi * gi * ( Is["I6"])
                    C_j1   += wi * gi * (6*Is["I12"] - 9*Is["I9"] - Is["I11"])
                    C_j2   += wi * gi * (3*Is["I12"] + 4*Is["I11"] - 3*Is["I14"])
                    C_zeta += wi * gi * ( Is["I16"])
                C_eps1 *= half; C_eps2 *= half; C_j1 *= half; C_j2 *= half; C_zeta *= half

        sqrt2pi = float(np.sqrt(2*np.pi))

        # SM78 A5 + eps1 sign so that Table 1 prints (-eps1*)
        eps1_star  = -3.0 * sqrt2pi * self.P_star * (A_eps1 + B_eps1 + C_eps1)
        eps2_star2 =  4.0 * sqrt2pi * self.P_star * (A_eps2 + B_eps2 + C_eps2)

        j1_star    = sqrt2pi * (x / j_eff) * self.P_star * (A_j1 + B_j1 + C_j1)
        j2_star2   = sqrt2pi * x * self.P_star * (A_j2 + B_j2 + C_j2)
        zeta_star2 = 2.0 * sqrt2pi * j_eff * self.P_star * (A_zeta + B_zeta + C_zeta)

        eps2_star2 = max(0.0, float(eps2_star2))
        j2_star2   = max(0.0, float(j2_star2))

        return dict(
            eps1_star=float(eps1_star),
            eps2_star2=float(eps2_star2),
            eps2_star=float(np.sqrt(eps2_star2)),
            j1_star=float(j1_star),
            j2_star2=float(j2_star2),
            j2_star=float(np.sqrt(j2_star2)),
            zeta_star2=float(zeta_star2),
            x_p=float(x_p),
            x_ap=float(x_ap),
            j_eff=float(j_eff),
        )


def compute_tables(
    X_TABLE: Array,
    J_GRID: Array,
    *,
    g0_bw: Callable[[Array], Array],
    P_star: float,
    x_D: float,
    n_theta: int = 48,
    n_x: int = 48,
    j_circular_eps: float = 1e-10,
) -> Dict[str, Array]:
    X_TABLE = np.asarray(X_TABLE, dtype=float)
    J_GRID  = np.asarray(J_GRID, dtype=float)

    calc = AppendixA(P_star=P_star, x_D=x_D, g0_bw=g0_bw,
                     n_theta=n_theta, n_x=n_x, j_circular_eps=j_circular_eps)

    nx, nj = X_TABLE.size, J_GRID.size
    out = {k: np.empty((nx, nj)) for k in
           ["eps1_star","eps2_star","eps2_star2","j1_star","j2_star","j2_star2","zeta_star2","x_p","x_ap","j_eff"]}

    for i,x in enumerate(X_TABLE):
        for k,j in enumerate(J_GRID):
            c = calc.coeffs_star(float(x), float(j))
            for key in out:
                out[key][i,k] = c[key]
    return out


def quick_checks(tables: Dict[str, Array], X_TABLE: Array, J_GRID: Array) -> Dict[str, float]:
    X_TABLE = np.asarray(X_TABLE, dtype=float)
    J_GRID  = np.asarray(J_GRID, dtype=float)

    k1 = int(np.argmax(J_GRID))
    r1 = np.median(tables["eps2_star"][:, k1] / (tables["j2_star"][:, k1] + 1e-300) / (2.0 * X_TABLE))

    k0 = int(np.argmin(J_GRID))
    r0 = np.median(tables["j2_star2"][:, k0] / (2.0 * J_GRID[k0] * tables["j1_star"][:, k0] + 1e-300))

    return {"A12a_like_median": float(r1), "A13_like_median": float(r0)}

# ---------------------------------------------------------------------
# Helpers: build g0_bw(x) from SM78 Table 2 and print "Table 1"-like block
# ---------------------------------------------------------------------

# SM78 Table 2 (central values only)
TABLE2_X = np.array([
    2.25e-1, 3.03e-1, 4.95e-1, 1.04e0, 1.26e0, 1.62e0, 2.35e0,
    5.00e0, 7.20e0, 8.94e0, 1.21e1, 1.97e1, 4.16e1, 5.03e1,
    6.46e1, 9.36e1, 1.98e2, 2.87e2, 3.56e2, 4.80e2, 7.84e2,
    1.65e3, 2.00e3, 2.57e3, 3.73e3
], dtype=float)

TABLE2_G = np.array([
    1.00, 1.07, 1.13, 1.60, 1.34, 1.37, 1.55,
    2.11, 2.22, 2.20, 2.41, 3.00, 3.50, 3.79,
    3.61, 3.66, 4.03, 3.98, 3.31, 2.92, 2.35,
    1.57, 0.85, 0.74, 0.20
], dtype=float)


def make_gbar_from_table2(
    *,
    x_tab: Array = TABLE2_X,
    g_tab: Array = TABLE2_G,
    extrapolate: str = "slope",
    below_behavior: str = "slope",
    g_floor: float = 1e-300,
) -> Callable[[Array], Array]:
    """
    Return a callable g0_bw(x) for x>0 built from SM78 Table 2.

    Interpolation is performed in (log x, log g) space.

    Parameters
    ----------
    extrapolate : {"slope","flat"}
        Behavior for x > max(x_tab).
        - "slope": extend last log-log slope
        - "flat":  hold g constant at last value
    below_behavior : {"slope","flat"}
        Behavior for 0 < x < min(x_tab).
        - "slope": extend first log-log slope
        - "flat":  hold g constant at first value (often ~1)
    g_floor : float
        Floor for g before taking logs.

    Notes
    -----
    This returns ONLY the bound-star isotropized DF for x>0.
    The module's AppendixA.bar_g() already handles x'<=0 via exp(x').
    """
    x_tab = np.asarray(x_tab, dtype=float)
    g_tab = np.asarray(g_tab, dtype=float)

    if np.any(x_tab <= 0):
        raise ValueError("x_tab must be > 0")
    if np.any(g_tab <= 0):
        raise ValueError("g_tab must be > 0 (use g_floor if needed)")

    # sort
    idx = np.argsort(x_tab)
    x_tab = x_tab[idx]
    g_tab = g_tab[idx]

    lx = np.log(x_tab)
    lg = np.log(np.clip(g_tab, g_floor, None))

    # end slopes in log-log
    slope_lo = (lg[1] - lg[0]) / (lx[1] - lx[0])
    slope_hi = (lg[-1] - lg[-2]) / (lx[-1] - lx[-2])

    def g0_bw(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)

        m = x > 0.0
        if not np.any(m):
            return out

        xx = x[m]
        lxx = np.log(xx)

        # inside range: interpolate
        lxx_clip = np.clip(lxx, lx[0], lx[-1])
        lg_i = np.interp(lxx_clip, lx, lg)

        # below range
        mb = lxx < lx[0]
        if np.any(mb):
            if below_behavior == "flat":
                lg_i[mb] = lg[0]
            elif below_behavior == "slope":
                lg_i[mb] = lg[0] + slope_lo * (lxx[mb] - lx[0])
            else:
                raise ValueError("below_behavior must be 'slope' or 'flat'")

        # above range
        ma = lxx > lx[-1]
        if np.any(ma):
            if extrapolate == "flat":
                lg_i[ma] = lg[-1]
            elif extrapolate == "slope":
                lg_i[ma] = lg[-1] + slope_hi * (lxx[ma] - lx[-1])
            else:
                raise ValueError("extrapolate must be 'slope' or 'flat'")

        out[m] = np.exp(np.clip(lg_i, _LOG_MIN_SUB, _LOG_MAX))
        return out

    return g0_bw


def print_table1_like(
    *,
    g0_bw: Callable[[Array], Array],
    P_star: float,
    x_D: float,
    x_values: Array = np.array([3.36e-1, 3.31e0, 3.27e1, 3.23e2, 3.18e3], dtype=float),
    j_values: Array = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=float),
    n_theta: int = 48,
    n_x: int = 48,
    j_circular_eps: float = 1e-10,
) -> None:
    """
    Compute and print the same 5x5 block structure as SM78 Table 1.

    Columns printed:
      x, j, -eps1*, eps2*, j1*, j2*, zeta*2
    """
    calc = AppendixA(
        P_star=P_star,
        x_D=x_D,
        g0_bw=g0_bw,
        n_theta=n_theta,
        n_x=n_x,
        j_circular_eps=j_circular_eps,
    )

    print("   x        j        -eps1*       eps2*        j1*         j2*       zeta*2")
    for xv in x_values:
        for jv in j_values:
            c = calc.coeffs_star(float(xv), float(jv))
            print(f"{xv:8.3g}  {jv:7.3f}  {-c['eps1_star']: .3e}  {c['eps2_star']: .3e}"
                  f"  {c['j1_star']: .3e}  {c['j2_star']: .3e}  {c['zeta_star2']: .3e}")
        print()
