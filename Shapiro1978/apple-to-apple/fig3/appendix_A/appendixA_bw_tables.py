# appendixA_bw_tables.py
"""
Compute the Appendix-A (Shapiro & Marchant 1978) dimensionless orbit-averaged
diffusion-coefficient tables for an isotropized field-star DF \bar g(x').

You provide:
  - g0_bw(x): your bound, isotropized DF for x>0 (vectorized over numpy arrays)
  - P_star:  dimensionless parameter from the paper (their P_*)
  - x_D:     cutoff in x' (no stars with x'>x_D)

Assumptions (as in the paper):
  - test star: bound x>0, 0<j<=1
  - unbound field stars (x'<=0): \bar g(x') = exp(x')
  - bound field stars:           \bar g(x') = g0_bw(x') for 0<x'<=x_D
  - for x'>x_D:                  \bar g(x') = 0

Outputs are in the paper’s “starred” units:
  eps1_star   = epsilon_1 / v0^2
  eps2_star2  = epsilon_2^2 / v0^4      (and eps2_star = sqrt(eps2_star2))
  j1_star     = j_1 / Jmax(E)
  j2_star2    = j_2^2 / Jmax(E)^2       (and j2_star = sqrt(j2_star2))
  zeta_star2  = zeta^2 / (v0^2 Jmax(E)) (their \zeta^{*2})

Numerics:
  - Gauss–Legendre in x' and theta, fixed order.
  - Pure numpy (no SciPy required).

If you need to experiment with the “factor-of-2” isotropization ambiguity, use
g0_scale and/or unbound_scale (defaults are 1.0).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

Array = np.ndarray


# -------------------------
# Quadrature helpers
# -------------------------


def _leggauss(n: int) -> Tuple[Array, Array]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x.astype(float), w.astype(float)


@dataclass(frozen=True)
class ThetaQuad:
    """Gauss–Legendre quadrature on theta in [0, pi/2], cached."""
    n: int = 48

    def __post_init__(self):
        if self.n < 8:
            raise ValueError("n_theta should be >= 8")
        t, w = _leggauss(self.n)
        theta = 0.25 * np.pi * (t + 1.0)
        ww = 0.25 * np.pi * w
        s = np.sin(theta)
        c = np.cos(theta)
        object.__setattr__(self, "_theta", theta)
        object.__setattr__(self, "_w", ww)
        object.__setattr__(self, "_sin2", s * s)
        object.__setattr__(self, "_cos2", c * c)
        object.__setattr__(self, "_cos4", (c * c) * (c * c))

    @property
    def theta(self) -> Array:
        return self._theta

    @property
    def w(self) -> Array:
        return self._w

    @property
    def sin2(self) -> Array:
        return self._sin2

    @property
    def cos2(self) -> Array:
        return self._cos2

    @property
    def cos4(self) -> Array:
        return self._cos4


@dataclass(frozen=True)
class GLQuad:
    """Gauss–Legendre nodes/weights on [-1,1], cached."""
    n: int = 48

    def __post_init__(self):
        if self.n < 8:
            raise ValueError("n_x should be >= 8")
        x, w = _leggauss(self.n)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_w", w)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def w(self) -> Array:
        return self._w

    def map(self, a: float, b: float) -> Tuple[Array, Array, float]:
        """Map nodes/weights from [-1,1] to [a,b]. Returns (x_mapped, w_ref, halfwidth)."""
        if b <= a:
            return np.empty(0), np.empty(0), 0.0
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xm = mid + half * self.x
        return xm, self.w, half


def _as_array_callable(f: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Ensure f can accept numpy arrays. If it only accepts scalars, wrap it."""
    def wrapped(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        try:
            y = f(x)
            return np.asarray(y, dtype=float)
        except Exception:
            # scalar fallback
            vf = np.vectorize(lambda t: float(f(float(t))), otypes=[float])
            return vf(x)
    return wrapped


# -------------------------
# Orbit geometry (paper eq. 7 and Appendix A eq. A7)
# -------------------------


def x_p_x_ap(x: float, j: float) -> Tuple[float, float]:
    """
    Compute x_p and x_ap from the standard Kepler orbit relations used by the paper.

    For their variables:
      r_{ap,p} = (r_a/(2x)) [1 ± sqrt(1-j^2)]
      x_{ap,p} = r_a/r_{ap,p} = 2x / [1 ± sqrt(1-j^2)]

    j->0: x_ap -> x, x_p -> +inf
    j=1 : x_ap = x_p = 2x
    """
    if x <= 0.0:
        raise ValueError("Require x>0")
    if not (0.0 < j <= 1.0):
        raise ValueError("Require 0<j<=1")
    s = np.sqrt(max(0.0, 1.0 - j * j))
    denom_p = max(1e-300, 1.0 - s)
    denom_ap = 1.0 + s
    return 2.0 * x / denom_p, 2.0 * x / denom_ap


def _inv_x1_x2_x3(x: float, xprime: float, x_p: float, x_ap: float) -> Tuple[float, float, float]:
    """
    Return (a1,a2,a3) where:
      a1 = 1/x1 = 1/max(x',x_ap) - 1/x_p
      a2 = 1/x2 = 1/min(x',x_ap) - 1/x_p
      a3 = 1/x3 = 1/x - 1/x_p

    Using inverse variables avoids overflow when x1,x2 become huge (j~1).
    """
    xmax = x_ap if xprime < x_ap else xprime
    xmin = xprime if xprime < x_ap else x_ap

    a1 = (1.0 / xmax) - (1.0 / x_p)
    a2 = (1.0 / xmin) - (1.0 / x_p)
    a3 = (1.0 / x) - (1.0 / x_p)

    tiny = 1e-300
    a1 = max(a1, tiny)
    a2 = max(a2, tiny)
    a3 = max(a3, tiny)
    return a1, a2, a3


def _inv_x1_for_I7(x_p: float, x_ap: float) -> float:
    """For I7 (region x'<x), max(x',x_ap)=x_ap, so a1 = 1/x_ap - 1/x_p."""
    tiny = 1e-300
    return max((1.0 / x_ap) - (1.0 / x_p), tiny)


# -------------------------
# Appendix A calculator
# -------------------------


@dataclass
class AppendixA:
    """
    Calculator for the starred diffusion coefficients at a single (x,j).

    Parameters
    ----------
    P_star : float
        The paper's dimensionless P_*.
    x_D : float
        Cutoff in x' for bound field stars.
    g0_bw : callable
        Your bound isotropized DF for x>0. Must accept numpy arrays.
    g0_scale : float
        Optional multiplier on the bound DF (to explore isotropization prefactors).
    unbound_scale : float
        Optional multiplier on exp(x') for x'<=0.
    n_theta, n_x : int
        Quadrature orders for theta and x' integrals.
    """
    P_star: float
    x_D: float
    g0_bw: Callable[[Array], Array]
    g0_scale: float = 1.0
    unbound_scale: float = 1.0
    n_theta: int = 48
    n_x: int = 48

    def __post_init__(self):
        self.g0_bw = _as_array_callable(self.g0_bw)
        self.theta = ThetaQuad(self.n_theta)
        self.glx = GLQuad(self.n_x)

    # ---- isotropized field-star DF \bar g(x') ----

    def bar_g(self, xprime: Array) -> Array:
        """
        \bar g(x') with paper assumptions:
          x'<=0 : unbound_scale * exp(x')
          0<x'<=x_D : g0_scale * g0_bw(x')
          x'>x_D : 0
        """
        xprime = np.asarray(xprime, dtype=float)
        out = np.zeros_like(xprime)

        m_unb = xprime <= 0.0
        if np.any(m_unb):
            out[m_unb] = self.unbound_scale * np.exp(xprime[m_unb])

        m_bnd = (xprime > 0.0) & (xprime <= self.x_D)
        if np.any(m_bnd):
            out[m_bnd] = self.g0_scale * self.g0_bw(xprime[m_bnd])

        return out

    def bar_g_pos(self, xprime_pos: Array) -> Array:
        """Bound part only (assumes xprime_pos>0)."""
        xprime_pos = np.asarray(xprime_pos, dtype=float)
        y = self.g0_scale * self.g0_bw(xprime_pos)
        y = np.asarray(y, dtype=float)
        y = np.where((xprime_pos > 0.0) & (xprime_pos <= self.x_D), y, 0.0)
        return y

    def _int_bar_g_minusinf_to_x(self, x: float) -> float:
        """
        ∫_{-∞}^x \bar g(x') dx' for x>0.
        For x>0:
          ∫_{-∞}^0 unbound_scale*exp(x') dx' = unbound_scale
          plus ∫_0^{min(x,x_D)} g0_scale*g0_bw(x') dx'
        """
        if x <= 0.0:
            return float(self.unbound_scale * np.exp(x))
        xb = min(x, self.x_D)
        if xb <= 0.0:
            return float(self.unbound_scale)
        xp, w, half = self.glx.map(0.0, xb)
        return float(self.unbound_scale + half * np.sum(w * self.bar_g_pos(xp)))

    # ---- analytic integrals I1 and I4 ----

    @staticmethod
    def I1(x: float) -> float:
        return 0.25 * np.pi * x ** (-1.5)

    @staticmethod
    def I4(x: float) -> float:
        return 0.25 * np.pi * x ** (-0.5)

    # ---- I7 (depends on x,j only) ----

    def I7(self, x: float, x_p: float, x_ap: float) -> float:
        """
        Stable form:
          I7 = sqrt(1/x) * ∫ ( (1/x_p) + (1/x1)*sin^2θ )^3 dθ
        because the x_p^3 cancellation is done analytically.
        """
        th = self.theta
        a1 = _inv_x1_for_I7(x_p=x_p, x_ap=x_ap)  # a1 = 1/x1
        inv_xp = 1.0 / x_p

        # z1s = z1/x_p = 1/x_p + (1/x1) sin^2θ
        z1s = inv_xp + a1 * th.sin2
        return float(np.sqrt(1.0 / x) * np.sum(th.w * (z1s ** 3)))

    # ---- The set of theta-integrals I2..I3..I16 for a given x' ----

    def Iset(self, x: float, j: float, xprime: float, x_p: float, x_ap: float) -> Dict[str, float]:
        """
        Stable evaluation of I2..I3..I16 using inverse variables (a1,a2,a3)
        and scaled z1 so all x_p powers cancel before forming big intermediates.
        """
        th = self.theta

        # a1=1/x1, a2=1/x2, a3=1/x3
        a1, a2, a3 = _inv_x1_x2_x3(x=x, xprime=xprime, x_p=x_p, x_ap=x_ap)

        sin2 = th.sin2
        cos2 = th.cos2
        cos4 = th.cos4
        w = th.w

        inv_xp = 1.0 / x_p

        # Scaled z1: z1s = z1/x_p = 1/x_p + (1/x1) sin^2θ
        z1s = inv_xp + a1 * sin2

        # Use ratios directly (avoid x2/x1 with huge x2,x1)
        r21 = a1 / a2   # = x2/x1
        r31 = a1 / a3   # = x3/x1

        z2 = 1.0 - r21 * sin2
        z3 = 1.0 - r31 * sin2

        # Numerical safety
        z2 = np.maximum(z2, 1e-300)
        z3 = np.maximum(z3, 1e-300)

        sqrtz2 = np.sqrt(z2)
        sqrtz3 = np.sqrt(z3)
        z2_15 = z2 * sqrtz2        # z2^(3/2)
        z3_15 = z3 * sqrtz3        # z3^(3/2)
        z3_25 = z3_15 * z3         # z3^(5/2)

        sx = float(np.sqrt(xprime))
        xprime_15 = float(xprime * sx)

        invx = 1.0 / x
        invx2 = invx * invx

        # Frequently used inverse combinations
        sqrt_a2_over_a3 = float(np.sqrt(a2 / a3))
        sqrt_a2_a3 = float(np.sqrt(a2 * a3))
        sqrt_a2 = float(np.sqrt(a2))
        sqrt_a3 = float(np.sqrt(a3))
        a2_15 = float(a2 * sqrt_a2)          # a2^(3/2)
        a3_15 = float(a3 * sqrt_a3)          # a3^(3/2)
        a3_25 = float(a3 * a3 * sqrt_a3)     # a3^(5/2)

        # I2: pref = (sqrt(x')/x)*sqrt(a2/a3) and uses z1s (already divided by x_p)
        I2 = (sx * invx) * sqrt_a2_over_a3 * float(np.sum(w * (z1s * sqrtz2 / sqrtz3)))

        # I3: pref = (sqrt(x')/x)*(a1/sqrt(a2 a3)) and uses z1s
        I3 = (sx * invx) * (a1 / sqrt_a2_a3) * float(np.sum(w * (cos2 * z1s / (sqrtz2 * sqrtz3))))

        # I5: pref = (x'^(3/2)/x) * a2^(3/2)/sqrt(a3)
        I5 = (xprime_15 * invx) * (a2_15 / sqrt_a3) * float(np.sum(w * (z2_15 / sqrtz3)))

        # I6: pref = (x'^(3/2)/x) * a1^2 / sqrt(a2 a3)
        I6 = (xprime_15 * invx) * ((a1 * a1) / sqrt_a2_a3) * float(np.sum(w * (cos4 / (sqrtz2 * sqrtz3))))

        # I8: same pref as I2, uses z1s^3
        I8 = (sx * invx) * sqrt_a2_over_a3 * float(np.sum(w * ((z1s ** 3) * sqrtz2 / sqrtz3)))

        # I9: same pref as I3, uses z1s^3
        I9 = (sx * invx) * (a1 / sqrt_a2_a3) * float(np.sum(w * (cos2 * (z1s ** 3) / (sqrtz2 * sqrtz3))))

        # I10: pref = (x'^(3/2)/x^2) * a2^(3/2)/a3^(3/2), uses z1s^3
        I10 = (xprime_15 * invx2) * (a2_15 / a3_15) * float(np.sum(w * ((z1s ** 3) * z2_15 / z3_15)))

        # I11: pref = (x'^(3/2)/x^2) * a1^2 / (sqrt(a2)*a3^(3/2)), uses z1s^3
        I11 = (xprime_15 * invx2) * ((a1 * a1) / (sqrt_a2 * a3_15)) * float(
            np.sum(w * (cos4 * (z1s ** 3) / (sqrtz2 * z3_15)))
        )

        # I12: pref = (2*sqrt(x')/x) * a1^2*sqrt(a2)/a3^(3/2), uses z1s^2
        I12 = (2.0 * sx * invx) * ((a1 * a1) * sqrt_a2 / a3_15) * float(
            np.sum(w * ((sin2 * cos2) * (z1s ** 2) * sqrtz2 / z3_15))
        )

        # I13: pref = (2*x'^(3/2)/x^2) * a1^2*a2^(3/2)/a3^(5/2), uses z1s^2
        I13 = (2.0 * xprime_15 * invx2) * ((a1 * a1) * a2_15 / a3_25) * float(
            np.sum(w * ((cos2 * sin2) * (z1s ** 2) * z2_15 / z3_25))
        )

        # I14: pref = (2*x'^(3/2)/x^2) * a1^3*sqrt(a2)/a3^(5/2), uses z1s^2
        I14 = (2.0 * xprime_15 * invx2) * ((a1 ** 3) * sqrt_a2 / a3_25) * float(
            np.sum(w * ((cos4 * sin2) * (z1s ** 2) * sqrtz2 / z3_25))
        )

        # I15: pref = (x'^(3/2)/x^2) * a2^(3/2)/a3^(3/2), uses z1s
        I15 = (xprime_15 * invx2) * (a2_15 / a3_15) * float(
            np.sum(w * (z1s * z2_15 / z3_15))
        )

        # I16: pref = (x'^(3/2)/x^2) * a1^2/(sqrt(a2)*a3^(3/2)), uses z1s
        I16 = (xprime_15 * invx2) * ((a1 * a1) / (sqrt_a2 * a3_15)) * float(
            np.sum(w * (cos4 * z1s / (sqrtz2 * z3_15)))
        )

        return {
            "I2": float(I2),
            "I3": float(I3),
            "I5": float(I5),
            "I6": float(I6),
            "I8": float(I8),
            "I9": float(I9),
            "I10": float(I10),
            "I11": float(I11),
            "I12": float(I12),
            "I13": float(I13),
            "I14": float(I14),
            "I15": float(I15),
            "I16": float(I16),
        }

    # -------------------------
    # Main starred-coefficient evaluation (Appendix A eqs. A5–A7)
    # -------------------------

    def coeffs_star(self, x: float, j: float) -> Dict[str, float]:
        """Compute (eps1*, eps2*^2, j1*, j2*^2, zeta*^2) at (x,j)."""
        x = float(x)
        j = float(j)
        if x <= 0.0:
            raise ValueError("Require x>0 (bound test star)")
        if not (0.0 < j <= 1.0):
            raise ValueError("Require 0<j<=1")

        x_p, x_ap = x_p_x_ap(x, j)

        # since \bar g(x')=0 for x'>x_D, truncate the x' integrations at x_p_eff:
        x_p_eff = min(x_p, self.x_D)

        # --- Region 1: x' < x  (Appendix A uses I1, I4, I7 here)
        I1 = self.I1(x)
        I4 = self.I4(x)
        I7 = self.I7(x=x, x_p=x_p, x_ap=x_ap)
        G1 = self._int_bar_g_minusinf_to_x(x)

        A_eps1 = I1 * G1
        A_eps2 = I4 * G1
        A_j1 = (2.0 * I7) * G1
        A_j2 = (4.0 * I7) * G1
        A_zeta = I1 * G1

        # --- Region 2: x <= x' < x_ap
        B_eps1 = B_eps2 = B_j1 = B_j2 = B_zeta = 0.0
        if x_ap > x and x_p_eff > x:
            b2 = min(x_ap, x_p_eff)
            if b2 > x:
                xnodes, w, half = self.glx.map(x, b2)
                gvals = self.bar_g_pos(xnodes)
                for xpr, wi, gi in zip(xnodes, w, gvals):
                    if gi == 0.0:
                        continue
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

        # --- Region 3: x_ap <= x' <= x_p_eff
        C_eps1 = C_eps2 = C_j1 = C_j2 = C_zeta = 0.0
        if x_p_eff > x_ap:
            a3 = max(x_ap, x)
            b3 = x_p_eff
            if b3 > a3:
                xnodes, w, half = self.glx.map(a3, b3)
                gvals = self.bar_g_pos(xnodes)
                for xpr, wi, gi in zip(xnodes, w, gvals):
                    if gi == 0.0:
                        continue
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

        # --- Combine with Appendix A prefactors (their eq. A5)
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
    g0_scale: float = 1.0,
    unbound_scale: float = 1.0,
    n_theta: int = 48,
    n_x: int = 48,
) -> Dict[str, Array]:
    """
    Compute coefficient tables on (X_TABLE, J_GRID).

    Returns dict of arrays with shape (len(X_TABLE), len(J_GRID)):
      eps1_star, eps2_star, eps2_star2, j1_star, j2_star, j2_star2, zeta_star2, x_p, x_ap
    """
    X_TABLE = np.asarray(X_TABLE, dtype=float)
    J_GRID = np.asarray(J_GRID, dtype=float)

    if np.any(X_TABLE <= 0.0):
        raise ValueError("X_TABLE must contain only x>0")
    if np.any(J_GRID <= 0.0) or np.any(J_GRID > 1.0):
        raise ValueError("J_GRID must be in (0,1]")

    calc = AppendixA(
        P_star=P_star,
        x_D=x_D,
        g0_bw=g0_bw,
        g0_scale=g0_scale,
        unbound_scale=unbound_scale,
        n_theta=n_theta,
        n_x=n_x,
    )

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


def save_tables_npz(path: str, tables: Dict[str, Array], X_TABLE: Array, J_GRID: Array) -> None:
    """Convenience saver."""
    np.savez(
        path,
        X_TABLE=np.asarray(X_TABLE, dtype=float),
        J_GRID=np.asarray(J_GRID, dtype=float),
        **{k: np.asarray(v) for k, v in tables.items()},
    )


# -------------------------
# Tiny diagnostics (optional)
# -------------------------


def quick_checks(tables: Dict[str, Array], X_TABLE: Array, J_GRID: Array) -> Dict[str, float]:
    """
    Return a couple of diagnostic medians that are commonly used sanity checks.

    These are *not* exact at finite quadrature; they should be "near 1" if you’re
    reproducing the paper’s consistency limits.

    - A12a-like:  eps2*/j2*  ≈ 2x as j->1
    - A13-like:   j2*^2/(2 j j1*) ≈ 1 as j->0
    """
    X_TABLE = np.asarray(X_TABLE, dtype=float)
    J_GRID = np.asarray(J_GRID, dtype=float)

    k1 = int(np.argmax(J_GRID))  # closest to 1
    ratio1 = np.median(
        tables["eps2_star"][:, k1] / (tables["j2_star"][:, k1] + 1e-300) / (2.0 * X_TABLE)
    )

    k0 = int(np.argmin(J_GRID))  # smallest j
    ratio0 = np.median(
        tables["j2_star2"][:, k0] / (2.0 * J_GRID[k0] * tables["j1_star"][:, k0] + 1e-300)
    )

    return {"A12a_like_median": float(ratio1), "A13_like_median": float(ratio0)}
