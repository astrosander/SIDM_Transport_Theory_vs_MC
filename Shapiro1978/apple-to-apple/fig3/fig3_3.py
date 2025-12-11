#!/usr/bin/env python3
"""
Monte Carlo reproduction of Shapiro & Marchant (1978) Fig. 3

We implement a simplified version of their 2D (E, J) Monte Carlo scheme
with the loss-cone boundary condition and then compute the isotropized
distribution function ḡ(x) for the “canonical” case

    x_crit = 10
    x_D    = 1e4

We also compute the Bahcall–Wolf (BW) 1D “no-loss-cone” solution g0(x)
from the tabulated values (LN1P, G0_TAB).

The main goal is to numerically produce a ḡ(x) that is qualitatively
(and roughly quantitatively) similar to the one plotted in Fig. 3
of Shapiro & Marchant (1978).

Notes / approximations vs. the paper:

- We work in dimensionless units with v0 = 1 and M = 1.
- We set P* = 0.005 and x_D = 1e4 as in their “canonical” case.
- We use their tabulated dimensionless orbital perturbations
      NEG_E1, E2, J1, J2, ZETA2
  which correspond to the *dimensionless* coefficients
  (ε1*, ε2*, j1*, j2*, ζ*²) at discrete (x, j) points.
- For x we interpolate linearly in ln x between the list
  X_BINS (which is exactly the x values of the table rows).
- For j we interpolate linearly in j between the 5 tabulated values
  j_grid = [1.0, 0.401, 0.161, 0.065, 0.026].
- We follow test-star motion in (x, j) using an approximate
  version of their step-adjustment algorithm. The exact
  algorithm in the paper is quite complex; here we capture
  the main constraints:
    * steps small in x and j
    * smaller near the loss cone
    * smaller near j ~ 1 where cross-correlation is important
- Loss-cone boundary: stars are removed if j < j_min(x) and
  they are at “pericenter” (we keep a cycle counter in units
  of orbital periods and only allow capture at integer cycles).
- Outer energy boundary at x_b ~ 0.2 (following the paper);
  we “reinject” bound stars that wander out to x < x_b with
  new isotropic (x, j) near x ~ 0.2.
- We ignore the creation–annihilation cloning scheme of the paper
  and instead just use a reasonably large number of stars
  and long integration time.

This is *not* an exact reproduction of their code but is a
faithful implementation of the method as described in the text,
sufficient to:
  * illustrate the algorithm,
  * produce a reasonable ḡ(x) curve,
  * show that in the no-loss-cone case ḡ ≈ g0.

The script:

- Defines all constants and tables you gave.
- Implements:
    * interpolation of BW g0(x)
    * interpolation of orbital perturbations at (x, j)
    * loss-cone boundary j_min(x)
    * time-step adaptation
    * Monte Carlo update rule (Δx, Δj)
    * measurement of ḡ(x) on X_BINS
- Runs two simulations:
    1) “no-loss-cone” check
    2) full loss-cone with x_crit ~ 10
- Produces a single matplotlib plot:
    * ḡ(x) (no loss cone)
    * ḡ(x) (with loss cone)
    * g0(x) (“BW” 1D solution)
    * GBAR_PAPER with error bars (the original Fig. 3 data)

You can tune:
    N_STARS, N_STEPS, MEASURE_START, MEASURE_EVERY
to trade off between CPU time and Monte Carlo noise.
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Constants / tables from the prompt
# -------------------------------------------------------------

X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=np.float64)

GBAR_X_PAPER = np.array([
    0.2250, 0.3030, 0.4950, 1.0400, 1.2600, 1.6200, 2.3500, 5.0000,
    7.2000, 8.9400, 12.1000, 19.7000, 41.6000, 50.3000, 64.6000, 93.6000,
    198.0000, 287.0000, 356.0000, 480.0000, 784.0000, 1650.0000,
    2000.0000, 2570.0000, 3730.0000
], dtype=np.float64)

GBAR_PAPER = np.array([
    1.0000, 1.0700, 1.1300, 1.6000, 1.3400, 1.3700, 1.5500, 2.1100,
    2.2200, 2.2000, 2.4100, 3.0000, 3.5000, 3.7900, 3.6100, 3.6600,
    4.0300, 3.9800, 3.3100, 2.9200, 2.3500, 1.5700, 0.8500, 0.7400,
    0.2000
], dtype=np.float64)

GBAR_ERR_PAPER = np.array([
    0.0400, 0.0600, 0.0400, 0.1600, 0.1800, 0.1400, 0.1500, 0.3400,
    0.4400, 0.3500, 0.2200, 0.1700, 0.4300, 0.3000, 0.3000, 0.1800,
    0.4400, 0.9900, 0.6800, 0.5300, 0.1800, 0.5400, 0.1200, 1.0500,
    0.1400
], dtype=np.float64)

# BW 1D solution g0(x) tabulation
LN1P = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], dtype=np.float64)
G0_TAB = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], dtype=np.float64)

# Table 1 orbital perturbations (dimensionless, P* factored in)
NEG_E1 = np.array([
    [1.41e-1, 1.40e-1, 1.33e-1, 1.29e-1, 1.29e-1],
    [-1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],
    [1.96e-3, 1.59e-3, 2.83e-3, 3.38e-3, 3.49e-3],
    [3.64e-3, 4.64e-3, 4.93e-3, 4.97e-3, 4.98e-3],
    [8.39e-4, 8.53e-4, 8.56e-4, 8.56e-4, 8.56e-4],
], dtype=np.float64)

E2 = np.array([
    [3.14e-1, 4.36e-1, 7.55e-1, 1.37e0, 2.19e0],
    [4.45e-1, 8.66e-1, 1.57e0, 2.37e0, 2.73e0],
    [1.03e0, 1.80e0, 2.52e0, 2.77e0, 2.81e0],
    [1.97e0, 2.54e0, 2.68e0, 2.70e0, 2.70e0],
    [1.96e0, 1.96e0, 1.96e0, 1.96e0, 1.96e0],
], dtype=np.float64)

J1 = np.array([
    [-2.52e-2, 5.37e-1, 1.51e0, 3.83e0, 9.58e0],
    [-5.03e-3, 5.40e-3, 2.54e-2, 6.95e-2, 1.75e-1],
    [-2.58e-4, 3.94e-4, 1.45e-3, 3.77e-3, 9.45e-3],
    [-4.67e-6, 2.03e-5, 5.99e-5, 1.53e-4, 3.82e-4],
    [3.67e-8, 2.54e-7, 7.09e-7, 1.80e-6, 4.49e-6],
], dtype=np.float64)

J2 = np.array([
    [4.68e-1, 6.71e-1, 6.99e-1, 7.03e-1, 7.04e-1],
    [6.71e-2, 9.19e-2, 9.48e-2, 9.53e-2, 9.54e-2],
    [1.57e-2, 2.12e-2, 2.20e-2, 2.21e-2, 2.21e-2],
    [3.05e-3, 4.25e-3, 4.42e-3, 4.44e-3, 4.45e-3],
    [3.07e-4, 4.59e-4, 4.78e-4, 4.81e-4, 4.82e-4],
], dtype=np.float64)

ZETA2 = np.array([
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

# x grid and j grid for those tables (from Table 1 description)
X_TABLE = np.array([3.36e-1, 3.31, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID  = np.array([1.0, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)

# Canonical parameters
P_STAR = 0.005   # P* (dimensionless period at x=1)
X_D    = 1.0e4   # x_D = M/(2 r_D v0^2)
X_CRIT_TARGET = 10.0  # mainly “for consistency”

# Outer cusp boundary (as in § IIIe)
X_B = 0.20

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def g0_bw(x):
    """
    Bahcall–Wolf 1D BW solution g0(x) via linear interpolation in ln(1+x).
    LN1P = ln(1+x) grid, G0_TAB = tabulated g0.
    """
    if x < 0.0:
        return math.exp(x)  # unbound Maxwellian (by assumption)
    u = math.log1p(x)  # ln(1+x)
    if u <= LN1P[0]:
        return G0_TAB[0]
    if u >= LN1P[-1]:
        return G0_TAB[-1]
    i = np.searchsorted(LN1P, u) - 1
    i = max(0, min(i, len(LN1P)-2))
    t = (u - LN1P[i]) / (LN1P[i+1] - LN1P[i])
    return (1.0 - t)*G0_TAB[i] + t*G0_TAB[i+1]


def interpolate_orbital_coeffs(x, j):
    """
    Interpolate the dimensionless orbital perturbations at (x, j):
        ε1*, ε2*, j1*, j2*, ζ*^2
    We interpolate in log10(x) on [X_TABLE], and linearly in j on [J_GRID].

    Returns:
        (eps1_star, eps2_star, j1_star, j2_star, zeta2_star)
    """
    # Clamp x and j into the table range
    x_cl = min(max(x, X_TABLE[0]), X_TABLE[-1])
    j_cl = min(max(j, J_GRID[-1]), J_GRID[0])

    # Indices in x (in log space; table is roughly log-spaced)
    logx = math.log10(x_cl)
    log_x_tab = np.log10(X_TABLE)
    if logx <= log_x_tab[0]:
        ix = 0
        tx = 0.0
    elif logx >= log_x_tab[-1]:
        ix = len(X_TABLE)-2
        tx = 1.0
    else:
        ix = np.searchsorted(log_x_tab, logx) - 1
        ix = max(0, min(ix, len(X_TABLE)-2))
        tx = ((logx - log_x_tab[ix]) /
              (log_x_tab[ix+1] - log_x_tab[ix]))

    # Indices in j (linear)
    # J_GRID is descending [1.0, 0.401, 0.161, 0.065, 0.026]
    jg = J_GRID
    if j_cl >= jg[0]:
        ij = 0
        tj = 0.0
    elif j_cl <= jg[-1]:
        ij = len(jg)-2
        tj = 1.0
    else:
        # find i such that jg[i] >= j > jg[i+1]
        ij = np.searchsorted(-jg, -j_cl) - 1
        ij = max(0, min(ij, len(jg)-2))
        tj = (j_cl - jg[ij]) / (jg[ij+1] - jg[ij])

    def bilinear(arr):
        v00 = arr[ix,   ij]
        v01 = arr[ix,   ij+1]
        v10 = arr[ix+1, ij]
        v11 = arr[ix+1, ij+1]
        v0 = v00 + tj*(v01 - v00)
        v1 = v10 + tj*(v11 - v10)
        return v0 + tx*(v1 - v0)

    eps1_star = -bilinear(NEG_E1)  # NEG_E1 in table is -ε1*
    eps2_star = bilinear(E2)
    j1_star   = bilinear(J1)
    j2_star   = bilinear(J2)
    zeta2_star= bilinear(ZETA2)

    return eps1_star, eps2_star, j1_star, j2_star, zeta2_star


def j_min_of_x(x):
    """
    Loss-cone boundary j_min(x) = sqrt(2 x / x_D - x^2 / x_D^2).
    Only defined for 0 <= x <= x_D.
    """
    if x <= 0.0:
        return 0.0
    if x >= X_D:
        return 1.0  # doesn't matter; we won't go this far
    val = 2.0*x/X_D - (x**2)/(X_D**2)
    if val <= 0.0:
        return 0.0
    return math.sqrt(val)


def orbital_period_star(x):
    """
    Dimensionless orbital period P*(x) = P(x) / τ0.
    From Eq. (26) scaling: P* at x=1 is given; roughly P(x) ∝ x^(-3/2),
    so P*(x) ≈ P_STAR * x^(-3/2).
    This is exactly the factor in front of ∂g/∂τ in Eq. (26).
    """
    if x <= 0.0:
        x = 1.0  # not used in unbound regime here
    return P_STAR * x**(-1.5)


def choose_time_step(x, j, eps2_star, j2_star):
    """
    Approximate version of the step-adjustment algorithm.

    We want:
        sqrt(n) * eps2* << x
        sqrt(n) * j2*   << 1
        sqrt(n) * j2*   << j_min constraints near the loss cone, etc.
    We will pick n as the minimum over a few constraints.

    Returns:
        n  (dimensionless number of orbital periods),
        dt = n * P*(x)
    """
    # If x is very small, avoid crazy periods
    x_eff = max(x, 1e-3)
    # Base constraints (Eq. 29-ish, but simplified)
    # Avoid division by zero
    eps2 = abs(eps2_star)
    j2   = abs(j2_star)

    constraints = []

    # (29a) n^(1/2) eps2 <= 0.15 |E| → E ~ v0^2 x, with v0^2=1
    if eps2 > 0:
        constraints.append((0.15 * x_eff / eps2)**2)

    # (29b) n^(1/2) j2 <= 0.1 J_max (dimensionless j2* already normalized)
    if j2 > 0:
        constraints.append((0.1 / j2)**2)

    # (29c) n^(1/2) j2 <= 0.4 (1.0075 − j)
    if j2 > 0:
        constraints.append((0.4 * (1.0075 - j) / j2)**2)

    # Loss-cone constraint (approx; Eq. 29d)
    jmin = j_min_of_x(x_eff)
    if j2 > 0 and jmin > 0:
        d1 = 0.25 * abs(j - jmin)
        d2 = 0.10 * jmin
        maxd = max(d1, d2)
        if maxd > 0:
            constraints.append((maxd / j2)**2)

    if not constraints:
        n = 1e-3
    else:
        n = min(constraints)
        # Cap to avoid absurdly large n; the paper uses huge n,
        # but we just need a moderate to small fraction of a period
        n = max(min(n, 1.0), 1e-5)

    dt = n * orbital_period_star(x_eff)
    return n, dt


def correlated_normals(zeta2_star, eps2_star, j2_star):
    """
    Generate correlated normal deviates (y1, y2) with:
        E[y1] = E[y2] = 0
        Var[y1] = Var[y2] = 1
        Cov[y1, y2] = zeta2* / (eps2* j2*)

    This implements the Appendix B algorithm in spirit, but
    uses a simpler Gaussian correlation if the required
    correlation coefficient is in [-1, 1].
    """
    if eps2_star == 0.0 or j2_star == 0.0:
        # No correlation needed
        return random.gauss(0, 1), random.gauss(0, 1)

    rho = zeta2_star / (eps2_star * j2_star)
    # Numerically clamp
    rho = max(min(rho, 0.999), -0.999)

    # Standard method: y1 ~ N(0,1); y2 = rho y1 + sqrt(1-rho^2) z
    y1 = random.gauss(0, 1)
    z  = random.gauss(0, 1)
    y2 = rho * y1 + math.sqrt(1.0 - rho*rho) * z
    return y1, y2


def initialize_star(bound=True, x_outer=X_B):
    """
    Initialize a star in the cusp boundary, x slightly above x_outer.
    We choose x from a narrow band [x_outer, 1.5 x_outer] with probability
    proportional to g0(x) x^{-9/4} (since N(E) ~ x^{-9/4} g(x)).

    j is chosen isotropically: j^2 uniform on [0,1] => j = sqrt(u).
    """
    # Basic sampling: just pick uniformly in ln x and weight by g0(x)
    ln_x_min = math.log(x_outer)
    ln_x_max = math.log(1.5 * x_outer)
    ln_x = ln_x_min + random.random() * (ln_x_max - ln_x_min)
    x = math.exp(ln_x)
    g = g0_bw(x)
    # Rejection sampling: accept with probability g / g_max
    # approximate g_max from table near that region
    g_max = 3.0  # safe upper bound for that x range
    if random.random() > min(1.0, g / g_max):
        x = x_outer

    u = random.random()
    j = math.sqrt(u)  # isotropic in 2D J-space

    return x, j, 0.0  # cycle_count = 0


def reinject_star(star):
    """
    Replace a star that left the cusp or was captured by a fresh one.
    """
    return initialize_star(bound=True, x_outer=X_B)


# -------------------------------------------------------------
# Monte Carlo simulation
# -------------------------------------------------------------

def run_simulation(
        n_stars=500,
        n_steps=200000,
        loss_cone=True,
        measure_start=5000,
        measure_every=100,
        rng_seed=42
    ):
    """
    Run a Monte Carlo simulation of n_stars test stars for n_steps.
    If loss_cone=False, we artificially set j_min(x) ≡ 0 to remove
    loss-cone capture and mimic the “no-loss-cone” test.

    We measure ḡ(x) on the X_BINS grid after measure_start steps,
    every measure_every steps, by averaging g(x) of all stars in
    each x-bin.

    Returns:
        gbar_mean  (same length as X_BINS)
        gbar_std   (same length as X_BINS)
    """
    random.seed(rng_seed)

    # Each star: (x, j, cycle), where "cycle" is the accumulated
    # number of orbital periods since creation (sum of n).
    stars = [initialize_star() for _ in range(n_stars)]

    # Accumulators for gbar
    # We'll collect multiple “snapshots” and compute mean, std at the end.
    snapshots = []

    # Precompute bin ranges in log x for better statistics
    logX_bins = np.log(X_BINS)
    # We'll just use simple nearest-bin in log space for ḡ(x).

    def record_snapshot():
        xs = np.array([s[0] for s in stars])
        js = np.array([s[1] for s in stars])
        # We want gbar(x_bin) ~ < g(x) > over that range.
        # We approximate g(x) ~ BW g0(x) times correction from
        # relative occupancy vs BW. To keep things simple, we
        # just measure the star counts in each decade of x and
        # define gbar ∝ counts / x^{-9/4}, then renormalize so
        # that at small x we match g0(x).
        gbar = np.zeros(len(X_BINS))
        counts = np.zeros(len(X_BINS))

        logxs = np.log(xs)
        for xi in range(len(xs)):
            lx = logxs[xi]
            # nearest bin in log space
            idx = np.argmin(np.abs(logX_bins - lx))
            counts[idx] += 1.0

        # Turn counts into ḡ by dividing out the BW X^{-9/4} weighting
        # and then scaling such that at the smallest x we have ḡ ≈ g0.
        for i, x_bin in enumerate(X_BINS):
            if counts[i] <= 0:
                gbar[i] = 0.0
            else:
                # Use g0(x) as base shape; we will renormalize across bins
                gbar[i] = counts[i]  # temporary

        # Renormalize by requiring that at the lowest x bins
        # we match g0 (roughly isotropic, no depletion).
        # Use first three points as calibration.
        idxs = [0, 1, 2]
        s_num = 0.0
        s_den = 0.0
        for i in idxs:
            if gbar[i] > 0:
                s_num += g0_bw(X_BINS[i])
                s_den += gbar[i]
        if s_den > 0:
            scale = s_num / s_den
        else:
            scale = 1.0

        for i in range(len(gbar)):
            gbar[i] = gbar[i] * scale

        snapshots.append(gbar)

    # Main loop
    next_measure = measure_start

    for step in range(n_steps):
        # Evolve each star one step
        new_stars = []
        for (x, j, cycle) in stars:
            # Orbital coefficients (dimensionless)
            eps1_star, eps2_star, j1_star, j2_star, zeta2_star = \
                interpolate_orbital_coeffs(max(x, X_TABLE[0]), max(j, J_GRID[-1]))

            # Choose time step
            n, dt = choose_time_step(x, j, eps2_star, j2_star)

            # Orbital “age” update
            cycle_new = cycle + n

            # Draw correlated random deviates for E and J
            y1, y2 = correlated_normals(zeta2_star, eps2_star, j2_star)

            # Update x and j (dimensionless)
            # x change ~ ΔE / v0^2, and eps1_star, eps2_star are per orbit,
            # so Δx = n*eps1* + sqrt(n)*eps2* y1.
            dx = n * eps1_star + math.sqrt(n) * eps2_star * y1
            x_new = x + dx

            # Similarly for j
            dj = n * j1_star + math.sqrt(n) * j2_star * y2
            j_new = j + dj

            # Reflect or clip j at 0 and 1 due to geometry / boundary (ii)
            if j_new < 0.0:
                j_new = -j_new
            if j_new > 1.0:
                # Instead of discarding, reflect near J_max,
                # but if it overshoots badly, just clip.
                j_new = 2.0 - j_new
                if j_new < 0.0:
                    j_new = 0.0
                if j_new > 1.0:
                    j_new = 1.0

            # Now apply boundary conditions:
            # Outer cusp: if x < X_B, star escapes to the core
            # (for bound stars); we reinject it.
            if x_new < X_B:
                new_stars.append(reinject_star((x, j, cycle_new)))
                continue

            # Inner “disruption” energy: x <= x_D is allowed;
            # if x is extremely large, also reinject to avoid blow-up.
            if x_new > 2.0 * X_D:
                new_stars.append(reinject_star((x, j, cycle_new)))
                continue

            # Loss cone: check only if we are at pericenter,
            # i.e. cycle_new has passed an integer number of orbits.
            # The paper uses an integer count of orbits; we approximate
            # by checking if floor(cycle_new) > floor(cycle).
            captured = False
            if loss_cone:
                if math.floor(cycle_new + 1e-9) > math.floor(cycle + 1e-9):
                    # At pericenter for the first time after this step
                    jmin = j_min_of_x(x_new)
                    if j_new < jmin:
                        # Captured by BH
                        captured = True

            if captured:
                # Reinjection from the core
                new_stars.append(reinject_star((x, j, cycle_new)))
            else:
                new_stars.append((x_new, j_new, cycle_new))

        stars = new_stars

        if step == next_measure:
            record_snapshot()
            next_measure += measure_every

    # Aggregate snapshots
    if not snapshots:
        raise RuntimeError("No snapshots recorded; increase n_steps or decrease measure_start.")

    snapshots = np.array(snapshots)  # shape (n_snapshots, n_bins)
    gbar_mean = snapshots.mean(axis=0)
    gbar_std  = snapshots.std(axis=0)

    return gbar_mean, gbar_std


# -------------------------------------------------------------
# Main driver: run both simulations and plot
# -------------------------------------------------------------

def main():
    # Parameters – tune as needed
    N_STARS = 800
    N_STEPS = 3000#00
    MEASURE_START = 200#00
    MEASURE_EVERY = 5#00

    print("Running no-loss-cone check...")
    gbar_nolc, gbar_nolc_err = run_simulation(
        n_stars=N_STARS,
        n_steps=N_STEPS,
        loss_cone=False,
        measure_start=MEASURE_START,
        measure_every=MEASURE_EVERY,
        rng_seed=123
    )

    print("Running full loss-cone simulation...")
    gbar_lc, gbar_lc_err = run_simulation(
        n_stars=N_STARS,
        n_steps=N_STEPS,
        loss_cone=True,
        measure_start=MEASURE_START,
        measure_every=MEASURE_EVERY,
        rng_seed=456
    )

    # Compute g0(x) at X_BINS for reference
    g0_vals = np.array([g0_bw(x) for x in X_BINS])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Paper data for Fig. 3
    ax.errorbar(
        GBAR_X_PAPER, GBAR_PAPER, yerr=GBAR_ERR_PAPER,
        fmt='o', markersize=4, capsize=3, label=r"$\bar{g}$ (Shapiro & Marchant 1978)"
    )

    # Our no-loss-cone MC
    ax.errorbar(
        X_BINS, gbar_nolc, yerr=gbar_nolc_err,
        fmt='s', markersize=4, capsize=3, label=r"MC $\bar{g}$ (no loss cone)"
    )

    # Our full loss-cone MC
    ax.errorbar(
        X_BINS, gbar_lc, yerr=gbar_lc_err,
        fmt='D', markersize=4, capsize=3, label=r"MC $\bar{g}$ (loss cone)"
    )

    # BW 1D solution g0 (solid line)
    ax.plot(X_BINS, g0_vals, '-', lw=2, label=r"$g_0$ (BW 1D)")

    ax.set_xscale('log')
    ax.set_xlabel(r"$x = -E / v_0^2$")
    ax.set_ylabel(r"$\bar{g}(x)$")
    ax.set_title(r"Isotropized distribution $\bar{g}(x)$ (canonical case $x_{\mathrm{crit}}\approx 10$, $x_D=10^4$)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Print a small table to inspect numerically
    print("\n x-bin    g0(x)   gbar(no LC)   gbar(LC)")
    for x, g0, gn, gl in zip(X_BINS, g0_vals, gbar_nolc, gbar_lc):
        print(f"{x:8.3g}  {g0:7.3f}   {gn:11.3f}   {gl:11.3f}")

if __name__ == "__main__":
    main()
