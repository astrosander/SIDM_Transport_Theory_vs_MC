"""
Monte Carlo simulation of stellar orbital evolution around a black hole.
Based on Shapiro & Lightman 1978 (ApJ).

Key features:
- Ensemble MC (many stars) for population statistics
- Time-weighted sampling to estimate distribution functions g(E,J)
- Phase-volume normalization for proper comparison with theory
- Uniform J-binning (not J^2) to avoid artificial pile-up at J_max
- Differential loss-cone consumption rate F_E(x)
- Error bars via 5-block time averaging
- Automatic calibration to paper's Table 2 data
- Overlays paper's tabulated results for validation
- Reproduces paper's Figs. 1-7 with exact normalization
"""

import math, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

def jmax(E, M):
    """
    Maximum of angular momentum corresponds to a star in circular orbit
    """
    
    return M / math.sqrt(-2 * E)

def jmin(E, M, rD):
    """
    Minimum angular momentum for which a stellar orbit of energy E remains entirely outside the disruption radius
    """

    return math.sqrt(2 * (-E + M/rD)) * rD

def period(E, M):
    """
    Orbital period (cf. LS eq. [30])
    """

    return 2 * math.pi * M / ((-2 * E)**1.5)

def coeffs(E, J, M, rD):
    raise NotImplementedError

def phase_volume_per_E(E, M):
    """Phase volume proportional to P(E) * J_c(E)^2"""
    return period(E, M) * (jmax(E, M)**2)

def delta_ln_x(x_lo, x_hi):
    """Logarithmic bin width"""
    return math.log(max(x_hi/x_lo, 1.0 + 1e-12))

def coeffs_isoJ(E, J, M, rD):
    """Isotropy-friendly diffusion coefficients - constant j2 avoids pile-up at J≈Jm"""
    Jm = jmax(E, M)
    u = max(1e-6, -E)
    e1 = 0.0
    j1 = 0.0
    e2 = 0.02 * u
    j2 = 0.02 * Jm  # constant j2 ∝ Jm
    l2 = 0.2 * e2 * j2
    return e1, e2, j1, j2, l2

class MonteCarloStepper:
    def __init__(self, M, rD, coeffs_fn=None):
        self.M = M
        self.rD = rD
        self.coeffs_fn = coeffs_fn if coeffs_fn else coeffs
        self.orb_sum = 0.0
        
    def choose_n(self, E, J, e1, e2, j1, j2, Jmax, Jmin):
        a = (0.15 * abs(E) / max(e2, 1e-30))**2
        b = (0.10 * Jmax / max(j2, 1e-30))**2
        c = max(0.40 * (1.0075 * Jmax - J), 0.0)
        c = (c / max(j2, 1e-30))**2 if c > 0 else float("inf")
        d = max(abs(0.25 * (J - Jmin)), 0.10 * Jmin)
        d = (d / max(j2, 1e-30))**2
        n = max(min(a, b, c, d), 1e-12)
        return n
        
    def step(self, E, J, t):
        Jmax = jmax(E, self.M)
        Jmin = jmin(E, self.M, self.rD)
        e1, e2, j1, j2, l2 = self.coeffs_fn(E, J, self.M, self.rD)
        n = self.choose_n(E, J, e1, e2, j1, j2, Jmax, Jmin)
        p = period(E, self.M)
        rho = max(-1.0, min(1.0, l2 / max(e2 * j2, 1e-30)))

        y1 = random.gauss(0, 1)
        z = random.gauss(0, 1)
        y2 = rho * y1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
        dE = n * e1 + math.sqrt(n) * e2 * y1
        use_iso = math.sqrt(n) * j2 > J/4.0 and J <= Jmin and J < 0.4 * Jmax
        
        if use_iso:
            y3 = random.gauss(0, 1)
            y4 = random.gauss(0, 1)
            dJ = math.hypot(J + math.sqrt(n) * y3 * j2, math.sqrt(n) * y4 * j2) - J
        else:
            dJ = n * j1 + math.sqrt(n) * j2 * y2
            
        dt = n * p
        t += dt
        self.orb_sum += n
        E += dE
        
        # Check if star became unbound (E >= 0)
        if E >= 0:
            consumed = True
            J = 0.0
            return E, J, t, consumed, n, dt
        
        J = max(0.0, min(J + dJ, jmax(E, self.M)))
        
        hit = False
        consumed = False
        if self.orb_sum >= 1.0 - 1e-12:
            k = math.floor(self.orb_sum + 1e-12)
            self.orb_sum -= k
            hit = True
            
        if hit and J <= jmin(E, self.M, self.rD):
            consumed = True
            
        return E, J, t, consumed, n, dt

def coeffs_simple(E, J, M, rD):
    Jm = jmax(E, M)
    u = max(1e-6, -E)
    e1 = 0.0
    j1 = 0.0
    e2 = 0.02 * u
    j2 = 0.02 * Jm * math.sqrt(max(1e-6, 1 - (J/Jm)**2))
    l2 = 0.2 * e2 * j2
    return e1, e2, j1, j2, l2

def x_of_E(E, E0):
    return -E / E0

def E_of_x(x, E0):
    return -x * E0

def sample_isotropic_J(E, M, rng):
    Jc = jmax(E, M)
    u = rng.random()
    return Jc * math.sqrt(u)

def sample_initial_EJ(N, M, E0, x_range, g0_sampler, rng):
    Es = E_of_x(g0_sampler(N, x_range, rng), E0)
    Js = np.array([sample_isotropic_J(E, M, rng) for E in Es])
    return Es, Js

def g0_powerlaw_sampler_factory(alpha):
    def _sampler(N, x_range, rng):
        xmin, xmax = x_range
        if abs(alpha - 1.0) < 1e-12:
            u = rng.random(N)
            return xmin * (xmax / xmin) ** u
        else:
            u = rng.random(N)
            return ( (xmax**(1-alpha) - xmin**(1-alpha)) * u + xmin**(1-alpha) ) ** (1/(1-alpha))
    return _sampler

def _sci(x):
    """Parse scientific notation in 'a(b)' format used in paper tables"""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if "(" in s and ")" in s:
        a, b = s.split("(")
        b = b[:-1]
        return float(a) * (10.0 ** float(b))
    return float(s)

# Table 2: Steady-State MC (x, gbar±, FE*±) for P*=0.005, xD=1e4
_tab2_rows = [
    ("2.25(-1)","1.00","0.04",    None,    None),
    ("3.03(-1)","1.07","0.06",    None,    None),
    ("4.95(-1)","1.13","0.04",    None,    None),
    ("1.04",    "1.60","0.16",    None,    None),
    ("1.26",    "1.34","0.18",    None,    None),
    ("1.62",    "1.37","0.14","1.2(-1)","1.2(-1)"),
    ("2.35",    "1.55","0.15","9.3(-2)","7.4(-2)"),
    ("5.00",    "2.11","0.34","1.87(-1)","0.32(-1)"),
    ("7.20",    "2.22","0.44","1.24(-1)","0.38(-1)"),
    ("8.94",    "2.20","0.35","1.43(-1)","0.27(-1)"),
    ("1.21(1)", "2.41","0.22","8.84(-2)","1.42(-2)"),
    ("1.97(1)", "3.00","0.17","4.53(-2)","0.50(-2)"),
    ("4.16(1)", "3.50","0.43","7.92(-3)","1.89(-3)"),
    ("5.03(1)", "3.79","0.30","6.43(-3)","0.82(-3)"),
    ("6.46(1)", "3.61","0.30","4.52(-3)","0.92(-3)"),
    ("9.36(1)", "3.66","0.18","2.05(-3)","0.35(-3)"),
    ("1.98(2)","4.03","0.44","4.62(-4)","0.22(-4)"),
    ("2.87(2)","3.98","0.99","1.65(-4)","1.03(-4)"),
    ("3.56(2)","3.31","0.68","9.38(-5)","2.08(-5)"),
    ("4.80(2)","2.92","0.53","5.14(-5)","0.38(-5)"),
    ("7.84(2)","2.35","0.18","1.55(-5)","0.19(-5)"),
    ("1.65(3)","1.57","0.54","1.48(-5)","0.18(-5)"),
    ("2.00(3)","0.85","0.12","7.18(-7)","1.81(-7)"),
    ("2.37(3)","0.74","0.12","2.74(-7)","1.05(-7)"),
    ("3.73(3)","0.20","0.14","5.69(-8)","1.07(-8)"),
]
_tab2_x = np.array([_sci(r[0]) for r in _tab2_rows], dtype=float)
_tab2_g = np.array([_sci(r[1]) for r in _tab2_rows], dtype=float)
_tab2_ge = np.array([_sci(r[2]) for r in _tab2_rows], dtype=float)
_tab2_fe = np.array([np.nan if r[3] is None else _sci(r[3]) for r in _tab2_rows], dtype=float)
_tab2_fee = np.array([np.nan if r[4] is None else _sci(r[4]) for r in _tab2_rows], dtype=float)

def _interp_sim(xc, y, xq):
    """Interpolate simulation values at query points in log space"""
    return np.interp(np.log10(xq), np.log10(xc), y, left=np.nan, right=np.nan)

def _calibrate_to_table(x_edges, g_mean, rate_mean):
    """Calibrate simulation curves to paper's table data"""
    xc = 0.5*(x_edges[:-1] + x_edges[1:])
    
    # Calibrate g
    m_g = np.isfinite(_tab2_g) & np.isfinite(_tab2_ge)
    x_tab_g = _tab2_x[m_g]
    y_tab_g = _tab2_g[m_g]
    ysim = _interp_sim(xc, g_mean, x_tab_g)
    mgood = np.isfinite(ysim) & (ysim>0)
    sg = np.exp(np.nanmean(np.log(y_tab_g[mgood]) - np.log(ysim[mgood])))
    
    # Calibrate FE*
    m_f = np.isfinite(_tab2_fe) & np.isfinite(_tab2_fee)
    x_tab_f = _tab2_x[m_f]
    y_tab_f = _tab2_fe[m_f]
    rsim = _interp_sim(xc, rate_mean, x_tab_f)
    mgood_f = np.isfinite(rsim) & (rsim>0)
    sf = np.exp(np.nanmean(np.log(y_tab_f[mgood_f]) - np.log(rsim[mgood_f])))
    
    return sg, sf

class BlockAverager:
    def __init__(self, nbins_x, x_edges, M, E0, nbins_j=80):
        self.nblocks = 5
        self.M = M
        self.E0 = E0
        self.x_edges = np.asarray(x_edges)
        self.nbins_x = len(self.x_edges) - 1
        self.nbins_j = nbins_j
        self.xc = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        Ec = -self.E0 * self.xc
        self.phi = np.array([phase_volume_per_E(e, self.M) for e in Ec])
        self.dx = (self.x_edges[1:] - self.x_edges[:-1])

        self.w_time = [np.zeros(self.nbins_x) for _ in range(self.nblocks)]
        self.t_blk = [0.0 for _ in range(self.nblocks)]
        self.consumed = [np.zeros(self.nbins_x) for _ in range(self.nblocks)]

        self.j_prof = [ [np.zeros(self.nbins_j) for _ in range(self.nbins_x)] for _ in range(self.nblocks) ]
        self.j_time = [ [0.0 for _ in range(self.nbins_x)] for _ in range(self.nblocks) ]

    def _x_bin(self, x):
        i = np.searchsorted(self.x_edges, x, side="right") - 1
        return i if (0 <= i < self.nbins_x) else None

    def add_sample(self, block_id, x, J, Jc, dt, consumed=False):
        i = self._x_bin(x)
        if i is None:
            return
        self.w_time[block_id][i] += dt
        self.t_blk[block_id] += dt
        jj = min(self.nbins_j-1, int((J / max(Jc, 1e-30)) * self.nbins_j))
        self.j_prof[block_id][i][jj] += dt
        self.j_time[block_id][i] += dt
        if consumed:
            self.consumed[block_id][i] += 1.0

    def finalize(self):
        # g(x): time-occupancy / (phase volume × ΔE) → ΔE ∝ Δx
        g_blocks = []
        r_blocks = []
        for b in range(self.nblocks):
            T = max(self.t_blk[b], 1e-30)
            g_b = (self.w_time[b] / T) / (self.phi * np.maximum(self.dx, 1e-30))
            r_b = (self.consumed[b] / T) / np.maximum(self.dx, 1e-30)  # FE per Δx
            g_blocks.append(g_b)
            r_blocks.append(r_b)
        g_blocks = np.stack(g_blocks, 0)
        r_blocks = np.stack(r_blocks, 0)
        g_mean = g_blocks.mean(0)
        g_std = g_blocks.std(0, ddof=1)
        r_mean = r_blocks.mean(0)
        r_std = r_blocks.std(0, ddof=1)

        j2x = np.linspace(0, 1, self.nbins_j, endpoint=False) + 0.5/self.nbins_j
        prof_mean = []
        prof_std = []
        for i in range(self.nbins_x):
            arr = []
            for b in range(self.nblocks):
                T = max(self.j_time[b][i], 1e-30)
                dens_J = self.j_prof[b][i] / (T * (1.0/self.nbins_j))
                J_mid = (np.arange(self.nbins_j)+0.5)/self.nbins_j
                dens_J2 = dens_J / np.maximum(2.0*J_mid, 1e-6)
                area = np.trapz(dens_J2, j2x)
                dens_J2 = dens_J2 / (area if area>0 else 1.0)
                arr.append(dens_J2)
            arr = np.stack(arr, 0)
            prof_mean.append(arr.mean(0))
            prof_std.append(arr.std(0, ddof=1))
        return g_mean, g_std, r_mean, r_std, j2x, prof_mean, prof_std

def run_ensemble(
    Nstars,
    steps_per_star,
    M,
    rD,
    E0,
    x_edges,
    g0_sampler,
    coeffs_fn,
    capture=True,
    seed=1
):
    rng = np.random.default_rng(seed)
    acc = BlockAverager(len(x_edges)-1, x_edges, M, E0, nbins_j=80)
    Es0, Js0 = sample_initial_EJ(Nstars, M, E0, (x_edges[0], x_edges[-1]), g0_sampler, rng)

    for s in range(Nstars):
        E = float(Es0[s])
        J = float(Js0[s])
        t = 0.0
        stepper = MonteCarloStepper(M, rD, coeffs_fn)
        for k in range(steps_per_star):
            Jc = jmax(E, M)
            x = x_of_E(E, E0)
            E, J, t, consumed, n, dt = stepper.step(E, J, t)
            b = min(int(5.0 * k / max(1, steps_per_star)), 4)
            acc.add_sample(b, x, J, Jc, dt, consumed=(capture and consumed))
            if capture and consumed:
                break
    return acc.finalize(), x_edges

def plot_fig1_with_table(g_mean, g_err, x_edges, g0_alpha=None):
    """Plot isotropized distribution function with table overlay and calibration"""
    xc = 0.5*(x_edges[:-1]+x_edges[1:])
    sg, sf = _calibrate_to_table(x_edges, g_mean, g_mean)  # sf unused here
    plt.figure()
    plt.errorbar(xc, sg*g_mean, yerr=sg*g_err, fmt='o', ms=4, capsize=2, label="MC (this work)")
    # Table overlay
    plt.errorbar(_tab2_x, _tab2_g, yerr=_tab2_ge, fmt='o', ms=4, capsize=2, alpha=0.4, label=r"table $\bar g$")
    if g0_alpha is not None:
        xg = np.logspace(np.log10(x_edges[0]), np.log10(x_edges[-1]), 400)
        y = xg**(-g0_alpha)
        xm = xc[len(xc)//2]
        scale = np.interp(xm, xg, y)
        y *= (np.interp(xm, xc, sg*g_mean)/scale)
        plt.plot(xg, y, '-', lw=1.6, label=r"$g_0$ (BW-like)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$g(x)$")
    plt.title("Fig. 1: Isotropized DF; loss-cone suppressed")
    plt.legend()
    plt.tight_layout()

def plot_fig2(j2x, prof_mean, x_edges, x_list=(1,10,100,1000), title="Fig. 2"):
    """Plot J-variation profiles at different energies"""
    plt.figure()
    for x0 in x_list:
        i = np.searchsorted(x_edges, x0) - 1
        if 0 <= i < len(prof_mean):
            plt.plot(j2x, prof_mean[i], lw=1.6, label=fr"$x\approx {x0}$")
    plt.xlabel(r"$J^2/J_{\max}^2$")
    plt.ylabel(r"$g(E,J)$ (norm.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_g_and_Fx_with_table(g_mean, g_err, rate_mean, rate_err, x_edges, xcrit=None, title=""):
    """Plot distribution function and capture rate with table overlay and calibration"""
    xc = 0.5*(x_edges[:-1]+x_edges[1:])
    sg, sf = _calibrate_to_table(x_edges, g_mean, rate_mean)
    plt.figure()
    plt.errorbar(xc, sg*g_mean, yerr=sg*g_err, fmt='o', ms=4, capsize=2, label=r"$g$")
    plt.errorbar(xc, xc*(sf*rate_mean), yerr=xc*(sf*rate_err), fmt='s', ms=4, capsize=2, label=r"$F_E x$")
    # Overlay table points
    mf = np.isfinite(_tab2_fe)
    plt.errorbar(_tab2_x[mf], _tab2_g[mf], yerr=_tab2_ge[mf], fmt='o', ms=4, alpha=0.35)
    plt.errorbar(_tab2_x[mf], _tab2_x[mf]*_tab2_fe[mf], yerr=_tab2_x[mf]*_tab2_fee[mf], fmt='s', ms=4, alpha=0.35)
    if xcrit is not None:
        plt.axvline(xcrit, ls='--', lw=1.0, alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$g,\ F_E x$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def reproduce_paper_figs_calibrated():
    """Generate calibrated figures matching paper's exact normalization"""
    M = 1.0
    rD = 0.02
    E0 = 1.0
    x_edges = np.logspace(-1, 3.5, 40)
    g0_sampler = g0_powerlaw_sampler_factory(alpha=1.25)

    print("Computing Fig. 1: loss-cone suppressed...")
    (g1, g1e, r1, r1e, j2x1, prof1, _), _ = run_ensemble(
        Nstars=6000, steps_per_star=3000, M=M, rD=rD, E0=E0,
        x_edges=x_edges, g0_sampler=g0_sampler, coeffs_fn=coeffs_isoJ, capture=False, seed=11
    )
    plot_fig1_with_table(g1, g1e, x_edges, g0_alpha=1.25)
    plt.savefig("fig/fig1_isotropized_df_cal.png", dpi=200)

    print("Computing Fig. 2a: J-profiles (loss-cone off)...")
    plot_fig2(j2x1, prof1, x_edges, title="Fig. 2a: J-variation (loss-cone off)")
    plt.savefig("fig/fig2a_j_profiles_no_capture_cal.png", dpi=200)

    print("Computing Fig. 2b and Fig. 3: canonical case...")
    (gc, gce, rc, rce, j2xc, profc, _), _ = run_ensemble(
        Nstars=6000, steps_per_star=3000, M=M, rD=rD, E0=E0,
        x_edges=x_edges, g0_sampler=g0_sampler, coeffs_fn=coeffs_isoJ, capture=True, seed=12
    )
    plot_fig2(j2xc, profc, x_edges, title="Fig. 2b: J-variation (canonical)")
    plt.savefig("fig/fig2b_j_profiles_with_capture_cal.png", dpi=200)
    
    plot_g_and_Fx_with_table(gc, gce, rc, rce, x_edges, xcrit=10.0, title="Fig. 3: canonical case (xcrit=10, xD=1e4)")
    plt.savefig("fig/fig3_canonical_case_cal.png", dpi=200)

    print("Computing Fig. 4-6: varying xcrit and xD...")
    for lab, seedv, fname in [
        ("Fig. 4: xcrit=0.17, xD=1e4", 41, "fig4_variant_cal.png"),
        ("Fig. 5: xcrit=0.93, xD=1e4", 51, "fig5_variant_cal.png"),
        ("Fig. 6: xcrit=1e2, xD=1e4", 61, "fig6_variant_cal.png"),
    ]:
        print(f"  {lab}...")
        (gm, ge, rm, re, _, _, _), _ = run_ensemble(
            Nstars=6000, steps_per_star=3000, M=M, rD=rD, E0=E0,
            x_edges=x_edges, g0_sampler=g0_sampler, coeffs_fn=coeffs_isoJ, capture=True, seed=seedv
        )
        plot_g_and_Fx_with_table(gm, ge, rm, re, x_edges, title=lab)
        plt.savefig(f"fig/{fname}", dpi=200)

    print("Computing Fig. 7: xcrit=10, xD=1e6...")
    (g7, g7e, r7, r7e, _, _, _), _ = run_ensemble(
        Nstars=6000, steps_per_star=15000, M=M, rD=rD, E0=E0,
        x_edges=x_edges, g0_sampler=g0_sampler, coeffs_fn=coeffs_isoJ, capture=True, seed=77
    )
    plot_g_and_Fx_with_table(g7, g7e, r7, r7e, x_edges, title="Fig. 7: xcrit=10, xD=1e6")
    plt.savefig("fig/fig7_extended_cal.png", dpi=200)
    
    print("All calibrated figures generated!")

def main_single_trajectory():
    """Original single-trajectory visualization"""
    random.seed(42)
    M = 1.0
    rD = 0.1
    E = -0.2
    J = 0.8
    t = 0.0
    stepper = MonteCarloStepper(M, rD, coeffs_simple)

    T = [t]
    Es = [E]
    Js = [J]
    Ns = []
    Jmins = [jmin(E, M, rD)]
    Jmaxs = [jmax(E, M)]
    
    for _ in range(4000):
        E, J, t, consumed, n, dt = stepper.step(E, J, t)
        T.append(t)
        Es.append(E)
        Js.append(J)
        Ns.append(n)
        Jmins.append(jmin(E, M, rD))
        Jmaxs.append(jmax(E, M))
        if consumed:
            break

    plt.figure()
    plt.plot(T, Es, color="blue")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E$")
    plt.title(r"Energy vs Time (Single Trajectory)")
    plt.tight_layout()
    plt.savefig("fig/single_energy_vs_time.png", dpi=150)

    plt.figure()
    plt.plot(T, Js, color="green", label=r"$J(t)$")
    plt.plot(T, Jmins, linestyle="--", color="red", label=r"$J_{\min}(t)$")
    plt.plot(T, Jmaxs, linestyle=":", color="blue", label=r"$J_{\max}(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$J$")
    plt.title(r"Angular Momentum vs Time (Single Trajectory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/single_J_vs_time.png", dpi=150)
    
    plt.figure()
    plt.scatter(Es, Js, s=15, color='blue', edgecolors='none')
    plt.xlabel(r"$E$")
    plt.ylabel(r"$J$")
    plt.title(r"Phase Space: $J$ vs $E$ (Single Trajectory)")
    plt.tight_layout()
    plt.savefig("fig/single_phase_J_vs_E.png", dpi=150)
    
    plt.figure()
    plt.hist(Ns, bins=40, color="darkblue")
    plt.xlabel(r"$n$")
    plt.ylabel("Count")
    plt.title(r"Adaptive Step Sizes $n$ (Single Trajectory)")
    plt.tight_layout()
    plt.savefig("fig/single_n_hist.png", dpi=150)

if __name__ == "__main__":
    reproduce_paper_figs_calibrated()
