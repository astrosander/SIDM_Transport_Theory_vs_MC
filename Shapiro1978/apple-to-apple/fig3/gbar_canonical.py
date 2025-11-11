#!/usr/bin/env python3
# Compute the isotropized distribution function ḡ(x) for the canonical case
# (x_crit = 10, x_D = 1e4) by time-averaged occupancy per Δx and per t0.
# Physics and step rules match the 2D Shapiro–Marchant MC:
# - drift/diffusion coeffs from Table 1 (NEG_E1, E2, J1, J2, ZETA2)
# - (29a–e) step limiter with tunable cone_gamma in (29d) and energy cap (29e)
# - Capture DISABLED during measurement to avoid loss-cone depletion
# - Unbiased time deposition: substeps in log-x space for long moves
# - measure ḡ by summing weights * dt0 in each x-bin / (total t0 * Δx)
# - Normalize so ḡ(0.225) = 1 to match Fig. 3 filled circles

import math, os, sys, time, argparse
import numpy as np

# Optional numba
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------- Canonical constants ----------------
X_D     = 1.0e4
PSTAR   = 0.005
X_BOUND = 0.2         # reservoir energy (Eb = -0.2 v0^2 -> x_b = 0.2)
SEED    = 20251028

# Fig. 3 bin centers (same set you’ve been using)
X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=np.float64)

def _edges_from_centers(c):
    e = np.empty(c.size + 1, dtype=c.dtype)
    for i in range(1, c.size):
        e[i] = math.sqrt(c[i-1]*c[i])
    e[0]  = c[0]**2 / e[1]
    e[-1] = c[-1]**2 / e[-2]
    return e

X_EDGES = _edges_from_centers(X_BINS)
DX      = X_EDGES[1:] - X_EDGES[:-1]

# ------------- Table-1 grids and starred coefficients -------------
X_GRID = np.array([3.36e-1, 3.31e0, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)  # descending

NEG_E1 = np.array([
    [ 1.41e-1,  1.40e-1,  1.33e-1,  1.29e-1,  1.29e-1],
    [ 1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],
    [ 1.96e-3,  1.59e-3,  2.83e-3,  3.38e-3,  3.49e-3],
    [ 3.64e-3,  4.64e-3,  4.93e-3,  4.97e-3,  4.98e-3],
    [ 8.39e-4,  8.53e-4,  8.56e-4,  8.56e-4,  8.56e-4],
], dtype=np.float64)

E2 = np.array([
    [ 3.14e-1,  4.36e-1,  7.55e-1,  1.37e0,  2.19e0],
    [ 4.45e-1,  8.66e-1,  1.57e0,  2.37e0,  2.73e0],
    [ 1.03e0,   1.80e0,   2.52e0,  2.77e0,  2.81e0],
    [ 1.97e0,   2.54e0,   2.68e0,  2.70e0,  2.70e0],
    [ 1.96e0,   1.96e0,   1.96e0,  1.96e0,  1.96e0],
], dtype=np.float64)

J1 = np.array([
    [-2.52e-2,  5.37e-1,  1.51e0,   3.83e0,   9.58e0],
    [-5.03e-3,  5.40e-3,  2.54e-2,  6.95e-2,  1.75e-1],
    [-2.58e-4,  3.94e-4,  1.45e-3,  3.77e-3,  9.45e-3],
    [-4.67e-6,  2.03e-5,  5.99e-5,  1.53e-4,  3.82e-4],
    [ 3.67e-8,  2.54e-7,  7.09e-7,  1.80e-6,  4.49e-6],
], dtype=np.float64)

J2 = np.array([
    [ 4.68e-1,  6.71e-1,  6.99e-1,  7.03e-1,  7.04e-1],
    [ 6.71e-2,  9.19e-2,  9.48e-2,  9.53e-2,  9.54e-2],
    [ 1.57e-2,  2.12e-2,  2.20e-2,  2.21e-2,  2.21e-2],
    [ 3.05e-3,  4.25e-3,  4.42e-3,  4.44e-3,  4.45e-3],
    [ 3.07e-4,  4.59e-4,  4.78e-4,  4.81e-4,  4.82e-4],
], dtype=np.float64)

ZETA2 = np.array([
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

# ---------------- Kernels (Numba or pure Python) ----------------
def _build_kernels(use_jit=True, cone_gamma=1.0):
    nb_njit = (lambda *a, **k: (lambda f: f))
    njit = nb.njit if (use_jit and HAVE_NUMBA) else nb_njit
    fast = dict(fastmath=True, nogil=True, cache=True) if (use_jit and HAVE_NUMBA) else {}

    @njit(**fast)
    def j_min_of_x(x):
        v = 2.0*x/X_D - (x/X_D)**2
        return math.sqrt(v) if v > 0.0 else 0.0

    @njit(**fast)
    def P_of_x(x):
        E = -x
        return 2.0*math.pi/((-2.0*E)**1.5)

    T0 = P_of_x(1.0)/PSTAR

    @njit(**fast)
    def which_bin(x):
        best = 0
        bestd = abs(X_BINS[0]-x)
        for i in range(1, X_BINS.size):
            d = abs(X_BINS[i]-x)
            if d < bestd:
                bestd = d; best = i
        return best

    @njit(**fast)
    def bilinear_coeffs(x, j):
        # clamp & bilinear in (log10 x, j) with J_GRID descending
        if x < X_GRID[0]: x = X_GRID[0]
        if x > X_GRID[-1]: x = X_GRID[-1]
        lx = math.log10(x)

        ix = 0
        for k in range(X_GRID.size-1):
            if X_GRID[k] <= x <= X_GRID[k+1]:
                ix = k; break
        x1 = X_GRID[ix]; x2 = X_GRID[ix+1]
        tx = (lx - math.log10(x1)) / (math.log10(x2) - math.log10(x1) + 1e-30)
        if tx < 0.0: tx = 0.0
        if tx > 1.0: tx = 1.0

        if j > J_GRID[0]: j = J_GRID[0]
        if j < J_GRID[-1]: j = J_GRID[-1]
        ij = 0
        for k in range(J_GRID.size-1):
            if (J_GRID[k] >= j) and (j >= J_GRID[k+1]):
                ij = k; break
        j1 = J_GRID[ij]; j2 = J_GRID[ij+1]
        tj = (j - j1) / (j2 - j1 + 1e-30)
        if tj < 0.0: tj = 0.0
        if tj > 1.0: tj = 1.0

        def get(A):
            a11 = A[ix,ij];   a12 = A[ix,ij+1]
            a21 = A[ix+1,ij]; a22 = A[ix+1,ij+1]
            a1 = a11*(1.0-tj) + a12*tj
            a2 = a21*(1.0-tj) + a22*tj
            return a1*(1.0-tx) + a2*tx

        e1  = -get(NEG_E1)     # table lists -ε1*; we need +ε1*
        e2v =  get(E2)
        j1v =  get(J1)
        j2v =  get(J2)
        z2v =  get(ZETA2)
        return e1, e2v, j1v, j2v, z2v

    @njit(**fast)
    def pick_n(x, j, e2v, j2v, cone_gamma_val):
        jmin = j_min_of_x(x)
        nmax = 1e30
        if e2v > 0.0:  # (29a)
            v = (0.15*abs(x)/max(e2v,1e-30))**2
            if v < nmax: nmax = v
        if j2v > 0.0:
            v = (0.10/j2v)**2                 # (29b)
            if v < nmax: nmax = v
            v = (0.40*max(1.0-j,0.0)/j2v)**2  # (29c)
            if v < nmax: nmax = v
            # (29d)
            floor = max(cone_gamma_val*(j - jmin), 0.10*jmin)
            if floor > 0.0:
                v = (floor/j2v)**2
                if v < nmax: nmax = v
        # (29e) practical energy-step cap: limit rms kick to small fraction of x
        kappa = 0.07
        if e2v > 0.0:
            v = (kappa*abs(x)/max(e2v,1e-30))**2
            if v < nmax: nmax = v
        return nmax if nmax > 1e-8 else 1e-8

    @njit(**fast)
    def correlated_normals(e2v, j2v, z2v):
        z1 = np.random.normal()
        z2 = np.random.normal()
        rho = z2v / max(e2v*j2v, 1e-30)
        if rho > 0.999: rho = 0.999
        if rho < -0.999: rho = -0.999
        y1 = z1
        y2 = rho*z1 + math.sqrt(max(0.0,1.0-rho*rho))*z2
        return y1, y2

    @njit(**fast)
    def step_one(x, j, phase, cone_gamma_val, disable_capture=False):
        # Return: x2, j2, phase2, n_used, captured?, crossed_pericenter?
        e1, e2v, j1v, j2v, z2v = bilinear_coeffs(x, j)
        n_raw = pick_n(x, j, e2v, j2v, cone_gamma_val)
        next_int = math.floor(phase) + 1.0
        crossed = (phase + n_raw >= next_int)
        n = next_int - phase if crossed else n_raw

        y1, y2 = correlated_normals(e2v, j2v, z2v)
        dE = n*e1 + math.sqrt(n)*y1*e2v
        x2 = x - dE
        if x2 < 1e-12: x2 = 1e-12

        # permissive 2D RW gate helps empty-loss-cone regime
        use_2d = (j < 0.6) or (math.sqrt(n)*j2v > max(1e-12, j/4.0))
        if use_2d:
            z1 = np.random.normal(); z2 = np.random.normal()
            j2 = math.sqrt((j + math.sqrt(n)*z1*j2v)**2 + (math.sqrt(n)*z2*j2v)**2)
        else:
            j2 = j + n*j1v + math.sqrt(n)*y2*j2v
        if j2 < 0.0: j2 = 0.0
        if j2 > 1.0: j2 = 1.0

        phase2 = phase + n
        # Disable capture during ḡ measurement to avoid loss-cone depletion
        if disable_capture:
            captured = False
        else:
            captured = (crossed and (j2 < j_min_of_x(x2)))
        return x2, j2, phase2, n, captured, crossed

    @njit(**fast)
    def T0_val():
        return T0

    return j_min_of_x, P_of_x, T0_val, which_bin, step_one

# -------------- Single-stream driver (no clones; just occupancy) --------------
def run_stream(n_relax_t0, warmup_t0, cone_gamma, use_jit, seed):
    j_min_of_x, P_of_x, T0_val, which_bin, step_one = _build_kernels(use_jit=use_jit, cone_gamma=cone_gamma)
    T0 = T0_val()

    # State
    rs = np.random.RandomState(seed)
    x   = X_BOUND
    j   = math.sqrt(rs.random())    # isotropic j
    w   = 1.0
    ph  = 0.0

    # Warm-up (no tallies)
    t0_used = 0.0
    while t0_used < warmup_t0:
        x_prev = x
        x, j, ph, n_used, cap, crossed = step_one(x, j, ph, cone_gamma)
        t0_used += (n_used * P_of_x(x_prev)) / T0
        if cap or (x < X_BOUND):
            # replacement at reservoir; keep phase (age)
            x = X_BOUND
            j = math.sqrt(rs.random())

    # Measurement (disable capture to measure background distribution)
    g_time = np.zeros_like(X_BINS, dtype=np.float64)
    t0_used = 0.0
    total_t0 = 0.0
    while t0_used < n_relax_t0:
        x_prev = x
        x, j, ph, n_used, cap, crossed = step_one(x, j, ph, cone_gamma, disable_capture=True)
        dt0 = (n_used * P_of_x(x_prev)) / T0
        t0_used += dt0
        total_t0 += dt0

        # Unbiased time deposition: substep long moves in log-x space
        dlogx = abs(math.log(max(x, 1e-12) / max(x_prev, 1e-12)))
        nsub = max(1, int(dlogx / 0.05))  # ≤ 5% in ln x per substep
        for k in range(nsub):
            tfrac = dt0 / nsub
            # Geometric midpoint in log-x for each substep
            ratio = max(x, 1e-12) / max(x_prev, 1e-12)
            xk = x_prev * (ratio ** ((k + 0.5) / nsub))
            g_time[which_bin(xk)] += w * tfrac

        # Still replace at reservoir boundary (but no capture removal during measurement)
        if x < X_BOUND:
            x = X_BOUND
            j = math.sqrt(rs.random())

    return g_time, total_t0

# -------------- Parallel aggregator --------------
def run_parallel(streams, windows, warmup, procs, cone_gamma, use_jit, seed, normalize):
    import concurrent.futures as cf
    rs = np.random.RandomState(seed)
    seeds = rs.randint(1, 2**31-1, size=streams, dtype=np.int64)

    gsum = np.zeros_like(X_BINS, dtype=np.float64)
    t0sum = 0.0

    if procs is None or procs < 1:
        procs = os.cpu_count() or 1

    start = time.time()
    with cf.ProcessPoolExecutor(max_workers=procs) as ex:
        futs = [ex.submit(run_stream, windows, warmup, cone_gamma, use_jit, int(seeds[i])) for i in range(streams)]
        done = 0
        for f in cf.as_completed(futs):
            g, t = f.result()
            gsum += g
            t0sum += t
            done += 1
            # simple progress
            if (time.time() - start) > 0.5:
                pct = 100.0 * done / streams
                sys.stderr.write(f"\rProgress: {done}/{streams} ({pct:5.1f}%)")
                sys.stderr.flush()
                start = time.time()
    sys.stderr.write("\n")

    # ḡ(x) = (time occupancy fraction per bin) / Δx
    gbar = np.zeros_like(X_BINS, dtype=np.float64)
    if t0sum > 0:
        tmp = gsum / t0sum
        np.divide(tmp, DX, out=gbar, where=(DX > 0))

    if normalize:
        # Normalize so ḡ(0.225) = 1 (first bin center is 0.225)
        norm_idx = 0  # X_BINS[0] = 0.225
        if gbar[norm_idx] > 0:
            gbar = gbar / gbar[norm_idx]

    return gbar

# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser(description="Compute isotropized ḡ(x) for canonical case via MC time-averaged occupancy.")
    ap.add_argument("--streams", type=int, default=400, help="number of independent streams")
    ap.add_argument("--windows", type=float, default=6.0, help="measurement time per stream in t0")
    ap.add_argument("--warmup", type=float, default=2.0, help="equilibration time in t0 before tallying")
    ap.add_argument("--procs", type=int, default=None, help="processes (defaults to all cores)")
    ap.add_argument("--nojit", action="store_true", help="disable numba JIT")
    ap.add_argument("--cone-gamma", type=float, default=1.0, help="prefactor in (29d): floor = max(gamma*(j-jmin), 0.10*jmin)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--normalize", action="store_true", help="scale so ḡ(x≈0.225)=1 (filled circles)")
    ap.add_argument("--plot", type=str, default="", help="optional output PNG path to plot filled circles")
    args = ap.parse_args()

    gbar = run_parallel(
        streams=args.streams,
        windows=args.windows,
        warmup=args.warmup,
        procs=args.procs,
        cone_gamma=args.cone_gamma,
        use_jit=(not args.nojit),
        seed=args.seed,
        normalize=args.normalize
    )

    # Print table
    print("# x_center    gbar(x)  (Δx-normalized, time-averaged per t0{})"
          .format("; normalized so ḡ(0.225)=1" if args.normalize else ""))
    for xb, val in zip(X_BINS, gbar):
        print(f"{xb:10.3g}  {val: .6e}")

    # Optional plot of filled circles
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogx(X_BINS, gbar, "o")  # filled circles
        plt.xlabel("x (dimensionless energy)")
        plt.ylabel("ḡ(x)")
        plt.title("Isotropized distribution ḡ(x) — canonical case")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)

if __name__ == "__main__":
    main()
