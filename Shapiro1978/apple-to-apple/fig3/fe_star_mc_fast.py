#!/usr/bin/env python3
# fe_star_mc_fast.py  (Fig. 3 reproduction - normalized & pericenter-truncated)
# Canonical 2D MC (Shapiro–Marchant 1978):
# - P* = 0.005, x_D = 1e4, x_crit ≈ 10 (canonical)
# - Boundary: replace at Eb = -0.2 v0^2 (x_b = 0.2) with isotropic j; keep the parent's time
# - Creation–annihilation via j^2 floors; clones share parent time
# - Exact step rules (29a–d). Use eq. (28) 2D RW when sqrt(n)*j2 > j/4
# - Pericenter-only capture check; truncate n to land exactly on next integer orbit when crossing
# - Measure ḡ(x) once per t0 across the run; rescale so ḡ(0.225) = 1 (≈ paper’s g(Eb)=1)
# - Output FE*(x)*x, Δx-normalized, per t0

import math, os, argparse, sys, time
import numpy as np

# -------- try numba ----------
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------- Canonical parameters ----------------
X_D    = 1.0e4
PSTAR  = 0.005
X_BOUND= 0.2        # Eb = -0.2 v0^2  => x_b = 0.2  (replacement energy)
SEED   = 20251028

# --------- Fig. 3 energy bin centers + geometric edges ----------
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

# ------------- Table-1 grids (x, j) and starred coeffs -------------
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
    [ 3.67e-8,  2.54e-7,  7.09e-7,  1.80e-6,  4.49e-6],  # <-- fixed 2.54e-7
], dtype=np.float64)

J2 = np.array([
    [ 4.68e-1,  6.71e-1,  6.99e-1,  7.03e-1,  7.04e-1],
    [ 6.71e-2,  9.19e-2,  9.48e-2,  9.53e-2,  9.54e-2],
    [ 1.57e-2,  2.12e-2,  2.20e-2,  2.21e-2,  2.21e-2],
    [ 3.05e-3,  4.25e-3,  4.42e-3,  4.44e-3,  4.45e-3],
    [ 3.07e-4,  4.59e-4,  4.78e-4,  4.81e-4,  4.82e-4],  # <-- fixed 4.59e-4
], dtype=np.float64)

# Verify Table-1 typos are fixed
assert abs(J1[-1,1] - 2.54e-7) < 1e-12, f"J1[-1,1] should be 2.54e-7, got {J1[-1,1]}"
assert abs(J2[-1,1] - 4.59e-4) < 1e-12, f"J2[-1,1] should be 4.59e-4, got {J2[-1,1]}"

ZETA2 = np.array([  # ζ*²
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

# ---------------- kernels (Numba or pure python) ----------------
def _build_kernels(use_jit=True, cone_gamma=0.25):
    nb_njit = (lambda *a, **k: (lambda f: f))
    njit = nb.njit if (use_jit and HAVE_NUMBA) else nb_njit
    fastmath = dict(fastmath=True, nogil=True, cache=True) if (use_jit and HAVE_NUMBA) else {}

    @njit(**fastmath)
    def j_min_of_x(x):
        v = 2.0*x/X_D - (x/X_D)**2
        return math.sqrt(v) if v > 0.0 else 0.0

    @njit(**fastmath)
    def P_of_x(x):  # orbital period P(E) with E=-x
        E = -x
        return 2.0*math.pi/((-2.0*E)**1.5)

    T0 = P_of_x(1.0)/PSTAR  # t0 from P*(x=1)

    @njit(**fastmath)
    def which_bin(x):
        best = 0
        bestd = abs(X_BINS[0]-x)
        for i in range(1, X_BINS.size):
            d = abs(X_BINS[i]-x)
            if d < bestd:
                bestd = d; best = i
        return best

    @njit(**fastmath)
    def bilinear_coeffs(x, j):
        # clamp and bilinear in (log10 x, j) with j_grid descending
        if x < X_GRID[0]: x = X_GRID[0]
        if x > X_GRID[-1]: x = X_GRID[-1]
        lx = math.log10(x)

        ix = 0
        for k in range(X_GRID.size-1):
            if X_GRID[k] <= x <= X_GRID[k+1]:
                ix = k; break
        x1 = X_GRID[ix]; x2 = X_GRID[ix+1]
        tx = (lx - math.log10(x1)) / (math.log10(x2) - math.log10(x1) + 1e-30)
        if tx < 0.0: tx=0.0
        if tx > 1.0: tx=1.0

        if j > J_GRID[0]: j = J_GRID[0]
        if j < J_GRID[-1]: j = J_GRID[-1]
        ij = 0
        for k in range(J_GRID.size-1):
            if (J_GRID[k] >= j) and (j >= J_GRID[k+1]):
                ij = k; break
        j1 = J_GRID[ij]; j2 = J_GRID[ij+1]
        tj = (j - j1) / (j2 - j1 + 1e-30)
        if tj < 0.0: tj=0.0
        if tj > 1.0: tj=1.0

        def get(A):
            a11 = A[ix,ij];   a12 = A[ix,ij+1]
            a21 = A[ix+1,ij]; a22 = A[ix+1,ij+1]
            a1 = a11*(1.0-tj) + a12*tj
            a2 = a21*(1.0-tj) + a22*tj
            return a1*(1.0-tx) + a2*tx

        e1  = -get(NEG_E1)   # table lists -ε1*; need ε1*
        e2v =  get(E2)
        j1v =  get(J1)
        j2v =  get(J2)
        z2v =  get(ZETA2)
        return e1, e2v, j1v, j2v, z2v

    @njit(**fastmath)
    def pick_n(x, j, e2v, j2v, cone_gamma_val):
        jmin = j_min_of_x(x)
        nmax = 1e30
        
        # (29a) energy rms limit
        if e2v > 0.0:
            v = (0.15*abs(x) / max(e2v, 1e-30))**2
            if v < nmax: nmax = v

        if j2v > 0.0:
            # (29b) absolute angular rms limit
            v = (0.10 / j2v)**2
            if v < nmax: nmax = v

            # (29c) distance to circular
            v = (0.40 * max(1.0 - j, 0.0) / j2v)**2
            if v < nmax: nmax = v

            # (29d) distance to loss cone (CORRECT FORM)
            # rms cap by max( gamma*(j - jmin), 0.10*jmin )
            gap = cone_gamma_val * (j - jmin)
            floor = max(gap, 0.10 * jmin)
            if floor > 0.0:
                v = (floor / j2v)**2
                if v < nmax: nmax = v

        # choose n (fraction of an orbit); no artificial 0.5 cap
        n = nmax if nmax > 1e-8 else 1e-8
        return n

    @njit(**fastmath)
    def correlated_normals(e2v, j2v, z2v):
        # Build correlated normals using rho = ζ^2/(e2*j2)
        z1 = np.random.normal()
        z2 = np.random.normal()
        denom = max(e2v*j2v, 1e-30)
        rho = z2v/denom
        if rho > 0.999: rho = 0.999
        if rho < -0.999: rho = -0.999
        y1 = z1
        y2 = rho*z1 + math.sqrt(max(0.0, 1.0 - rho*rho))*z2
        return y1, y2

    @njit(**fastmath)
    def step_one(x, j, phase, cone_gamma_val):
        e1, e2v, j1v, j2v, z2v = bilinear_coeffs(x, j)
        n_raw = pick_n(x, j, e2v, j2v, cone_gamma_val)

        # land exactly on next pericenter if we'd cross it
        next_int = math.floor(phase) + 1.0
        crossed_peri = (phase + n_raw >= next_int)
        n = next_int - phase if crossed_peri else n_raw

        # correlated normals
        y1, y2 = correlated_normals(e2v, j2v, z2v)

        # energy update (x = -E)
        dE = n*e1 + math.sqrt(n)*y1*e2v
        x_new = x - dE
        if x_new < 1e-12: x_new = 1e-12

        # 2-D RW gate: small-j OR large RMS kick vs J
        use_2d = (j < 0.4) or (math.sqrt(n)*j2v > max(1e-12, j/4.0))
        if use_2d:
            z1 = np.random.normal()
            z2 = np.random.normal()
            j_new = math.sqrt( (j + math.sqrt(n)*z1*j2v)**2 + (math.sqrt(n)*z2*j2v)**2 )
        else:
            j_new = j + n*j1v + math.sqrt(n)*y2*j2v
        if j_new < 0.0: j_new = 0.0
        if j_new > 1.0: j_new = 1.0

        # phase advance
        phase_before = phase
        phase = phase + n

        # capture only if we actually hit pericenter on this step
        captured = False
        if crossed_peri and (j_new < j_min_of_x(x_new)):
            captured = True

        return x_new, j_new, phase, n, captured, crossed_peri

    @njit(**fastmath)
    def run_stream(n_relax, floors, clones_per_split, x_bins, dx, seed, cone_gamma_val):
        np.random.seed(seed)

        # parent (time carrier)
        x = X_BOUND
        j = math.sqrt(np.random.random())  # isotropic j
        phase = 0.0
        w = 1.0

        # clone pool
        MAX_CLONES = 512
        cx  = np.zeros(MAX_CLONES, dtype=np.float64)
        cj  = np.zeros(MAX_CLONES, dtype=np.float64)
        cph = np.zeros(MAX_CLONES, dtype=np.float64)
        cw  = np.zeros(MAX_CLONES, dtype=np.float64)
        cfloor = np.zeros(MAX_CLONES, dtype=np.float64)
        cactive = np.zeros(MAX_CLONES, dtype=np.uint8)
        ccount = 0

        captures = np.zeros(x_bins.size, dtype=np.float64)
        t0_used = 0.0

        # Diagnostic counters
        crosses = 0
        caps = 0
        escapes = 0

        # time-weighted occupancy per x-bin (t0 units)
        g_time = np.zeros(x_bins.size, dtype=np.float64)
        total_t0 = 0.0

        # helpers
        def bin_index(xv):
            b = 0
            bestd = abs(x_bins[0]-xv)
            for bi in range(1, x_bins.size):
                d = abs(x_bins[bi]-xv)
                if d < bestd:
                    bestd = d; b = bi
            return b

        while t0_used < n_relax:
            x_prev = x; j_prev = j
            x, j, phase, n_used, cap, crossed = step_one(x, j, phase, cone_gamma_val)
            
            # dt0 (this step) in t0 units from the *parent's* previous state
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0
            total_t0 += dt0

            # time-average occupancy at the *start* of the step for each active object (weights sum ≈ 1)
            g_time[bin_index(x_prev)] += w * dt0
            for ii in range(ccount):
                if cactive[ii]:
                    g_time[bin_index(cx[ii])] += cw[ii] * dt0

            # count pericenter crossings
            if crossed: crosses += 1

            # 1) capture first
            if cap:
                caps += 1
                b = bin_index(x)            # energy *at capture*
                captures[b] += w
                x = X_BOUND                 # replace at reservoir
                j = math.sqrt(np.random.random())

            else:
                # 2) outward escape replacement
                if x < X_BOUND:
                    escapes += 1
                    x = X_BOUND
                    j = math.sqrt(np.random.random())  # keep phase

            # 3) split on w=j^2 floors *after* cap/escape
            w_now = j*j
            for fi in range(floors.size):
                f = floors[fi]
                if w_now < f and w >= f:
                    alpha = 1.0/(1.0 + clones_per_split)
                    w = w * alpha
                    for k in range(clones_per_split):
                        if ccount >= MAX_CLONES:
                            break
                        cx[ccount] = x
                        cj[ccount] = j
                        cph[ccount]= phase
                        cw[ccount] = w
                        cfloor[ccount] = f
                        cactive[ccount] = 1
                        ccount += 1
                    break

            # step active clones (share time; do not add to t0_used)
            i = 0
            while i < ccount:
                if cactive[i] == 0:
                    i += 1; continue
                x_c, j_c, ph_c = cx[i], cj[i], cph[i]
                x_c2, j_c2, ph_c2, n_c, cap_c, crossed_c = step_one(x_c, j_c, ph_c, cone_gamma_val)
                cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2

                # count pericenter crossings
                if crossed_c: crosses += 1

                # 1) capture first
                if cap_c:
                    b = bin_index(x_c2)
                    captures[b] += cw[i]
                    caps += 1
                    cactive[i] = 0
                    i += 1
                    continue

                # 2) escape replacement
                if x_c2 < X_BOUND:
                    escapes += 1
                    x_c2 = X_BOUND
                    j_c2 = math.sqrt(np.random.random())
                    cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2

                # 3) floor removal last (don't suppress a capture)
                if j_c2*j_c2 >= cfloor[i]:
                    cactive[i] = 0
                    i += 1
                    continue

                # 4) clone splitting
                w_c = j_c2*j_c2
                for fi in range(floors.size):
                    f = floors[fi]
                    if w_c < f and cw[i] >= f:
                        alpha = 1.0/(1.0 + clones_per_split)
                        cw[i] = cw[i] * alpha
                        for k in range(clones_per_split):
                            if ccount >= MAX_CLONES:
                                break
                            cx[ccount] = x_c2
                            cj[ccount] = j_c2
                            cph[ccount]= ph_c2
                            cw[ccount] = cw[i]
                            cfloor[ccount] = f
                            cactive[ccount] = 1
                            ccount += 1
                        break
                i += 1

        # --- return raw tallies; NO normalization here ---
        # captures: counts per x-bin (weighted); g_time: time-weighted occupancy; total_t0: total time
        return captures, g_time, total_t0, crosses, caps

    return which_bin, P_of_x, T0, run_stream

# ---------------- multiprocessing worker (single stream) ----------------
def _worker_one(args):
    sid, n_relax, floors, clones_per_split, stream_seeds, use_jit, cone_gamma = args
    which_bin, P_of_x, T0, run_stream = _build_kernels(use_jit=use_jit, cone_gamma=cone_gamma)
    # warmup tiny call to pull kernels from cache
    _ = run_stream(1e-6, floors, clones_per_split, X_BINS, DX,
                   int(stream_seeds[sid] ^ 0xABCDEF), cone_gamma)
    return run_stream(n_relax, floors, clones_per_split, X_BINS, DX,
                      int(stream_seeds[sid]), cone_gamma)

# ---------------- parallel driver ----------------
def run_parallel(n_streams=400, n_relax=6.0, floors=None, clones_per_split=9,
                 procs=None, use_jit=True, seed=SEED, show_progress=True, cone_gamma=0.25):

    if floors is None:
        floors = np.array([10.0**(-k) for k in range(0, 9)], dtype=np.float64)  # 1 ... 1e-8
    else:
        floors = np.array(floors, dtype=np.float64)

    if procs is None or procs < 1:
        procs = os.cpu_count() or 1

    import concurrent.futures as cf
    import multiprocessing as mp

    # seeds per stream
    rs = np.random.RandomState(seed)
    stream_seeds = rs.randint(1, 2**31-1, size=n_streams, dtype=np.int64)

    # Progress tracking
    if show_progress:
        start_time = time.time()
        print(f"Starting {n_streams} streams across {procs} processes...", file=sys.stderr)

    cap_total  = np.zeros_like(X_BINS, dtype=np.float64)
    gtime_total= np.zeros_like(X_BINS, dtype=np.float64)
    t0_total   = 0.0
    peri_total = 0
    caps_total = 0
    try:
        with cf.ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [ex.submit(_worker_one, (sid, n_relax, floors, clones_per_split, stream_seeds, use_jit, cone_gamma))
                    for sid in range(n_streams)]
            
            completed = 0
            for f in cf.as_completed(futs):
                cap_b, gtime_b, t0_b, peri_b, caps_b = f.result()
                cap_total  += cap_b
                gtime_total+= gtime_b
                t0_total   += t0_b
                peri_total += peri_b
                caps_total += caps_b
                completed += 1
                
                if show_progress:
                    elapsed = time.time() - start_time
                    rate = completed / max(elapsed, 1e-9)
                    eta = (n_streams - completed) / max(rate, 1e-9)
                    print(f"\rProgress: {completed}/{n_streams} ({100*completed/n_streams:5.1f}%) | "
                          f"Rate: {rate:5.1f} streams/s | ETA: {eta:5.0f}s",
                          end="", flush=True, file=sys.stderr)
                    
    except KeyboardInterrupt:
        if show_progress:
            print(f"\nSimulation interrupted after {completed}/{n_streams} streams", file=sys.stderr)
        raise
    finally:
        if show_progress:
            elapsed_total = time.time() - start_time
            print(f"\nCompleted {completed}/{n_streams} streams in {elapsed_total:.1f}s", file=sys.stderr)

    # ---- FE from capture histogram ----
    denom = float(n_streams) * float(n_relax)
    FE = np.zeros_like(X_BINS)
    np.divide(cap_total, denom*DX, out=FE, where=(DX>0))
    FE_x = FE * X_BINS

    # ---- ḡ(x) from *time-weighted* occupancy ----
    gbar = np.zeros_like(X_BINS)
    if t0_total > 0.0:
        tmp = gtime_total / t0_total
        np.divide(tmp, DX, out=gbar, where=(DX>0))

    # boundary bin containing x_b = 0.2 (by edges)
    norm_idx = 0
    for i in range(X_EDGES.size-1):
        if X_EDGES[i] <= X_BOUND < X_EDGES[i+1]:
            norm_idx = i; break

    scale = 1.0/gbar[norm_idx] if gbar[norm_idx] > 0 else 1.0
    FE_x *= scale

    # ---- robust diags + safety guard ----
    print(f"[norm] bin at x_b=0.2 is index {norm_idx} (center {X_BINS[norm_idx]:.3g}), "
          f"gbar={gbar[norm_idx]:.3e}, scale={scale:.2g}", file=sys.stderr)
    print(f"[norm] total captures (weighted): {cap_total.sum():.3f}, max bin: {cap_total.max():.3f}", file=sys.stderr)
    print(f"[diag] pericenter crossings (total): {peri_total}", file=sys.stderr)
    print(f"[diag] captures (events, unweighted): {caps_total}", file=sys.stderr)

    # If no capture mass was tallied, do NOT print fake FE
    if cap_total.sum() <= 0.0:
        FE_x[:] = 0.0

    # Extra sanity: if caps_total>0 but cap_total is zero, fail fast to catch wiring bugs
    assert not (caps_total > 0 and cap_total.sum() == 0.0), \
        "Got capture events but capture histogram is zero — check worker/unpack/return order."

    return FE_x

# ---------------- CLI ----------------
def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--streams", type=int, default=400, help="number of non-clone streams")
        ap.add_argument("--windows", type=float, default=6.0, help="t0 windows per stream (n_relax)")
        ap.add_argument("--procs", type=int, default=None, help="process count (defaults to all cores)")
        ap.add_argument("--floors_min_exp", type=int, default=10, help="min exponent k for floors 10^{-k}")
        ap.add_argument("--clones", type=int, default=9, help="clones per split")
        ap.add_argument("--nojit", action="store_true", help="disable numba JIT (slow fallback)")
        ap.add_argument("--no-progress", action="store_true", help="disable progress display")
        ap.add_argument("--seed", type=int, default=SEED)
        ap.add_argument("--cone_gamma", type=float, default=0.25, help="prefactor in (29d): floor = max(gamma*(j-jmin), 0.10*jmin)")
        args = ap.parse_args()

        floors = np.array([10.0**(-k) for k in range(0, args.floors_min_exp+1)], dtype=np.float64)

        print(f"Running Monte Carlo simulation with {args.streams} streams...", file=sys.stderr)
        FE = run_parallel(
            n_streams=args.streams,
            n_relax=args.windows,
            floors=floors,
            clones_per_split=args.clones,
            procs=args.procs,
            use_jit=(not args.nojit),
            seed=args.seed,
            show_progress=(not args.no_progress),
            cone_gamma=args.cone_gamma
        )

        print(f"Simulation completed successfully!", file=sys.stderr)
        print("# x_center    FE_star_x  (dimensionless; Δx-normalized; per-stream global time; normalized ḡ(0.225)=1)")
        for xb, val in zip(X_BINS, FE):
            print(f"{xb:10.3g}  {val: .6e}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
