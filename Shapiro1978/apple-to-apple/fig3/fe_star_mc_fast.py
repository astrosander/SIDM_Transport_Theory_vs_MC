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
def _build_kernels(use_jit=True):
    nb_njit = (lambda *a, **k: (lambda f: f))
    njit = nb.njit if (use_jit and HAVE_NUMBA) else nb_njit
    fastmath = dict(fastmath=True, nogil=True, cache=False) if (use_jit and HAVE_NUMBA) else {}

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
    def pick_n(x, j, e2v, j2v):
        jmin = j_min_of_x(x)
        nmax = 1e30

        # (29a)  sqrt(n)*e2 <= 0.15*|E|  (x = -E)
        if e2v > 0.0:
            v = (0.15*abs(x) / max(e2v, 1e-30))**2
            if v < nmax: nmax = v

        if j2v > 0.0:
            # (29b)  |ΔJ|_rms <= 0.10   (ABSOLUTE cap; do NOT divide by j)
            v = (0.10 / j2v)**2
            if v < nmax: nmax = v

            # (29c)  |ΔJ|_rms <= 0.40*(1.0075 − j)   (linear distance to circular)
            v = (0.40 * max(1.0075 - j, 0.0) / j2v)**2
            if v < nmax: nmax = v

            # (29d)  |ΔJ|_rms <= max(|0.25*j − jmin|, 0.10*jmin)
            floor = max(abs(0.25*j - jmin), 0.10 * jmin)
            if floor > 0.0:
                v = (floor / j2v)**2
                if v < nmax: nmax = v

        # choose n; tiny safety floor to avoid zero
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
    def step_one(x, j, phase):
        # One stochastic step (E=-x): returns x_new, j_new, phase_new, n_used, captured?
        e1, e2v, j1v, j2v, z2v = bilinear_coeffs(x, j)
        n_raw = pick_n(x, j, e2v, j2v)

        # Pericenter truncation: if we would cross an integer, land exactly there
        next_int = math.floor(phase) + 1.0
        if phase + n_raw > next_int:
            n = next_int - phase
        else:
            n = n_raw

        # Correlated diffusion
        y1, y2 = correlated_normals(e2v, j2v, z2v)
        dE = n*e1 + math.sqrt(n)*y1*e2v
        x_new = x - dE
        if x_new < 1e-12:
            x_new = 1e-12

        # J update: 2D RW only inside loss cone and small j (paper's gate)
        use_2d = (
            (math.sqrt(n)*j2v > max(1e-12, j/4.0)) and  # "step large vs J"
            (j < 0.4) and                               # small j
            (j < j_min_of_x(x))                         # INSIDE the loss cone
        )
        if use_2d:
            # 2D RW (eq. 28)
            z1 = np.random.normal()
            z2 = np.random.normal()
            j_new = math.sqrt( (j + math.sqrt(n)*z1*j2v)**2 + (math.sqrt(n)*z2*j2v)**2 )
            used_2d = True
        else:
            # 1D Gaussian with correlation (eq. 27b)
            j_new = j + n*j1v + math.sqrt(n)*y2*j2v
            used_2d = False
        if j_new < 0.0: j_new = 0.0
        if j_new > 1.0: j_new = 1.0

        # advance phase by n; pericenter if integer crossed
        phase_before = phase
        phase = phase + n
        captured = False
        if int(math.floor(phase)) > int(math.floor(phase_before)):
            if j_new < j_min_of_x(x_new):
                captured = True

        return x_new, j_new, phase, n, captured, used_2d

    @njit(**fastmath)
    def run_stream(n_relax, floors, clones_per_split, x_bins, dx, seed):
        np.random.seed(seed)

        # parent (time carrier)
        x = X_BOUND
        j = math.sqrt(np.random.random())  # isotropic j
        phase = 0.0
        w = 1.0

        # clone pool
        MAX_CLONES = 2048
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
        pericross = 0
        caps = 0
        escapes = 0
        g_boundary_sum = 0.0
        n_2d = 0
        n_1d = 0

        # ḡ snapshots (once per t0): count weighted stars in each x-bin
        g_counts = np.zeros(x_bins.size, dtype=np.float64)
        n_snaps  = 0
        next_snap = 1.0  # 1,2,3,... t0

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
            x, j, phase, n_used, cap, used_2d = step_one(x, j, phase)
            # advance time
            t0_used += (n_used * P_of_x(x_prev)) / T0
            pericross += 1  # Count steps as proxy for pericenter crossings
            if used_2d:
                n_2d += 1
            else:
                n_1d += 1

            # Check for escape from cusp (outward diffusion)
            if x < X_BOUND:
                # star left the cusp -> replace at reservoir (no capture counted)
                escapes += 1
                x = X_BOUND
                j = math.sqrt(np.random.random())   # isotropic J
                # phase unchanged (keep age)
                escaped = True
            else:
                escaped = False

            if not escaped:
                # capture at pericenter
                if cap:
                    caps += 1
                    b = bin_index(x)
                    captures[b] += w
                    # replace at x_b with isotropic j; keep phase (per paper)
                    x = X_BOUND
                    j = math.sqrt(np.random.random())

                # splitting on w=j^2 floors (parent not deleted)
                w_now = j*j
                for fi in range(floors.size):
                    f = floors[fi]
                    if w_now < f and w >= f:
                        alpha = 1.0/(1.0 + clones_per_split)
                        new_w = w*alpha
                        w = new_w
                        for k in range(clones_per_split):
                            if ccount < MAX_CLONES:
                                cx[ccount] = x
                                cj[ccount] = j
                                cph[ccount]= phase
                                cw[ccount] = new_w
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
                x_c2, j_c2, ph_c2, n_c, cap_c, used_2d_c = step_one(x_c, j_c, ph_c)
                cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2
                if used_2d_c:
                    n_2d += 1
                else:
                    n_1d += 1

                # outward across its floor => remove
                if j_c2*j_c2 >= cfloor[i]:
                    cactive[i] = 0; i += 1; continue

                # escaped the cusp? -> replace at reservoir (do NOT count a capture)
                if x_c2 < X_BOUND:
                    # replace the clone state at reservoir
                    escapes += 1
                    x_c2 = X_BOUND
                    j_c2 = math.sqrt(np.random.random())
                    # keep phase
                    cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2
                    # (do not mark inactive; keep evolving this clone)
                    # proceed to possible further splitting below
                else:
                    # pericenter-capture?
                    if cap_c:
                        b = bin_index(x_c2)
                        captures[b] += cw[i]
                        cactive[i] = 0
                        i += 1
                        continue

                # further splitting for clone
                w_c = j_c2*j_c2
                for fi in range(floors.size):
                    f = floors[fi]
                    if w_c < f and cw[i] >= f:
                        alpha = 1.0/(1.0 + clones_per_split)
                        new_w = cw[i]*alpha
                        cw[i] = new_w
                        for k in range(clones_per_split):
                            if ccount < MAX_CLONES:
                                cx[ccount] = x_c2
                                cj[ccount] = j_c2
                                cph[ccount]= ph_c2
                                cw[ccount] = new_w
                                cfloor[ccount] = f
                                cactive[ccount] = 1
                                ccount += 1
                        break
                i += 1

            # snapshot ḡ once per t0 boundary
            while t0_used >= next_snap:
                # parent
                g_counts[bin_index(x)] += w
                # parent at boundary?
                if abs(x - X_BOUND) < 1e-12:
                    g_boundary_sum += w
                # active clones
                for ii in range(ccount):
                    if cactive[ii]:
                        g_counts[bin_index(cx[ii])] += cw[ii]
                        # clones at boundary?
                        if abs(cx[ii] - X_BOUND) < 1e-12:
                            g_boundary_sum += cw[ii]
                n_snaps += 1
                next_snap += 1.0

        # --- return raw tallies; NO normalization here ---
        # captures: counts per x-bin (weighted); g_counts: snapshot weights per x-bin; n_snaps: #snapshots
        return captures, g_counts, n_snaps

    return which_bin, P_of_x, T0, run_stream

# ---------------- multiprocessing worker ----------------
def _worker(args):
    try:
        batch_ids, n_relax, floors, clones_per_split, stream_seeds, use_jit, _ = args
        which_bin, P_of_x, T0, run_stream = _build_kernels(use_jit=use_jit)
        cap_sum  = np.zeros_like(X_BINS, dtype=np.float64)
        g_sum    = np.zeros_like(X_BINS, dtype=np.float64)
        snap_sum = 0.0

        for sid in batch_ids:
            cap_i, g_i, n_snaps_i = run_stream(n_relax, floors, clones_per_split, X_BINS, DX, int(stream_seeds[sid]))
            cap_sum += cap_i
            g_sum   += g_i
            snap_sum += float(n_snaps_i)

        # return a tuple
        return cap_sum, g_sum, snap_sum
    except Exception as e:
        print(f"Worker error: {e}", file=sys.stderr)
        z = np.zeros_like(X_BINS, dtype=np.float64)
        return z, z, 0.0

# ---------------- parallel driver ----------------
def run_parallel(n_streams=400, n_relax=6.0, floors=None, clones_per_split=9,
                 procs=None, use_jit=True, seed=SEED, show_progress=True):

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

    # split indices
    batch_sizes = [n_streams // procs] * procs
    for i in range(n_streams % procs):
        batch_sizes[i] += 1
    batches = []
    idx = 0
    for bsz in batch_sizes:
        if bsz > 0:
            batches.append(tuple(range(idx, idx+bsz)))
        idx += bsz

    # Progress tracking
    if show_progress:
        completed_streams = 0
        start_time = time.time()
        last_update = start_time
        
        print(f"Starting {n_streams} streams across {procs} processes...", file=sys.stderr)

    cap_total  = np.zeros_like(X_BINS, dtype=np.float64)
    g_total    = np.zeros_like(X_BINS, dtype=np.float64)
    snap_total = 0.0
    try:
        with cf.ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [ex.submit(_worker, (batch, n_relax, floors, clones_per_split, stream_seeds, use_jit, None))
                    for batch in batches if len(batch)>0]
            
            # Monitor progress
            if show_progress:
                for i, f in enumerate(cf.as_completed(futs)):
                    cap_b, g_b, snap_b = f.result()
                    cap_total  += cap_b
                    g_total    += g_b
                    snap_total += snap_b
                    completed_streams += len(batches[i])
                    
                    # Update progress display
                    current_time = time.time()
                    if current_time - last_update >= 1.0:  # Update every second
                        elapsed = current_time - start_time
                        rate = completed_streams / elapsed if elapsed > 0 else 0
                        eta = (n_streams - completed_streams) / rate if rate > 0 else 0
                        
                        progress_pct = (completed_streams / n_streams) * 100
                        print(f"\rProgress: {completed_streams:4d}/{n_streams} ({progress_pct:5.1f}%) | "
                              f"Rate: {rate:5.1f} streams/s | ETA: {eta:5.0f}s", 
                              end="", flush=True, file=sys.stderr)
                        last_update = current_time
            else:
                for f in cf.as_completed(futs):
                    cap_b, g_b, snap_b = f.result()
                    cap_total  += cap_b
                    g_total    += g_b
                    snap_total += snap_b
                    
    except KeyboardInterrupt:
        if show_progress:
            print(f"\nSimulation interrupted after {completed_streams}/{n_streams} streams", file=sys.stderr)
        raise
    finally:
        if show_progress:
            elapsed_total = time.time() - start_time
            print(f"\nCompleted {completed_streams}/{n_streams} streams in {elapsed_total:.1f}s", file=sys.stderr)

    # ---- compute FE*(x)*x from raw totals (no per-stream scaling) ----
    # FE(x) = captures / (n_streams * n_relax * Δx)
    FE = np.zeros_like(X_BINS, dtype=np.float64)
    denom = float(n_streams) * float(n_relax)
    for bi in range(X_BINS.size):
        if DX[bi] > 0.0 and denom > 0.0:
            FE[bi] = cap_total[bi] / (denom * DX[bi])

    FE_x = FE * X_BINS  # what Fig. 3 plots

    # ---- build global ḡ(x) from snapshots and normalize at boundary ----
    gbar = np.zeros_like(X_BINS, dtype=np.float64)
    if snap_total > 0.0:
        for bi in range(X_BINS.size):
            if DX[bi] > 0.0:
                gbar[bi] = (g_total[bi] / snap_total) / DX[bi]

    # find the bin that contains x_b = 0.2 by edges
    norm_idx = 0
    for i in range(X_EDGES.size - 1):
        if (X_EDGES[i] <= X_BOUND) and (X_BOUND < X_EDGES[i+1]):
            norm_idx = i
            break

    scale = 1.0
    if gbar[norm_idx] > 0.0:
        scale = 1.0 / gbar[norm_idx]

    FE_x *= scale

    # Debug normalization info
    print(f"[norm] bin at x_b=0.2 is index {norm_idx} (center {X_BINS[norm_idx]:.3g}), "
          f"gbar={gbar[norm_idx]:.3e}, scale={scale:.3g}", file=sys.stderr)
    print(f"[norm] total captures: {cap_total.sum():.0f}, max single bin: {cap_total.max():.0f}", file=sys.stderr)

    return FE_x

# ---------------- CLI ----------------
def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--streams", type=int, default=400, help="number of non-clone streams")
        ap.add_argument("--windows", type=float, default=6.0, help="t0 windows per stream (n_relax)")
        ap.add_argument("--procs", type=int, default=None, help="process count (defaults to all cores)")
        ap.add_argument("--floors_min_exp", type=int, default=8, help="min exponent k for floors 10^{-k}")
        ap.add_argument("--clones", type=int, default=9, help="clones per split")
        ap.add_argument("--nojit", action="store_true", help="disable numba JIT (slow fallback)")
        ap.add_argument("--no-progress", action="store_true", help="disable progress display")
        ap.add_argument("--seed", type=int, default=SEED)
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
            show_progress=(not args.no_progress)
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
