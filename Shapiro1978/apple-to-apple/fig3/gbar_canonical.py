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
from multiprocessing import Manager
import multiprocessing as mp
import threading

# Optional numba
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# Kernel cache (per-process, avoids recompilation)
_K_CACHE = None
_K_KEY = None

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
LOGX_GRID = np.log10(X_GRID)  # Precomputed for speed
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

    # Fast P_of_x: avoid pow, use CP / (x*sqrt(x)) where CP = 2π/(2^1.5)
    CP = 2.0 * math.pi / (2.0**1.5)  # constant precomputed
    
    @njit(**fast)
    def P_of_x(x):
        # P = 2π/(-2E)^1.5 with E=-x, so P = 2π/(2x)^1.5 = CP/(x*sqrt(x))
        return CP / (x * math.sqrt(x))

    T0 = P_of_x(1.0)/PSTAR

    @njit(**fast)
    def bin_of(xv, edges):
        # Fast binary search via searchsorted
        i = np.searchsorted(edges, xv, side='right') - 1
        if i < 0:
            i = 0
        if i >= edges.size - 1:
            i = edges.size - 2
        return i

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
        # Use precomputed LOGX_GRID
        lx1 = LOGX_GRID[ix]; lx2 = LOGX_GRID[ix+1]
        tx = (lx - lx1) / (lx2 - lx1 + 1e-30)
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
    def pick_n(x, j, e2v, j2v, cone_gamma_val, disable_capture=False,
               capB=0.10, capC=0.40, kappaE=0.10):
        jmin = j_min_of_x(x)
        nmax = 1e30
        if e2v > 0.0:  # (29a)
            v = (0.15*abs(x)/max(e2v,1e-30))**2
            if v < nmax: nmax = v
        if j2v > 0.0:
            v = (capB/j2v)**2                 # (29b)
            if v < nmax: nmax = v
            v = (capC*max(1.0-j,0.0)/j2v)**2  # (29c)
            if v < nmax: nmax = v
            # (29d) Skip when measuring gbar (no capture during measurement)
            if not disable_capture:
                floor = max(cone_gamma_val*(j - jmin), 0.10*jmin)
                if floor > 0.0:
                    v = (floor/j2v)**2
                    if v < nmax: nmax = v
        # (29e) practical energy-step cap
        kappa = kappaE if disable_capture else 0.07
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
    def step_one(x, j, phase, cone_gamma_val, disable_capture=False, snap_peri=True,
                 capB=0.10, capC=0.40, kappaE=0.10):
        # Return: x2, j2, phase2, n_used, captured?, crossed_pericenter?
        e1, e2v, j1v, j2v, z2v = bilinear_coeffs(x, j)
        n_raw = pick_n(x, j, e2v, j2v, cone_gamma_val, disable_capture, capB, capC, kappaE)
        if snap_peri:
            next_int = math.floor(phase) + 1.0
            crossed = (phase + n_raw >= next_int)
            n = next_int - phase if crossed else n_raw
        else:
            # Don't truncate to pericenter; just note if a crossing happened (diagnostic only).
            n = n_raw
            crossed = (math.floor(phase) != math.floor(phase + n))

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
        # Disable capture during ḡ measurement
        if disable_capture:
            captured = False
        else:
            captured = (crossed and (j2 < j_min_of_x(x2)))
        return x2, j2, phase2, n, captured, crossed

    @njit(**fast)
    def T0_val():
        return T0

    # -------- JIT full per-stream loop (warm-up + measurement) --------
    @njit(**fast)
    def run_stream_jit(n_relax_t0, warmup_t0, cone_gamma_val, seed,
                       snap_peri_meas, capB, capC, kappaE,
                       X_EDGES_local, X_BINS_local):
        # local handles so numba sees shapes
        np.random.seed(seed)
        T0loc = T0_val()
        g_time = np.zeros(X_BINS_local.size, dtype=np.float32)

        # init at reservoir
        x = X_BOUND
        j = math.sqrt(np.random.random())
        w = 1.0
        ph = 0.0

        # warm-up (captures enabled so the system equilibrates)
        t0_used = 0.0
        while t0_used < warmup_t0:
            x_prev = x
            x, j, ph, n_used, cap, crossed = step_one(x, j, ph, cone_gamma_val,
                                                      disable_capture=False, snap_peri=True,
                                                      capB=0.10, capC=0.40, kappaE=0.10)
            t0_used += (n_used * P_of_x(x_prev)) / T0loc
            if cap or (x < X_BOUND):
                x = X_BOUND
                j = math.sqrt(np.random.random())

        # measurement (disable capture, use relaxed parameters)
        t0_used = 0.0
        total_t0 = 0.0
        while t0_used < n_relax_t0:
            x_prev = x
            x, j, ph, n_used, cap, crossed = step_one(x, j, ph, cone_gamma_val,
                                                      disable_capture=True,
                                                      snap_peri=snap_peri_meas,
                                                      capB=capB, capC=capC, kappaE=kappaE)
            dt0 = (n_used * P_of_x(x_prev)) / T0loc
            t0_used += dt0
            total_t0 += dt0

            # adaptive time deposition (looser threshold for speed)
            xp = x_prev if x_prev > 1e-12 else 1e-12
            xn = x if x > 1e-12 else 1e-12
            dlogx = abs(math.log(xn / xp))
            if dlogx <= 0.50:  # looser threshold; measurement is smooth in log x
                xm = math.sqrt(xp * xn)
                b = bin_of(xm, X_EDGES_local)
                g_time[b] += w * dt0
            else:
                nsub = min(3, 1 + int(dlogx / 0.50))  # at most 3 substeps
                ratio = xn / xp
                for k in range(nsub):
                    xm = xp * (ratio ** ((k + 0.5) / nsub))
                    b = bin_of(xm, X_EDGES_local)
                    g_time[b] += w * (dt0 / nsub)

            if x < X_BOUND:
                x = X_BOUND
                j = math.sqrt(np.random.random())

        return g_time, total_t0

    return j_min_of_x, P_of_x, T0_val, bin_of, step_one, run_stream_jit

def _ensure_kernels(use_jit, cone_gamma):
    """Build kernels once per process and cache them."""
    global _K_CACHE, _K_KEY
    key = (bool(use_jit), float(cone_gamma))
    if (_K_CACHE is None) or (_K_KEY != key):
        _K_CACHE = _build_kernels(use_jit=use_jit, cone_gamma=cone_gamma)
        _K_KEY = key
    return _K_CACHE

# -------------- Single-stream driver (no clones; just occupancy) --------------
def run_stream(n_relax_t0, warmup_t0, cone_gamma, use_jit, seed, progress_q=None, hb_dt0=0.5,
               snap_peri_meas=False, capB=0.15, capC=0.60, kappaE=0.20):
    # Build kernels once per process, then reuse
    j_min_of_x, P_of_x, T0_val, bin_of, step_one = _ensure_kernels(use_jit, cone_gamma)
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
        # Keep original pericenter snapping and strict caps in warm-up
        x, j, ph, n_used, cap, crossed = step_one(x, j, ph, cone_gamma,
                                                  disable_capture=False, snap_peri=True,
                                                  capB=0.10, capC=0.40, kappaE=0.10)
        t0_used += (n_used * P_of_x(x_prev)) / T0
        if cap or (x < X_BOUND):
            # replacement at reservoir; keep phase (age)
            x = X_BOUND
            j = math.sqrt(rs.random())

    # Measurement (disable capture to measure background distribution)
    # Use float32 for speed (fine for diagnostics)
    g_time = np.zeros_like(X_BINS, dtype=np.float32)
    t0_used = 0.0
    total_t0 = 0.0
    since_hb = 0.0
    while t0_used < n_relax_t0:
        x_prev = x
        x, j, ph, n_used, cap, crossed = step_one(x, j, ph, cone_gamma,
                                                  disable_capture=True,
                                                  snap_peri=snap_peri_meas,
                                                  capB=capB, capC=capC, kappaE=kappaE)
        dt0 = (n_used * P_of_x(x_prev)) / T0
        t0_used += dt0
        total_t0 += dt0
        since_hb += dt0

        # Adaptive coarse time deposition: midpoint for small moves, substep only for large
        dlogx = abs(math.log(max(x, 1e-12) / max(x_prev, 1e-12)))
        if dlogx <= 0.50:  # looser threshold; measurement is smooth in log x
            xm = math.sqrt(x_prev * x)
            g_time[bin_of(xm, X_EDGES)] += w * dt0
        else:
            # At most 3 substeps for very large moves
            nsub = min(3, 1 + int(dlogx / 0.50))
            ratio = max(x, 1e-12) / max(x_prev, 1e-12)
            for k in range(nsub):
                xm = x_prev * (ratio ** ((k + 0.5) / nsub))
                g_time[bin_of(xm, X_EDGES)] += w * (dt0 / nsub)

        # Still replace at reservoir boundary (but no capture removal during measurement)
        if x < X_BOUND:
            x = X_BOUND
            j = math.sqrt(rs.random())

        # heartbeat (optional, time-based) – avoids big msgs for long windows
        if progress_q is not None and hb_dt0 > 0.0 and since_hb >= hb_dt0:
            try:
                progress_q.put_nowait(since_hb / n_relax_t0)
            except Exception:
                pass
            since_hb = 0.0

    return g_time, total_t0

# -------------- Per-stream worker (for frequent progress updates) --------------
def _worker_one(stream_idx, windows, warmup, cone_gamma, use_jit, seed, progress_q=None,
                hb_dt0=0.5, snap_peri_meas=False, capB=0.15, capC=0.60, kappaE=0.20):
    """Run exactly one stream (JIT)."""
    # ensure kernels compiled once at process start
    j_min_of_x, P_of_x, T0_val, bin_of, step_one, run_stream_jit = _ensure_kernels(use_jit, cone_gamma)
    # Use jitted version for full loop (no Python↔Numba overhead)
    g, t = run_stream_jit(windows, warmup, cone_gamma, int(seed),
                          snap_peri_meas, capB, capC, kappaE,
                          X_EDGES, X_BINS)
    # Progress ping (simplified - jitted loop doesn't support heartbeats, but that's fine)
    if progress_q is not None:
        try:
            progress_q.put_nowait(1.0)
        except Exception:
            pass
    return g, t

# -------------- Chunked worker (for reduced overhead when needed) --------------
def _worker_chunk(start_idx, count, windows, warmup, cone_gamma, use_jit, seeds, progress_q=None,
                  hb_dt0=0.5, snap_peri_meas=False, capB=0.15, capC=0.60, kappaE=0.20):
    """Run count streams starting from start_idx, aggregate results."""
    # compile kernels ONCE for the whole chunk
    j_min_of_x, P_of_x, T0_val, bin_of, step_one, run_stream_jit = _ensure_kernels(use_jit, cone_gamma)
    gsum = np.zeros_like(X_BINS, dtype=np.float32)
    t0sum = 0.0
    for i in range(count):
        idx = start_idx + i
        # Use jitted version for full loop (no Python↔Numba overhead)
        g, t = run_stream_jit(windows, warmup, cone_gamma, int(seeds[idx]),
                             snap_peri_meas, capB, capC, kappaE,
                             X_EDGES, X_BINS)
        gsum += g
        t0sum += t
        # Progress ping per stream
        if progress_q is not None:
            try:
                progress_q.put_nowait(1.0)
            except Exception:
                pass
    return gsum, t0sum

# -------------- Parallel aggregator --------------
def run_parallel(streams, windows, warmup, procs, cone_gamma, use_jit, seed, normalize,
                 hb_dt0=0.5, snap_peri_meas=False, capB=0.15, capC=0.60, kappaE=0.20, chunk=8):
    import concurrent.futures as cf
    import multiprocessing as mp
    rs = np.random.RandomState(seed)
    seeds = rs.randint(1, 2**31-1, size=streams, dtype=np.int64)

    gsum = np.zeros_like(X_BINS, dtype=np.float64)
    t0sum = 0.0

    if procs is None or procs < 1:
        procs = os.cpu_count() or 1

    total_start = time.time()
    
    # Ensure immediate flushing even when stderr is piped
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    
    # Batch several streams per worker to amortize JIT/overhead.
    # Aim ~2–4 tasks per process outstanding (so for 48 procs, ~100–200 chunks total).
    CHUNK = chunk
    use_chunks = (CHUNK > 1)
    
    if use_chunks:
        chunks = []
        for start in range(0, streams, CHUNK):
            count = min(CHUNK, streams - start)
            chunks.append((start, count))
        sys.stderr.write(f"Starting: {streams} streams ({len(chunks)} chunks), {procs} processes\n")
    else:
        sys.stderr.write(f"Starting: {streams} streams, {procs} processes\n")
    sys.stderr.flush()
    
    # Progress queue + printer thread for per-stream updates
    # Note: Manager().Queue() is required for ProcessPoolExecutor (mp.Queue() isn't picklable)
    mgr = Manager()
    progress_q = mgr.Queue()
    done = 0
    stop_event = threading.Event()
    
    def printer():
        nonlocal done
        last = 0.0
        while not stop_event.is_set() or done < streams:
            try:
                inc = progress_q.get(timeout=0.2)
                done += float(inc)
            except Exception:
                pass
            now = time.time()
            if now - last >= 0.2 or (stop_event.is_set() and done >= streams):
                # clamp to guard against tiny FP overshoot
                if done > streams:
                    done = float(streams)
                pct = 100.0 * done / streams if streams > 0 else 0.0
                elapsed = now - total_start
                # Robust ETA: only compute if we have at least one completion
                if done > 0:
                    rate = done / elapsed
                    eta = (streams - done) / rate if rate > 0 else float('inf')
                else:
                    rate = 0.0
                    eta = float('inf')
                
                # Format time nicely
                def fmt_time(sec):
                    if not math.isfinite(sec):
                        return "  —  "
                    if sec < 60:
                        return f"{sec:.1f}s"
                    elif sec < 3600:
                        return f"{sec/60:.1f}m"
                    else:
                        h = int(sec // 3600)
                        m = int((sec % 3600) // 60)
                        return f"{h}h{m}m"
                
                eta_str = fmt_time(eta)
                elapsed_str = fmt_time(elapsed)
                # \x1b[K clears to end-of-line so old characters don't linger
                sys.stderr.write(
                    f"\rProgress: {done:6.1f}/{streams} ({pct:5.1f}%) | "
                    f"Rate: {rate:5.2f} streams/s | "
                    f"Elapsed: {elapsed_str} | "
                    f"Remaining: {eta_str}\x1b[K"
                )
                sys.stderr.flush()
                last = now
        # Drain any remaining items
        while True:
            try:
                inc = progress_q.get_nowait()
                done += float(inc)
            except Exception:
                break
        # Final update and newline
        if done > 0:
            pct = 100.0 * done / streams if streams > 0 else 0
            elapsed = time.time() - total_start
            # Format time nicely (reuse same logic)
            if elapsed < 60:
                elapsed_str = f"{elapsed:.1f}s"
            elif elapsed < 3600:
                elapsed_str = f"{elapsed/60:.1f}m"
            else:
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                elapsed_str = f"{h}h{m}m"
            sys.stderr.write(
                f"\rProgress: {done:6.1f}/{streams} ({pct:5.1f}%) | "
                f"Elapsed: {elapsed_str}\x1b[K"
            )
            sys.stderr.flush()
        sys.stderr.write("\n")
        sys.stderr.flush()
    
    t_prn = threading.Thread(target=printer, daemon=True)
    t_prn.start()
    
    try:
        # Prefer 'fork' so compiled machine code is inherited (Linux)
        try:
            ctx = mp.get_context("fork")
        except (ValueError, RuntimeError):
            ctx = mp.get_context()  # fallback (e.g., macOS uses 'spawn')
        with cf.ProcessPoolExecutor(max_workers=procs, mp_context=ctx) as ex:
            if use_chunks:
                futs = {ex.submit(_worker_chunk, start, count, windows, warmup, cone_gamma,
                                  use_jit, seeds, progress_q, hb_dt0,
                                  snap_peri_meas, capB, capC, kappaE): (start, count)
                        for start, count in chunks}
            else:
                # Per-stream tasks (rare after chunking enabled)
                futs = {ex.submit(_worker_one, sid, windows, warmup, cone_gamma, use_jit,
                                  seeds[sid], progress_q, hb_dt0,
                                  snap_peri_meas, capB, capC, kappaE): sid
                        for sid in range(streams)}
            
            for f in cf.as_completed(futs):
                if use_chunks:
                    g_chunk, t_chunk = f.result()
                    gsum += g_chunk.astype(np.float64)
                    t0sum += t_chunk
                else:
                    g_one, t_one = f.result()
                    gsum += g_one.astype(np.float64)
                    t0sum += t_one
    finally:
        # Stop printer thread
        stop_event.set()
        t_prn.join(timeout=1.0)
    
    # Final status
    total_elapsed = time.time() - total_start
    def fmt_time(sec):
        if sec < 60:
            return f"{sec:.1f}s"
        elif sec < 3600:
            return f"{sec/60:.1f}m"
        else:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            return f"{h}h{m}m"
    sys.stderr.write(f"Completed: {streams} streams, {t0sum:.2f} total t0 measured in {fmt_time(total_elapsed)}\n")
    sys.stderr.flush()

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
    ap.add_argument("--streams", type=int, default=200, help="number of independent streams (default: 200). For faster progress, use more streams with shorter windows (e.g., 768 streams × 12 t0)")
    ap.add_argument("--windows", type=float, default=48.0, help="measurement time per stream in t0 (default: 48). For faster progress updates, use shorter windows like 8-12 t0 with more streams")
    ap.add_argument("--warmup", type=float, default=2.0, help="equilibration time in t0 before tallying")
    ap.add_argument("--procs", type=int, default=None, help="processes (defaults to all cores)")
    ap.add_argument("--nojit", action="store_true", help="disable numba JIT")
    ap.add_argument("--cone-gamma", type=float, default=1.0, help="prefactor in (29d): floor = max(gamma*(j-jmin), 0.10*jmin)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--normalize", action="store_true", help="scale so ḡ(x≈0.225)=1 (filled circles)")
    ap.add_argument("--plot", type=str, default="", help="optional output PNG path to plot filled circles")
    ap.add_argument("--hb", type=float, default=0.5, help="heartbeat interval in t0 per worker (default 0.5)")
    ap.add_argument("--chunk", type=int, default=8, help="streams per task (1 = per-stream tasks; default 8)")
    ap.add_argument("--snap-peri-meas", action="store_true",
                    help="during measurement, still snap steps to pericenter (slower; off by default)")
    ap.add_argument("--capB", type=float, default=0.15, help="(29b) factor during measurement (default 0.15; warm-up uses 0.10)")
    ap.add_argument("--capC", type=float, default=0.60, help="(29c) factor during measurement (default 0.60; warm-up uses 0.40)")
    ap.add_argument("--kappaE", type=float, default=0.20, help="(29e) energy-step cap during measurement (default 0.20; warm-up uses 0.10)")
    args = ap.parse_args()

    gbar = run_parallel(
        streams=args.streams,
        windows=args.windows,
        warmup=args.warmup,
        procs=args.procs,
        cone_gamma=args.cone_gamma,
        use_jit=(not args.nojit),
        seed=args.seed,
        normalize=args.normalize,
        hb_dt0=args.hb,
        snap_peri_meas=args.snap_peri_meas,
        capB=args.capB, capC=args.capC, kappaE=args.kappaE,
        chunk=args.chunk
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
