# fe_star_mc_table1_per_stream.py
# New MC variant:
# - Each non-clone "stream" runs until its own elapsed time >= n_relax * t0.
# - Denominator = sum of all non-clone stream durations.
# - Clones share the parent's time (stepped together) but DO NOT contribute to the denominator.
# - Uses Table-1 (e1*, e2*, j1*, j2*) with bilinear interpolation, step rules (29a–29d),
#   pericenter-only capture, w=j^2 cloning, Δx normalization, births from BW-II g0(ln(1+x)).

import numpy as np
import math, random

# ------------------------- Canonical parameters -------------------------
X_D   = 1.0e4
PSTAR = 0.005
SEED  = 20251028
rng = random.Random(SEED)

# ------------------------- Energy-bin centers (Fig. 3) + Δx -------------------------
x_bins = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=float)

def geometric_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(len(centers) + 1, dtype=float)
    # internal edges geometric mean
    for i in range(1, len(centers)):
        edges[i] = math.sqrt(centers[i-1]*centers[i])
    # end edges extrapolated geometrically
    edges[0]  = centers[0]**2 / edges[1]
    edges[-1] = centers[-1]**2 / edges[-2]
    return edges

x_edges = geometric_edges_from_centers(x_bins)
dx = x_edges[1:] - x_edges[:-1]

def which_bin(x: float) -> int:
    return int(np.argmin(np.abs(x_bins - x)))

# ------------------------- Loss-cone boundary -------------------------
def j_min_of_x(x: float) -> float:
    v = 2.0*x/X_D - (x/X_D)**2
    return math.sqrt(v) if v > 0.0 else 0.0

# ------------------------- Orbital period and t0 -------------------------
def P_of_x(x: float) -> float:
    # Keplerian near BH: P(E) = 2π / (-2E)^(3/2), with x = -E (take v0=1)
    E = -x
    return 2.0*math.pi / ((-2.0*E)**1.5)

T0 = P_of_x(1.0) / PSTAR  # t0 = P(E=-1)/P*

# ------------------------- Table-1 starred coefficients grid -------------------------
x_grid = np.array([3.36e-1, 3.31e0, 3.27e1, 3.23e2, 3.18e3], dtype=float)
j_grid = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=float)

neg_e1 = np.array([
    [ 1.41e-1,  1.40e-1,  1.33e-1,  1.29e-1,  1.29e-1],
    [ 1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],
    [ 1.96e-3,  1.59e-3,  2.83e-3,  3.38e-3,  3.49e-3],
    [ 3.64e-3,  4.64e-3,  4.93e-3,  4.97e-3,  4.98e-3],
    [ 8.39e-4,  8.53e-4,  8.56e-4,  8.56e-4,  8.56e-4],
], dtype=float)
e2 = np.array([
    [ 3.14e-1,  4.36e-1,  7.55e-1,  1.37e0,  2.19e0],
    [ 4.45e-1,  8.66e-1,  1.57e0,  2.37e0,  2.73e0],
    [ 1.03e0,   1.80e0,   2.52e0,  2.77e0,  2.81e0],
    [ 1.97e0,   2.54e0,   2.68e0,  2.70e0,  2.70e0],
    [ 1.96e0,   1.96e0,   1.96e0,  1.96e0,  1.96e0],
], dtype=float)
j1 = np.array([
    [-2.52e-2,  5.37e-1,  1.51e0,   3.83e0,   9.58e0],
    [-5.03e-3,  5.40e-3,  2.54e-2,  6.95e-2,  1.75e-1],
    [-2.58e-4,  3.94e-4,  1.45e-3,  3.77e-3,  9.45e-3],
    [-4.67e-6,  2.03e-5,  5.99e-5,  1.53e-4,  3.82e-4],
    [ 3.67e-8,  2.54e-4,  7.09e-7,  1.80e-6,  4.49e-6],
], dtype=float)
j2 = np.array([
    [ 4.68e-1,  6.71e-1,  6.99e-1,  7.03e-1,  7.04e-1],
    [ 6.71e-2,  9.19e-2,  9.48e-2,  9.53e-2,  9.54e-2],
    [ 1.57e-2,  2.12e-2,  2.20e-2,  2.21e-2,  2.21e-2],
    [ 3.05e-3,  4.25e-3,  4.42e-3,  4.44e-3,  4.45e-3],
    [ 3.07e-4,  4.59e-3,  4.78e-4,  4.81e-4,  4.82e-4],
], dtype=float)

e1 = -neg_e1
lx_grid = np.log10(x_grid)

def bilinear_coeffs(x: float, j: float):
    """Interpolate (e1,e2,j1,j2) over (log10 x, j)."""
    lx = np.log10(max(min(x, x_grid[-1]), x_grid[0]))
    jj = max(min(j, j_grid[0]), j_grid[-1])
    i1 = np.searchsorted(lx_grid, lx) - 1
    i1 = max(0, min(i1, len(lx_grid)-2))
    i2 = i1 + 1
    t = (lx - lx_grid[i1]) / (lx_grid[i2] - lx_grid[i1] + 1e-30)
    # j_grid descends
    k1 = 0
    while k1 < len(j_grid)-1 and not (j_grid[k1] >= jj >= j_grid[k1+1]):
        k1 += 1
    k1 = max(0, min(k1, len(j_grid)-2))
    k2 = k1 + 1
    u = (jj - j_grid[k2]) / (j_grid[k1] - j_grid[k2] + 1e-30)

    def interp(arr):
        v11 = arr[i1, k1]; v12 = arr[i1, k2]
        v21 = arr[i2, k1]; v22 = arr[i2, k2]
        v1 = v11*(1-u) + v12*u
        v2 = v21*(1-u) + v22*u
        return v1*(1-t) + v2*t

    return interp(e1), interp(e2), interp(j1), interp(j2)

# ------------------------- Step rules (29a–29d) -------------------------
def choose_n(x, j, e2_val, j2_val, jmin):
    bounds = []
    if e2_val > 0.0:
        bounds.append((0.15*x/e2_val)**2)
    if j2_val > 0.0:
        bounds.append((0.10/j2_val)**2)
        bounds.append((0.40*max(1.0075 - j,0.0)/j2_val)**2)
        floor = max(abs(0.25*j - jmin), 0.10*jmin)
        if floor > 0.0:
            bounds.append((floor/j2_val)**2)
    n = min(bounds) if bounds else 1.0
    return max(1e-6, min(1.0, n))

# ------------------------- Reservoir parameters -------------------------
X_RES_MIN, X_RES_MAX = 0.33, 0.38  # reservoir window near outer boundary

# ------------------------- BW-II Table-1: g0(ln(1+x)) births -------------------------
ln1p = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], dtype=float)
g0_tab = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], dtype=float)

# Piecewise-constant pdf over y-bins (y=ln(1+x)), weight ∝ g0 * Δy
y_edges = np.empty(len(ln1p) + 1, dtype=float)
y_edges[1:-1] = 0.5*(ln1p[:-1] + ln1p[1:])
y_edges[0]    = max(0.0, ln1p[0] - 0.5*(ln1p[1]-ln1p[0]))
y_edges[-1]   = min(9.21, ln1p[-1] + 0.5*(ln1p[-1]-ln1p[-2]))
dy = y_edges[1:] - y_edges[:-1]
w  = np.maximum(g0_tab, 0.0) * np.maximum(dy, 0.0)

# Truncate g0 sampling at reservoir minimum and renormalize
y_res_min = math.log(1.0 + X_RES_MIN)
mask_res = y_edges[:-1] >= y_res_min
w_res = w.copy()
w_res[~mask_res] = 0.0
cumw_res = np.cumsum(w_res); totw_res = cumw_res[-1]

def sample_birth_x():
    """Sample x from reservoir window (0.33-0.38) for replacements."""
    return X_RES_MIN + (X_RES_MAX - X_RES_MIN) * rng.random()

def sample_birth_x_from_g0():
    """Sample x from truncated g0 distribution (x >= 0.33)."""
    if totw_res <= 0:
        return sample_birth_x()  # fallback to reservoir window
    
    u = rng.random() * totw_res
    k = int(np.searchsorted(cumw_res, u))
    k = max(0, min(k, len(dy)-1))
    # uniform within the y-bin
    y0, y1 = y_edges[k], y_edges[k+1]
    y = y0 + (y1 - y0) * rng.random() if w_res[k] > 0 else 0.5*(y0+y1)
    x = math.exp(y) - 1.0
    return min(max(x, X_RES_MIN), 1.0e4)

def sample_birth_j():
    # isotropic: j = sqrt(U), pdf f(j)=2j
    return math.sqrt(rng.random())

# ------------------------- Star + cloning -------------------------
class Star:
    __slots__ = ("x","j","phase","w","is_clone","w_floor")
    def __init__(self, x, j, w=1.0, is_clone=False, w_floor=None):
        self.x = float(x)
        self.j = float(j)
        self.phase = 0.0
        self.w = float(w)
        self.is_clone = bool(is_clone)
        self.w_floor = w_floor

W_FLOORS = [10.0**(-k) for k in range(0, 9)]  # 1 .. 1e-8
W_FLOORS.sort(reverse=True)

def maybe_clone(star, j_prev, clones_list):
    w_prev = j_prev*j_prev
    w_now  = star.j*star.j
    if w_now < w_prev:
        for wf in W_FLOORS:
            if w_now <= wf < w_prev:
                star.w *= 0.1
                star.is_clone = True
                star.w_floor = wf
                for _ in range(9):
                    c = Star(star.x, star.j, w=star.w, is_clone=True, w_floor=wf)
                    c.phase = star.phase
                    clones_list.append(c)
                break

def keep_star(star):
    return not (star.is_clone and (star.j*star.j > (star.w_floor + 1e-16)))

# ------------------------- Simulate ONE non-clone stream -------------------------
def run_one_stream(n_relax: float, max_iters: int = 10_000_000):
    """Return (captures_per_bin, elapsed_t0_stream)."""
    # parent star (non-clone) - use g0 distribution for initial birth
    parent = Star(sample_birth_x_from_g0(), sample_birth_j(), w=1.0, is_clone=False)
    clones = []  # list[Star]
    captures = np.zeros_like(x_bins, dtype=float)
    t_stream = 0.0  # in units of t0

    iters = 0
    while t_stream < n_relax and iters < max_iters:
        iters += 1

        # ---- Step parent first (defines the stream time increment) ----
        x_prev_p = parent.x
        j_prev_p = parent.j
        e1v, e2v, j1v, j2v = bilinear_coeffs(parent.x, parent.j)
        jmin = j_min_of_x(parent.x)
        n_p = choose_n(parent.x, parent.j, e2v, j2v, jmin)

        # Time increment for THIS stream (clones share it; they do not add)
        dt0 = (n_p * P_of_x(parent.x)) / T0
        
        # Guard against huge dt0 steps that destabilize the random walk
        if dt0 > 0.05:
            # substep to keep per-iteration time increments tame
            m = math.ceil(dt0 / 0.05)
            n_p /= m
            dt0 /= m
        
        t_stream += dt0

        # Apply parent kicks
        dE = n_p*e1v + math.sqrt(n_p)*rng.gauss(0.0, e2v)
        dJ = n_p*j1v + math.sqrt(n_p)*rng.gauss(0.0, j2v)
        if math.sqrt(n_p)*j2v > max(1e-12, parent.j/4.0):
            mag = abs(math.sqrt(n_p)*rng.gauss(0.0, j2v))
            dJ = (rng.random()*2.0 - 1.0)*mag

        parent.x = max(1e-10, parent.x + (-dE))
        parent.j = min(1.0, max(0.0, parent.j + dJ))
        parent.phase += n_p

        captured_parent = False
        while parent.phase >= 1.0 - 1e-12:
            parent.phase -= 1.0
            if parent.j < j_min_of_x(parent.x):
                captures[which_bin(parent.x)] += parent.w  # w=1 for parent
                captured_parent = True
                break

        # Parent capture => replace from reservoir; keep stream running
        if captured_parent:
            parent = Star(sample_birth_x(), sample_birth_j(), w=1.0, is_clone=False)
            # don't "continue"; clones still need stepping this iteration

        # maybe clone parent on w=j^2 floors
        new_clones = []
        maybe_clone(parent, j_prev_p, new_clones)

        # ---- Step clones (share parent's time; each uses its own n for dynamics) ----
        # We do NOT advance t_stream for clones; they are variance-reduction only.
        kept_clones = []
        for c in clones + new_clones:
            x_prev = c.x
            j_prev = c.j
            e1v, e2v, j1v, j2v = bilinear_coeffs(c.x, c.j)
            jmin = j_min_of_x(c.x)
            n_c = choose_n(c.x, c.j, e2v, j2v, jmin)

            dE = n_c*e1v + math.sqrt(n_c)*rng.gauss(0.0, e2v)
            dJ = n_c*j1v + math.sqrt(n_c)*rng.gauss(0.0, j2v)
            if math.sqrt(n_c)*j2v > max(1e-12, c.j/4.0):
                mag = abs(math.sqrt(n_c)*rng.gauss(0.0, j2v))
                dJ = (rng.random()*2.0 - 1.0)*mag

            c.x = max(1e-10, c.x + (-dE))
            c.j = min(1.0, max(0.0, c.j + dJ))
            c.phase += n_c

            captured_clone = False
            while c.phase >= 1.0 - 1e-12:
                c.phase -= 1.0
                if c.j < j_min_of_x(c.x):
                    captures[which_bin(c.x)] += c.w
                    captured_clone = True
                    break

            if captured_clone:
                # drop clone
                continue

            # clone w-floor logic (creation–annihilation)
            to_add = []
            maybe_clone(c, j_prev, to_add)
            # retain clone only if it hasn't diffused outward past its floor
            if keep_star(c):
                kept_clones.append(c)
            # append any new sub-clones (they inherit clone status)
            kept_clones.extend(to_add)

        clones = kept_clones

    return captures, t_stream

# ------------------------- Run many streams & aggregate -------------------------
def run_mc(n_nonclone=200, n_relax=6.0):
    total_captures = np.zeros_like(x_bins, dtype=float)
    total_time_t0  = 0.0
    for _ in range(n_nonclone):
        print(f"{_}/{n_nonclone}")
        caps, t_stream = run_one_stream(n_relax=n_relax)
        total_captures += caps
        total_time_t0  += t_stream

    # Differential rate per x: FE* = captures / (total_time_t0 * Δx)
    FE_star = np.zeros_like(x_bins, dtype=float)
    mask = dx > 0
    FE_star[mask] = total_captures[mask] / (max(total_time_t0, 1e-30) * dx[mask])
    FE_star_times_x = FE_star * x_bins
    return FE_star_times_x, total_captures, total_time_t0

# ------------------------- Main -------------------------
if __name__ == "__main__":
    FE_star_x, captures, total_t0 = run_mc(n_nonclone=300, n_relax=6.0)
    print("# x_center    FE_star_x  (dimensionless; Δx-normalized; per-stream time)")
    for x, y in zip(x_bins, FE_star_x):
        print(f"{x:10.3g}  {y: .6e}")
    print(f"\n# Debug: total_time_t0 (sum over streams) = {total_t0:.6f}, total weighted captures = {captures.sum():.6f}")
    print(f"# Reservoir window: x ∈ [{X_RES_MIN:.3f}, {X_RES_MAX:.3f}]")
