# SM78 Exact Physics Implementation Guide

This guide shows exactly where in `gbar_mc_fast_sm78_fig3_v2.py` to replace parametric approximations with the exact SM78 formulas from the paper.

## Overview

The code currently uses **tunable parameters** (`--E2-x-power`, `--J2-scale`, `--lc_scale`, etc.) that approximate SM78's physics. To match the paper exactly, you need to:

1. Replace parametric diffusion coefficients with exact SM78 formulas
2. Lock in loss-cone boundary to exact SM78 definition
3. Fix outer boundary/injection to match paper's prescription
4. Remove tunable scaling factors (they become constants from the paper)

---

## 1. Diffusion Coefficients (Lines 305-408)

### Current Location: `bilinear_coeffs()` function

**EXACT CODE FROM YOUR FILE (lines 357-408):**

```python
# Lines 357-408 in bilinear_coeffs()
PSTAR_CANON = 0.005

if zero_coeffs_flag:
    e1_star = 0.0
    E2_star = 0.0
    J2_star = 0.0
    j1_star = 0.0
    covEJ_star = 0.0
else:
    # Interpolate from Table 1
    e1_star = get(NEG_E1)
    E2_star = max(get(E2), 0.0)
    J2_star = max(get(J2), 0.0)
    j1_star = get(J1)
    covEJ_star = get(ZETA2)
    
    # Scale by pstar (THIS IS OK - matches paper's P* scaling)
    scale = pstar_val / PSTAR_CANON
    e1_star *= scale
    E2_star *= scale
    j1_star *= scale
    J2_star *= scale
    covEJ_star *= scale

# Zero out drift/diffusion if diagnostic flags set
if zero_drift_flag:
    e1_star = 0.0
    j1_star = 0.0

if zero_diffusion_flag:
    E2_star = 0.0
    J2_star = 0.0
    covEJ_star = 0.0

# TUNABLE SCALINGS (THESE NEED TO BE REMOVED/REPLACED):
E2_star *= E2_scale_local          # <-- REMOVE: should be fixed by paper
J2_star *= J2_scale_local          # <-- REMOVE: should be fixed by paper
covEJ_star *= covEJ_scale_local    # <-- REMOVE: should be fixed by paper

# TUNABLE ENERGY-DEPENDENT SCALING (THIS NEEDS TO BE REPLACED):
if abs(E2_x_power_local) > 1e-10:
    x_scale = x_clamp / E2_x_ref_local
    if x_scale > 100.0 and E2_x_power_local > 0.0:
        x_scale = 100.0
    E2_star *= (x_scale ** E2_x_power_local)  # <-- REPLACE with exact formula

# Convert to physical units
v0_sq = 1.0
Jmax = 1.0 / math.sqrt(2.0 * x_clamp)

e1 = -e1_scale_val * e1_star * v0_sq      # <-- e1_scale_val should be fixed
sigE_star = math.sqrt(max(E2_star, 0.0))
sigE = sigE_star * v0_sq                   # <-- This becomes exact DE2 from paper

j1 = j1_star * Jmax
sigJ_star = math.sqrt(max(J2_star, 0.0))
sigJ = sigJ_star * Jmax                    # <-- This becomes exact DJ2 from paper

covEJ = covEJ_star * v0_sq * Jmax
```

### What to Change:

**Step 1:** Add a new function `diff_coeffs_sm78_exact()` that implements the paper's formulas directly:

```python
# ADD THIS NEW FUNCTION near line 300 (before bilinear_coeffs)

@njit(**fastmath)
def diff_coeffs_sm78_exact(x, j, sm78_params):
    """
    Exact SM78 orbit-averaged diffusion coefficients.
    
    Args:
        x: dimensionless energy (x = -E/v0^2 in SM78 notation)
        j: dimensionless angular momentum (j = J/J_circ(E))
        sm78_params: struct/namespace with SM78 constants:
            - lnLambda: Coulomb logarithm
            - m_star: stellar mass
            - M_BH: black hole mass
            - r_influence: influence radius
            - ... (all constants from paper)
    
    Returns:
        (e1, E2, j1, J2, covEJ) in physical units
    """
    # TODO: COPY EXACT FORMULAS FROM SM78 PAPER HERE
    # Example structure (REPLACE WITH ACTUAL PAPER FORMULAS):
    
    # Energy drift: <ΔE> = e1
    # From SM78 eq. (XX):
    e1 = (sm78_params.A_e1 
          * x**sm78_params.alpha_e1 
          * (1.0 + sm78_params.beta_e1 * j**2))
    
    # Energy diffusion: <(ΔE)^2> = E2
    # From SM78 eq. (YY):
    E2 = (sm78_params.A_E2 
          * x**sm78_params.alpha_E2 
          * (1.0 + sm78_params.beta_E2 * j**2))
    
    # Angular momentum drift: <ΔJ> = j1
    j1 = (sm78_params.A_j1 
          * x**sm78_params.alpha_j1 
          * j * (1.0 + sm78_params.beta_j1 * j**2))
    
    # Angular momentum diffusion: <(ΔJ)^2> = J2
    J2 = (sm78_params.A_J2 
          * x**sm78_params.alpha_J2 
          * j**2 * (1.0 + sm78_params.beta_J2 * j**2))
    
    # E-J covariance: <ΔE ΔJ> = covEJ
    covEJ = (sm78_params.A_covEJ 
             * x**sm78_params.alpha_covEJ 
             * j * (1.0 + sm78_params.beta_covEJ * j**2))
    
    return e1, E2, j1, J2, covEJ
```

**Step 2:** Modify `bilinear_coeffs()` to use the exact formulas when `use_sm78_physics=True`:

```python
# MODIFY bilinear_coeffs() around line 305:

@njit(**fastmath)
def bilinear_coeffs(x, j, pstar_val, e1_scale_val, zero_coeffs_flag,
                    zero_drift_flag, zero_diffusion_flag,
                    E2_scale_local, J2_scale_local, covEJ_scale_local,
                    E2_x_power_local, E2_x_ref_local,
                    use_sm78_physics=False, sm78_params=None):
    # ... existing x/j clamping code ...
    
    if use_sm78_physics:
        # USE EXACT SM78 FORMULAS
        e1, E2, j1, J2, covEJ = diff_coeffs_sm78_exact(x, j, sm78_params)
        
        # Apply only the pstar scaling (if needed by paper's definition)
        scale = pstar_val / PSTAR_CANON
        e1 *= scale
        E2 *= scale
        j1 *= scale
        J2 *= scale
        covEJ *= scale
        
        # NO OTHER SCALING (remove E2_scale, J2_scale, E2_x_power, etc.)
        
    else:
        # OLD CODE: interpolate Table 1 + apply tunable scalings
        e1_star = get(NEG_E1)
        E2_star = max(get(E2), 0.0)
        # ... rest of current code ...
        # (keep this for backward compatibility / diagnostics)
    
    # Convert to physical units (v0_sq, Jmax) as before
    v0_sq = 1.0
    Jmax = 1.0 / math.sqrt(2.0 * x_clamp)
    
    e1_phys = -e1_scale_val * e1_star * v0_sq  # or e1 directly if using SM78
    sigE = math.sqrt(max(E2_star, 0.0)) * v0_sq  # or sqrt(E2) if using SM78
    # ... etc
```

**Step 3:** Add SM78 constants struct at top of file:

```python
# ADD NEAR TOP OF FILE (around line 20, after constants)

class SM78Params:
    """Constants from SM78 paper for exact physics implementation."""
    def __init__(self):
        # TODO: Fill in exact values from SM78 paper
        self.lnLambda = 15.0  # Example - get from paper
        self.m_star = 1.0  # Example - get from paper
        self.M_BH = 1.0e6  # Example - get from paper
        self.r_influence = 1.0  # Example - get from paper
        
        # Diffusion coefficient prefactors (from paper's formulas)
        self.A_e1 = 1.0  # TODO: from paper
        self.alpha_e1 = 1.0  # TODO: from paper
        self.beta_e1 = 1.0  # TODO: from paper
        # ... etc for all coefficients

SM78_PARAMS = SM78Params()  # Global instance
```

---

## 2. Loss-Cone Boundary (Lines 249-254)

### Current Location: `j_min_of_x()` function

**EXACT CODE FROM YOUR FILE (lines 249-254):**

```python
@njit(**fastmath)
def j_min_of_x(x, lc_scale_val, noloss_flag):
    if noloss_flag:
        return 0.0
    v = 2.0 * x / X_D - (x / X_D) ** 2
    jmin = math.sqrt(v) if v > 0.0 else 0.0
    return lc_scale_val * jmin  # <-- REMOVE lc_scale_val (should be 1.0 for exact SM78)
```

**Where it's used in `pick_n()` (lines 422-457):**

```python
# Line 426: Get loss-cone boundary
jmin = j_min_of_x(x, lc_scale_val, noloss_flag)

# Lines 450-457: Loss-cone step limiting with floor
if noloss_flag:
    n_J_lc = n_max
else:
    Jmin = jmin * Jmax
    gap_scale = cone_gamma_val if lc_gap_scale is None else lc_gap_scale
    gap = gap_scale * abs(J - Jmin)                    # <-- REMOVE gap scaling
    if lc_floor_frac > 0.0:
        floor = lc_floor_frac * Jmin                   # <-- REMOVE floor
    else:
        floor = 0.0
    n_J_lc = (step_size_factor_val * floor / sigJ) ** 2  # <-- Use exact j_lc, no floor
```

### What to Change:

**Replace the exact code above with SM78 formulas:**

```python
# MODIFY j_min_of_x() around line 249:

@njit(**fastmath)
def j_min_of_x(x, lc_scale_val, noloss_flag, use_sm78_physics=False, sm78_params=None):
    if noloss_flag:
        return 0.0
    
    if use_sm78_physics:
        # EXACT SM78 LOSS-CONE DEFINITION
        # TODO: Copy exact formula from SM78 paper here
        # The current formula v = 2.0 * x / X_D - (x / X_D) ** 2
        # is likely already correct, but remove the lc_scale_val multiplier:
        v = 2.0 * x / X_D - (x / X_D) ** 2
        jmin = math.sqrt(v) if v > 0.0 else 0.0
        return jmin  # NO lc_scale_val scaling
    else:
        # OLD CODE: parametric approximation (keep for backward compatibility)
        v = 2.0 * x / X_D - (x / X_D) ** 2
        jmin = math.sqrt(v) if v > 0.0 else 0.0
        return lc_scale_val * jmin
```

**Also modify `pick_n()` function (around line 450-457):**

```python
# MODIFY pick_n() around line 450-457:

if noloss_flag:
    n_J_lc = n_max
else:
    Jmin = jmin * Jmax
    
    if use_sm78_physics:
        # SM78: No artificial floor or gap scaling
        # Use exact distance to loss-cone boundary
        n_J_lc = (step_size_factor_val * abs(J - Jmin) / sigJ) ** 2
    else:
        # OLD CODE: with floor and gap (keep for backward compatibility)
        gap_scale = cone_gamma_val if lc_gap_scale is None else lc_gap_scale
        gap = gap_scale * abs(J - Jmin)
        if lc_floor_frac > 0.0:
            floor = lc_floor_frac * Jmin
        else:
            floor = 0.0
        n_J_lc = (step_size_factor_val * floor / sigJ) ** 2
```

---

## 3. Outer Boundary / Injection (Multiple locations)

### Current Locations:
- Line 19: `X_BOUND = 0.2`
- Lines 702-712, 754-764, 854-882, 931-960: Injection logic

**Current code pattern:**
```python
# Example from line 702-712:
if cap or (x < X_BOUND):
    if noloss_flag:
        x = sample_x_from_g0_jit()  # Sample from Bahcall-Wolf reservoir
    else:
        if outer_injection:
            x = sample_x_from_g0_jit()
            while x < outer_inj_x_min:
                x = sample_x_from_g0_jit()
        else:
            x = X_BOUND  # Fixed injection at x = 0.2
    j = math.sqrt(np.random.random())  # Isotropic J
```

### What to Change:

**Implement exact SM78 outer boundary condition:**

```python
# ADD NEW FUNCTION near line 700:

@njit(**fastmath)
def inject_star_sm78(use_sm78_physics=False, sm78_params=None):
    """
    Inject star according to SM78 outer boundary condition.
    
    SM78 boundary condition (i): 
    "Each consumed or escaping non-clone star is replaced by a core star 
    at the outer boundary of the cusp with the maximum cusp energy and 
    isotropic J."
    """
    if use_sm78_physics:
        # SM78: Inject at x_b = 0.2 (outer boundary of cusp)
        x = X_BOUND  # = 0.2, matching paper
        
        # Isotropic angular momentum distribution
        j = math.sqrt(np.random.random())
        
        return x, j
    else:
        # OLD CODE: allow tunable injection
        # (keep for backward compatibility)
        pass

# THEN REPLACE all injection blocks (lines 702-712, etc.) with:
if cap or (x < X_BOUND):
    if use_sm78_physics:
        x, j = inject_star_sm78(use_sm78_physics=True, sm78_params=sm78_params)
    else:
        # OLD CODE: existing injection logic
        if noloss_flag:
            x = sample_x_from_g0_jit()
        # ... etc
```

---

## 4. Add CLI Flag to Enable SM78 Physics

**Add to argparse section (around line 1448):**

```python
ap.add_argument(
    "--use-sm78-physics",
    action="store_true",
    help=(
        "Use exact SM78 formulas for diffusion coefficients and loss-cone, "
        "instead of parametric approximations. Requires implementing "
        "diff_coeffs_sm78_exact() with formulas from the paper."
    ),
)
```

**Update function signatures to pass this flag through:**

- `_build_kernels()` - add `use_sm78_physics` parameter
- `bilinear_coeffs()` - add `use_sm78_physics` and `sm78_params` parameters
- `j_min_of_x()` - add `use_sm78_physics` and `sm78_params` parameters
- `pick_n()` - add `use_sm78_physics` and `sm78_params` parameters
- `step_one()` - add `use_sm78_physics` and `sm78_params` parameters
- `run_stream()` - add `use_sm78_physics` and `sm78_params` parameters

---

## 5. Production Commands (After Implementation)

Once you've implemented the exact SM78 formulas:

### Occupancy ḡ run:
```bash
python3 gbar_mc_fast_sm78_fig3_v2.py \
  --streams 800 --windows 10 --warmup 5 --replicates 5 --procs 48 \
  --pstar 0.005 --gexp 2.5 --clones 9 --floors_min_exp 8 --floor-step 1 \
  --snapshots --snaps-per-t0 80 --use-x-bound-injection \
  --gbar-x-norm 0.225 \
  --use-sm78-physics \
  > gbar_sm78_exact_occupancy.log 2>&1
```

### Flux ḡ run:
```bash
python3 gbar_mc_fast_sm78_fig3_v2.py \
  --streams 800 --windows 10 --warmup 5 --replicates 5 --procs 48 \
  --pstar 0.005 --gexp 2.5 --clones 9 --floors_min_exp 8 --floor-step 1 \
  --snapshots --snaps-per-t0 80 --use-x-bound-injection \
  --gbar-x-norm 0.225 \
  --use-sm78-physics \
  --gbar-from-flux --gbar-flux-exp 3.10 \
  > gbar_sm78_exact_flux.log 2>&1
```

**Note:** No more `--E2-x-power`, `--J2-scale`, `--lc_scale`, `--cone-gamma`, `--lc-floor-frac` - these become fixed constants from the paper.

---

## 6. What You Need from the Paper

To complete this implementation, you need to extract from SM78:

1. **Diffusion coefficient formulas:**
   - Exact expressions for ⟨ΔE⟩, ⟨(ΔE)²⟩, ⟨ΔJ⟩, ⟨(ΔJ)²⟩, ⟨ΔE ΔJ⟩
   - All numerical prefactors (Coulomb log, mass ratios, etc.)
   - Energy and angular momentum dependencies

2. **Loss-cone definition:**
   - Exact formula for J_lc(E) or j_lc(x)
   - Tidal radius or capture radius definition
   - Any interpolation between empty/full loss-cone regimes

3. **Outer boundary condition:**
   - Exact value of x_b (outer boundary of cusp)
   - Distribution function f(E) at outer boundary
   - Injection rule (fixed flux vs fixed f(E))

4. **Constants:**
   - All physical constants (masses, radii, Coulomb log, etc.)
   - Dimensionless scaling factors

---

## 7. Testing After Implementation

Once implemented, verify:

1. **Slope match:** `best-fit g(x) ∝ x^α` should match paper's slope (≈ +0.2 to +0.3)
2. **χ² improvement:** Should drop from ~50 to < 5
3. **Peak location:** ḡ peak should be near x ≈ 200-300 (matching paper)
4. **Consistency:** Occupancy and flux ḡ should still match each other

If these don't match, the issue is in the formulas you copied from the paper, not in the ḡ computation (which is now correct).

