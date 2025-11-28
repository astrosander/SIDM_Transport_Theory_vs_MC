# SM78 Exact Physics Implementation Status

## Overview

This document describes the current implementation of the SM78 exact physics mode (`--use-sm78-physics`), focusing on the diffusion coefficients and their scaling behavior.

## Current Implementation

### Diffusion Coefficients Structure

The SM78 exact physics mode is implemented in `diff_coeffs_sm78_exact_star()` (lines 356-389), which returns dimensionless "starred" coefficients in the same units as the original Table 1 interpolation.

**Function signature:**
```python
@njit(**fastmath)
def diff_coeffs_sm78_exact_star(x, j):
    """
    Returns: (e1_star, E2_star, j1_star, J2_star, covEJ_star)
    All coefficients are for pstar = PSTAR_CANON (0.005), BEFORE any pstar scaling.
    """
```

### Example: E₂ (Energy Diffusion Coefficient)

**Current implementation (lines 382-383):**
```python
E2_star = A_E2 * x**ALPHA_E2 * (1.0 + BETA_E2 * j*j)
E2_star = max(E2_star, 0.0)  # Enforce non-negativity
```

**Current parameter values (lines 37-39):**
```python
SM78_A_E2 = 1.0      # Placeholder - needs actual SM78 value
SM78_ALPHA_E2 = 1.0  # Placeholder - needs actual SM78 value  
SM78_BETA_E2 = 0.0   # Placeholder - needs actual SM78 value
```

**Functional form:**
\[
E_2^*(x, j) = A_{E2} \cdot x^{\alpha_{E2}} \cdot (1 + \beta_{E2} \cdot j^2)
\]

**Scaling behavior:**
- **Energy dependence (x)**: Currently `E2_star ∝ x^1.0` (linear in x)
- **Angular momentum dependence (j)**: Currently `E2_star ∝ (1 + 0.0·j²) = constant` (no j dependence)

### All Coefficients

All five diffusion coefficients follow the same parametric form:

1. **e₁* (Energy drift)**: `A_e1 * x^ALPHA_e1 * (1 + BETA_e1 * j²)`
2. **E₂* (Energy diffusion)**: `A_E2 * x^ALPHA_E2 * (1 + BETA_E2 * j²)`
3. **j₁* (Angular momentum drift)**: `A_j1 * x^ALPHA_j1 * j * (1 + BETA_j1 * j²)`
4. **J₂* (Angular momentum diffusion)**: `A_J2 * x^ALPHA_J2 * j² * (1 + BETA_J2 * j²)`
5. **covEJ* (E-J covariance)**: `A_covEJ * x^ALPHA_covEJ * j * (1 + BETA_covEJ * j²)`

**Current placeholder values:**
- All `A_*` = 1.0
- All `ALPHA_*` = 1.0  
- All `BETA_*` = 0.0
- `A_covEJ` = 0.0 (no covariance)

### Unit Conversion

The starred coefficients are converted to physical units in `bilinear_coeffs()` (lines 470-500):

```python
# After pstar scaling (lines 456-461):
scale = pstar_val / PSTAR_CANON
E2_star *= scale
# ... (all coefficients scaled)

# Physical unit conversion (lines 470-500):
v0_sq = 1.0
Jmax = 1.0 / math.sqrt(2.0 * x_clamp)

sigE = math.sqrt(max(E2_star, 0.0)) * v0_sq
```

So the final energy diffusion step size is:
\[
\sigma_E = \sqrt{E_2^*} \cdot v_0^2
\]

## What Needs to Be Done

### 1. Extract Exact SM78 Formulas

The current implementation uses placeholder power-law forms. The actual SM78 paper provides:
- Exact orbit-averaged expressions for each coefficient
- Dependencies on Coulomb log (lnΛ), stellar mass, BH mass, etc.
- Specific x and j scaling laws

**Action required:** Replace the placeholder formulas in `diff_coeffs_sm78_exact_star()` with the exact expressions from the SM78 paper.

### 2. Sanity Check: E₂ Scaling

To verify the implementation, compare the **expected** E₂(x, j) scaling from SM78 theory against what the MC simulation produces:

**Expected behavior (from SM78 theory):**
- At fixed j, how does E₂ scale with x?
- At fixed x, how does E₂ scale with j?
- What is the overall normalization (A_E2)?

**Observed behavior (from MC runs):**
- Check the ḡ(x) slope in noloss runs
- Compare capture rates vs. energy
- Analyze the steady-state distribution shape

**Diagnostic approach:**
1. Run noloss test with `--use-sm78-physics`
2. Measure N(E) slope → should match Bahcall-Wolf (N ∝ x^-2.25)
3. If slope is wrong, the E₂(x) scaling is likely incorrect
4. Compare to parametric runs with `--E2-x-power` to see what exponent would fix it

### 3. Current Issues

From recent runs with `--use-sm78-physics`:
- **Capture rate too high**: ~70× more captures than expected
  - Likely fixed by unified pstar scaling (already implemented)
  - May also indicate wrong normalization in diffusion coefficients
- **ḡ(x) slope wrong**: Negative slope instead of positive
  - Suggests E₂(x) scaling doesn't match SM78
  - Current placeholder: E₂ ∝ x^1.0, but may need different exponent

## Next Steps

1. **Fill in actual SM78 constants** from the paper
2. **Run diagnostic noloss test** with exact formulas
3. **Compare N(E) slope** to expected Bahcall-Wolf (x^-2.25)
4. **Adjust formulas** if needed based on MC results
5. **Re-run full BH case** and compare to Fig. 3

## Code Locations

- **SM78 constants**: Lines 28-51
- **Exact diffusion function**: Lines 356-389 (`diff_coeffs_sm78_exact_star`)
- **Coefficient usage**: Lines 450-500 (`bilinear_coeffs`)
- **CLI flag**: Line ~1863 (`--use-sm78-physics`)

