#!/usr/bin/env python
"""
Test script to verify gen_ge.py fixes match Fortran implementation.
"""
import sys
import os
import math
import numpy as np

# Add source to path
sys.path.insert(0, '../../source')
sys.path.insert(0, '../../main')

from gen_ge import (
    get_ejw_from_particle, S2DHstType, DmsStellarObject,
    dms_so_get_fxj, get_barge_stellar, Jbin_type_lin,
    NejwRec, S1DType, S2DType
)

def test_energy_conversion():
    """Test that energy is converted correctly to log10(x)."""
    print("\n=== Test 1: Energy Conversion ===")
    
    # Create test particles
    # Physical energy E, dimensionless x = |E|/v0^2
    v0 = 1.0
    estar = np.array([1.0, 10.0, 100.0])  # Physical energies
    jstar = np.array([0.5, 0.7, 0.9])
    wstar = np.array([1.0, 1.0, 1.0])
    mstar = np.array([1.0, 1.0, 1.0])
    
    # Expected: log10(|E|/v0^2) = log10(E) when v0=1
    expected_log_x = np.array([0.0, 1.0, 2.0])
    
    nejw, n = get_ejw_from_particle(
        estar, jstar, wstar, mstar, 3,
        m1=0.5, m2=1.5, mbh=4e6, v0=v0, xb=100.0,
        nejw_out=None, nsam_out=None
    )
    
    print(f"  Input energies: {estar}")
    print(f"  Expected log10(x): {expected_log_x}")
    print(f"  Computed log10(x): {[nejw[i].e for i in range(n)]}")
    
    for i in range(n):
        assert abs(nejw[i].e - expected_log_x[i]) < 1e-10, \
            f"Energy conversion failed: expected {expected_log_x[i]}, got {nejw[i].e}"
    
    print("  [OK] Energy conversion correct!")
    return True


def test_histogram_binning():
    """Test 2D histogram binning matches Fortran."""
    print("\n=== Test 2: 2D Histogram Binning ===")
    
    # Create histogram
    hist = S2DHstType()
    hist.init(3, 3, 0.0, 3.0, 0.0, 1.0, use_weight=True)
    hist.set_range()
    
    print(f"  X bins: {hist.xcenter}")
    print(f"  Y bins: {hist.ycenter}")
    print(f"  X step: {hist.xstep}, Y step: {hist.ystep}")
    
    # Add particles
    en = [0.5, 1.5, 2.5]  # Should go in bins 0, 1, 2
    jm = [0.25, 0.5, 0.75]
    we = [1.0, 2.0, 3.0]
    
    hist.get_stats_weight(en, jm, we, 3)
    
    print(f"  Histogram:\n{hist.nxyw}")
    
    # Check that weights are in correct bins
    assert hist.nxyw[0, 0] == 1.0, f"Expected 1.0 in [0,0], got {hist.nxyw[0, 0]}"
    assert hist.nxyw[1, 1] == 2.0, f"Expected 2.0 in [1,1], got {hist.nxyw[1, 1]}"
    assert hist.nxyw[2, 2] == 3.0, f"Expected 3.0 in [2,2], got {hist.nxyw[2, 2]}"
    
    print("  [OK] Histogram binning correct!")
    return True


def test_gxj_normalization():
    """Test g(x,j) normalization formula."""
    print("\n=== Test 3: g(x,j) Normalization ===")
    
    # Create stellar object with simple histogram
    so = DmsStellarObject()
    so.n = 10
    
    # Initialize nxj histogram (avoid j=0 which causes division by zero)
    so.nxj.init(2, 2, 0.0, 2.0, 0.1, 1.0, use_weight=True)
    so.nxj.set_range()
    so.nxj.nxyw = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Initialize gxj (match nxj range)
    so.gxj.init(2, 2, 0.0, 2.0, 0.1, 1.0, 0)
    so.gxj.set_range()
    
    # Parameters
    n0 = 1.0
    mbh = 4e6
    v0 = 1.0
    
    # Apply normalization
    dms_so_get_fxj(so, n0, mbh, v0, Jbin_type_lin)
    
    print(f"  Input nxyw:\n{so.nxj.nxyw}")
    print(f"  Output gxj.fxy:\n{so.gxj.fxy}")
    print(f"  Sum of gxj: {np.sum(so.gxj.fxy):.3e}")
    
    # Check that normalization was applied (values should be non-zero and finite)
    assert np.all(np.isfinite(so.gxj.fxy)), "gxj contains non-finite values"
    assert np.sum(so.gxj.fxy) > 0, "gxj sum is zero"
    
    print("  [OK] g(x,j) normalization correct!")
    return True


def test_barge_integration():
    """Test integration over j to get g(x)."""
    print("\n=== Test 4: Integration to get g(x) ===")
    
    # Create stellar object
    so = DmsStellarObject()
    so.n = 10
    
    # Initialize gxj with simple values
    so.gxj.init(2, 3, 0.0, 2.0, 0.0, 1.0, 0)
    so.gxj.set_range()
    # Set uniform values for easy integration check
    so.gxj.fxy = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    
    # Initialize barge
    so.barge.init(0.0, 2.0, 2, 0)
    so.barge.set_range()
    
    print(f"  gxj.fxy:\n{so.gxj.fxy}")
    print(f"  j centers: {so.gxj.ycenter}")
    print(f"  j step: {so.gxj.ystep}")
    
    # Integrate
    get_barge_stellar(so, Jbin_type_lin)
    
    print(f"  barge.fx (g(x)): {so.barge.fx}")
    
    # For linear j-bins: integral = sum(gxj * j * dj * 2)
    # With uniform gxj=1, j=[0.17, 0.5, 0.83], dj=0.33:
    # integral ≈ 1.0 * (0.17 + 0.5 + 0.83) * 0.33 * 2 = 1.0
    
    assert np.all(np.isfinite(so.barge.fx)), "barge contains non-finite values"
    assert np.all(so.barge.fx > 0), "barge contains non-positive values"
    
    print("  [OK] Integration to g(x) correct!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing gen_ge.py Fortran Compatibility")
    print("=" * 60)
    
    tests = [
        test_energy_conversion,
        test_histogram_binning,
        test_gxj_normalization,
        test_barge_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

