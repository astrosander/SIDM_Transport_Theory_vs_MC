import os
import sys

print("Verification Script for GNC Simulation")
print("="*60)

print("\n1. Checking required files...")
files_to_check = [
    'examples/1comp/model.in',
    'examples/1comp/mfrac.in',
    'gnc_sim.py'
]

for f in files_to_check:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} - MISSING!")
        sys.exit(1)

print("\n2. Checking Python packages...")
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except ImportError:
    print("  ✗ numpy - NOT INSTALLED")
    sys.exit(1)

try:
    import h5py
    print(f"  ✓ h5py {h5py.__version__}")
except ImportError:
    print("  ✗ h5py - NOT INSTALLED")
    sys.exit(1)

try:
    import scipy
    print(f"  ✓ scipy {scipy.__version__}")
except ImportError:
    print("  ✗ scipy - NOT INSTALLED")
    sys.exit(1)

print("\n3. Testing simulation import...")
try:
    with open('gnc_sim.py', 'r') as f:
        code = f.read()
    compile(code, 'gnc_sim.py', 'exec')
    print("  ✓ gnc_sim.py compiles successfully")
except SyntaxError as e:
    print(f"  ✗ Syntax error in gnc_sim.py: {e}")
    sys.exit(1)

print("\n4. Creating test output directories...")
os.makedirs('examples/1comp/output/ecev/dms', exist_ok=True)
os.makedirs('examples/1comp/output/ini/hdf5', exist_ok=True)
print("  ✓ Output directories created")

print("\n5. Testing HDF5 file creation...")
try:
    test_file = 'test_output.hdf5'
    with h5py.File(test_file, 'w') as f:
        grp = f.create_group('test')
        grp.create_dataset('data', data=[1, 2, 3])
    os.remove(test_file)
    print("  ✓ HDF5 file creation works")
except Exception as e:
    print(f"  ✗ HDF5 error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All checks passed! Ready to run simulation.")
print("="*60)
print("\nTo run the simulation:")
print("  python gnc_sim.py")
print("\nTo plot results:")
print("  cd examples/1comp/plot")
print("  python ge.py")

