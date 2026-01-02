import sys
import os

sys.stdout = open('sim_log.txt', 'w', buffering=1)
sys.stderr = sys.stdout

print("="*60)
print("Starting Galactic Nuclear Cluster Simulation")
print("="*60)

try:
    print("\n1. Importing libraries...")
    import numpy as np
    print("   - numpy imported")
    import h5py
    print("   - h5py imported")
    from scipy import integrate, interpolate
    print("   - scipy imported")
    
    print("\n2. Reading configuration files...")
    base_dir = 'examples/1comp'
    model_file = f'{base_dir}/model.in'
    mfrac_file = f'{base_dir}/mfrac.in'
    
    print(f"   - Model file: {model_file}")
    print(f"   - Mass fraction file: {mfrac_file}")
    
    print("\n3. Initializing simulation...")
    from simulation import GalacticNuclearCluster
    gnc = GalacticNuclearCluster(model_file=model_file, mfrac_file=mfrac_file)
    
    print(f"\n4. Configuration:")
    print(f"   - MBH = {gnc.mbh:.2e} Msun")
    print(f"   - Number of snapshots = {gnc.num_snapshots}")
    print(f"   - Updates per snapshot = {gnc.num_update_per_snap}")
    print(f"   - gx_bins = {gnc.gx_bins}")
    print(f"   - Relaxation time = {gnc.trlx:.2e}")
    
    print("\n5. Running simulation...")
    output_dir = f'{base_dir}/output'
    gnc.run_simulation(output_dir=output_dir)
    
    print("\n" + "="*60)
    print("Simulation completed successfully!")
    print("="*60)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    sys.stdout.close()

