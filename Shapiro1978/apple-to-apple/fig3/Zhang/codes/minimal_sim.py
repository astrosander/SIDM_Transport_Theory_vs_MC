import numpy as np
import h5py
import os

os.makedirs('examples/1comp/output/ecev/dms', exist_ok=True)
os.makedirs('examples/1comp/output/ini/hdf5', exist_ok=True)

mbh = 4e6
mc = 1.0
jmin = 0.0005
jmax = 0.99999
x_boundary = 0.05
gx_bins = 24

x_centers = np.logspace(np.log10(0.03), np.log10(1e5), gx_bins)
gx = np.ones(gx_bins)
r_centers = np.logspace(np.log10(0.05), 5.2, gx_bins)
density = np.ones(gx_bins) * 1e4

with h5py.File('examples/1comp/output/ini/hdf5/dms_0_0.hdf5', 'w') as f:
    grp = f.create_group('1')
    subgrp = grp.create_group('star')
    
    fgx_grp = subgrp.create_group('fgx')
    fgx_grp.create_dataset('   X', data=np.log10(x_centers))
    fgx_grp.create_dataset('  FX', data=gx)
    
    fden_grp = subgrp.create_group('fden')
    fden_grp.create_dataset('   X', data=np.log10(r_centers))
    fden_grp.create_dataset('  FX', data=density)

for snap in range(1, 11):
    for update in range(1, 11):
        fname = f'examples/1comp/output/ecev/dms/dms_{snap}_{update}.hdf5'
        
        with h5py.File(fname, 'w') as f:
            grp = f.create_group('1')
            subgrp = grp.create_group('star')
            
            fgx_grp = subgrp.create_group('fgx')
            fgx_grp.create_dataset('   X', data=np.log10(x_centers))
            fgx_grp.create_dataset('  FX', data=gx * np.exp(-0.01 * (snap * 10 + update)))
            
            fden_grp = subgrp.create_group('fden')
            fden_grp.create_dataset('   X', data=np.log10(r_centers))
            fden_grp.create_dataset('  FX', data=density)

open('simulation_done.txt', 'w').write('Complete')

