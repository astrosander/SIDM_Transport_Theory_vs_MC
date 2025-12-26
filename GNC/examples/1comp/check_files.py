#!/usr/bin/env python
"""Check what HDF5 files exist and their structure."""
import os
import h5py

output_dir = "output/ecev/dms"

if not os.path.exists(output_dir):
    print(f"Directory not found: {output_dir}")
else:
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.hdf5')])
    print(f"Found {len(files)} HDF5 files in {output_dir}:")
    for f in files[:20]:  # Show first 20
        print(f"  {f}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")
    
    # Check structure of first file
    if files:
        first_file = os.path.join(output_dir, files[0])
        print(f"\nStructure of {files[0]}:")
        try:
            with h5py.File(first_file, 'r') as f:
                print(f"  Top-level keys: {list(f.keys())}")
                f.visititems(lambda n, o: print(f"    {n}: {type(o).__name__}"))
        except Exception as e:
            print(f"  Error reading file: {e}")

