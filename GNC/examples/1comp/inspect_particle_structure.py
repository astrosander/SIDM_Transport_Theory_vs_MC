#!/usr/bin/env python
"""Inspect the structure of particle sample files."""
import os
import sys
import pickle
import numpy as np

# Add source directory to path for unpickling
sys.path.insert(0, '../../source')
sys.path.insert(0, '../../main')

def inspect_file(filepath):
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        print(f"Size: {os.path.getsize(filepath)} bytes")
        
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"\nFirst element type: {type(data[0])}")
                first = data[0]
                if hasattr(first, '__dict__'):
                    attrs = list(vars(first).keys())
                    print(f"Attributes ({len(attrs)} total): {attrs[:15]}")
                    for attr in ['en', 'jm', 'm', 'w', 'exit_flag'][:5]:  # Key attributes
                        if hasattr(first, attr):
                            val = getattr(first, attr)
                            if isinstance(val, np.ndarray):
                                print(f"  {attr}: array shape={val.shape}")
                            else:
                                print(f"  {attr}: {type(val).__name__} = {val}")
                elif isinstance(first, dict):
                    print(f"Dict keys: {list(first.keys())}")
                    for key in list(first.keys())[:10]:
                        val = first[key]
                        print(f"  {key}: {type(val)} = {val}")
                        
        elif isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
            for key in list(data.keys())[:15]:
                val = data[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: array shape={val.shape}, dtype={val.dtype}")
                    if val.size > 0 and val.size < 10:
                        print(f"    values: {val}")
                    elif val.size > 0:
                        print(f"    first few: {val.flat[:5]}")
                else:
                    print(f"  {key}: {type(val)} = {val}")
                    
        elif hasattr(data, '__dict__'):
            print(f"Object attributes: {list(vars(data).keys())}")
            for attr in list(vars(data).keys())[:10]:
                val = getattr(data, attr)
                print(f"  {attr}: {type(val)}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Check several particle files
base_dir = "output/ecev/bin/single"

if os.path.exists(base_dir):
    pkl_files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} .pkl files in {base_dir}")
    print(f"\nFile names (first 10): {pkl_files[:10]}")
    
    # Inspect first file
    if pkl_files:
        inspect_file(os.path.join(base_dir, pkl_files[0]))
        
        # Inspect a different snapshot if available
        if len(pkl_files) > 10:
            inspect_file(os.path.join(base_dir, pkl_files[10]))
else:
    print(f"Directory not found: {base_dir}")

