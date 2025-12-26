#!/usr/bin/env python
"""
run_simulation.py - Run the 1-component GNC simulation.

This script runs the simulation from the 1comp example directory.

Usage:
    python run_simulation.py              # Run full simulation
    python run_simulation.py --cfuns      # Generate cfuns first (if needed)
    python run_simulation.py --help       # Show help

Requirements:
    - Python 3.8+
    - numpy
    - h5py (for HDF5 output)
    - mpi4py (optional, for parallel execution)
    - matplotlib (for plotting results)
"""
from __future__ import annotations

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.join(script_dir, '..', '..', 'main')
source_dir = os.path.join(script_dir, '..', '..', 'source')

sys.path.insert(0, main_dir)
sys.path.insert(0, source_dir)

# Change to this directory
os.chdir(script_dir)

# Import and run
from run import main

if __name__ == "__main__":
    main()

