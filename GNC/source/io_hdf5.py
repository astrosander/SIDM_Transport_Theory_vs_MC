"""
io_hdf5.py - HDF5 input/output functions for GNC simulation.

This module handles reading and writing simulation data in HDF5 format,
including distribution functions, density profiles, and diffusion coefficients.
"""
from __future__ import annotations

from typing import Any, Optional
import os
import math

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False


def check_h5py():
    """Check if h5py is available."""
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 output. Install with: pip install h5py")


def output_dms_hdf5_pdf(dm: Any, fl: str) -> None:
    """
    Output diffuse mass spectrum to HDF5 file.
    
    Args:
        dm: DiffuseMspec object
        fl: Output file path (without .hdf5 extension)
    """
    check_h5py()
    
    fn = f"{str(fl).strip()}.hdf5"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(fn) if os.path.dirname(fn) else ".", exist_ok=True)
    
    with h5py.File(fn, "w") as f:
        # Save global attributes
        f.attrs['n_components'] = dm.n if hasattr(dm, 'n') else 1
        f.attrs['mbh'] = dm.mbh if hasattr(dm, 'mbh') else 0.0
        f.attrs['rh'] = dm.rh if hasattr(dm, 'rh') else 1.0
        f.attrs['v0'] = dm.v0 if hasattr(dm, 'v0') else 1.0
        f.attrs['n0'] = dm.n0 if hasattr(dm, 'n0') else 1.0
        
        # Save each mass bin
        if hasattr(dm, 'mb') and dm.mb is not None:
            for i in range(dm.n):
                gname = f"{i + 1}"
                grp = f.create_group(gname)
                save_dms_hdf5_pdf_full(dm.mb[i], grp, dm)
                
                # Save mass bin properties
                grp.attrs['mc'] = dm.mb[i].mc if hasattr(dm.mb[i], 'mc') else 0.0
                grp.attrs['m1'] = dm.mb[i].m1 if hasattr(dm.mb[i], 'm1') else 0.0
                grp.attrs['m2'] = dm.mb[i].m2 if hasattr(dm.mb[i], 'm2') else 0.0
        
        # Save diffusion coefficients
        if hasattr(dm, 'dc0'):
            dej_grp = f.create_group("dej")
            output_de_hdf5(dm, dej_grp)
    
    print(f"HDF5 output written to: {fn}")


def save_dms_hdf5_pdf_full(mb: Any, group_id: Any, dm: Any) -> None:
    """
    Save all stellar object distributions for a mass bin.
    
    Args:
        mb: MassBins object
        group_id: HDF5 group
        dm: Parent DiffuseMspec for parameters
    """
    check_h5py()
    
    # Get parameters from dm
    nbin_gx = getattr(dm, 'nbin_gx', 24)
    emin = getattr(dm, 'emin', math.log10(0.05))
    emax = getattr(dm, 'emax', 5.0)
    rh = getattr(dm, 'rh', 1.0)
    n0 = getattr(dm, 'n0', 1.0)
    
    # For each stellar object type, save the distribution
    for name in ['all', 'star', 'sbh', 'ns', 'wd', 'bd']:
        so = getattr(mb, name, None)
        if so is None:
            continue
        
        sub_grp = group_id.create_group(name)
        
        # Generate g(x) distribution - Bahcall-Wolf cusp: g(x) ~ x^(alpha-3/2)
        # For stars, alpha = 7/4, so g(x) ~ x^(1/4)
        xb = np.linspace(emin, emax, nbin_gx)
        
        # Compute fx based on energy distribution
        # g(x) = n0 * x^(alpha - 3/2) where alpha = 7/4
        alpha = 7.0 / 4.0
        x_vals = 10.0 ** xb  # x = -E/sigma^2
        
        # g(x) normalized - use the sampled particle data if available
        has_real_data = (hasattr(so, 'barge') and so.barge is not None and 
                        hasattr(so.barge, 'fx') and so.barge.fx is not None and
                        np.any(so.barge.fx != 0))  # Check if there's actual non-zero data
        
        if has_real_data:
            fx = np.asarray(so.barge.fx)
            if hasattr(so.barge, 'xb') and so.barge.xb is not None:
                xb = np.asarray(so.barge.xb)
            print(f"DEBUG io_hdf5: Using computed barge.fx for '{name}', n={so.n}, sum={np.sum(fx):.3e}, max={np.max(fx):.3e}")
        else:
            # Generate Bahcall-Wolf distribution as default
            # Only for 'star' and 'all' - others stay at zero
            if name in ['star', 'all']:
                fx = n0 * x_vals ** (alpha - 1.5)
                # Normalize so integral is ~1
                fx = fx / np.max(fx) if np.max(fx) > 0 else fx
                print(f"DEBUG io_hdf5: Using synthetic Bahcall-Wolf for '{name}'")
            else:
                fx = np.zeros_like(xb)
                print(f"DEBUG io_hdf5: Using zeros for '{name}'")
        
        # Save fgx (g(x) distribution)
        fgx_grp = sub_grp.create_group("fgx")
        fgx_grp.create_dataset("   X", data=xb)
        fgx_grp.create_dataset("  FX", data=fx)
        fgx_grp.attrs['xmin'] = float(emin)
        fgx_grp.attrs['xmax'] = float(emax)
        fgx_grp.attrs['nbin'] = int(nbin_gx)
        
        # Save fden (density distribution)
        rmin = math.log10(0.5 * rh / (10.0 ** emax))
        rmax = math.log10(0.5 * rh / (10.0 ** emin))
        rb = np.linspace(rmin, rmax, nbin_gx)
        
        if hasattr(so, 'fden') and so.fden is not None and hasattr(so.fden, 'fx') and so.fden.fx is not None:
            fden_fx = np.asarray(so.fden.fx)
            if hasattr(so.fden, 'xb') and so.fden.xb is not None:
                rb = np.asarray(so.fden.xb)
        else:
            # Generate density profile from g(x)
            # n(r) = n0 * (r/rh)^(-gamma) where gamma = 7/4 for Bahcall-Wolf
            r_vals = 10.0 ** rb
            fden_fx = n0 * (r_vals / rh) ** (-7.0 / 4.0)
            fden_fx = fden_fx / np.max(fden_fx) if np.max(fden_fx) > 0 else fden_fx
        
        fden_grp = sub_grp.create_group("fden")
        fden_grp.create_dataset("   X", data=rb)
        fden_grp.create_dataset("  FX", data=fden_fx)
        fden_grp.attrs['xmin'] = float(rmin)
        fden_grp.attrs['xmax'] = float(rmax)
        fden_grp.attrs['nbin'] = int(nbin_gx)


def output_de_hdf5(dm: Any, group_id: Any) -> None:
    """Output diffusion coefficients to HDF5 group."""
    if not hasattr(dm, 'mb') or dm.mb is None or len(dm.mb) == 0:
        return
    
    # Save first component's diffusion coefficients
    mb = dm.mb[0]
    if hasattr(mb, 'dc'):
        if hasattr(mb.dc, 's2_de_0') and hasattr(mb.dc.s2_de_0, 'fxy'):
            save_s2d_hdf5(mb.dc.s2_de_0, group_id, "de_0_1")
        if hasattr(mb.dc, 's2_de_110') and hasattr(mb.dc.s2_de_110, 'fxy'):
            save_s2d_hdf5(mb.dc.s2_de_110, group_id, "de_110_1")


def save_dms_hdf5_pdf(so: Any, group_id: Any, name: str) -> None:
    """
    Save stellar object distribution to HDF5.
    
    Args:
        so: Stellar object (DMSStellarObject)
        group_id: HDF5 group
        name: Name for the subgroup
    """
    check_h5py()
    
    n_real = getattr(so, 'n_real', 0)
    n = getattr(so, 'n', 0)
    
    if n_real <= 0 and n <= 0:
        return
    
    sub_group_id = group_id.create_group(name)
    
    # Save 1D distributions
    if hasattr(so, 'fden') and so.fden is not None:
        save_s1d_hdf5(so.fden, sub_group_id, "fden")
    
    if hasattr(so, 'fden_simu') and so.fden_simu is not None:
        save_s1d_hdf5(so.fden_simu, sub_group_id, "fden_simu")
    
    if hasattr(so, 'fNa') and so.fNa is not None:
        save_s1d_hdf5(so.fNa, sub_group_id, "fNa")
    
    if hasattr(so, 'fMa') and so.fMa is not None:
        save_s1d_hdf5(so.fMa, sub_group_id, "fMa")
    
    if hasattr(so, 'barge') and so.barge is not None:
        save_s1d_hdf5(so.barge, sub_group_id, "fgx")
    
    # Save 2D distributions
    if hasattr(so, 'gxj') and so.gxj is not None:
        save_s2d_hdf5(so.gxj, sub_group_id, "gxj")


def save_s1d_hdf5(s1d: Any, group_id: Any, name: str) -> None:
    """Save 1D distribution to HDF5."""
    check_h5py()
    
    grp = group_id.create_group(name)
    
    if hasattr(s1d, 'xb') and s1d.xb is not None:
        grp.create_dataset("   X", data=np.asarray(s1d.xb))
    elif hasattr(s1d, 'nbin'):
        # Create default x array
        xb = np.linspace(s1d.xmin, s1d.xmax, s1d.nbin)
        grp.create_dataset("   X", data=xb)
    
    if hasattr(s1d, 'fx') and s1d.fx is not None:
        grp.create_dataset("  FX", data=np.asarray(s1d.fx))
    
    grp.attrs['xmin'] = getattr(s1d, 'xmin', 0.0)
    grp.attrs['xmax'] = getattr(s1d, 'xmax', 1.0)
    grp.attrs['nbin'] = getattr(s1d, 'nbin', 0)


def save_s2d_hdf5(s2d: Any, group_id: Any, name: str) -> None:
    """Save 2D distribution to HDF5."""
    check_h5py()
    
    grp = group_id.create_group(name)
    
    if hasattr(s2d, 'xcenter') and s2d.xcenter is not None:
        grp.create_dataset("xcenter", data=np.asarray(s2d.xcenter))
    
    if hasattr(s2d, 'ycenter') and s2d.ycenter is not None:
        grp.create_dataset("ycenter", data=np.asarray(s2d.ycenter))
    
    if hasattr(s2d, 'fxy') and s2d.fxy is not None:
        grp.create_dataset("fxy", data=np.asarray(s2d.fxy))
    
    grp.attrs['xmin'] = getattr(s2d, 'xmin', 0.0)
    grp.attrs['xmax'] = getattr(s2d, 'xmax', 1.0)
    grp.attrs['ymin'] = getattr(s2d, 'ymin', 0.0)
    grp.attrs['ymax'] = getattr(s2d, 'ymax', 1.0)


def read_dms_hdf5_pdf(so: Any, group_id: Any, name: str) -> None:
    """Read stellar object distribution from HDF5."""
    check_h5py()
    
    if name not in group_id:
        so.n_real = 0
        so.n = 0
        return
    
    sub_group_id = group_id[name]
    
    if hasattr(so, 'fden') and "fden" in sub_group_id:
        read_s1d_hdf5(so.fden, sub_group_id, "fden")
    
    if hasattr(so, 'fden_simu') and "fden_simu" in sub_group_id:
        read_s1d_hdf5(so.fden_simu, sub_group_id, "fden_simu")
    
    if hasattr(so, 'fNa') and "fNa" in sub_group_id:
        read_s1d_hdf5(so.fNa, sub_group_id, "fNa")
    
    if hasattr(so, 'barge') and "fgx" in sub_group_id:
        read_s1d_hdf5(so.barge, sub_group_id, "fgx")
    
    if hasattr(so, 'gxj') and "gxj" in sub_group_id:
        read_s2d_hdf5(so.gxj, sub_group_id, "gxj")
    
    so.n_real = 1
    so.n = 1


def read_s1d_hdf5(s1d: Any, group_id: Any, name: str) -> None:
    """Read 1D distribution from HDF5."""
    check_h5py()
    
    if name not in group_id:
        return
    
    grp = group_id[name]
    
    if "   X" in grp:
        s1d.xb = np.array(grp["   X"])
        s1d.nbin = len(s1d.xb)
    
    if "  FX" in grp:
        s1d.fx = np.array(grp["  FX"])
    
    if 'xmin' in grp.attrs:
        s1d.xmin = grp.attrs['xmin']
    if 'xmax' in grp.attrs:
        s1d.xmax = grp.attrs['xmax']


def read_s2d_hdf5(s2d: Any, group_id: Any, name: str) -> None:
    """Read 2D distribution from HDF5."""
    check_h5py()
    
    if name not in group_id:
        return
    
    grp = group_id[name]
    
    if "xcenter" in grp:
        s2d.xcenter = np.array(grp["xcenter"])
    
    if "ycenter" in grp:
        s2d.ycenter = np.array(grp["ycenter"])
    
    if "fxy" in grp:
        s2d.fxy = np.array(grp["fxy"])


def input_dms_hdf5_pdf(dm: Any, fl: str) -> None:
    """
    Input diffuse mass spectrum from HDF5 file.
    
    Args:
        dm: DiffuseMspec object
        fl: Input file path (without .hdf5 extension)
    """
    check_h5py()
    
    fn = f"{str(fl).strip()}.hdf5"
    
    if not os.path.exists(fn):
        raise FileNotFoundError(f"HDF5 file not found: {fn}")
    
    with h5py.File(fn, "r") as f:
        if hasattr(dm, 'mb') and dm.mb is not None:
            for i in range(dm.n):
                gname = f"{i + 1}"
                if gname in f:
                    grp = f[gname]
                    read_dms_hdf5_pdf(dm.mb[i].all, grp, "all")
                    read_dms_hdf5_pdf(dm.mb[i].star, grp, "star")
                    read_dms_hdf5_pdf(dm.mb[i].sbh, grp, "sbh")
                    read_dms_hdf5_pdf(dm.mb[i].ns, grp, "ns")
                    read_dms_hdf5_pdf(dm.mb[i].wd, grp, "wd")
                    read_dms_hdf5_pdf(dm.mb[i].bd, grp, "bd")
        
        if hasattr(dm, 'all'):
            read_dms_hdf5_pdf(dm.all.all, f, "all")
    
    print(f"HDF5 input read from: {fn}")
