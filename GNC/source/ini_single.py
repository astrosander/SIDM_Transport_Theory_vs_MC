"""
ini_single.py - Initialize particle samples for single-component simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from com_main_gw import (
    PI, ctl, MBH, mbh, mbh_radius, rh, rhmax, jmin_value, emax_factor,
    log10emin_factor, AU_SI, my_unit_vel_c, rd_sun,
    star_type_MS, star_type_BH, star_type_NS, star_type_WD, star_type_BD,
    flag_ini_or, flag_ini_ini, exit_normal, state_ae_evl, rid,
    ParticleSampleType, ParticleSamplesArrType, ChainType, ChainPointer,
    BinaryOrbit, Particle
)
import com_main_gw as cmg


def rnd(a: float, b: float) -> float:
    """Random number between a and b."""
    return float(a + (b - a) * np.random.random())


def fpowerlaw(alpha: float, xmin: float, xmax: float) -> float:
    """Generate random number from power-law distribution."""
    if xmin <= 0.0 or xmax <= xmin:
        return xmin
    u = np.random.random()
    a = float(alpha)
    if abs(a + 1.0) < 1e-14:
        return xmin * (xmax / xmin) ** u
    p = a + 1.0
    return (u * (xmax ** p - xmin ** p) + xmin ** p) ** (1.0 / p)


def get_mass_idx(m: float) -> int:
    """Get mass bin index for a given mass."""
    for i in range(ctl.m_bins):
        if m >= ctl.bin_mass_m1[i] and m <= ctl.bin_mass_m2[i]:
            return i + 1
    return -1


def create_init_clone_particle(ptbk, en0: float, time: float) -> None:
    """Create initial clone particle (placeholder)."""
    pass


def by_em2st(by: BinaryOrbit) -> None:
    """Convert eccentricity/mean anomaly to state vectors (placeholder)."""
    pass


def by_split_from_rd(by: BinaryOrbit) -> None:
    """Split binary from reduced mass frame (placeholder)."""
    pass


def track_init(bkps: ParticleSampleType, n: int) -> None:
    """Initialize particle track."""
    bkps.track = []


def set_particle_sample_other(bkps: ParticleSampleType) -> None:
    """Set additional particle sample properties."""
    bkps.byot.ms.m = bkps.m
    bkps.byot.mm.m = cmg.mbh
    bkps.byot.mtot = bkps.m + cmg.mbh
    bkps.byot.Inc = rnd(0.0, PI)
    bkps.byot.Om = rnd(0.0, 2.0 * PI)
    bkps.byot.pe = rnd(0.0, 2.0 * PI)
    bkps.byot.bname = "byot"
    bkps.byot_bf.bname = "byot_bf"
    bkps.byot.ms.id = bkps.id
    bkps.byot.mm.obtype = star_type_BH
    bkps.byot.mm.radius = cmg.mbh / 1e8
    bkps.byot.me = rnd(0.0, 2.0 * PI)
    by_em2st(bkps.byot)
    by_split_from_rd(bkps.byot)


def init_particle_sample_common(bkps: ParticleSampleType) -> None:
    """Initialize common particle sample properties."""
    bkps.en = -cmg.MBH / (2.0 * bkps.byot.a_bin)
    bkps.En = bkps.en
    bkps.en0 = bkps.en
    bkps.jm = math.sqrt(max(0, 1.0 - bkps.byot.e_bin ** 2))
    bkps.Jm = bkps.jm
    bkps.jm0 = bkps.jm
    bkps.djp = 0.0
    bkps.elp = 0.0
    bkps.id = int(rnd(0.0, 100000000.0))
    bkps.byot.ms.id = bkps.id
    bkps.state_flag_last = state_ae_evl
    bkps.exit_flag = exit_normal
    bkps.rid = rid
    set_particle_sample_other(bkps)
    bkps.byot_ini = bkps.byot
    track_init(bkps, 0)


def set_ini_byot_abin() -> float:
    """Set initial semi-major axis from power-law distribution."""
    xb = ctl.x_boundary if ctl.x_boundary > 0 else 0.05
    amax = cmg.rhmax if cmg.rhmax > 0 else cmg.rh * cmg.emax_factor * 2
    amin = 0.01 * cmg.rh / (xb * 2.0)
    return float(fpowerlaw(ctl.alpha_ini, amin, amax))


def star_Radius(mass: float) -> float:
    """Calculate stellar radius from mass."""
    if mass <= 0.06:
        return 0.1 * rd_sun
    if 0.06 < mass <= 1.0:
        return (mass ** 0.8) * rd_sun
    return (mass ** 0.56) * rd_sun


def white_dwarf_radius(mass: float) -> float:
    """Calculate white dwarf radius from mass."""
    if mass < 1.44:
        return 0.01 * rd_sun * (mass ** (-1.0 / 3.0))
    raise RuntimeError("error, white dwarf mass should be smaller than 1.44 solar mass")


def set_star_radius(pr: Particle) -> None:
    """Set stellar radius based on type."""
    if pr.obtype == star_type_MS:
        pr.radius = float(star_Radius(pr.m))
    elif pr.obtype == star_type_BD:
        pr.radius = pr.m / (my_unit_vel_c ** 2)
    elif pr.obtype == star_type_BH:
        pr.radius = pr.m / (my_unit_vel_c ** 2)
    elif pr.obtype == star_type_WD:
        pr.radius = float(white_dwarf_radius(pr.m))
    elif pr.obtype == star_type_NS:
        pr.radius = 1.0e4 / AU_SI
    else:
        pr.radius = 1e-10
        print(f"Warning: Unknown star type={pr.obtype}")


def get_sample_r_td_single(sp: ParticleSampleType) -> None:
    """Calculate tidal radius for single particle."""
    if sp.obtype == star_type_MS:
        if sp.byot.ms.radius <= 0.0:
            sp.byot.ms.radius = star_Radius(sp.m)
        sp.r_td = (3.0 * cmg.mbh / sp.m) ** (1.0 / 3.0) * sp.byot.ms.radius
    elif sp.obtype in (star_type_BH, star_type_NS, star_type_WD, star_type_BD):
        sp.r_td = 16.0 * cmg.mbh_radius / (1.0 + sp.byot.e_bin)
    else:
        sp.r_td = 16.0 * cmg.mbh_radius


def get_sample_r_td(bkps: ParticleSampleType) -> None:
    """Get tidal radius for particle sample."""
    get_sample_r_td_single(bkps)


def set_jm_init(bkps: ParticleSampleType) -> None:
    """Set initial angular momentum from eccentricity."""
    if bkps.byot.e_bin < 0.0:
        bkps.byot.e_bin = 0.0
    if bkps.byot.e_bin >= 1.0:
        bkps.byot.e_bin = 1.0 - 1e-12
    bkps.jm = math.sqrt(max(0, 1.0 - bkps.byot.e_bin ** 2))


def init_particle_sample_one_model_rnd(bkps: ParticleSampleType, flag: int) -> None:
    """Initialize particle sample with random orbital parameters."""
    if bkps.obtype == 0 or bkps.obidx == 0:
        raise RuntimeError(f"error: particle type not assigned {bkps.obtype} {bkps.obidx}")
    if bkps.m <= 0.0:
        raise RuntimeError(f"error: particle mass should be assigned {bkps.m}")

    if flag == flag_ini_or:
        if bkps.byot.a_bin == 0.0:
            raise RuntimeError("error! flag_ini_or should assume abin first")
        bkps.byot.ms.m = bkps.m
        bkps.byot.ms.obtype = bkps.obtype
        bkps.byot.ms.obidx = bkps.obidx
        set_star_radius(bkps.byot.ms)
        get_sample_r_td(bkps)
        bkps.rp = bkps.byot.a_bin * (1.0 - bkps.byot.e_bin)
        init_particle_sample_common(bkps)
        if bkps.weight_real == 0.0:
            bkps.weight_real = 1.0

    elif flag == flag_ini_ini:
        bkps.byot.a_bin = set_ini_byot_abin()
        # Initialize with thermal eccentricity distribution
        bkps.byot.e_bin = np.random.uniform(0.0, 0.99)
        bkps.byot.ms.m = bkps.m
        bkps.byot.ms.obtype = bkps.obtype
        bkps.byot.ms.obidx = bkps.obidx
        set_star_radius(bkps.byot.ms)
        get_sample_r_td(bkps)
        set_jm_init(bkps)
        bkps.rp = bkps.byot.a_bin * (1.0 - bkps.byot.e_bin)
        init_particle_sample_common(bkps)
        midx = get_mass_idx(bkps.m)
        if midx > 0 and midx <= len(ctl.weight_n):
            bkps.weight_n = float(ctl.weight_n[midx - 1])
            bkps.weight_N = bkps.weight_n
        else:
            bkps.weight_n = 1.0
            bkps.weight_N = 1.0
        bkps.weight_real = 1.0
    else:
        raise RuntimeError("undefined flag in init_particle_sample_one_model_rnd")


def init_particle_sample_one(sample: ParticleSampleType, m: float, flag: int) -> None:
    """Initialize a single particle sample."""
    sample.m = float(m)
    init_particle_sample_one_model_rnd(sample, flag)


def get_numbers_each_bin(n: int):
    """Get number of particles in each mass bin by stellar type."""
    nstar = np.zeros(n, dtype=int)
    nsbh = np.zeros(n, dtype=int)
    nns = np.zeros(n, dtype=int)
    nwd = np.zeros(n, dtype=int)
    nbd = np.zeros(n, dtype=int)

    for i in range(ctl.m_bins):
        nstar[i] = int(ctl.asymptot[1, i] * ctl.bin_mass_particle_number[i])
        nsbh[i] = int(ctl.asymptot[2, i] * ctl.bin_mass_particle_number[i]) if ctl.idxsbh != -1 else 0
        nns[i] = int(ctl.asymptot[3, i] * ctl.bin_mass_particle_number[i]) if ctl.idxns != -1 else 0
        nwd[i] = int(ctl.asymptot[4, i] * ctl.bin_mass_particle_number[i]) if ctl.idxwd != -1 else 0
        nbd[i] = int(ctl.asymptot[5, i] * ctl.bin_mass_particle_number[i]) if ctl.idxbd != -1 else 0

    nstar_tot = int(nstar.sum())
    nsbh_tot = int(nsbh.sum()) if ctl.idxsbh != -1 else 0
    nwd_tot = int(nwd.sum()) if ctl.idxwd != -1 else 0
    nns_tot = int(nns.sum()) if ctl.idxns != -1 else 0
    nbd_tot = int(nbd.sum()) if ctl.idxbd != -1 else 0

    return nstar_tot, nsbh_tot, nns_tot, nwd_tot, nbd_tot, nstar, nsbh, nns, nwd, nbd


def get_init_samples_given(bksps_arr_ini: ParticleSamplesArrType) -> None:
    """Generate initial samples with given mass distribution."""
    (nstar_tot, nsbh_tot, nns_tot, nwd_tot, nbd_tot,
     nstar, nsbh, nns, nwd, nbd) = get_numbers_each_bin(ctl.m_bins)

    total = nstar_tot + nsbh_tot + nwd_tot + nns_tot + nbd_tot
    
    if total == 0:
        print("Warning: No particles to generate, using default")
        total = int(ctl.bin_mass_particle_number[0])
        nstar = np.array([total])
        nstar_tot = total
    
    bksps_arr_ini.init(total)
    print(f"Generating {total} particles...")

    nsg0 = 0
    for i in range(ctl.m_bins):
        # Stars
        for j in range(int(nstar[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_MS
            sp.obidx = int(ctl.idxstar) if ctl.idxstar > 0 else 1
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nstar[i])

        # Stellar black holes
        for j in range(int(nsbh[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_BH
            sp.obidx = int(ctl.idxsbh) if ctl.idxsbh > 0 else 2
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nsbh[i])

        # White dwarfs
        for j in range(int(nwd[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_WD
            sp.obidx = int(ctl.idxwd) if ctl.idxwd > 0 else 4
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nwd[i])

        # Neutron stars
        for j in range(int(nns[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_NS
            sp.obidx = int(ctl.idxns) if ctl.idxns > 0 else 3
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nns[i])

        # Brown dwarfs
        for j in range(int(nbd[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_BD
            sp.obidx = int(ctl.idxbd) if ctl.idxbd > 0 else 5
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nbd[i])
    
    print(f"Generated {bksps_arr_ini.n} particles")


def get_init_samples(bksps_arr_ini: ParticleSamplesArrType) -> None:
    """Get initial particle samples."""
    get_init_samples_given(bksps_arr_ini)


def set_chain_samples(cbk: ChainType, bksps_arr: ParticleSamplesArrType) -> None:
    """Set chain samples from array."""
    set_chain_samples_single(cbk, bksps_arr)


def set_chain_samples_single(cbk: ChainType, bksps_arr: ParticleSamplesArrType) -> None:
    """Set single-component chain samples from array."""
    cbk.init(bksps_arr.n)
    ptbk = cbk.head
    for i in range(bksps_arr.n):
        if ptbk is None:
            break
        ca = bksps_arr.sp[i]
        ca.create_time = 0.0
        ca.simu_bgtime = 0.0
        ca.en0 = -cmg.mbh / (2.0 * ca.byot.a_bin) if ca.byot.a_bin > 0 else -1.0
        ca.jm0 = math.sqrt(max(0, 1.0 - ca.byot.e_bin ** 2))
        if ca.jm0 < cmg.jmin_value:
            ca.jm0 = cmg.jmin_value
        if ctl.clone_scheme >= 1:
            create_init_clone_particle(ptbk, ca.en0, 0.0)
        ptbk.ob = ca
        ptbk = ptbk.next
    
    print(f"Chain samples set: {cbk.n} particles")
