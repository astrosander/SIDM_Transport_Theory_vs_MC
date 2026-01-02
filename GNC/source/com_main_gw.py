"""
com_main_gw.py - Central module that provides all shared state and utility functions.
This module unifies all the common definitions needed by the GNC Python simulation.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, TextIO, Tuple

import numpy as np

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    _RID = _COMM.Get_rank()
    _NTASKS = _COMM.Get_size()
except ImportError:
    MPI = None
    _COMM = None
    _RID = 0
    _NTASKS = 1

# ========== Constants ==========
PI = math.pi
AU_SI = 1.496e11  # meters
rd_sun = 6.957e8 / AU_SI  # Solar radius in AU

# Star types
star_type_MS = 1
star_type_BH = 2
star_type_NS = 3
star_type_WD = 4
star_type_BD = 5

# J-bin types
Jbin_type_lin = 1
Jbin_type_log = 2
jbin_type_sqr = 3
# Lowercase aliases for consistency
jbin_type_lin = Jbin_type_lin
jbin_type_log = Jbin_type_log

# Exit flags
exit_normal = 0
exit_tidal = 2
exit_max_reach = 3
exit_boundary_min = 4
exit_boundary_max = 5
exit_invtransit = 7
exit_tidal_empty = 8
exit_tidal_full = 9
exit_ejection = 11
exit_gw_iso = 99
exit_plunge_single = 10

# Model integration types
model_intej_fast = 1
method_int_nearst = 1
method_int_linear = 2

# Boundary types
boundary_fj_iso = 1
boundary_fj_ls = 2

# Flags
flag_ini_or = 1
flag_ini_ini = 2
state_ae_evl = 0

# Track recording
record_track_nes = 1
record_track_detail = 2

# Component count
n_tot_comp = 7

# MPI globals
rid: int = _RID
proc_id: int = os.getpid()
mpi_master_id: int = 0
MPI_COMM_WORLD = _COMM

# Physical globals (will be set during initialization)
MBH: float = 4e6
mbh: float = MBH
mbh_radius: float = MBH / (299792.458 ** 2)  # Schwarzschild radius approximation
rh: float = 1.0  # influence radius
rhmin: float = 0.0
rhmax: float = 1.0
log10rh: float = 0.0

# Energy/angular momentum bounds
jmin_value: float = 5e-4
jmax_value: float = 0.99999
emin_factor: float = 0.03
emax_factor: float = 1e5
log10emin_factor: float = math.log10(0.03)
log10emax_factor: float = math.log10(1e5)
clone_e0_factor: float = 1.0
clone_emax: float = 1.0
log10clone_emax: float = 0.0

# Semi-major axis bounds (set in init_pro)
aomax: float = 0.0  # Will be set to rh / 2.0 / emin_factor
aomin: float = 0.0  # Will be set to rh / 2.0 / emax_factor
my_unit_vel_c: float = 299792.458  # km/s

# Output unit
chattery_out_unit: TextIO = None

# Boundary statistics
boundary_sts_emin_cros: int = 0
boundary_sts_emin_dir: int = 0
boundary_sts_create: int = 0
boundary_sts_replace: int = 0
boundary_cross_up: int = 0
boundary_cross_down: int = 0
boundary_cross_net: int = 0
boundary_sts_emax_cros: int = 0

# Placeholder for diffusion functions
df = None


@dataclass
class CoreCompType:
    """Core component type for stellar populations."""
    bk_evl_mode: int = 0
    blkmass: float = 0.0
    gamma: float = 0.0
    n0: float = 0.0
    alpha: float = 7.0 / 4.0
    N_in_rh: float = 0.0
    Mass_in_rh: float = 0.0
    N_pre_in: float = 0.0
    N_pre: int = 0
    nbd_tot: int = 0
    rb_min: float = 0.0
    rb_max: float = 0.0
    frac: float = 0.0
    rb: float = 0.0
    rb_ini: float = 0.0
    alpha_ini: float = 0.0
    rb_min_ini: float = 0.0
    rb_max_ini: float = 0.0
    clone_amplifier: int = 0
    str_bktypemodel: str = ""
    str_bkmassmodel: str = ""
    str_bkacmodel: str = ""
    str_bkacinimodel: str = ""
    bktypemodel: int = 0
    bkmassmodel: int = 0
    bkacmodel: int = 0
    bkacinimodel: int = 0


@dataclass
class ControlType:
    """Main control structure for simulation parameters."""
    cc: List[CoreCompType] = field(default_factory=list)

    mass_ref: float = 0.0
    bin_mass_min: float = 0.0
    bin_mass_max: float = 0.0
    n_basic: float = 0.0
    n0: float = 1.0
    v0: float = 1.0

    bin_mass: np.ndarray = field(default_factory=lambda: np.zeros(20))
    bin_mass_m1: np.ndarray = field(default_factory=lambda: np.zeros(20))
    bin_mass_m2: np.ndarray = field(default_factory=lambda: np.zeros(20))
    asymptot: np.ndarray = field(default_factory=lambda: np.zeros((8, 20)))
    ini_weight_n: np.ndarray = field(default_factory=lambda: np.zeros(20))
    weight_n: np.ndarray = field(default_factory=lambda: np.zeros(20))

    alpha_ini: float = 0.25
    total_time: float = 0.0
    sigma: float = 0.0
    energy0: float = 1.0
    energy_min: float = 0.0
    energy_max: float = 0.0

    clone_e0: float = 0.0
    rbd: float = 0.0
    x_boundary: float = 0.05
    energy_boundary: float = 0.0

    ts_spshot: float = 0.0
    tnr: float = 0.0
    ts_snap_input: float = 0.1
    update_dt: float = 0.0

    str_jbin_bd: str = "LOG"
    str_fj_bd: str = "ISO"
    time_unit: str = "TNR"
    str_model_intej: str = "FAST"
    str_method_interpolate: str = "near"
    cfs_file_dir: str = "../../common_data/cfuns_34"
    burn_in_dir: str = ""

    bin_mass_particle_number: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=int))
    clone_factor: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=int))

    EJ_mode: int = 0
    num_bk_comp: int = 0
    num_clone_created: int = 0
    num_boundary_created: int = 0
    num_boundary_elim: int = 0
    boundary_method: int = 0
    boundary_fj: int = boundary_fj_iso

    num_update_per_snap: int = 10
    n_spshot: int = 10
    n_spshot_total: int = 10
    include_loss_cone: int = 0
    model_intej: int = model_intej_fast

    same_rseed_evl: int = 1
    same_rseed_ini: int = 1
    clone_scheme: int = 1
    trace_all_sample: int = 0
    del_cross_clone: int = 0
    burn_in_snap: int = 0
    jbin_type: int = Jbin_type_log

    idxstar: int = 1
    idxsbh: int = -1
    idxns: int = -1
    idxwd: int = -1
    idxbd: int = -1

    method_interpolate: int = method_int_nearst
    m_bins: int = 1
    grid_bins: int = 72
    gx_bins: int = 24
    grid_type: int = 0
    idx_ref: int = 1

    debug: int = 0

    bin_mass_flux_in: np.ndarray = field(default_factory=lambda: np.zeros((20, 4), dtype=int))
    bin_mass_flux_out: np.ndarray = field(default_factory=lambda: np.zeros((20, 4), dtype=int))
    bin_mass_emax_out: np.ndarray = field(default_factory=lambda: np.zeros((20, 4), dtype=int))
    bin_mass_N: np.ndarray = field(default_factory=lambda: np.zeros((20, 4)))

    chattery: int = 0
    ntasks: int = _NTASKS
    ntask_total: int = _NTASKS
    seed_value: int = 100
    ntask_bg: int = 0

    nblock_mpi_bg: int = 1
    nblock_mpi_ed: int = 1
    nblock_size: int = 1

    output_track_td: int = 0
    output_track_emri: int = 0
    output_track_plunge: int = 0

    burn_in_phase: bool = False


# Global control instance
ctl = ControlType()


# ========== Utility Functions ==========

def skip_comments(fp: TextIO, comment_char: str = "#") -> None:
    """Skip comment lines in a file."""
    while True:
        pos = fp.tell()
        line = fp.readline()
        if not line:
            break
        if not line.strip().startswith(comment_char):
            fp.seek(pos)
            break


def readpar_int_sp(funit: int, fp: TextIO, comment: str, delim: str) -> int:
    """Read an integer parameter from file."""
    skip_comments(fp, comment)
    line = fp.readline()
    if delim in line:
        parts = line.split(delim)
        return int(parts[-1].strip())
    return int(line.strip().split()[0])


def readpar_dbl_sp(funit: int, fp: TextIO, comment: str, delim: str) -> float:
    """Read a double parameter from file."""
    skip_comments(fp, comment)
    line = fp.readline()
    if delim in line:
        parts = line.split(delim)
        val_str = parts[-1].strip().replace('d', 'e').replace('D', 'E')
        return float(val_str)
    val_str = line.strip().split()[0].replace('d', 'e').replace('D', 'E')
    return float(val_str)


def readpar_str_sp(funit: int, fp: TextIO, comment: str, delim: str) -> str:
    """Read a string parameter from file."""
    skip_comments(fp, comment)
    line = fp.readline()
    if delim in line:
        parts = line.split(delim)
        return parts[-1].strip()
    return line.strip()


def READPAR_STR_AUTO_sp(funit: int, fp: TextIO, comment: str, delim1: str, delim2: str, nmax: int) -> Tuple[str, List[str], int]:
    """Read multiple string parameters from a line."""
    skip_comments(fp, comment)
    line = fp.readline().strip()
    # Replace Fortran-style doubles
    line = line.replace('d', 'e').replace('D', 'E')
    parts = line.replace(delim1, delim2).split(delim2)
    parts = [p.strip() for p in parts if p.strip()]
    return line, parts[:nmax], min(len(parts), nmax)


def get_tnr_timescale_at_rh() -> float:
    """Calculate two-body relaxation time at influence radius."""
    global ctl, MBH, rh
    nm2_tot = 0.0
    for i in range(ctl.m_bins):
        nm2_tot += ctl.n0 * ctl.asymptot[0, i] * (ctl.bin_mass[i] ** 2)
    
    if nm2_tot <= 0:
        nm2_tot = ctl.n0 * ctl.bin_mass[0] ** 2
    
    lglambda = math.log(MBH / ctl.bin_mass[0]) if ctl.bin_mass[0] > 0 else 15.0
    tnr = 0.34 * (ctl.v0 ** 3) / nm2_tot / lglambda / 2.0 / PI / 1e6

    if rid == 0:
        print(f"TNR(rh)= {tnr:.4e} Myr")
    
    ctl.tnr = tnr
    return tnr


def get_ccidx_from_type(idxstar: int, bktype: int) -> int:
    """Get component index from stellar type."""
    for i in range(ctl.num_bk_comp):
        if ctl.cc[i].bktypemodel == bktype:
            return i + 1
    return -1


def mpi_barrier(comm=None) -> None:
    """MPI barrier synchronization."""
    if _COMM is not None:
        _COMM.Barrier()


def init_mpi() -> None:
    """Initialize MPI."""
    global rid, proc_id
    if MPI is None:
        print("Warning: mpi4py not available, running in single-process mode")
        rid = 0
        ctl.ntasks = 1
    else:
        comm = MPI.COMM_WORLD
        rid = comm.Get_rank()
        ctl.ntasks = comm.Get_size()
    proc_id = os.getpid()


def stop_mpi() -> None:
    """Finalize MPI."""
    if MPI is not None:
        try:
            MPI.Finalize()
        except Exception:
            pass
    print(f"MPI finalized, rid={rid}")


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(path) or os.path.exists(path + ".bin") or os.path.exists(path + ".hdf5")


# ========== Particle and Sample Types ==========

@dataclass
class Particle:
    """Basic particle type."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vx: np.ndarray = field(default_factory=lambda: np.zeros(3))
    m: float = 0.0
    radius: float = 0.0
    id: int = 0
    obtype: int = 0
    obidx: int = 0


@dataclass
class BinaryOrbit:
    """Binary orbit parameters."""
    a_bin: float = 0.0
    e_bin: float = 0.0
    mtot: float = 0.0
    Jc: float = 0.0
    Inc: float = 0.0
    inc: float = 0.0
    Om: float = 0.0
    om: float = 0.0
    pe: float = 0.0
    me: float = 0.0
    t0: float = 0.0
    f0: float = 0.0
    e: float = 0.0
    l: float = 0.0
    k: float = 0.0
    miu: float = 0.0
    lum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bname: str = ""
    an_in_mode: int = 0
    ms: Particle = field(default_factory=Particle)
    mm: Particle = field(default_factory=Particle)
    rd: Particle = field(default_factory=Particle)


@dataclass
class ParticleSampleType:
    """Particle sample for Monte Carlo simulation."""
    m: float = 0.0
    obtype: int = 0
    obidx: int = 0
    En: float = 0.0
    en: float = 0.0
    en0: float = 0.0
    Jm: float = 0.0
    jm: float = 0.0
    jm0: float = 0.0
    djp: float = 0.0
    djp0: float = 0.0
    elp: float = 0.0
    den: float = 0.0
    pd: float = 0.0
    rp: float = 0.0
    tgw: float = 0.0
    id: int = 0
    idx: int = 0
    length: int = 0
    state_flag_last: int = 0
    exit_flag: int = exit_normal
    within_jt: int = 0
    write_down_track: int = 0
    rid: int = 0
    r_td: float = 0.0
    create_time: float = 0.0
    exit_time: float = 0.0
    simu_bgtime: float = 0.0
    weight_real: float = 1.0
    weight_N: float = 0.0
    weight_n: float = 0.0
    weight_clone: float = 1.0
    byot: BinaryOrbit = field(default_factory=BinaryOrbit)
    byot_ini: BinaryOrbit = field(default_factory=BinaryOrbit)
    byot_bf: BinaryOrbit = field(default_factory=BinaryOrbit)
    track: Optional[List[Any]] = None

    def track_init(self, n: int) -> None:
        self.track = []


@dataclass
class ParticleSamplesArrType:
    """Array of particle samples."""
    n: int = 0
    sp: List[ParticleSampleType] = field(default_factory=list)

    def init(self, n: int) -> None:
        self.n = int(n)
        self.sp = [ParticleSampleType() for _ in range(self.n)]
    
    def select(self, out: "ParticleSamplesArrType", exitflag: int, emin: float, emax: float) -> None:
        """Select particles matching criteria."""
        selected = []
        for i in range(self.n):
            if self.sp[i].exit_flag == exitflag:
                selected.append(self.sp[i])
        out.n = len(selected)
        out.sp = selected


# Alias for compatibility
particle_samples_arr_type = ParticleSamplesArrType


@dataclass
class ChainPointer:
    """Linked list node for particle chains."""
    ob: Optional[ParticleSampleType] = None
    next: Optional["ChainPointer"] = None
    append_left: Optional["ChainPointer"] = None
    append_right: Optional["ChainPointer"] = None


@dataclass
class ChainType:
    """Linked list chain of particles."""
    n: int = 0
    head: Optional[ChainPointer] = None

    def init(self, n: int) -> None:
        self.n = int(n)
        if self.n <= 0:
            self.head = None
            return
        self.head = ChainPointer(ob=ParticleSampleType())
        p = self.head
        for _ in range(1, self.n):
            p.next = ChainPointer(ob=ParticleSampleType())
            p = p.next

    def get_length(self, type: int = 0) -> int:
        count = 0
        p = self.head
        while p is not None:
            count += 1
            p = p.next
        return count

    def destory(self) -> None:
        self.head = None
        self.n = 0

    def output_bin(self, path: str) -> None:
        """Output chain to binary file (non-recursive to avoid stack overflow)."""
        import pickle
        
        # Convert linked list to a regular list to avoid recursion issues
        particles = []
        pt = self.head
        while pt is not None:
            if pt.ob is not None:
                # Create a copy of the particle data without the chain pointers
                particles.append(pt.ob)
            pt = pt.next
        
        # Save as list with count
        data = {'n': self.n, 'particles': particles}
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def input_bin(self, path: str) -> None:
        """Input chain from binary file."""
        import pickle
        pth = f"{path}.pkl" if not path.endswith(".pkl") else path
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                data = pickle.load(f)
            
            # Reconstruct chain from list
            if isinstance(data, dict) and 'particles' in data:
                particles = data['particles']
                self.n = data.get('n', len(particles))
                
                if len(particles) == 0:
                    self.head = None
                    return
                
                # Rebuild the linked list
                self.head = ChainPointer(ob=particles[0])
                pt = self.head
                for i in range(1, len(particles)):
                    pt.next = ChainPointer(ob=particles[i])
                    pt = pt.next
            else:
                # Old format - try to update from dict
                self.__dict__.update(data.__dict__)


# Global instances
bksams_arr_ini = ParticleSamplesArrType()
bksams = ChainType()
bksams_arr = ParticleSamplesArrType()
bksams_arr_norm = ParticleSamplesArrType()
bksams_pointer_arr = None

# Event tracking placeholders
pteve_star = None
pteve_sbh = None
pteve_bd = None
pteve_ns = None
pteve_wd = None


# ========== Import cfuns from md_cfuns ==========
from md_cfuns import CfunsType, cfs


print(f"com_main_gw loaded: rid={rid}, ntasks={ctl.ntasks}")

