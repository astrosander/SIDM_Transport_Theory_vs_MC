"""
gen_ge.py - Generate distribution function g(x) from particle samples.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    _RID = _COMM.Get_rank()
    _NTASKS = _COMM.Get_size()
except Exception:
    MPI = None
    _COMM = None
    _RID = 0
    _NTASKS = 1


Jbin_type_lin = 1
Jbin_type_log = 2
Jbin_type_sqr = 3

model_intej_fast = 1

star_type_MS = 1
star_type_BH = 2
star_type_WD = 3
star_type_NS = 4
star_type_BD = 5

exit_normal = 0


def set_seed(same_seed: int, seed_value: int) -> None:
    """Set random seed for reproducibility."""
    if same_seed > 0:
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        random.seed()
        np.random.seed()


def collection_data_single_bympi(smsa: List[Any], n: int) -> None:
    """Collect particle sample data via MPI."""
    if _COMM is None or _NTASKS <= 1:
        # Single process mode - just copy local data
        if len(smsa) > 0:
            smsa[0] = bksams_arr_norm
        return
    
    # MPI mode
    update_arrays_single()
    _COMM.Barrier()
    
    if _RID == mpi_master_id:
        print("single: start_collect")
        # Root collects from all processes
        for i in range(_NTASKS):
            if i != mpi_master_id:
                # Receive from other processes
                data = _COMM.recv(source=i, tag=0)
                smsa[i] = data
            else:
                smsa[mpi_master_id] = bksams_arr_norm
        print("single: end_collect")
    else:
        # Send to root
        _COMM.send(bksams_arr_norm, dest=mpi_master_id, tag=0)
    
    print(f"collection finished rid={_RID}")


def get_dms(dm: "DiffuseMspec") -> None:
    """Get diffuse mass spectrum with MPI synchronization."""
    if _COMM is not None:
        _COMM.Barrier()
    
    # Broadcast barge and fden
    bcast_dms_barge(dm)
    bcast_dms_fden(dm)
    
    print(f"start get diffuse coefficients rid={_RID}")
    dm_get_dc_mpi(dm)
    print(f"cal dms finished rid={_RID}")
    
    if _COMM is not None:
        _COMM.Barrier()


def bcast_dms_barge(dm: "DiffuseMspec") -> None:
    """Broadcast barge arrays via MPI."""
    if _COMM is None or _NTASKS <= 1:
        return
    # Placeholder - in full implementation would broadcast S1D arrays
    pass


def bcast_dms_fden(dm: "DiffuseMspec") -> None:
    """Broadcast fden arrays via MPI."""
    if _COMM is None or _NTASKS <= 1:
        return
    # Placeholder - in full implementation would broadcast S1D arrays
    pass


@dataclass
class Control:
    x_boundary: float = 0.0
    energy0: float = 1.0
    energy_boundary: float = 0.0
    v0: float = 1.0
    n0: float = 1.0
    m_bins: int = 0
    grid_bins: int = 0
    gx_bins: int = 0
    idx_ref: int = 1
    jbin_type: int = Jbin_type_lin
    grid_type: int = 0
    model_intej: int = model_intej_fast
    nblock_mpi_bg: int = 1
    nblock_mpi_ed: int = 1
    nblock_size: int = 1
    ntasks: int = 1
    cc: Optional[list] = None
    bin_mass: Optional[np.ndarray] = None
    bin_mass_m1: Optional[np.ndarray] = None
    bin_mass_m2: Optional[np.ndarray] = None
    asymptot: Optional[np.ndarray] = None


@dataclass
class CCT:
    alpha: float = 7.0 / 4.0


@dataclass
class S1DType:
    xmin: float = 0.0
    xmax: float = 0.0
    nbin: int = 0
    xb: Optional[np.ndarray] = None
    fx: Optional[np.ndarray] = None
    sts_type: int = 0

    def init(self, xmin: float, xmax: float, nbin: int, sts_type: int) -> None:
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.nbin = int(nbin)
        self.sts_type = int(sts_type)
        self.xb = np.zeros(self.nbin, dtype=float)
        self.fx = np.zeros(self.nbin, dtype=float)

    def set_range(self) -> None:
        if self.xb is None:
            return
        if self.nbin <= 1:
            self.xb[:] = self.xmin
        else:
            self.xb[:] = np.linspace(self.xmin, self.xmax, self.nbin)

    def get_value_l(self, x: float) -> float:
        xb = self.xb
        fx = self.fx
        if xb is None or fx is None or xb.size == 0:
            return 0.0
        if x <= xb[0]:
            return float(fx[0])
        if x >= xb[-1]:
            return float(fx[-1])
        return float(np.interp(x, xb, fx))


@dataclass
class S2DType:
    nbinx: int = 0
    nbiny: int = 0
    nx: int = 0  # Alias for nbinx
    ny: int = 0  # Alias for nbiny
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    xstep: float = 0.0
    ystep: float = 0.0
    xcenter: Optional[np.ndarray] = None
    ycenter: Optional[np.ndarray] = None
    fxy: Optional[np.ndarray] = None
    sts_type: int = 0

    def init(self, nbinx: int, nbiny: int, xmin: float, xmax: float, ymin: float, ymax: float, sts_type: int) -> None:
        self.nbinx = int(nbinx)
        self.nbiny = int(nbiny)
        self.nx = self.nbinx
        self.ny = self.nbiny
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.sts_type = int(sts_type)
        self.xcenter = np.zeros(self.nbinx, dtype=float)
        self.ycenter = np.zeros(self.nbiny, dtype=float)
        self.fxy = np.zeros((self.nbinx, self.nbiny), dtype=float)

    def set_range(self) -> None:
        if self.nbinx > 1:
            self.xcenter[:] = np.linspace(self.xmin, self.xmax, self.nbinx)
            self.xstep = (self.xmax - self.xmin) / float(self.nbinx - 1) if self.nbinx > 1 else 0.0
        else:
            self.xcenter[:] = self.xmin
            self.xstep = 0.0
        if self.nbiny > 1:
            self.ycenter[:] = np.linspace(self.ymin, self.ymax, self.nbiny)
            self.ystep = (self.ymax - self.ymin) / float(self.nbiny - 1) if self.nbiny > 1 else 0.0
        else:
            self.ycenter[:] = self.ymin
            self.ystep = 0.0


@dataclass
class DiffuseCoeffGrid:
    s2_de_110: S2DType = field(default_factory=S2DType)
    s2_de_0: S2DType = field(default_factory=S2DType)
    s2_dee: S2DType = field(default_factory=S2DType)
    s2_dj_111: S2DType = field(default_factory=S2DType)
    s2_dj_rest: S2DType = field(default_factory=S2DType)
    s2_djj: S2DType = field(default_factory=S2DType)
    s2_dej: S2DType = field(default_factory=S2DType)

    def init(self, nbin_grid: int, emin: float, emax: float, jmin: float, jmax: float, grid_type: int) -> None:
        """Initialize all diffusion coefficient grids."""
        sts_type = 0  # Default statistics type
        self.s2_de_110.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_de_110.set_range()
        self.s2_de_0.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_de_0.set_range()
        self.s2_dee.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_dee.set_range()
        self.s2_dj_111.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_dj_111.set_range()
        self.s2_dj_rest.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_dj_rest.set_range()
        self.s2_djj.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_djj.set_range()
        self.s2_dej.init(nbin_grid, nbin_grid, emin, emax, jmin, jmax, sts_type)
        self.s2_dej.set_range()


@dataclass
class NejwRec:
    e: float
    j: float
    w: float
    idx: int


@dataclass
class S2DHstType:
    """2D histogram type with weights."""
    nx: int = 0
    ny: int = 0
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    xstep: float = 0.0
    ystep: float = 0.0
    xcenter: Optional[np.ndarray] = None
    ycenter: Optional[np.ndarray] = None
    nxyw: Optional[np.ndarray] = None  # Weighted histogram
    
    def init(self, nbinx: int, nbiny: int, xmin: float, xmax: float, ymin: float, ymax: float, use_weight: bool = True) -> None:
        self.nx = int(nbinx)
        self.ny = int(nbiny)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.xcenter = np.zeros(self.nx, dtype=float)
        self.ycenter = np.zeros(self.ny, dtype=float)
        self.nxyw = np.zeros((self.nx, self.ny), dtype=float)
        
    def set_range(self) -> None:
        if self.nx > 1:
            self.xcenter[:] = np.linspace(self.xmin, self.xmax, self.nx)
            self.xstep = (self.xmax - self.xmin) / float(self.nx - 1) if self.nx > 1 else 0.0
        else:
            self.xcenter[:] = self.xmin
            self.xstep = 0.0
        if self.ny > 1:
            self.ycenter[:] = np.linspace(self.ymin, self.ymax, self.ny)
            self.ystep = (self.ymax - self.ymin) / float(self.ny - 1) if self.ny > 1 else 0.0
        else:
            self.ycenter[:] = self.ymin
            self.ystep = 0.0
    
    def get_stats_weight(self, en: List[float], jm: List[float], we: List[float], n: int) -> None:
        """Bin particles into weighted histogram.
        
        Replicates Fortran bin2_weight with flag=0 (sts_type_dstr).
        See sts_fc.f90 lines 175-190 and return_idxy lines 306-321.
        """
        # Compute step sizes (Fortran flag=0 uses simple division)
        xstep = (self.xmax - self.xmin) / float(self.nx) if self.nx > 1 else 1.0
        ystep = (self.ymax - self.ymin) / float(self.ny) if self.ny > 1 else 1.0
        
        for k in range(n):
            x_val = en[k]
            y_val = jm[k]
            
            # Find x bin index (flag=0 sts_type_dstr, Fortran lines 306-312)
            if x_val >= self.xmin and x_val < self.xmax:
                ix = int((x_val - self.xmin) / xstep) + 1  # Fortran is 1-indexed
            elif abs(x_val - self.xmax) < 1e-10:  # x == xmax
                ix = self.nx
            else:
                ix = -9999  # Out of bounds
            
            # Find y bin index
            if y_val >= self.ymin and y_val < self.ymax:
                iy = int((y_val - self.ymin) / ystep) + 1  # Fortran is 1-indexed
            elif abs(y_val - self.ymax) < 1e-10:  # y == ymax
                iy = self.ny
            else:
                iy = -9999  # Out of bounds
            
            # Convert to 0-indexed for Python and add weight
            if ix > 0 and ix <= self.nx and iy > 0 and iy <= self.ny:
                self.nxyw[ix - 1, iy - 1] += we[k]


@dataclass
class DmsStellarObject:
    n: int = 0
    n_real: float = 0.0
    nejw: Optional[np.ndarray] = None
    barge: S1DType = field(default_factory=S1DType)
    nxj: S2DHstType = field(default_factory=S2DHstType)
    gxj: S2DType = field(default_factory=S2DType)
    
    def dms_so_get_nxj_from_nejw(self, jbtype: int) -> None:
        """Bin particles into nxj histogram."""
        if self.n <= 0:
            self.n_real = 0.0
            return
        if self.nejw is None:
            return
        
        en = [rec.e for rec in self.nejw[:self.n]]
        jm_raw = [rec.j for rec in self.nejw[:self.n]]
        we = [rec.w for rec in self.nejw[:self.n]]
        
        # Convert j to appropriate space
        if jbtype == Jbin_type_lin:
            jm = jm_raw
        elif jbtype == Jbin_type_log:
            jm = [math.log10(v) if v > 0 else -10 for v in jm_raw]
        elif jbtype == Jbin_type_sqr:
            jm = [v * v for v in jm_raw]
        else:
            raise RuntimeError(f"dms_nxj_newj:error! define jbtype {jbtype}")
        
        self.nxj.get_stats_weight(en, jm, we, self.n)
        self.n_real = sum(we)


@dataclass
class MassBins:
    mc: float = 0.0
    m1: float = 0.0
    m2: float = 0.0
    nbin_grid: int = 0
    nbin_gx: int = 0
    emin: float = 0.0
    emax: float = 0.0
    jmin: float = 0.0
    jmax: float = 0.0
    mbh: float = 0.0
    v0: float = 1.0
    n0: float = 1.0
    rh: float = 1.0
    dc: DiffuseCoeffGrid = field(default_factory=DiffuseCoeffGrid)
    all: DmsStellarObject = field(default_factory=DmsStellarObject)
    star: DmsStellarObject = field(default_factory=DmsStellarObject)
    sbh: DmsStellarObject = field(default_factory=DmsStellarObject)
    wd: DmsStellarObject = field(default_factory=DmsStellarObject)
    ns: DmsStellarObject = field(default_factory=DmsStellarObject)
    bd: DmsStellarObject = field(default_factory=DmsStellarObject)


@dataclass
class DiffuseMspec:
    n: int = 0
    mb: Optional[list] = None
    mbh: float = 0.0
    v0: float = 1.0
    n0: float = 1.0
    rh: float = 1.0
    emin: float = 0.0
    emax: float = 0.0
    jmin: float = 0.0
    jmax: float = 0.0
    nbin_grid: int = 0
    nbin_gx: int = 0
    x_boundary: float = 0.0
    idx_ref: int = 1
    jbin_type: int = Jbin_type_lin
    grid_type: int = 0
    weight_asym: float = 1.0
    dc0: DiffuseCoeffGrid = field(default_factory=DiffuseCoeffGrid)
    all: MassBins = field(default_factory=MassBins)

    def set_diffuse_mspec(self, nbin_grid: int, nbin_gx: int, emin: float, emax: float, jmin: float, jmax: float,
                         mbh: float, v0: float, n0: float, rh: float, xb: float, idx_ref: int, jb_type: int,
                         grid_type: int) -> None:
        self.nbin_grid = int(nbin_grid)
        self.nbin_gx = int(nbin_gx)
        self.emin = float(emin)
        self.emax = float(emax)
        self.jmin = float(jmin)
        self.jmax = float(jmax)
        self.mbh = float(mbh)
        self.v0 = float(v0)
        self.n0 = float(n0)
        self.rh = float(rh)
        self.x_boundary = float(xb)
        self.idx_ref = int(idx_ref)
        self.jbin_type = int(jb_type)
        self.grid_type = int(grid_type)

    def init(self, n: int) -> None:
        self.n = int(n)
        self.mb = [MassBins() for _ in range(self.n)]
        
        # Initialize dc0 (global diffusion coefficients)
        self.dc0.init(self.nbin_grid, self.emin, self.emax, self.jmin, self.jmax, self.grid_type)
        
        # Initialize each mass bin's dc grid and parameters
        for i in range(self.n):
            mb = self.mb[i]
            mb.nbin_grid = self.nbin_grid
            mb.nbin_gx = self.nbin_gx
            mb.emin = self.emin
            mb.emax = self.emax
            mb.jmin = self.jmin
            mb.jmax = self.jmax
            mb.mbh = self.mbh
            mb.v0 = self.v0
            mb.n0 = self.n0
            mb.rh = self.rh
            # Initialize diffusion coefficient grid for this mass bin
            mb.dc.init(self.nbin_grid, self.emin, self.emax, self.jmin, self.jmax, self.grid_type)
            # Initialize barge, nxj, and gxj for each stellar object type
            for so_name in ['all', 'star', 'sbh', 'wd', 'ns', 'bd']:
                so = getattr(mb, so_name)
                # Initialize barge (g(x) distribution)
                so.barge.init(self.emin, self.emax, self.nbin_gx, 0)
                so.barge.set_range()
                # Initialize nxj (weighted histogram)
                so.nxj.init(self.nbin_gx, self.nbin_gx, self.emin, self.emax, self.jmin, self.jmax, True)
                so.nxj.set_range()
                # Initialize gxj (normalized distribution function)
                so.gxj.init(self.nbin_gx, self.nbin_gx, self.emin, self.emax, self.jmin, self.jmax, 0)
                so.gxj.set_range()

    def get_asymp_norm_factor(self) -> None:
        get_asymp_norm_factor(self)

    def normalize_barge(self) -> None:
        normalize_barge(self)

    def get_dens0(self) -> None:
        get_dens0(self)

    def get_nxj(self) -> None:
        get_nxj(self)

    def get_fxj0(self) -> None:
        get_fxj0(self)

    def get_barge0(self) -> None:
        get_barge0(self)

    def output_bin(self, fl: str) -> None:
        """Output DiffuseMspec to binary file."""
        import pickle
        with open(fl, "wb") as f:
            # Save basic parameters
            pickle.dump({
                'n': self.n,
                'mbh': self.mbh,
                'v0': self.v0,
                'n0': self.n0,
                'rh': self.rh,
                'emin': self.emin,
                'emax': self.emax,
                'jmin': self.jmin,
                'jmax': self.jmax,
                'nbin_grid': self.nbin_grid,
                'nbin_gx': self.nbin_gx,
                'x_boundary': self.x_boundary,
                'idx_ref': self.idx_ref,
                'jbin_type': self.jbin_type,
                'grid_type': self.grid_type,
                'weight_asym': self.weight_asym,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save mass bins (simplified - just the essential data)
            mb_data = []
            if self.mb is not None:
                for mb in self.mb:
                    mb_data.append({
                        'mc': mb.mc,
                        'm1': mb.m1,
                        'm2': mb.m2,
                    })
            pickle.dump(mb_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def input_bin(self, fl: str) -> None:
        """Input DiffuseMspec from binary file."""
        import pickle
        import os
        if not os.path.exists(fl):
            print(f"Warning: DMS file not found: {fl}")
            return
        with open(fl, "rb") as f:
            params = pickle.load(f)
            for key, val in params.items():
                if hasattr(self, key):
                    setattr(self, key, val)
            
            mb_data = pickle.load(f)
            if len(mb_data) > 0:
                self.init(len(mb_data))
                for i, mbd in enumerate(mb_data):
                    for key, val in mbd.items():
                        if hasattr(self.mb[i], key):
                            setattr(self.mb[i], key, val)

    def print_norm(self, unit: int) -> None:
        """Print normalization info."""
        print(f"DMS norm: n={self.n}, weight_asym={self.weight_asym}")


@dataclass
class ParticleSample:
    en: float = 0.0
    jm: float = 0.0
    m: float = 0.0
    weight_real: float = 0.0


@dataclass
class ParticleSamplesArr:
    n: int = 0
    sp: Optional[np.ndarray] = None

    def select(self, out: "ParticleSamplesArr", exitflag: int, emin: float, emax: float) -> None:
        if self.sp is None or self.n <= 0:
            out.n = 0
            out.sp = np.zeros(0, dtype=object)
            return
        out.n = self.n
        out.sp = self.sp.copy()


ctl = Control(ntasks=_NTASKS)
rid = _RID
mpi_master_id = 0

dms = DiffuseMspec()

log10emin_factor = 0.0
log10emax_factor = 0.0
jmin_value = 0.0
jmax_value = 0.0
mbh = 0.0
rh = 1.0
emax_factor = 1.0
emin_factor = 1.0

cct_share: Optional[CCT] = None
fc_share: Optional[S1DType] = None
fgx_g0: float = 0.0


def mpi_barrier() -> None:
    if _COMM is not None:
        _COMM.Barrier()


def collect_data_mpi(a: np.ndarray, nbin: int, nbg: int, ned: int, nblock: int, ntasks: int) -> None:
    if _COMM is None:
        return
    if ntasks <= 1:
        return
    full = np.empty_like(a)
    _COMM.Allreduce(a, full, op=MPI.SUM)
    a[:, :] = full[:, :]


def smmerge_arr_single(smsa: Sequence[ParticleSamplesArr], n: int) -> ParticleSamplesArr:
    parts = []
    for k in range(int(n)):
        if smsa[k].sp is not None and smsa[k].n > 0:
            parts.append(smsa[k].sp[:smsa[k].n])
    if len(parts) == 0:
        return ParticleSamplesArr(n=0, sp=np.zeros(0, dtype=object))
    sp = np.concatenate(parts, axis=0)
    return ParticleSamplesArr(n=int(sp.shape[0]), sp=sp)


def sams_get_weight_clone_single(sms_arr_single: ParticleSamplesArr) -> None:
    return


def set_real_weight_arr_single(sms_arr_single: ParticleSamplesArr) -> None:
    return


def sams_arr_select_type_single(bks: ParticleSamplesArr, out: ParticleSamplesArr, stype: int) -> None:
    if bks.sp is None or bks.n <= 0:
        out.n = 0
        out.sp = np.zeros(0, dtype=object)
        return
    out.n = bks.n
    out.sp = bks.sp.copy()


def all_chain_to_arr_single(chain: Any, out: ParticleSamplesArr) -> None:
    """Convert chain of particles to array format."""
    if chain is None or not hasattr(chain, 'head') or chain.head is None:
        out.n = 0
        out.sp = np.zeros(0, dtype=object)
        return
    
    # Count particles in chain
    count = 0
    ptr = chain.head
    while ptr is not None:
        if hasattr(ptr, 'ob') and ptr.ob is not None:
            count += 1
        ptr = ptr.next if hasattr(ptr, 'next') else None
    
    if count == 0:
        out.n = 0
        out.sp = np.zeros(0, dtype=object)
        return
    
    # Extract particles from chain
    particles = np.empty(count, dtype=object)
    ptr = chain.head
    idx = 0
    while ptr is not None and idx < count:
        if hasattr(ptr, 'ob') and ptr.ob is not None:
            particles[idx] = ptr.ob
            idx += 1
        ptr = ptr.next if hasattr(ptr, 'next') else None
    
    out.n = idx
    out.sp = particles[:idx]
    print(f"DEBUG all_chain_to_arr_single: Extracted {out.n} particles from chain")


def set_sample_arr_indexs_rid_particle(arr: ParticleSamplesArr, rid: int) -> None:
    return


def convert_sams_pointer_arr(chain: Any, out: Any, type: int = 1) -> None:
    return


def bcast_dms_asym_weights(dm: DiffuseMspec) -> None:
    return


def set_clone_weight(chain: Any) -> None:
    return


def set_real_weight(chain: Any) -> None:
    return


def get_sigma0(energyx: float, fgx_func: Callable[[float], float], out: Any) -> float:
    return 0.0


def get_coeff_sigma_funcs_cfs_grid(energy: float, jum: float, barg: Callable[[float], float],
                                   s110: Any, s111: Any, s131: Any, s130: Any, s13_1: Any, s330: Any, s310: Any) -> Tuple[float, float, float, float, float, float, float]:
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


@dataclass
class CoeffType:
    e: float = 0.0
    j: float = 0.0
    ee: float = 0.0
    jj: float = 0.0
    ej: float = 0.0
    e_110: float = 0.0
    e_0: float = 0.0
    j_111: float = 0.0
    j_rest: float = 0.0
    m_avg: float = 0.0


def get_coeff_ej(jum: float, sigma0: float, sigma110: float, sigma111: float, sigma131: float,
                 sigma130: float, sigma13_1: float, sigma330: float, sigma310: float) -> CoeffType:
    coe = CoeffType()
    coe.e_110 = sigma110
    coe.e_0 = -sigma0
    coe.ee = 4.0 / 3.0 * (sigma0 + sigma13_1)
    coe.j_111 = -jum * sigma111
    coe.j_rest = (1.0 / jum) * (((5.0 - 3.0 * jum * jum) / 12.0) * sigma0 + sigma310 - sigma330 / 3.0)
    coe.jj = ((5.0 - 3.0 * jum * jum) / 6.0) * sigma0 + (jum * jum / 2.0) * sigma131 - (jum * jum / 2.0) * sigma111 + 2.0 * sigma310 - 2.0 / 3.0 * sigma330
    coe.ej = -2.0 / 3.0 * jum * (sigma0 + sigma130)
    return coe


def get_dc0(dm: DiffuseMspec) -> None:
    return


def get_asymp_norm_factor(dm: DiffuseMspec) -> None:
    return


def normalize_barge(dm: DiffuseMspec) -> None:
    return


def get_dens0(dm: DiffuseMspec) -> None:
    return


def get_nxj(dm: DiffuseMspec) -> None:
    """Get nxj histogram from particle (e,j,w) values."""
    if dm.mb is None:
        return
    
    n_tot_comp = 6  # star, sbh, ns, wd, bd, (bstar, bbh not used here)
    
    for i in range(dm.n):
        mb = dm.mb[i]
        # Process each stellar object type
        for so_name in ['star', 'sbh', 'ns', 'wd', 'bd', 'all']:
            so = getattr(mb, so_name, None)
            if so is not None and hasattr(so, 'dms_so_get_nxj_from_nejw'):
                so.dms_so_get_nxj_from_nejw(dm.jbin_type)
        
        # Sum all components into 'all'
        if hasattr(mb.all, 'nxj') and hasattr(mb.all.nxj, 'nxyw'):
            mb.all.nxj.nxyw = np.zeros_like(mb.all.nxj.nxyw)
            for so_name in ['star', 'sbh', 'ns', 'wd', 'bd']:
                so = getattr(mb, so_name, None)
                if so is not None and hasattr(so, 'nxj') and hasattr(so.nxj, 'nxyw'):
                    mb.all.nxj.nxyw += so.nxj.nxyw


def get_fxj0(dm: DiffuseMspec) -> None:
    """Normalize nxj histogram to create gxj distribution function."""
    if dm.mb is None:
        return
    
    for i in range(dm.n):
        mb = dm.mb[i]
        # Process each stellar object
        for so_name in ['all', 'star', 'sbh', 'ns', 'wd', 'bd']:
            so = getattr(mb, so_name, None)
            if so is not None:
                dms_so_get_fxj(so, mb.n0, mb.mbh, mb.v0, dm.jbin_type)


def get_barge0(dm: DiffuseMspec) -> None:
    """Integrate gxj over j to get barge (g(x) distribution)."""
    if dm.mb is None:
        return
    
    for i in range(dm.n):
        mb = dm.mb[i]
        # Process each stellar object
        for so_name in ['all', 'star', 'sbh', 'ns', 'wd', 'bd']:
            so = getattr(mb, so_name, None)
            if so is not None:
                get_barge_stellar(so, dm.jbin_type)


def get_barge_stellar(so: Any, jbtype: int) -> None:
    """Integrate gxj over j to get barge (g(x) distribution).
    
    Replicates get_barge_stellar from stellar_obj.f90 lines 193-256.
    """
    if not hasattr(so, 'n') or so.n == 0:
        return
    if not hasattr(so, 'barge') or not hasattr(so, 'gxj'):
        return
    
    barge = so.barge
    gxj = so.gxj
    
    if not hasattr(barge, 'fx') or not hasattr(gxj, 'fxy'):
        return
    if barge.nbin != gxj.nx:
        print(f"Error: barge.nbin={barge.nbin} should equal gxj.nx={gxj.nx}")
        return
    
    log10_val = math.log(10.0)
    
    # Fortran lines 200-248
    for i in range(barge.nbin):
        int_out = 0.0
        # Fortran line 207: Set x-coordinate
        barge.xb[i] = gxj.xcenter[i]
        
        if jbtype == Jbin_type_lin:
            # Fortran lines 210-216: Linear j-bins
            # Integration: sum of g(x,j) * j * dj * 2
            for j in range(gxj.ny):
                int_out += gxj.fxy[i, j] * gxj.ycenter[j] * gxj.ystep * 2.0
        
        elif jbtype == Jbin_type_log:
            # Fortran lines 225-232: Log j-bins
            # Integration: sum of g(x,j) * (10^log_j)^2 * d(log_j) * 2 * log(10)
            for j in range(gxj.ny):
                int_out += gxj.fxy[i, j] * (10.0 ** gxj.ycenter[j]) ** 2 * gxj.ystep * 2.0 * log10_val
        
        elif jbtype == Jbin_type_sqr:
            # Fortran lines 240-242: Squared j-bins
            # Integration: sum of g(x,j) * j^2 * d(j^2) * 2
            for j in range(gxj.ny):
                int_out += gxj.fxy[i, j] * gxj.ycenter[j] * gxj.ystep * 2.0
        
        barge.fx[i] = int_out
        
        # Check for NaN (Fortran lines 219-223)
        if math.isnan(barge.fx[i]):
            print(f"get_barge_stellar: fx[{i}] is NaN, int_out={int_out}")
            return
    
    # Debug output
    if hasattr(barge, 'fx') and barge.fx is not None:
        fx_sum = np.sum(barge.fx) if isinstance(barge.fx, np.ndarray) else sum(barge.fx)
        fx_max = np.max(barge.fx) if isinstance(barge.fx, np.ndarray) else max(barge.fx)
        print(f"DEBUG get_barge_stellar: so.n={so.n}, sum(fx)={fx_sum:.3e}, max(fx)={fx_max:.3e}")


def get_ejw_from_particle(estar: np.ndarray, jstar: np.ndarray, wstar: np.ndarray, mstar: np.ndarray, n_star: int,
                          m1: float, m2: float, mbh: float, v0: float, xb: float,
                          nejw_out: Any, nsam_out: Any) -> Tuple[np.ndarray, int]:
    """Convert particle (e,j,mass,weight) to nejw format for binning.
    
    This replicates get_ejw_from_particle from dms.f90 lines 520-572.
    Energy is converted from physical energy to log10(x) where x = |E|/v0^2.
    """
    if n_star <= 0:
        return np.zeros(0, dtype=object), 0
    
    # Convert energy to dimensionless form: x = |E|/v0^2 (Fortran line 530)
    xstar = np.abs(estar[:n_star]) / (v0 ** 2)
    
    # Filter particles by mass range (Fortran line 532)
    mask = (mstar[:n_star] >= m1) & (mstar[:n_star] <= m2)
    idx = np.where(mask)[0]
    n_selected = len(idx)
    
    if n_selected == 0:
        return np.zeros(0, dtype=object), 0
    
    # Create nejw records with log10(x) for energy (Fortran line 537)
    nejw_list = []
    for i in idx:
        rec = NejwRec(
                e=float(math.log10(xstar[i])),  # Store as log10(x)
                j=float(jstar[i]),
                w=float(wstar[i]),
                idx=int(i)
            )
        nejw_list.append(rec)
    
    return np.array(nejw_list, dtype=object), len(nejw_list)


def get_ge_by_root(smsa: Sequence[ParticleSamplesArr], n: int, norm_in: bool) -> None:
    if rid == mpi_master_id:
        smstot = smmerge_arr_single(smsa, n)
        print("smstot%n=", smstot.n)
        get_ge(dms, smstot, norm_in)
        smstot.sp = None


def get_ge(dm: DiffuseMspec, sms_arr_single: ParticleSamplesArr, norm_in: bool) -> None:
    sams_get_weight_clone_single(sms_arr_single)
    if norm_in:
        dm.weight_asym = 1.0
    set_real_weight_arr_single(sms_arr_single)

    sms_arr_star = ParticleSamplesArr()
    sms_arr_sbh = ParticleSamplesArr()
    sms_arr_wd = ParticleSamplesArr()
    sms_arr_ns = ParticleSamplesArr()
    sms_arr_bd = ParticleSamplesArr()

    separate_to_species(sms_arr_single, sms_arr_star, sms_arr_sbh, sms_arr_wd, sms_arr_ns, sms_arr_bd)

    gen_gx(dm, sms_arr_star, sms_arr_sbh, sms_arr_wd, sms_arr_ns, sms_arr_bd)

    if norm_in:
        dm.get_asymp_norm_factor()
        dm.normalize_barge()

    print("start get_dens0")
    print("weight_asym=", dm.weight_asym)
    dm.get_dens0()
    print("finished get_dens0")

    set_real_weight_arr_single(sms_arr_single)

    if norm_in:
        separate_to_species(sms_arr_single, sms_arr_star, sms_arr_sbh, sms_arr_wd, sms_arr_ns, sms_arr_bd)
        conv_dms_nejw(dm, sms_arr_star, sms_arr_sbh, sms_arr_wd, sms_arr_ns, sms_arr_bd)

    sms_arr_star.sp = None
    sms_arr_sbh.sp = None
    sms_arr_wd.sp = None
    sms_arr_ns.sp = None
    sms_arr_bd.sp = None

    print("finished get_ge")


def get_fma(fna: S1DType, fma: S1DType, mass: float) -> None:
    if fna.fx is None or fma.fx is None:
        return
    fma.fx[:fna.nbin] = fna.fx[:fna.nbin] * float(mass)


def dm_get_dc_mpi(dm: DiffuseMspec) -> None:
    if dm.mb is None:
        return
    for i in range(dm.n):
        mb_get_dc_mpi(dm.mb[i])
    get_dc0(dm)


def mb_get_dc_mpi(mb: MassBins) -> None:
    n = int(mb.nbin_grid)
    ycenter = np.zeros(n, dtype=float)
    s0 = np.zeros((n, n), dtype=float)
    s110 = np.zeros((n, n), dtype=float)
    s111 = np.zeros((n, n), dtype=float)
    s131 = np.zeros((n, n), dtype=float)
    s130 = np.zeros((n, n), dtype=float)
    s13_1 = np.zeros((n, n), dtype=float)
    s330 = np.zeros((n, n), dtype=float)
    s310 = np.zeros((n, n), dtype=float)

    if ctl.jbin_type == Jbin_type_lin:
        ycenter[:] = mb.dc.s2_de_110.ycenter[:n]
    elif ctl.jbin_type == Jbin_type_log:
        ycenter[:] = 10.0 ** mb.dc.s2_de_110.ycenter[:n]
    elif ctl.jbin_type == Jbin_type_sqr:
        ycenter[:] = np.sqrt(mb.dc.s2_de_110.ycenter[:n])
    else:
        raise RuntimeError("error! define jbin_type")

    if ctl.model_intej == model_intej_fast:
        xb = 10.0 ** mb.dc.s2_de_110.xcenter[:n]
        mb_get_dc_mpi_sigma(mb.mc, n, xb, ycenter, s0, s110, s111, s131, s130, s13_1, s330, s310, mb.all.barge, mb.all.n)
    else:
        raise RuntimeError("model_intej:error! define")

    kappa = (4.0 * math.pi * mb.mc) ** 2 * math.log(mb.mbh / mb.mc)
    sigma32 = (2.0 * math.pi * mb.v0 ** 2) ** (-3.0 / 2.0)
    n0 = mb.n0

    for i in range(n):
        for j in range(n):
            cej = get_coeff_ej(float(ycenter[j]),
                               float(s0[i, j]), float(s110[i, j]), float(s111[i, j]), float(s131[i, j]),
                               float(s130[i, j]), float(s13_1[i, j]), float(s330[i, j]), float(s310[i, j]))
            mb.dc.s2_de_110.fxy[i, j] = cej.e_110 * sigma32 * n0 * kappa
            mb.dc.s2_de_0.fxy[i, j] = cej.e_0 * sigma32 * n0 * kappa
            ee = abs(cej.ee) if cej.ee < 0.0 else cej.ee
            mb.dc.s2_dee.fxy[i, j] = ee * sigma32 * n0 * kappa
            mb.dc.s2_dj_111.fxy[i, j] = cej.j_111 * sigma32 * n0 * kappa
            mb.dc.s2_dj_rest.fxy[i, j] = cej.j_rest * sigma32 * n0 * kappa
            jj = abs(cej.jj) if cej.jj < 0.0 else cej.jj
            mb.dc.s2_djj.fxy[i, j] = jj * sigma32 * n0 * kappa
            mb.dc.s2_dej.fxy[i, j] = cej.ej * sigma32 * n0 * kappa


def get_steps_grid(dm: DiffuseMspec, mi: int, dt: S2DType) -> None:
    dt.init(dm.nbin_grid, dm.nbin_grid, dm.emin, dm.emax, dm.jmin, dm.jmax, 0)


def conv_j_space(jin: float, jbin_type: int) -> float:
    if jbin_type == Jbin_type_log:
        return math.log10(jin)
    if jbin_type == Jbin_type_lin:
        return jin
    if jbin_type == Jbin_type_sqr:
        return jin * jin
    raise RuntimeError("define jbin type")


def set_mass_bin_mass_given(dm: DiffuseMspec, masses: np.ndarray, m1: np.ndarray, m2: np.ndarray, asym: np.ndarray, n: int) -> None:
    if dm.mb is None:
        return
    for i in range(int(n)):
        dm.mb[i].mc = float(masses[i])
        dm.mb[i].m1 = float(m1[i])
        dm.mb[i].m2 = float(m2[i])


def set_dm_init(dm: DiffuseMspec) -> None:
    # Import the actual configured ctl from com_main_gw to get real values
    import com_main_gw as cmg
    actual_ctl = cmg.ctl
    
    # Use global values from com_main_gw for physical parameters
    actual_rh = cmg.rh if hasattr(cmg, 'rh') else rh
    actual_mbh = cmg.mbh if hasattr(cmg, 'mbh') else mbh
    actual_log10emin = cmg.log10emin_factor if hasattr(cmg, 'log10emin_factor') else log10emin_factor
    actual_log10emax = cmg.log10emax_factor if hasattr(cmg, 'log10emax_factor') else log10emax_factor
    actual_jmin = cmg.jmin_value if hasattr(cmg, 'jmin_value') else jmin_value
    actual_jmax = cmg.jmax_value if hasattr(cmg, 'jmax_value') else jmax_value
    
    dm.set_diffuse_mspec(
        actual_ctl.grid_bins, actual_ctl.gx_bins,
        actual_log10emin, actual_log10emax,
        actual_jmin, actual_jmax,
        actual_mbh, actual_ctl.v0, actual_ctl.n0, actual_rh, actual_ctl.x_boundary,
        actual_ctl.idx_ref, actual_ctl.jbin_type, actual_ctl.grid_type
    )
    dm.init(actual_ctl.m_bins)
    if actual_ctl.bin_mass is not None and actual_ctl.bin_mass_m1 is not None and actual_ctl.bin_mass_m2 is not None and actual_ctl.asymptot is not None:
        set_mass_bin_mass_given(dm, actual_ctl.bin_mass, actual_ctl.bin_mass_m1, actual_ctl.bin_mass_m2, actual_ctl.asymptot, actual_ctl.m_bins)


def fx_g(x: float) -> float:
    global cct_share, fgx_g0
    alpha = 0.0 if cct_share is None else float(cct_share.alpha)
    if x <= 0.0:
        return float(fgx_g0) * math.exp(x)
    return float(fgx_g0) * (x / float(ctl.x_boundary)) ** (alpha - 1.5)


def fgx_mb_star(x: float) -> float:
    global fc_share, fgx_g0
    if fc_share is None or fc_share.xb is None:
        if x <= 0.0:
            return math.exp(x) * float(fgx_g0)
        return 0.0
    xb = fc_share.xb
    nbin = int(fc_share.nbin)
    if x >= 10.0 ** xb[0] and x <= 10.0 ** xb[nbin - 1]:
        return float(fc_share.get_value_l(math.log10(x)))
    if x <= 0.0:
        return math.exp(x) * float(fgx_g0)
    if x > 10.0 ** xb[nbin - 1] or (x >= 0.0 and x < 10.0 ** xb[0]):
        return 0.0
    return 0.0


def fgx_mb(x: float) -> float:
    global fc_share, fgx_g0
    if fc_share is None or fc_share.xb is None or fc_share.fx is None:
        if x <= 0.0:
            return math.exp(x) * float(fgx_g0)
        return 0.0
    if x > float(ctl.x_boundary) and x <= float(emax_factor):
        return float(np.interp(math.log10(x), fc_share.xb[:fc_share.nbin], fc_share.fx[:fc_share.nbin]))
    if x <= 0.0:
        return math.exp(x) * float(fgx_g0)
    if x > float(emax_factor) or (x <= float(ctl.x_boundary) and x > 0.0):
        return 0.0
    return 0.0


def mb_get_dc_mpi_sigma(mc: float, nbin: int, xb: np.ndarray, yb: np.ndarray,
                        s0: np.ndarray, s110: np.ndarray, s111: np.ndarray, s131: np.ndarray,
                        s130: np.ndarray, s13_1: np.ndarray, s330: np.ndarray, s310: np.ndarray,
                        barge: S1DType, asymp: float) -> None:
    global cct_share, fc_share, fgx_g0
    if ctl.cc is None:
        ctl.cc = [CCT()]
    if len(ctl.cc) == 0:
        ctl.cc = [CCT()]
    cct_share = ctl.cc[0]
    ctl.cc[0].alpha = 7.0 / 4.0
    fc_share = barge
    fgx_g0 = float(asymp)

    if rid == 0:
        print("mc, fgx_g0, bin=", mc, fgx_g0, nbin)

    if fc_share.fx is None or np.all(fc_share.fx == 0.0):
        s0[:, :] = 0.0
        s110[:, :] = 0.0
        s111[:, :] = 0.0
        s131[:, :] = 0.0
        s130[:, :] = 0.0
        s13_1[:, :] = 0.0
        s330[:, :] = 0.0
        s310[:, :] = 0.0
        return

    nbg = int(ctl.nblock_mpi_bg)
    ned = int(ctl.nblock_mpi_ed)
    nblock = int(ctl.nblock_size)
    ntasks = int(ctl.ntasks)

    for i in range(nbg - 1, ned):
        energyx = float(xb[i])
        v = get_sigma0(energyx, fgx_mb, None)
        s0[i, 0] = float(v)
        s0[i, 1:nbin] = s0[i, 0]
        for j in range(nbin):
            jum = float(yb[j])
            a110, a111, a131, a130, a13_1, a330, a310 = get_coeff_sigma_funcs_cfs_grid(energyx, jum, fgx_mb, None, None, None, None, None, None, None)
            s110[i, j] = float(a110)
            s111[i, j] = float(a111)
            s131[i, j] = float(a131)
            s130[i, j] = float(a130)
            s13_1[i, j] = float(a13_1)
            s330[i, j] = float(a330)
            s310[i, j] = float(a310)

    mpi_barrier()
    collect_data_mpi(s0, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s110, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s111, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s131, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s130, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s13_1, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s330, nbin, nbg, ned, nblock, ntasks)
    mpi_barrier()
    collect_data_mpi(s310, nbin, nbg, ned, nblock, ntasks)


def conv_dms_nejw_obj(dm: DiffuseMspec, en: np.ndarray, jm: np.ndarray, m: np.ndarray, w_real: np.ndarray, nobj: int, obj_type: int) -> None:
    if nobj <= 0:
        return
    wobj = w_real[:nobj].astype(float) / float(ctl.ntasks)
    obj_type_star = 1
    obj_type_sbh = 2
    obj_type_wd = 5
    obj_type_ns = 6
    obj_type_bd = 7

    if dm.mb is None:
        return

    for i in range(dm.n):
        mb = dm.mb[i]
        m1 = float(mb.m1)
        m2 = float(mb.m2)
        if obj_type == obj_type_star:
            mb.star.nejw, mb.star.n = get_ejw_from_particle(en[:nobj], jm[:nobj], wobj, m[:nobj], nobj, m1, m2, dms.mbh, dms.v0, dms.x_boundary, mb.star.nejw, mb.star.n)
        elif obj_type == obj_type_sbh:
            mb.sbh.nejw, mb.sbh.n = get_ejw_from_particle(en[:nobj], jm[:nobj], wobj, m[:nobj], nobj, m1, m2, dms.mbh, dms.v0, dms.x_boundary, mb.sbh.nejw, mb.sbh.n)
        elif obj_type == obj_type_wd:
            mb.wd.nejw, mb.wd.n = get_ejw_from_particle(en[:nobj], jm[:nobj], wobj, m[:nobj], nobj, m1, m2, dms.mbh, dms.v0, dms.x_boundary, mb.wd.nejw, mb.wd.n)
        elif obj_type == obj_type_ns:
            mb.ns.nejw, mb.ns.n = get_ejw_from_particle(en[:nobj], jm[:nobj], wobj, m[:nobj], nobj, m1, m2, dms.mbh, dms.v0, dms.x_boundary, mb.ns.nejw, mb.ns.n)
        elif obj_type == obj_type_bd:
            mb.bd.nejw, mb.bd.n = get_ejw_from_particle(en[:nobj], jm[:nobj], wobj, m[:nobj], nobj, m1, m2, dms.mbh, dms.v0, dms.x_boundary, mb.bd.nejw, mb.bd.n)
        else:
            raise RuntimeError("error! define obj_type")


def conv_dms_newj_obj_one(dm: DiffuseMspec, bk: ParticleSamplesArr, i_obj: int) -> None:
    if bk.sp is None or bk.n <= 0:
        return
    en = np.array([bk.sp[k].en for k in range(bk.n)], dtype=float)
    jm = np.array([bk.sp[k].jm for k in range(bk.n)], dtype=float)
    mass = np.array([bk.sp[k].m for k in range(bk.n)], dtype=float)
    weight_real = np.array([bk.sp[k].weight_real for k in range(bk.n)], dtype=float)
    conv_dms_nejw_obj(dm, en, jm, mass, weight_real, bk.n, i_obj)


def conv_dms_nejw(dm: DiffuseMspec, bkstar: ParticleSamplesArr, bksbh: ParticleSamplesArr,
                  bkwd: ParticleSamplesArr, bkns: ParticleSamplesArr, bkbd: ParticleSamplesArr) -> None:
    obj_type_star = 1
    obj_type_sbh = 2
    obj_type_wd = 5
    obj_type_ns = 6
    obj_type_bd = 7

    conv_dms_newj_obj_one(dm, bkstar, obj_type_star)
    conv_dms_newj_obj_one(dm, bksbh, obj_type_sbh)
    conv_dms_newj_obj_one(dm, bkwd, obj_type_wd)
    conv_dms_newj_obj_one(dm, bkns, obj_type_ns)
    conv_dms_newj_obj_one(dm, bkbd, obj_type_bd)

    if dm.mb is None:
        return
    for i in range(dm.n):
        mb = dm.mb[i]
        mb.all.n = mb.star.n + mb.sbh.n + mb.wd.n + mb.ns.n + mb.bd.n


def separate_to_species(bks: ParticleSamplesArr, bkstar: ParticleSamplesArr, bksbh: ParticleSamplesArr,
                        bkwd: ParticleSamplesArr, bkns: ParticleSamplesArr, bkbd: ParticleSamplesArr) -> None:
    separate_to_species_single(bks, bkstar, bksbh, bkwd, bkns, bkbd)


def separate_to_species_single(bks: ParticleSamplesArr, bkstar: ParticleSamplesArr, bksbh: ParticleSamplesArr,
                               bkwd: ParticleSamplesArr, bkns: ParticleSamplesArr, bkbd: ParticleSamplesArr) -> None:
    sams_arr_select_type_single(bks, bkstar, star_type_MS)
    sams_arr_select_type_single(bks, bksbh, star_type_BH)
    sams_arr_select_type_single(bks, bkwd, star_type_WD)
    sams_arr_select_type_single(bks, bkns, star_type_NS)
    sams_arr_select_type_single(bks, bkbd, star_type_BD)


def gen_gx(dm: DiffuseMspec, bkstar: ParticleSamplesArr, bksbh: ParticleSamplesArr,
           bkwd: ParticleSamplesArr, bkns: ParticleSamplesArr, bkbd: ParticleSamplesArr) -> None:
    conv_dms_nejw(dm, bkstar, bksbh, bkwd, bkns, bkbd)
    dm.get_nxj()
    dm.get_fxj0()
    dm.get_barge0()


def get_num_all(so: DmsStellarObject) -> Tuple[float, float]:
    if so.nejw is None or so.n <= 0:
        return 0.0, 0.0
    nb = 0.0
    nbw = 0.0
    lo = math.log10(float(ctl.x_boundary))
    hi = math.log10(float(emax_factor))
    for i in range(int(so.n)):
        rec = so.nejw[i]
        if rec.e > lo and rec.e < hi:
            nbw += float(rec.w)
            nb += 1.0
    return nb, nbw


def get_num_boundary(so: DmsStellarObject) -> Tuple[float, float, float, float]:
    if so.nejw is None or so.n <= 0:
        return 0.0, 0.0, 0.0, 0.0
    nb = 0.0
    nbw = 0.0
    nbj1 = 0.0
    nbj2 = 0.0
    xb_log = math.log10(float(ctl.x_boundary))
    for i in range(int(so.n)):
        rec = so.nejw[i]
        if rec.e < xb_log:
            nbw += float(rec.w)
            nb += 1.0
            if float(rec.j) > 0.5:
                nbj1 += 1.0
            else:
                nbj2 += 1.0
        if rec.e < float(log10emin_factor):
            print("error!??:", rec.e, log10emin_factor)
    return nb, nbj1, nbj2, nbw


def print_num_all(dm: DiffuseMspec) -> None:
    if dm.mb is None or dm.n <= 0:
        print("print_num_all: no mass bins")
        return
    nb = np.zeros((dm.n, 10), dtype=float)
    nbw = np.zeros((dm.n, 10), dtype=float)
    print("print_num_all==========================")
    print("i  mc          star         sbh          bbh          ns           wd           bd")
    for i in range(dm.n):
        b, w = get_num_all(dm.mb[i].star)
        nb[i, 0], nbw[i, 0] = b, w
        b, w = get_num_all(dm.mb[i].sbh)
        nb[i, 1], nbw[i, 1] = b, w
        b, w = get_num_all(dm.mb[i].ns)
        nb[i, 3], nbw[i, 3] = b, w
        b, w = get_num_all(dm.mb[i].wd)
        nb[i, 4], nbw[i, 4] = b, w
        b, w = get_num_all(dm.mb[i].bd)
        nb[i, 5], nbw[i, 5] = b, w
    for i in range(dm.n):
        print(i + 1, dm.mb[i].mc, *nb[i, 0:6])
    print("i  mc          starw        sbhw         bbhw         nsw          wdw          bdw")
    for i in range(dm.n):
        print(i + 1, dm.mb[i].mc, *nbw[i, 0:6])
    print("end of print_num_all===================")


def print_num_boundary(dm: DiffuseMspec) -> None:
    if dm.mb is None or dm.n <= 0:
        print("print_num_boundary: no mass bins")
        return
    nb = np.zeros((dm.n, 10), dtype=float)
    nbw = np.zeros((dm.n, 10), dtype=float)
    nbj1 = np.zeros((dm.n, 10), dtype=float)
    nbj2 = np.zeros((dm.n, 10), dtype=float)
    print("print_num_boundary==========================")
    print("i  mc          star         sbh          bbh          ns           wd           bd")
    for i in range(dm.n):
        b, j1, j2, w = get_num_boundary(dm.mb[i].star)
        nb[i, 0], nbj1[i, 0], nbj2[i, 0], nbw[i, 0] = b, j1, j2, w
        b, j1, j2, w = get_num_boundary(dm.mb[i].sbh)
        nb[i, 1], nbj1[i, 1], nbj2[i, 1], nbw[i, 1] = b, j1, j2, w
        b, j1, j2, w = get_num_boundary(dm.mb[i].ns)
        nb[i, 3], nbj1[i, 3], nbj2[i, 3], nbw[i, 3] = b, j1, j2, w
        b, j1, j2, w = get_num_boundary(dm.mb[i].wd)
        nb[i, 4], nbj1[i, 4], nbj2[i, 4], nbw[i, 4] = b, j1, j2, w
        b, j1, j2, w = get_num_boundary(dm.mb[i].bd)
        nb[i, 5], nbj1[i, 5], nbj2[i, 5], nbw[i, 5] = b, j1, j2, w
    for i in range(dm.n):
        print(i + 1, dm.mb[i].mc, *nb[i, 0:6])
    print("i  mc          starw        sbhw         bbhw         nsw          wdw          bdw")
    for i in range(dm.n):
        print(i + 1, dm.mb[i].mc, *nbw[i, 0:6])
    print("end of print_num_boundary===================")


def get_vr(r: float, a: float, e: float) -> float:
    return 0.0


def get_fden_sample_particle(en: np.ndarray, jum: np.ndarray, w: np.ndarray, n: int, fden: S1DType) -> None:
    a = mbh / (-10.0 ** en[:n] * float(ctl.energy0)) / 2.0
    e = np.sqrt(1.0 - jum[:n] ** 2)
    for i in range(fden.nbin):
        r = 10.0 ** float(fden.xb[i])
        ivr = np.array([get_vr(r, float(a[j]), float(e[j])) for j in range(n)], dtype=float)
        ivrsum = float(np.sum(ivr * w[:n]))
        fden.fx[i] = ivrsum / math.pi / (4.0 * math.pi * 10.0 ** (float(fden.xb[i]) * 2.0))


def update_weights() -> None:
    bcast_dms_asym_weights(dms)
    set_clone_weight(bksams)


def set_weights_for_all_samples(sms_single: Any, sms_arr_single: ParticleSamplesArr) -> None:
    sams_get_weight_clone_single(sms_arr_single)
    set_real_weight_arr_single(sms_arr_single)
    set_clone_weight(sms_single)
    set_real_weight(sms_single)


# Import bksams ChainType instance from com_main_gw (the particle chain)
# Keep local ParticleSamplesArr instances (different type - uses numpy arrays)
from com_main_gw import bksams
bksams_arr = ParticleSamplesArr()
bksams_pointer_arr = None
bksams_arr_norm = ParticleSamplesArr()


def update_arrays_single() -> None:
    all_chain_to_arr_single(bksams, bksams_arr)
    set_sample_arr_indexs_rid_particle(bksams_arr, rid)
    convert_sams_pointer_arr(bksams, bksams_pointer_arr, type=1)
    bksams_arr.select(bksams_arr_norm, exit_normal, -1.0, -1.0)
    bksams_arr.sp = None


def update_density() -> None:
    sms_arr_star = ParticleSamplesArr()
    sms_arr_sbh = ParticleSamplesArr()
    sms_arr_wd = ParticleSamplesArr()
    sms_arr_ns = ParticleSamplesArr()
    sms_arr_bd = ParticleSamplesArr()
    sams_get_weight_clone_single(bksams_arr_norm)
    set_real_weight_arr_single(bksams_arr_norm)
    separate_to_species(bksams_arr_norm, sms_arr_star, sms_arr_sbh, sms_arr_wd, sms_arr_ns, sms_arr_bd)
    gen_gx(dms, sms_arr_star, sms_arr_sbh, sms_arr_wd, sms_arr_ns, sms_arr_bd)
    dms.get_dens0()


def my_integral_none(a: float, b: float, f: Callable[[float], float], n: int = 4096) -> float:
    if b == a:
        return 0.0
    if n < 8:
        n = 8
    if n % 2 == 1:
        n += 1
    xs = np.linspace(a, b, n + 1)
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    h = (b - a) / float(n)
    return float((h / 3.0) * (ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-1:2])))


def get_fna(fden: S1DType, fna: S1DType) -> None:
    for i in range(fna.nbin):
        xb_i = float(fna.xb[i])
        def integrand(x: float) -> float:
            y_out = float(fden.get_value_l(x))
            return y_out * (10.0 ** x) ** 3 * math.log(10.0)
        int_out = my_integral_none(float(fden.xmin), xb_i, integrand)
        fna.fx[i] = 4.0 * math.pi * int_out


def get_fden(gx: S1DType, fden: S1DType, n0: float, v0: float, rh: float, xmin: float) -> None:
    for i in range(fden.nbin):
        xb_i = float(fden.xb[i])
        upper = float(rh) / (10.0 ** xb_i)
        def integrand(x: float) -> float:
            y_out = float(gx.get_value_l(math.log10(x)))
            inside = float(rh) / (10.0 ** xb_i) - x
            return y_out * math.sqrt(abs(inside))
        int_out = my_integral_none(10.0 ** float(xmin), upper, integrand)
        fden.fx[i] = 2.0 / math.sqrt(math.pi) * int_out * float(n0)


def get_den_u(fdenu: S1DType, rmin: float, rmax: float, nbin: int, r0: float, n0: float) -> None:
    fdenu.init(rmin, rmax, nbin, 0)
    fdenu.set_range()
    for i in range(nbin):
        r = 10.0 ** float(fdenu.xb[i])
        phi = float(r0) / r
        if phi < 20.0:
            fdenu.fx[i] = float(n0) * (2.0 / math.sqrt(math.pi) * math.sqrt(phi) + math.exp(phi) * math.erfc(math.sqrt(phi)))
        else:
            fdenu.fx[i] = float(n0) * (2.0 / math.sqrt(math.pi) * math.sqrt(phi) + 1.0 / math.sqrt(math.pi) / math.sqrt(phi) * (1.0 - 1.0 / phi / 2.0))


def dms_so_get_fxj(so: DmsStellarObject, n0: float, mbh: float, v0: float, jbtype: int) -> None:
    """Convert weighted histogram nxj to distribution function gxj.
    
    Replicates dms_so_get_fxj from stellar_obj.f90 lines 145-192.
    """
    if not hasattr(so, 'n') or so.n == 0:
        return
    if not hasattr(so, 'nxj') or not hasattr(so, 'gxj'):
        return
    
    nxj = so.nxj
    gxj = so.gxj
    
    if not hasattr(nxj, 'nxyw') or not hasattr(gxj, 'fxy'):
        return
    
    PI = math.pi
    log10_val = math.log(10.0)
    
    # Match Fortran exactly - lines 152-191
    if jbtype == Jbin_type_lin:
        # Fortran lines 153-163
        for i in range(nxj.nx):
            x = 10.0 ** nxj.xcenter[i]
            for j in range(nxj.ny):
                jm = nxj.ycenter[j]
                # Fortran line 157-159: exact formula
                # Skip if jm is zero to avoid division by zero
                if jm > 0:
                    gxj.fxy[i, j] = (nxj.nxyw[i, j] / (x * log10_val) / nxj.xstep / nxj.ystep *
                                    PI ** (-1.5) * v0 ** 6 * x ** 2.5 / jm / n0 / mbh ** 3)
                else:
                    gxj.fxy[i, j] = 0.0
    
    elif jbtype == Jbin_type_log:
        # Fortran lines 164-175
        for i in range(nxj.nx):
            x = 10.0 ** nxj.xcenter[i]
            for j in range(nxj.ny):
                jm = 10.0 ** nxj.ycenter[j]
                # Fortran line 169-171: exact formula
                if jm > 0:
                    gxj.fxy[i, j] = (nxj.nxyw[i, j] / (x * log10_val) / nxj.xstep / nxj.ystep *
                                    PI ** (-1.5) * v0 ** 6 * x ** 2.5 / (jm ** 2 * log10_val) / n0 / mbh ** 3)
                else:
                    gxj.fxy[i, j] = 0.0
    
    elif jbtype == Jbin_type_sqr:
        # Fortran lines 176-187
        for i in range(nxj.nx):
            x = 10.0 ** nxj.xcenter[i]
            for j in range(nxj.ny):
                jm = nxj.ycenter[j] ** 0.5
                # Fortran line 181-183: exact formula
                gxj.fxy[i, j] = (nxj.nxyw[i, j] / (x * log10_val) / nxj.xstep / nxj.ystep *
                                PI ** (-1.5) * v0 ** 6 * x ** 2.5 / 2.0 / n0 / mbh ** 3)
    
    # Debug output
    nxyw_sum = np.sum(nxj.nxyw)
    gxj_sum = np.sum(gxj.fxy)
    print(f"DEBUG dms_so_get_fxj: so.n={so.n}, sum(nxyw)={nxyw_sum:.3e}, sum(gxj.fxy)={gxj_sum:.3e}")


def get_fxj(mb: MassBins, jbtype: int) -> None:
    dms_so_get_fxj(mb.all, mb.n0, mb.mbh, mb.v0, jbtype)
    for so in (mb.star, mb.sbh, mb.ns, mb.wd, mb.bd):
        dms_so_get_fxj(so, mb.n0, mb.mbh, mb.v0, jbtype)
