import pickle
import numpy as np


f_log = 0
f_linear = 1
dct_x = 1
dct_y = 2

sts_type_grid = 0
sts_type_dstr = 1


def get_dstr_num_in_each_bin(x, n: int, xbg: float, xstep: float, nbin: int, nb_out=None):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    idx = np.floor((xv - xbg) / xstep).astype(np.int64) + 1
    mask = (idx >= 1) & (idx <= nbin)
    nb = np.zeros((nbin,), dtype=np.int64) if nb_out is None else nb_out
    nb[:] = 0
    if np.any(mask):
        ii = idx[mask] - 1
        np.add.at(nb, ii, 1)
    ns = int(np.sum(mask))
    return nb, ns


def get_dstr_num_in_each_bin_weight(x, w, n: int, xbg: float, xstep: float, nbin: int, nbw_out=None):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    wv = np.asarray(w, dtype=np.float64).reshape((n,))
    idx = np.floor((xv - xbg) / xstep).astype(np.int64) + 1
    mask = (idx >= 1) & (idx <= nbin)
    nbw = np.zeros((nbin,), dtype=np.float64) if nbw_out is None else nbw_out
    nbw[:] = 0.0
    if np.any(mask):
        ii = idx[mask] - 1
        np.add.at(nbw, ii, wv[mask])
    nsw = float(np.sum(wv[mask])) if np.any(mask) else 0.0
    return nbw, nsw


def init_s1d_basic(obj, xmin: float, xmax: float, n: int, bin_type: int):
    obj.nbin = int(n)
    obj.xmin = float(xmin)
    obj.xmax = float(xmax)
    obj.bin_type = int(bin_type)
    obj.xstep = (obj.xmax - obj.xmin) / float(obj.nbin) if obj.nbin > 0 else 0.0
    obj.xb = np.zeros((obj.nbin,), dtype=np.float64)
    obj.fx = np.zeros((obj.nbin,), dtype=np.float64)
    if obj.nbin > 0:
        obj.xb[:] = obj.xmin + (np.arange(obj.nbin, dtype=np.float64) + 0.5) * obj.xstep


class S1DBasicType:
    def __init__(self):
        self.nbin = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.xstep = 0.0
        self.bin_type = sts_type_dstr
        self.xb = None
        self.fx = None

    def deallocate(self):
        self.xb = None
        self.fx = None
        self.nbin = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.xstep = 0.0


class S1DHstBasicType(S1DBasicType):
    def __init__(self):
        super().__init__()
        self.nb = None
        self.fxw = None
        self.nbw = None
        self.nsw = 0.0
        self.ns = 0
        self.use_weight = False

    def init_s1_hst_basic(self, xmin: float, xmax: float, n: int, use_weight: bool = False):
        init_s1d_basic(self, xmin, xmax, n, sts_type_dstr)
        self.use_weight = bool(use_weight)
        self.nb = np.zeros((self.nbin,), dtype=np.int64)
        self.ns = 0
        if self.use_weight:
            self.nbw = np.zeros((self.nbin,), dtype=np.float64)
            self.fxw = np.zeros((self.nbin,), dtype=np.float64)
            self.nsw = 0.0
        else:
            self.nbw = np.zeros((self.nbin,), dtype=np.float64)
            self.fxw = np.zeros((self.nbin,), dtype=np.float64)
            self.nsw = 0.0
        return self

    def print(self, str_: str | None = None):
        if str_ is not None:
            print("for sts s1d=", str(str_).strip())
        str_bin_type = "GRID" if self.bin_type == sts_type_grid else "DSTR"
        if self.use_weight:
            print(f"{'bin_type':>15}{'nbin':>15}{'xmin':>15}{'xmax':>15}{'xstep':>15}{'NS':>15}{'NSW':>15}")
            print(f"{str_bin_type:>15}{self.nbin:15d}{self.xmin:15.5e}{self.xmax:15.5e}{self.xstep:15.5e}{self.ns:15d}{self.nsw:15.5e}")
            print(f"{'X':>20}{'FX':>20}{'FXW':>20}{'NB':>20}{'NBW':>20}")
            for i in range(self.nbin):
                print(f"{self.xb[i]:20.10e}{self.fx[i]:20.10e}{self.fxw[i]:20.10e}{int(self.nb[i]):20d}{self.nbw[i]:20.10e}")
        else:
            print(f"{'bin_type':>15}{'nbin':>15}{'xmin':>15}{'xmax':>15}{'xstep':>15}{'NS':>15}")
            print(f"{str_bin_type:>15}{self.nbin:15d}{self.xmin:15.5e}{self.xmax:15.5e}{self.xstep:15.5e}{self.ns:15d}")
            print(f"{'X':>20}{'FX':>20}{'NB':>20}")
            for i in range(self.nbin):
                print(f"{self.xb[i]:20.10e}{self.fx[i]:20.10e}{int(self.nb[i]):20d}")

    def get_s1d_hst(self, x, n: int):
        if self.use_weight:
            pass
        nb, ns = get_dstr_num_in_each_bin(np.asarray(x)[:n], n, self.xmin, self.xstep, self.nbin, self.nb)
        self.nb[:] = nb
        self.ns = int(ns)
        self.fx[:] = self.nb.astype(np.float64) / self.xstep if self.xstep != 0.0 else 0.0

    def get_s1d_hst_weight(self, x, w, n: int):
        if not self.use_weight:
            raise ValueError("error! s1_hst.use_weight=False")
        nbw, nsw = get_dstr_num_in_each_bin_weight(np.asarray(x)[:n], np.asarray(w)[:n], n, self.xmin, self.xstep, self.nbin, self.nbw)
        self.nbw[:] = nbw
        self.nsw = float(nsw)
        self.fxw[:] = self.nbw / self.xstep if self.xstep != 0.0 else 0.0

        nb, ns = get_dstr_num_in_each_bin(np.asarray(x)[:n], n, self.xmin, self.xstep, self.nbin, self.nb)
        self.nb[:] = nb
        self.ns = int(ns)
        self.fx[:] = self.nb.astype(np.float64) / self.xstep if self.xstep != 0.0 else 0.0

    def get_hst(self, x, w=None, n: int | None = None):
        if n is None:
            n = len(x)
        if w is None:
            return self.get_s1d_hst(x, int(n))
        return self.get_s1d_hst_weight(x, w, int(n))


def output_s1d_hst(s1d_hst: S1DHstBasicType, fn: str):
    fn = fn.strip()
    with open(f"{fn}_s1d_hst.txt", "w") as f:
        if s1d_hst.use_weight:
            f.write(f"{'xb':>27}{'fx':>27}{'fxw':>27}{'nb':>27}{'nbw':>27}\n")
            for i in range(s1d_hst.nbin):
                f.write(
                    f"{s1d_hst.xb[i]:27.12e} {s1d_hst.fx[i]:27.12e} {s1d_hst.fxw[i]:27.12e} "
                    f"{int(s1d_hst.nb[i]):27d} {s1d_hst.nbw[i]:27.12e}\n"
                )
        else:
            f.write(f"{'xb':>27}{'fx':>27}{'nb':>27}\n")
            for i in range(s1d_hst.nbin):
                f.write(f"{s1d_hst.xb[i]:27.12e} {s1d_hst.fx[i]:27.12e} {int(s1d_hst.nb[i]):27d}\n")


class S1DHstType(S1DHstBasicType):
    def __init__(self):
        super().__init__()

    def init(self, xmin: float, xmax: float, n: int, use_weight: bool = False):
        return self.init_s1_hst_basic(xmin, xmax, n, use_weight)

    def write_s1d_hst(self, file_unit):
        payload = {
            "nbin": self.nbin,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "use_weight": bool(self.use_weight),
            "xb": None if self.xb is None else self.xb.copy(),
            "fx": None if self.fx is None else self.fx.copy(),
            "nb": None if self.nb is None else self.nb.copy(),
            "ns": int(self.ns),
            "nbw": None if self.nbw is None else self.nbw.copy(),
            "fxw": None if self.fxw is None else self.fxw.copy(),
            "nsw": float(self.nsw),
        }
        pickle.dump(payload, file_unit, protocol=pickle.HIGHEST_PROTOCOL)

    def read_s1d_hst(self, file_unit):
        d = pickle.load(file_unit)
        self.init(d["xmin"], d["xmax"], int(d["nbin"]), bool(d["use_weight"]))
        self.xb[:] = d["xb"]
        self.fx[:] = d["fx"]
        self.nb[:] = d["nb"]
        self.ns = int(d["ns"])
        if self.use_weight:
            self.nbw[:] = d["nbw"]
            self.fxw[:] = d["fxw"]
            self.nsw = float(d["nsw"])
