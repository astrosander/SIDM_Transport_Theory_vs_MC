from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional


Jbin_type_lin = 1
Jbin_type_log = 2
Jbin_type_sqr = 3
sts_type_dstr = 1


@dataclass
class NeJW:
    e: float
    j: float
    w: float
    idx: int = 0


@dataclass
class DmsStellarObject:
    n: int = 0
    n_real: float = 0.0
    nejw: List[NeJW] = field(default_factory=list)
    barge: object = None
    nxj: object = None
    gxj: object = None
    fden_simu: object = None
    fden: object = None
    fNa: object = None
    fMa: object = None
    asymp: float = 0.0

    def init(self, nbin_grid: int, nbin_gx: int, emin: float, emax: float, tmin: float, tmax: float, rh: float, jb_type: int):
        rmin = math.log10(0.5 * rh / (10.0 ** emax))
        rmax = math.log10(0.5 * rh / (10.0 ** emin))

        self.barge.init(emin, emax, nbin_gx, sts_type_dstr)
        self.barge.set_range()

        self.fden.init(rmin, rmax, nbin_gx, sts_type_dstr)
        self.fden.set_range()
        self.fden_simu.init(rmin, rmax, nbin_gx, sts_type_dstr)
        self.fden_simu.set_range()

        self.fNa.init(rmin, rmax, nbin_gx, sts_type_dstr)
        self.fNa.set_range()
        self.fMa.init(rmin, rmax, nbin_gx, sts_type_dstr)
        self.fMa.set_range()

        self.gxj.init(nbin_gx, nbin_gx, emin, emax, tmin, tmax, sts_type_dstr)
        self.gxj.set_range()
        self.nxj.init(nbin_gx, nbin_gx, emin, emax, tmin, tmax, use_weight=True)
        self.nxj.set_range()

        self.n = 0
        self.n_real = 0.0

    def deallocation(self):
        self.fden.deallocate()
        self.fNa.deallocate()
        self.fMa.deallocate()
        self.barge.deallocate()

    def dms_so_get_nxj_from_nejw(self, jbtype: int):
        if self.n <= 0:
            self.n_real = 0.0
            return

        en = [x.e for x in self.nejw[: self.n]]
        jm = [x.j for x in self.nejw[: self.n]]
        we = [x.w for x in self.nejw[: self.n]]

        if jbtype == Jbin_type_lin:
            self.nxj.get_stats_weight(en, jm, we, self.n)
        elif jbtype == Jbin_type_log:
            self.nxj.get_stats_weight(en, [math.log10(v) for v in jm], we, self.n)
        elif jbtype == Jbin_type_sqr:
            self.nxj.get_stats_weight(en, [v * v for v in jm], we, self.n)
        else:
            raise RuntimeError(f"dms_nxj_newj:error! define jbtype {jbtype}")

        self.n_real = sum(we)

    def get_asymp_norm_factor_one(self, x_boundary: float) -> float:
        if self.n_real <= 0.0:
            raise RuntimeError(f"error! dso%n_real={self.n_real}")

        cnorm = self.barge.get_value_l(math.log10(x_boundary))
        if (cnorm == 0.0) or math.isnan(cnorm):
            print("star:cnorm is nan or 0", cnorm)
            self.barge.print("dso%barge")
            raise RuntimeError("cnorm invalid")

        return self.asymp / cnorm

    def normalize_barge_one(self, norm: float):
        self.barge.fx = self.barge.fx * norm

    def normalize_gxj_one(self, norm: float):
        self.gxj.fxy = self.gxj.fxy * norm

    def dms_so_get_fxj(self, n0: float, mbh: float, v0: float, jbtype: int):
        if self.n == 0:
            return

        pi = math.pi
        log10 = math.log(10.0)

        nx = self.nxj.nx
        ny = self.nxj.ny

        for i in range(nx):
            x = 10.0 ** self.nxj.xcenter[i]
            for j in range(ny):
                if jbtype == Jbin_type_lin:
                    jm = self.nxj.ycenter[j]
                    denom_j = jm
                    denom_extra = 1.0
                elif jbtype == Jbin_type_log:
                    jm = 10.0 ** self.nxj.ycenter[j]
                    denom_j = jm * jm
                    denom_extra = log10
                elif jbtype == Jbin_type_sqr:
                    jm = self.nxj.ycenter[j] ** 0.5
                    denom_j = 2.0
                    denom_extra = 1.0
                else:
                    raise RuntimeError("fxj error!")

                self.gxj.fxy[i][j] = (
                    self.nxj.nxyw[i][j]
                    / (x * log10)
                    / self.nxj.xstep
                    / self.nxj.ystep
                    * (pi ** (-1.5))
                    * (v0 ** 6)
                    * (x ** 2.5)
                    / denom_j
                    / denom_extra
                    / n0
                    / (mbh ** 3)
                )

    def get_barge_stellar(self, jbtype: int):
        if self.n == 0:
            return

        if self.barge.nbin != self.gxj.nx:
            raise RuntimeError("error! barge%nbin should = gxj%nx")

        log10 = math.log(10.0)

        for i in range(self.barge.nbin):
            int_out = 0.0
            self.barge.xb[i] = self.gxj.xcenter[i]

            if jbtype == Jbin_type_lin:
                for j in range(self.gxj.ny):
                    int_out += self.gxj.fxy[i][j] * self.gxj.ycenter[j] * self.gxj.ystep * 2.0
                self.barge.fx[i] = int_out
            elif jbtype == Jbin_type_log:
                for j in range(self.gxj.ny):
                    int_out += self.gxj.fxy[i][j] * (10.0 ** self.gxj.ycenter[j]) ** 2 * self.gxj.ystep * 2.0 * log10
                self.barge.fx[i] = int_out
            elif jbtype == Jbin_type_sqr:
                for j in range(self.gxj.ny):
                    int_out += self.gxj.fxy[i][j] * self.gxj.ycenter[j] * self.gxj.ystep * 2.0
                self.barge.fx[i] = int_out
            else:
                raise RuntimeError(f"dms_nxj_newj:error! define jbtype {jbtype}")

            if math.isnan(self.barge.fx[i]):
                print("get_barge_stellar:fx is NaN:", int_out)
                self.gxj.print()
                raise RuntimeError("barge fx is NaN")

    def get_dens(self, n0: float, v0: float, rh: float, emin: float, weight_asym: float):
        if self.n > 0:
            get_fden(self.barge, self.fden, n0, v0, rh, emin)

            en = [x.e for x in self.nejw[: self.n]]
            jm = [x.j for x in self.nejw[: self.n]]
            we = [x.w for x in self.nejw[: self.n]]

            get_fden_sample_particle(en, jm, we, self.n, self.fden_simu)
            get_fna(self.fden_simu, self.fNa)
        else:
            self.fden_simu.fx = 0.0
