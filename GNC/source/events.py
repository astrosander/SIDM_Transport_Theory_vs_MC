import math
import pickle
from dataclasses import dataclass, field
from typing import BinaryIO, Optional, Union

import numpy as np


def farravg(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))


def farrsct(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.std(a, ddof=0))


@dataclass
class EventRateType:
    ntot_eve_simu_w: Optional[np.ndarray] = None
    ntot_eve_simu: Optional[np.ndarray] = None
    eve_rate: Optional[np.ndarray] = None
    t_snap: Optional[np.ndarray] = None
    p_eve: Optional[np.ndarray] = None
    dt: float = 0.0
    numavg_flyby: Optional[np.ndarray] = None
    numavg_kl: Optional[np.ndarray] = None
    numavg_encnt: Optional[np.ndarray] = None
    nsp: int = 0

    def init_event(self, tspan: float, nsp: int) -> None:
        self.nsp = int(nsp)
        n = self.nsp
        self.ntot_eve_simu_w = np.zeros(n, dtype=float)
        self.ntot_eve_simu = np.zeros(n, dtype=float)
        self.eve_rate = np.zeros(n, dtype=float)
        self.p_eve = np.zeros(n, dtype=float)
        self.numavg_flyby = np.zeros(n, dtype=float)
        self.numavg_kl = np.zeros(n, dtype=float)
        self.numavg_encnt = np.zeros(n, dtype=float)
        self.t_snap = (np.arange(1, n + 1, dtype=float) * (float(tspan) / float(n)))
        self.dt = float(tspan) / float(n)

    def write_sample(self, f: BinaryIO) -> None:
        payload = {
            "ntot_eve_simu_w": self.ntot_eve_simu_w,
            "t_snap": self.t_snap,
            "ntot_eve_simu": self.ntot_eve_simu,
            "eve_rate": self.eve_rate,
            "p_eve": self.p_eve,
            "numavg_flyby": self.numavg_flyby,
            "numavg_kl": self.numavg_kl,
            "numavg_encnt": self.numavg_encnt,
            "dt": self.dt,
            "nsp": self.nsp,
        }
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_sample(self, f: BinaryIO) -> None:
        payload = pickle.load(f)
        self.ntot_eve_simu_w = payload["ntot_eve_simu_w"]
        self.t_snap = payload["t_snap"]
        self.ntot_eve_simu = payload["ntot_eve_simu"]
        self.eve_rate = payload["eve_rate"]
        self.p_eve = payload["p_eve"]
        self.numavg_flyby = payload["numavg_flyby"]
        self.numavg_kl = payload["numavg_kl"]
        self.numavg_encnt = payload["numavg_encnt"]
        self.dt = float(payload["dt"])
        self.nsp = int(payload["nsp"])


@dataclass
class EventSetsStar:
    nsp: int = 0
    tot_time: float = 0.0
    etot: EventRateType = field(default_factory=EventRateType)
    enorm: EventRateType = field(default_factory=EventRateType)
    enorm_withinbd: EventRateType = field(default_factory=EventRateType)
    etd: EventRateType = field(default_factory=EventRateType)
    etdfull: EventRateType = field(default_factory=EventRateType)
    etdempty: EventRateType = field(default_factory=EventRateType)
    eemin: EventRateType = field(default_factory=EventRateType)
    eemax: EventRateType = field(default_factory=EventRateType)
    emaxsteps: EventRateType = field(default_factory=EventRateType)

    def init(self, tot_time: float, nsap: int) -> None:
        self.tot_time = float(tot_time)
        self.nsp = int(nsap)
        self.etot.init_event(self.tot_time, self.nsp)
        self.enorm.init_event(self.tot_time, self.nsp)
        self.enorm_withinbd.init_event(self.tot_time, self.nsp)
        self.etd.init_event(self.tot_time, self.nsp)
        self.etdfull.init_event(self.tot_time, self.nsp)
        self.etdempty.init_event(self.tot_time, self.nsp)
        self.emaxsteps.init_event(self.tot_time, self.nsp)
        self.eemin.init_event(self.tot_time, self.nsp)
        self.eemax.init_event(self.tot_time, self.nsp)


@dataclass
class EventSetsComp(EventSetsStar):
    egw: EventRateType = field(default_factory=EventRateType)
    egw_emri: EventRateType = field(default_factory=EventRateType)
    eother: EventRateType = field(default_factory=EventRateType)
    egw_plunge: EventRateType = field(default_factory=EventRateType)

    def init(self, tot_time: float, nsap: int) -> None:
        super().init(tot_time, nsap)
        self.egw.init_event(self.tot_time, self.nsp)
        self.egw_emri.init_event(self.tot_time, self.nsp)
        self.egw_plunge.init_event(self.tot_time, self.nsp)
        self.eother.init_event(self.tot_time, self.nsp)
        self.egw_plunge.init_event(self.tot_time, self.nsp)


def print_events(et: EventRateType, enm: EventRateType, j: int) -> None:
    jj = int(j) - 1
    print("isnap=", j)
    print("Ntot, Ntotw=", float(et.ntot_eve_simu[jj]), float(et.ntot_eve_simu_w[jj]))
    print("Nnorm, Nnormw=", float(enm.ntot_eve_simu[jj]), float(enm.ntot_eve_simu_w[jj]))


def output_eveset_avgscatter_rate(eveset: Union[EventSetsStar, EventSetsComp], fn: str, frac: float) -> None:
    nsp = int(eveset.nsp)
    nbg = int(float(nsp) * float(frac))
    ned = nsp
    nar = ned - nbg + 1
    i0 = max(0, nbg - 1)
    i1 = ned
    with open(f"{fn}_avgrates.txt", "w") as f:
        if isinstance(eveset, EventSetsComp):
            f.write(f'{"Rate_plunge(Gyr-1)":>25}{"S_plunge":>25}\n')
            a = eveset.egw_plunge.eve_rate[i0:i1]
            f.write(f"{farravg(a):25.10f}{farrsct(a):25.10f}\n")
        else:
            f.write(
                f'{"Rate_td(Gyr-1)":>25}{"S_td":>25}{"R_tdfull":>25}{"S_tdfull":>25}'
                f'{"R_tdempty":>25}{"s_tmpfull":>25}{"R_emax":>25}{"s_emax":>25}\n'
            )
            a0 = eveset.etd.eve_rate[i0:i1]
            a1 = eveset.etdfull.eve_rate[i0:i1]
            a2 = eveset.etdempty.eve_rate[i0:i1]
            a3 = eveset.eemax.eve_rate[i0:i1]
            f.write(
                f"{farravg(a0):25.10f}{farrsct(a0):25.10f}"
                f"{farravg(a1):25.10f}{farrsct(a1):25.10f}"
                f"{farravg(a2):25.10f}{farrsct(a2):25.10f}"
                f"{farravg(a3):25.10f}{farrsct(a3):25.10f}\n"
            )


def output_eveset_rate(eveset: Union[EventSetsStar, EventSetsComp], fn: str) -> None:
    with open(f"{fn}_rates.txt", "w") as f:
        if isinstance(eveset, EventSetsComp):
            f.write(
                f'{"Tsnap(Myr)":>20}{"Rate_td(Gyr-1)":>20}{"R_tdfull":>20}'
                f'{"R_tmpfull":>20}{"R_emax":>20}{"R_plunge":>20}\n'
            )
            for i in range(eveset.nsp):
                f.write(
                    f"{float(eveset.etot.t_snap[i]):20.10f}"
                    f"{float(eveset.etd.eve_rate[i]):20.10f}"
                    f"{float(eveset.etdfull.eve_rate[i]):20.10f}"
                    f"{float(eveset.etdempty.eve_rate[i]):20.10f}"
                    f"{float(eveset.eemax.eve_rate[i]):20.10f}"
                    f"{float(eveset.egw_plunge.eve_rate[i]):20.10f}\n"
                )
        else:
            f.write(
                f'{"Tsnap(Myr)":>25}{"Rate_td(Gyr-1)":>25}{"R_tdfull":>25}{"R_tmpfull":>25}{"R_emax":>25}\n'
            )
            for i in range(eveset.nsp):
                f.write(
                    f"{float(eveset.etot.t_snap[i]):20.10f}"
                    f"{float(eveset.etd.eve_rate[i]):20.10f}"
                    f"{float(eveset.etdfull.eve_rate[i]):20.10f}"
                    f"{float(eveset.etdempty.eve_rate[i]):20.10f}"
                    f"{float(eveset.eemax.eve_rate[i]):20.10f}\n"
                )


def output_eveset_N(eveset: Union[EventSetsStar, EventSetsComp], fout: str, flag: int) -> None:
    with open(f"{fout}_event_Nweight.txt", "w") as f:
        if isinstance(eveset, EventSetsComp):
            if int(flag) == 1:
                f.write(f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_emax":>25}{"N_plunge":>25}\n')
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.egw_plunge.ntot_eve_simu_w[i]):25.10f}\n"
                    )
            else:
                f.write(
                    f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_td":>25}{"N_tdfull":>25}'
                    f'{"N_tmpfull":>25}{"N_emax":>25}{"N_plunge":>25}\n'
                )
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.etd.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.etdfull.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.etdempty.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.egw_plunge.ntot_eve_simu_w[i]):25.10f}\n"
                    )
        else:
            if int(flag) == 1:
                f.write(f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_emax":>25}\n')
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu_w[i]):25.10f}\n"
                    )
            else:
                f.write(
                    f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_td":>25}{"N_tdfull":>25}'
                    f'{"N_tmpfull":>25}{"N_emax":>25}\n'
                )
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.etd.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.etdfull.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.etdempty.ntot_eve_simu_w[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu_w[i]):25.10f}\n"
                    )

    with open(f"{fout}_event_N.txt", "w") as f:
        if isinstance(eveset, EventSetsComp):
            if int(flag) == 1:
                f.write(f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_emax":>25}{"N_plunge":>25}\n')
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.egw_plunge.ntot_eve_simu[i]):25.10f}\n"
                    )
            else:
                f.write(
                    f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_td":>25}{"N_tdfull":>25}'
                    f'{"N_tdempty":>25}{"N_emax":>25}{"N_plunge":>25}\n'
                )
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.etd.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.etdfull.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.etdempty.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.egw_plunge.ntot_eve_simu[i]):25.10f}\n"
                    )
        else:
            if int(flag) == 1:
                f.write(f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_emax":>25}\n')
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu[i]):25.10f}\n"
                    )
            else:
                f.write(
                    f'{"Tsnap":>25}{"N_all":>25}{"N_norm":>25}{"N_norm_bd":>25}{"N_td":>25}{"N_tdfull":>25}'
                    f'{"N_tdempty":>25}{"N_emax":>25}\n'
                )
                for i in range(eveset.nsp):
                    f.write(
                        f"{float(eveset.etot.t_snap[i]):25.10f}"
                        f"{float(eveset.etot.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.enorm_withinbd.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.etd.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.etdfull.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.etdempty.ntot_eve_simu[i]):25.10f}"
                        f"{float(eveset.eemax.ntot_eve_simu[i]):25.10f}\n"
                    )


def output_eveset_txt(eveset: Union[EventSetsStar, EventSetsComp], fout: str, flag: int) -> None:
    output_eveset_N(eveset, fout, int(flag))
    if int(flag) > 1:
        output_eveset_rate(eveset, fout)
        output_eveset_avgscatter_rate(eveset, fout, 0.5)


frac_nbh: float = 0.0
frac_mbh: float = 0.0
frac_rate_supply: float = 0.0

pteve_star = EventSetsStar()
pteve_sbh = EventSetsComp()
pteve_wd = EventSetsComp()
pteve_ns = EventSetsComp()
pteve_bd = EventSetsComp()
