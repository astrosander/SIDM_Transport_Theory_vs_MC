from __future__ import annotations
import os
import math
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from mpi4py import MPI
except Exception:
    MPI = None

from model_basic import ctl

rid: int = 0
proc_id: int = 0
mpi_master_id: int = 0

MPI_COMMAND_END: int = -1
MPI_COMMAND_comu_sent: int = 4
MPI_COMMAND_comu_recv: int = 5
mpi_command_own: int = 6

nint_particle: int = 10
nreal_particle: int = 19

nint_by: int = 10
nreal_by: int = 41
nstr_by: int = 100


def init_mpi() -> None:
    global rid, proc_id
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    rid = comm.Get_rank()
    ctl.ntasks = comm.Get_size()
    proc_id = os.getpid()
    if getattr(ctl, "chattery", 0) > 4:
        print("mpi initializtion finished")


def stop_mpi() -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    MPI.Finalize()
    print("mpi finalized", rid)


def collect_data_mpi(
    fxy: np.ndarray,
    nbin: int,
    nbg: int,
    ned: int,
    nblocks: int,
    ntasks: int,
) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    if nbin % ntasks != 0:
        raise RuntimeError("error in collect data mpi!: ntasks is not times of nbin")

    nd_tot = nbin * nbin
    recvbuffer = np.zeros(nd_tot, dtype=np.float64)
    sendbuffer = np.zeros(nd_tot, dtype=np.float64)

    sent_count = nbin * nblocks

    for i in range(1, ntasks + 1):
        for j in range(nbg, ned + 1):
            start = (i - 1) * sent_count + (j - nbg) * nbin
            end = start + nbin
            sendbuffer[start:end] = fxy[j - 1, 0:nbin]

    send2 = sendbuffer.reshape((ntasks, sent_count))
    recv2 = np.empty_like(send2)
    comm.Alltoall(send2, recv2)
    recvbuffer[:] = recv2.ravel()

    for i in range(1, nbin + 1):
        start = (i - 1) * nbin
        end = i * nbin
        fxy[i - 1, 0:nbin] = recvbuffer[start:end]


def collect_to_root_sps_single(sps_send: Any, sps: List[Any], n: int) -> None:
    for i in range(1, ctl.ntasks + 1):
        if i != mpi_master_id + 1:
            send_particle_sample_arr_mpi(sps_send, sps[i - 1], i - 1, mpi_master_id)
        else:
            sps[mpi_master_id] = sps_send


def bcast_dms_barge(dm: Any) -> None:
    for i in range(1, dm.n + 1):
        bcast_s1d_mpi(dm.mb[i - 1].all.barge)
        bcast_s1d_mpi(dm.mb[i - 1].sbh.barge)
        bcast_s1d_mpi(dm.mb[i - 1].star.barge)
        bcast_s1d_mpi(dm.mb[i - 1].bbh.barge)
    bcast_s1d_mpi(dm.all.all.barge)


def bcast_dms_fden(dm: Any) -> None:
    for i in range(1, dm.n + 1):
        bcast_s1d_mpi(dm.mb[i - 1].all.fden)
        bcast_s1d_mpi(dm.mb[i - 1].all.fden_simu)
    bcast_s1d_mpi(dm.all.all.fden)
    bcast_s1d_mpi(dm.all.all.fden_simu)


def send_particle_sample_arr_mpi(
    sps_send: Any,
    sps_recv: Any,
    proc_id_source: int,
    proc_id_dest: int,
) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD

    if rid == proc_id_source:
        n = int(sps_send.n)

        intarr_sp = np.zeros((nint_particle, n), dtype=np.int32)
        realarr_sp = np.zeros((nreal_particle, n), dtype=np.float64)

        intarr_by = np.zeros((nint_by, n), dtype=np.int32)
        realarr_by = np.zeros((nreal_by, n), dtype=np.float64)
        strarr_by = np.zeros((n,), dtype=f"S{nstr_by}")

        for i in range(1, n + 1):
            ia, ra = conv_sp_int_real_arrays(sps_send.sp[i - 1])
            intarr_sp[:, i - 1] = ia
            realarr_sp[:, i - 1] = ra

        comm.send(n, dest=proc_id_dest, tag=0)
        comm.Send([intarr_sp, MPI.INT], dest=proc_id_dest, tag=0)
        comm.Send([realarr_sp, MPI.DOUBLE], dest=proc_id_dest, tag=0)

        for i in range(1, n + 1):
            ia, ra, sa = conv_by_arrays(sps_send.sp[i - 1].byot)
            intarr_by[:, i - 1] = ia
            realarr_by[:, i - 1] = ra
            strarr_by[i - 1] = np.asarray(sa.encode("utf-8")[:nstr_by], dtype=f"S{nstr_by}")

        comm.Send([intarr_by, MPI.INT], dest=proc_id_dest, tag=0)
        comm.Send([realarr_by, MPI.DOUBLE], dest=proc_id_dest, tag=0)
        comm.Send([strarr_by, MPI.CHAR], dest=proc_id_dest, tag=0)

        for i in range(1, n + 1):
            ia, ra, sa = conv_by_arrays(sps_send.sp[i - 1].byot_ini)
            intarr_by[:, i - 1] = ia
            realarr_by[:, i - 1] = ra
            strarr_by[i - 1] = np.asarray(sa.encode("utf-8")[:nstr_by], dtype=f"S{nstr_by}")

        comm.Send([intarr_by, MPI.INT], dest=proc_id_dest, tag=0)
        comm.Send([realarr_by, MPI.DOUBLE], dest=proc_id_dest, tag=0)
        comm.Send([strarr_by, MPI.CHAR], dest=proc_id_dest, tag=0)

        for i in range(1, n + 1):
            ia, ra, sa = conv_by_arrays(sps_send.sp[i - 1].byot_bf)
            intarr_by[:, i - 1] = ia
            realarr_by[:, i - 1] = ra
            strarr_by[i - 1] = np.asarray(sa.encode("utf-8")[:nstr_by], dtype=f"S{nstr_by}")

        comm.Send([intarr_by, MPI.INT], dest=proc_id_dest, tag=0)
        comm.Send([realarr_by, MPI.DOUBLE], dest=proc_id_dest, tag=0)
        comm.Send([strarr_by, MPI.CHAR], dest=proc_id_dest, tag=0)

    elif rid == proc_id_dest:
        n = int(comm.recv(source=proc_id_source, tag=0))
        sps_recv.n = n
        if hasattr(sps_recv, "init"):
            sps_recv.init(n)
        else:
            sps_recv.sp = [None] * n

        intarr_sp = np.zeros((nint_particle, n), dtype=np.int32)
        realarr_sp = np.zeros((nreal_particle, n), dtype=np.float64)

        intarr_by = np.zeros((nint_by, n), dtype=np.int32)
        realarr_by = np.zeros((nreal_by, n), dtype=np.float64)
        strarr_by = np.zeros((n,), dtype=f"S{nstr_by}")

        comm.Recv([intarr_sp, MPI.INT], source=proc_id_source, tag=0)
        comm.Recv([realarr_sp, MPI.DOUBLE], source=proc_id_source, tag=0)

        for i in range(1, n + 1):
            sp = sps_recv.sp[i - 1]
            if sp is None:
                sp = type(sps_send.sp[0])()
                sps_recv.sp[i - 1] = sp
            conv_int_real_arrays_sp(sp, intarr_sp[:, i - 1], realarr_sp[:, i - 1])

        comm.Recv([intarr_by, MPI.INT], source=proc_id_source, tag=0)
        comm.Recv([realarr_by, MPI.DOUBLE], source=proc_id_source, tag=0)
        comm.Recv([strarr_by, MPI.CHAR], source=proc_id_source, tag=0)

        for i in range(1, n + 1):
            conv_arrays_by(
                sps_recv.sp[i - 1].byot,
                intarr_by[:, i - 1],
                realarr_by[:, i - 1],
                bytes(strarr_by[i - 1]).decode("utf-8", errors="ignore").rstrip("\x00").rstrip(),
            )

        comm.Recv([intarr_by, MPI.INT], source=proc_id_source, tag=0)
        comm.Recv([realarr_by, MPI.DOUBLE], source=proc_id_source, tag=0)
        comm.Recv([strarr_by, MPI.CHAR], source=proc_id_source, tag=0)

        for i in range(1, n + 1):
            conv_arrays_by(
                sps_recv.sp[i - 1].byot_ini,
                intarr_by[:, i - 1],
                realarr_by[:, i - 1],
                bytes(strarr_by[i - 1]).decode("utf-8", errors="ignore").rstrip("\x00").rstrip(),
            )

        comm.Recv([intarr_by, MPI.INT], source=proc_id_source, tag=0)
        comm.Recv([realarr_by, MPI.DOUBLE], source=proc_id_source, tag=0)
        comm.Recv([strarr_by, MPI.CHAR], source=proc_id_source, tag=0)

        for i in range(1, n + 1):
            conv_arrays_by(
                sps_recv.sp[i - 1].byot_bf,
                intarr_by[:, i - 1],
                realarr_by[:, i - 1],
                bytes(strarr_by[i - 1]).decode("utf-8", errors="ignore").rstrip("\x00").rstrip(),
            )


def conv_sp_int_real_arrays(sp: Any) -> Tuple[np.ndarray, np.ndarray]:
    intarr = np.zeros((nint_particle,), dtype=np.int32)
    realarr = np.zeros((nreal_particle,), dtype=np.float64)

    intarr[0:5] = np.array([sp.id, sp.obtype, sp.obidx, sp.state_flag_last, sp.exit_flag], dtype=np.int32)
    intarr[5:10] = np.array([sp.within_jt, sp.rid, sp.idx, sp.length, sp.write_down_track], dtype=np.int32)

    realarr[0:5] = np.array([sp.create_time, sp.den, sp.djp, sp.djp0, sp.elp], dtype=np.float64)
    realarr[5:10] = np.array([sp.En, sp.en0, sp.exit_time, sp.Jm, sp.weight_clone], dtype=np.float64)
    realarr[10:14] = np.array([sp.weight_real, sp.m, sp.r_td, sp.jm0], dtype=np.float64)
    realarr[14:19] = np.array([sp.pd, sp.rp, sp.tgw, sp.weight_N, sp.simu_bgtime], dtype=np.float64)

    return intarr, realarr


def conv_int_real_arrays_sp(sp: Any, intarr: np.ndarray, realarr: np.ndarray) -> None:
    sp.id = int(intarr[0])
    sp.obtype = int(intarr[1])
    sp.obidx = int(intarr[2])
    sp.state_flag_last = int(intarr[3])
    sp.exit_flag = int(intarr[4])

    sp.within_jt = int(intarr[5])
    sp.rid = int(intarr[6])
    sp.idx = int(intarr[7])
    sp.length = int(intarr[8])
    sp.write_down_track = int(intarr[9])

    if hasattr(sp, "track_init"):
        sp.track_init(0)

    sp.create_time = float(realarr[0])
    sp.den = float(realarr[1])
    sp.djp = float(realarr[2])
    sp.djp0 = float(realarr[3])
    sp.elp = float(realarr[4])

    sp.En = float(realarr[5])
    sp.en0 = float(realarr[6])
    sp.exit_time = float(realarr[7])
    sp.Jm = float(realarr[8])
    sp.weight_clone = float(realarr[9])

    sp.weight_real = float(realarr[10])
    sp.m = float(realarr[11])
    sp.r_td = float(realarr[12])
    sp.jm0 = float(realarr[13])

    sp.pd = float(realarr[14])
    sp.rp = float(realarr[15])
    sp.tgw = float(realarr[16])
    sp.weight_N = float(realarr[17])
    sp.simu_bgtime = float(realarr[18])


def conv_arrays_by(by: Any, intarr: np.ndarray, realarr: np.ndarray, strarr: str) -> None:
    by.ms.x = np.array(realarr[0:3], dtype=np.float64)
    by.ms.vx = np.array(realarr[3:6], dtype=np.float64)
    by.ms.m = float(realarr[6])
    by.ms.radius = float(realarr[7])
    by.ms.id = int(intarr[0])
    by.ms.obtype = int(intarr[1])
    by.ms.obidx = int(intarr[2])

    by.mm.x = np.array(realarr[8:11], dtype=np.float64)
    by.mm.vx = np.array(realarr[11:14], dtype=np.float64)
    by.mm.m = float(realarr[14])
    by.mm.radius = float(realarr[15])
    by.mm.id = int(intarr[3])
    by.mm.obtype = int(intarr[4])
    by.mm.obidx = int(intarr[5])

    by.rd.x = np.array(realarr[16:19], dtype=np.float64)
    by.rd.vx = np.array(realarr[19:22], dtype=np.float64)
    by.rd.m = float(realarr[22])
    by.rd.radius = float(realarr[23])
    by.rd.id = int(intarr[6])
    by.rd.obtype = int(intarr[7])
    by.rd.obidx = int(intarr[8])

    by.e = float(realarr[24])
    by.l = float(realarr[25])
    by.k = float(realarr[26])
    by.miu = float(realarr[27])
    by.mtot = float(realarr[28])
    by.Jc = float(realarr[29])
    by.a_bin = float(realarr[30])
    by.e_bin = float(realarr[31])
    by.lum = np.array(realarr[32:35], dtype=np.float64)
    by.f0 = float(realarr[35])
    by.inc = float(realarr[36])
    by.om = float(realarr[37])
    by.pe = float(realarr[38])
    by.t0 = float(realarr[39])
    by.me = float(realarr[40])

    by.an_in_mode = int(intarr[9])
    by.bname = strarr


def conv_by_arrays(by: Any) -> Tuple[np.ndarray, np.ndarray, str]:
    intarr = np.zeros((nint_by,), dtype=np.int32)
    realarr = np.zeros((nreal_by,), dtype=np.float64)

    realarr[0:3] = np.asarray(by.ms.x, dtype=np.float64)
    realarr[3:6] = np.asarray(by.ms.vx, dtype=np.float64)
    realarr[6:8] = np.array([by.ms.m, by.ms.radius], dtype=np.float64)
    intarr[0:3] = np.array([by.ms.id, by.ms.obtype, by.ms.obidx], dtype=np.int32)

    realarr[8:11] = np.asarray(by.mm.x, dtype=np.float64)
    realarr[11:14] = np.asarray(by.mm.vx, dtype=np.float64)
    realarr[14:16] = np.array([by.mm.m, by.mm.radius], dtype=np.float64)
    intarr[3:6] = np.array([by.mm.id, by.mm.obtype, by.mm.obidx], dtype=np.int32)

    realarr[16:19] = np.asarray(by.rd.x, dtype=np.float64)
    realarr[19:22] = np.asarray(by.rd.vx, dtype=np.float64)
    realarr[22:24] = np.array([by.rd.m, by.rd.radius], dtype=np.float64)
    intarr[6:9] = np.array([by.rd.id, by.rd.obtype, by.rd.obidx], dtype=np.int32)

    realarr[24:41] = np.array(
        [
            by.e,
            by.l,
            by.k,
            by.miu,
            by.mtot,
            by.Jc,
            by.a_bin,
            by.e_bin,
            by.lum[0],
            by.lum[1],
            by.lum[2],
            by.f0,
            by.inc,
            by.om,
            by.pe,
            by.t0,
            by.me,
        ],
        dtype=np.float64,
    )

    intarr[9] = int(by.an_in_mode)
    s = str(getattr(by, "bname", ""))
    if len(s) > nstr_by:
        s = s[:nstr_by]
    return intarr, realarr, s


def bcast_s1d_mpi(s1d: Any) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    intarr = np.zeros((int(s1d.type_int_size),), dtype=np.int32)
    realarr = np.zeros((int(s1d.type_real_size),), dtype=np.float64)
    logarr = np.zeros((int(s1d.type_log_size),), dtype=np.bool_)

    if rid == mpi_master_id:
        conv_s1d_int_real_arrays(s1d, intarr, realarr, logarr)

    comm.Bcast(intarr, root=mpi_master_id)
    comm.Bcast(realarr, root=mpi_master_id)
    comm.Bcast(logarr, root=mpi_master_id)

    if rid != mpi_master_id:
        conv_int_real_arrays_s1d(s1d, intarr, realarr, logarr)


def bcast_fc_mpi(fc: Any) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    intarr = np.zeros((int(fc.type_int_size),), dtype=np.int32)
    realarr = np.zeros((int(fc.type_real_size),), dtype=np.float64)
    logarr = np.zeros((int(fc.type_log_size),), dtype=np.bool_)

    if rid == mpi_master_id:
        conv_fc_int_real_arrays(fc, intarr, realarr, logarr)

    comm.Bcast(intarr, root=mpi_master_id)
    comm.Bcast(realarr, root=mpi_master_id)
    comm.Bcast(logarr, root=mpi_master_id)

    if rid != mpi_master_id:
        conv_int_real_arrays_fc(fc, intarr, realarr, logarr)


def bcast_s2d_mpi(s2d: Any) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    comm.Bcast(s2d.fxy, root=mpi_master_id)


def bcast_dms_gxj(dm: Any) -> None:
    for i in range(1, dm.n + 1):
        bcast_s2d_mpi(dm.mb[i - 1].all.gxj)


def get_dms(dm: Any) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    comm.Barrier()
    bcast_dms_barge(dm)
    bcast_dms_fden(dm)
    print("start get diffuse coefficients", rid)
    dm_get_dc_mpi(dm)
    print("cal dms finished", rid)
    comm.Barrier()


def bcast_dms_asym_weights(dm: Any) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    weights = np.zeros((6, int(dm.n)), dtype=np.float64)

    if rid == mpi_master_id:
        for i in range(1, dm.n + 1):
            weights[0:6, i - 1] = np.array(
                [
                    dm.mb[i - 1].all.asymp,
                    dm.mb[i - 1].star.asymp,
                    dm.mb[i - 1].bstar.asymp,
                    dm.mb[i - 1].sbh.asymp,
                    dm.mb[i - 1].bbh.asymp,
                    dm.weight_asym,
                ],
                dtype=np.float64,
            )

    comm.Bcast(weights, root=mpi_master_id)

    if rid != mpi_master_id:
        for i in range(1, dm.n + 1):
            mb = dm.mb[i - 1]
            mb.all.asymp = float(weights[0, i - 1])
            mb.star.asymp = float(weights[1, i - 1])
            mb.bstar.asymp = float(weights[2, i - 1])
            mb.sbh.asymp = float(weights[3, i - 1])
            mb.bbh.asymp = float(weights[4, i - 1])
        dm.weight_asym = float(weights[5, 0])


def collection_data_single_bympi(smsa: List[Any], n: int) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required")
    comm = MPI.COMM_WORLD
    update_arrays_single()
    comm.Barrier()
    if rid == mpi_master_id:
        print("single:start_collect")
        collect_to_root_sps_single(bksams_arr_norm, smsa, ctl.ntasks)
        print("single:end_collect")
    else:
        send_particle_sample_arr_mpi(bksams_arr_norm, smsa[rid], rid, mpi_master_id)
    print("collection finished rid=", rid)


def conv_s1d_int_real_arrays(s1d: Any, intarr: np.ndarray, realarr: np.ndarray, logarr: np.ndarray) -> None:
    raise NotImplementedError


def conv_int_real_arrays_s1d(s1d: Any, intarr: np.ndarray, realarr: np.ndarray, logarr: np.ndarray) -> None:
    raise NotImplementedError


def conv_fc_int_real_arrays(fc: Any, intarr: np.ndarray, realarr: np.ndarray, logarr: np.ndarray) -> None:
    raise NotImplementedError


def conv_int_real_arrays_fc(fc: Any, intarr: np.ndarray, realarr: np.ndarray, logarr: np.ndarray) -> None:
    raise NotImplementedError


def dm_get_dc_mpi(dm: Any) -> None:
    raise NotImplementedError


def update_arrays_single() -> None:
    raise NotImplementedError


try:
    from model_basic import bksams_arr_norm
except Exception:
    bksams_arr_norm = None
