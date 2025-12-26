from __future__ import annotations

import numpy as np

try:
    import h5py
except Exception as _e:  # noqa: F841
    h5py = None


def save_stsfcweight_hdf5(x, w, n, group_id, tname, xmin, xmax, rxn, fc_flag):
    if h5py is None:
        raise ImportError("h5py is required")
    fc = sts_fc_type()
    fc.init(xmin, xmax, rxn, fc_flag, use_weight=True)
    get_fc_weight(np.asarray(x)[:n], np.asarray(w)[:n], n, fc)
    save_fc_hdf5(fc, group_id, tname)


def save_stsfcweight_auto_hdf5(x, w, n, group_id, groupname, rxn, fc_flag):
    if h5py is None:
        raise ImportError("h5py is required")
    x = np.asarray(x)[:n]
    w = np.asarray(w)[:n]
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    fc = sts_fc_type()
    fc.init(xmin, xmax, rxn, fc_flag, use_weight=True)
    get_fc_weight(x, w, n, fc)
    save_fc_hdf5(fc, group_id, str(groupname).strip())


def input_dms_hdf5_pdf(dm, fl):
    if h5py is None:
        raise ImportError("h5py is required")
    fn = f"{str(fl).strip()}.hdf5"
    try:
        with h5py.File(fn, "r") as f:
            for i in range(dm.n):
                gname = f"{i + 1:4d}".strip()
                grp = f[gname]
                read_dms_hdf5_pdf(dm.mb[i].all, grp, "all")
                read_dms_hdf5_pdf(dm.mb[i].star, grp, "star")
                read_dms_hdf5_pdf(dm.mb[i].bd, grp, "bd")
                read_dms_hdf5_pdf(dm.mb[i].sbh, grp, "sbh")
                read_dms_hdf5_pdf(dm.mb[i].wd, grp, "wd")
                read_dms_hdf5_pdf(dm.mb[i].ns, grp, "ns")

            read_dms_hdf5_pdf(dm.all.all, f, "all")
            read_dms_hdf5_pdf(dm.all.star, f, "star")
            read_dms_hdf5_pdf(dm.all.sbh, f, "sbh")
            read_dms_hdf5_pdf(dm.all.wd, f, "wd")
            read_dms_hdf5_pdf(dm.all.ns, f, "ns")
            read_dms_hdf5_pdf(dm.all.bd, f, "bd")
    except OSError:
        raise FileNotFoundError("error! file may not exist!")


def output_dms_hdf5_pdf(dm, fl):
    if h5py is None:
        raise ImportError("h5py is required")
    fn = f"{str(fl).strip()}.hdf5"
    with h5py.File(fn, "w") as f:
        for i in range(dm.n):
            gname = f"{i + 1:4d}".strip()
            grp = f.create_group(gname)
            save_dms_hdf5_pdf(dm.mb[i].all, grp, "all")
            save_dms_hdf5_pdf(dm.mb[i].star, grp, "star")
            save_dms_hdf5_pdf(dm.mb[i].bd, grp, "bd")
            save_dms_hdf5_pdf(dm.mb[i].sbh, grp, "sbh")
            save_dms_hdf5_pdf(dm.mb[i].wd, grp, "wd")
            save_dms_hdf5_pdf(dm.mb[i].ns, grp, "ns")

        save_dms_hdf5_pdf(dm.all.all, f, "all")
        save_dms_hdf5_pdf(dm.all.star, f, "star")
        save_dms_hdf5_pdf(dm.all.sbh, f, "sbh")
        save_dms_hdf5_pdf(dm.all.wd, f, "wd")
        save_dms_hdf5_pdf(dm.all.ns, f, "ns")
        save_dms_hdf5_pdf(dm.all.bd, f, "bd")

        sub_group_id = f.create_group("dej")
        output_de_hdf5(dm, sub_group_id)


def output_de_hdf5(dm, group_id):
    dm.mb[0].dc.s2_de_0.save_hdf5(group_id, "de_0_1")

    s2d = s2d_type()
    s2d.init(dm.nbin_grid, dm.nbin_grid, dm.emin, dm.emax, dm.jmin, dm.jmax, sts_type_dstr)
    s2d.xcenter = np.array(dm.mb[0].dc.s2_de_110.xcenter, copy=True)
    s2d.ycenter = np.array(dm.mb[0].dc.s2_de_110.ycenter, copy=True)
    s2d.fxy = np.array(dm.dc0.s2_de_0.fxy, copy=True)
    for i in range(dm.n):
        s2d.fxy = s2d.fxy + dm.mb[0].mc / dm.mb[i].mc * dm.mb[i].dc.s2_de_110.fxy
    s2d.save_hdf5(group_id, "de1")
    dm.mb[0].dc.s2_de_110.save_hdf5(group_id, "de_110_1")

    if dm.n > 1:
        s2d = s2d_type()
        s2d.init(dm.nbin_grid, dm.nbin_grid, dm.emin, dm.emax, dm.jmin, dm.jmax, sts_type_dstr)
        s2d.xcenter = np.array(dm.mb[0].dc.s2_de_110.xcenter, copy=True)
        s2d.ycenter = np.array(dm.mb[0].dc.s2_de_110.ycenter, copy=True)
        s2d.fxy = np.array(dm.dc0.s2_de_0.fxy, copy=True)
        for i in range(dm.n):
            s2d.fxy = s2d.fxy + dm.mb[1].mc / dm.mb[i].mc * dm.mb[i].dc.s2_de_110.fxy
        s2d.save_hdf5(group_id, "de2")
        dm.mb[1].dc.s2_de_110.save_hdf5(group_id, "de_110_2")

    dm.dc0.s2_dee.save_hdf5(group_id, "dee")
    dm.dc0.s2_dej.save_hdf5(group_id, "dej")
    dm.dc0.s2_djj.save_hdf5(group_id, "djj")

    s2d.fxy = np.array(dm.dc0.s2_dj_rest.fxy, copy=True)
    for i in range(dm.n):
        s2d.fxy = s2d.fxy + (dm.mb[0].mc + dm.mb[i].mc) / dm.mb[i].mc / 2.0 * dm.mb[i].dc.s2_dj_111.fxy
    s2d.save_hdf5(group_id, "dj1")

    if dm.n > 1:
        s2d.fxy = np.array(dm.dc0.s2_dj_rest.fxy, copy=True)
        for i in range(dm.n):
            s2d.fxy = s2d.fxy + (dm.mb[1].mc + dm.mb[i].mc) / dm.mb[i].mc / 2.0 * dm.mb[i].dc.s2_dj_111.fxy
        s2d.save_hdf5(group_id, "dj2")


def read_dms_hdf5_pdf(so, group_id, str_):
    name = str(str_).strip()
    if name in group_id:
        sub_group_id = group_id[name]
        so.fden.read_hdf5(sub_group_id, "fden")
        so.fna.read_hdf5(sub_group_id, "fNa")
        so.barge.read_hdf5(sub_group_id, "fgx")
        so.fma.read_hdf5(sub_group_id, "fMa")
        so.fden_simu.read_hdf5(sub_group_id, "fden_simu")
        so.gxj.read_hdf5(sub_group_id, "gxj")
        so.n_real = 1
        so.n = 1
    else:
        so.n_real = 0
        so.n = 0


def save_dms_hdf5_pdf(so, group_id, str_):
    if getattr(so, "n_real", 0) > 0:
        name = str(str_).strip()
        sub_group_id = group_id.create_group(name)
        so.fden.save_hdf5(sub_group_id, "fden")
        so.fna.save_hdf5(sub_group_id, "fNa")
        so.barge.save_hdf5(sub_group_id, "fgx")
        so.fma.save_hdf5(sub_group_id, "fMa")
        so.fden_simu.save_hdf5(sub_group_id, "fden_simu")
        so.gxj.save_hdf5(sub_group_id, "gxj")
