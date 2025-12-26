import math

def print_flux():
    global ctl, rid, chattery_out_unit
    chattery_out_unit.write("print_flux\n")
    for i in range(ctl.m_bins):
        vals = [rid, i + 1]
        vals += list(ctl.bin_mass_flux_in[i][0:ctl.num_bk_comp])
        vals += list(ctl.bin_mass_flux_out[i][0:ctl.num_bk_comp])
        vals += list(ctl.bin_mass_emax_out[i][0:ctl.num_bk_comp])
        chattery_out_unit.write((" ".join(str(int(v)) for v in vals)) + "\n")
    for i in range(ctl.m_bins):
        for j in range(ctl.num_bk_comp):
            ctl.bin_mass_flux_in[i][j] = 0
            ctl.bin_mass_flux_out[i][j] = 0
            ctl.bin_mass_emax_out[i][j] = 0

def reset_create_time_zero(sps):
    ps = sps.head
    while ps is not None:
        ob = getattr(ps, "ob", None)
        if ob is not None and getattr(ob, "exit_flag", None) == exit_normal:
            ob.create_time = 0.0
        ps = ps.next

def reset_create_time(sps):
    sp = sps.head
    while sp is not None:
        ob = getattr(sp, "ob", None)
        if ob is None:
            raise RuntimeError("warnning, why it is not allocated?")
        if getattr(ob, "exit_flag", None) == exit_normal:
            ob.simu_bgtime = ob.exit_time
        sp = sp.next

def reset_j_for_boundary(sps):
    sp = sps.head
    while sp is not None:
        ob = getattr(sp, "ob", None)
        if ob is None:
            raise RuntimeError("warnning, why it is not allocated?")
        if ob.exit_flag == exit_normal and ob.en > ctl.energy_boundary:
            set_jm_init(ob)
            ob.jm0 = ob.jm
        ob.rp = ob.byot.a_bin * (1.0 - ob.byot.e_bin)
        sp = sp.next

def set_real_weight_arr_single(sms_single):
    for i in range(sms_single.n):
        get_sample_weight_real(sms_single.sp[i])

def set_real_weight(sms):
    pt = sms.head
    while pt is not None and getattr(pt, "ob", None) is not None:
        get_sample_weight_real(pt.ob)
        pt = pt.next

def convert_sams_convert():
    arr_to_chain_single(bksams_arr_norm, bksams)
    reset_create_time(bksams)
    convert_sams_pointer_arr(bksams, bksams_pointer_arr, type=1)

def arr_to_chain_single(bkarr, chain):
    chain.init(bkarr.n)
    pt = chain.head
    for i in range(bkarr.n):
        pt.ob = particle_sample_type()
        pt.ob = bkarr.sp[i]
        pt.idx = i + 1
        pt = pt.next

def convert_sams():
    refine_chain(bksams)
    convert_sams_pointer_arr(bksams, bksams_pointer_arr, type=1)
    reset_create_time(bksams)

def convert_sams_no_refine():
    convert_sams_pointer_arr(bksams, bksams_pointer_arr, type=1)
    reset_create_time(bksams)

def convert_sams_copy():
    bksams_out = chain_type()
    chain_select(bksams, bksams_out, exit_normal)
    bksams_out.copy(bksams)
    bksams_out.destory()
    reset_create_time(bksams)
    convert_sams_pointer_arr(bksams, bksams_pointer_arr, type=1)

def run_one_snap(cur_time_i, cur_time_f, smsa, n, update_dms):
    global rid
    if rid == 0:
        t1 = cpu_time()
    update_arrays_single()
    chattery_out_unit.write(f"sg:rid, cur_time_f= {rid} {cur_time_f}\n")
    RR_mpi(bksams, cur_time_f)
    mpi_barrier(mpi_comm_world)
    collection_data_single_bympi(smsa, ctl.ntasks)
    if rid == 0:
        print("reset time")
    reset_create_time(bksams)
    if rid == 0:
        t2 = cpu_time()
        chattery_out_unit.write(f"one snap running time: {t2 - t1}\n")
    if update_dms:
        set_dm_init(dms)
        if not hasattr(run_one_snap, "_norm"):
            run_one_snap._norm = 5
        if run_one_snap._norm > 0:
            get_ge_by_root(smsa, ctl.ntasks, True)
            update_weights()
            run_one_snap._norm -= 1
        else:
            get_ge_by_root(smsa, ctl.ntasks, False)
        get_dms(dms)
        if rid == 0:
            dms.print_norm(chattery_out_unit)
            print_num_boundary(dms)
            print_num_all(dms)
    mpi_barrier(mpi_comm_world)
    reset_create_time(bksams)
    reset_j_for_boundary(bksams)

def print_minj():
    pt = bksams.head
    jmin = 1.0
    jmin0 = 1.0
    sid = -1
    while pt is not None:
        ob = pt.ob
        if jmin > ob.jm and ob.en > ctl.energy_boundary:
            jmin = ob.jm
            jmin0 = ob.jm0
            sid = ob.id
        pt = pt.next
    print("rid, jmin=", rid, jmin, jmin0, sid)

def get_vr(r, a, e):
    ra = a * (1.0 + e)
    rp = a * (1.0 - e)
    if r > rp and r < ra:
        return (r / a) * ((r - rp) * (ra - r)) ** (-0.5)
    return 0.0

def get_collection_memory_usage(smsa, n):
    global nsize_tot
    nsize_smsa = 0
    for i in range(n):
        nsize_smsa += sizeof(smsa[i].sp) // 1024
    nsize_tot += nsize_smsa
    chattery_out_unit.write(f"nsize_smsa= {nsize_smsa}\n")

def show_memory_usage():
    global nsize_chain_bk, nsize_arr_bk, nsize_arr_bk_norm, nsize_arr_bk_pointer, nsize_tot_bk
    global nsize_chain_by, nsize_arr_by, nsize_arr_by_norm, nsize_arr_by_pointer, nsize_tot_by
    global nsize_tot
    get_memo_usage(proc_id)
    nsize_chain_bk = bksams.head.get_sizeof() // 1024
    nsize_arr_bk = sizeof(bksams_arr.sp) // 1024
    nsize_arr_bk_norm = sizeof(bksams_arr_norm.sp) // 1024
    nsize_arr_bk_pointer = sizeof(bksams_pointer_arr.pt) // 1024
    nsize_tot_bk = nsize_arr_bk + nsize_chain_bk + nsize_arr_bk_norm + nsize_arr_bk_pointer
    chattery_out_unit.write(
        f"rid:nsize_bk, bksam, bkarr, bkarrnorm, bkp: {nsize_tot_bk} {nsize_chain_bk} "
        f"{nsize_arr_bk} {nsize_arr_bk_norm} {nsize_arr_bk_pointer}\n"
    )
    nsize_chain_by = 0
    nsize_arr_by = 0
    nsize_arr_by_norm = 0
    nsize_arr_by_pointer = 0
    nsize_tot_by = 0
    nsize_cfs_holder = {"v": 0}
    cfs.get_size(nsize_cfs_holder)
    nsize_cfs = nsize_cfs_holder["v"]
    nsize_tot = nsize_tot_bk + nsize_tot_by
    chattery_out_unit.write(f"rid:cfs: {nsize_cfs}\n")

def show_total_memory_usage():
    nsizetot_all = mpi_gather(nsize_tot, root=0, comm=mpi_comm_world)
    if rid == 0:
        print("tot memory usage:", sum(nsizetot_all), " kb")

def print_single_data_size():
    n_holder = {"v": 0}
    bksams.get_length(n_holder)
    n = n_holder["v"]
    chattery_out_unit.write(f"rid, bksams, bkarr, bks_norm= {rid} {n} {bksams_arr.n} {bksams_arr_norm.n}\n")

def refine_chain(chain):
    ps = chain.head
    while ps is not None:
        pt = ps.next
        if ps.ob.exit_flag != exit_normal:
            ps.ob = None
            if ps.prev is None:
                if pt is not None:
                    chain.head = pt
                    pt.prev = None
                    destroy_attach_pointer_chain_type(ps)
                    pt.set_head()
                else:
                    chain.head = None
            else:
                chain_pointer_delete_item_chain_type(ps)
        ps = pt

def particle_sample_get_weight_clone(en, clone, amplifier, e0):
    if clone >= 1:
        if en / e0 < 0.0:
            return 1.0
        nlvl = int(math.log10(en / e0))
        if nlvl < 0:
            nlvl = 0
        weight_clone = float(amplifier) ** (-float(nlvl))
        if (not math.isfinite(weight_clone)) or math.isnan(weight_clone):
            raise RuntimeError(f"error!weight_clone= {weight_clone}\namplifier, nlvl, en, e0= {amplifier} {nlvl} {en} {e0}")
        return weight_clone
    return 1.0

def prepare_ini_data(tmprid):
    bksams.input_bin(f"output/ini/bin/single/samchn{tmprid.strip()}")
    dms.input_bin("output/ini/bin/dms.bin")
    print("readin dms finished!")
    update_arrays_single()
