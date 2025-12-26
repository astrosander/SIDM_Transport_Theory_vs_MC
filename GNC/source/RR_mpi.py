from com_main_gw import (
    ctl,
    rid,
    mpi_master_id,
    chattery_out_unit,
    bksams,
    dms,
    dc_grid_xstep,
    dc_grid_ystep,
    star_type_MS,
    star_type_BH,
    record_track_detail,
    exit_normal,
    exit_invtransit,
    exit_boundary_min,
    run_one_sample,
    chain_pointer_delete_item_chain_type,
    destroy_attach_pointer_chain_type,
)


def RR_mpi(ch, total_time):
    global dc_grid_xstep, dc_grid_ystep

    if ctl.chattery >= 1:
        chattery_out_unit.write(f"proc {rid} starting...\n")

    if rid == mpi_master_id:
        chattery_out_unit.write("simu begin\n")
        if ctl.chattery >= 1:
            chattery_out_unit.write(f"total number of samples: {bksams.n}\n")
            chattery_out_unit.write(f"total time: {total_time} Myr\n")
            chattery_out_unit.write(f"total number of procs: {ctl.ntasks}\n")

    if ctl.chattery >= 1:
        chattery_out_unit.write("-------result type idx nhiar cpuid eid ngene\n")

    dc_grid_xstep = dms.dc0.s2_dee.xstep
    dc_grid_ystep = dms.dc0.s2_dee.ystep

    pt = ch.head
    while pt is not None:
        if pt.ob.exit_flag == exit_normal:
            if ctl.trace_all_sample == -1:
                if pt.ob.obtype == star_type_MS:
                    pt.ob.write_down_track = record_track_detail
                else:
                    pt.ob.write_down_track = 0
            elif ctl.trace_all_sample == -2:
                if pt.ob.obtype == star_type_BH:
                    pt.ob.write_down_track = record_track_detail
                else:
                    pt.ob.write_down_track = 0

            run_one_sample(pt, total_time)

        ps = pt
        pt = pt.next

        if ctl.clone_scheme >= 1:
            cond1 = (ps.ob.exit_flag == exit_invtransit and ctl.del_cross_clone >= 1) or (
                ps.ob.exit_flag == exit_boundary_min
            )
            if cond1:
                if ps.prev is not None:
                    if getattr(ps, "ob", None) is not None:
                        ps.ob = None
                    chain_pointer_delete_item_chain_type(ps)
                else:
                    if pt is None:
                        break
                    destroy_attach_pointer_chain_type(ch.head)
                    ch.head = pt
                    pt.set_head()
                    pt.prev = None
                continue
