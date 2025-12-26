from __future__ import annotations


def pro():
    readin_model_par("model.in")
    init_model()
    init_pro()

    init_sams_events()

    for isnap in range(1, ctl.n_spshot_total + 1):
        print("isnap=", isnap)

        tmpssnapid = f"{isnap:4d}"
        tmpj = f"{ctl.num_update_per_snap:4d}"

        print(tmpssnapid.strip() + "_" + tmpj.strip())

        fdir = "output/ecev/dms/dms_" + tmpssnapid.strip() + "_" + tmpj.strip()
        ex = file_exists(fdir)

        if ex:
            dms.input_bin(fdir.strip())
        else:
            print(isnap, ctl.num_update_per_snap, "not exist")

        gethering_samples_single("output/ecev/", isnap, bksams, bksams_arr, ex)
        print("gathering finished")

        if ex:
            pro_single(isnap)

    output_all_sts("output/")


if __name__ == "__main__":
    pro()
