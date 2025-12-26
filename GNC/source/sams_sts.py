from com_main_gw import ctl, pteve_star, pteve_sbh, pteve_wd, pteve_ns, pteve_bd


def sams_get_eve_num(weights, n, eve, isnap):
    neve = float(n)
    neve_w = 0.0
    for i in range(n):
        neve_w += weights[i]
    eve.ntot_eve_simu_w[isnap] = neve_w / float(ctl.ntask_total)
    eve.ntot_eve_simu[isnap] = neve / float(ctl.ntask_total)


def get_sams_event_rate(eve, evenorm, evetot, isnap):
    neve_w = eve.ntot_eve_simu_w[isnap]
    eve.eve_rate[isnap] = float(neve_w) / eve.dt * 1.0e3
    if eve.eve_rate[isnap] != 0.0:
        eve.p_eve[isnap] = eve.eve_rate[isnap] / (neve_w ** 0.5)
    else:
        eve.p_eve[isnap] = 0.0


def init_sams_events():
    pteve_star.init(ctl.total_time, ctl.n_spshot_total)
    pteve_sbh.init(ctl.total_time, ctl.n_spshot_total)
    pteve_wd.init(ctl.total_time, ctl.n_spshot_total)
    pteve_ns.init(ctl.total_time, ctl.n_spshot_total)
    pteve_bd.init(ctl.total_time, ctl.n_spshot_total)
