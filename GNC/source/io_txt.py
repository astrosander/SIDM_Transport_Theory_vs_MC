"""
io_txt.py - Text file input/output functions for GNC simulation.
"""
from __future__ import annotations

import os
from typing import Any


def output_sams_sg_track_txt(sps: Any, fl: str) -> None:
    """
    Output particle sample tracks to text files.
    
    Args:
        sps: ParticleSamplesArrType object
        fl: Output directory path
    """
    from com_main_gw import (
        ctl, rid, star_type_MS, star_type_BH,
        exit_boundary_max, exit_tidal_empty, exit_tidal_full, exit_plunge_single, exit_normal
    )
    
    nummax = 500
    print(f"track n= {sps.n}")

    if not hasattr(sps, "sp") or sps.sp is None:
        print("sps_arr not allocated")
        return

    # Ensure output directories exist
    for subdir in ["emri/sg/ms", "emri/sg/bh", "plunge/bh", "td"]:
        os.makedirs(os.path.join(fl, subdir), exist_ok=True)

    num = 0
    num2 = 0

    record_track_nes = 1
    record_track_detail = 2

    for i in range(sps.n):
        sp = sps.sp[i]
        
        write_down = getattr(sp, 'write_down_track', 0)
        trace_all = getattr(ctl, 'trace_all_sample', 0)
        
        if write_down >= record_track_nes or trace_all >= record_track_detail:
            itmp = f"{i + 1 + rid * nummax:10d}".strip()
            exit_flag = getattr(sp, 'exit_flag', exit_normal)
            obtype = getattr(sp, 'obtype', star_type_MS)

            if exit_flag == exit_boundary_max:
                num2 += 1
                if num2 < nummax and getattr(ctl, 'output_track_emri', 0) >= 1:
                    if obtype == star_type_MS:
                        output_sample_track_txt(sp, f"{fl.strip()}/emri/sg/ms/evl_sg_{itmp}")
                    elif obtype == star_type_BH:
                        output_sample_track_txt(sp, f"{fl.strip()}/emri/sg/bh/evl_sg_{itmp}")

            elif exit_flag == exit_plunge_single:
                if obtype == star_type_BH and getattr(ctl, 'output_track_plunge', 0) >= 1:
                    output_sample_track_txt(sp, f"{fl.strip()}/plunge/bh/evl_sg_{itmp}")

            elif exit_flag in (exit_tidal_empty, exit_tidal_full):
                num += 1
                if num < nummax and getattr(ctl, 'output_track_td', 0) >= 1:
                    output_sample_track_txt(sp, f"{fl.strip()}/td/evl_sg_{itmp}")


def output_sample_track_txt(sp: Any, fl: str) -> None:
    """
    Output single particle track to text file.
    
    Args:
        sp: ParticleSampleType object
        fl: Output file path
    """
    track = getattr(sp, 'track', None)
    if track is None or len(track) == 0:
        return
    
    path = f"{fl.strip()}_simple.txt"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(path, "w") as f:
        f.write(f"{'time':29s}{'ac':20s}{'ec':20s}{'inc':20s}{'om':20s}{'flag':10s}\n")
        for tr in track:
            f.write(
                f"{tr.time:29.15E}"
                f"{tr.ac:20.10E}"
                f"{tr.ec:20.10E}"
                f"{tr.incout:20.10E}"
                f"{tr.omout:20.10E}"
                f"{int(tr.state_flag):10d}\n"
            )


def output_sg_sample_track_txt(sp: Any, fl: str) -> None:
    """
    Output single-component sample track to text file.
    
    Args:
        sp: ParticleSampleType object
        fl: Output file path
    """
    output_sample_track_txt(sp, fl)


def get_tnr_timescale_at_rh() -> float:
    """Get two-body relaxation timescale at influence radius."""
    from com_main_gw import get_tnr_timescale_at_rh as _get_tnr
    return _get_tnr()


def output_eveset_txt(pteve: Any, fl: str, flag: int) -> None:
    """Output event set to text file (placeholder)."""
    pass
