import os
from typing import List, Tuple


def gethering_samples_single(fdir: str, nsnap: int, bks: ChainType, bksar: ParticleSamplesArrType):
    smsa: List[ParticleSamplesArrType] = [ParticleSamplesArrType() for _ in range(ctl.ntask_total)]
    sms: List[ChainType] = [ChainType() for _ in range(ctl.ntask_total)]

    tmpspid = f"{nsnap:4d}"
    nvalid = 0
    ex = False

    for i in range(1, ctl.ntask_total + 1):
        tmprid = f"{i:4d}"
        str_ = f"{tmprid.strip()}_{tmpspid.strip()}"
        fl = os.path.join(fdir.strip(), "bin", "single", f"samchn{str_.strip()}")
        ex = os.path.exists(f"{fl.strip()}.bin")
        if ex:
            nvalid += 1
            sms[nvalid - 1].input_bin(fl.strip())
            all_chain_to_arr_single(sms[nvalid - 1], smsa[nvalid - 1])
        else:
            print(f"{fl.strip()} does not exist")
            return ex

    print("sms chain gathering finished")

    smmerge_arr_single(smsa[:nvalid], nvalid, bksar)
    smmerge(sms[:nvalid], nvalid, bks)

    if nvalid >= 1:
        bks.destory()

    nvalid = 0
    return ex
