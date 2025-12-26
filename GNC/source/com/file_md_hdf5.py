import numpy as np
import h5py


class HDF5:
    def __init__(self):
        self.h5f = HDF5File()
        self.h5g = HDF5Group()
        self.h5t = HDF5Table()

    def open_file(self, fd: str):
        self.h5f.open(fd)
        return self

    def close_file(self):
        self.h5f.close()
        return self

    def create_group(self, gname: str, IDin=None):
        upper = IDin if IDin is not None else self.h5f.file
        self.h5g.create(upper, gname)
        return self

    def close_group(self):
        self.h5g.close()
        return self

    @staticmethod
    def write_2d_arr(group, arr, nx: int, ny: int, aname: str):
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (nx, ny):
            a = a.reshape((nx, ny))
        if aname in group:
            del group[aname]
        group.create_dataset(aname.strip(), data=a)
