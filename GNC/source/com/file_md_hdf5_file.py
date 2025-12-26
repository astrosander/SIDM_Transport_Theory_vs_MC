import h5py


class HDF5File:
    HDF5_FILE_FLAG_READ = 1
    HDF5_FILE_FLAG_WRITE = 2
    HDF5_FILE_FLAG_RW = 3

    def __init__(self):
        self.file = None
        self.error = 0
        self.fd = ""

    def open(self, fd: str, flag: int | None = None):
        self.fd = fd
        try:
            if flag == self.HDF5_FILE_FLAG_READ:
                self.file = h5py.File(fd, "r")
            elif flag == self.HDF5_FILE_FLAG_RW:
                self.file = h5py.File(fd, "r+")
            elif flag == self.HDF5_FILE_FLAG_WRITE:
                self.file = h5py.File(fd, "w")
            else:
                self.file = h5py.File(fd, "w")
            self.error = 0
        except Exception:
            self.file = None
            self.error = 1
        return self

    def close(self):
        try:
            if self.file is not None:
                self.file.close()
            self.error = 0
        except Exception:
            self.error = 1
        finally:
            self.file = None
