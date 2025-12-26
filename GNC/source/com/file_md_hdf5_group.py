import h5py


class HDF5Group:
    def __init__(self):
        self.group_id_upper = None
        self.group = None
        self.group_name = ""

    def create(self, group_id_upper, gname: str):
        self.group_id_upper = group_id_upper
        self.group = group_id_upper.create_group(gname)
        self.group_name = gname.strip()
        return self

    def close(self):
        self.group = None
        self.group_id_upper = None
