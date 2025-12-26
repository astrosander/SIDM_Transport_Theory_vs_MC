import numpy as np
import h5py


class HDF5Table:
    def __init__(self):
        self.nfields = 0
        self.nrecords = 0
        self.si = 0
        self.group = None
        self.tablename = ""
        self.field_names = []
        self.field_types = []
        self.type_sizes = []
        self.dataset = None

    def init_table(self, nfin: int, nrin: int, tname: str):
        if nrin < 0:
            raise ValueError("number of records must be >= 0")
        self.nfields = int(nfin)
        self.nrecords = int(nrin)
        self.si = 0
        self.tablename = str(tname).strip()
        self.field_names = [""] * self.nfields
        self.field_types = [None] * self.nfields
        self.type_sizes = [1] * self.nfields
        self.dataset = None
        return self

    def set_field(self, i: int, name: str, dtype, type_size: int = 1):
        self.field_names[i] = str(name).strip()
        self.field_types[i] = dtype
        self.type_sizes[i] = int(type_size)
        return self

    def _build_dtype(self):
        fields = []
        for name, dt, ts in zip(self.field_names, self.field_types, self.type_sizes):
            if not name:
                raise ValueError("field name is empty")
            if "," in name:
                raise ValueError("field name must not contain ','")
            if dt is None:
                raise ValueError(f"field dtype missing for {name}")

            if isinstance(dt, str):
                k = dt.lower()
                if k in ("real", "float", "float64", "r8"):
                    base = np.dtype(np.float64)
                elif k in ("int", "int32", "i4"):
                    base = np.dtype(np.int32)
                elif k in ("int64", "i8", "long"):
                    base = np.dtype(np.int64)
                elif k in ("str", "string") or k.startswith("str"):
                    base = np.dtype(f"S{ts}")
                    ts = 1
                else:
                    base = np.dtype(dt)
            else:
                base = np.dtype(dt)

            if base.kind in ("S", "U"):
                fields.append((name, base))
            else:
                if ts == 1:
                    fields.append((name, base))
                else:
                    fields.append((name, base, (ts,)))
        return np.dtype(fields)

    def prepare_write_table(self, group):
        self.group = group
        dt = self._build_dtype()
        if self.tablename in group:
            del group[self.tablename]
        self.dataset = group.create_dataset(self.tablename, shape=(self.nrecords,), dtype=dt)
        self.si = 0
        return self

    def prepare_read_table(self, group):
        self.group = group
        self.dataset = group[self.tablename]
        self.nrecords = int(self.dataset.shape[0])
        names = list(self.dataset.dtype.names or [])
        self.nfields = len(names)
        self.field_names = names
        self.field_types = [self.dataset.dtype.fields[n][0] for n in names]
        self.type_sizes = []
        for n in names:
            d = self.dataset.dtype.fields[n][0]
            if d.subdtype is not None:
                self.type_sizes.append(int(np.prod(d.subdtype[1])))
            elif d.kind in ("S", "U"):
                self.type_sizes.append(int(d.itemsize) if d.kind == "S" else int(d.itemsize // 4))
            else:
                self.type_sizes.append(1)
        self.si = 0
        return self

    def write_column_real(self, da):
        name = self.field_names[self.si]
        self.dataset[name] = np.asarray(da, dtype=np.float64)
        self.si += 1
        return self

    def write_column_int(self, da):
        name = self.field_names[self.si]
        self.dataset[name] = np.asarray(da, dtype=np.int64)
        self.si += 1
        return self

    def write_column_str(self, da, str_size: int):
        name = self.field_names[self.si]
        a = np.asarray(da)
        if a.dtype.kind == "U":
            b = np.array([s.encode("utf-8")[:str_size] for s in a.tolist()], dtype=f"S{str_size}")
        elif a.dtype.kind == "S":
            b = a.astype(f"S{str_size}")
        else:
            b = np.array([str(s).encode("utf-8")[:str_size] for s in a.tolist()], dtype=f"S{str_size}")
        self.dataset[name] = b
        self.si += 1
        return self

    def read_column_int(self, field_name: str | None = None):
        if field_name is None:
            name = self.field_names[self.si]
            self.si += 1
        else:
            name = field_name
        return np.asarray(self.dataset[name])

    def read_column_real(self, field_name: str | None = None):
        if field_name is None:
            name = self.field_names[self.si]
            self.si += 1
        else:
            name = field_name
        return np.asarray(self.dataset[name], dtype=np.float64)

    def read_column_str(self, str_size: int, field_name: str | None = None):
        if field_name is None:
            name = self.field_names[self.si]
            self.si += 1
        else:
            name = field_name
        a = np.asarray(self.dataset[name])
        if a.dtype.kind == "S":
            return np.array([x.decode("utf-8", errors="ignore") for x in a.tolist()], dtype=f"U{str_size}")
        return a.astype(f"U{str_size}")
