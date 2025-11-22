import numpy as np


class Exporter:
    def __init__(self, n):
        self.n = n
        self._cache = dict()
        self.num_cells = 1
        for ni in self.n:
            self.num_cells *= ni
        print(f"Num cells: {self.num_cells}")
        self.__cache_grid()

    def write_vtk(self, filename: str, data: dict = None):
        with open(filename, "wb") as file_id:
            self.__write_header(file_id)
            self.__write_points(file_id)
            self.__write_cells(file_id)
            if data:
                self.__write_data(file_id, data)

    def __cache_grid(self):
        if len(self.n) == 2:
            points = np.meshgrid(range(self.n[0] + 1), range(self.n[1] + 1), indexing="ij")
            cell0 = np.meshgrid(range(self.n[0]), range(self.n[1]), indexing="ij")
        elif len(self.n) == 3:
            points = np.meshgrid(range(self.n[0] + 1), range(self.n[1] + 1), range(self.n[2] + 1), indexing="ij")
            cell0 = np.meshgrid(range(self.n[0]), range(self.n[1]), range(self.n[2]), indexing="ij")
        else:
            raise Exception("Invalid dimensions")
        points = np.array(np.column_stack([r_i.flatten() for r_i in points]), dtype=np.float32)
        if len(self.n) == 2:
            points = np.insert(points, np.arange(2, points.size + 1, 2), 0.0)
        cell0 = np.column_stack([r_i.flatten() for r_i in cell0])
        n_p = tuple([ni + 1 for ni in self.n])
        if len(self.n) == 2:
            cells = np.array([[4,
                               np.ravel_multi_index(cell_i + np.array([0, 0]), n_p),
                               np.ravel_multi_index(cell_i + np.array([1, 0]), n_p),
                               np.ravel_multi_index(cell_i + np.array([1, 1]), n_p),
                               np.ravel_multi_index(cell_i + np.array([0, 1]), n_p)] for cell_i in cell0], dtype=np.int32)
        elif len(self.n) == 3:
            cells = np.array([[8,
                               np.ravel_multi_index(cell_i + np.array([0, 0, 0]), n_p),
                               np.ravel_multi_index(cell_i + np.array([1, 0, 0]), n_p),
                               np.ravel_multi_index(cell_i + np.array([1, 1, 0]), n_p),
                               np.ravel_multi_index(cell_i + np.array([0, 1, 0]), n_p),
                               np.ravel_multi_index(cell_i + np.array([0, 0, 1]), n_p),
                               np.ravel_multi_index(cell_i + np.array([1, 0, 1]), n_p),
                               np.ravel_multi_index(cell_i + np.array([1, 1, 1]), n_p),
                               np.ravel_multi_index(cell_i + np.array([0, 1, 1]), n_p)] for cell_i in cell0], dtype=np.int32)
        else:
            raise Exception("Invalid dimensions")
        self._cache["points"] = points.flatten()
        self._cache["cells"] = cells.flatten()

    @staticmethod
    def __write_header(file_id):
        file_id.write(b"# vtk DataFile Version 3.0\n")
        file_id.write(b"lbm solver data\n")
        file_id.write(b"BINARY\n")
        file_id.write(b"DATASET UNSTRUCTURED_GRID\n")

    def __write_points(self, file_id):
        file_id.write(b"POINTS ")
        file_id.write(f'{int(len(self._cache["points"])/3)}'.encode("ascii"))
        file_id.write(b" float\n")
        if np.little_endian:
            self._cache["points"].byteswap().tofile(file_id, sep="")
        else:
            self._cache["points"].tofile(file_id, sep="")
        file_id.write(b"\n")

    def __write_cells(self, file_id):
        if len(self.n) == 2:
            size = 4
        elif len(self.n) == 3:
            size = 8
        else:
            raise Exception("Invalid dimensions")
        file_id.write(b"CELLS ")
        file_id.write(f'{self.num_cells} {int((size + 1) * self.num_cells)}'.encode("ascii"))
        file_id.write(b"\n")
        if np.little_endian:
            self._cache["cells"].byteswap().tofile(file_id, sep="")
        else:
            self._cache["cells"].tofile(file_id, sep="")

        file_id.write(b"\n")
        file_id.write(b"CELL_TYPES ")
        file_id.write(f'{self.num_cells}'.encode("ascii"))
        file_id.write(b"\n")
        if size == 4:  # quads
            cell_type = 9 * np.ones(self.num_cells, dtype=np.int32)
        elif size == 8:  # hexahedron
            cell_type = 12 * np.ones(self.num_cells, dtype=np.int32)
        else:
            raise Exception("Invalid dimensions")

        if np.little_endian:
            cell_type.byteswap().tofile(file_id, sep="")
        else:
            cell_type.tofile(file_id, sep="")
        file_id.write(b"\n")

    def __write_data(self, file_id, data: dict):
        file_id.write(b"CELL_DATA ")
        file_id.write(f"{self.num_cells}".encode("ascii"))
        file_id.write(b"\n")

        file_id.write(b"SCALARS density float 1\nLOOKUP_TABLE default\n")
        if np.little_endian:
            data["density"].flatten().astype(np.float32).byteswap().tofile(file_id, sep="")
        else:
            data["density"].flatten().astype(np.float32).tofile(file_id, sep="")
        file_id.write(b"\n")

        file_id.write(b"VECTORS velocity float\n")
        if len(self.n) == 2:
            data["velocity"] = np.pad(data["velocity"], ((0, 0), (0, 0), (0, 1)), mode="constant")
        if np.little_endian:
            data["velocity"].flatten().astype(np.float32).byteswap().tofile(file_id, sep="")
        else:
            data["velocity"].flatten().astype(np.float32).tofile(file_id, sep="")
        file_id.write(b"\n")
