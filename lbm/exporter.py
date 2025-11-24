import cupy as cp
import numpy as np  # 需要 numpy 来处理最终的文件写入
import sys


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
        """
        完全向量化生成网格，避免使用 Python 循环遍历 GPU 数组。
        """
        # 1. 生成节点坐标 (Points)
        if len(self.n) == 2:
            # 生成坐标网格
            coords = cp.meshgrid(cp.arange(self.n[0] + 1), cp.arange(self.n[1] + 1), indexing="ij")
            # 堆叠并展平 -> (N, 2)
            points = cp.stack(coords, axis=-1).reshape(-1, 2).astype(cp.float32)
            # 2D 情况下补一个 Z=0 的列 -> (N, 3)
            points = cp.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        elif len(self.n) == 3:
            coords = cp.meshgrid(cp.arange(self.n[0] + 1), cp.arange(self.n[1] + 1), cp.arange(self.n[2] + 1),
                                 indexing="ij")
            points = cp.stack(coords, axis=-1).reshape(-1, 3).astype(cp.float32)
        else:
            raise Exception("Invalid dimensions")

        # 2. 生成单元连接关系 (Cells)
        # 计算每个维度的坐标索引
        if len(self.n) == 2:
            nx, ny = self.n
            # 生成所有单元格左下角的基准索引 (i, j)
            i, j = cp.meshgrid(cp.arange(nx), cp.arange(ny), indexing="ij")

            # 节点维数 (用于计算线性索引)
            n_p = (nx + 1, ny + 1)

            # 计算四个顶点的线性索引 (Vectorized)
            # 对应顺序: (0,0), (1,0), (1,1), (0,1)
            p0 = cp.ravel_multi_index((i, j), n_p)
            p1 = cp.ravel_multi_index((i + 1, j), n_p)
            p2 = cp.ravel_multi_index((i + 1, j + 1), n_p)
            p3 = cp.ravel_multi_index((i, j + 1), n_p)

            # 创建单元类型列 (全是 4)
            num_pts = cp.full_like(p0, 4)

            # 堆叠 -> (NumCells, 5) -> 展平
            cells = cp.stack((num_pts, p0, p1, p2, p3), axis=-1).reshape(-1).astype(cp.int32)

        elif len(self.n) == 3:
            nx, ny, nz = self.n
            i, j, k = cp.meshgrid(cp.arange(nx), cp.arange(ny), cp.arange(nz), indexing="ij")
            n_p = (nx + 1, ny + 1, nz + 1)

            # 8个顶点 (VTK Hexahedron ordering)
            p0 = cp.ravel_multi_index((i, j, k), n_p)
            p1 = cp.ravel_multi_index((i + 1, j, k), n_p)
            p2 = cp.ravel_multi_index((i + 1, j + 1, k), n_p)
            p3 = cp.ravel_multi_index((i, j + 1, k), n_p)
            p4 = cp.ravel_multi_index((i, j, k + 1), n_p)
            p5 = cp.ravel_multi_index((i + 1, j, k + 1), n_p)
            p6 = cp.ravel_multi_index((i + 1, j + 1, k + 1), n_p)
            p7 = cp.ravel_multi_index((i, j + 1, k + 1), n_p)

            num_pts = cp.full_like(p0, 8)
            cells = cp.stack((num_pts, p0, p1, p2, p3, p4, p5, p6, p7), axis=-1).reshape(-1).astype(cp.int32)

        else:
            raise Exception("Invalid dimensions")

        self._cache["points"] = points.flatten()
        self._cache["cells"] = cells

    @staticmethod
    def __write_header(file_id):
        file_id.write(b"# vtk DataFile Version 3.0\n")
        file_id.write(b"lbm solver data\n")
        file_id.write(b"BINARY\n")
        file_id.write(b"DATASET UNSTRUCTURED_GRID\n")

    def __write_points(self, file_id):
        # 注意：这里只取一次 .get() 到 CPU，避免多次传输
        points_cpu = self._cache["points"].get()

        file_id.write(b"POINTS ")
        file_id.write(f'{int(len(points_cpu) / 3)}'.encode("ascii"))
        file_id.write(b" float\n")

        # VTK legacy binary 默认为 Big Endian
        # 如果当前系统是 Little Endian (绝大多数PC都是)，则需要 swap
        if sys.byteorder == 'little':
            points_cpu.astype(np.float32).byteswap().tofile(file_id, sep="")
        else:
            points_cpu.astype(np.float32).tofile(file_id, sep="")
        file_id.write(b"\n")

    def __write_cells(self, file_id):
        if len(self.n) == 2:
            size = 4
        elif len(self.n) == 3:
            size = 8
        else:
            raise Exception("Invalid dimensions")

        cells_cpu = self._cache["cells"].get()

        file_id.write(b"CELLS ")
        file_id.write(f'{self.num_cells} {int((size + 1) * self.num_cells)}'.encode("ascii"))
        file_id.write(b"\n")

        if sys.byteorder == 'little':
            cells_cpu.byteswap().tofile(file_id, sep="")
        else:
            cells_cpu.tofile(file_id, sep="")

        file_id.write(b"\n")
        file_id.write(b"CELL_TYPES ")
        file_id.write(f'{self.num_cells}'.encode("ascii"))
        file_id.write(b"\n")

        # 在 GPU 上生成 cell types 数组
        if size == 4:  # quads
            cell_type_gpu = 9 * cp.ones(self.num_cells, dtype=cp.int32)
        elif size == 8:  # hexahedron
            cell_type_gpu = 12 * cp.ones(self.num_cells, dtype=cp.int32)

        # 传输回 CPU
        cell_type_cpu = cell_type_gpu.get()

        if sys.byteorder == 'little':
            cell_type_cpu.byteswap().tofile(file_id, sep="")
        else:
            cell_type_cpu.tofile(file_id, sep="")
        file_id.write(b"\n")

    def __write_data(self, file_id, data: dict):
        file_id.write(b"CELL_DATA ")
        file_id.write(f"{self.num_cells}".encode("ascii"))
        file_id.write(b"\n")

        # --- 处理 Density ---
        file_id.write(b"SCALARS density float 1\nLOOKUP_TABLE default\n")
        # .get() 传回 CPU
        density_cpu = data["density"].flatten().astype(np.float32).get()

        if sys.byteorder == 'little':
            density_cpu.byteswap().tofile(file_id, sep="")
        else:
            density_cpu.tofile(file_id, sep="")
        file_id.write(b"\n")

        # --- 处理 Velocity ---
        file_id.write(b"VECTORS velocity float\n")

        # 处理 GPU 上的数据 padding
        velocity_gpu = data["velocity"]
        if len(self.n) == 2:
            # 如果是 2D (Nx, Ny, 2)，需要 pad 到 (Nx, Ny, 3)
            # 假设 velocity 形状是 (Nx, Ny, 2) 或 (N_cells, 2)
            # cp.pad 语法: ((前,后), (前,后)...)
            # 这里假设输入已经是 (N, 2) 形状的数组
            if velocity_gpu.ndim == 2 and velocity_gpu.shape[1] == 2:
                velocity_gpu = cp.pad(velocity_gpu, ((0, 0), (0, 1)), mode="constant")
            elif velocity_gpu.ndim == 3:  # (Nx, Ny, 2)
                velocity_gpu = cp.pad(velocity_gpu, ((0, 0), (0, 0), (0, 1)), mode="constant")

        # .get() 传回 CPU
        velocity_cpu = velocity_gpu.flatten().astype(np.float32).get()

        if sys.byteorder == 'little':
            velocity_cpu.byteswap().tofile(file_id, sep="")
        else:
            velocity_cpu.tofile(file_id, sep="")
        file_id.write(b"\n")