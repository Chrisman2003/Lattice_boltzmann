import numpy as np
from numba import cuda
from mpi4py import MPI
from lbm.constants import inv_cs2, inv_cs4

class LatticeMPI:
    def __init__(self, n_global, stencil, comm: MPI.Comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # One rank per GPU
        cuda.select_device(self.rank % len(cuda.gpus))

        self.stencil = stencil
        self.q = stencil.q
        self.c = cuda.to_device(stencil.c)
        self.w = cuda.to_device(stencil.w)

        nx, ny, nz = n_global
        assert nz % self.size == 0, "nz must be divisible by MPI size"

        self.nz_local = nz // self.size
        self.nx, self.ny = nx, ny

        # +2 for halo planes
        self.shape = (nx, ny, self.nz_local + 2)

        self.f = cuda.device_array(self.shape + (self.q,), np.float32)
        self.f_new = cuda.device_array_like(self.f)
        self.rho = cuda.device_array(self.shape, np.float32)
        self.u = cuda.device_array(self.shape + (3,), np.float32)

    def exchange_halos(self):
        """Exchange halo planes along z-direction (CUDA-aware MPI)"""
        if self.rank > 0:
            self.comm.Sendrecv(
                self.f[:, :, 1, :], self.rank - 1,
                recvbuf=self.f[:, :, 0, :], source=self.rank - 1
            )
        if self.rank < self.size - 1:
            self.comm.Sendrecv(
                self.f[:, :, self.nz_local, :], self.rank + 1,
                recvbuf=self.f[:, :, self.nz_local + 1, :], source=self.rank + 1
            )

    @staticmethod
    @cuda.jit(fastmath=True)
    def moments_kernel(f, rho, u, c, q, nx, ny, nz_local):
        i, j, k = cuda.grid(3)
        if i >= nx or j >= ny or k < 1 or k > nz_local:
            return

        s = 0.0
        ux = 0.0
        uy = 0.0
        uz = 0.0

        for iq in range(q):
            fi = f[i, j, k, iq]
            s += fi
            ux += fi * c[iq, 0]
            uy += fi * c[iq, 1]
            uz += fi * c[iq, 2]

        rho[i, j, k] = s
        if s > 0.0:
            inv = 1.0 / s
            u[i, j, k, 0] = ux * inv
            u[i, j, k, 1] = uy * inv
            u[i, j, k, 2] = uz * inv

    @staticmethod
    @cuda.jit(fastmath=True)
    def collide_stream_kernel(f, f_new, rho, u, c, w, q,
                              omega, inv_cs2, inv_cs4,
                              nx, ny, nz_local):
        i, j, k = cuda.grid(3)
        if i >= nx or j >= ny or k < 1 or k > nz_local:
            return

        ux = u[i, j, k, 0]
        uy = u[i, j, k, 1]
        uz = u[i, j, k, 2]
        rho_ = rho[i, j, k]
        uu = ux*ux + uy*uy + uz*uz

        for iq in range(q):
            ci0, ci1, ci2 = c[iq]
            uc = ux*ci0 + uy*ci1 + uz*ci2
            feq = w[iq] * rho_ * (
                1.0 + inv_cs2*uc + 0.5*inv_cs4*uc*uc - 0.5*inv_cs2*uu
            )
            f_post = f[i, j, k, iq] - omega * (f[i, j, k, iq] - feq)

            it = i + ci0
            jt = j + ci1
            kt = k + ci2

            if it < 0: it += nx
            if it >= nx: it -= nx
            if jt < 0: jt += ny
            if jt >= ny: jt -= ny

            f_new[it, jt, kt, iq] = f_post
