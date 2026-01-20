import cupy as cp
import numpy as np
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil
from mpi4py import MPI

class Lattice:
    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        self.stencil_c = cp.asarray(stencil.c)
        self.stencil_w = cp.asarray(stencil.w)

        self.f = cp.zeros(n + (stencil.q,), dtype=cp.float64)
        self.u = cp.zeros(n + (stencil.d,), dtype=cp.float64)
        self.rho = cp.ones(n, dtype=cp.float64)
        self.gamma = 1.4

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def init_data(self):
        self.f = self.feq()

    def print_f(self):
        f_cpu = cp.asnumpy(self.f)
        if self.stencil.d == 2:
            for j in range(f_cpu.shape[1]):
                for i in range(f_cpu.shape[0]):
                    ff = [f"{f_cpu[i, j, iq]}" for iq in range(self.stencil.q)]
                    print(f"[{i}, {j}] {ff}")
        elif self.stencil.d == 3:
            for k in range(f_cpu.shape[2]):
                for j in range(f_cpu.shape[1]):
                    for i in range(f_cpu.shape[0]):
                        ff = [f"{f_cpu[i, j, k, iq]}" for iq in range(self.stencil.q)]
                        print(f"[{i}, {j}, {k}] {ff}")

    def streaming(self):
        axis = tuple(range(self.stencil.d))
        idx = (slice(None),) * self.stencil.d
        for iq in range(self.stencil.q):
            self.f[idx + (iq,)] = cp.roll(self.f[idx + (iq,)], self.stencil.c[iq], axis=axis)

    def density(self):
        self.rho = cp.sum(self.f, axis=-1)

    def velocity(self):
        self.u = cp.tensordot(self.f, self.stencil_c, axes=(-1, 0)) / self.rho[..., None]

    def collision(self, omega):
        self.density()
        self.velocity()
        self.f -= omega * (self.f - self.feq())

    def feq(self):
        uc = cp.tensordot(self.u, self.stencil_c, axes=(-1, 1))
        uu = cp.sum(self.u**2, axis=-1, keepdims=True)
        feq = self.stencil_w * self.rho[..., None] * (
                1.0 +
                inv_cs2 * uc +
                0.5 * inv_cs4 * uc * uc -
                0.5 * inv_cs2 * uu
        )
        return feq

    def exchange_halos_z(self):
        """Example halo exchange along last axis (z)"""
        send_rank = (self.rank + 1) % self.size
        recv_rank = (self.rank - 1) % self.size

        # last slice along z
        sendbuf = self.rho[..., -1].copy()
        recvbuf = cp.zeros_like(sendbuf)

        # GPU-aware MPI sendrecv
        self.comm.Sendrecv(sendbuf=sendbuf, dest=send_rank,
                           recvbuf=recvbuf, source=recv_rank)

        # Update first slice
        self.rho[..., 0] = recvbuf


