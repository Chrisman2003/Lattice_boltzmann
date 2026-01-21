import cupy as cp
from mpi4py import MPI
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil


class Lattice:
    def __init__(self, n_global, stencil: Stencil):
        self.stencil = stencil
        self.stencil_c = cp.asarray(stencil.c, dtype=cp.float32)
        self.stencil_w = cp.asarray(stencil.w, dtype=cp.float32)

        # MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        nx, ny, nz = n_global
        assert nz % self.size == 0, "nz must be divisible by MPI size"

        self.nz_local = nz // self.size

        # +2 halo planes in z
        self.shape = (nx, ny, self.nz_local + 2)

        self.f = cp.zeros(self.shape + (stencil.q,), dtype=cp.float32)
        self.u = cp.zeros(self.shape + (stencil.d,), dtype=cp.float32)
        self.rho = cp.ones(self.shape, dtype=cp.float32)

    def init_data(self):
        self.f[...] = self.feq()

    def density(self):
        self.rho[...] = cp.sum(self.f, axis=-1, dtype=cp.float32)

    def velocity(self):
        self.u[...] = (
            cp.tensordot(self.f, self.stencil_c, axes=(-1, 0))
            / self.rho[..., None]
        )

    def collision(self, omega):
        self.density()
        self.velocity()
        self.f -= cp.float32(omega) * (self.f - self.feq())

    def feq(self):
        uc = cp.tensordot(self.u, self.stencil_c, axes=(-1, 1))
        uu = cp.sum(self.u ** 2, axis=-1, keepdims=True)

        feq = self.stencil_w * self.rho[..., None] * (
            cp.float32(1.0)
            + inv_cs2 * uc
            + cp.float32(0.5) * inv_cs4 * uc * uc
            - cp.float32(0.5) * inv_cs2 * uu
        )
        return feq.astype(cp.float32)

    def exchange_halos_z(self):
        """Exchange distribution-function halos in z (GPU-aware MPI)."""
        # Define neighbors
        prev_rank = self.rank - 1
        next_rank = self.rank + 1

        # -----------------------------------------------------------
        # STEP 1: Exchange with DOWN neighbor (Rank - 1)
        # We send our Bottom Real Layer (index 1) -> They receive in Top Halo
        # We receive from their Top Real Layer -> Into our Bottom Halo (index 0)
        # -----------------------------------------------------------
        if prev_rank >= 0:
            # 1. Pack: Copy the non-contiguous slice to a contiguous block
            send_buf = cp.ascontiguousarray(self.f[:, :, 1, :])
            recv_buf = cp.empty_like(send_buf)

            # 2. Exchange: Send contiguous data
            self.comm.Sendrecv(
                sendbuf=send_buf, dest=prev_rank,
                recvbuf=recv_buf, source=prev_rank
            )

            # 3. Unpack: Copy contiguous data back into the halo slice
            self.f[:, :, 0, :] = recv_buf

        # -----------------------------------------------------------
        # STEP 2: Exchange with UP neighbor (Rank + 1)
        # We send our Top Real Layer (index -2) -> They receive in Bottom Halo
        # We receive from their Bottom Real Layer -> Into our Top Halo (index -1)
        # -----------------------------------------------------------
        if next_rank < self.size:
            # 1. Pack
            send_buf = cp.ascontiguousarray(self.f[:, :, -2, :])
            recv_buf = cp.empty_like(send_buf)

            # 2. Exchange
            self.comm.Sendrecv(
                sendbuf=send_buf, dest=next_rank,
                recvbuf=recv_buf, source=next_rank
            )

            # 3. Unpack
            self.f[:, :, -1, :] = recv_buf

    def streaming(self):
        for iq in range(self.stencil.q):
            cx, cy, cz = self.stencil.c[iq]
            self.f[:, :, 1:-1, iq] = cp.roll(
                self.f[:, :, 1:-1, iq],
                shift=(cx, cy, cz),
                axis=(0, 1, 2)
            )
