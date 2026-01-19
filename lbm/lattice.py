import numpy as np
from numba import cuda
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil


class Lattice:
    """
    MPI + Numba-CUDA Lattice for LBM.
    Domain decomposition is handled outside this class.
    """

    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        self.n = n  # (nx, ny, nz_local_with_halo)

        self.f = np.zeros(n + (stencil.q,), dtype=np.float32)
        self.rho = np.ones(n, dtype=np.float32)
        self.u = np.zeros(n + (stencil.d,), dtype=np.float32)

    def init_data(self):
        self.f = self.feq()

    def feq(self):
        feq = np.zeros_like(self.f)
        uu = np.sum(self.u**2, axis=-1)

        for iq, c_i, w_i in zip(
            range(self.stencil.q), self.stencil.c, self.stencil.w
        ):
            uc = (
                self.u[..., 0] * c_i[0]
                + self.u[..., 1] * c_i[1]
                + self.u[..., 2] * c_i[2]
            )
            feq[..., iq] = w_i * self.rho * (
                1.0
                + inv_cs2 * uc
                + 0.5 * inv_cs4 * uc**2
                - 0.5 * inv_cs2 * uu
            )
        return feq

    # ============================================================
    # ================= CUDA KERNELS =============================
    # ============================================================

    @staticmethod
    @cuda.jit(fastmath=True)
    def moments_kernel(f, rho, u, c, q, nx, ny, nz):
        i, j, k = cuda.grid(3)

        if i >= nx or j >= ny or k >= nz:
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
        else:
            u[i, j, k, 0] = 0.0
            u[i, j, k, 1] = 0.0
            u[i, j, k, 2] = 0.0

    @staticmethod
    @cuda.jit(fastmath=True)
    def collision_stream_kernel(
        f_in, f_out, rho, u, c, w,
        q, omega, inv_cs2, inv_cs4,
        nx, ny, nz
    ):
        i, j, k = cuda.grid(3)

        if i >= nx or j >= ny or k >= nz:
            return

        ux = u[i, j, k, 0]
        uy = u[i, j, k, 1]
        uz = u[i, j, k, 2]
        rho_ijk = rho[i, j, k]

        uu = ux*ux + uy*uy + uz*uz
        common = -0.5 * inv_cs2 * uu

        for iq in range(q):
            cx = c[iq, 0]
            cy = c[iq, 1]
            cz = c[iq, 2]
            wi = w[iq]

            uc = ux*cx + uy*cy + uz*cz
            feq = wi * rho_ijk * (
                1.0
                + inv_cs2 * uc
                + 0.5 * inv_cs4 * uc * uc
                + common
            )

            f_post = f_in[i, j, k, iq] - omega * (f_in[i, j, k, iq] - feq)

            ii = i + cx
            jj = j + cy
            kk = k + cz

            if ii < 0: ii += nx
            if ii >= nx: ii -= nx
            if jj < 0: jj += ny
            if jj >= ny: jj -= ny
            if kk < 0: kk += nz
            if kk >= nz: kk -= nz

            f_out[ii, jj, kk, iq] = f_post
