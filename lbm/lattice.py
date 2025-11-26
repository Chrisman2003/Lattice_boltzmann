import cupy as cp
import numba as nb
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil


class Lattice:
    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        self.f = cp.zeros((n + (stencil.q,)), dtype=cp.float64)
        self.u = cp.zeros((n + (stencil.d,)), dtype=cp.float64)
        self.rho = cp.ones(n, dtype=cp.float64)
        self.gamma = 1.4

    def init_data(self):
        self.f = self.feq()

    def print_f(self):
        if self.stencil.d == 2:
            for j in cp.arange(self.f.shape[1]):
                for i in cp.arange(self.f.shape[0]):
                    ff = [f"{self.f[i, j, iq]}" for iq in cp.arange(self.stencil.q)]
                    print(f"[{i}, {j}] {ff}")
        elif self.stencil.d == 3:
            for k in cp.arange(self.f.shape[2]):
                for j in cp.arange(self.f.shape[1]):
                    for i in cp.arange(self.f.shape[0]):
                        ff = [f"{self.f[i, j, k, iq]}" for iq in cp.arange(self.stencil.q)]
                        print(f"[{i}, {j}, {k}] {ff}")

    def streaming(self):

        spatial_axes = tuple(range(self.stencil.d))

        for iq in range(self.stencil.q):

            if hasattr(self.stencil, 'c_cpu') and self.stencil.c_cpu is not None:
                vec = self.stencil.c_cpu[iq]
            else:

                vec = self.stencil.c[iq].get()

            shift = tuple(map(int, vec))

            self.f[..., iq] = cp.roll(self.f[..., iq], shift, axis=spatial_axes)

    def density(self):
        self.rho = cp.sum(self.f, axis=self.f.ndim-1)

    def velocity(self):
        idx = (slice(None),) * self.stencil.d
        for i in cp.arange(self.stencil.d):
            self.u[idx + (i,)] = cp.dot(self.f[idx], self.stencil.c[:, i])/self.rho

    def collision(self, omega):
        self.density()
        self.velocity()
        self.f -= omega * (self.f - self.feq())

    def feq(self):
        idx = (slice(None),) * self.stencil.d
        feq = cp.zeros_like(self.f)
        uu = cp.sum(self.u**2, axis=self.f.ndim-1)
        for iq, c_i, w_i in zip(cp.arange(self.stencil.q), self.stencil.c, self.stencil.w):
            uc = cp.dot(self.u[idx], c_i)
            feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu)
        return feq
