import numpy as np
import cupy as cp
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil


class Lattice:
    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        self.stencil_c = cp.asarray(stencil.c)
        self.stencil_w = cp.asarray(stencil.w)

        self.f = cp.zeros((n + (stencil.q,)), dtype=cp.float64)
        self.u = cp.zeros((n + (stencil.d,)), dtype=cp.float64)
        self.rho = cp.ones(n, dtype=cp.float64)
        self.gamma = 1.4

    def init_data(self):
        self.f = self.feq()

    def print_f(self):
        # Transfer to CPU for printing
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
        axis = tuple([i for i in range(self.stencil.d)])
        idx = (slice(None),) * self.stencil.d
        for iq in range(self.stencil.q):
            # print(f"Roll {iq}: {self.stencil.c[iq]} :: {self.f[idx + (iq,)].shape}")
            self.f[idx + (iq,)] = cp.roll(self.f[idx + (iq,)], self.stencil.c[iq], axis=axis)

    def density(self):
        self.rho = cp.sum(self.f, axis=self.f.ndim - 1)

    def velocity(self):
        # idx = (slice(None),) * self.stencil.d
        # for i in range(self.stencil.d):
        #     self.u[idx + (i,)] = cp.dot(self.f[idx], self.stencil_c[:, i]) / self.rho
        self.u = cp.tensordot(self.f, self.stencil_c, axes=(-1, 0)) / self.rho[..., None]
    def collision(self, omega):
        self.density()
        self.velocity()
        self.f -= omega * (self.f - self.feq())

    def feq(self):
        # idx = (slice(None),) * self.stencil.d
        # feq = cp.zeros_like(self.f)
        # uu = cp.sum(self.u ** 2, axis=self.f.ndim - 1)
        # for iq, c_i, w_i in zip(range(self.stencil.q), self.stencil_c, self.stencil_w):
        #     uc = cp.dot(self.u[idx], c_i)
        #     feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc ** 2 - 0.5 * inv_cs2 * uu)

        uc = cp.tensordot(self.u, self.stencil_c, axes=(-1, 1))

        uu = cp.sum(self.u ** 2, axis=-1, keepdims=True)

        feq = self.stencil_w * self.rho[..., None] * (
                1.0 +
                inv_cs2 * uc +
                0.5 * inv_cs4 * uc * uc -
                0.5 * inv_cs2 * uu
        )
        return feq
