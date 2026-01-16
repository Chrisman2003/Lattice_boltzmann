import numpy as np
import numba as nb
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil

"""
Density (Zeroth Moment) = Sum of all populations.
Momentum (First Moment) = Sum of (Population x Lattice Direction).
Velocity (Derived Moment) =	Momentum divided by Density.
Mass (Integral) = Sum of Density over all grid nodes.
"""

class Lattice:
    def __init__(self, n, stencil: Stencil):
        self.stencil = stencil
        self.f = np.zeros((n + (stencil.q,)), dtype=np.float64)
        self.u = np.zeros((n + (stencil.d,)), dtype=np.float64)
        self.rho = np.ones(n, dtype=np.float64)
        self.gamma = 1.4

    def init_data(self):
        self.f = self.feq()

    def print_f(self):
        if self.stencil.d == 2:
            for j in range(self.f.shape[1]):
                for i in range(self.f.shape[0]):
                    ff = [f"{self.f[i, j, iq]}" for iq in range(self.stencil.q)]
                    print(f"[{i}, {j}] {ff}")
        elif self.stencil.d == 3:
            for k in range(self.f.shape[2]):
                for j in range(self.f.shape[1]):
                    for i in range(self.f.shape[0]):
                        ff = [f"{self.f[i, j, k, iq]}" for iq in range(self.stencil.q)]
                        print(f"[{i}, {j}, {k}] {ff}")

    def streaming(self):
        axis = tuple([i for i in range(self.stencil.d)])
        idx = (slice(None),) * self.stencil.d
        for iq in range(self.stencil.q):
            #print(f"Roll {iq}: {self.stencil.c[iq]} :: {self.f[idx + (iq,)].shape}")
            self.f[idx + (iq,)] = np.roll(self.f[idx + (iq,)], self.stencil.c[iq], axis=axis)

    def density(self):
        self.rho = np.sum(self.f, axis=self.f.ndim-1)

    def velocity(self):
        idx = (slice(None),) * self.stencil.d
        for i in range(self.stencil.d):
            self.u[idx + (i,)] = np.dot(self.f[idx], self.stencil.c[:, i])/self.rho

    def collision(self, omega):
        self.density()
        self.velocity()
        self.f -= omega * (self.f - self.feq())

    # Equilibrium distribution function
    def feq(self):
        idx = (slice(None),) * self.stencil.d
        feq = np.zeros_like(self.f)
        uu = np.sum(self.u**2, axis=self.f.ndim-1)
        for iq, c_i, w_i in zip(range(self.stencil.q), self.stencil.c, self.stencil.w):
            uc = np.dot(self.u[idx], c_i)
            feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu)
        return feq
