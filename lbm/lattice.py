import numpy as np
from numba import cuda, int32

@cuda.jit(device=True)
def feq_kernel(iq, rho, ux, uy, c, w, INV_CS2, INV_CS4):
    cx = c[iq, 0]
    cy = c[iq, 1]
    uc = cx * ux + cy * uy
    u_sq = ux * ux + uy * uy
    feq = w[iq] * rho * (1.0 + INV_CS2 * uc + 0.5 * INV_CS4 * uc * uc - 0.5 * INV_CS2 * u_sq)
    return feq

class Lattice:
    def __init__(self, n, stencil):
        self.stencil = stencil
        self.f = np.zeros((n + (stencil.q,)), dtype=np.float32)
        self.u = np.zeros((n + (stencil.d,)), dtype=np.float32)
        self.rho = np.ones(n, dtype=np.float32)
        self.gamma = 1.4

    def init_data(self, inv_cs2, inv_cs4):
        self.f = self.feq(inv_cs2, inv_cs4)

    def feq(self, inv_cs2, inv_cs4):
        idx = (slice(None),) * self.stencil.d
        feq = np.zeros_like(self.f)
        uu = np.sum(self.u**2, axis=self.f.ndim - 1)
        for iq, c_i, w_i in zip(range(self.stencil.q), self.stencil.c, self.stencil.w):
            uc = np.dot(self.u[idx], c_i)
            feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu)
        return feq

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

    @staticmethod
    @cuda.jit(fastmath=True)
    def streaming_kernel(f_in, f_out, c):
        ix, iy, iq = cuda.grid(3)
        Nx, Ny, Q = f_in.shape
        if ix < Nx and iy < Ny and iq < Q:
            sx = int32(c[iq, 0])
            sy = int32(c[iq, 1])
            source_x = ix - sx
            if source_x < 0:
                source_x += Nx
            elif source_x >= Nx:
                source_x -= Nx
            source_y = iy - sy
            if source_y < 0:
                source_y += Ny
            elif source_y >= Ny:
                source_y -= Ny
            f_out[ix, iy, iq] = f_in[source_x, source_y, iq]

    @staticmethod
    @cuda.jit(fastmath=True)
    def density_kernel(f_in, rho_out):
        ix, iy = cuda.grid(2)
        Nx, Ny, Q = f_in.shape
        if ix < Nx and iy < Ny:
            row = f_in[ix, iy]
            s = 0.0
            for iq in range(Q):
                s += row[iq]
            rho_out[ix, iy] = s

    @staticmethod
    @cuda.jit(fastmath=True)
    def velocity_kernel(f_in, rho_in, u_out, c_in):
        ix, iy = cuda.grid(2)
        Nx, Ny, Q = f_in.shape
        d = u_out.shape[2]
        if ix < Nx and iy < Ny:
            r = rho_in[ix, iy]
            if r > 1e-12:
                row = f_in[ix, iy]
                m0 = 0.0
                m1 = 0.0
                has3 = d == 3
                if has3:
                    m2 = 0.0
                for iq in range(Q):
                    fv = row[iq]
                    m0 += fv * c_in[iq, 0]
                    m1 += fv * c_in[iq, 1]
                    if has3:
                        m2 += fv * c_in[iq, 2]
                ir = 1.0 / r
                u_out[ix, iy, 0] = m0 * ir
                u_out[ix, iy, 1] = m1 * ir
                if has3:
                    u_out[ix, iy, 2] = m2 * ir
            else:
                for dim in range(d):
                    u_out[ix, iy, dim] = 0.0
    @staticmethod
    @cuda.jit(fastmath=True)
    def collision_kernel(f_in, rho_in, u_in, omega, c, w, INV_CS2, INV_CS4):
        ix, iy, iq = cuda.grid(3)
        Nx, Ny, Q = f_in.shape
        if ix < Nx and iy < Ny and iq < Q:
            rho = rho_in[ix, iy]
            ux = u_in[ix, iy, 0]
            uy = u_in[ix, iy, 1]
            feq_val = feq_kernel(iq, rho, ux, uy, c, w, INV_CS2, INV_CS4)
            f_old = f_in[ix, iy, iq]
            f_new = f_old - omega * (f_old - feq_val)
            f_in[ix, iy, iq] = f_new