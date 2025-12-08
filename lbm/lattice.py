import numpy as np
import numba as nb
from numba import cuda
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.stencil import Stencil

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

    def feq(self):
        idx = (slice(None),) * self.stencil.d
        feq = np.zeros_like(self.f)
        uu = np.sum(self.u**2, axis=self.f.ndim-1)
        for iq, c_i, w_i in zip(range(self.stencil.q), self.stencil.c, self.stencil.w):
            uc = np.dot(self.u[idx], c_i)
            feq[idx + (iq,)] = w_i * self.rho * (1.0 + inv_cs2 * uc + 0.5 * inv_cs4 * uc**2 - 0.5 * inv_cs2 * uu)
        return feq

    
    @staticmethod
    @cuda.jit(fastmath=True)
    def moments_kernel(f, rho, u, c, q):
        i, j, k = cuda.grid(3)
        nx, ny, nz = rho.shape
        
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
        rho_ijk = s
        rho[i, j, k] = rho_ijk
        if rho_ijk > 0.0:
            inv_rho = 1.0 / rho_ijk
            u[i, j, k, 0] = ux * inv_rho
            u[i, j, k, 1] = uy * inv_rho
            u[i, j, k, 2] = uz * inv_rho
        else:
            u[i, j, k, 0] = 0.0
            u[i, j, k, 1] = 0.0
            u[i, j, k, 2] = 0.0

    @staticmethod 
    @cuda.jit(fastmath=True)
    def collision_and_stream_kernel(f_in, f_out, rho, u, c, w, q, omega, inv_cs2, inv_cs4, nx, ny, nz):
        i, j, k = cuda.grid(3) 
        
        if i >= nx or j >= ny or k >= nz:
            return
        ux = u[i, j, k, 0]
        uy = u[i, j, k, 1]
        uz = u[i, j, k, 2]
        rho_ijk = rho[i, j, k]
        uu = ux*ux + uy*uy + uz*uz
        common_term = -0.5 * inv_cs2 * uu 

        for iq in range(q):
            ci0 = c[iq, 0]
            ci1 = c[iq, 1]
            ci2 = c[iq, 2]
            wi = w[iq]
            uc = ux*ci0 + uy*ci1 + uz*ci2
            uc_term = inv_cs2 * uc
            uc2_term = 0.5 * inv_cs4 * uc * uc
            
            feq_val = wi * rho_ijk * (1.0 + uc_term + uc2_term + common_term)
            f_old = f_in[i, j, k, iq]
            f_star = f_old - omega * (f_old - feq_val)
            f_star = max(f_star, 1e-16)
            itarget = i + ci0
            jtarget = j + ci1
            ktarget = k + ci2
            if itarget < 0: itarget += nx
            elif itarget >= nx: itarget -= nx
            
            if jtarget < 0: jtarget += ny
            elif jtarget >= ny: jtarget -= ny
            
            if ktarget < 0: ktarget += nz
            elif ktarget >= nz: ktarget -= nz
            f_out[itarget, jtarget, ktarget, iq] = f_star


        






