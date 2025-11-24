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
        # 修正 1：axis 必须是 Python 的整数 tuple，不能用 cp.arange
        # 这里的 axis 对应空间维度 (例如 2D 就是 (0, 1))
        spatial_axes = tuple(range(self.stencil.d))

        # 遍历所有离散方向 q
        for iq in range(self.stencil.q):
            # 修正 2：shift (位移量) 必须是 Python 的 int tuple
            # 我们优先尝试使用 c_cpu (如果你更新了 Stencil 类)
            # 如果没有 c_cpu，则从 GPU 取回 (.get())
            if hasattr(self.stencil, 'c_cpu') and self.stencil.c_cpu is not None:
                vec = self.stencil.c_cpu[iq]
            else:
                # 慢速回退方案：如果 Stencil 类没更新，从 GPU 拷贝回 CPU
                vec = self.stencil.c[iq].get()

            # 确保转为 Python int tuple
            shift = tuple(map(int, vec))

            # 这里的 self.f 是 GPU 数组，cp.roll 在 GPU 上执行
            # 但 shift 和 axis 参数必须是 CPU 上的整数
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
