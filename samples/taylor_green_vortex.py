import sys
import time
import cupy as cp
from lbm.stencil import Stencil
from lbm.lattice import Lattice
from lbm.constants import cs
from lbm.exporter import Exporter


def main():
    d = 3
    q = 19
    n = 128
    Ma = 0.1
    Re = 1600
    L = n/(2 * cp.pi)
    rho0 = 1.0
    p0 = rho0 * cs**2
    v0 = Ma * cs
    nu = L * v0 / Re
    q_criterion = Ma * (v0/L)**2
    t_c = L/v0
    tau = nu/cs**2 + 0.5
    omega = 1.0/tau

    print(f"Mach        = {Ma}")
    print(f"Re          = {Re}")
    print(f"v0          = {v0}")
    print(f"Tau         = {tau}")
    print(f"omega       = {omega}")
    print(f"Q-criterion = {q_criterion}")
    print(f"t_c         = {t_c} \t 20 * t_c = {20 * t_c}")
    sys.stdout.flush()

    stencil = Stencil(d, q)
    lattice = Lattice((n, n, n), stencil)
    # Initial data
    x, y, z = cp.meshgrid(cp.arange(n), cp.arange(n), cp.arange(n), indexing="ij")
    x = x/L + cp.pi/2
    y = y/L + cp.pi/2
    z = z/L + cp.pi/2
    lattice.u[:, :, :, 0] = +v0 * cp.sin(x) * cp.cos(y) * cp.sin(z)
    lattice.u[:, :, :, 1] = -v0 * cp.cos(x) * cp.sin(y) * cp.sin(z)
    lattice.u[:, :, :, 2] = 0
    lattice.rho[:] = p0 / cs**2
    lattice.init_data()

    mod_it = int(t_c/2)
    max_it = 2*20*mod_it
    print(f"max it {max_it} \t mod it {mod_it}")

    exporter = Exporter((n, n, n))
    filename = f"tgv-{0}.vtk"
    exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})
    t0 = time.perf_counter()
    for it in cp.arange(max_it):
        print(it + 1)
        lattice.collision(omega)
        lattice.streaming()
        if cp.mod(it + 1, mod_it) == 0:
            filename = f"tgv-{it + 1}.vtk"
            exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})
            print(f"Time: {time.perf_counter() - t0}")
    print(f"Time: {time.perf_counter() - t0}")


if __name__ == '__main__':
    main()
