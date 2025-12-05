import sys
sys.path.append("/cephyr/users/jingyang/Vera/workspace/Lattice_boltzmann")
import time
import numpy as np
from lbm.stencil import Stencil
from lbm.lattice import Lattice
from lbm.constants import cs
from lbm.exporter import Exporter
import cupy as cp

def main():
    d = 3
    q = 19
    n = 128
    Ma = 0.1
    Re = 1600
    L = n/(2 * np.pi)
    rho0 = 1.0
    p0 = rho0 * cs**2
    v0 = Ma * cs
    nu = L * v0 / Re
    q_criterion = Ma * (v0/L)**2
    t_c = L/v0
    tau = nu/cs**2 + 0.5
    omega = 1.0/tau
    first_hit_time = None
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
    x, y, z = np.meshgrid(range(n), range(n), range(n), indexing="ij")
    x = x/L + np.pi/2
    y = y/L + np.pi/2
    z = z/L + np.pi/2
    lattice.u[:, :, :, 0] = cp.asarray(+v0 * np.sin(x) * np.cos(y) * np.sin(z))
    lattice.u[:, :, :, 1] = cp.asarray(-v0 * np.cos(x) * np.sin(y) * np.sin(z))
    lattice.u[:, :, :, 2] = 0
    lattice.rho[:] = cp.asarray(p0 / cs**2)
    lattice.init_data()

    mod_it = int(t_c/2)
    max_it = 2*20*mod_it
    print(f"max it {max_it} \t mod it {mod_it}")
    print(max_it/mod_it)
    exporter = Exporter((n, n, n))
    filename = f"tgv-{0}.vtk"
    exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})
    t0 = time.perf_counter()
    for it in range(max_it):
        #print(it + 1)
        lattice.collision(omega)
        lattice.streaming()
        if first_hit_time is None:
            	first_hit_time =time.perf_counter()-t0
            	est_total = first_hit_time*(max_it // mod_it)
            	print(f"Estimated total runtime: {est_total:.2f} seconds")
        if np.mod(it + 1, mod_it) == 0:
            filename = f"tgv-{it + 1}.vtk"
            exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})
            print(f"Time: {time.perf_counter() - t0}")
    print(f"Time: {time.perf_counter() - t0}")


if __name__ == '__main__':
    main()
