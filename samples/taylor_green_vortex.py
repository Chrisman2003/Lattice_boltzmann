import sys
sys.path.append("/cephyr/users/ispas/Vera/Lattice_boltzmann")

import time
import math
import numpy as np
from numba import cuda
from lbm.stencil import Stencil
from lbm.lattice import Lattice
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.exporter import Exporter


def main():
    d = 3
    q = 19
    n =100 # set to 129 for 7040 reference vtk file
    Ma = 0.1
    Re = 1600  #10 22 46 100 215 464
    L = n / (2 * np.pi)
    rho0 = 1.0
    p0 = rho0 * cs**2
    v0 = Ma * cs
    nu = L * v0 / Re
    q_criterion = Ma * (v0 / L)**2
    t_c = L / v0
    tau = nu / cs**2 + 0.5
    omega = 1.0 / tau

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

    x, y, z = np.meshgrid(range(n), range(n), range(n), indexing="ij")
    x = x / L + np.pi / 2
    y = y / L + np.pi / 2
    z = z / L + np.pi / 2

    lattice.u[:, :, :, 0] = +v0 * np.sin(x) * np.cos(y) * np.sin(z)
    lattice.u[:, :, :, 1] = -v0 * np.cos(x) * np.sin(y) * np.sin(z)
    lattice.u[:, :, :, 2] = 0
    lattice.rho[:] = p0 / cs**2
    lattice.init_data()

    
    d_f = cuda.to_device(lattice.f)
    d_f_new = cuda.device_array_like(lattice.f)
    d_rho = cuda.to_device(lattice.rho)
    d_u = cuda.to_device(lattice.u)
    d_c = cuda.to_device(stencil.c)
    d_w = cuda.to_device(stencil.w)
    

    threads = (8, 8, 4)
    blocks = (
        math.ceil(n / threads[0]),
        math.ceil(n / threads[1]),
        math.ceil(n / threads[2])
    )

    exporter = Exporter((n, n, n))
    exporter.write_vtk("tgvgpu2-0.vtk", {"density": lattice.rho, "velocity": lattice.u})

    mod_it = int(t_c / 2)
    max_it = 2 * 20 * mod_it
    print(f"max_it = {max_it} \t mod_it = {mod_it}")
    t0 = time.perf_counter()

    for it in range(max_it):
        Lattice.moments_kernel[blocks, threads](d_f, d_rho, d_u, d_c, q)
        Lattice.collision_and_stream_kernel[blocks, threads](
            d_f, d_f_new, d_rho, d_u, d_c, d_w, q, omega, inv_cs2, inv_cs4,n,n,n,
        )
        d_f, d_f_new = d_f_new, d_f

        if (it + 1) % mod_it == 0:
            print(f"Iteration {it+1}")

            rho_host = d_rho.copy_to_host()
            u_host = d_u.copy_to_host()
            total_mass = rho_host.sum()

            print(f"Total mass: {total_mass}")

            fname = f"tgvGPU-{it+1}.vtk"
            exporter.write_vtk(fname, {"density": rho_host, "velocity": u_host})

            print(f"Time: {time.perf_counter() - t0}")
    print("Total time:", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
