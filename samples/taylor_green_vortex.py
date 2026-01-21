import sys
sys.path.append("/cephyr/users/jingyang/Vera/workspace/Lattice_boltzmann")

import time
import numpy as np
import cupy as cp
from mpi4py import MPI

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

    L = n / (2 * np.pi)
    rho0 = 1.0
    p0 = rho0 * cs**2
    v0 = Ma * cs
    nu = L * v0 / Re
    t_c = L / v0
    tau = nu / cs**2 + 0.5
    omega = 1.0 / tau

    comm = MPI.COMM_WORLD       # <--- define comm
    rank = comm.rank            # <--- define rank
    size = comm.size            # <--- define size

    if rank == 0:
        print(f"Mach        = {Ma}")
        print(f"Re          = {Re}")
        print(f"v0          = {v0}")
        print(f"Tau         = {tau}")
        print(f"omega       = {omega}")
        print(f"t_c         = {t_c} \t 20 * t_c = {20 * t_c}")
        sys.stdout.flush()

    stencil = Stencil(d, q)
    lattice = Lattice((n, n, n), stencil)

    nz_local = lattice.nz_local
    z0 = rank * nz_local

    x, y, z = np.meshgrid(
        np.arange(n),
        np.arange(n),
        np.arange(z0, z0 + nz_local),
        indexing="ij"
    )

    x = x / L + np.pi / 2
    y = y / L + np.pi / 2
    z = z / L + np.pi / 2

    lattice.u[:, :, 1:-1, 0] = cp.asarray(
        +v0 * np.sin(x) * np.cos(y) * np.sin(z),
        dtype=cp.float32
    )
    lattice.u[:, :, 1:-1, 1] = cp.asarray(
        -v0 * np.cos(x) * np.sin(y) * np.sin(z),
        dtype=cp.float32
    )
    lattice.u[:, :, 1:-1, 2] = cp.float32(0.0)

    lattice.rho[:, :, 1:-1] = cp.float32(p0 / cs**2)
    lattice.init_data()

    mod_it = int(t_c / 2)
    max_it = 2 * 20 * mod_it

    if rank == 0:
        print(f"max it {max_it} \t mod it {mod_it}")
        print(max_it / mod_it)

    exporter = Exporter((n, n, n))
    t0 = time.perf_counter()
    first_hit_time = None

    for it in range(max_it):
        lattice.collision(omega)
        lattice.exchange_halos_z()
        lattice.streaming()

        if first_hit_time is None:
            first_hit_time = time.perf_counter() - t0
            if rank == 0:
                est = first_hit_time * (max_it // mod_it)
                print(f"Estimated total runtime: {est:.2f} seconds")


        #if (it + 1) % mod_it == 0 and rank == 0:
        #    print(f"Time: {time.perf_counter() - t0}")
        
        # Write VTK every mod_it steps
        if (it + 1) % mod_it == 0:
            # Gather full domain to rank 0
            local_rho = lattice.rho[:, :, 1:-1]
            local_u = lattice.u[:, :, 1:-1, :]
            # Move to CPU for MPI
            local_rho_cpu = cp.asnumpy(local_rho)
            local_u_cpu = cp.asnumpy(local_u)
            # Prepare full arrays on rank 0
            rho_full = None
            u_full = None
            if rank == 0:
                rho_full = np.zeros((n, n, n), dtype=np.float32)
                u_full = np.zeros((n, n, n, 3), dtype=np.float32)
            counts = np.full(size, nz_local * n * n, dtype=int)
            displs = np.array([i * nz_local * n * n for i in range(size)], dtype=int)
            # Gather rho
            comm.Gatherv(sendbuf=local_rho_cpu.flatten(),
                         recvbuf=(rho_full.flatten() if rank == 0 else None, counts, displs, MPI.FLOAT),
                         root=0)
            # Gather u
            comm.Gatherv(sendbuf=local_u_cpu.flatten(),
                         recvbuf=(u_full.flatten() if rank == 0 else None, counts * 3, displs * 3, MPI.FLOAT),
                         root=0)
            if rank == 0:
                filename = f"tgv-{it + 1}.vtk"
                exporter.write_vtk(filename, {"density": rho_full, "velocity": u_full})
                print(f"Time: {time.perf_counter() - t0}")


    if rank == 0:
        print(f"Time: {time.perf_counter() - t0}")


if __name__ == "__main__":
    main()
