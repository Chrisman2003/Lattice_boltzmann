from mpi4py import MPI
from numba import cuda
import numpy as np
import math
import time

from lbm.lattice import Lattice
from lbm.stencil import Stencil
from lbm.constants import cs, inv_cs2, inv_cs4


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cuda.select_device(rank)


def exchange_halos(comm, d_f, nx, ny, local_nz, q, rank, size):
    left = (rank - 1) % size
    right = (rank + 1) % size

    send_up = d_f[:, :, local_nz, :].copy_to_host()
    send_dn = d_f[:, :, 1, :].copy_to_host()

    recv_dn = np.empty_like(send_dn)
    recv_up = np.empty_like(send_up)

    comm.Sendrecv(send_up, dest=right, recvbuf=recv_dn, source=left)
    comm.Sendrecv(send_dn, dest=left,  recvbuf=recv_up, source=right)

    d_f[:, :, 0, :] = recv_dn
    d_f[:, :, local_nz + 1, :] = recv_up


def main():
    d = 3
    q = 19
    n = 128

    Ma = 0.1
    Re = 1600

    L = n / (2 * np.pi)
    v0 = Ma * cs
    nu = L * v0 / Re
    tau = nu / cs**2 + 0.5
    omega = 1.0 / tau

    local_nz = n // size
    nz_loc = local_nz + 2

    stencil = Stencil(d, q)
    lattice = Lattice((n, n, nz_loc), stencil)

    z_global = np.arange(rank * local_nz, (rank + 1) * local_nz)
    x, y, z = np.meshgrid(
        np.arange(n),
        np.arange(n),
        z_global,
        indexing="ij"
    )

    x = x / L + np.pi/2
    y = y / L + np.pi/2
    z = z / L + np.pi/2

    lattice.u[:, :, 1:local_nz+1, 0] = +v0 * np.sin(x) * np.cos(y) * np.sin(z)
    lattice.u[:, :, 1:local_nz+1, 1] = -v0 * np.cos(x) * np.sin(y) * np.sin(z)

    lattice.init_data()

    d_f = cuda.to_device(lattice.f)
    d_f_new = cuda.device_array_like(d_f)
    d_rho = cuda.to_device(lattice.rho)
    d_u = cuda.to_device(lattice.u)
    d_c = cuda.to_device(stencil.c.astype(np.float32))
    d_w = cuda.to_device(stencil.w.astype(np.float32))

    threads = (8, 8, 4)
    blocks = (
        math.ceil(n / threads[0]),
        math.ceil(n / threads[1]),
        math.ceil(nz_loc / threads[2]),
    )

    mod_it = 50
    max_it = 500

    t0 = time.perf_counter()

    for it in range(max_it):
        Lattice.moments_kernel[blocks, threads](
            d_f, d_rho, d_u, d_c, q, n, n, nz_loc
        )

        Lattice.collision_stream_kernel[blocks, threads](
            d_f, d_f_new, d_rho, d_u, d_c, d_w,
            q, omega, inv_cs2, inv_cs4,
            n, n, nz_loc
        )

        d_f, d_f_new = d_f_new, d_f
        exchange_halos(comm, d_f, n, n, local_nz, q, rank, size)

        if rank == 0 and (it + 1) % mod_it == 0:
            print(f"Iteration {it+1}, Time {time.perf_counter()-t0:.2f}s")

    if rank == 0:
        print("Simulation finished")


if __name__ == "__main__":
    main()
