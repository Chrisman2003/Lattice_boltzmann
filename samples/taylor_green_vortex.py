from mpi4py import MPI
import numpy as np
from numba import cuda
from lbm.stencil import Stencil
from lbm.lattice import LatticeMPI
from lbm.constants import cs, inv_cs2, inv_cs4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters
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

if rank == 0:
    print(f"MPI ranks = {comm.Get_size()}")
    print(f"omega = {omega}")

stencil = Stencil(d, q)
lattice = LatticeMPI((n, n, n), stencil, comm)

# Initial condition (Taylor–Green vortex)
x, y, z = np.meshgrid(
    np.arange(lattice.nx),
    np.arange(lattice.ny),
    np.arange(lattice.nz_local + 2),
    indexing="ij"
)

z_global = z + lattice.rank * lattice.nz_local - 1

x = x / L + np.pi / 2
y = y / L + np.pi / 2
z = z_global / L + np.pi / 2

u0 = v0 * np.sin(x) * np.cos(y) * np.sin(z)
v1 = -v0 * np.cos(x) * np.sin(y) * np.sin(z)

lattice.u[:, :, :, 0] = nu0
lattice.u[:, :, :, 1] = v1
lattice.u[:, :, :, 2] = 0.0
lattice.rho[:] = 1.0

threads = (8, 8, 4)
blocks = (
    (lattice.nx + threads[0] - 1) // threads[0],
    (lattice.ny + threads[1] - 1) // threads[1],
    (lattice.nz_local + 2 + threads[2] - 1) // threads[2],
)

max_it = 200
for it in range(max_it):
    lattice.moments_kernel[blocks, threads](
        lattice.f, lattice.rho, lattice.u,
        lattice.c, lattice.q,
        lattice.nx, lattice.ny, lattice.nz_local
    )

    lattice.exchange_halos()

    lattice.collide_stream_kernel[blocks, threads](
        lattice.f, lattice.f_new,
        lattice.rho, lattice.u,
        lattice.c, lattice.w, lattice.q,
        omega, inv_cs2, inv_cs4,
        lattice.nx, lattice.ny, lattice.nz_local
    )

    lattice.f, lattice.f_new = lattice.f_new, lattice.f

    if it % 50 == 0:
        local_mass = lattice.rho.copy_to_host().sum()
        total_mass = comm.allreduce(local_mass, op=MPI.SUM)
        if rank == 0:
            print(f"it={it}, total_mass={total_mass}")
