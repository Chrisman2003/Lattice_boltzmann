# module purge
# module load foss/2025b
# module load CUDA/12.9.1
# module load numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

import sys

sys.path.append("/cephyr/users/konkala/Vera/Downloads/lbm_code")
import time
import numpy as np
from numba import cuda
from lbm.stencil import Stencil
from lbm.lattice_kernal_2 import Lattice
from lbm.constants import cs, inv_cs2, inv_cs4
from lbm.exporter import Exporter


# def get_block_grid_2d(nx, ny):
#     N = nx * ny 
#     threadsperblock = 1024
#     blockspergrid = (N + threadsperblock - 1) // threadsperblock
#     return blockspergrid, threadsperblock

# def get_block_grid_3d(nx,ny,nq):
#     N = nx*ny*nq
#     threadsperblock = 1024
#     blockspergrid = (N + threadsperblock - 1) // threadsperblock
#     return blockspergrid, threadsperblock

def get_block_grid_2d(nx, ny):
    threadsperblock = (16, 16)
    blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return blockspergrid, threadsperblock


def get_block_grid_3d(nx, ny, nq):
    threadsperblock = (8, 8, 4)
    blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (nq + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    return blockspergrid, threadsperblock


def main():
    d = 2
    q = 9
    nx = 400
    ny = 400
    Ma = 0.05 / cs
    b = 0.5
    Re = 100
    gamma = 1.4
    L = b * np.sqrt(np.log(2))
    rho0 = 1.0
    u0 = Ma * cs
    u_dir = np.array([1.0, 0.0])
    u_dir /= np.linalg.norm(u_dir)
    u = u0 * u_dir[0]
    v = u0 * u_dir[1]
    nu = L * u0 / Re
    tau = nu / cs ** 2 + 0.5
    tau = 0.52
    omega = 1.0 / tau
    omega = np.float32(omega)
    inv_cs2_f32 = np.float32(inv_cs2)
    inv_cs4_f32 = np.float32(inv_cs4)

    print(f"Mach        = {Ma}")
    print(f"Re          = {Re}")
    print(f"Tau         = {tau}")
    print(f"omega       = {omega}")
    sys.stdout.flush()

    stencil = Stencil(d, q)
    lattice = Lattice((nx, ny), stencil)

    x, y = np.meshgrid(range(nx), range(ny), indexing="ij")
    x = x / 20
    y = y / 20
    x0 = (nx - 1) / 2 / 20
    y0 = (ny - 1) / 2 / 20
    r2 = (x - x0) ** 2 + (y - y0) ** 2

    lattice.u[:, :, 0] = u + u0 * 2 / np.pi * np.exp(0.5 * (1 - r2 / b ** 2)) * (y - y0) / b
    lattice.u[:, :, 1] = v - u0 * 2 / np.pi * np.exp(0.5 * (1 - r2 / b ** 2)) * (x - x0) / b
    alpha = (2 / np.pi * Ma) ** 2 * np.exp(1 - r2 / b ** 2)
    lattice.rho[:] = rho0 * (1 - (gamma - 1) / 2 * alpha) ** (1 / (gamma - 1))

    lattice.init_data(inv_cs2_f32, inv_cs4_f32)

    d_f_old = cuda.to_device(lattice.f)
    d_f_new = cuda.device_array_like(lattice.f)
    d_rho = cuda.to_device(lattice.rho)
    d_u = cuda.to_device(lattice.u)
    d_c = cuda.to_device(stencil.c.astype(np.float32))
    d_w = cuda.to_device(stencil.w.astype(np.float32))

    blocks_2d, threads_2d = get_block_grid_2d(nx, ny)
    blocks_3d, threads_3d = get_block_grid_3d(nx, ny, q)

    mod_it = 50
    max_it = 500
    print(f"max it {max_it} \t mod it {mod_it}")

    exporter = Exporter((nx, ny))
    filename = f"iv-{0}.vtk"
    exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})

    t0 = time.perf_counter()
    for it in range(max_it):
        lattice.collision_kernel[blocks_3d, threads_3d](
            d_f_old, d_rho, d_u, omega, d_c, d_w, inv_cs2_f32, inv_cs4_f32
        )

        lattice.streaming_kernel[blocks_3d, threads_3d](
            d_f_old, d_f_new, d_c
        )

        d_f_old, d_f_new = d_f_new, d_f_old

        lattice.density_kernel[blocks_2d, threads_2d](
            d_f_old, d_rho
        )
        lattice.velocity_kernel[blocks_2d, threads_2d](
            d_f_old, d_rho, d_u, d_c
        )

        if (it + 1) % mod_it == 0:
            d_f_old.copy_to_host(lattice.f)
            d_rho.copy_to_host(lattice.rho)
            d_u.copy_to_host(lattice.u)

            total_mass = lattice.f.sum()
            print(f"Total mass: {total_mass}")
            filename = f"iv-{it + 1}.vtk"
            exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})
            print(f"Time: {time.perf_counter() - t0}")
            if np.isnan(total_mass):
                break

    print(f"Time: {time.perf_counter() - t0}")


if __name__ == '__main__':
    main()