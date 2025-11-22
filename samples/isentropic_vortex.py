import sys
import time
import numpy as np
from lbm.stencil import Stencil
from lbm.lattice import Lattice
from lbm.constants import cs
from lbm.exporter import Exporter


def main():
    d = 2
    q = 9
    nx = 400
    ny = 400
    Ma = 0.05/cs  # 0.0866
    #Ma = 0.4
    b = 0.5
    Re = 100
    gamma = 1.4
    L = b*np.sqrt(np.log(2))
    rho0 = 1.0
    u0 = Ma * cs
    u_dir = np.array([1.0, 0.0])
    u_dir /= np.linalg.norm(u_dir)
    u = u0 * u_dir[0]
    v = u0 * u_dir[1]
    sos = 340.2940
    u0_real = Ma * sos
    rh0_real = 1.2250
    mu_real = 1.7894e-05
    #Re = rh0_real * u0_real * L/20 / mu_real
    nu = L * u0 / Re
    tau = nu/cs**2 + 0.5
    tau = 0.52
    omega = 1.0/tau

    print(f"Mach        = {Ma}")
    print(f"L           = {L}")
    print(f"Re          = {Re}")
    print(f"v0          = {u0}")
    print(f"Tau         = {tau}")
    print(f"omega       = {omega}")
    sys.stdout.flush()

    stencil = Stencil(d, q)
    lattice = Lattice((nx, ny), stencil)
    # Initial data
    x, y = np.meshgrid(range(nx), range(ny), indexing="ij")
    x = x/20
    y = y/20
    x0 = (nx - 1)/2/20
    y0 = (ny - 1)/2/20
    print(x0, y0, u0)
    r2 = (x - x0)**2 + (y - y0)**2
    #
    lattice.u[:, :, 0] = u + u0*2/np.pi * np.exp(0.5 * (1 - r2/b**2)) * (y - y0)/b
    lattice.u[:, :, 1] = v - u0*2/np.pi * np.exp(0.5 * (1 - r2/b**2)) * (x - x0)/b
    alpha = (2/np.pi * Ma)**2 * np.exp(1 - r2/b**2)
    lattice.rho[:] = rho0 * (1 - (gamma - 1)/2 * alpha)**(1/(gamma - 1))
    
    #eps = 5.0
    #lattice.u[:, :, 0] = u0 * (1 - eps/(2*np.pi) * np.exp(0.5*(1 - r2))) * (y - y0)
    #lattice.u[:, :, 1] = u0 * (1 + eps/(2*np.pi) * np.exp(0.5*(1 - r2))) * (x - x0)
    #lattice.rho[:] = rho0 * (1 - (gamma - 1) * eps**2/(8 * np.pi**2) * np.exp(1 - r2))**(1/(gamma - 1))
    
    #K = 0.125
    #lattice.u[:, :, 0] = u0 - K/(2*np.pi) * np.exp(0.5*(1 - r2)) * (y - y0)
    #lattice.u[:, :, 1] = u0 + K/(2*np.pi) * np.exp(0.5*(1 - r2)) * (x - x0)
    #lattice.rho[:] = rho0 * (1 + K**2*(gamma - 1)/(8 * cs**2 * np.pi**2) * np.exp(1 - r2))**(1/(gamma - 1))
    #tau = 0.54
    #omega = 1/tau
    #
    lattice.init_data()

    mod_it = 50
    max_it = 500
    print(f"max it {max_it} \t mod it {mod_it}")

    exporter = Exporter((nx, nx))
    filename = f"iv-{0}.vtk"
    exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})

    t0 = time.perf_counter()
    for it in range(max_it):
        lattice.collision(omega)
        lattice.streaming()
        if np.mod(it + 1, mod_it) == 0:
            total_mass = np.sum(lattice.f.flatten())
            print(f"Total mass: {total_mass}")
            filename = f"iv-{it + 1}.vtk"
            exporter.write_vtk(filename, {"density": lattice.rho, "velocity": lattice.u})
            print(f"Time: {time.perf_counter() - t0}")
            if np.isnan(total_mass):
                break
    print(f"Time: {time.perf_counter() - t0}")


if __name__ == '__main__':
    main()
