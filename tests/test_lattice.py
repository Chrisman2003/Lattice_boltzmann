import pytest
from pytest import approx
import numpy as np
from lbm.constants import cs
from lbm.stencil import Stencil
from lbm.lattice import Lattice


def setup_test_data(lattice: Lattice):
    lattice.rho[:] = 0.88
    if lattice.stencil.d == 2:
        for j in range(lattice.f.shape[1]):
            for i in range(lattice.f.shape[0]):
                for d in range(lattice.stencil.d):
                    lattice.u[i, j, d] = float(i + lattice.f.shape[0] * j) * 0.1 + d * 0.01
                for q in range(lattice.stencil.q):
                    lattice.f[i, j, q] = i + lattice.f.shape[0] * j + q * 0.01
    elif lattice.stencil.d == 3:
        for k in range(lattice.f.shape[2]):
            for j in range(lattice.f.shape[1]):
                for i in range(lattice.f.shape[0]):
                    for d in range(lattice.stencil.d):
                        lattice.u[i, j, k, d] = float(i + lattice.f.shape[0] * (j + lattice.f.shape[1] * k)) * 0.1 + d * 0.01
                    for q in range(lattice.stencil.q):
                        lattice.f[i, j, k, q] = i + lattice.f.shape[0] * (j + lattice.f.shape[1] * k) + q * 0.01


@pytest.mark.parametrize("d,q", [(2, 9), (3, 19)])
def test_streaming(d, q):
    stencil = Stencil(d, q)
    size = 4
    n = tuple([size for _ in range(d)])
    lattice = Lattice(n, stencil)
    setup_test_data(lattice)
    f0 = lattice.f.copy()
    lattice.streaming()

    idx = tuple([ni // 2 for ni in n])  # Find some cell in the middle
    for iq, c_i in zip(range(stencil.q), stencil.c):
        idx_i = tuple([idx[i] - c_i[i] for i in range(len(c_i))])
        assert lattice.f[idx + (iq,)] == approx(f0[idx_i + (iq,)])


@pytest.mark.parametrize("d,q", [(2, 9), (3, 19)])
def test_density(d, q):
    stencil = Stencil(d, q)
    size = 4
    n = tuple([size for _ in range(d)])
    lattice = Lattice(n, stencil)
    setup_test_data(lattice)
    f0 = lattice.f.copy()
    lattice.density()

    idx = tuple([ni // 2 for ni in n])  # Find some cell in the middle
    assert lattice.rho[idx] == approx(np.sum(f0[idx]))


@pytest.mark.parametrize("d,q", [(2, 9), (3, 19)])
def test_velocity(d, q):
    stencil = Stencil(d, q)
    size = 4
    n = tuple([size for _ in range(d)])
    lattice = Lattice(n, stencil)
    setup_test_data(lattice)
    f0 = lattice.f.copy()
    lattice.density()
    lattice.velocity()
    idx = tuple([ni // 2 for ni in n])  # Find some cell in the middle
    density = np.sum(f0[idx])
    velocity = np.zeros(d)
    for iq, c_i in zip(range(stencil.q), stencil.c):
        velocity += c_i * f0[idx + (iq,)]
    velocity /= density
    for i in range(d):
        assert lattice.u[idx + (i,)] == approx(velocity[i])


@pytest.mark.parametrize("d,q", [(2, 9), (3, 19)])
def test_equilibrium(d, q):
    stencil = Stencil(d, q)
    size = 4
    n = tuple([size for _ in range(d)])
    lattice = Lattice(n, stencil)
    setup_test_data(lattice)
    feq = lattice.feq()

    idx = tuple([ni // 2 for ni in n])  # Find some cell in the middle
    u = lattice.u[idx]
    rho = lattice.rho[idx]
    for iq, c_i, w_i in zip(range(stencil.q), stencil.c, stencil.w):
        feq_i = w_i * rho * (1 + np.dot(u, c_i)/cs**2 + 0.5 * np.dot(u, c_i)**2/cs**4 - 0.5 * np.dot(u, u)/cs**2)
        assert feq[idx + (iq,)] == approx(feq_i)


@pytest.mark.parametrize("d,q", [(2, 9), (3, 19)])
def test_collision(d, q):
    omega = 0.6
    stencil = Stencil(d, q)
    size = 4
    n = tuple([size for _ in range(d)])
    lattice = Lattice(n, stencil)
    setup_test_data(lattice)
    lattice.init_data()
    f0 = lattice.f.copy()
    feq = lattice.feq()
    f0 -= omega * (f0 - feq)
    lattice.collision(omega)

    idx = tuple([ni // 2 for ni in n])  # Find some cell in the middle
    for iq, c_i, w_i in zip(range(stencil.q), stencil.c, stencil.w):
        assert lattice.f[idx + (iq,)] == approx(f0[idx + (iq,)])
