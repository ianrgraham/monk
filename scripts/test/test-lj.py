from typing import List, Optional, Sequence, Union
import time

import numpy as np
from numpy.typing import ArrayLike
import hoomd
import gsd.hoomd


def init_rng(
    seed: Union[int, Sequence[int]]
) -> Union[np.random.Generator, Sequence[np.random.Generator]]:
    '''Simple helper function to spawn random number generators.'''
    return np.random.default_rng(seed)


def len_from_phi(N: int, phi: float, dim: int = 3):
    """Calculate regular box length for a given particle density"""
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    return np.power(N / phi, 1 / dim)


def len_from_vol_frac(diams: ArrayLike, vol_frac: float, dim: int = 3):
    """Calculate regular box length for a given volume fraction"""
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    part_vol = np.sum(np.square(diams / 2) * np.pi)
    return np.power(part_vol / vol_frac, 1 / dim)


def approx_euclidean_snapshot(
        N: int,
        L: float,
        rng: np.random.Generator,
        dim: int = 3,
        particle_types: Optional[List[str]] = None,
        ratios: Optional[List[int]] = None,
        diams: Optional[List[float]] = None) -> gsd.hoomd.Snapshot:
    '''Constucts hoomd simulation snapshot with regularly spaced particles on a
    euclidian lattice.

    Easy way to initialize simulations states for pair potentials like
    Lennard-Jones where large particle overlaps can leading to particles being
    ejected from the box. Simulation should be thoroughly equilibrated after
    such setup.

    Arguments
    ---------
        `N`: Number of particles.
        `L`: Side length of the simulation box.
        `rng`: `numpy` RNG to choose where to place species.
        `dim`: Physical dimension of the box (default=3).
        `particle_types`: List of particle labels (default=['A', 'B']).
        `ratios`: List of particle ratios (default=[50, 50]).
        `diams`: List of particle diameters for visualization.

    Returns
    -------
        `Snapshot`: A valid `hoomd` simulation state.
    '''

    if particle_types is None:
        particle_types = ['A', 'B']

    if ratios is None:
        ratios = [50, 50]

    # only valid dims in hoomd are 2 and 3
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    assert L > 0, "Box length cannot be <= 0"
    assert N > 0, "Number of particles cannot be <= 0"
    len_types = len(particle_types)
    assert np.sum(ratios) == 100, "Ratios must sum to 100"
    assert len_types == len(
        ratios), "Lens of 'particle_types' and 'ratios' must match"
    if diams is not None:
        assert len_types == len(diams)

    n = int(np.ceil(np.power(N, 1 / dim)))
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X = [x for _ in range(dim)]
    if dim == 2:
        X.append(np.zeros(1))
    grid = np.meshgrid(*X)
    grid = [x.flatten() for x in grid]
    pos = np.stack(grid, axis=-1)

    if dim == 2:
        Lz = 0.0
    else:
        Lz = L
    # build snapshot and populate particles positions
    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = N
    snapshot.particles.position = pos[:N]
    snapshot.configuration.box = [L, L, Lz, 0.0, 0.0, 0.0]
    snapshot.particles.types = particle_types
    snapshot.particles.typeid = [0] * N
    snapshot.particles.diameter = [0] * N

    # assign particle labels with rng
    idx = 0
    limits = np.cumsum(ratios)
    j = 0
    for i in rng.permutation(np.arange(N)):
        while j / N * 100 >= limits[idx]:
            idx += 1
        snapshot.particles.typeid[i] = idx
        snapshot.particles.diameter[i] = diams[idx]

        j += 1

    return snapshot


def quick_sim(
        N: int,
        phi: float,
        device: hoomd.device.Device,
        dim: int = 3,
        particle_types: Optional[List[str]] = None,
        ratios: Optional[List[int]] = None,
        diams: Optional[List[float]] = None,
        seed: int = 0) -> hoomd.Simulation:
    '''Construct a hoomd simulation using `approx_euclidean_snapshot`.
    
    Arguments
    ---------
        `N`: Number of particles.
        `phi`: # density for a cubic (or square) simulation box.
        `rng`: `numpy` RNG to choose where to place species.
        `dim`: Physical dimension of the box (default=3).
        `particle_types`: List of particle labels (default=['A', 'B']).
        `ratios`: List of particle ratios (default=[50, 50]).
        `diams`: List of particle diameters for visualization.

    Returns
    -------
        `Snapshot`: A valid `hoomd` simulation state.
    '''
    
    rng = np.random.default_rng(seed + 1) # offset rng seed by 1

    sim = hoomd.Simulation(device, seed)

    L = len_from_phi(N, phi, dim)

    snapshot = approx_euclidean_snapshot(
        N,
        L,
        rng,
        dim,
        particle_types,
        ratios,
        diams)
    
    sim.create_state_from_snapshot(snapshot)

    return sim


if __name__ == "__main__":
    gpu = hoomd.device.GPU([0])
    sim = quick_sim(
        10_000,
        1.2,
        gpu,
        particle_types=['A', 'B'],
        ratios=[80,20],
        diams=[1.0, 0.88]
    )

    integrator = hoomd.md.Integrator(dt=0.0025)
    nvt = hoomd.md.methods.NVT(hoomd.filter.All(), kT=1.5, tau=0.25)
    tree = hoomd.md.nlist.Tree(buffer=0.3)
    lj = hoomd.md.pair.LJ(nlist=tree, default_r_cut=2.5)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    lj.params[('A', 'B')] = dict(epsilon=1.5, sigma=0.8)
    lj.params[('B', 'B')] = dict(epsilon=0.5, sigma=0.88)

    integrator.methods.append(nvt)
    integrator.forces.append(lj)

    sim.operations.integrator = integrator

    start = time.time()
    sim.run(10_000)
    print(time.time() - start)

