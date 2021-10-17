from typing import List, Sequence, Union

import numpy as np
import hoomd
import gsd.hoomd


def init_rng(
        seed: Union[int, Sequence[int]]
        ) -> Union[np.random.Generator, Sequence[np.random.Generator]]:
    '''Simple helper function to spawn random number generators
    '''
    return np.random.default_rng(seed)


def approx_euclidean_snapshot(
        N: int,
        L: float,
        rng: np.random.Generator,
        dim: int = 2,
        particle_types: List[chr] = ['A', 'B'],
        ratios: List[int] = [50, 50]
        ) -> gsd.hoomd.Snapshot:
    '''Constucts hoomd simulation snapshot with regularly spaced particles.

    Arguments
    ---------
    * `N`: Number of particles\n
    * `L`: Side length of the simulation box\n
    * `dim`: Physical dimension of the box (default=2)\n
    * `particle_types`: List of particle labels (default=['A', 'B'])
    * `ratios`: List of particle ratios (default=[50, 50])
    '''

    # only valid dims in hoomd are 2 and 3
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    assert L > 0, "Box length cannot be <= 0"
    assert N > 0, "Number of particles cannot be <= 0"
    len_types = len(particle_types)
    assert np.sum(ratios) == 100, "Ratios must sum to 100"
    assert len_types == len(ratios), "Lens of 'particle_types' and 'ratios' must match" 

    n = int(np.ceil(np.power(N, 1/dim)))
    x = np.linspace(-L/2, L/2, n, endpoint=False)
    X = [x for _ in range(dim)]
    if dim == 2:
        X.append([0])
    grid = np.meshgrid(*X)
    grid = [x.flatten() for x in grid]
    pos = np.stack(grid, axis=-1)

    if dim == 2:
        Lz = 0
    else:
        Lz = L
    # build snapshot and populate particles positions
    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = N
    snapshot.particles.position = pos[:N]
    snapshot.configuration.box = [L, L, Lz, 0, 0, 0]
    snapshot.particles.types = particle_types
    snapshot.particles.typeid = [0] * N

    # assign particle labels with rng
    idx = 0
    limits = np.cumsum(ratios)
    j = 0
    for i in rng.permutation(np.arange(N)):
        snapshot.particles.typeid[i] = idx
        if j/N*100 > limits[idx]:
            idx += 1
        j += 1

    return snapshot