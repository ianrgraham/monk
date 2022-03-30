import argparse

from typing import Iterable, List, Optional, Sequence, Union, Tuple, Callable, Any
from inspect import signature

import numpy as np
import hoomd
import gsd.hoomd

from . import pair as monk_pair

class SimulationArgumentParser(argparse.ArgumentParser):
    """Command line parser for `hoomd` simulations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def init_rng(
        seed: Union[int, Sequence[int]]
        ) -> Union[np.random.Generator, Sequence[np.random.Generator]]:
    '''Simple helper function to spawn random number generators.'''
    return np.random.default_rng(seed)


def len_from_phi(N: int, phi: float, dim: int = 3):
    """Calculate regular box length for a given particle density"""
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    return np.power(N / phi, 1/dim)


def search_for_pair(pair: List) -> Tuple[Callable[..., hoomd.md.pair.Pair], Tuple]:
    """Search for function matching `pair` in `monk.pair`."""

    pair_len = len(pair)
    assert(pair_len >= 1)
    pair_name = pair[0]
    if pair_len > 1:
        pair_args = tuple(pair[1:])
    else:
        pair_args = tuple()

    pair_func: Any = getattr(monk_pair, pair_name)
    signatures = list(signature(pair_func).parameters.values())[1:]
    # TODO apply more assert statements regarding the function signature
    arguments = []
    for arg, sig in zip(pair_args, signatures):
        arguments.append(sig.annotation(arg))
    arguments_tuple = tuple(arguments)

    return (pair_func, arguments_tuple)

def vary_potential_parameters(
    sim: hoomd.Simulation,
    nlist: hoomd.md.nlist.NeighborList,
    pair_func: Callable[..., hoomd.md.pair.Pair],
    param_iter: Iterable[Tuple],
    steps: int,
):  
    """Interlace variations to a pair potential and the running of a simulation.

    Commonly used to equilibrate unstable parameter regions of a potential by
    starting from a stable state and slowly tuning the parameters. Of course,
    this function will mutate the simulation state forward in time, as well as
    repeatedly clearing and resetting the active forces in the system. Has the
    side effect of clearing any existing potentials of the system and leaving 
    the last applied potential active.

    Arguments
    ---------
        `sim`: Currently running simulation instance.
        `nlist`: Neighborlist to apply to the potential.
        `pair_func`: Function that returns a `Pair` object.
        `param_iter`: Iterable that can be expanded to the argument list of `pair_func`.
        `steps`: Number of simulation steps to be run after each iteration.
    """

    integrator: hoomd.md.Integrator = sim.operations.integrator

    for params in param_iter:
        integrator.forces.clear()
        pot_pair = pair_func(nlist, *params)
        integrator.forces.append(pot_pair)
        sim.run(steps)
        

def approx_euclidean_snapshot(
        N: int,
        L: float,
        rng: np.random.Generator,
        dim: int = 2,
        particle_types: Optional[List[str]] = None,
        ratios: Optional[List[int]] = None
        ) -> gsd.hoomd.Snapshot:
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
        `dim`: Physical dimension of the box (default=2).
        `particle_types`: List of particle labels (default=['A', 'B']).
        `ratios`: List of particle ratios (default=[50, 50]).

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
    assert len_types == len(ratios), "Lens of 'particle_types' and 'ratios' must match" 

    n = int(np.ceil(np.power(N, 1/dim)))
    x = np.linspace(-L/2, L/2, n, endpoint=False)
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