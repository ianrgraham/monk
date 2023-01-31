# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.pair_plugin.pair import *
import hoomd

import itertools
import pytest
import numpy as np


def hertzian(dx, params, r_cut, shift=False):
    epsilon = params["epsilon"]
    sigma = params["sigma"]

    dr = np.linalg.norm(dx)

    if dr >= r_cut:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0

    f = epsilon / sigma * np.power(1 - dr / sigma, 1.5) * np.array(dx, dtype=np.float64) / dr
    e = 0.4 * np.power(1 - dr / sigma, 2.5)

    return f, e


def mlj(dx, params, r_cut, shift=False):
    epsilon = params["epsilon"]
    sigma = params["sigma"]
    delta = params["delta"]

    dsigma = (1.0 - delta/2.0**(1/6)) * sigma

    dr = np.linalg.norm(dx)

    if dr >= r_cut:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0

    f = 4.0 * epsilon * (12.0 * dsigma ** 12 / (dr - delta) ** 13 - 6.0 * dsigma ** 6 / (dr - delta) ** 7) * np.array(dx, dtype=np.float64) / dr
    e = 4.0 * epsilon * (dsigma ** 12 / (dr - delta) ** 12 - dsigma ** 6 / (dr - delta) ** 6)
    if shift:
        e -= 4.0 * epsilon * (dsigma ** 12 / (r_cut - delta) ** 12 - dsigma ** 6 / (r_cut - delta) ** 6)

    return f, e


def wlj(dx, params, r_cut, shift=False):
    epsilon = params["epsilon_a"]
    sigma = params["sigma"]
    delta = params["delta_a"]
    e_shift = 0.0

    dr = np.linalg.norm(dx)

    if dr < sigma * 2 ** (1/6):
        epsilon = params["epsilon"]
        delta = params["delta"]
        e_shift = -params["epsilon_a"] + params["epsilon"]

    dsigma = (1.0 - delta/2.0**(1/6)) * sigma

    if dr >= r_cut:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0

    f = 4.0 * epsilon * (12.0 * dsigma ** 12 / (dr - delta) ** 13 - 6.0 * dsigma ** 6 / (dr - delta) ** 7) * np.array(dx, dtype=np.float64) / dr
    e = 4.0 * epsilon * (dsigma ** 12 / (dr - delta) ** 12 - dsigma ** 6 / (dr - delta) ** 6) + e_shift
    if shift:
        e -= 4.0 * epsilon * (dsigma ** 12 / (r_cut - delta) ** 12 - dsigma ** 6 / (r_cut - delta) ** 6)

    return f, e

# Build up list of parameters.
distances = np.linspace(1.0, 2.0, 5)
pair_list = [Hertzian, MLJ, WLJ]
pot_list = [hertzian, mlj, wlj]
keys_list = [
    ("epsilon", "sigma"),
    ("epsilon", "sigma", "delta"),
    ("epsilon", "sigma", "delta", "epsilon_a", "delta_a")
]
mins = [0.5, 0.5, 0.0, 0.0, 0.0]
maxs = [1.5, 1.0, 0.4, 1.0, 0.4]
# No need to test "xplor", as that is handled outside of the plugin impl.
modes = ["none", "shift"]

params = []
for pair, pot, keys in list(zip(pair_list, pot_list, keys_list)):
    key_values = []
    for i in range(len(keys)):
        key = keys[i]
        key_values.append(np.linspace(mins[i], maxs[i], 2))
    print(keys)
    print(key_values)
    
    all_values = list(itertools.product(*key_values))
    for values in all_values:
        if pot == hertzian:
            r_cut = values[1]
        else:
            r_cut = values[1]*2.5
        pot_dict = {}
        for i in range(len(keys)):
            pot_dict[keys[i]] = values[i]
        params.append((pair, pot, pot_dict, r_cut))

testdata = list(itertools.product(distances, params, modes))


@pytest.mark.parametrize("distance, params, mode", testdata)
def test_force_and_energy_eval(simulation_factory,
                               two_particle_snapshot_factory, distance,
                               params, mode):

    # Build the simulation from the factory fixtures defined in
    # hoomd/conftest.py.
    sim = simulation_factory(two_particle_snapshot_factory(d=distance))

    pair = params[0]
    pot = params[1]
    pair_params = params[2]
    r_cut = params[3]

    # Setup integrator and force.
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.NVE(hoomd.filter.All())

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    
    example_pair: hoomd.md.pair.Pair = pair(
        cell, default_r_cut=r_cut, mode=mode)
    example_pair.params[("A", "A")] = pair_params
    integrator.forces = [example_pair]
    integrator.methods = [nve]

    sim.operations.integrator = integrator

    sim.run(0)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        vec_dist = snap.particles.position[1] - snap.particles.position[0]

        # Compute force and energy from Python
        shift = mode == "shift"
        f, e = pot(vec_dist, pair_params, r_cut, shift)
        e /= 2.0

    # Test that the forces and energies match that predicted by the Python
    # implementation.
    forces = example_pair.forces
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(forces, [-f, f], decimal=4)

    energies = example_pair.energies
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(energies, [e, e], decimal=4)
