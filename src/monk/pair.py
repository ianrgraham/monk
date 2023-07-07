from typing import Dict

import hoomd
import hoomd.pair_plugin.pair as p_pair
import numpy as np

from typing import Optional


def KA_LJ(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88

    lj = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj


def Wahn(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Lennard-Jones potential with Wahn interaction parameters
    '''
    r_cutoff = 2.5
    eps_AA = 1.0
    eps_AB = 1.0
    eps_BB = 1.0
    sig_AA = 1.0
    sig_AB = 1.1
    sig_BB = 1.2

    lj = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj


def KA_LJ_DPD(nlist: hoomd.md.nlist.NeighborList,
              kT: float) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88

    lj = hoomd.md.pair.DPDLJ(nlist=nlist, kT=kT, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA, gamma=4.5)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB, gamma=4.5)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB, gamma=4.5)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj


def bi_hertz(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Standard bidisperse hertzian potential
    '''
    eps_AA = 1.0
    eps_AB = 1.0
    eps_BB = 1.0
    sig_AA = 14 / 12
    sig_AB = 1.0
    sig_BB = 10 / 12

    hertz = p_pair.Hertzian(nlist)
    hertz.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    hertz.r_cut[('A', 'A')] = sig_AA
    hertz.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    hertz.r_cut[('A', 'B')] = sig_AB
    hertz.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    hertz.r_cut[('B', 'B')] = sig_BB

    return hertz


def KA_WCA(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Wicks-Chandler-Anderson potential
    '''
    r_cutoff = 2.0**(1.0 / 6.0)  # minima of the LJ potential in units of sigma
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88

    lj = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj


def KA_ModLJ(nlist: hoomd.md.nlist.NeighborList,
             delta: float) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential with modified well width
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    lj = p_pair.ModLJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA, delta=delta)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB, delta=delta)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB, delta=delta)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj


def KA_MLJ(nlist: hoomd.md.nlist.NeighborList,
           delta: float) -> hoomd.md.pair.Pair:
    return KA_ModLJ(nlist, delta)


def KA_WLJ(nlist: hoomd.md.nlist.NeighborList,
           delta: float = 0.0,
           epsilon_a: float = 1.0,
           delta_a: Optional[float] = None,
           ) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential with modified well width and separate attractive/repulsive parts
    '''

    if delta_a is None:
        delta_a = delta

    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    lj = p_pair.WLJ(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA,
                                 delta=delta, epsilon_a=epsilon_a*eps_AA, delta_a=delta_a)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB,
                                 delta=delta, epsilon_a=epsilon_a*eps_AB, delta_a=delta_a)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB,
                                 delta=delta, epsilon_a=epsilon_a*eps_BB, delta_a=delta_a)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj


def KA_LJ1208(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential with 12-8 modification
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88

    lj = hoomd.md.pair.LJ1208(nlist=nlist, mode="shift")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff * sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff * sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff * sig_BB

    return lj
