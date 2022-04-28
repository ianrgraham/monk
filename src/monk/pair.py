from typing import Dict

import hoomd
import hoomd.pair_plugin.pair as p_pair
import numpy as np

def WCA_pot_smooth(r, rmin, rmax, epsilon, sigma):
    V = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6) + \
        epsilon - epsilon * (36*2**(-1/3.)*(r-rmax)**2) / sigma**2
    F = 4 * epsilon / r * (12 * (sigma / r)**12 - 6 * (sigma / r)**6) + \
        epsilon * (36*2**(-1/3.)*(r-rmax)) * 2 / sigma**2
    return (V, F)


def mLJ_pot_force_shifted(r, rmin, rmax, epsilon, sigma, delta):
    delt = delta*sigma
    sig = sigma - delt/np.power(2, 1./6)
    v = lambda x: 4 * epsilon * ( (sig / (x-delt))**12 - (sig / (x-delt))**6)
    shift = v(rmax)
    V = v(r) - shift
    F = 4 * epsilon / (r-delt) * ( 12 * (sig / (r-delt))**12 - 6 * (sig / (r-delt))**6)
    return (V, F)


def harm_pot(r, rmin, rmax, sigma):
    V = 0.5*(1-r/sigma)**2
    F = (1/sigma)*(1-r/sigma)
    return (V, F)


def hertz_pot(r, rmin, rmax, sigma):
    V = 0.4*np.power(1-r/sigma, 2.5)
    F = (1/sigma)*np.power(1-r/sigma, 1.5)
    return (V, F)


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
    r_on_cutoff = 0.0
    # specify Lennard-Jones interactions between particle pairs
    lj = hoomd.md.pair.LJ(nlist=nlist, mode="xplor")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff*sig_AA
    lj.r_on[('A', 'A')] = r_on_cutoff*sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff*sig_AB
    lj.r_on[('A', 'B')] = r_on_cutoff*sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff*sig_BB
    lj.r_on[('B', 'B')] = r_on_cutoff*sig_BB

    return lj


def bi_hertz(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Standard bidisperse hertzian potential
    '''
    eps_AA = 1.0
    eps_AB = 1.0
    eps_BB = 1.0
    sig_AA = 1
    sig_AB = 1.2
    sig_BB = 1.4
    r_on_cutoff = 0.0
    # specify Hertzian interactions between particle pairs
    hertz = p_pair.Hertzian(nlist)
    hertz.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    hertz.r_cut[('A', 'A')] = sig_AA
    hertz.r_on[('A', 'A')] = r_on_cutoff*sig_AA
    hertz.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    hertz.r_cut[('A', 'B')] = sig_AB
    hertz.r_on[('A', 'B')] = r_on_cutoff*sig_AB
    hertz.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    hertz.r_cut[('B', 'B')] = sig_BB
    hertz.r_on[('B', 'B')] = r_on_cutoff*sig_BB

    return hertz

def KA_WCA(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Wicks-Chandler-Anderson potential
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    r_on_cutoff = 0.0
    # specify Lennard-Jones interactions between particle pairs
    lj = hoomd.md.pair.LJ(nlist=nlist, mode="xplor")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff*sig_AA
    lj.r_on[('A', 'A')] = r_on_cutoff*sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff*sig_AB
    lj.r_on[('A', 'B')] = r_on_cutoff*sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff*sig_BB
    lj.r_on[('B', 'B')] = r_on_cutoff*sig_BB

    return lj

def KA_ModLJ(nlist: hoomd.md.nlist.NeighborList, delta: float) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential with modified well width
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    r_on_cutoff = 0.0
    # specify Lennard-Jones interactions between particle pairs
    lj = p_pair.ModLJ(nlist=nlist)
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA, delta=delta)
    lj.r_cut[('A', 'A')] = r_cutoff*sig_AA
    lj.r_on[('A', 'A')] = r_on_cutoff*sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB, delta=delta)
    lj.r_cut[('A', 'B')] = r_cutoff*sig_AB
    lj.r_on[('A', 'B')] = r_on_cutoff*sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB, delta=delta)
    lj.r_cut[('B', 'B')] = r_cutoff*sig_BB
    lj.r_on[('B', 'B')] = r_on_cutoff*sig_BB

    return lj

def LJ1208(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential with 12-8 modification
    '''
    r_cutoff = 2.5
    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    r_on_cutoff = 0.0
    # specify Lennard-Jones interactions between particle pairs
    lj = hoomd.md.pair.LJ1208(nlist=nlist, mode="xplor")
    lj.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    lj.r_cut[('A', 'A')] = r_cutoff*sig_AA
    lj.r_on[('A', 'A')] = r_on_cutoff*sig_AA
    lj.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    lj.r_cut[('A', 'B')] = r_cutoff*sig_AB
    lj.r_on[('A', 'B')] = r_on_cutoff*sig_AB
    lj.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    lj.r_cut[('B', 'B')] = r_cutoff*sig_BB
    lj.r_on[('B', 'B')] = r_on_cutoff*sig_BB

    return lj

# TODO need to update the table params to HOOMD v3
def _Harmonic(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Harmonic potential
    '''
    myPair = hoomd.md.pair.Table(width=1000, nlist=nlist)
    myPair.pair_coeff.set("A", "A", func=harm_pot, rmin=0.0,
                          rmax=5/6, coeff=dict(sigma=5/6))
    myPair.pair_coeff.set("B", "B", func=harm_pot, rmin=0.0,
                          rmax=7/6, coeff=dict(sigma=7/6))
    myPair.pair_coeff.set("A", "B", func=harm_pot, rmin=0.0,
                          rmax=1.0, coeff=dict(sigma=1.0))
    return myPair

# TODO need to update the table params to HOOMD v3
def _Hertzian(nlist: hoomd.md.nlist.NeighborList) -> hoomd.md.pair.Pair:
    '''Hertzian potential
    '''
    myPair = hoomd.md.pair.Table(width=1000, nlist=nlist)
    myPair.pair_coeff.set("A", "A", func=hertz_pot, rmin=0.0,
                          rmax=5/6, coeff=dict(sigma=5/6))
    myPair.pair_coeff.set("B", "B", func=hertz_pot, rmin=0.0,
                          rmax=7/6, coeff=dict(sigma=7/6))
    myPair.pair_coeff.set("A", "B", func=hertz_pot, rmin=0.0,
                          rmax=1.0, coeff=dict(sigma=1.0))
    return myPair