import hoomd
import numpy as np


def WCA_pot_smooth(r, rmin, rmax, epsilon, sigma):
    V = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6) + \
        epsilon - epsilon * (36*2**(-1/3.)*(r-rmax)**2) / sigma**2
    F = 4 * epsilon / r * (12 * (sigma / r)**12 - 6 * (sigma / r)**6) + \
        epsilon * (36*2**(-1/3.)*(r-rmax)) * 2 / sigma**2
    return (V, F)


def mLJ_pot_shifted(r, rmin, rmax, epsilon, sigma, delta):
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


def LJ(NeighborList: hoomd.md.nlist.NList) -> hoomd.md.pair.Pair:
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
    myLjPair = hoomd.md.pair.LJ(nlist=NeighborList, mode="xplor")
    myLjPair.params[('A', 'A')] = dict(epsilon=eps_AA, sigma=sig_AA)
    myLjPair.r_cut[('A', 'A')] = r_cutoff*sig_AA
    myLjPair.r_on[('A', 'A')] = r_on_cutoff*sig_AA
    myLjPair.params[('A', 'B')] = dict(epsilon=eps_AB, sigma=sig_AB)
    myLjPair.r_cut[('A', 'B')] = r_cutoff*sig_AB
    myLjPair.r_on[('A', 'B')] = r_on_cutoff*sig_AB
    myLjPair.params[('B', 'B')] = dict(epsilon=eps_BB, sigma=sig_BB)
    myLjPair.r_cut[('B', 'B')] = r_cutoff*sig_BB
    myLjPair.r_on[('B', 'B')] = r_on_cutoff*sig_BB

    return myLjPair


def WCA(NeighborList: hoomd.md.nlist.NList) -> hoomd.md.pair.Pair:
    '''WCA potential
    '''
    r_cutoff = 2**(1./6)
    r_min = 0.722462048309373
    TABLEWIDTH = 1000  # number of particle pairs

    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    # specify WCA interactions between particle pairs
    myLjPair = hoomd.md.pair.Table(width=TABLEWIDTH, nlist=NeighborList)
    myLjPair.pair_coeff.set('A', 'A',
                            func=WCA_pot_smooth,
                            rmin=r_min*sig_AA,
                            rmax=r_cutoff*sig_AA,
                            coeff=dict(epsilon=eps_AA, sigma=sig_AA))
    myLjPair.pair_coeff.set('A', 'B',
                            func=WCA_pot_smooth,
                            rmin=r_min*sig_AB,
                            rmax=r_cutoff*sig_AB,
                            coeff=dict(epsilon=eps_AB, sigma=sig_AB))
    myLjPair.pair_coeff.set('B', 'B',
                            func=WCA_pot_smooth,
                            rmin=r_min*sig_BB,
                            rmax=r_cutoff*sig_BB,
                            coeff=dict(epsilon=eps_BB, sigma=sig_BB))
    return myLjPair


def mLJ(NeighborList: hoomd.md.nlist.NList, delta: float) -> hoomd.md.pair.Pair:
    '''Kob-Anderson Lennard-Jones potential with modified well width
    '''
    r_cutoff = 2.5
    r_min = 0.722462048309373

    eps_AA = 1
    eps_AB = 1.5
    eps_BB = 0.5
    sig_AA = 1
    sig_AB = 0.8
    sig_BB = 0.88
    # specify WCA interactions between particle pairs
    myLjPair = hoomd.md.pair.Table(nlist=NeighborList)
    myLjPair.pair_coeff.set('A', 'A',
                            func=mLJ_pot_shifted,
                            rmin=r_min*sig_AA,
                            rmax=r_cutoff*sig_AA,
                            coeff=dict(epsilon=eps_AA, sigma=sig_AA, delta=delta))
    myLjPair.pair_coeff.set('A', 'B',
                            func=mLJ_pot_shifted,
                            rmin=r_min*sig_AB,
                            rmax=r_cutoff*sig_AB,
                            coeff=dict(epsilon=eps_AB, sigma=sig_AB, delta=delta))
    myLjPair.pair_coeff.set('B', 'B',
                            func=mLJ_pot_shifted,
                            rmin=r_min*sig_BB,
                            rmax=r_cutoff*sig_BB,
                            coeff=dict(epsilon=eps_BB, sigma=sig_BB, delta=delta))
    return myLjPair


def Harmonic(NeighborList: hoomd.md.nlist.NList) -> hoomd.md.pair.Pair:
    '''Harmonic potential
    '''
    myPair = hoomd.md.pair.Table(width=1000, nlist=NeighborList)
    myPair.pair_coeff.set("A", "A", func=harm_pot, rmin=0.0,
                          rmax=5/6, coeff=dict(sigma=5/6))
    myPair.pair_coeff.set("B", "B", func=harm_pot, rmin=0.0,
                          rmax=7/6, coeff=dict(sigma=7/6))
    myPair.pair_coeff.set("A", "B", func=harm_pot, rmin=0.0,
                          rmax=1.0, coeff=dict(sigma=1.0))
    return myPair


def Hertzian(NeighborList: hoomd.md.nlist.NList) -> hoomd.md.pair.Pair:
    '''Hertzian potential
    '''
    myPair = hoomd.md.pair.Table(width=1000, nlist=NeighborList)
    myPair.pair_coeff.set("A", "A", func=hertz_pot, rmin=0.0,
                          rmax=5/6, coeff=dict(sigma=5/6))
    myPair.pair_coeff.set("B", "B", func=hertz_pot, rmin=0.0,
                          rmax=7/6, coeff=dict(sigma=7/6))
    myPair.pair_coeff.set("A", "B", func=hertz_pot, rmin=0.0,
                          rmax=1.0, coeff=dict(sigma=1.0))
    return myPair