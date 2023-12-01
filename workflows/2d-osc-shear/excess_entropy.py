from numba import njit
import numpy as np

def compute_rdf(nlist, bins=100, r_max=3):
    r = np.linspace(0, r_max, bins + 1)
    H, _ = np.histogram(nlist.distances, bins=r)
    bin_centers = (r[1:] + r[:-1]) / 2
    dr = r[1] - r[0]
    # normalize by volume
    H = H / (2 * np.pi * bin_centers * dr) / len(nlist.distances) * np.pi * r_max ** 2
    return bin_centers, H

@njit
def excess_entropy(r, g, dr):
    prelim = (g*np.log(g)-g+1)*r*dr
    prelim[np.isnan(prelim)] = 0
    return -np.pi*np.sum(prelim)

@njit
def excess_entropy_prelim(r, g, dr):
    prelim = (g*np.log(g)-g+1)*r*dr
    prelim[np.isnan(prelim)] = 0
    return -np.pi*prelim

@njit
def binary_excess_entropy(r, g_aa, g_ab, g_bb, f_a, f_b, dr):
    prelim_aa = f_a**2*excess_entropy_prelim(r, g_aa, dr)
    prelim_ab = 2*f_b*f_b*excess_entropy_prelim(r, g_ab, dr)
    prelim_bb = f_b**2*excess_entropy_prelim(r, g_bb, dr)
    return np.sum(prelim_aa + prelim_ab + prelim_bb)


@njit
def local_excess_entropy(r, sfs, dr):
    vol = 4*np.pi*(r[-1]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    for i in range(len(sfs)):
        sf = sfs[i,::2] + sfs[i,1::2]
        inv_n = 1/np.sum(sf)
        g = sf*inv_shell*inv_n*vol
        out[i] = excess_entropy(r, g, dr)
    return out


@njit
def local_excess_entropy_binary(r, sfs, dr):
    vol = 4*np.pi*(r[-1]**2)
    inv_shell = 1/(2*np.pi*r*dr)
    out = np.zeros(len(sfs))
    for i in range(len(sfs)):
        na = np.sum(sfs[i,::2])
        nb = np.sum(sfs[i,1::2])
        n = na + nb
        inv_na = 1/na
        inv_nb = 1/nb
        g_a = sfs[i,::2]*inv_shell*inv_na*vol
        g_b = sfs[i,1::2]*inv_shell*inv_nb*vol
        out[i] = na/n*excess_entropy(r, g_a, dr) + nb/n*excess_entropy(r, g_b, dr)
    return out

def excess_entropy_ka_2d(r, rdf_aa, rdf_ab, rdf_bb):
    return binary_excess_entropy(r, rdf_aa, rdf_ab, rdf_bb, 0.6, 0.4, r[1] - r[0])