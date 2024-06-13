"""Quasi-localized mode computer - local!"""

from scipy.sparse.linalg import eigsh
import scipy.linalg
import scipy.sparse as ssp
import gsd.hoomd
import numpy as np
import numba
from numba import njit
from numba.experimental import jitclass
from typing import Callable
from freud.locality import AABBQuery
from freud.box import Box
from tqdm import tqdm
import time

import jax
from jax import grad, jit, lax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')


class TypeParamDict(dict):

    def __init__(self, grad2_pots, grad3_pots, pot_factory):
        self._pot_factory = pot_factory
        self.grad2_pots = grad2_pots
        self.grad3_pots = grad3_pots
        super().__init__()

    def __setitem__(self, key, value):
        # convert to alpha
        key = tuple(sorted([ord(k.upper()) - 65 for k in key]))
        # key = tuple(sorted(key))
        # M_2 = D^2{ f(x) } = H
        g2 = jit(grad(jit(grad(self._pot_factory(**value)))))
        g3 = jit(grad(g2))  # M_3 = D{ H }
        self.grad2_pots[key] = g2
        self.grad3_pots[key] = g3
        super().__setitem__(key, value)


class Pair:
    """Pair potential from which quasi-localized modes may be calculated.

    This class leverages the auto-grad tool `jax` to automatically
    differentiate and JIT compile a given pair potential, enabling fast
    construction of the Hessian (dynamical matrix) to compute eigenvalues and
    eigenvectors.

    Arguments
    ---------
    - `pot_factory`: `Callable` - Factory function that takes in a variable
      number of parameters as arguments and returns a `jax` compatible pair
      potential of the type signature `(float) -> float`.

    Example
    -------
    ``` python

    def pot_factory(sigma):

        # define pair potential compatible with `jax`
        def _lambda(x):
            term = 1-x/sigma
            return 0.4/(sigma)*lax.sqrt(term)*(term)

        # handle any conditional using `lax.cond`
        return lambda x: (
            lax.cond(
                x < sigma,
                _lambda,
                lambda x: 0.0,
                x
            )
        )

    pair = Pair(pot_factory)

    pair.params[("A","A")] = dict(sigma=14/12)
    pair.params[("A","B")] = dict(sigma=1.0)
    pair.params[("B","B")] = dict(sigma=10/12)

    ```
    """

    def __init__(self, pot_factory: Callable):
        self._pot_factory = pot_factory
        self._cutoffs = {}
        self._grad2_pots = {}
        self._grad3_pots = {}

        self._typeparam_dict = TypeParamDict(self._grad2_pots,
                                             self._grad3_pots,
                                             self._pot_factory)

    @property
    def params(self):
        return self._typeparam_dict

    @property
    def cutoffs(self):
        return self._cutoffs

    def max_cutoff(self):
        max_cut = 0.0
        for cutoff in self._cutoffs.values():
            max_cut = max(max_cut, cutoff)
        return max_cut


class BidispHertz(Pair):

    def __init__(self):

        # it's necessary to have function that generates our pair potential
        # between types given a set of parameters
        def hertzian(sigma):

            # define pair potential compatible with `jax`
            def _lambda(x):
                term = 1 - x / sigma
                return 0.4 / (sigma) * lax.sqrt(term) * (term)

            # handle any conditional using `lax.cond`
            return lambda x: (lax.cond(x < sigma, _lambda, lambda x: 0.0, x))

        super().__init__(hertzian)

        # connect pair dictionary definitions and lazily produce hessian and
        # mode-filtering gradient functions
        self.params[("A", "A")] = dict(sigma=14 / 12)
        self.params[("A", "B")] = dict(sigma=1.0)
        self.params[("B", "B")] = dict(sigma=10 / 12)

        self.cutoffs[("A", "A")] = 14 / 12
        self.cutoffs[("A", "B")] = 1.0
        self.cutoffs[("B", "B")] = 10 / 12


class KobAndersenLJ(Pair):

    def __init__(self):

        def lj(epsilon, sigma):

            # define pair potential compatible with `jax`
            def _lambda(r):
                x = sigma / r
                x2 = x * x
                x4 = x2 * x2
                x6 = x4 * x2
                return 4 * epsilon * (x6 * x6 - x6)

            return _lambda

        super().__init__(lj)

        # connect pair dictionary definitions and lazily produce hessian and
        # mode-filtering gradient functions
        self.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
        self.params[("A", "B")] = dict(epsilon=1.5, sigma=0.8)
        self.params[("B", "B")] = dict(epsilon=0.5, sigma=0.88)

        self.cutoffs[("A", "A")] = 2.5
        self.cutoffs[("A", "B")] = 2.5 * 0.8
        self.cutoffs[("B", "B")] = 2.5 * 0.88


# NOTE ATM this function accepts a dense matrix as the hessian.
# It would be a whole lot more memory efficient if we used a sparse
# representation
@njit
def _compute_dense_hessian(edges, grad1_us, grad2_us, edge_vecs, dim, hessian):
    # loop over all edges in the system
    for edge_idx in np.arange(len(edges)):

        # don't forget the prefactor of 1/2 from overcounting
        grad1 = grad1_us[edge_idx] * np.eye(dim)
        k_vec = edge_vecs[edge_idx]
        k_outer = 2 * grad2_us[edge_idx] * np.outer(k_vec, k_vec) + grad1

        

        # loop over all combinations of the particles relating to the current
        # edge
        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                if i == j:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] += k_outer
                else:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] -= k_outer


@njit
def _make_coo_data(edges, grad1_us, grad2_us, edge_vecs, dim, N, row, col, data):

    idx = 0

    for edge_idx in np.arange(len(edges)):

        k_vec  = edge_vecs[edge_idx]
        k_outer = 2.0 * grad2_us[edge_idx] * np.outer(k_vec, k_vec) + grad1_us[edge_idx] * np.eye(dim)

        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                sign = 1.0
                if i != j:
                    sign = -1.0
                # hessian[i * dim:(i + 1) * dim,
                #         j * dim:(j + 1) * dim] -= k_outer
                d = sign * k_outer
                ii = i * dim
                jj = j * dim
                k = 0
                for l in range(dim):
                    for k in range(dim):
                        row[idx] = ii + k
                        col[idx] = jj + l
                        data[idx] = d[k, l]
                        idx += 1

                

def _compute_csr_hessian(edges, grad1_us, grad2_us, edge_vecs, dim, N):

    row = np.zeros(4*len(edges)*dim*dim, dtype=np.int32)
    col = np.zeros(4*len(edges)*dim*dim, dtype=np.int32)
    data = np.zeros(4*len(edges)*dim*dim, dtype=np.float64)
    _make_coo_data(edges, grad1_us, grad2_us, edge_vecs, dim, N, row, col, data)
    

    hessian = ssp.coo_array((data, (row, col)), shape=(N*dim, N*dim))
    return hessian.tocsr()


@njit
def _tensor_dot(v: np.ndarray, p: np.ndarray):
    """`v` is a rank-3 tensor of shape (l,n,m) and `p` is a rank-2 tensor of
    shape (n,m).
    This function contracts along indices n & m, resulting in a vector of size
    (l)."""
    shape = v.shape
    out = np.zeros(shape[0])
    for i in np.arange(shape[1]):
        for j in np.arange(shape[2]):
            out += v[:, i, j] * p[i, j]
    return out


@njit
def _filter_mode(vec, edges, u3s, v3s, dim, N):
    # the inner workings of this function are a little confusing.
    # the math is essentially contracting the input `vec`
    # (first transformed into a rank-2 tensor with an outer poduct)
    # with a rank-3 tensor that is constructed from
    # `u3s` and `v3s`. `u3s` are the 3rd-order radial derivates
    # of the pair potential, while `v3s` are the rank-3 tensor product
    # of the unit vector separating particles `i` and `j` found in an
    # edge

    # allocate space for the post-filtration vector
    filt_vec = np.zeros_like(vec)

    # perform a tensor product along the input vector `vec`
    self_outers = np.zeros((N, dim, dim))
    for idx in range(N):
        u1 = vec[idx * dim:(idx + 1) * dim]
        self_outers[idx] = np.outer(u1, u1)

    # now loop over edges, contracting a rank-3 tensor with the above tensor
    # product to get out the filtered vec
    for idx in np.arange(edges.shape[0]):
        edge = edges[idx]
        grad3_u = u3s[idx]
        v = v3s[idx]
        part_i = edge[0]
        part_j = edge[1]

        # everything below is basically unreadable tensor math
        u1 = vec[part_i * dim:(part_i + 1) * dim]
        u2 = vec[part_j * dim:(part_j + 1) * dim]

        v1 = self_outers[part_i]
        v2 = np.outer(u1, u2)
        v3 = self_outers[part_j]

        t1 = _tensor_dot(v, v1)
        t2 = _tensor_dot(v, v2)
        t3 = _tensor_dot(v, v3)

        out = grad3_u * (t1 - 2 * t2 + t3)  # and we finally have the answer!

        filt_vec[part_i * dim:(part_i + 1) * dim] += out
        filt_vec[part_j * dim:(part_j + 1) * dim] -= out

    return filt_vec


spec = [
    ('epsilon', numba.float64[:]),
    ('sigma', numba.float64[:]),
    ('cutoff', numba.float64[:]),
]
@jitclass(spec)
class KATypeParamDict:
    def __init__(self):
        self.epsilon = np.array([1.0, 1.5, 0.5])
        self.sigma = np.array([1.0, 0.8, 0.88])
        self.cutoff = np.array([2.5, 2.5*0.8, 2.5*0.88])

    @property
    def ntypes(self):
        return 2
    
    def get_params(self, i, j):
        return self.epsilon[i+j], self.sigma[i+j], self.cutoff[i+j]

@njit
def _lj_ka_grad_old(r, epsilon, sigma, cutoff):
    if r > cutoff:
        return 0.0
    x = sigma / r
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    return - 24.0 * epsilon * (2.0 * x6 * x6 - x6) / r

@njit
def _lj_ka_grad(r, epsilon, sigma, cutoff):
    if r > cutoff:
        return 0.0
    x = sigma / r
    x2 = x * x
    r2 = r * r
    x4 = x2 * x2
    x6 = x4 * x2
    return - 12.0 * epsilon * (2.0 * x6 * x6 - x6) / r2

@njit
def _lj_ka_grad2_old(r, epsilon, sigma, cutoff):
    if r > cutoff:
        return 0.0
    x = sigma / r
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    # comute the second derivative of the pair potential
    return 24.0 * epsilon * (26.0 * x6 * x6 - 7.0 * x6) / r**2

@njit
def _lj_ka_grad2(r, epsilon, sigma, cutoff):
    if r > cutoff:
        return 0.0
    x = sigma / r
    r2 = r * r
    r4 = r2 * r2
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    # comute the second derivative of the pair potential
    return epsilon * (168.0 * x6 * x6 - 48.0 * x6) / r4


@njit
def _lj_ka_grad3(r, epsilon, sigma, cutoff):
    if r > cutoff:
        return 0.0
    x = sigma / r
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    return - 24.0 * epsilon * (364.0 * x6 * x6 - 56.0 * x6) / r**3


@njit
def _compute_1gs_fast(edges, dists, types, params):

    grad1_us = np.zeros_like(dists)

    for idx, (edge, dist) in enumerate(zip(edges, dists)):
        type_i = types[edge[0]]
        type_j = types[edge[1]]
        epsilon, sigma, cutoff = params.get_params(type_i, type_j)
        grad1_u = _lj_ka_grad(dist, epsilon, sigma, cutoff)
        grad1_us[idx] = grad1_u

    return grad1_us


@njit
def _compute_2gs_fast(edges, dists, types, params):

    grad2_us = np.zeros_like(dists)

    for idx, (edge, dist) in enumerate(zip(edges, dists)):
        type_i = types[edge[0]]
        type_j = types[edge[1]]
        epsilon, sigma, cutoff = params.get_params(type_i, type_j)
        grad2_u = _lj_ka_grad2(dist, epsilon, sigma, cutoff)
        grad2_us[idx] = grad2_u

    return grad2_us


@njit
def _compute_3gs_fast(edges, unit_vecs, dists, types, dim, params):

    grad3_us = np.zeros_like(dists)

    grad3_ts = np.zeros((len(dists), dim, dim, dim))

    for idx, (edge, dist, vec) in enumerate(zip(edges, dists, unit_vecs)):
        type_i = types[edge[0]]
        type_j = types[edge[1]]
        epsilon, sigma, cutoff = params.get_params(type_i, type_j)
        grad3_u = _lj_ka_grad3(dist, epsilon, sigma, cutoff)

        grad3_us[idx] = grad3_u

        outer = np.outer(vec, vec)

        for i in range(dim):
            grad3_ts[idx, i] = vec[i] * outer

    return grad3_us, grad3_ts

@njit
def _spring_grad(r, k, r0):
    return 0.5 * k * (1 - r0/r)

@njit
def _spring_grad2(r, k, r0):
    return 0.25 * k * r0 / r**3

@njit
def _spring_grad_old(r, k, r0):
    return k * (r - r0)

@njit
def _spring_grad2_old(r, k, r0):
    return k

@njit
def _spring_grad3_old(r, k, r0):
    return 0.0

class QLM():
    """Computes the quasi-localized modes for a glassy configuration."""

    def __init__(self, pair: Pair):
        # nothing else to really do here.
        # NOTE we could instead keep a list of interactions (in place of a
        # single  pair-wise interaction).
        # Then we could compute the QLMs for system that have combinations of
        # bonded, non-bonded, anisotropic, diheadral, etc.
        self._pair = pair
        self._param_dict = KATypeParamDict()

    def _compute_1gs(self, edges, dists, types) -> np.ndarray:

        grad1_us = _compute_1gs_fast(edges, dists, types, self._param_dict)

        return grad1_us

    def _compute_2gs_slow(self, edges, dists, types) -> np.ndarray:

        grad2_us = np.zeros_like(dists)

        for idx, (edge, dist) in enumerate(zip(edges, dists)):
            type_i = types[edge[0]]
            type_j = types[edge[1]]
            grad2_u = self._pair._grad2_pots[tuple(sorted([type_i, type_j]))]
            grad2_us[idx] = grad2_u(dist)

        return grad2_us
    
    def _compute_2gs(self, edges, dists, types) -> np.ndarray:

        grad2_us = _compute_2gs_fast(edges, dists, types, self._param_dict)

        return grad2_us


    def _compute_3gs_slow(self, edges, unit_vecs, dists, types, dim) -> np.ndarray:

        grad3_us = np.zeros_like(dists)

        grad3_ts = np.zeros((len(dists), dim, dim, dim))

        for idx, (edge, dist, vec) in enumerate(zip(edges, dists, unit_vecs)):
            type_i = types[edge[0]]
            type_j = types[edge[1]]
            grad3_u = self._pair._grad3_pots[tuple(sorted([type_i, type_j]))]

            grad3_us[idx] = grad3_u(dist)

            grad3_ts[idx] = np.tensordot(vec, np.outer(vec, vec), axes=0)

        return grad3_us, grad3_ts
    
    def _compute_3gs(self, edges, unit_vecs, dists, types, dim) -> np.ndarray:

        grad3_us, grad3_ts = _compute_3gs_fast(edges, unit_vecs, dists, types, dim, self._param_dict)

        return grad3_us, grad3_ts

    def _compute_nlist(self, system):
        max_cutoff = self._pair.max_cutoff()

        query_args = dict(mode='ball', r_max=max_cutoff, exclude_ii=True)

        aq = AABBQuery.from_system(system)
        nlist = aq.query(aq.points, query_args).toNeighborList()

        edges = nlist[:]
        dists = nlist.distances[:]

        return edges, dists

    def _compute_uvecs(self, pos, edges, dists, box, dim):
        # unit_vecs = np.zeros((len(edges), dim))
        # for idx, (i, j) in enumerate(edges):
        #     unit_vecs[idx] = box.wrap(pos[j] - pos[i])[:dim] / dists[idx]
        # return unit_vecs
        unit_vecs = box.wrap(pos[edges[:, 1]] - pos[edges[:, 0]])[:, :dim]
        # TODO: I think this is a bug, there is no need to normalize these vectors
        # for i in range(dim):
        #     unit_vecs[:,i] /= dists
        return unit_vecs

    def compute(self,
                system: gsd.hoomd.Snapshot,
                k=10,
                filter=False,
                sigma=0,
                dense=False):
        """WARNING: Only use dense=True on small systems. """
        dim = system.configuration.dimensions
        N = system.particles.N
        box = Box.from_box(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)

        # TODO need to run a pass on the computed Hessian submatrices and
        # ensure no particles are rattlers (at least dim+1 contacts)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)

        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)

        if dense:
            hessian_dense = np.zeros((N * dim, N * dim))
            _compute_dense_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, hessian_dense)
            eig_vals, eig_vecs = scipy.linalg.eigh(hessian_dense)
            eig_vecs = list(eig_vecs.T)
        else:
            hessian_csr = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            eig_vals, eig_vecs = eigsh(hessian_csr, k=k, sigma=sigma)
            eig_vecs = list(eig_vecs.T)

        if filter:

            grad3_us, grad3_ts = self._compute_3gs(edges, unit_vecs, dists,
                                                   types, dim)

            filtered_vecs = [
                _filter_mode(v, edges, grad3_us, grad3_ts, dim, N)
                for v in eig_vecs
            ]

            reshaped_vecs = [v.reshape(N, dim) for v in filtered_vecs]

            del grad3_us, grad3_ts

            return eig_vals, reshaped_vecs

        else:
            return eig_vals, eig_vecs
        
    def compute_with_nlist(self,
                pos,
                dists,
                edges,
                types,
                box,
                k=10,
                filter=False,
                sigma=0,
                dense=False):
        """WARNING: Only use dense=True on small systems. """
        dim = pos.shape[1]
        N = len(pos)

        # TODO need to run a pass on the computed Hessian submatrices and
        # ensure no particles are rattlers (at least dim+1 contacts)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)

        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)

        if dense:
            hessian_dense = np.zeros((N * dim, N * dim))
            _compute_dense_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, hessian_dense)
            eig_vals, eig_vecs = scipy.linalg.eigh(hessian_dense)
            eig_vecs = list(eig_vecs.T)
        else:
            hessian_csr = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            eig_vals, eig_vecs = eigsh(hessian_csr, k=k, sigma=sigma)
            eig_vecs = list(eig_vecs.T)

        if filter:

            grad3_us, grad3_ts = self._compute_3gs(edges, unit_vecs, dists,
                                                   types, dim)

            filtered_vecs = [
                _filter_mode(v, edges, grad3_us, grad3_ts, dim, N)
                for v in eig_vecs
            ]

            reshaped_vecs = [v.reshape(N, dim) for v in filtered_vecs]

            del grad3_us, grad3_ts

            return eig_vals, reshaped_vecs

        else:
            return eig_vals, eig_vecs
        

    def compute_hessian_with_nlist(self,
                pos,
                dists,
                edges,
                types,
                box,
                dim,
                dense=False):
        """WARNING: Only use dense=True on small systems. """
        # dim = pos.shape[1]
        N = len(pos)

        # TODO need to run a pass on the computed Hessian submatrices and
        # ensure no particles are rattlers (at least dim+1 contacts)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)

        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)

        if dense:
            hessian_dense = np.zeros((N * dim, N * dim))
            _compute_dense_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, hessian_dense)
            return hessian_dense
        else:
            hessian_csr = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            return hessian_csr
        
    def filter_modes(self, system, eig_vecs):
        
        dim = system.configuration.dimensions
        N = system.particles.N
        box = Box.from_box(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)
        # print("a")
        grad3_us, grad3_ts = self._compute_3gs(edges, unit_vecs, dists,
                                                   types, dim)
        # print("b")
        # filtered_vecs = [
        #     _filter_mode(v, edges, grad3_us, grad3_ts, dim, N)
        #     for v in eig_vecs
        # ]
        filtered_vecs = []
        for v in tqdm(eig_vecs):
            filtered_vecs.append(_filter_mode(v, edges, grad3_us, grad3_ts, dim, N))
        # print("c")
        # reshaped_vecs = [v.reshape(N, dim) for v in filtered_vecs]

        del grad3_us, grad3_ts

        return filtered_vecs
    

    def filter_modes_slow(self, system, eig_vecs):
        
        dim = system.configuration.dimensions
        N = system.particles.N
        box = Box.from_box(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)
        # print("a")
        grad3_us, grad3_ts = self._compute_3gs_slow(edges, unit_vecs, dists,
                                                   types, dim)
        # print("b")
        # filtered_vecs = [
        #     _filter_mode(v, edges, grad3_us, grad3_ts, dim, N)
        #     for v in eig_vecs
        # ]
        filtered_vecs = []
        for v in tqdm(eig_vecs):
            filtered_vecs.append(_filter_mode(v, edges, grad3_us, grad3_ts, dim, N))
        # print("c")
        # reshaped_vecs = [v.reshape(N, dim) for v in filtered_vecs]

        del grad3_us, grad3_ts

        return filtered_vecs

        
    def compute_hessian(self,
                system: gsd.hoomd.Snapshot,
                dense=False):
        """WARNING: Only use dense=True on small systems. """
        dim = system.configuration.dimensions
        N = system.particles.N
        box = Box.from_box(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)

        # TODO need to run a pass on the computed Hessian submatrices and
        # ensure no particles are rattlers (at least dim+1 contacts)

        # NOTE this can probably be refactored to remove the for loop
        # use numba to compute array of naive dist_vecs for all pairs,
        # then pass this entire array to the freud.Box to wrap
        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)
        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)

        # now lets construct the hessian and convert it to a sparse
        # replresentation
        # NOTE I really should look into a more memory efficient approach. For
        # large systems the matrix might take up 10-100s of MB or more.
        # hessian_csr = ssp.csr_matrix(hessian_dense)
        # del hessian_dense, grad2_us

        if dense:
            hessian = np.zeros((N * dim, N * dim))
            _compute_dense_hessian(edges, grad2_us, unit_vecs, dim, hessian)
        else:
            hessian = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            
        return hessian
    
    def compute_partial(self,
                        system: gsd.hoomd.Snapshot,
                        dead_idx: int,
                        k=10,
                        filter=False,
                        sigma=0,
                        dense=False):
        
        dim = system.configuration.dimensions
        N = system.particles.N
        box = Box.from_box(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid


class MLWF:

    def __init__(self, pair: Pair):
        self._qlm = QLM(pair)

    def compute_eigs(self,
                system: gsd.hoomd.Snapshot,
                k=10,
                filter=False,
                sigma=0,
                dense=False):
        
        self._system = system
        self._dim = system.configuration.dimensions
        box = system.configuration.box
        self._box = Box.from_box(box)
        box[3:] = 0.0
        self._square_box = Box.from_box(box)
        self._k = k

        eig_vals, eig_vecs = self._qlm.compute(system,
                                               k=k,
                                               filter=filter,
                                               sigma=sigma,
                                               dense=dense)
        
        self._eig_vals = eig_vals
        self._eig_vecs = eig_vecs

    def localization_func(self, vecs, pos, box, dim):
        # this impl is a bit naive, but it should do for now (and be correct
        # for square boxes)

        assert dim == 2

        # L = np.array([box.Lx, box.Ly])
        N = len(pos)

        unit_box = Box.from_box([1, 1, 0, 0, 0, 0])

        omega = 0

        for vec in vecs:
            mean_eta = np.zeros(dim)
            mean_ce = np.zeros(dim)

            for p, v in zip(pos, vec):

                vnorm = np.linalg.norm(v)
                theta = p*2*np.pi

                eta = np.cos(theta)
                ce = np.sin(theta)

                mean_eta += eta * vnorm
                mean_ce += ce * vnorm

            mean_theta = np.arctan2(mean_ce, mean_eta) + np.pi
            com = mean_theta/(2*np.pi)

            diffs = unit_box.wrap(pos - com)
            vnorm = np.linalg.norm(vec, axis=1)
            var = np.sum(np.mean((diffs*vnorm)**2, axis=0))

            omega += var

        return omega
    
    def compute_gradient(self, vecs, pos, box, dim):

        assert dim == 2

        # L = np.array([box.Lx, box.Ly])
        N = len(pos)

        unit_box = Box.from_box([1, 1, 0, 0, 0, 0])

        d_omegas = []
        d_rs = []

        for vec in vecs:
            mean_eta = np.zeros(dim)
            mean_ce = np.zeros(dim)

            for p, v in zip(pos, vec):

                vnorm = np.linalg.norm(v)
                theta = p*2*np.pi

                eta = np.cos(theta)
                ce = np.sin(theta)

                mean_eta += eta * vnorm
                mean_ce += ce * vnorm

            mean_theta = np.arctan2(mean_ce, mean_eta) + np.pi
            com = mean_theta/(2*np.pi)

            diffs = unit_box.wrap(pos - com)
            vnorm = np.linalg.norm(vec, axis=1)
            r = diffs*vnorm
            d_rs.append(r)

            d_omegas.append()

        return omega

    def solve_mlfs(self, dt, n_iter, sigma=0):
        # get unitary transformation
        # sigma is a noise parameter
        pass
        pos = self._system.particles.position
        box = self._box
        square = self._square_box
        pos = box.make_fractional(pos)[:, :self._dim]
        

        for i in range(self._k):
            pass
