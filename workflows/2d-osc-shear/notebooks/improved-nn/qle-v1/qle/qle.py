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
import cupyx.scipy.sparse as csp
import cupy as cp
import cupyx as cpx
from cupyx import cusparse
from cupyx.scipy.sparse.linalg import eigsh as ceigsh

from .data import numba_scalar, np_scalar, hess_scalar
from .ka_lj import KATypeParamDict
from .linalg import _compute_dense_hessian, _compute_csr_hessian, _compute_dense_hessian_jac, _compute_sparse_hessian_jac, _filter_modes_new_dense, filter_sparse_new
from .boxdim import BoxDim

@njit
def _compute_1gs_fast(edges, dists, types, params):

    grad1_us = np.zeros_like(dists, dtype=np_scalar)

    for idx, (edge, dist) in enumerate(zip(edges, dists)):
        type_i = types[edge[0]]
        type_j = types[edge[1]]
        epsilon, sigma, cutoff = params.get_params(type_i, type_j)
        grad1_u = params._grad(dist, epsilon, sigma, cutoff)
        grad1_us[idx] = grad1_u

    return grad1_us


@njit
def _compute_2gs_fast(edges, dists, types, params):

    grad2_us = np.zeros_like(dists, dtype=np_scalar)

    for idx, (edge, dist) in enumerate(zip(edges, dists)):
        type_i = types[edge[0]]
        type_j = types[edge[1]]
        epsilon, sigma, cutoff = params.get_params(type_i, type_j)
        grad2_u = params._grad2(dist, epsilon, sigma, cutoff)
        grad2_us[idx] = grad2_u

    return grad2_us

@njit
def _compute_3gs_fast(edges, unit_vecs, dists, types, dim, params):

    grad3_us = np.zeros_like(dists)
    # grad3_ts = np.zeros(len(dists))  

    for idx, (edge, dist, vec) in enumerate(zip(edges, dists, unit_vecs)):
        type_i = types[edge[0]]
        type_j = types[edge[1]]
        epsilon, sigma, cutoff = params.get_params(type_i, type_j)
        grad3_u = params._grad3(dist, epsilon, sigma, cutoff)

        grad3_us[idx] = grad3_u

    return grad3_us


class QLM():
    """Computes the quasi-localized modes for a glassy configuration."""

    def __init__(self, parm_dict=None):
        if parm_dict is not None:
            self._param_dict = parm_dict
        else:
            self._param_dict = KATypeParamDict()

    def max_cutoff(self):
        return self._param_dict.max_cutoff()

    def _compute_1gs(self, edges, dists, types) -> np.ndarray:

        grad1_us = _compute_1gs_fast(edges, dists, types, self._param_dict)

        return grad1_us
    
    def _compute_2gs(self, edges, dists, types) -> np.ndarray:

        grad2_us = _compute_2gs_fast(edges, dists, types, self._param_dict)

        return grad2_us
    
    def _compute_3gs(self, edges, unit_vecs, dists, types, dim) -> np.ndarray:

        grad3_us = _compute_3gs_fast(edges, unit_vecs, dists, types, dim, self._param_dict)

        return grad3_us

    def _compute_nlist(self, system):
        max_cutoff = self.max_cutoff()

        query_args = dict(mode='ball', r_max=max_cutoff, exclude_ii=True)

        aq = AABBQuery.from_system(system)
        nlist = aq.query(aq.points, query_args).toNeighborList()

        edges = nlist[:]
        dists = nlist.distances[:]

        return edges, dists

    def _compute_uvecs(self, pos, edges, dists, box, dim):
        unit_vecs = box.wrap(pos[edges[:, 1]] - pos[edges[:, 0]])[:, :dim]
        return unit_vecs

    def compute(self,
                system: gsd.hoomd.Frame,
                k=10,
                filter=False,
                sigma=0,
                dense=False,
                dense_filter=False,
                alt=True,
                gpu=False,
                dont_apply_filter=False,
                evals_only=False,
                bsr_size=None):
        """WARNING: Only use dense=True on small systems. """
        dim = system.configuration.dimensions
        N = system.particles.N
        box = BoxDim(np.array(system.configuration.box, dtype=np_scalar))
        pos = system.particles.position.astype(np_scalar)
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)
        dists = dists.astype(np_scalar)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)

        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)

        if dense:
            hessian_dense = np.zeros((N * dim, N * dim), dtype=np_scalar)

            _compute_dense_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N, hessian_dense)
            # if hess_scalar != np_scalar:
            #     hessian
            if evals_only:
                eig_vals = scipy.linalg.eigvalsh(hessian_dense)
                return eig_vals
            eig_vals, eig_vecs = scipy.linalg.eigh(hessian_dense)
            eig_vecs = eig_vecs
            hessian = hessian_dense
        else:
            hessian_csr = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            start = time.time()
            if hess_scalar != np_scalar:
                print("converted")
                hessian_csr = hessian_csr.astype(hess_scalar)
            if gpu:
                hessian_gpu = cp.sparse.csr_matrix(hessian_csr)
                # if bsr_size is not None:
                #     hessian_gpu = cusparse.csr2bsr(hessian_gpu, bsr_size)
                if evals_only:
                    evals = ceigsh(hessian_gpu, k=k, which="SA", return_eigenvectors=False)
                    print("eigsh time:", time.time() - start)
                    return evals.get()
                eig_vals, eig_vecs = ceigsh(hessian_gpu, k=k, which="SA")
                eig_vals = eig_vals.get()
                eig_vecs = eig_vecs.get()
            else:
                if evals_only:
                    evals = eigsh(hessian_csr, k=k, sigma=sigma, return_eigenvectors=False)
                    print("eigsh time:", time.time() - start)
                    return evals
                eig_vals, eig_vecs = eigsh(hessian_csr, k=k, sigma=sigma)
            print("eigsh time:", time.time() - start)
            eig_vecs = eig_vecs
            hessian = hessian_csr

        if filter or dense_filter:
            start = time.time()
            grad3_us = self._compute_3gs(edges, unit_vecs, dists, types, dim)
            if dense_filter:
                hessian_jac = np.zeros((N * dim, N * dim, N * dim), dtype=np_scalar)
                _compute_dense_hessian_jac(edges, grad2_us, grad3_us, unit_vecs, dim, hessian_jac)
                if dont_apply_filter:
                    return eig_vals, eig_vecs, hessian, hessian_jac
                filtered_vecs = [
                    _filter_modes_new_dense(v, hessian_jac)
                    for v in eig_vecs.T
                ]
                reshaped_vecs = np.array(filtered_vecs).T

                # del grad3_us, grad3_ts

                print("filter time:", time.time() - start)

                return eig_vals, reshaped_vecs, hessian, hessian_jac
            else:
                cond = edges[:, 0] < edges[:, 1]
                reduced_edges = edges[cond]
                grad2_us = grad2_us[cond]
                grad3_us = grad3_us[cond]
                unit_vecs = unit_vecs[cond]
                hessian_jac = _compute_sparse_hessian_jac(reduced_edges, grad2_us, grad3_us, unit_vecs, dim)
                if dont_apply_filter:
                    return eig_vals, eig_vecs, hessian, hessian_jac, reduced_edges
                filtered_vecs = [
                    filter_sparse_new(reduced_edges, np.ascontiguousarray(v), hessian_jac, dim, N)
                    for v in eig_vecs.T
                ]

                reshaped_vecs = np.array(filtered_vecs).T

                # del grad3_us, grad3_ts
                print("filter time:", time.time() - start)


                return eig_vals, reshaped_vecs, hessian, hessian_jac, reduced_edges

        else:
            return eig_vals, eig_vecs, hessian
        
    def compute_with_nlist(self,
                pos,
                dists,
                edges,
                types,
                box,
                dim=None,
                k=10,
                filter=False,
                dense_filter=False,
                sigma=0,
                dense=False,
                alt=True,
                N=None,
                gpu=False):
        """WARNING: Only use dense=True on small systems. """
        if dim is None:
            dim = pos.shape[1]
        if N is None:
            N = len(pos)

        # TODO need to run a pass on the computed Hessian submatrices and
        # ensure no particles are rattlers (at least dim+1 contacts)

        pos = pos.astype(np_scalar)
        dists = dists.astype(np_scalar)
        if alt:
            unit_vecs = (pos[edges[:, 1]] - pos[edges[:, 0]])[:, :dim]
        else:
            unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)#.astype(np_scalar)

        grad1_us = self._compute_1gs(edges, dists, types)#.astype(np_scalar)
        grad2_us = self._compute_2gs(edges, dists, types)#.astype(np_scalar)



        if dense:
            hessian_dense = np.zeros((N * dim, N * dim), dtype=np_scalar)
            _compute_dense_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N, hessian_dense)
            eig_vals, eig_vecs = scipy.linalg.eigh(hessian_dense)
            eig_vecs = eig_vecs
            hessian = hessian_dense
        else:
            hessian_csr = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            start = time.time()
            if gpu:
                hessian_gpu = cp.sparse.csr_matrix(hessian_csr)
                eig_vals, eig_vecs = ceigsh(hessian_gpu, k=k, which="SA")
                eig_vals = eig_vals.get()
                eig_vecs = eig_vecs.get()
            else:
                eig_vals, eig_vecs = eigsh(hessian_csr, k=k, sigma=sigma)
            print("eigsh time:", time.time() - start)
            eig_vecs = eig_vecs
            hessian = hessian_csr

        if filter or dense_filter:
            start = time.time()
            grad3_us = self._compute_3gs(edges, unit_vecs, dists, types, dim)
            if dense_filter:
                hessian_jac = np.zeros((N * dim, N * dim, N * dim), dtype=np_scalar)
                _compute_dense_hessian_jac(edges, grad2_us, grad3_us, unit_vecs, dim, hessian_jac)
                filtered_vecs = [
                    _filter_modes_new_dense(v, hessian_jac)
                    for v in eig_vecs.T
                ]
                reshaped_vecs = np.array(filtered_vecs).T

                # del grad3_us, grad3_ts

                print("filter time:", time.time() - start)

                return eig_vals, reshaped_vecs, hessian, hessian_jac
            else:
                cond = edges[:, 0] < edges[:, 1]
                reduced_edges = edges[cond]
                grad2_us = grad2_us[cond]
                grad3_us = grad3_us[cond]
                unit_vecs = unit_vecs[cond]
                hessian_jac = _compute_sparse_hessian_jac(reduced_edges, grad2_us, grad3_us, unit_vecs, dim)
                filtered_vecs = [
                    filter_sparse_new(reduced_edges, np.ascontiguousarray(v), hessian_jac, dim, N)
                    for v in eig_vecs.T
                ]

                reshaped_vecs = np.array(filtered_vecs).T

                # del grad3_us, grad3_ts
                print("filter time:", time.time() - start)


                return eig_vals, reshaped_vecs, hessian, hessian_jac, reduced_edges

        else:
            return eig_vals, eig_vecs, hessian
        

    def compute_hessian_with_nlist(self,
                pos,
                dists,
                edges,
                types,
                box,
                dim,
                dense=False,
                alt=False,
                hessianjac=False):
        """WARNING: Only use dense=True on small systems. """
        N = len(pos)

        pos = pos.astype(np_scalar)
        dists = dists.astype(np_scalar)

        if alt:
            unit_vecs = (pos[edges[:, 1]] - pos[edges[:, 0]])[:, :dim]
        else:
            unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)

        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)


        if dense:
            hessian_dense = np.zeros((N * dim, N * dim), dtype=np_scalar)
            _compute_dense_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, hessian_dense)
            if hessianjac:
                grad3_us = self._compute_3gs(edges, unit_vecs, dists, types, dim)
                hessian_jac = np.zeros((N * dim, N * dim, N * dim), dtype=np_scalar)
                _compute_dense_hessian_jac(edges, grad2_us, grad3_us, unit_vecs, dim, hessian_jac)
                return hessian_dense, hessian_jac
            else:
                return hessian_dense
        else:
            hessian_csr = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            return hessian_csr

        
    def compute_hessian(self,
                system: gsd.hoomd.Frame,
                dense=False):
        """WARNING: Only use dense=True on small systems. """
        dim = system.configuration.dimensions
        N = system.particles.N
        box = BoxDim(system.configuration.box)
        pos = system.particles.position
        types = system.particles.typeid

        edges, dists = self._compute_nlist(system)

        unit_vecs = self._compute_uvecs(pos, edges, dists, box, dim)
        grad1_us = self._compute_1gs(edges, dists, types)
        grad2_us = self._compute_2gs(edges, dists, types)

        if dense:
            hessian = np.zeros((N * dim, N * dim))
            _compute_dense_hessian(edges, grad2_us, unit_vecs, dim, hessian)
        else:
            hessian = _compute_csr_hessian(edges, grad1_us, grad2_us, unit_vecs, dim, N)
            
        return hessian
    
    @staticmethod
    def filter_vec(vec, hessian_jac):
        return _filter_modes_new_dense(vec, hessian_jac)
    
    @staticmethod
    def filter_vec_sparse(edges, vec, hessian_jac, dim, N):
        return filter_sparse_new(edges, np.ascontiguousarray(vec), hessian_jac, dim, N)

# @njit
def compute_output(evals, evecs, modes=None, toss=None):
    output = np.zeros(evecs.shape[1])

    if toss is None:
        toss = 0
    if modes is None:
        modes = len(evals) + toss
    modes = min(modes, len(evals))
    for i in range(toss, modes):
        output += (1/evals[i]) * np.sum(np.square(evecs[i]), axis=-1)
    return output