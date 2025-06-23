from numba import njit
import numpy as np
import scipy.sparse as ssp

from .data import np_scalar

# NOTE ATM this function accepts a dense matrix as the hessian.
# It would be a whole lot more memory efficient if we used a sparse
# representation
@njit
def _compute_dense_hessian(edges, grad1_us, grad2_us, edge_vecs, dim, N, hessian):
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
                if i >= N or j >= N:
                    continue
                if i == j:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] += k_outer
                else:
                    hessian[i * dim:(i + 1) * dim,
                            j * dim:(j + 1) * dim] -= k_outer
                    

@njit
def _compute_dense_hessian_jac(edges, grad2_us, grad3_us, edge_vecs, dim, hessian_jac):

    k_outer3 = np.zeros((dim, dim, dim))

    grad2_d3 = np.zeros((dim, dim, dim))

    term = np.zeros((dim, dim, dim))

    for edge_idx in np.arange(len(edges)):

        if edges[edge_idx][0] < edges[edge_idx][1]:
            continue

        grad2 = 4.0 * grad2_us[edge_idx] * np.eye(dim)

        grad3 = 8.0 * grad3_us[edge_idx]

        k_vec = edge_vecs[edge_idx]
        k_outer = np.outer(k_vec, k_vec)
        for i in range(dim):
            k_outer3[i] = k_vec[i] * k_outer * grad3
            grad2_d3[i] = k_vec[i] * grad2

        # add additional terms to the hessian_jac

        term = k_outer3 + grad2_d3 + grad2_d3.transpose((1, 0, 2)) + grad2_d3.transpose((2, 1, 0))
        
        for idx in range(2):
            for jdx in range(2):
                for kdx in range(2):
                    i = edges[edge_idx][idx]
                    j = edges[edge_idx][jdx]
                    k = edges[edge_idx][kdx]
                    idx_mod_2 = (idx + jdx + kdx) % 2
                    sign = -1.0 if idx_mod_2 == 0 else 1.0

                    hessian_jac[i * dim:(i + 1) * dim,
                                j * dim:(j + 1) * dim,
                                k * dim:(k + 1) * dim] += sign * term


@njit 
def _compute_sparse_hessian_jac(edges, grad2_us, grad3_us, edge_vecs, dim):
    """edges must be 'half'"""

    output = np.zeros((len(edges), dim, dim, dim))

    k_outer3 = np.zeros((dim, dim, dim))

    grad2_d3 = np.zeros((dim, dim, dim))

    for edge_idx in np.arange(len(edges)):

        grad2 = 4.0 * grad2_us[edge_idx] * np.eye(dim)

        grad3 = 8.0 * grad3_us[edge_idx]

        k_vec = edge_vecs[edge_idx]
        k_outer = np.outer(k_vec, k_vec)
        for i in range(dim):
            k_outer3[i] = k_vec[i] * k_outer * grad3
            grad2_d3[i] = k_vec[i] * grad2

        output[edge_idx] = k_outer3 + grad2_d3 + grad2_d3.transpose((1, 0, 2)) + grad2_d3.transpose((2, 1, 0))

    return output


@njit
def _filter_sparse_stage_1(edges, vec, dim, sparse_hj, N, row, col, data):
    """edges must be 'half'"""

    # reshape vector into (N x dim)
    v = vec.reshape(-1, dim)
    ldx = 0

    mat_i = np.zeros((dim, dim))
    mat_j = np.zeros((dim, dim))

    for edge_idx in np.arange(len(edges)):

        edge = edges[edge_idx]
        part_i = edge[0]
        part_j = edge[1]

        hj = sparse_hj[edge_idx]

        mat_i[:] = 0.0
        mat_j[:] = 0.0
        for idx in range(dim):
            mat_i += v[part_i][idx] * hj[idx, :]
            mat_j += v[part_j][idx] * hj[idx, :]

        mats = (mat_i, mat_j)

        for idx in range(2):
            for jdx in range(2):
                for kdx in range(2):
                    i = edges[edge_idx][idx]
                    j = edges[edge_idx][jdx]
                    k = edges[edge_idx][kdx]
                    if i >= N or j >= N or k >= N:
                        continue
                    idx_mod_2 = (idx + jdx + kdx) % 2
                    sign = -1.0 if idx_mod_2 == 0 else 1.0

                    d = sign * mats[idx]
                    ii = j * dim
                    jj = k * dim
                    for l in range(dim):
                        for m in range(dim):
                            row[ldx] = ii + m
                            col[ldx] = jj + l
                            data[ldx] = d[m, l]
                            ldx += 1

def filter_sparse_new(edges, vec, sparse_hj, dim, N):
    """edges must be 'half'"""

    row = np.zeros(8*len(edges)*dim*dim, dtype=np.int32)
    col = np.zeros(8*len(edges)*dim*dim, dtype=np.int32)
    data = np.zeros(8*len(edges)*dim*dim, dtype=np_scalar)
    _filter_sparse_stage_1(edges, vec, dim, sparse_hj, N, row, col, data)

    stage_1_coo = ssp.coo_array((data, (row, col)), shape=(N*dim, N*dim))

    # convert to csr
    stage_1_csr = stage_1_coo.tocsr()

    # dot with vec
    return stage_1_csr.dot(vec)

@njit
def _make_coo_data(edges, grad1_us, grad2_us, edge_vecs, dim, N, row, col, data):

    idx = 0

    for edge_idx in np.arange(len(edges)):

        k_vec  = edge_vecs[edge_idx]
        k_outer = 2.0 * grad2_us[edge_idx] * np.outer(k_vec, k_vec) + grad1_us[edge_idx] * np.eye(dim)

        for i in edges[edge_idx]:
            for j in edges[edge_idx]:
                if i >= N or j >= N:
                    continue
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
    data = np.zeros(4*len(edges)*dim*dim, dtype=np_scalar)
    _make_coo_data(edges, grad1_us, grad2_us, edge_vecs, dim, N, row, col, data)
    

    hessian = ssp.coo_array((data, (row, col)), shape=(N*dim, N*dim))
    return hessian.tocsr()


def _filter_modes_new_dense(vec, hess_jac):
    return np.einsum('ijk,j,k->i', hess_jac, vec, vec, optimize=True)