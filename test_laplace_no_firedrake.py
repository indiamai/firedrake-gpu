import cupyx as cpx
import cupy as cp
import numpy as np


data = np.load("firedrake_laplace_data.npz")

derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float32)
derivs_cg2_gpu = cp.asarray(data["grad_basis_cg2"], dtype=cp.float32)
derivs_cg3_gpu = cp.asarray(data["grad_basis_cg3"], dtype=cp.float32)
cg2_node_map_gpu = cp.asarray(data["cg2_map"])
cg3_node_map_gpu = cp.asarray(data["cg3_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float32)
weights_gpu = cp.asarray(data["weights"])


# Do all cells in one set of instructions
cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)
# i is number of cells, j coordinate basis, k spatial dim, l number of quad points
jacobians = cp.einsum("ijk,jlm->ilmk", cell_coords, derivs_gpu)
# Pointwise non linear operations go here
inv_J = cp.linalg.pinv(jacobians)
JT = cp.transpose(inv_J, axes=[0,1,3,2])
grad_v = cp.einsum("lmij,kmi->lmkj", JT, derivs_cg2_gpu)
grad_u = cp.einsum("lmij,kmi->lmkj", JT, derivs_cg3_gpu)
basis_funcs_gpu = cp.einsum("cqkd, cqbd-> cqkb", grad_v, grad_u)
det_jacobians = cp.fabs(cp.linalg.det(jacobians))
contracted = cp.einsum("ij,ijkm,j->ikm", det_jacobians, basis_funcs_gpu, weights_gpu)
for c in range(len(cg2_node_map_gpu)):
    map = cp.ix_(cg2_node_map_gpu[c], cg3_node_map_gpu[c])
    cpx.scatter_add(cg_data_gpu, map, contracted[c])


output = cg_data_gpu.get()


#print("GPU:", output)
#print("Firedrake:", data["expected"])
assert(np.allclose(data["expected"], output, atol=1e-6))
print("Success")
