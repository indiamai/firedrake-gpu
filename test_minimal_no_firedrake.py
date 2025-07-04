import cupyx as cpx
import cupy as cp
import numpy as np


# Code not dependent on firedrake (with data file)
data = np.load("firedrake_data.npz")
derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float32)
basis_funcs_gpu = cp.asarray(data["basis"])
cg_node_map_gpu = cp.asarray(data["cg_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float32)
weights_gpu = cp.asarray(data["weights"])



# Do all cells in one set of instructions
cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)
# i is number of cells, j coordinate basis, k spatial dim, l number of quad points
jacobians = cp.einsum("ijk,jlm->ilkm", cell_coords, derivs_gpu)
# Pointwise non linear operations go here
det_jacobians = cp.fabs(cp.linalg.det(jacobians))
contracted = cp.einsum("ij,kj,j->ik", det_jacobians, basis_funcs_gpu, weights_gpu)
cpx.scatter_add(cg_data_gpu, cg_node_map_gpu, contracted)


output = cg_data_gpu

print("GPU:", output)
print("Firedrake:", data["expected"])
assert(np.allclose(data["expected"], output))
