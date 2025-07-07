import cupyx as cpx
import cupy as cp
import numpy as np

# Code not dependent on firedrake (with data file)
data = np.load("test_coefficient_data.npz")
derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float64)
cg1_basis_gpu = cp.asarray(data["cg1_basis"])
cg2_basis_gpu = cp.asarray(data["cg2_basis"])
cg3_basis_gpu = cp.asarray(data["cg3_basis"])
cg1_node_map_gpu = cp.asarray(data["cg1_map"])
cg2_node_map_gpu = cp.asarray(data["cg2_map"])
cg3_node_map_gpu = cp.asarray(data["cg3_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)
weights_gpu = cp.asarray(data["weights"])
f_data_gpu = cp.asarray(data["f_data"])
g_data_gpu = cp.asarray(data["g_data"])




# Do all cells in one set of instructions

#Gather
cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)
f_coeffs = cp.take(f_data_gpu, cg1_node_map_gpu, axis=0)
g_coeffs = cp.take(g_data_gpu, cg2_node_map_gpu, axis=0)

# i is number of cells, j the basis, k spatial dim, l number of quad points
jacobians = cp.einsum("ijk,jlm->ilkm", cell_coords, derivs_gpu)
# Pointwise non linear operations go here
det_jacobians = cp.fabs(cp.linalg.det(jacobians))
f = cp.einsum("ij,jl->il", f_coeffs, cg1_basis_gpu)
g = cp.einsum("ij,jl->il", g_coeffs, cg2_basis_gpu)
f_add_g = cp.add(f,g)
contracted = cp.einsum("il,il,jl,l->ij", det_jacobians, f_add_g, cg3_basis_gpu, weights_gpu)

cpx.scatter_add(cg_data_gpu, cg3_node_map_gpu, contracted)


output = cg_data_gpu

print("GPU:", output)
print("Firedrake:", data["expected"])
assert(np.allclose(data["expected"], output))
print("Success")
