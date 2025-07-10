import cupyx as cpx
import cupy as cp
import numpy as np

# This file demonstrates the limitation that 

# Code not dependent on firedrake (with data file)
data = np.load("firedrake_data.npz")
derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float64)
basis_funcs_gpu = cp.asarray(data["basis"])
cg_node_map_gpu = cp.asarray(data["cg_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)
weights_gpu = cp.asarray(data["weights"])


s1 = cp.cuda.Stream(non_blocking=True)
with s1:
    s1.begin_capture()
    # Do all cells in one set of instructions
    cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)
    # i is number of cells, j coordinate basis, k spatial dim, l number of quad points
    jacobians = cp.einsum("ijk,jlm->ilkm", cell_coords, derivs_gpu)

    g = s1.end_capture()

g.launch(stream=s1)
s1.synchronize()



