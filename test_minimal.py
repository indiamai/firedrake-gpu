import cupyx as cpx
import cupy as cp
import numpy as np

# Code dependent on firedrake
from firedrake import *
from FIAT import make_quadrature
from finat.element_factory import as_fiat_cell

mesh = UnitSquareMesh(50,50)
cg_space = FunctionSpace(mesh, "CG", 2)

# For checking our work
v = TestFunction(cg_space)
form = v*dx
form_a = assemble(form)

v = Function(cg_space)
Q = make_quadrature(as_fiat_cell(mesh.ufl_cell()), 5)
coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
coordinates = mesh.coordinates.dat.data_ro
phi_xq = cg_space.finat_element.fiat_equivalent.tabulate(0, Q.get_points())[(0,0)]
psi_xq = coordinate_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
derivs = np.dstack((psi_xq[(1,0)], psi_xq[(0,1)]))
weights = Q.get_weights()
cg_node_map = cg_space.cell_node_list
coord_node_map = coordinate_space.cell_node_list

np.savez("firedrake_data.npz", coords=coordinates, coords_map=coord_node_map, basis=phi_xq, grad_basis=derivs, empty_data=v.dat.data_ro, weights=weights, cg_map=cg_node_map, expected=form_a.dat.data_ro)

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
