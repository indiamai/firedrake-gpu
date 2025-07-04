import cupyx as cpx
import cupy as cp
import numpy as np

from firedrake import *
from FIAT import make_quadrature
from finat.element_factory import as_fiat_cell

mesh = UnitSquareMesh(1,1)
cg_space = FunctionSpace(mesh, "CG", 1)
v = Function(cg_space)

Q = make_quadrature(as_fiat_cell(mesh.ufl_cell()), 2)
coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
coordinates = mesh.coordinates.dat.data_ro
phi_xq = cg_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
psi_xq = coordinate_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
derivs = np.dstack((psi_xq[(1,0)], psi_xq[(0,1)]))
derivs_cg = np.dstack((phi_xq[(1,0)], phi_xq[(0,1)]))
weights = Q.get_weights()
cg_node_map = cg_space.cell_node_list
coord_node_map = coordinate_space.cell_node_list

np.savez("firedrake_data.npz", coords=coordinates, coords_map=coord_node_map, basis=phi_xq[(0,0)], grad_basis_cg=derivs_cg, grad_basis=derivs, empty_data=np.outer(v.dat.data_ro,v.dat.data_ro), weights=weights, cg_map=cg_node_map)

data = np.load("firedrake_data.npz")

derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float32)
derivs_cg_gpu = cp.asarray(data["grad_basis_cg"], dtype=cp.float32)
basis_funcs_gpu = cp.asarray(data["basis"])
cg_node_map_gpu = cp.asarray(data["cg_map"])
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
grad_u = cp.einsum("lmij,kmi->lmkj", JT, derivs_cg)
basis_funcs_gpu = cp.einsum("cqkd, cqbd-> cqkb", grad_u, grad_u)
det_jacobians = cp.fabs(cp.linalg.det(jacobians))
contracted = cp.einsum("ij,ijkm,j->ikm", det_jacobians, basis_funcs_gpu, weights_gpu)
for c in range(len(cg_node_map_gpu)):
    map = cp.ix_(cg_node_map_gpu[c], cg_node_map_gpu[c])
    cpx.scatter_add(cg_data_gpu, map, contracted[c])


output = cg_data_gpu.get()


# Checking our work
v = TestFunction(cg_space)
u = TrialFunction(cg_space)
form = dot(grad(v),grad(u))*dx
form_a = assemble(form)

print("GPU:", output)
print("Firedrake:", form_a.M.values)

assert(np.allclose(form_a.M.values, output, atol=1e-6))
