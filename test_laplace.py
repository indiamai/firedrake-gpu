import cupyx as cpx
import cupy as cp
import numpy as np

from firedrake import *
from FIAT import make_quadrature
from finat.element_factory import as_fiat_cell


mesh = UnitSquareMesh(10,10)
cg2_space = FunctionSpace(mesh, "CG", 2)
cg3_space = FunctionSpace(mesh, "CG", 3)

# Checking our work
v = TestFunction(cg2_space)
u = TrialFunction(cg3_space)
form = dot(grad(v),grad(u))*dx
form_a = assemble(form)

v = Function(cg2_space)
u = Function(cg3_space)

Q = make_quadrature(as_fiat_cell(mesh.ufl_cell()), 2)
coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
coordinates = mesh.coordinates.dat.data_ro
phi_xq_cg2 = cg2_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
phi_xq_cg3 = cg3_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
psi_xq = coordinate_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
derivs = np.dstack((psi_xq[(1,0)], psi_xq[(0,1)]))
derivs_cg2 = np.dstack((phi_xq_cg2[(1,0)], phi_xq_cg2[(0,1)]))
derivs_cg3 = np.dstack((phi_xq_cg3[(1,0)], phi_xq_cg3[(0,1)]))
weights = Q.get_weights()
cg2_node_map = cg2_space.cell_node_list
cg3_node_map = cg3_space.cell_node_list
coord_node_map = coordinate_space.cell_node_list

np.savez("firedrake_laplace_data.npz", coords=coordinates, coords_map=coord_node_map, 
                               grad_basis_cg2=derivs_cg2, grad_basis_cg3=derivs_cg3, grad_basis=derivs,
                               empty_data=np.outer(v.dat.data_ro,u.dat.data_ro), weights=weights,
                               cg2_map=cg2_node_map, cg3_map=cg3_node_map, expected=form_a.M.values)

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
