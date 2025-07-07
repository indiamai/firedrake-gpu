import cupyx as cpx
import cupy as cp
import numpy as np

# Code dependent on firedrake
from firedrake import *
from FIAT import make_quadrature
from finat.element_factory import as_fiat_cell

mesh = UnitSquareMesh(50,50)
cg1_space = FunctionSpace(mesh, "CG", 1)
cg2_space = FunctionSpace(mesh, "CG", 2)
cg3_space = FunctionSpace(mesh, "CG", 3)

# For checking our work
x = SpatialCoordinate(mesh)
f = Function(cg1_space)
g = Function(cg2_space)
f.assign(1)
g.interpolate(x[0] + 10*x[1])
v = TestFunction(cg3_space)
form = (f+g)*v*dx
form_a = assemble(form)

v = Function(cg3_space)
Q = make_quadrature(as_fiat_cell(mesh.ufl_cell()), 5)
weights = Q.get_weights()
coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
coordinates = mesh.coordinates.dat.data_ro
cg1_basis_xq = cg1_space.finat_element.fiat_equivalent.tabulate(0, Q.get_points())[(0,0)]
cg2_basis_xq = cg2_space.finat_element.fiat_equivalent.tabulate(0, Q.get_points())[(0,0)]
cg3_basis_xq = cg3_space.finat_element.fiat_equivalent.tabulate(0, Q.get_points())[(0,0)]
coordinate_basis_xq = coordinate_space.finat_element.fiat_equivalent.tabulate(1, Q.get_points())
derivs = np.dstack((coordinate_basis_xq[(1,0)], coordinate_basis_xq[(0,1)]))
weights = Q.get_weights()
cg1_node_map = cg1_space.cell_node_list
cg2_node_map = cg2_space.cell_node_list
cg3_node_map = cg3_space.cell_node_list
coord_node_map = coordinate_space.cell_node_list

np.savez("test_coefficient_data.npz", coords=coordinates, coords_map=coord_node_map, cg1_basis=cg1_basis_xq,
                               cg2_basis=cg2_basis_xq, cg3_basis=cg3_basis_xq, grad_basis=derivs,
                               empty_data=v.dat.data_ro, weights=weights, cg1_map=cg1_node_map,
                               cg2_map=cg2_node_map, cg3_map=cg3_node_map, f_data = f.dat.data_ro,
                               g_data = g.dat.data_ro, expected=form_a.dat.data_ro)

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
