import cupyx as cpx
import cupy as cp
import numpy as np

# Code dependent on firedrake
#from firedrake import *
#from FIAT import make_quadrature
#from finat.element_factory import as_fiat_cell
#
#mesh = UnitSquareMesh(50,50)
#cg1_space = FunctionSpace(mesh, "CG", 1)
#cg2_space = FunctionSpace(mesh, "CG", 2)
#cg3_space = FunctionSpace(mesh, "CG", 3)
#
#with device("gpu") as compute_device:
#    # For checking our work
#    x = SpatialCoordinate(mesh)
#    f = Function(cg1_space)
#    g = Function(cg2_space)
#    f.assign(1)
#    g.interpolate(x[0] + 10*x[1])
#    v = TestFunction(cg3_space)
#    form = (f+g)*v*dx
#    form_a = assemble(form)
#    import os
#    if compute_device.kernel_string == "":
#        raise ValueError("Missing kernel, please run firedrake-clean")
#    with open("./temp_kernel_coefficients.py",'w') as file:
#        file.write("import cupy as cp\n")
#        for i, kernel in enumerate(compute_device.kernel_string):
#            file.write(kernel.replace("cupy_kernel", f"cupy_kernel{i}") + "\n")
#
#v = Function(cg3_space)
#
#coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
#coordinates = mesh.coordinates.dat.data_ro
#cg1_node_map = cg1_space.cell_node_list
#cg2_node_map = cg2_space.cell_node_list
#cg3_node_map = cg3_space.cell_node_list
#coord_node_map = coordinate_space.cell_node_list
#
#np.savez("coefficient_data.npz", coords=coordinates, coords_map=coord_node_map,
#                               empty_data=v.dat.data_ro, cg1_map=cg1_node_map,
#                               cg2_map=cg2_node_map, cg3_map=cg3_node_map, f_data = f.dat.data_ro,
#                               g_data = g.dat.data_ro, expected=form_a.dat.data_ro)

# Code not dependent on firedrake (with data file)
data = np.load("coefficient_data.npz")
from temp_kernel_coefficients import cupy_kernel1 as cupy_kernel 
cg1_node_map_gpu = cp.asarray(data["cg1_map"])
cg2_node_map_gpu = cp.asarray(data["cg2_map"])
cg3_node_map_gpu = cp.asarray(data["cg3_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)
f_data_gpu = cp.asarray(data["f_data"])
g_data_gpu = cp.asarray(data["g_data"])


#Gather
cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)
f_coeffs = cp.take(f_data_gpu, cg1_node_map_gpu, axis=0)
g_coeffs = cp.take(g_data_gpu, cg2_node_map_gpu, axis=0)

for i in range(len(cell_coords)):
    A = cp.zeros_like(cg3_node_map_gpu[i], dtype=cp.float64)
    cupy_kernel(cell_coords[i].flatten(), f_coeffs[i], g_coeffs[i], A)
    cpx.scatter_add(cg_data_gpu, cg3_node_map_gpu[i], A)

output = cg_data_gpu.get()

print("GPU:", output)
print("Firedrake:", data["expected"])
assert(np.allclose(data["expected"], output))
print("Success")
