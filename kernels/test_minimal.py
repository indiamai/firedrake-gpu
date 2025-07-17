import cupyx as cpx
import cupy as cp
import numpy as np

# Code dependent on firedrake
#from firedrake import *
#from FIAT import make_quadrature
#from finat.element_factory import as_fiat_cell
#
#mesh = UnitSquareMesh(50,50)
#cg_space = FunctionSpace(mesh, "CG", 2)
#
#with device("gpu") as compute_device:
#    # For checking our work
#    v = TestFunction(cg_space)
#    form = v*dx
#    form_a = assemble(form)
#    import os
#    if compute_device.kernel_string == "":
#        raise ValueError("Missing kernel, please run firedrake-clean")
#    with open("./temp_kernel_minimal.py",'w') as file:
#        file.write("import cupy as cp\n")
#        for i, kernel in enumerate(compute_device.kernel_string):
#            file.write(kernel.replace("cupy_kernel", f"cupy_kernel{i}") + "\n")
#
#v = Function(cg_space)
#coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
#coordinates = mesh.coordinates.dat.data_ro
#cg_node_map = cg_space.cell_node_list
#coord_node_map = coordinate_space.cell_node_list
#
#np.savez("minimal_data.npz", coords=coordinates, coords_map=coord_node_map,
#                             empty_data=v.dat.data_ro, cg_map=cg_node_map, expected=form_a.dat.data_ro)

# Code not dependent on firedrake (with kernel and data file)
from temp_kernel_minimal import cupy_kernel0 as cupy_kernel 
data = np.load("minimal_data.npz")
cg_node_map_gpu = cp.asarray(data["cg_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)


cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)

for i in range(len(cell_coords)):
    A = cp.zeros_like(cg_node_map_gpu[i], dtype=cp.float64)
    cupy_kernel(cell_coords[i].flatten(), A)
    cpx.scatter_add(cg_data_gpu, cg_node_map_gpu[i], A)


output = cg_data_gpu

print("GPU:", output)
print("Firedrake:", data["expected"])
assert(np.allclose(data["expected"], output))
