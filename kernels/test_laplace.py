import cupyx as cpx
import cupy as cp
import numpy as np

#from firedrake import *
#from FIAT import make_quadrature
#from finat.element_factory import as_fiat_cell
#
#
#mesh = UnitSquareMesh(20, 20)
#cg2_space = FunctionSpace(mesh, "CG", 2)
#cg3_space = FunctionSpace(mesh, "CG", 3)
#
#with device("gpu") as compute_device:
#    # Checking our work
#    v = TestFunction(cg2_space)
#    u = TrialFunction(cg3_space)
#    f = Function(cg3_space)
#    f.assign(0.5)
#    form = dot(grad(v),grad(u))*dx
#    form_a = assemble(action(form,f))
#    expected_val = form_a.dat.data
#    import os
#    if compute_device.kernel_string == "":
#        raise ValueError("Missing kernel, please run firedrake-clean")
#    with open("./temp_kernel_laplace.py",'w') as file:
#        file.write("import cupy as cp\n")
#        for i, kernel in enumerate(compute_device.kernel_string):
#            file.write(kernel.replace("cupy_kernel", f"cupy_kernel{i}") + "\n")
#
#v = Function(cg2_space)
#u = Function(cg3_space)
#
#coordinate_space = FunctionSpace(mesh, mesh._ufl_coordinate_element)
#coordinates = mesh.coordinates.dat.data_ro
#cg2_node_map = cg2_space.cell_node_list
#cg3_node_map = cg3_space.cell_node_list
#coord_node_map = coordinate_space.cell_node_list
#
#np.savez("laplace_data.npz", coords=coordinates, coords_map=coord_node_map, 
#                               empty_data=v.dat.data_ro, cg2_map=cg2_node_map,
#                               cg3_map=cg3_node_map, f=f.dat.data, expected=expected_val)

data = np.load("laplace_data.npz")
from temp_kernel_laplace import cupy_kernel0 as cupy_kernel 

cg2_node_map_gpu = cp.asarray(data["cg2_map"])
cg3_node_map_gpu = cp.asarray(data["cg3_map"])
cg_data_gpu = cp.empty_like(data["empty_data"])
coord_node_map_gpu = cp.asarray(data["coords_map"])
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)
f_data_gpu = cp.asarray(data["f"], dtype=cp.float64)


cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis = 0)
f_coeffs = cp.take(f_data_gpu, cg3_node_map_gpu, axis=0)

for i in range(len(cell_coords)):
    A = cp.zeros_like(cg2_node_map_gpu[i], dtype=cp.float64)
    cupy_kernel(cell_coords[i].flatten(), f_coeffs[i], A)
    cpx.scatter_add(cg_data_gpu, cg2_node_map_gpu[i], A)


output = cg_data_gpu.get()


print("GPU:", output)
print("Firedrake:", data["expected"])
assert(np.allclose(data["expected"], output))
print("Success")
