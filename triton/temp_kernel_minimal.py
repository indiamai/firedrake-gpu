import cupy as cp
def cupy_kernel0(coords,A):
	t0=cp.array([[ 0.22222222,  0.44444444,  0.44444444, -0.11111111,  0.11111111,
        -0.11111111],
       [-0.11111111,  0.11111111,  0.44444444, -0.11111111,  0.44444444,
         0.22222222],
       [-0.11111111,  0.44444444,  0.11111111,  0.22222222,  0.44444444,
        -0.11111111]])
	t1=cp.array([0.16666667, 0.16666667, 0.16666667])
	t2=cp.array(-1.)
	is3=cp.einsum(",->", cp.add(cp.einsum(",->", t2,coords[0]), coords[4]),cp.add(cp.einsum(",->", t2,coords[1]), coords[3]))
	is4=t1[:]
	is5=abs(cp.add(cp.einsum(",->", cp.add(cp.einsum(",->", t2,coords[0]), coords[2]),cp.add(cp.einsum(",->", t2,coords[1]), coords[5])), cp.einsum(",->", t2,is3)))
	A[:]=cp.array(cp.einsum("BA->A", cp.einsum("BA,B->BA", t0[:,:],cp.einsum("A,->A", is4,is5))))

import numpy as np
data = np.load("tiny_data.npz")

coords = cp.array(data["coords"], dtype=cp.float32)
coord_node_map = cp.array(data["coords_map"])
cell_coords = cp.take(coords, coord_node_map, axis=0)
cell_coords = cell_coords.reshape((cell_coords.shape[0], -1))
for c in cell_coords:
    A = cp.zeros(6)
    cupy_kernel0(c, A)
    print(A)
