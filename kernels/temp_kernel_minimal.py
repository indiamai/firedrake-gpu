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
	A[:]=cp.array(cp.einsum("BA->B", cp.einsum("AB,A->BA", t0[:,:],cp.einsum("A,->A", is4,is5))))
