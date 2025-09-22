import cupy as cp
import cupyx as cpx
import torch
import numpy as np
import triton
import triton.language as tl
from triton.language.extra import libdevice
import pdb
DEVICE=triton.runtime.driver.active.get_active_torch_device()

# Ops for slicing (take/put) local tensor (extension to https://github.com/triton-lang/triton/pull/2715)
# extension found https://github.com/triton-lang/triton/issues/656
@triton.jit
def _indicator_(n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim: tl.constexpr):
    tl.static_assert(idx < n_dims)
    tl.static_assert(pos < pos_dim)
    y = tl.arange(0, pos_dim)
    y = tl.where(y==pos, 1, 0)
    
    for n in tl.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = tl.expand_dims(y, n)
    return y

@triton.jit
def _take_slice_(x, n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim:tl.constexpr, keep_dim: tl.constexpr = True):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = tl.sum(x * ind, n_dims - 1 - idx)
    if keep_dim:
        y = tl.expand_dims(y, n_dims - 1 - idx)

    return y

@triton.jit
def _put_slice_(x, n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim:tl.constexpr, input_slice):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = tl.where(ind==1, input_slice, x)
    return y

@triton.jit
def subkernel1(coords, w_0, t0, t1, inter4, inter6, size_C:tl.constexpr, size_Q:tl.constexpr, BLOCK_SIZE_C:tl.constexpr, BLOCK_SIZE_Q:tl.constexpr):
	pid_C = tl.program_id(axis=0)
	pid_Q = tl.program_id(axis=1)
	coords_dim1:tl.constexpr = 8
	coords_dim0:tl.constexpr = BLOCK_SIZE_C
	w_0_dim1:tl.constexpr = 8
	w_0_dim0:tl.constexpr = BLOCK_SIZE_C
	t0_dim0:tl.constexpr = BLOCK_SIZE_Q
	t1_dim1:tl.constexpr = 8
	t1_dim0:tl.constexpr = BLOCK_SIZE_Q
	inter4_dim1:tl.constexpr = 8
	inter4_dim0:tl.constexpr = BLOCK_SIZE_Q
	inter6_dim1:tl.constexpr = BLOCK_SIZE_Q
	inter6_dim0:tl.constexpr = BLOCK_SIZE_C
	coords_size1:tl.constexpr = 6
	coords_size0:tl.constexpr = BLOCK_SIZE_C * (pid_C+1) if BLOCK_SIZE_C * (pid_C+1) < size_C else size_C
	w_0_size1:tl.constexpr = 6
	w_0_size0:tl.constexpr = BLOCK_SIZE_C * (pid_C+1) if BLOCK_SIZE_C * (pid_C+1) < size_C else size_C
	t0_size0:tl.constexpr = BLOCK_SIZE_Q * (pid_Q+1) if BLOCK_SIZE_Q * (pid_Q+1) < size_Q else size_Q
	t1_size1:tl.constexpr = 6
	t1_size0:tl.constexpr = BLOCK_SIZE_Q * (pid_Q+1) if BLOCK_SIZE_Q * (pid_Q+1) < size_Q else size_Q
	inter4_size1:tl.constexpr = 6
	inter4_size0:tl.constexpr = BLOCK_SIZE_Q * (pid_Q+1) if BLOCK_SIZE_Q * (pid_Q+1) < size_Q else size_Q
	inter6_size1:tl.constexpr = BLOCK_SIZE_Q * (pid_Q+1) if BLOCK_SIZE_Q * (pid_Q+1) < size_Q else size_Q
	inter6_size0:tl.constexpr = BLOCK_SIZE_C * (pid_C+1) if BLOCK_SIZE_C * (pid_C+1) < size_C else size_C
	coords_stride1:tl.constexpr = 1
	coords_stride0:tl.constexpr = 6
	w_0_stride1:tl.constexpr = 1
	w_0_stride0:tl.constexpr = 6
	t0_stride0:tl.constexpr = 1
	t1_stride1:tl.constexpr = 1
	t1_stride0:tl.constexpr = 6
	inter4_stride1:tl.constexpr = 1
	inter4_stride0:tl.constexpr = 6
	inter6_stride1:tl.constexpr = 1
	inter6_stride0:tl.constexpr = size_Q
	inter6_offsets1 = (pid_Q * BLOCK_SIZE_Q + tl.arange(0, inter6_dim1)) 
	inter6_offsets0 = (pid_C * BLOCK_SIZE_C + tl.arange(0, inter6_dim0)) 
	inter6_offsets = inter6_offsets0[:,None]*inter6_stride0 + inter6_offsets1[None,:]*inter6_stride1
	inter6_mask = inter6_offsets <  (inter6_offsets0*inter6_stride0)[:,None] + inter6_size1
	inter6_ptr = inter6
	inter4_offsets1 = tl.arange(0, inter4_dim1)
	inter4_offsets0 = (pid_Q * BLOCK_SIZE_Q + tl.arange(0, inter4_dim0)) 
	inter4_offsets = inter4_offsets0[:,None]*inter4_stride0 + inter4_offsets1[None,:]*inter4_stride1
	inter4_mask = inter4_offsets <  (inter4_offsets0*inter4_stride0)[:,None] + inter4_size1
	inter4_ptr = inter4
	coords_offsets1 = tl.arange(0, coords_dim1)
	coords_offsets0 = (pid_C * BLOCK_SIZE_C + tl.arange(0, coords_dim0)) 
	coords_offsets = coords_offsets0[:,None]*coords_stride0 + coords_offsets1[None,:]*coords_stride1
	coords_mask = coords_offsets <  (coords_offsets0*coords_stride0)[:,None] + coords_size1
	coords_ptr = coords
	coords = tl.load(coords + coords_offsets, mask=coords_mask, other=0)
	w_0_offsets1 = tl.arange(0, w_0_dim1)
	w_0_offsets0 = (pid_C * BLOCK_SIZE_C + tl.arange(0, w_0_dim0)) 
	w_0_offsets = w_0_offsets0[:,None]*w_0_stride0 + w_0_offsets1[None,:]*w_0_stride1
	w_0_mask = w_0_offsets <  (w_0_offsets0*w_0_stride0)[:,None] + w_0_size1
	w_0_ptr = w_0
	w_0 = tl.load(w_0 + w_0_offsets, mask=w_0_mask, other=0)
	t0_offsets0 = (pid_Q * BLOCK_SIZE_Q + tl.arange(0, t0_dim0)) 
	t0_offsets = t0_offsets0[:]*t0_stride0
	t0_mask = t0_offsets < t0_size0
	t0_ptr = t0
	t0 = tl.load(t0 + t0_offsets, mask=t0_mask, other=0)
	t1_offsets1 = tl.arange(0, t1_dim1)
	t1_offsets0 = (pid_Q * BLOCK_SIZE_Q + tl.arange(0, t1_dim0)) 
	t1_offsets = t1_offsets0[:,None]*t1_stride0 + t1_offsets1[None,:]*t1_stride1
	t1_mask = t1_offsets <  (t1_offsets0*t1_stride0)[:,None] + t1_size1
	t1_ptr = t1
	t1 = tl.load(t1 + t1_offsets, mask=t1_mask, other=0)
	coords0 = _take_slice_(coords, len(coords.shape), 0, 0, coords.shape[1]).reshape(coords.shape[0])
	inter0 = (-1.0 * coords0)
	coords1 = _take_slice_(coords, len(coords.shape), 0, 1, coords.shape[1]).reshape(coords.shape[0])
	inter1 = (-1.0 * coords1)
	coords2 = _take_slice_(coords, len(coords.shape), 0, 2, coords.shape[1]).reshape(coords.shape[0])
	coords5 = _take_slice_(coords, len(coords.shape), 0, 5, coords.shape[1]).reshape(coords.shape[0])
	coords4 = _take_slice_(coords, len(coords.shape), 0, 4, coords.shape[1]).reshape(coords.shape[0])
	coords3 = _take_slice_(coords, len(coords.shape), 0, 3, coords.shape[1]).reshape(coords.shape[0])
	inter2 = tl.abs((((inter0 + coords2) * (inter1 + coords5)) + (-1.0 * ((inter0 + coords4) * (inter1 + coords3)))))
	inter3 = t0
	inter4 = t1
	inter5 = tl.sum((inter4 * w_0), 1)
	inter6 = ((inter3 * inter2[:, None]) * inter5)
	tl.store(inter4_ptr + inter4_offsets, inter4, mask = inter4_mask)
	tl.store(inter6_ptr + inter6_offsets, inter6, mask = inter6_mask)

@triton.jit
def subkernel2(A, inter4, inter6, size_C:tl.constexpr, size_Q:tl.constexpr, dim_Q:tl.constexpr, BLOCK_SIZE_C:tl.constexpr):
	pid_C = tl.program_id(axis=0)
	pid_Q = 0
	A_dim1:tl.constexpr = 8
	A_dim0:tl.constexpr = BLOCK_SIZE_C
	inter4_dim1:tl.constexpr = 8
	inter4_dim0:tl.constexpr = dim_Q
	inter6_dim1:tl.constexpr = dim_Q
	inter6_dim0:tl.constexpr = BLOCK_SIZE_C
	A_size1:tl.constexpr = 6
	A_size0:tl.constexpr = BLOCK_SIZE_C * (pid_C+1) if BLOCK_SIZE_C * (pid_C+1) < size_C else size_C
	inter4_size1:tl.constexpr = 6
	inter4_size0:tl.constexpr = size_Q * (pid_Q+1) if size_Q * (pid_Q+1) < size_Q else size_Q
	inter6_size1:tl.constexpr = size_Q * (pid_Q+1) if size_Q * (pid_Q+1) < size_Q else size_Q
	inter6_size0:tl.constexpr = BLOCK_SIZE_C * (pid_C+1) if BLOCK_SIZE_C * (pid_C+1) < size_C else size_C
	A_stride1:tl.constexpr = 1
	A_stride0:tl.constexpr = 6
	inter4_stride1:tl.constexpr = 1
	inter4_stride0:tl.constexpr = 6
	inter6_stride1:tl.constexpr = 1
	inter6_stride0:tl.constexpr = size_Q
	inter6_offsets1 = (pid_Q * size_Q + tl.arange(0, inter6_dim1)) 
	inter6_offsets0 = (pid_C * BLOCK_SIZE_C + tl.arange(0, inter6_dim0)) 
	inter6_offsets = inter6_offsets0[:,None]*inter6_stride0 + inter6_offsets1[None,:]*inter6_stride1
	inter6_mask = inter6_offsets <  (inter6_offsets0*inter6_stride0)[:,None] + inter6_size1
	inter6_ptr = inter6
	inter6 = tl.load(inter6 + inter6_offsets, mask=inter6_mask, other=0)
	inter4_offsets1 = tl.arange(0, inter4_dim1)
	inter4_offsets0 = (pid_Q * size_Q + tl.arange(0, inter4_dim0)) 
	inter4_offsets = inter4_offsets0[:,None]*inter4_stride0 + inter4_offsets1[None,:]*inter4_stride1
	inter4_mask = inter4_offsets <  (inter4_offsets0*inter4_stride0)[:,None] + inter4_size1
	inter4_ptr = inter4
	inter4 = tl.load(inter4 + inter4_offsets, mask=inter4_mask, other=0)
	A_offsets1 = tl.arange(0, A_dim1)
	A_offsets0 = (pid_C * BLOCK_SIZE_C + tl.arange(0, A_dim0)) 
	A_offsets = A_offsets0[:,None]*A_stride0 + A_offsets1[None,:]*A_stride1
	A_mask = A_offsets <  (A_offsets0*A_stride0)[:,None] + A_size1
	A_ptr = A
	A = tl.load(A + A_offsets, mask=A_mask, other=0)
	A_res=tl.sum((inter4 * inter6[:, None]), 1)
	tl.store(A_ptr + A_offsets, A_res, mask = A_mask)

def triton_kernel0(A, coords, w_0, t0, t1, size_C, size_Q, BLOCK_SIZE_C, BLOCK_SIZE_Q):
	inter4 = torch.from_numpy(np.zeros((size_Q,6))).float().to(DEVICE)
	inter6 = torch.from_numpy(np.zeros((size_C,size_Q))).float().to(DEVICE)
	grid = lambda meta: (triton.cdiv(size_C, meta['BLOCK_SIZE_C']),triton.cdiv(size_Q, meta['BLOCK_SIZE_Q']), )
	subkernel1[grid](coords,w_0,t0,t1,inter4,inter6,size_C,size_Q,BLOCK_SIZE_C,BLOCK_SIZE_Q)
	torch.cuda.current_stream().synchronize()
	print(inter6)
	print(inter4)
	dim_Q = int(np.power(2, np.ceil(np.log2(size_Q))))
	grid = lambda meta: (triton.cdiv(size_C, meta['BLOCK_SIZE_C']), )
	subkernel2[grid](A,inter4,inter6,size_C,size_Q,dim_Q,BLOCK_SIZE_C)
	print(A)
	breakpoint()

def gpu_parloop():
	a0 = cp.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
	m0 = cp.array([[0, 1, 2],
       [2, 3, 1]])
	a1 = cp.array([[0., 1.],
       [1., 1.],
       [1., 0.],
       [0., 0.]])
	m1 = cp.array([[0, 1, 2],
       [2, 3, 1]])
	a2 = cp.array([3., 3., 3., 3., 3., 3., 3., 3., 3.])
	m2 = cp.array([[5, 4, 3, 0, 2, 1],
       [5, 3, 8, 6, 7, 2]], dtype=cp.int32)
	t0 = torch.from_numpy(np.array([0.11169079, 0.11169079, 0.11169079, 0.05497587, 0.05497587,
       0.05497587])).float().to(DEVICE)
	t1 = torch.from_numpy(np.array([[-0.04820838,  0.79548023,  0.19283351, -0.04820838,  0.19283351,
        -0.08473049],
       [-0.08473049,  0.19283351,  0.19283351, -0.04820838,  0.79548023,
        -0.04820838],
       [-0.04820838,  0.19283351,  0.79548023, -0.08473049,  0.19283351,
        -0.04820838],
       [ 0.51763234,  0.29921523,  0.29921523, -0.07480381,  0.03354481,
        -0.07480381],
       [-0.07480381,  0.29921523,  0.03354481,  0.51763234,  0.29921523,
        -0.07480381],
       [-0.07480381,  0.03354481,  0.29921523, -0.07480381,  0.29921523,
         0.51763234]])).float().to(DEVICE)
	grid = lambda meta: (triton.cdiv(2, meta['BLOCK_SIZE_C']), )
	a_g0 = cp.zeros_like(m0, dtype=cp.float64)
	a_g0 = torch.from_numpy(a_g0.get()).float().to(DEVICE)
	a_g1 = cp.take(a1, m1, axis=0)
	a_g1 = torch.from_numpy(a_g1.get()).float().to(DEVICE)
	a_g2 = cp.take(a2, m2, axis=0)
	a_g2 = torch.from_numpy(a_g2.get()).float().to(DEVICE)
	triton_kernel0(a_g0,a_g1,a_g2,t0,t1,size_C=2,size_Q=6,BLOCK_SIZE_C=1,BLOCK_SIZE_Q=4)
	cpx.scatter_add(a0, m0, a_g0)
	print(a0)
