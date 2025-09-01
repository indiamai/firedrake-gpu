import cupy as cp
import cupyx as cpx
import numpy as np
import triton
import triton.language as tl
import torch
import triton.tools as tl_tools
import pdb
DEVICE = triton.runtime.driver.active.get_active_torch_device()

def next_power2(num):
    return int(np.power(2,np.ceil(np.log2(num))))

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
def _split_indicator_(n_dims: tl.constexpr, idx: tl.constexpr, stride: tl.constexpr, pos_dim: tl.constexpr, off: tl.constexpr, end_size:tl.constexpr):
    tl.static_assert(idx < n_dims)
    tl.static_assert(off < pos_dim)
    y = tl.arange(0, pos_dim)
    y = tl.where(y%stride==off, y, 0)
    count = tl.sum(tl.where(y!=0, 1, 0), 0)
    y = tl.sort(y)
    
    for n in tl.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = tl.expand_dims(y, n)
    return y

@triton.jit
def _take_split_slice_(x, n_dims: tl.constexpr, idx: tl.constexpr, stride: tl.constexpr, pos_dim:tl.constexpr, off:tl.constexpr, end_size:tl.constexpr, keep_dim: tl.constexpr = True):
    ind = _split_indicator_(n_dims, idx, stride, pos_dim, off, end_size)
    breakpoint()
    y = tl.gather(x, ind, n_dims - 1 - idx)
    # need to check that end_size is power of two - here or in generation?
    y = tl.trans(y.reshape(2, end_size, x.shape[1:]))
    y = tl.split(y)[1]
    if keep_dim:
        y = tl.expand_dims(y, n_dims - 1 - idx)

    return y

@triton.jit
def _put_slice_(x, n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim:tl.constexpr, input_slice):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = tl.where(ind==1, input_slice, x)
    return y

@triton.jit
def add_kernel(cell_coords,
               t0,
               t1,
               output_ptr,  # *Pointer* to output vector.
               c, 
               coords_c: tl.constexpr, 
               stride_coords_c, stride_coords_d, coords_size,
               t0_d0: tl.constexpr, t0_d1: tl.constexpr, t0_stride0, t0_stride1, t0_size0, t0_size1,
               t1_dim: tl.constexpr, t1_stride, t1_size,
               BLOCK_SIZE_Q: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_C: tl.constexpr,
               ):
    coords_d: tl.constexpr = 8
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    block_start = pid * BLOCK_SIZE_C
    cell_offsets = (block_start + tl.arange(0, BLOCK_SIZE_C)) % c
    sub_offsets = tl.arange(0, coords_d)
    offsets =  cell_offsets[:, None]*stride_coords_c + sub_offsets[None, :]*stride_coords_d 
    mask = offsets < (cell_offsets * stride_coords_c + coords_size)[:, None] 
    coord_cells = tl.load(cell_coords + offsets, mask=mask, other=0)
    c = _take_split_slice_(coord_cells, len(coord_cells.shape), 0, 2, coord_cells.shape[1],1, 4)
    breakpoint()
    coords0 = _take_slice_(coord_cells, len(coord_cells.shape), 0, 0, coord_cells.shape[1]).reshape(coord_cells.shape[0])
    coord_0 = _take_slice_(coord_cells, len(coord_cells.shape), 0, 0, coord_cells.shape[1]).reshape(coord_cells.shape[0])
    coord_1 = _take_slice_(coord_cells, BLOCK_SIZE_C, 0, 1, coords_d).reshape(BLOCK_SIZE_C)
    coord_2 = _take_slice_(coord_cells, BLOCK_SIZE_C, 0, 2, coords_d).reshape(BLOCK_SIZE_C)
    coord_3 = _take_slice_(coord_cells, BLOCK_SIZE_C, 0, 3, coords_d).reshape(BLOCK_SIZE_C)
    coord_4 = _take_slice_(coord_cells, BLOCK_SIZE_C, 0, 4, coords_d).reshape(BLOCK_SIZE_C)
    coord_5 = _take_slice_(coord_cells, BLOCK_SIZE_C, 0, 5, coords_d).reshape(BLOCK_SIZE_C)
    js = (-1 * coord_0 + coord_2)*(-1 * coord_1 + coord_5) - (-1 * coord_0 + coord_4)*(-1 * coord_1 + coord_3)
    t0_offsets0 = tl.arange(0, t0_d0)
    t0_offsets1 = tl.arange(0, t0_d1)

    t0_offsets =  t0_offsets0[:, None]*t0_stride0 + t0_offsets1[None, :]*t0_stride1
    t0 = tl.load(t0 + t0_offsets, mask= t0_offsets < (t0_offsets0*t0_stride0 + t0_size1)[:, None], other=0)
    t1_offsets = tl.arange(0, t1_dim)
    t1 = tl.load(t1 + t1_offsets, mask= t1_offsets < t1_size, other=0)
 
    res = tl.load(output_ptr + t1_offsets)
    res = tl.sum(t0*(tl.abs(js)[:, None]*t1)[:, :, None], 1)
    tl.store(output_ptr + offsets, res, mask=mask)

def add():

    data = np.load("tiny_data.npz")
    cg_data_gpu = cp.zeros_like(data["empty_data"])
    cg_node_map_gpu = cp.array(data["cg_map"])
    coord_node_map_cp = cp.array(data["coords_map"])
    coord_data_cp = cp.array(data["coords"], dtype=cp.float32)
    cell_coords = cp.take(coord_data_cp, coord_node_map_cp, axis=0)
    cell_coords = cell_coords.reshape((cell_coords.shape[0], -1))
    coords_gpu = torch.from_numpy(cell_coords.get()).float().to(DEVICE)

    num_cells = coord_node_map_cp.shape[0] 
    n_quads = 25 
    grid = lambda meta: (triton.cdiv(n_quads, meta['BLOCK_SIZE_Q']) * triton.cdiv(num_cells, meta['BLOCK_SIZE_C']), )

    output = torch.from_numpy(np.zeros((num_cells,6), dtype=np.float32)).to(DEVICE)
    t0=torch.from_numpy(np.array([[ 0.22222222,  0.44444444,  0.44444444, -0.11111111,  0.11111111,-0.11111111],
        [-0.11111111,0.11111111,0.44444444, -0.11111111,  0.44444444,0.22222222],
        [-0.11111111,  0.44444444,  0.11111111,  0.22222222,  0.44444444,-0.11111111]], dtype=np.float32)).to(DEVICE)
    t1= torch.from_numpy(np.array([0.16666667, 0.16666667, 0.16666667], dtype=np.float32)).to(DEVICE)

    #next_power2(cell_coords.shape[1]),

    add_kernel[grid](coords_gpu, t0, t1, output,
                     num_cells, next_power2(cell_coords.shape[0]),
                                          coords_gpu.stride(0), coords_gpu.stride(1), len(cell_coords[0]), 
                     next_power2(t0.shape[0]), next_power2(t0.shape[1]), t0.stride(0), t0.stride(1), t0.shape[0], t0.shape[1],
                     next_power2(len(t1)), t1.stride(0), len(t1), 
                     BLOCK_SIZE_Q=5, BLOCK_SIZE_C=8)
    torch.cuda.current_stream().synchronize()
    cpx.scatter_add(cg_data_gpu, cg_node_map_gpu, output)
    return cg_data_gpu

torch.manual_seed(0)
output_triton = add()
print("triton",output_triton)
      #f'{torch.max(torch.abs(output_torch - output_triton))}')


#print("GPU:", output)
#print("Firedrake:", data["expected"])
#assert(np.allclose(data["expected"], output))
