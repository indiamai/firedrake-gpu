import cupy as cp
import cupyx as cpx
import numpy as np
import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def next_power2(num):
    return int(np.power(2,np.ceil(np.log2(num))))


@triton.jit
def add_kernel(cell_coords,
               t0,
               t1,
               js_ptr,
               output_ptr,  # *Pointer* to output vector.
               c, 
               coords_c: tl.constexpr, coords_d: tl.constexpr,
               stride_coords_c, stride_coords_d, coords_size,
               t0_d0: tl.constexpr, t0_d1: tl.constexpr, t0_stride0, t0_stride1, t0_size,
               t1_dim: tl.constexpr, t1_stride, t1_size,
               BLOCK_SIZE_Q: tl.constexpr,  # Number of elements each program should process.
               BLOCK_SIZE_C: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    block_start = pid * BLOCK_SIZE_C
    cell_offsets = (block_start + tl.arange(0, BLOCK_SIZE_C)) % c
    sub_offsets = tl.arange(0, coords_d)
    offsets =  cell_offsets[:, None]*stride_coords_c + sub_offsets[None, :]*stride_coords_d 
    
    mask = offsets < coords_size 
    coord_cells = tl.load(cell_coords + offsets, mask=mask)
    
    t0_offsets0 = tl.arange(0, t0_d0)
    t0_offsets1 = tl.arange(0, t0_d1)

    t0_offsets =  t0_offsets0[:, None]*t0_stride0 + t0_offsets1[None, :]*t0_stride1
    t0 = tl.load(t0 + t0_offsets, mask= t0_offsets < t0_size, other=0)

    t1_offsets = tl.arange(0, t1_dim)
    t1 = tl.load(t1 + t1_offsets, mask= t1_offsets < t1_size, other=0)
    js = tl.load(js_ptr + cell_offsets, mask = cell_offsets < c, other = 0)
    is5 = tl.abs(js)
    res = tl.load(output_ptr + t1_offsets)
    res = tl.sum(t0*(t1*is5)[:, None], 0)
    tl.store(output_ptr + offsets, res[None, :], mask=mask)

def add():

    data = np.load("tiny_data.npz")
    derivs_gpu = torch.from_numpy(data["grad_basis"])
    basis_funcs_gpu = torch.from_numpy(data["basis"])
    cg_node_map_gpu = torch.from_numpy(data["cg_map"])
    cg_data_gpu = cp.empty_like(data["empty_data"])
    coord_node_map_cp = cp.array(data["coords_map"])
    coord_data_cp = cp.array(data["coords"], dtype=cp.float32)
    cell_coords = cp.take(coord_data_cp, coord_node_map_cp, axis=0)
    cell_coords = cell_coords.reshape((cell_coords.shape[0], -1))
    coords_gpu = torch.from_numpy(cell_coords.get()).float().to(DEVICE)
    js = (-1 * cell_coords[:, 0] + cell_coords[:,2])*(-1 * cell_coords[:, 1] + cell_coords[:,5]) - (-1 * cell_coords[:, 0] + cell_coords[:, 4])*(-1 * cell_coords[:,1] + cell_coords[:, 3])
    weights_gpu = torch.from_numpy(data["weights"])
    print(cell_coords)
    num_cells = coord_node_map_cp.shape[0] 
    n_quads = basis_funcs_gpu.shape[-1] 
    grid = lambda meta: (triton.cdiv(n_quads, meta['BLOCK_SIZE_Q']) * triton.cdiv(num_cells, meta['BLOCK_SIZE_C']), )
    coords_size = len(cell_coords.flatten())
    output = torch.from_numpy(np.zeros((num_cells,6), dtype=np.float32)).to(DEVICE)
    t0=torch.from_numpy(np.array([[ 0.22222222,  0.44444444,  0.44444444, -0.11111111,  0.11111111,-0.11111111],
        [-0.11111111,0.11111111,0.44444444, -0.11111111,  0.44444444,0.22222222],
        [-0.11111111,  0.44444444,  0.11111111,  0.22222222,  0.44444444,-0.11111111]], dtype=np.float32)).to(DEVICE)
    t1= torch.from_numpy(np.array([0.16666667, 0.16666667, 0.16666667], dtype=np.float32)).to(DEVICE)
    js = torch.from_numpy(js.get()).to(DEVICE)
    add_kernel[grid](coords_gpu, t0, t1, js, output,
                     num_cells, next_power2(cell_coords.shape[0]),
                     next_power2(cell_coords.shape[1]),
                     coords_gpu.stride(0), coords_gpu.stride(1), len(cell_coords.flatten()), 
                     next_power2(t0.shape[0]), next_power2(t0.shape[1]), t0.stride(0), t0.stride(1), len(t0.flatten()),
                     next_power2(len(t1)), t1.stride(0), len(t1), 
                     BLOCK_SIZE_Q=5, BLOCK_SIZE_C=1)
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
