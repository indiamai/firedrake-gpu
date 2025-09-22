from triton.language.extra import libdevice
import torch
import triton
import triton.language as tl
#import inspect
#import os

#from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    x = libdevice.pow(x, 2)
    tl.store(y_ptr+offsets, x, mask =mask)

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE)
output_torch = torch.pow(x,2)
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
