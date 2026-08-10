import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "gfx1030"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1030"
os.environ["CUDA_HOME"] = "/opt/rocm-7.2.0"

KERNEL_DIR = "/home/chenco_adm/vllm_humanwork/csrc/rocm"
KERNEL_FILES = [os.path.join(KERNEL_DIR, "q_gemm_w8a16_fp8_rdna2.cu")]
STANDALONE_CPP = "/tmp/standalone_w8a16.cpp"

CPP = """#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

namespace vllm {
namespace w8a16_fp8_dense_rdna2 {
void gemm_w8a16_fp8_dense(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scales,
    torch::Tensor c,
    int64_t group_size);
}
}

void gemm_w8a16_fp8_dense_standalone(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scales,
    torch::Tensor c,
    int64_t group_size) {
  vllm::w8a16_fp8_dense_rdna2::gemm_w8a16_fp8_dense(
      a, b_q, b_scales, c, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_w8a16_fp8_dense", &gemm_w8a16_fp8_dense_standalone,
        "W8A16 FP8 dense GEMM for RDNA2 (gfx1030)");
}
"""

with open(STANDALONE_CPP, "w") as f:
    f.write(CPP)

import torch
from torch.utils.cpp_extension import load

ext = load(
    name="gemm_w8a16_fp8_dense_rdna2_standalone",
    sources=[STANDALONE_CPP] + KERNEL_FILES,
    extra_cflags=["-O3", "-std=c++17", "-fPIC", "-DSTANDALONE_W8A16_BUILD"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-fPIC", "-DSTANDALONE_W8A16_BUILD", "--offload-arch=gfx1030"],
    extra_include_paths=[KERNEL_DIR],
    verbose=False,
)
print("Build complete!")
print("Has gemm_w8a16_fp8_dense:", hasattr(ext, "gemm_w8a16_fp8_dense"))