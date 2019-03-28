#include <torch/torch.h>

at::Tensor bilinear_forward_gpu_kernel_wrapper(at::Tensor& z);

at::Tensor bilinear_backward_gpu_kernel_wrapper(at::Tensor& z);

