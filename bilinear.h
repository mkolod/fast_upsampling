#pragma once

#include <ATen/ATen.h>

at::Tensor bilinear_cuda_forward(at::Tensor& in, const int new_h, const int new_w);

at::Tensor bilinear_cuda_backward(at::Tensor& in, const int orig_h, const int orig_w);
