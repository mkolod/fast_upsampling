#pragma once

#include <ATen/ATen.h>

at::Tensor bilinear_cuda_forward(at::Tensor z);
