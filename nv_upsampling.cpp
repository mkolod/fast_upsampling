#include <torch/extension.h>

#include "bilinear.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// at::Tensor bilinear_cuda_forward(at::Tensor z);

template <typename T>
inline T lerp(T v0, T v1, T t) {
    return (1-t)*v0 + t*v1;
}

at::Tensor bilinear_forward(at::Tensor z) {
  CHECK_INPUT(z);
  return bilinear_cuda_forward(z);
}

at::Tensor bilinear_backward(at::Tensor z) {
  CHECK_INPUT(z);
  return z;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bilinear_forward, "bilinear forward");
  m.def("backward", &bilinear_backward, "bilinear backward");
}
