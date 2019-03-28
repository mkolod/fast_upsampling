#include <torch/torch.h>

template <typename T>
inline T lerp(T v0, T v1, T t) {
    return (1-t)*v0 + t*v1;
}

at::Tensor bilinear_forward(at::Tensor& z) {
  // Makes sure it's NCHW
  assert(z.ndim() == 4);
  int n = z.size(0);
  int c = z.size(1);
  int h = z.size(2);
  int w = z.size(3);

  at::Tensor to = at::zeros_like(z);
  float* out = to.data<float>();

  // TODO: support multiple types
  float* t = static_cast<float*>(z.data<float>());

//  #pragma omp parallel for
  out[0] = lerp(t[0], t[1], 0.5f);

/*
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < c; ++j) {
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          if (l < (w - 1)) {
	    int addr = n * (c*h*w) + c * (h * w) + h * w + w;
            out[0] = lerp(t[0], t[1], 0.5f); //lerp(t[addr], t[addr + 1], 0.5f);
	    // TODO: Support multiple types
	    
          }
        }
      }
    }
  }
*/

  return to;
}

at::Tensor bilinear_backward(at::Tensor& z) {
  return z;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilinear_forward", &bilinear_forward, "bilinear forward");
  m.def("bilinear_backward", &bilinear_backward, "bilinear backward");
}
