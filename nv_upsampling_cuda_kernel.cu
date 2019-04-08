#include <ATen/ATen.h>

#include "bilinear.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
  TODO: Check out the MXNet implementation: https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu
*/

// __forceinline__
__device__ void fastFp16AtomicAdd(__half* __restrict__ tensor,
                                  int index, int numel,
                                  __half value) {

  int addr = __alignof(tensor);
  bool tensor_aligned = addr % 4 == 0;

  #if defined(__CUDA_ARCH)
  #else
    #error CUDA_ARCH is not defined
  #endif

  #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
    #error Need CUDA arch >= 700
  #endif

  #if ((CUDA_VERSION >= 10000) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))

  // Actually it's not that index < numel - 1
  // but that it's not the last element if numel is even
  if (tensor_aligned) {
    __half2 value2;

    if (index % 2 == 0 && index < (numel - 1)) {
      value2.x = value;
      value2.y = __float2half(0.0f); //0;
    }
    if (index % 2 == 1) {
      value2.x = __float2half(0.0f); //0;
      value2.y = value;
    }

    atomicAdd(reinterpret_cast<__half2*>(tensor) + index/2, value2);

   } else if (index == numel - 1) {
     atomicAdd(tensor + index, value);
   }
   #else
       #pragma message __CUDA_ARCH__
       #error This extension requires CUDA 10 and arch >= 7.0, found CUDA CUDA_VERSION and arch __CUDA_ARCH__
   #endif
}

template<typename scalar_t>
__device__ bool between(scalar_t value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}


at::Tensor bilinear_cuda_forward(at::Tensor& in, at::Tensor out, int out_h, int out_w) {

  // TODO: grid
  // TODO: block
  // TODO: make sure to specialize for half2 case
  
  return in; 
}

// kernel borrowed from MXNet: https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu
template<typename scalar_t>
__global__ void bilinearForwardKernel(const int i_c, const int i_h,
                                      const int i_w, const scalar_t* data,
                                      const scalar_t* __restrict__ grid, const int o_n,
                                      const int o_c, const int o_h,
                                      const int o_w, scalar_t* __restrict__ out) {

  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_c * o_h * o_w;
       index += blockDim.x * gridDim.x * gridDim.y) {

    // (n, c, h, w) is the element in out
    int w = index % o_w;
    int h = (index / o_w) % o_h;
    int c = (index / o_w / o_h) % o_c;
    int n = index / o_w / o_h / o_c;
    int out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
    int grid_index = n * o_h * o_w * 2 + h * o_w + w;
    scalar_t y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    scalar_t x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    scalar_t top_left_y_w = 1.0 - (y_real - top_left_y);
    scalar_t top_left_x_w = 1.0 - (x_real - top_left_x);
    int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
    scalar_t top_left_v = 0;
    scalar_t top_right_v = 0;
    scalar_t bottom_left_v = 0;
    scalar_t bottom_right_v = 0;
    if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1))
      top_left_v = *(data + data_index);
    if (between(top_left_x + 1, 0, i_w-1) && between(top_left_y, 0, i_h-1))
      top_right_v = *(data + data_index + 1);
    if (between(top_left_x, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
      bottom_left_v = *(data + data_index + i_w);
    if (between(top_left_x+1, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
      bottom_right_v = *(data + data_index + i_w + 1);
    *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                        top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                        bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                        bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
  }

}


/*
template<typename DType, int Req1, int Req2>
__global__ void BilinearSamplerBackwardKernel(const int i_c, const int i_h,
                                              const int i_w, const DType* grad,
                                              const DType* data, const int o_n,
                                              const int o_c, const int o_h,
                                              const int o_w, DType* g_input,
                                              const DType* grid_src,
                                              DType* grad_grid) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_h * o_w;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) is the element in grad
    int w = index % o_w;
    int h = (index / o_w) % o_h;
    int n = index / o_w / o_h;
    DType top_left_y_gw = 0.0;
    DType top_left_x_gw = 0.0;
    int grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
    DType y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    DType x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;

    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    DType top_left_y_w = 1.0 - (y_real - top_left_y);
    DType top_left_x_w = 1.0 - (x_real - top_left_x);
    for (int c = 0; c < o_c; ++c) {
      int grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
      int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
      // calc 4 vertex value in input data
      DType top_left_v = 0;
      DType top_right_v = 0;
      DType bottom_left_v = 0;
      DType bottom_right_v = 0;
      // calc input grad
      if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
        if (Req1 != mxnet::kNullOp) {
          atomicAdd(&g_input[data_index], *(grad + grad_index) * top_left_y_w * top_left_x_w);
        }
        top_left_v = *(data + data_index);
      }
      if (between(top_left_x+1, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
        if (Req1 != mxnet::kNullOp) {
          atomicAdd(&g_input[data_index + 1],
                    *(grad + grad_index) * top_left_y_w * (1.0 - top_left_x_w));
        }
        top_right_v = *(data + data_index + 1);
      }
      if (between(top_left_x, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
        if (Req1 != mxnet::kNullOp) {
          atomicAdd(&g_input[data_index+ i_w],
                    *(grad + grad_index) * (1.0 - top_left_y_w) * top_left_x_w);
        }
        bottom_left_v = *(data + data_index + i_w);
      }
      if (between(top_left_x+1, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
        if (Req1 != mxnet::kNullOp) {
          atomicAdd(&g_input[data_index+ i_w + 1],
                    *(grad + grad_index) * (1.0 - top_left_y_w) * (1.0 - top_left_x_w));
        }
        bottom_right_v = *(data + data_index + i_w + 1);
      }
      // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
      top_left_y_gw -= *(grad + grad_index) * (top_right_v - bottom_right_v +
                        (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                        * top_left_x_w);
      top_left_x_gw -= *(grad + grad_index) * (bottom_left_v - bottom_right_v +
                        (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                        * top_left_y_w);
    }
    if (Req2 != mxnet::kNullOp) {
      // calc grad of grid
      *(grad_grid + grid_src_index + o_h * o_w) += top_left_y_gw * (i_h - 1) / 2;
      *(grad_grid + grid_src_index) += top_left_x_gw * (i_w - 1) / 2;
    }
  }
}
*/
