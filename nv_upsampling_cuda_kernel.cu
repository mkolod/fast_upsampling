#include <ATen/ATen.h>

#include "bilinear.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>



/*
  TODO: Check out the MXNet implementation: https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu
*/

__device__ __forceinline__ void fastFp16AtomicAdd(__half* __restrict__ tensor,
                                  int index, int numel,
                                  __half value) {

  int addr = __alignof(tensor);
  bool tensor_aligned = addr % 4 == 0;

  // Actually it's not that index < numel - 1
  // but that it's not the last element if numel is even
  if (tensor_aligned) {
    __half2 value2;

    if (index % 2 == 0 && index < (numel - 1)) {
      value2.x = value;
      value2.y = __int2half_rz(0);
    }
    if (index % 2 == 1) {
      value2.x = __int2half_rz(0);
      value2.y = value;
    }

    atomicAdd(reinterpret_cast<__half2*>(tensor) + index/2, value2);

   } else if (index == numel - 1) {
     atomicAdd(tensor + index, value);
   }
}

template<typename scalar_t>
__device__ bool between(scalar_t value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

/*
def bilinear_upsample(input, new_h, new_w):
    orig_h, orig_w, channels = input.shape
    result = np.zeros((new_h, new_w, channels)).astype(np.float32)
    y_scale = orig_h / new_h
    x_scale = orig_w / new_w
    for h in range(new_h):
        for w in range(new_w):
            for c in range(channels):
                y_scaled_pos = h * y_scale
                x_scaled_pos = w * x_scale
                y1 = int(y_scaled_pos)
                x1 = int(x_scaled_pos)
                y2 = min(y1 + 1, orig_h - 1)
                x2 = min(x1 + 1, orig_w - 1)
                q11 = input[y1][x1][c]
                q12 = input[y1][x2][c]
                q21 = input[y2][x1][c]
                q22 = input[y2][x2][c]
                yfloat = y_scaled_pos - y1
                xfloat = x_scaled_pos - x1
                xmix1 = (1 - xfloat) * q11 + xfloat * q12
                xmix2 = (1 - xfloat) * q21 + xfloat * q22
                ymix = (1 - yfloat) * xmix1 + yfloat * xmix2
                result[h, w, c] = ymix
               
    return result
*/


// https://github.com/pytorch/pytorch/blob/master/aten/src/THCUNN/TemporalUpSamplingLinear.cu

/*

template<typename Acctype>
__device__ __forceinline__
static Acctype linear_upsampling_compute_source_index(
                          Acctype scale, int dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    Acctype src_idx = scale * (dst_index + Acctype(0.5)) - Acctype(0.5);
    return src_idx < Acctype(0) ? Acctype(0) : src_idx;
  }
}
*/

/*
__global__ void caffe_gpu_interp2_kernel(const int n,
    const Acctype rwidth, const bool align_corners,
    const THCDeviceTensor<Dtype, 3> data1, THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][w1];
          data2[n][c][w2] = val;
        }
      }
      return;
    }
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const Acctype val = w0lambda * data1[n][c][w1]
                            + w1lambda * data1[n][c][w1+w1p];
        data2[n][c][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }
  }
}
*/

template<typename scalar_t>
__global__ void bilinearForwardKernel(const int i_n, const int i_c, const int i_h,
                                      const int i_w,
                                      const scalar_t* const __restrict__ in,
                                      const int o_h,
                                      const int o_w, scalar_t* const __restrict__ out) {

  // grid-stride loop

  int globIdx = blockIdx.x * blockDim.x + threadIdx.x;
/*
  for (int globIdx = blockIdx.x * blockDim.x + threadIdx.x; 
       globIdx < n; 
       globIdx += blockDim.x * gridDim.x) { 
*/

{

    if (globIdx > i_n * i_c * o_h * o_w) {
      return;
    }

// https://github.com/pytorch/pytorch/blob/master/aten/src/THCUNN/TemporalUpSamplingLinear.cu

    const int inDim = i_n * i_c * i_h * i_w;

    const int w = globIdx % o_w;
    const int h = (globIdx / o_w) % o_h;
    const int c = (globIdx / o_w / o_h) % i_c;
    // TODO: check if this should be c or i_c
    const int n = globIdx / o_w / o_h / i_c;

    const int outIdx = n * i_c * o_h * o_w + c * o_h * o_w + h * o_w + w;

    const float hScaledPos = 1.0f * i_h / o_h * h;

    const float wScaledPos = 1.0f * i_w / o_w * w;

    const int h1 = static_cast<int>(floor(hScaledPos));
    const int w1 = static_cast<int>(floor(wScaledPos));
    const int h2 = min(h1 + 1, i_h - 1);
    const int w2 = min(w1 + 1, i_w - 1);

    const int in_idx11 = n * i_c * i_h * i_w + c * i_h * i_w + h1 * i_w + w1;
    const int in_idx12 = n * i_c * i_h * i_w + c * i_h * i_w + h1 * i_w + w2;
    const int in_idx21 = n * i_c * i_h * i_w + c * i_h * i_w + h2 * i_w + w1;
    const int in_idx22 = n * i_c * i_h * i_w + c * i_h * i_w + h2 * i_w + w2;

    // TODO: don't return, actually take other values like in MXNet

    if (in_idx11 > inDim || in_idx12 > inDim ||
        in_idx21 > inDim || in_idx22 > inDim) {

      return;
    }

    const scalar_t q11 = __ldg(&in[in_idx11]);
    const scalar_t q12 = __ldg(&in[in_idx12]);
    const scalar_t q21 = __ldg(&in[in_idx21]);
    const scalar_t q22 = __ldg(&in[in_idx22]);

    const scalar_t yfl = hScaledPos - h1;
    const scalar_t xfl = wScaledPos - w1;

    const scalar_t xMix1 = (1 - xfl) * q11 + xfl * q12;
    const scalar_t xMix2 = (1 - xfl) * q21 + xfl * q22;
    const scalar_t yMix = static_cast<scalar_t>((1 - yfl) * xMix1 + yfl * xMix2);

/*
    printf("n=%d, c=%d, h=%d, w=%d, h1=%d, w1=%d, h2=%d, w2=%d, in_idx11=%d, in_idx12=%d, in_idx21=%d, in_idx22=%d, q11=%f, q12=%f, q21=%f, q22=%f, xMix1=%f, xMix2=%f, yMix=%f, outIdx=%d\n",
           n, c, h, w, h1, w1, h2, w2, in_idx11, in_idx12, in_idx21, in_idx22,
           static_cast<float>(q11), static_cast<float>(q12), static_cast<float>(q21), static_cast<float>(q22),
           static_cast<float>(xMix1), static_cast<float>(xMix2), static_cast<float>(yMix), outIdx);
*/

    out[outIdx] = yMix;
}
//  }

}

at::Tensor bilinear_cuda_forward(at::Tensor& in, const int new_h, const int new_w) {

  // TODO: grid
  // TODO: block
  // TODO: make sure to specialize for half2 case

  // TODO: input dimensions

  // TODO: create new tensor here

  const int nIn = in.size(0);
  const int cIn = in.size(1);
  const int hIn = in.size(2);
  const int wIn = in.size(3);

  std::cout << "n = " << nIn << ", c = " << cIn << ", hIn = " << hIn << ", wIn = " << wIn << std::endl;

  at::Tensor out = at::empty({nIn, cIn, new_h, new_w}, in.options());

  int outSize = nIn * cIn * new_h * new_w;
  dim3 block(1024);
  dim3 grid((outSize + block.x - 1) / block.x);

//  dim3 grid(1, 1, 1);
//  dim3 block(1024, 1, 1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(in.type(), "bilinearForwardKernel", ([&]
    {
     
        bilinearForwardKernel<scalar_t><<<grid, block>>>(nIn, cIn, hIn,
                                      wIn, in.data<scalar_t>(),
                                      new_h, new_w, 
                                      out.data<scalar_t>()
                                      ); 

    }));

    std::cout << "Checking" << std::endl;
    AT_CHECK(cudaGetLastError() == cudaSuccess,
          "issue with bilinearForwardKernel, CUDA code ",
          cudaGetLastError());

    std::cout << "Checked" << std::endl;

  return out; 
}

/*
def bilinear_upsample(input, new_h, new_w):
    orig_h, orig_w, channels = input.shape
    result = np.zeros((new_h, new_w, channels)).astype(np.float32)
    y_scale = orig_h / new_h
    x_scale = orig_w / new_w
    for h in range(new_h):
        for w in range(new_w):
            for c in range(channels):
                y_scaled_pos = h * y_scale
                x_scaled_pos = w * x_scale
                y1 = int(y_scaled_pos)
                x1 = int(x_scaled_pos)
                y2 = min(y1 + 1, orig_h - 1)
                x2 = min(x1 + 1, orig_w - 1)
                q11 = input[y1][x1][c]
                q12 = input[y1][x2][c]
                q21 = input[y2][x1][c]
                q22 = input[y2][x2][c]
                yfloat = y_scaled_pos - y1
                xfloat = x_scaled_pos - x1
                xmix1 = (1 - xfloat) * q11 + xfloat * q12
                xmix2 = (1 - xfloat) * q21 + xfloat * q22
                ymix = (1 - yfloat) * xmix1 + yfloat * xmix2
                result[h, w, c] = ymix
               
    return result
In [7]:

*/

// kernel borrowed from MXNet: https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu

//template<typename scalar_t>
//__global__ void bilinearForwardKernel(const int i_c, const int i_h,
//                                      const int i_w, const scalar_t* data,
//                                      const scalar_t* __restrict__ grid, const int o_n,
//                                      const int o_c, const int o_h,
//                                      const int o_w, scalar_t* __restrict__ out) {
//
//  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
//       index < o_n * o_c * o_h * o_w;
//       index += blockDim.x * gridDim.x * gridDim.y) {
//
//    // (n, c, h, w) is the element in out
//    int w = index % o_w;
//    int h = (index / o_w) % o_h;
//    int c = (index / o_w / o_h) % o_c;
//    int n = index / o_w / o_h / o_c;
//    int out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
//    int grid_index = n * o_h * o_w * 2 + h * o_w + w;
//    scalar_t y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
//    scalar_t x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
//    int top_left_y = static_cast<int>(floor(y_real));
//    int top_left_x = static_cast<int>(floor(x_real));
//    scalar_t top_left_y_w = 1.0 - (y_real - top_left_y);
//    scalar_t top_left_x_w = 1.0 - (x_real - top_left_x);
//    int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
//    scalar_t top_left_v = 0;
//    scalar_t top_right_v = 0;
//    scalar_t bottom_left_v = 0;
//    scalar_t bottom_right_v = 0;
//    if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1))
//      top_left_v = *(data + data_index);
//    if (between(top_left_x + 1, 0, i_w-1) && between(top_left_y, 0, i_h-1))
//      top_right_v = *(data + data_index + 1);
//    if (between(top_left_x, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
//      bottom_left_v = *(data + data_index + i_w);
//    if (between(top_left_x+1, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
//      bottom_right_v = *(data + data_index + i_w + 1);
//    *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
//                        top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
//                        bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
//                        bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
//  }
//
//}

/*
template<typename scalar_t, int Req1, int Req2>
__global__ void BilinearSamplerBackwardKernel(const int i_c, const int i_h,
                                              const int i_w, const scalar_t* __restrict__ grad,
                                              const scalar_t* __restrict__ data, const int o_n,
                                              const int o_c, const int o_h,
                                              const int o_w, scalar_t* __restrict__ g_input,
                                              const scalar_t* __restrict__ grid_src,
                                              scalar_t* __restrict__ grad_grid) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_h * o_w;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) is the element in grad
    int w = index % o_w;
    int h = (index / o_w) % o_h;
    int n = index / o_w / o_h;
    scalar_t top_left_y_gw = 0.0;
    scalar_t top_left_x_gw = 0.0;
    int grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
    scalar_t y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
    scalar_t x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;

    int top_left_y = static_cast<int>(floor(y_real));
    int top_left_x = static_cast<int>(floor(x_real));
    scalar_t top_left_y_w = 1.0 - (y_real - top_left_y);
    scalar_t top_left_x_w = 1.0 - (x_real - top_left_x);
    for (int c = 0; c < o_c; ++c) {
      int grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
      int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
      // calc 4 vertex value in input data
      scalar_t top_left_v = 0;
      scalar_t top_right_v = 0;
      scalar_t bottom_left_v = 0;
      scalar_t bottom_right_v = 0;
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
