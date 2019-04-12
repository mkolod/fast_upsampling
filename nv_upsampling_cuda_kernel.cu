#include <type_traits>

#include <ATen/ATen.h>

//#include "caffe2/core/context_gpu.h"
//#include "caffe2/operators/upsample_op.h"
//#include "caffe2/utils/math.h"

#include "bilinear.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>



/*
  TODO: Check out the MXNet implementation: https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu

  TODO: use type traits (std::is_same<scalar_t, __half>::value) to see when to call this instead of vanilla atomicAdd

*/

__device__ __forceinline__ void fastFp16AtomicAdd(__half* __restrict__ tensor,
                                  int index, int numel,
                                  __half value) {

  int addr = __alignof(tensor);
  bool tensor_aligned = addr % 4 == 0;

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

__device__ __forceinline__ int idx(
    const int n,
    const int num_channels,
    const int c,
    const int height,
    const int width,
    const int y,
    const int x) {
  return ((n * num_channels + c) * height + y) * width + x;
}

// input is X, output is Y
template <typename scalar_t>
__global__ void bilinearForwardKernel(
    const int output_size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const scalar_t* const  __restrict__ X,
    scalar_t* const __restrict__ Y) {

  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x;
       index < output_size; index += blockDim.x * gridDim.x) {

    int indexTemp = index;
    const int out_x = indexTemp % output_width;
    indexTemp /= output_width;
    const int out_y = indexTemp % output_height;
    indexTemp /= output_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const float height_scale = 1.0f * output_height / input_height;
    const float width_scale = 1.0f * output_width / input_width;

    const int in_y = fminf(out_y / height_scale, input_height - 1);
    const int in_x = fminf(out_x / width_scale, input_width - 1);

    const float rheight =
        output_height > 1 ? (input_height - 1.f) / (output_height - 1.f) : 0.f;
    const float rwidth =
        output_width > 1 ? (input_width - 1.f) / (output_width - 1.f) : 0.f;

    // Compute Y axis lambdas
    const float h1r = rheight * out_y;
    const int h1 = static_cast<int>(h1r);
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.f - h1lambda;

    // Compute X axis lambdas
    const float w1r = rwidth * out_x;
    const int w1 = static_cast<int>(w1r);
    const int w1p = (w1 < input_width - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.f - w1lambda;

    Y[index] =
        static_cast<scalar_t>(h0lambda *
             (w0lambda *
                  X[idx(
                      n, num_channels, c, input_height, input_width, h1, w1)] +
              w1lambda *
                  X[idx(
                      n,
                      num_channels,
                      c,
                      input_height,
                      input_width,
                      h1,
                      w1 + w1p)]) +
         h1lambda *
             (w0lambda *
                  X[idx(
                      n,
                      num_channels,
                      c,
                      input_height,
                      input_width,
                      h1 + h1p,
                      w1)] +
              w1lambda *
                  X[idx(
                      n,
                      num_channels,
                      c,
                      input_height,
                      input_width,
                      h1 + h1p,
                      w1 + w1p)]));
  }
}

// input is dY, output is dX
template <typename scalar_t>
__global__ void bilinearBackwardKenel(
    const int input_size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const scalar_t* const __restrict__ dY,
    scalar_t* const __restrict__ dX) {

    for (size_t index = blockDim.x * blockIdx.x + threadIdx.x;
         index < input_size; index += blockDim.x * gridDim.x) {

    int indexTemp = index;
    const int in_x = indexTemp % input_width;
    indexTemp /= input_width;
    const int in_y = indexTemp % input_height;
    indexTemp /= input_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_y = fminf(in_y / height_scale, output_height - 1);
    const int out_x = fminf(in_x / width_scale, output_width - 1);

    const float rheight =
        output_height > 1 ? (output_height - 1.f) / (input_height - 1.f) : 0.f;
    const float rwidth =
        output_width > 1 ? (output_width - 1.f) / (input_width - 1.f) : 0.f;

    // Compute Y axis lambdas
    const float h1r = rheight * in_y;
    const int h1 = static_cast<int>(h1r);
    const int h1p = (h1 < output_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.f - h1lambda;

    // Compute X axis lambdas
    const float w1r = rwidth * in_x;
    const int w1 = static_cast<int>(w1r);
    const int w1p = (w1 < output_width - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.f - w1lambda;

    const scalar_t dYi = __ldg(&dY[index]);


    if (std::is_same<scalar_t, __half>::value) {

      fastFp16AtomicAdd(&dX, 
          idx(n, num_channels, c, output_height, output_width, h1, w1),
          numel,
          h0lambda * w0lambda * dYi
      );

/*
__device__ __forceinline__ void fastFp16AtomicAdd(__half* __restrict__ tensor,
                                  int index, int numel,
                                  __half value)
*/

    } else {

      atomicAdd(
          &dX[idx(n, num_channels, c, output_height, output_width, h1, w1)],
          h0lambda * w0lambda * dYi);
      atomicAdd(
          &dX[idx(n, num_channels, c, output_height, output_width, h1, w1 + w1p)],
          h0lambda * w1lambda * dYi);
      atomicAdd(
          &dX[idx(n, num_channels, c, output_height, output_width, h1 + h1p, w1)],
          h1lambda * w0lambda * dYi);
      atomicAdd(
        &dX[idx(
            n,
            num_channels,
            c,
            output_height,
            output_width,
            h1 + h1p,
            w1 + w1p)],
        h1lambda * w1lambda * dYi);
  }
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

  at::Tensor out = at::empty({nIn, cIn, new_h, new_w}, in.options());

  const int outSize = nIn * cIn * new_h * new_w;
  const dim3 block(1024);
  const dim3 grid((outSize + block.x - 1) / block.x);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(in.type(), "bilinearForwardKernel", ([&]
    {
     
        bilinearForwardKernel<scalar_t><<<grid, block>>>(
                                        out.numel(),
                                        cIn,
                                        hIn,
                                        wIn,
                                        new_h,
                                        new_w,
                                        in.data<scalar_t>(),
                                        out.data<scalar_t>()
                                      ); 

    }));

    AT_CHECK(cudaGetLastError() == cudaSuccess,
          "issue with bilinearForwardKernel, CUDA code ",
          cudaGetLastError());

  return out; 
}

at::Tensor bilinear_cuda_backward(at::Tensor& in, const int out_h, const int out_w) {

  const int nIn = in.size(0);
  const int cIn = in.size(1);
  const int hIn = in.size(2);
  const int wIn = in.size(3);

  at::Tensor out = at::empty({nIn, cIn, out_h, out_w}, in.options());

  const int inSize = nIn * cIn * hIn * wIn;
  const dim3 block(1024);
  const dim3 grid((inSize + block.x - 1) / block.x);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(in.type(), "bilinearBackwardKernel", ([&]
    {
     
        bilinearBackwardKernel<scalar_t><<<grid, block>>>(
                                        in.numel(),
                                        cIn,
                                        hIn,
                                        wIn,
                                        out_h,
                                        out_w,
                                        in.data<scalar_t>(),
                                        out.data<scalar_t>()
                                      ); 

    }));

    AT_CHECK(cudaGetLastError() == cudaSuccess,
          "issue with bilinearForwardKernel, CUDA code ",
          cudaGetLastError());
}
