#include <type_traits>

#include <ATen/ATen.h>

#include "bilinear.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>

#if __CUDA_ARCH__ >= 350
// Device has __ldg
template <typename T>
__device__ __forceinline__ T __ldg(const T *ptr) {
  typedef typename detail::working_array<T>::type aliased;
  aliased storage = detail::load_storage<T>::impl(ptr);
  return detail::fuse<T>(storage);
}

#else
template <typename T>
__device__ __forceinline__ T __ldg(const T *ptr) {
  return *ptr;
}

#endif

template <typename scalar_t, typename std::enable_if<std::is_same<
                                 c10::Half, scalar_t>::value>::type * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(scalar_t *tensor,
                                                         int index, int numel,
                                                         scalar_t value) {
#if ((CUDA_VERSION < 10000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(reinterpret_cast<at::Half *>(tensor) + index,
            static_cast<at::Half>(value));
#else
  if (index % 2 == 0 && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2 *>(tensor) + index / 2, value2);

  } else if (index % 2 == 1) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2 *>(tensor) + index / 2, value2);

  } else {
    atomicAdd(reinterpret_cast<__half *>(tensor) + index,
              static_cast<__half>(value));
  }
#endif
}

template <typename scalar_t, typename std::enable_if<!std::is_same<
                                 c10::Half, scalar_t>::value>::type * = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(scalar_t *tensor,
                                                         int index, int numel,
                                                         scalar_t value) {
  atomicAdd(tensor + index, value);
}

template <class scalar_t>
__device__ __forceinline__ void fastAtomicAdd(scalar_t *__restrict__ tensor,
                                              int index, int numel,
                                              scalar_t value) {
  fastSpecializedAtomicAdd(tensor, index, numel, value);
}

__device__ __forceinline__ int idx(const int n, const int num_channels,
                                   const int c, const int height,
                                   const int width, const int y, const int x) {
  return ((n * num_channels + c) * height + y) * width + x;
}

// input is X, output is Y
template <typename scalar_t>
__global__ void bilinearForwardKernel(
    const int output_size, const int num_channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const scalar_t *const __restrict__ X, scalar_t *const __restrict__ Y) {
  const float height_scale = 1.0f * output_height / input_height;
  const float width_scale = 1.0f * output_width / input_width;

  const int batch_size =
      output_size / num_channels / output_height / output_width;

  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  int indexTemp = index;
  const int out_x = indexTemp % output_width;
  indexTemp /= output_width;
  const int out_y = indexTemp % output_height;

  const int in_y = fminf(out_y / height_scale, input_height - 1);
  const int in_x = fminf(out_x / width_scale, input_width - 1);

  const float rheight =
      output_height > 1 ? (input_height - 1.f) / (output_height - 1.f) : 0.f;
  const float rwidth =
      output_width > 1 ? (input_width - 1.f) / (output_width - 1.f) : 0.f;

  const float h1r = rheight * out_y;
  const int h1 = static_cast<int>(h1r);
  const int h1p = (h1 < input_height - 1) ? 1 : 0;
  const float h1lambda = h1r - h1;
  const float h0lambda = 1.f - h1lambda;

  const float w1r = rwidth * out_x;
  const int w1 = static_cast<int>(w1r);
  const int w1p = (w1 < input_width - 1) ? 1 : 0;
  const float w1lambda = w1r - w1;
  const float w0lambda = 1.f - w1lambda;

  for (int n = 0; n < batch_size; n++) {
    for (int c = 0; c < num_channels; c++) {
      Y[idx(n, num_channels, c, output_height, output_width, out_y, out_x)] =
          static_cast<scalar_t>(
              h0lambda *
                  (w0lambda * __ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1, w1)]) +
                   w1lambda * __ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1, w1 + w1p)])) +
              h1lambda *
                  (w0lambda * __ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1 + h1p, w1)]) +
                   w1lambda * __ldg(&X[idx(n, num_channels, c, input_height,
                                           input_width, h1 + h1p, w1 + w1p)])));
    }
  }
}

// TODO: Launch this with thread per gradInput instead of gradOutput
// input is dY, output is dX
template <typename scalar_t>
__global__ void bilinearBackwardKernel2(
    const int input_size, const int num_channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const scalar_t *const __restrict__ dY, scalar_t *const __restrict__ dX) {
  const float height_scale = 1.0f * output_height / input_height;
  const float width_scale = 1.0f * output_width / input_width;

  const int index = blockDim.x * blockIdx.x + threadIdx.x;

  int indexTemp = index;
  const int in_x = indexTemp % input_width;
  indexTemp /= input_width;
  const int in_y = indexTemp % input_height;
  indexTemp /= input_height;
  const int c = indexTemp % num_channels;
  indexTemp /= num_channels;

  //  const int n = indexTemp;
  const int n = 0;

  if (index > input_size - 1) {
    return;
  }

  const int dst_idx =
      idx(n, num_channels, c, input_height, input_width, in_y, in_x);

  // accumulator
  float acc = 0.0f;

  // TODO: figure out which Ys to loop over
  // TODO: figure out what lambdas to use for which Y

  const int y_window = ceilf(height_scale);
  const int x_window = ceilf(width_scale);

  // Not expecting overflow here, static_cast<int>(roundf(x)) is ugly
  const int y_base_idx = lroundf(in_y * height_scale);
  const int x_base_idx = lroundf(in_x * width_scale);

  int ctr = 0;

  for (int out_y = y_base_idx;
       out_y <= min(y_base_idx + y_window, output_height - 1); out_y++) {
    for (int out_x = x_base_idx;
         out_x <= min(x_base_idx + x_window, output_width - 1); out_x++) {
      ctr += 1;

      const int src_idx =
          idx(n, num_channels, c, output_height, output_width, out_y, out_x);

      // TODO: calculate lambdas for y and x !!

      acc += dY[src_idx];
    }
  }

  if (index == 100) {
    printf("index = %d, ctr = %d\n", index, ctr);
  }

  dX[dst_idx] = static_cast<scalar_t>(acc);
}

// input is dY, output is dX
template <typename scalar_t>
__global__ void bilinearBackwardKernel(
    const int input_size, const int num_channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const scalar_t *const __restrict__ dY, scalar_t *const __restrict__ dX) {
  const float height_scale = 1.0f * output_height / input_height;
  const float width_scale = 1.0f * output_width / input_width;

  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < input_size;
       index += blockDim.x * gridDim.x) {
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

    const int out_numel = input_size / (input_height * input_width) *
                          output_height * output_width;

    if (n == 2 && c == 1 && h1 == 12 && w1 == 14) {
      int idx0 = idx(n, num_channels, c, output_height, output_width, h1, w1);

      printf("n = %d, c = %d, h1 = %d, w1 = %d, dX idx = %d, dY idx = %d\n", n,
             c, h1, w1, idx0, (int)index);
    }

    fastAtomicAdd<scalar_t>(
        dX, idx(n, num_channels, c, output_height, output_width, h1, w1),
        out_numel, static_cast<scalar_t>(h0lambda * w0lambda * dYi));

    fastAtomicAdd<scalar_t>(
        dX, idx(n, num_channels, c, output_height, output_width, h1, w1 + w1p),
        out_numel, static_cast<scalar_t>(h0lambda * w1lambda * dYi));

    fastAtomicAdd<scalar_t>(
        dX, idx(n, num_channels, c, output_height, output_width, h1 + h1p, w1),
        out_numel, static_cast<scalar_t>(h1lambda * w0lambda * dYi));

    fastAtomicAdd<scalar_t>(dX, idx(n, num_channels, c, output_height,
                                    output_width, h1 + h1p, w1 + w1p),
                            out_numel,
                            static_cast<scalar_t>(h1lambda * w1lambda * dYi));
  }
}

at::Tensor bilinear_cuda_forward(at::Tensor &in, const int new_h,
                                 const int new_w) {
  const int nIn = in.size(0);
  const int cIn = in.size(1);
  const int hIn = in.size(2);
  const int wIn = in.size(3);

  at::Tensor out = at::empty({nIn, cIn, new_h, new_w}, in.options());

  const int outSize = nIn * cIn * new_h * new_w;
  const dim3 block(256);
  const dim3 grid(((outSize / cIn / nIn) + block.x - 1) / block.x);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      in.type(), "bilinearForwardKernel", ([&] {

        bilinearForwardKernel<scalar_t><<<grid, block>>>(
            out.numel(), cIn, hIn, wIn, new_h, new_w, in.data<scalar_t>(),
            out.data<scalar_t>());

      }));

  AT_CHECK(cudaGetLastError() == cudaSuccess,
           "issue with bilinearForwardKernel, CUDA code ", cudaGetLastError());

  return out;
}

at::Tensor bilinear_cuda_backward(at::Tensor &in, const int out_h,
                                  const int out_w) {
  const int nIn = in.size(0);
  const int cIn = in.size(1);
  const int hIn = in.size(2);
  const int wIn = in.size(3);

  at::Tensor out = at::empty({nIn, cIn, out_h, out_w}, in.options());

  /*
    const int outSize = nIn * cIn * out_h * out_w;
    const dim3 block(256);
    const dim3 grid((outSize + block.x - 1) / block.x);
  */

  const int inSize = nIn * cIn * hIn * wIn;
  const dim3 block(256);
  const dim3 grid((inSize + block.x - 1) / block.x);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      in.type(), "bilinearBackwardKernel", ([&] {

        // bilinearBackwardKernel2
        bilinearBackwardKernel<scalar_t><<<grid, block>>>(
            in.numel(), cIn, hIn, wIn, out_h, out_w, in.data<scalar_t>(),
            out.data<scalar_t>());

      }));

  AT_CHECK(cudaGetLastError() == cudaSuccess,
           "issue with bilinearForwardKernel, CUDA code ", cudaGetLastError());

  return out;
}
