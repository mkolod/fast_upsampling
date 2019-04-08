#include <ATen/ATen.h>

#include "bilinear.h"

/*
  TODO: Check out the MXNet implementation: https://github.com/apache/incubator-mxnet/blob/master/src/operator/bilinear_sampler.cu
*/

at::Tensor bilinear_cuda_forward(at::Tensor z) {
  return z;
}

template<typename scalar_t>
__device__ __forceinline__
static scalar_t linear_upsampling_compute_source_index(
                          scalar_t scale, int dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t ptfive = static_cast<scalar_t>(0.5);
    scalar_t src_idx = scale * (dst_index + ptfive) - ptfive;
    return src_idx < 0 ? 0 : src_idx;
  }
}

template <typename scalar_t>
__global__ void bilinear_cuda_forward2(
    int n,
    int c,
    int h, 
    int w,
    int new_h,
    int new_w,
    const bool align_corners,
    const scalar_t* const __restrict__ data1,
    scalar_t* const __restrict__ data2) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // n = batchSize, c = channels, h = width1, w = width2

  // TODO: use __ldg() for load of input


  if (index < n) {

    const int w2 = index % width2;
    // special case: just copy

    if (w == new_w) {

      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {

//          const Dtype val = data1[n][c][w1];
//          data2[n][c][w2] = val;
        }
      }
      return;
    } else {

    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;


    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const Acctype val = w0lambda * data1[n][c][w1]
                            + w1lambda * data1[n][c][w1+w1p];
        data2[n][c][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }

    }
  }
}


/*
template <typename Dtype, typename Acctype>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const Acctype rwidth, const bool align_corners,
    THCDeviceTensor<Dtype, 3> data1, const THCDeviceTensor<Dtype, 3> data2){
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
          const Dtype val = data2[n][c][w1];
          data1[n][c][w2] += val;
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
        const Dtype d2val = data2[n][c][w2];
        atomicAdd(data1[n][c][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(w0lambda * d2val));
        atomicAdd(data1[n][c][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(w1lambda * d2val));
      }
    }
  }
}

*/
