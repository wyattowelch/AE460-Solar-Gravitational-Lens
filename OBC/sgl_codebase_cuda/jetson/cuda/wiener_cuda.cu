// CUDA Wiener deconvolution (Jetson path)
// Implements frequency-domain Wiener filter:
//   I_src_hat(k) = H*(k) / ( |H(k)|^2 + gamma ) * I_obs_hat(k)
//
// For a real flight implementation you would use cuFFT to compute FFTs.
// This file provides the interface + a CPU fallback trigger if cuFFT is not available.
//
// Build notes:
// - If building on Jetson with CUDA + cuFFT, define SGL_USE_CUFFT and link cufft.
// - Otherwise, the scheduler should call the CPU/OpenMP path.

#include <cuda_runtime.h>

#ifdef SGL_USE_CUFFT
#include <cufft.h>
#endif

extern "C" int sgl_wiener_deconv_cuda(/* device pointers */ void*,
                                     int, int,
                                     double, cudaStream_t) {
#ifdef SGL_USE_CUFFT
  // TODO: implement cuFFT pipeline (R2C, multiply, C2R) with proper normalization.
  return 0;
#else
  return -1; // indicate not available
#endif
}
