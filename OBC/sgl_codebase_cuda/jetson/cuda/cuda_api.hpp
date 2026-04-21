#pragma once
#include <cstddef>
#ifdef SGL_ENABLE_CUDA
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif

extern "C" void sgl_psf_radial_cuda(const double* d_rho, double* d_psf, int n,
                                   double lambda_m, double z_m, double rg_m,
                                   cudaStream_t stream);
extern "C" int sgl_wiener_deconv_cuda(void* opaque,
                                     int w, int h,
                                     double gamma, cudaStream_t stream);
