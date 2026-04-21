// CUDA PSF kernel (Jetson path)
// Implements the SGL PSF intensity profile:
//   PSF(ρ) = μ0 [ J0^2(u) + J1^2(u) ]
//   u = (2π ρ / λ) * sqrt( 2 r_g / z ),  r_g = 2GM/c^2
//
// Notes:
// - CUDA does not provide Bessel J0/J1 natively on all toolchains.
// - For Jetson (CUDA), use device implementations based on cephes-like approximations.
// - This file provides a stable approximation suitable for onboard deconvolution kernels.
// - The μ0 scaling is handled outside (can be absorbed into normalization).

#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ double j0_approx(double x) {
  double ax = fabs(x);
  if (ax < 8.0) {
    double y = x * x;
    double p = 57568490574.0 + y*(-13362590354.0 + y*(651619640.7 + y*(-11214424.18 + y*(77392.33017 + y*(-184.9052456)))));
    double q = 57568490411.0 + y*(1029532985.0 + y*(9494680.718 + y*(59272.64853 + y*(267.8532712 + y*1.0))));
    return p / q;
  } else {
    double z = 8.0 / ax;
    double y = z * z;
    double xx = ax - 0.785398164;
    double p = 1.0 + y*(-0.1098628627e-2 + y*(0.2734510407e-4 + y*(-0.2073370639e-5 + y*0.2093887211e-6)));
    double q = -0.1562499995e-1 + y*(0.1430488765e-3 + y*(-0.6911147651e-5 + y*(0.7621095161e-6 - y*0.934945152e-7)));
    return sqrt(0.636619772 / ax) * (cos(xx)*p - z*sin(xx)*q);
  }
}

__device__ __forceinline__ double j1_approx(double x) {
  double ax = fabs(x);
  if (ax < 8.0) {
    double y = x * x;
    double ans1 = x*(72362614232.0 + y*(-7895059235.0 + y*(242396853.1 + y*(-2972611.439 + y*(15704.48260 + y*(-30.16036606))))));
    double ans2 = 144725228442.0 + y*(2300535178.0 + y*(18583304.74 + y*(99447.43394 + y*(376.9991397 + y*1.0))));
    return ans1/ans2;
  } else {
    double z = 8.0 / ax;
    double y = z * z;
    double xx = ax - 2.356194491;
    double ans1 = 1.0 + y*(0.183105e-2 + y*(-0.3516396496e-4 + y*(0.2457520174e-5 + y*(-0.240337019e-6))));
    double ans2 = 0.04687499995 + y*(-0.2002690873e-3 + y*(0.8449199096e-5 + y*(-0.88228987e-6 + y*0.105787412e-6)));
    double ans = sqrt(0.636619772 / ax) * (cos(xx)*ans1 - z*sin(xx)*ans2);
    return (x < 0.0) ? -ans : ans;
  }
}

__global__ void psf_radial_kernel(const double* rho, double* psf, int n,
                                 double lambda_m, double z_m,
                                 double rg_m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double u = (2.0 * CUDART_PI * rho[i] / lambda_m) * sqrt(fmax(2.0 * rg_m / z_m, 0.0));
  u = fmin(fmax(u, -1e6), 1e6);
  double j0 = j0_approx(u);
  double j1 = j1_approx(u);
  psf[i] = j0*j0 + j1*j1;
}

extern "C" void sgl_psf_radial_cuda(const double* d_rho, double* d_psf, int n,
                                   double lambda_m, double z_m, double rg_m,
                                   cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  psf_radial_kernel<<<grid, block, 0, stream>>>(d_rho, d_psf, n, lambda_m, z_m, rg_m);
}
