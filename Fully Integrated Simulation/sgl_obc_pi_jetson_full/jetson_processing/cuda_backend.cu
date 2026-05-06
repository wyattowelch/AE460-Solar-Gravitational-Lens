#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sgl {

namespace {

__global__ void finalize_accum_kernel(const float* acc_r,
                                      const float* acc_g,
                                      const float* acc_b,
                                      const float* wsum,
                                      float wmax,
                                      int n,
                                      unsigned char* out_rgb,
                                      float* out_conf) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float w = fmaxf(1e-6f, wsum[i]);
  float r = acc_r[i] / w;
  float g = acc_g[i] / w;
  float b = acc_b[i] / w;
  r = fminf(255.0f, fmaxf(0.0f, r));
  g = fminf(255.0f, fmaxf(0.0f, g));
  b = fminf(255.0f, fmaxf(0.0f, b));
  out_rgb[3 * i + 0] = static_cast<unsigned char>(r + 0.5f);
  out_rgb[3 * i + 1] = static_cast<unsigned char>(g + 0.5f);
  out_rgb[3 * i + 2] = static_cast<unsigned char>(b + 0.5f);
  out_conf[i] = fminf(1.0f, fmaxf(0.0f, w / fmaxf(1e-6f, wmax)));
}

static const char* cuda_err(cudaError_t e) {
  return cudaGetErrorString(e);
}

}  // namespace

bool cuda_runtime_available(std::string& reason) {
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e != cudaSuccess) {
    reason = std::string("cuda_runtime_error: ") + cuda_err(e);
    return false;
  }
  if (ndev <= 0) {
    reason = "cuda_runtime_no_devices";
    return false;
  }
  reason = "cuda_runtime_ok";
  return true;
}

bool cuda_finalize_accum_to_image(const std::vector<float>& acc_r,
                                  const std::vector<float>& acc_g,
                                  const std::vector<float>& acc_b,
                                  const std::vector<float>& wsum,
                                  float wmax,
                                  unsigned outW,
                                  unsigned outH,
                                  std::vector<uint8_t>& out_rgba,
                                  std::vector<float>& out_conf,
                                  std::string& err) {
  const int n = static_cast<int>(outW * outH);
  if (n <= 0) {
    err = "empty output shape";
    return false;
  }
  if ((int)acc_r.size() != n || (int)acc_g.size() != n || (int)acc_b.size() != n || (int)wsum.size() != n) {
    err = "accumulator size mismatch";
    return false;
  }

  float *d_r = nullptr, *d_g = nullptr, *d_b = nullptr, *d_w = nullptr, *d_conf = nullptr;
  unsigned char* d_rgb = nullptr;
  std::vector<unsigned char> rgb3(static_cast<size_t>(n) * 3ull);
  out_conf.assign(static_cast<size_t>(n), 0.0f);
  out_rgba.assign(static_cast<size_t>(n) * 4ull, 255);

  auto cleanup = [&]() {
    cudaFree(d_r); cudaFree(d_g); cudaFree(d_b); cudaFree(d_w); cudaFree(d_rgb); cudaFree(d_conf);
  };

  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&d_r, sizeof(float) * n); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMalloc(&d_g, sizeof(float) * n); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMalloc(&d_b, sizeof(float) * n); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMalloc(&d_w, sizeof(float) * n); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMalloc(&d_rgb, sizeof(unsigned char) * 3ull * n); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMalloc(&d_conf, sizeof(float) * n); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }

  e = cudaMemcpy(d_r, acc_r.data(), sizeof(float) * n, cudaMemcpyHostToDevice); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMemcpy(d_g, acc_g.data(), sizeof(float) * n, cudaMemcpyHostToDevice); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMemcpy(d_b, acc_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMemcpy(d_w, wsum.data(), sizeof(float) * n, cudaMemcpyHostToDevice); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }

  constexpr int block = 256;
  const int grid = (n + block - 1) / block;
  finalize_accum_kernel<<<grid, block>>>(d_r, d_g, d_b, d_w, wmax, n, d_rgb, d_conf);
  e = cudaGetLastError();
  if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }

  e = cudaMemcpy(rgb3.data(), d_rgb, sizeof(unsigned char) * 3ull * n, cudaMemcpyDeviceToHost); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }
  e = cudaMemcpy(out_conf.data(), d_conf, sizeof(float) * n, cudaMemcpyDeviceToHost); if (e != cudaSuccess) { err = cuda_err(e); cleanup(); return false; }

  for (int i = 0; i < n; ++i) {
    out_rgba[4ull * i + 0] = rgb3[3ull * i + 0];
    out_rgba[4ull * i + 1] = rgb3[3ull * i + 1];
    out_rgba[4ull * i + 2] = rgb3[3ull * i + 2];
    out_rgba[4ull * i + 3] = 255;
  }

  cleanup();
  return true;
}

}  // namespace sgl
