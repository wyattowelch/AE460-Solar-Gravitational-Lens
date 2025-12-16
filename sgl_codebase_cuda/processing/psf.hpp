#pragma once
#include <vector>
#include <cstdint>

namespace sgl {

// SGL PSF intensity model (monochromatic):
//   PSF(ρ) = μ0 [ J0^2(u) + J1^2(u) ],  u = (2πρ/λ) sqrt(2 r_g / z),  r_g=2GM/c^2
double psf_radial(double rho, double lambda_m, double z_m);

// Build a small PSF kernel (odd size) normalized to sum=1 (demo Gaussian kernel).
std::vector<double> build_psf_kernel(int ksize, double sigma_px);

// Wiener deconvolution (CPU/OpenMP), frequency-domain filter:
//   I_src_hat(k) = H*(k) / (|H(k)|^2 + γ) * I_obs_hat(k)
void wiener_deconvolve_gray(const std::vector<double>& input,
                            int w, int h,
                            const std::vector<double>& psf,
                            int ksize,
                            double gamma,
                            std::vector<double>& output);

} // namespace sgl
