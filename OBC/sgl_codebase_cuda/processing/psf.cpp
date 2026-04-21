#include "psf.hpp"
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sgl {

/*
Paper-equation form (Turyshev/Toth style):

Let r_g = 2GM/c^2, z = heliocentric distance of receiver (m), wavelength λ (m),
and ρ be radial offset in the image plane (m). Define:
  u(ρ) = (2π ρ / λ) * sqrt( 2 r_g / z ).

Then an often-used (monochromatic, scalar) SGL PSF intensity model is:
  PSF(ρ) = μ0 [ J0^2(u) + J1^2(u) ]

In the flight pipeline, PSF enters as the convolution kernel H such that:
  I_obs = H * I_src + n

and reconstruction uses Wiener/Tikhonov in Fourier domain:
  I_src_hat(k) = H*(k) / (|H(k)|^2 + γ) * I_obs_hat(k)

This file implements a stable approximation for J0/J1 so it runs on plain C++.
*/

static inline double j0_approx(double x){
  double ax = std::fabs(x);
  if(ax < 8.0){
    double y = x*x;
    double p = 57568490574.0 + y*(-13362590354.0 + y*(651619640.7 + y*(-11214424.18 + y*(77392.33017 + y*(-184.9052456)))));
    double q = 57568490411.0 + y*(1029532985.0 + y*(9494680.718 + y*(59272.64853 + y*(267.8532712 + y*1.0))));
    return p/q;
  } else {
    double z = 8.0/ax;
    double y = z*z;
    double xx = ax - 0.785398164;
    double p = 1.0 + y*(-0.1098628627e-2 + y*(0.2734510407e-4 + y*(-0.2073370639e-5 + y*0.2093887211e-6)));
    double q = -0.1562499995e-1 + y*(0.1430488765e-3 + y*(-0.6911147651e-5 + y*(0.7621095161e-6 - y*0.934945152e-7)));
    return std::sqrt(0.636619772/ax) * (std::cos(xx)*p - z*std::sin(xx)*q);
  }
}
static inline double j1_approx(double x){
  double ax = std::fabs(x);
  if(ax < 8.0){
    double y = x*x;
    double ans1 = x*(72362614232.0 + y*(-7895059235.0 + y*(242396853.1 + y*(-2972611.439 + y*(15704.48260 + y*(-30.16036606))))));
    double ans2 = 144725228442.0 + y*(2300535178.0 + y*(18583304.74 + y*(99447.43394 + y*(376.9991397 + y*1.0))));
    return ans1/ans2;
  } else {
    double z = 8.0/ax;
    double y = z*z;
    double xx = ax - 2.356194491;
    double ans1 = 1.0 + y*(0.183105e-2 + y*(-0.3516396496e-4 + y*(0.2457520174e-5 + y*(-0.240337019e-6))));
    double ans2 = 0.04687499995 + y*(-0.2002690873e-3 + y*(0.8449199096e-5 + y*(-0.88228987e-6 + y*0.105787412e-6)));
    double ans = std::sqrt(0.636619772/ax) * (std::cos(xx)*ans1 - z*std::sin(xx)*ans2);
    return (x < 0.0) ? -ans : ans;
  }
}

double psf_radial(double rho, double lambda_m, double z_m){
  constexpr double G = 6.67430e-11;
  constexpr double c = 299792458.0;
  constexpr double M = 1.98847e30;
  const double rg = 2.0*G*M/(c*c);
  const double u = (2.0*M_PI*rho/lambda_m) * std::sqrt(std::max(2.0*rg/z_m, 0.0));
  double uu = std::clamp(u, -1e6, 1e6);
  double j0=j0_approx(uu), j1=j1_approx(uu);
  return j0*j0 + j1*j1;
}

std::vector<double> build_psf_kernel(int ksize, double sigma_px){
  if(ksize<3) ksize=3;
  if(ksize%2==0) ksize+=1;
  int r = ksize/2;
  std::vector<double> k(ksize*ksize,0.0);
  double sum=0;
  #pragma omp parallel for reduction(+:sum) collapse(2) schedule(static)
  for(int y=-r;y<=r;y++){
    for(int x=-r;x<=r;x++){
      double rr = std::sqrt(double(x*x+y*y));
      double v = std::exp(-(rr*rr)/(2.0*sigma_px*sigma_px));
      k[(y+r)*ksize + (x+r)] = v;
      sum += v;
    }
  }
  for(auto& v: k) v /= sum;
  return k;
}

static inline int wrap(int a, int n){ a%=n; if(a<0) a+=n; return a; }

static void dft2(const std::vector<double>& in, int w, int h,
                 std::vector<double>& re, std::vector<double>& im, bool inverse){
  re.assign(w*h,0.0); im.assign(w*h,0.0);
  double sgn = inverse ? +1.0 : -1.0;
  double norm = inverse ? 1.0/(w*h) : 1.0;
  #pragma omp parallel for collapse(2) schedule(static)
  for(int ky=0;ky<h;ky++){
    for(int kx=0;kx<w;kx++){
      double sumre=0, sumim=0;
      for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
          double ang = 2*M_PI*( (double)kx*x/w + (double)ky*y/h );
          double c = std::cos(ang);
          double s = std::sin(ang)*sgn;
          double v = in[y*w+x];
          sumre += v*c;
          sumim += v*s;
        }
      }
      re[ky*w+kx] = sumre*norm;
      im[ky*w+kx] = sumim*norm;
    }
  }
}

void wiener_deconvolve_gray(const std::vector<double>& input,
                            int w, int h,
                            const std::vector<double>& psf,
                            int ksize,
                            double gamma,
                            std::vector<double>& output){
  std::vector<double> H(w*h,0.0);
  int r = ksize/2;
  for(int y=0;y<ksize;y++){
    for(int x=0;x<ksize;x++){
      int yy = wrap(y-r, h);
      int xx = wrap(x-r, w);
      H[yy*w+xx] = psf[y*ksize+x];
    }
  }
  std::vector<double> Ire,Iim, Hre,Him;
  dft2(input,w,h,Ire,Iim,false);
  dft2(H,w,h,Hre,Him,false);
  std::vector<double> Ore(w*h,0.0), Oim(w*h,0.0);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<w*h;i++){
    double a=Hre[i], b=Him[i];
    double denom = a*a + b*b + gamma;
    double cr = a/denom;
    double ci = -b/denom;
    Ore[i] = cr*Ire[i] - ci*Iim[i];
    Oim[i] = cr*Iim[i] + ci*Ire[i];
  }
  // inverse (approx: real-part)
  std::vector<double> tmpi;
  dft2(Ore,w,h,output,tmpi,true);
  for(auto& v: output){ v = std::clamp(v, 0.0, 1.0); }
}

} // namespace sgl
