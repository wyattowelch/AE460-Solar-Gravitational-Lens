// Einstein Ring from Image + Full Tile Rows + Reconstruction + PSF (OpenMP + PNG/PGM)
// ----------------------------------------------------------------------------------
// 1) Put a PNG image next to the executable (e.g., bluemarble.png).
// 2) Build: g++ -O3 -fopenmp -std=c++17 main.cpp lodepng.cpp -o sgl_ring_from_image
// 3) Run:   ./sgl_ring_from_image
//
// Outputs in ./out/<BASE_NAME>/:
//   * <base>_ring_gray.pgm, <base>_ring_gray.png         (continuous ring preview)
//   * <base>_ring_color.png                              (colored ring preview)
//   * <base>_src_with_tiles.png                          (source + grid overlay)
//   * <base>_tiles.csv                                   (tile_id,row,col,mean_R,mean_G,mean_B,mean_L)
//   * <base>_ring_from_tiles.png                         (equator-row preview)
//   * <base>_reconstructed_from_tiles.png                (mosaic at tile resolution)
//   * <base>_ring_from_row_rXX.png                       (one ring per tile row)
//
// Ring radial profile can be:
//   - Gaussian (old behavior), or
//   - PSF-like: J0^2( scale * (r - RING_RADIUS) ).
// ----------------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lodepng.h"

// ===================== USER PARAMETERS =============================

// Base name for outputs (folder out/BASE_NAME/)
static std::string BASE_NAME   = "earth_demo";

// Input source image (PNG)
static std::string SRC_PNG     = "bluemarble.png";

// Ring render (display canvas)
static int    IMG_N        = 1536;   // output resolution (ring canvases)
static double RING_RADIUS  = 0.38;   // radius in normalized [-1,1] coords
static double RING_SIGMA   = 0.04;   // characteristic radial scale (~ring thickness)
static bool   WRITE_PGM    = true;   // also write PGM (besides PNG)

// Choose radial profile:
//   USE_PSF = false -> Gaussian profile
//   USE_PSF = true  -> PSF-like J0^2 profile
static bool   USE_PSF      = true;

// PSF scale factor: J0 zero is at ~2.4048, so
// arg = PSF_SCALE * (r - RING_RADIUS)
// with PSF_SCALE ~ 2.4048 / RING_SIGMA puts first zero ~1*RING_SIGMA away.
static double PSF_SCALE    = 2.4048255577 / RING_SIGMA;

// Colormap toggle for grayscale → false ⇒ grayscale ring; true ⇒ inferno-like colors
static bool   COLOR_MAP    = true;

// Tile "scan" parameters (piece-by-piece), driven by image resolution:
// approximate tile size in pixels; code will derive tile counts automatically.
static int    TILE_PIX_X    = 8;    // desired tile width (px) on the source image
static int    TILE_PIX_Y    = 8;    // desired tile height (px) on the source image
static int    MAX_TILES_DIM = 256;   // safety cap on tiles per dimension

// Draw grid on source overlay?
static bool   DRAW_TILE_GRID_ON_SRC = true;

// Output control
static bool   WRITE_ROW_RINGS = true;  // write one ring per tile row
// ==================================================================

// Simple RGB struct
struct RGBd { double r,g,b; };
struct RGBA8 { uint8_t r,g,b,a; };

// Inferno-like colormap
static RGBd lerp(const RGBd& a, const RGBd& b, double t){
  return { a.r + (b.r-a.r)*t, a.g + (b.g-a.g)*t, a.b + (b.b-a.b)*t };
}
static RGBd inferno_like(double x){
  x = std::clamp(x, 0.0, 1.0);
  static const double s[5] = {0.0, 0.25, 0.50, 0.75, 1.0};
  static const RGBd   c[5] = {
    {0.00, 0.00, 0.00},
    {0.16, 0.05, 0.33},
    {0.73, 0.22, 0.33},
    {0.98, 0.56, 0.05},
    {0.99, 0.99, 0.64}
  };
  if (x <= s[0]) return c[0];
  if (x >= s[4]) return c[4];
  int i=0; while (i<4 && x>s[i+1]) ++i;
  double t = (x - s[i]) / (s[i+1] - s[i]);
  return lerp(c[i], c[i+1], t);
}

// PNG helpers
static bool write_png_rgba(const std::string& filename,
                           const std::vector<uint8_t>& rgba,
                           unsigned w, unsigned h)
{
  unsigned err = lodepng::encode(filename, rgba, w, h);
  if (err) {
    std::cerr << "PNG encode error " << err << ": " << lodepng_error_text(err) << "\n";
    return false;
  }
  return true;
}

static bool write_png_gray8(const std::string& filename,
                            const std::vector<uint8_t>& gray,
                            unsigned w, unsigned h)
{
  std::vector<uint8_t> rgba(4ull*w*h);
  for (size_t i=0;i<gray.size();++i){
    uint8_t g = gray[i];
    rgba[4*i+0]=g; rgba[4*i+1]=g; rgba[4*i+2]=g; rgba[4*i+3]=255;
  }
  return write_png_rgba(filename, rgba, w, h);
}

static bool write_pgm_gray8(const std::string& filename,
                            const std::vector<uint8_t>& gray,
                            unsigned w, unsigned h)
{
  std::ofstream f(filename, std::ios::binary);
  if(!f){ std::cerr<<"Failed to open "<<filename<<" for write\n"; return false; }
  f<<"P5\n"<<w<<" "<<h<<"\n255\n";
  f.write(reinterpret_cast<const char*>(gray.data()), gray.size());
  return true;
}

// Load PNG (RGBA8)
static bool load_png_rgba(const std::string& path,
                          std::vector<uint8_t>& rgba,
                          unsigned& w, unsigned& h)
{
  std::vector<unsigned char> img;
  unsigned err = lodepng::decode(img, w, h, path);
  if (err) {
    std::cerr << "PNG decode error " << err << ": " << lodepng_error_text(err)
              << "\nWhile opening: " << path << "\n";
    return false;
  }
  rgba.assign(img.begin(), img.end());
  return true;
}

// Draw a simple rectangle outline onto an RGBA image
static void draw_rect_outline(std::vector<uint8_t>& img, unsigned w, unsigned h,
                              int x0, int y0, int x1, int y1, RGBA8 color)
{
  x0 = std::max(0,x0); y0 = std::max(0,y0);
  x1 = std::min((int)w-1,x1); y1 = std::min((int)h-1,y1);
  for(int x=x0; x<=x1; ++x){
    int y=y0; size_t i=4*(y*w + x);
    img[i]=color.r; img[i+1]=color.g; img[i+2]=color.b; img[i+3]=color.a;
    y=y1; i=4*(y*w + x);
    img[i]=color.r; img[i+1]=color.g; img[i+2]=color.b; img[i+3]=color.a;
  }
  for(int y=y0; y<=y1; ++y){
    int x=x0; size_t i=4*(y*w + x);
    img[i]=color.r; img[i+1]=color.g; img[i+2]=color.b; img[i+3]=color.a;
    x=x1; i=4*(y*w + x);
    img[i]=color.r; img[i+1]=color.g; img[i+2]=color.b; img[i+3]=color.a;
  }
}

// Fill rectangle solid color on RGBA
static void fill_rect(std::vector<uint8_t>& img, unsigned w, unsigned h,
                      int x0, int y0, int x1, int y1,
                      double R, double G, double B)
{
  x0 = std::max(0,x0); y0 = std::max(0,y0);
  x1 = std::min((int)w-1,x1); y1 = std::min((int)h-1,y1);
  for(int y=y0; y<=y1; ++y){
    size_t i = 4ull*(y*w + x0);
    for(int x=x0; x<=x1; ++x, i+=4){
      img[i+0] = (uint8_t)std::round(255.0*std::clamp(R,0.0,1.0));
      img[i+1] = (uint8_t)std::round(255.0*std::clamp(G,0.0,1.0));
      img[i+2] = (uint8_t)std::round(255.0*std::clamp(B,0.0,1.0));
      img[i+3] = 255;
    }
  }
}

// Sample source by angle (θ→column) from equator (for continuous ring preview)
static RGBd sample_source_by_theta_equator(const std::vector<uint8_t>& src,
                                           unsigned sw, unsigned sh, double theta)
{
  double u = (theta + M_PI) / (2.0*M_PI); // 0..1
  double colf = u * sw;
  int c0 = (int)std::floor(colf) % (int)sw;
  int c1 = (c0 + 1) % (int)sw;
  double t = colf - std::floor(colf);

  int row = (int)(0.5 * sh); // central row (equator)
  const uint8_t* p0 = &src[4*(row*sw + c0)];
  const uint8_t* p1 = &src[4*(row*sw + c1)];
  RGBd A{ p0[0]/255.0, p0[1]/255.0, p0[2]/255.0 };
  RGBd B{ p1[0]/255.0, p1[1]/255.0, p1[2]/255.0 };
  return lerp(A,B,t);
}

// === Radial profile: Gaussian vs PSF (J0^2) =======================

// Simple Gaussian ring profile
static inline double radial_gaussian(double r, double r0, double sigma){
  double dr = r - r0;
  return std::exp(-(dr*dr)/(2.0*sigma*sigma));
}

// PSF-like profile using J0^2 around r0.
// arg = PSF_SCALE * (r - r0)
static inline double radial_psf(double r, double r0){
  double dr = r - r0;

  // Optional cutoff: ignore far-away pixels for speed
  if (std::fabs(dr) > 5.0 * RING_SIGMA) return 0.0;

  double arg = PSF_SCALE * dr;

  // J0 is even: J0(-x) = J0(x). Some libstdc++ builds throw on negative args,
  // so we take fabs(arg) to stay in the allowed domain.
  double j0  = std::cyl_bessel_j(0.0, std::fabs(arg));

  double val = j0*j0; // intensity
  if (val < 0.0) val = 0.0;
  return val;
}

// Wrapper: choose profile based on USE_PSF
static inline double radial_profile(double r, double r0, double sigma){
  if (USE_PSF) return radial_psf(r, r0);
  return radial_gaussian(r, r0, sigma);
}
// ==================================================================

int main(){
  namespace fs = std::filesystem;

  // Load source PNG
  std::vector<uint8_t> src_rgba;
  unsigned sw=0, sh=0;
  if(!load_png_rgba(SRC_PNG, src_rgba, sw, sh)) {
    std::cerr<<"Supply a PNG image named "<<SRC_PNG<<" (convert JPEG to PNG if needed).\n";
    return 1;
  }
  std::cerr<<"Loaded source "<<SRC_PNG<<" ("<<sw<<"x"<<sh<<")\n";

  fs::path outdir = fs::path("out") / BASE_NAME;
  fs::create_directories(outdir);

  const double xmin=-1.0, xmax=+1.0;
  const double scale = (xmax - xmin) / (double)IMG_N;

  // === 1) Continuous Einstein ring preview (equator-sampled) ===
  std::vector<uint8_t> ring_gray(IMG_N*IMG_N, 0);

  #pragma omp parallel for collapse(2) schedule(static)
  for(int y=0; y<IMG_N; ++y){
    for(int x=0; x<IMG_N; ++x){
      double X = xmin + (x+0.5)*scale;
      double Y = xmin + (y+0.5)*scale;
      double r = std::sqrt(X*X + Y*Y);
      double theta = std::atan2(Y,X);

      double I_rad = radial_profile(r, RING_RADIUS, RING_SIGMA); // Gaussian or PSF

      RGBd C = sample_source_by_theta_equator(src_rgba, sw, sh, theta);
      double L = std::clamp(0.2126*C.r + 0.7152*C.g + 0.0722*C.b, 0.0, 1.0);

      double val = std::clamp(I_rad * L, 0.0, 1.0);
      ring_gray[y*IMG_N + x] = (uint8_t)std::round(val*255.0);
    }
  }

  std::string pgm_name = (outdir / (BASE_NAME + "_ring_gray.pgm")).string();
  std::string png_gray = (outdir / (BASE_NAME + "_ring_gray.png")).string();
  if (WRITE_PGM) {
    if (write_pgm_gray8(pgm_name, ring_gray, IMG_N, IMG_N))
      std::cerr<<"Wrote "<<pgm_name<<"\n";
  }
  if (write_png_gray8(png_gray, ring_gray, IMG_N, IMG_N))
    std::cerr<<"Wrote "<<png_gray<<"\n";

  std::vector<uint8_t> ring_rgba(4ull*IMG_N*IMG_N, 0);
  #pragma omp parallel for collapse(2) schedule(static)
  for(int y=0; y<IMG_N; ++y){
    for(int x=0; x<IMG_N; ++x){
      double X = xmin + (x+0.5)*scale;
      double Y = xmin + (y+0.5)*scale;
      double r = std::sqrt(X*X + Y*Y);
      double theta = std::atan2(Y,X);

      double I_rad = radial_profile(r, RING_RADIUS, RING_SIGMA);

      RGBd Csrc = sample_source_by_theta_equator(src_rgba, sw, sh, theta);
      double L = std::clamp(0.2126*Csrc.r + 0.7152*Csrc.g + 0.0722*Csrc.b, 0.0, 1.0);

      RGBd Cmap = COLOR_MAP ? inferno_like(L) : RGBd{L,L,L};
      double R = std::clamp(I_rad*Cmap.r, 0.0, 1.0);
      double G = std::clamp(I_rad*Cmap.g, 0.0, 1.0);
      double B = std::clamp(I_rad*Cmap.b, 0.0, 1.0);

      size_t i = 4ull*(y*IMG_N + x);
      ring_rgba[i+0] = (uint8_t)std::round(255.0*R);
      ring_rgba[i+1] = (uint8_t)std::round(255.0*G);
      ring_rgba[i+2] = (uint8_t)std::round(255.0*B);
      ring_rgba[i+3] = 255;
    }
  }
  std::string png_color = (outdir / (BASE_NAME + "_ring_color.png")).string();
  if (write_png_rgba(png_color, ring_rgba, IMG_N, IMG_N))
    std::cerr<<"Wrote "<<png_color<<"\n";

  // === 2) Derive tile counts from image resolution and tile size ===
  int tileW = std::max(1, TILE_PIX_X);
  int tileH = std::max(1, TILE_PIX_Y);

  int tx = std::max(1, (int)sw / tileW);
  int ty = std::max(1, (int)sh / tileH);

  tx = std::min(tx, MAX_TILES_DIM);
  ty = std::min(ty, MAX_TILES_DIM);

  std::cerr << "Tiling source into " << tx << " x " << ty
            << " tiles (approx "
            << (double)sw/tx << " x " << (double)sh/ty << " px per tile)\n";

  // Precompute tile boundaries using fractional mapping:
  // x0 = floor(c * sw / tx), x1 = floor((c+1)*sw/tx) - 1
  std::vector<int> x0s(tx), x1s(tx), y0s(ty), y1s(ty);
  for (int c = 0; c < tx; ++c) {
    int x0 = (int)std::floor((double)c     * (int)sw / tx);
    int x1 = (int)std::floor((double)(c+1) * (int)sw / tx) - 1;
    if (x1 < x0) x1 = x0;
    x0s[c] = x0;
    x1s[c] = x1;
  }
  for (int r = 0; r < ty; ++r) {
    int y0 = (int)std::floor((double)r     * (int)sh / ty);
    int y1 = (int)std::floor((double)(r+1) * (int)sh / ty) - 1;
    if (y1 < y0) y1 = y0;
    y0s[r] = y0;
    y1s[r] = y1;
  }

  // === 3) Compute per-tile means and overlay grid ===
  std::vector<std::array<double,4>> tile_stats; // mean R,G,B,L per tile
  tile_stats.reserve((size_t)ty * (size_t)tx);

  std::vector<uint8_t> src_grid = src_rgba;

  std::ofstream csv(outdir / (BASE_NAME + "_tiles.csv"));
  csv << "tile_id,row,col,mean_R,mean_G,mean_B,mean_L\n";

  int tile_id = 0;
  for(int r=0; r<ty; ++r){
    for(int c=0; c<tx; ++c){
      int x0 = x0s[c];
      int x1 = x1s[c];
      int y0 = y0s[r];
      int y1 = y1s[r];

      double sumR=0, sumG=0, sumB=0, sumL=0;
      long long cnt=0;
      for(int y=y0; y<=y1; ++y){
        for(int x=x0; x<=x1; ++x){
          const uint8_t* p = &src_rgba[4*(y*sw + x)];
          double R=p[0]/255.0, G=p[1]/255.0, B=p[2]/255.0;
          double L = 0.2126*R + 0.7152*G + 0.0722*B;
          sumR+=R; sumG+=G; sumB+=B; sumL+=L; ++cnt;
        }
      }
      double mR=sumR/cnt, mG=sumG/cnt, mB=sumB/cnt, mL=sumL/cnt;
      tile_stats.push_back({mR,mG,mB,mL});
      csv<<tile_id<<","<<r<<","<<c<<","
         <<std::fixed<<std::setprecision(6)
         <<mR<<","<<mG<<","<<mB<<","<<mL<<"\n";

      if (DRAW_TILE_GRID_ON_SRC){
        draw_rect_outline(src_grid, sw, sh, x0, y0, x1, y1, RGBA8{255,0,0,255});
      }
      ++tile_id;
    }
  }
  csv.close();

  std::string png_src_tiles = (outdir / (BASE_NAME + "_src_with_tiles.png")).string();
  if (write_png_rgba(png_src_tiles, src_grid, sw, sh))
    std::cerr<<"Wrote "<<png_src_tiles<<"\n";
  std::cerr<<"Wrote "<<(outdir / (BASE_NAME + "_tiles.csv")).string()<<"\n";

  // === 4) Legacy equator-row preview (ring_from_tiles) ===
  {
    std::vector<uint8_t> ring_tiles(4ull*IMG_N*IMG_N, 0);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int y=0; y<IMG_N; ++y){
      for(int x=0; x<IMG_N; ++x){
        double X = xmin + (x+0.5)*scale;
        double Y = xmin + (y+0.5)*scale;
        double r = std::sqrt(X*X + Y*Y);
        double theta = std::atan2(Y,X);

        double I_rad = radial_profile(r, RING_RADIUS, RING_SIGMA);

        double u = (theta + M_PI) / (2.0*M_PI);
        int c = std::min(tx-1, std::max(0, (int)std::floor(u*tx)));
        int r_equator = ty/2;
        int idx = r_equator*tx + c;
        auto m = tile_stats[idx];

        double R = std::clamp(I_rad * m[0], 0.0, 1.0);
        double G = std::clamp(I_rad * m[1], 0.0, 1.0);
        double B = std::clamp(I_rad * m[2], 0.0, 1.0);

        size_t i = 4ull*(y*IMG_N + x);
        ring_tiles[i+0] = (uint8_t)std::round(255.0*R);
        ring_tiles[i+1] = (uint8_t)std::round(255.0*G);
        ring_tiles[i+2] = (uint8_t)std::round(255.0*B);
        ring_tiles[i+3] = 255;
      }
    }
    std::string png_ring_tiles = (outdir / (BASE_NAME + "_ring_from_tiles.png")).string();
    if (write_png_rgba(png_ring_tiles, ring_tiles, IMG_N, IMG_N))
      std::cerr<<"Wrote "<<png_ring_tiles<<" (equator row)\n";
  }

  // === 5) Reconstructed mosaic from per-tile means ===
  {
    std::vector<uint8_t> recon(4ull*sw*sh, 255);
    int tile_id2 = 0;
    for(int r=0; r<ty; ++r){
      for(int c=0; c<tx; ++c, ++tile_id2){
        int x0 = x0s[c];
        int x1 = x1s[c];
        int y0 = y0s[r];
        int y1 = y1s[r];
        auto m = tile_stats[tile_id2];
        fill_rect(recon, sw, sh, x0,y0,x1,y1, m[0], m[1], m[2]);
      }
    }
    std::string recon_png = (outdir / (BASE_NAME + "_reconstructed_from_tiles.png")).string();
    if (write_png_rgba(recon_png, recon, sw, sh))
      std::cerr<<"Wrote "<<recon_png<<" (mosaic at "<<tx<<"x"<<ty<<" tile resolution)\n";
  }

  // === 6) One ring per row (uses each row's tile means around the ring) ===
  if (WRITE_ROW_RINGS){
    for(int rr=0; rr<ty; ++rr){
      std::vector<uint8_t> row_ring(4ull*IMG_N*IMG_N, 0);

      #pragma omp parallel for collapse(2) schedule(static)
      for(int y=0; y<IMG_N; ++y){
        for(int x=0; x<IMG_N; ++x){
          double X = xmin + (x+0.5)*scale;
          double Y = xmin + (y+0.5)*scale;
          double r = std::sqrt(X*X + Y*Y);
          double theta = std::atan2(Y,X);

          double I_rad = radial_profile(r, RING_RADIUS, RING_SIGMA);

          double u = (theta + M_PI) / (2.0*M_PI);
          int c = std::min(tx-1, std::max(0, (int)std::floor(u*tx)));
          int idx = rr*tx + c;
          auto m = tile_stats[idx];

          double R = std::clamp(I_rad * m[0], 0.0, 1.0);
          double G = std::clamp(I_rad * m[1], 0.0, 1.0);
          double B = std::clamp(I_rad * m[2], 0.0, 1.0);

          size_t i = 4ull*(y*IMG_N + x);
          row_ring[i+0] = (uint8_t)std::round(255.0*R);
          row_ring[i+1] = (uint8_t)std::round(255.0*G);
          row_ring[i+2] = (uint8_t)std::round(255.0*B);
          row_ring[i+3] = 255;
        }
      }

      char namebuf[64];
      std::snprintf(namebuf, sizeof(namebuf), "%s_ring_from_row_r%02d.png",
                    BASE_NAME.c_str(), rr);
      std::string png_row = (outdir / namebuf).string();
      if (write_png_rgba(png_row, row_ring, IMG_N, IMG_N))
        std::cerr<<"Wrote "<<png_row<<"\n";
    }
  }

  std::cerr<<"\nAll done. Check "<<outdir.string()<<"\n";
  return 0;
}
