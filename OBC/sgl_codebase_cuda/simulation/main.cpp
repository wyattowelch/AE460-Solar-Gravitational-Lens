// SGL Simulation: tile -> ring slices -> reconstruct (CPU, OpenMP)
// Outputs PPM/PGM (portable formats). Convert to PNG if needed.
// Build (WSL/Linux):
//   mkdir -p build && cd build && cmake .. && make -j
// Run:
//   ./sgl_sim --config ../config/config.json
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common/logger.hpp"
#include "../common/config.hpp"
#include "../common/image_io.hpp"
#include "../processing/psf.hpp"

namespace fs = std::filesystem;
using sgl::Logger;
using sgl::LogLevel;

static inline double luma01(uint8_t r,uint8_t g,uint8_t b){
  return 0.2126*(r/255.0) + 0.7152*(g/255.0) + 0.0722*(b/255.0);
}

static void draw_grid_overlay(sgl::ImageRGBA& img, int tile_px_x, int tile_px_y){
  unsigned w=img.w,h=img.h;
  for(unsigned y=0;y<h;y++){
    if(tile_px_y>0 && (y % tile_px_y)==0){
      for(unsigned x=0;x<w;x++){
        size_t i=4ull*(y*w+x);
        img.rgba[i+0]=255; img.rgba[i+1]=0; img.rgba[i+2]=0;
      }
    }
  }
  for(unsigned x=0;x<w;x++){
    if(tile_px_x>0 && (x % tile_px_x)==0){
      for(unsigned y=0;y<h;y++){
        size_t i=4ull*(y*w+x);
        img.rgba[i+0]=255; img.rgba[i+1]=0; img.rgba[i+2]=0;
      }
    }
  }
}

struct TileStat { double r,g,b,l; };

static std::vector<TileStat> compute_tiles(const sgl::ImageRGBA& src, int tile_px_x, int tile_px_y,
                                          int& tx, int& ty){
  tx = std::max(1, (int)((src.w + tile_px_x - 1)/tile_px_x));
  ty = std::max(1, (int)((src.h + tile_px_y - 1)/tile_px_y));
  std::vector<TileStat> stats(tx*ty);
  #pragma omp parallel for collapse(2) schedule(static)
  for(int r=0;r<ty;r++){
    for(int c=0;c<tx;c++){
      int x0 = c*tile_px_x;
      int y0 = r*tile_px_y;
      int x1 = std::min((int)src.w, x0+tile_px_x);
      int y1 = std::min((int)src.h, y0+tile_px_y);
      double sr=0,sg=0,sb=0,sl=0; long long cnt=0;
      for(int y=y0;y<y1;y++){
        for(int x=x0;x<x1;x++){
          const uint8_t* p=&src.rgba[4ull*(y*src.w+x)];
          double R=p[0]/255.0, G=p[1]/255.0, B=p[2]/255.0;
          double L=0.2126*R+0.7152*G+0.0722*B;
          sr+=R; sg+=G; sb+=B; sl+=L; cnt++;
        }
      }
      if(cnt==0) cnt=1;
      stats[r*tx+c] = TileStat{sr/cnt, sg/cnt, sb/cnt, sl/cnt};
    }
  }
  return stats;
}

// Render a ring image (RGBA) from tile stats by mapping theta->tile col and y->tile row
static sgl::ImageRGBA render_ring_from_tiles(const std::vector<TileStat>& tiles, int tx, int ty,
                                            int N, double ring_radius, double ring_sigma){
  sgl::ImageRGBA out; out.w=N; out.h=N; out.rgba.assign(4ull*N*N, 255);
  const double xmin=-1.0, xmax=+1.0;
  const double scale=(xmax-xmin)/N;
  #pragma omp parallel for collapse(2) schedule(static)
  for(int y=0;y<N;y++){
    for(int x=0;x<N;x++){
      double X=xmin+(x+0.5)*scale;
      double Y=xmin+(y+0.5)*scale;
      double r = std::sqrt(X*X+Y*Y);
      double theta = std::atan2(Y,X); // [-pi,pi]
      double dr=r-ring_radius;
      double I = std::exp(-(dr*dr)/(2.0*ring_sigma*ring_sigma));
      // map theta->tile column (wrap)
      double u = (theta + M_PI)/(2.0*M_PI); //0..1
      int c = std::clamp((int)std::floor(u*tx), 0, tx-1);
      // map Y -> tile row (0..ty-1) (simple equirectangular mapping)
      double v = (Y - xmin)/(xmax-xmin); // 0..1
      int rr = std::clamp((int)std::floor(v*ty), 0, ty-1);
      const TileStat& t = tiles[rr*tx + c];
      double R = std::clamp(I*t.r,0.0,1.0);
      double G = std::clamp(I*t.g,0.0,1.0);
      double B = std::clamp(I*t.b,0.0,1.0);
      size_t i=4ull*(y*N+x);
      out.rgba[i+0]=(uint8_t)std::lround(255*R);
      out.rgba[i+1]=(uint8_t)std::lround(255*G);
      out.rgba[i+2]=(uint8_t)std::lround(255*B);
      out.rgba[i+3]=255;
    }
  }
  return out;
}

// Reconstruct tiled image from ring samples (toy inverse): invert mapping by painting vertical arcs back into tiles
static sgl::ImageRGBA reconstruct_from_tiles(const std::vector<TileStat>& tiles, int tx, int ty,
                                            unsigned outW, unsigned outH){
  sgl::ImageRGBA rec; rec.w=outW; rec.h=outH; rec.rgba.assign(4ull*outW*outH,255);
  // nearest-neighbor: each tile mean fills its region
  unsigned tileW = (outW + tx - 1)/tx;
  unsigned tileH = (outH + ty - 1)/ty;
  #pragma omp parallel for collapse(2) schedule(static)
  for(int r=0;r<ty;r++){
    for(int c=0;c<tx;c++){
      const TileStat& t = tiles[r*tx+c];
      unsigned x0=c*tileW, y0=r*tileH;
      unsigned x1=std::min(outW, x0+tileW);
      unsigned y1=std::min(outH, y0+tileH);
      for(unsigned y=y0;y<y1;y++){
        for(unsigned x=x0;x<x1;x++){
          size_t i=4ull*(y*outW+x);
          rec.rgba[i+0]=(uint8_t)std::lround(255*std::clamp(t.r,0.0,1.0));
          rec.rgba[i+1]=(uint8_t)std::lround(255*std::clamp(t.g,0.0,1.0));
          rec.rgba[i+2]=(uint8_t)std::lround(255*std::clamp(t.b,0.0,1.0));
          rec.rgba[i+3]=255;
        }
      }
    }
  }
  return rec;
}

static void usage(){
  std::cerr<<"Usage: sgl_sim --config path/to/config.json\n";
}

int main(int argc, char** argv){
  std::string cfgPath="config/config.json";
  for(int i=1;i<argc;i++){
    std::string a=argv[i];
    if(a=="--config" && i+1<argc){ cfgPath=argv[++i]; }
    else if(a=="-h"||a=="--help"){ usage(); return 0; }
  }

  Logger log;
  fs::create_directories("out/logs");
  log.open("out/logs/sgl_sim.log");

  sgl::Config C; std::string err;
  if(!sgl::load_config_json(cfgPath, C, err)){
    log.log(LogLevel::ERROR, "Config load failed: %s", err.c_str());
    return 1;
  }
  log.log(LogLevel::INFO, "Loaded config: source=%s tile_px=%dx%d lowres=%d highres=%d",
          C.source_image.c_str(), C.tile_px_x, C.tile_px_y, C.lowres_N, C.highres_N);

  // Load source (PPM P6)
  sgl::ImageRGBA src; 
  if(!sgl::read_ppm(C.source_image, src, err)){
    log.log(LogLevel::ERROR, "Source read failed: %s", err.c_str());
    log.log(LogLevel::ERROR, "Tip: convert with: python3 scripts/convert_to_ppm.py input.jpg %s", C.source_image.c_str());
    return 1;
  }
  log.log(LogLevel::INFO, "Loaded source image: %ux%u", src.w, src.h);

  fs::create_directories(C.out_dir);

  // Compute tiles
  int tx=0, ty=0;
  auto tiles = compute_tiles(src, C.tile_px_x, C.tile_px_y, tx, ty);
  log.log(LogLevel::INFO, "Computed tiles: tx=%d ty=%d (%d tiles)", tx, ty, tx*ty);

  // Write source grid overlay
  sgl::ImageRGBA srcGrid = src;
  draw_grid_overlay(srcGrid, C.tile_px_x, C.tile_px_y);
  if(C.write_ppm){
    sgl::write_ppm(fs::path(C.out_dir)/"src_with_tiles.ppm", srcGrid);
  }

  // Low-res ring and reconstruction
  auto ringLow = render_ring_from_tiles(tiles, tx, ty, C.lowres_N, C.ring_radius, C.ring_sigma);
  if(C.write_ppm) sgl::write_ppm(fs::path(C.out_dir)/"ring_lowres.ppm", ringLow);

  auto recLow = reconstruct_from_tiles(tiles, tx, ty, src.w, src.h);
  if(C.write_ppm) sgl::write_ppm(fs::path(C.out_dir)/"reconstructed_from_tiles.ppm", recLow);

  // High-res ring
  auto ringHigh = render_ring_from_tiles(tiles, tx, ty, C.highres_N, C.ring_radius, C.ring_sigma);
  if(C.write_ppm) sgl::write_ppm(fs::path(C.out_dir)/"ring_highres.ppm", ringHigh);

  log.log(LogLevel::INFO, "Wrote outputs to %s (PPM/PGM).", C.out_dir.c_str());
  log.log(LogLevel::INFO, "Done.");
  return 0;
}
