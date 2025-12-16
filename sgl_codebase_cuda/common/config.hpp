#pragma once
#include <string>
#include <unordered_map>

namespace sgl {

struct Config {
  // Power policy
  double power_cap_W = 25.0;
  double nominal_fraction = 0.75; // target = available * nominal_fraction
  // Reconstruction policy
  int lowres_N = 256;
  int highres_N = 1024;
  int refine_steps = 4; // how many refinement passes
  // Tiling policy
  int tile_px_x = 64;
  int tile_px_y = 64;
  // Ring model (demo)
  double ring_radius = 0.38;
  double ring_sigma  = 0.04;
  // IO
  std::string source_image = "bluemarble.ppm"; // PPM recommended (convert from PNG/JPEG)
  std::string out_dir = "out";
  bool write_ppm = true;
  bool write_pgm = true;
};

bool load_config_json(const std::string& path, Config& C, std::string& err);

} // namespace sgl
