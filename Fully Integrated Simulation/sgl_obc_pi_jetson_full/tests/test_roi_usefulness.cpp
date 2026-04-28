#include <cmath>
#include <iostream>
#include <vector>

#include "../common/reconstruction.hpp"

int main() {
  sgl::ImageRGBA img;
  img.w = 512;
  img.h = 512;
  img.rgba.assign(4ull * img.w * img.h, 255);

  // Mostly uniform background.
  for (unsigned y = 0; y < img.h; ++y) {
    for (unsigned x = 0; x < img.w; ++x) {
      const size_t i = 4ull * (static_cast<size_t>(y) * img.w + x);
      img.rgba[i] = 210;
      img.rgba[i + 1] = 210;
      img.rgba[i + 2] = 210;
      img.rgba[i + 3] = 255;
    }
  }

  // High-contrast/checker region in lower-right quadrant.
  const unsigned rx0 = 320, ry0 = 320, rx1 = 480, ry1 = 480;
  for (unsigned y = ry0; y < ry1; ++y) {
    for (unsigned x = rx0; x < rx1; ++x) {
      const bool dark = (((x - rx0) / 8 + (y - ry0) / 8) % 2) == 0;
      const uint8_t v = dark ? 30 : 245;
      const size_t i = 4ull * (static_cast<size_t>(y) * img.w + x);
      img.rgba[i] = v;
      img.rgba[i + 1] = v;
      img.rgba[i + 2] = v;
    }
  }

  int tx = 0, ty = 0;
  auto tiles = sgl::compute_tiles(img, 32, 32, tx, ty);
  std::vector<sgl::proto::RegionOfInterest> rois;
  auto _coarse = sgl::reconstruct_coarse_from_tiles(tiles, tx, ty, 4, 4, 256, 256, &rois, 8, nullptr, 1);
  (void)_coarse;
  if (rois.empty()) {
    std::cerr << "no ROIs selected\n";
    return 2;
  }

  // Compute expected coarse group for contrast patch center.
  const int gx = std::max(1, (tx + 4 - 1) / 4);
  const int gy = std::max(1, (ty + 4 - 1) / 4);
  const unsigned cx = (rx0 + rx1) / 2;
  const unsigned cy = (ry0 + ry1) / 2;
  const int tile_x = std::min(tx - 1, static_cast<int>((static_cast<double>(cx) / img.w) * tx));
  const int tile_y = std::min(ty - 1, static_cast<int>((static_cast<double>(cy) / img.h) * ty));
  const int target_gx = std::min((tx + gx - 1) / gx - 1, tile_x / gx);
  const int target_gy = std::min((ty + gy - 1) / gy - 1, tile_y / gy);

  bool found = false;
  for (const auto& r : rois) {
    if (std::abs(r.x - target_gx) <= 1 && std::abs(r.y - target_gy) <= 1) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::cerr << "high-contrast region not selected by ROI scoring\n";
    return 3;
  }
  return 0;
}

