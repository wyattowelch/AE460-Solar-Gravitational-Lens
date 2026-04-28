#include <cmath>
#include <filesystem>
#include <iostream>

#include "../common/image_io.hpp"
#include "../common/reconstruction.hpp"
#include "../jetson_processing/processor.hpp"

int main() {
  namespace fs = std::filesystem;
  const fs::path out_dir = fs::path("/tmp") / "sgl_ring_observation_test";
  fs::create_directories(out_dir);

  // Build a synthetic ring-like observation.
  sgl::ImageRGBA ring;
  ring.w = 512;
  ring.h = 512;
  ring.rgba.assign(4ull * ring.w * ring.h, 255);
  const double cx = 300.0;
  const double cy = 240.0;
  const double r0 = 150.0;
  const double sigma = 18.0;
  for (unsigned y = 0; y < ring.h; ++y) {
    for (unsigned x = 0; x < ring.w; ++x) {
      const double dx = static_cast<double>(x) - cx;
      const double dy = static_cast<double>(y) - cy;
      const double r = std::sqrt(dx * dx + dy * dy);
      const double a = std::exp(-((r - r0) * (r - r0)) / (2.0 * sigma * sigma));
      const uint8_t rr = static_cast<uint8_t>(std::lround(30 + 200 * a));
      const uint8_t gg = static_cast<uint8_t>(std::lround(20 + 120 * a));
      const uint8_t bb = static_cast<uint8_t>(std::lround(10 + 70 * a));
      const size_t i = 4ull * (static_cast<size_t>(y) * ring.w + x);
      ring.rgba[i] = rr;
      ring.rgba[i + 1] = gg;
      ring.rgba[i + 2] = bb;
      ring.rgba[i + 3] = 255;
    }
  }

  const fs::path ring_in = out_dir / "ring_input.ppm";
  if (!sgl::write_ppm(ring_in.string(), ring)) {
    std::cerr << "failed to write synthetic ring input\n";
    return 2;
  }

  std::string csv, ring_preview, err;
  unsigned sw = 0, sh = 0;
  const bool ok = sgl::generate_payload_dataset_from_ring_observation(
      ring_in.string(), 64, 64, out_dir.string(), csv, ring_preview, sw, sh, err);
  if (!ok) {
    std::cerr << "ring observation extraction failed: " << err << "\n";
    return 3;
  }
  if (!fs::exists(ring_preview)) {
    std::cerr << "ring preview missing\n";
    return 4;
  }
  if (!fs::exists(out_dir / "ring_detect_overlay.ppm")) {
    std::cerr << "ring detect overlay missing\n";
    return 5;
  }

  auto coarse = sgl::process_coarse_job(csv, 256, 256, 4, 4, 8, {}, 1, 1, (out_dir / "scratch").string(), "cpu", true);
  if (!coarse.success || coarse.image_ppm.empty()) {
    std::cerr << "coarse reconstruction from ring dataset failed\n";
    return 6;
  }
  return 0;
}
