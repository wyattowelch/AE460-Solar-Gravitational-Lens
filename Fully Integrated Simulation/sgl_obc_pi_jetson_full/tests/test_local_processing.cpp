#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../common/reconstruction.hpp"
#include "../jetson_processing/processor.hpp"

int main() {
  namespace fs = std::filesystem;
  const fs::path repo = fs::path(__FILE__).parent_path().parent_path();
  const fs::path source = repo / "bluemarble.ppm";
  const fs::path out_dir = fs::path("/tmp") / "sgl_local_processing_test";
  fs::create_directories(out_dir);

  std::string csv, ring_preview, err;
  unsigned sw = 0, sh = 0;
  const bool ok = sgl::generate_payload_dataset(source.string(), 64, 64, 256, 0.38, 0.04, out_dir.string(), csv, ring_preview, sw, sh, err);
  if (!ok) {
    std::cerr << "generate_payload_dataset failed: " << err << "\n";
    return 2;
  }

  auto coarse = sgl::process_coarse_job(csv, 256, 256, 4, 4, 8, {}, 1, 1, (out_dir / "scratch").string(), "cpu", true);
  if (!coarse.success || coarse.image_ppm.empty() || coarse.rois.empty()) {
    std::cerr << "local coarse processing failed\n";
    return 3;
  }

  auto refine = sgl::process_refine_job(csv, 256, 256, 4, 4, coarse.rois, 1, (out_dir / "scratch").string(), "cpu", true);
  if (!refine.success || refine.image_ppm.empty()) {
    std::cerr << "local refine processing failed\n";
    return 4;
  }

  return 0;
}
