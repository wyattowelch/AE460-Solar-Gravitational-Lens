#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sgl {

bool cuda_runtime_available(std::string& reason);

bool cuda_finalize_accum_to_image(const std::vector<float>& acc_r,
                                  const std::vector<float>& acc_g,
                                  const std::vector<float>& acc_b,
                                  const std::vector<float>& wsum,
                                  float wmax,
                                  unsigned outW,
                                  unsigned outH,
                                  std::vector<uint8_t>& out_rgba,
                                  std::vector<float>& out_conf,
                                  std::string& err);

}  // namespace sgl
