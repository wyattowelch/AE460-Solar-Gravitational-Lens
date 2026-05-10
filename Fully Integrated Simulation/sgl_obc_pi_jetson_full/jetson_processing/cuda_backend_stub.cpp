#include "cuda_backend.hpp"

namespace sgl {

bool cuda_runtime_available(std::string& reason) {
  reason = "no_cuda_build";
  return false;
}

bool cuda_finalize_accum_to_image(const std::vector<float>&,
                                  const std::vector<float>&,
                                  const std::vector<float>&,
                                  const std::vector<float>&,
                                  float,
                                  unsigned,
                                  unsigned,
                                  std::vector<uint8_t>&,
                                  std::vector<float>&,
                                  std::string& err) {
  err = "cuda backend not built";
  return false;
}

}  // namespace sgl
