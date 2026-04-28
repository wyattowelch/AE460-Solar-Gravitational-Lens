#pragma once
#include "sgl/star_tracker_types.hpp"

namespace sgl {

class AdcsController {
public:
    explicit AdcsController(const AdcsControllerConfig& cfg = {});
    void set_config(const AdcsControllerConfig& cfg);
    Vec3 compute_body_torque(const FusedAttitudeState& state,
                             const AdcsCommand& command) const;

private:
    AdcsControllerConfig cfg_{};
};

} // namespace sgl
