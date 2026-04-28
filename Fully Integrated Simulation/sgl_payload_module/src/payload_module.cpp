#include "sgl/payload_module.hpp"

namespace sgl::payload {
void PayloadModel::reset() {
  acquire_countdown_ = 0;
  acquisition_progress_ = 0;
  dataset_counter_ = 0;
}

PayloadTelemetry PayloadModel::step(const PayloadInput& in) {
  const int dec = (in.dt_s >= 1.0) ? static_cast<int>(in.dt_s) : 1;
  if (acquire_countdown_ > 0) acquire_countdown_ -= dec;
  if (acquire_countdown_ < 0) acquire_countdown_ = 0;

  if (acquisition_progress_ <= 0 && acquire_countdown_ <= 0) acquisition_progress_ = 3;

  bool active = (acquisition_progress_ > 0);
  bool dataset_ready = false;
  int acquisition_stage = 0;
  if (active) {
    acquisition_stage = 4 - acquisition_progress_;
    acquisition_progress_ -= dec;
    if (acquisition_progress_ <= 0) {
      acquisition_progress_ = 0;
      active = false;
      dataset_ready = true;
      acquire_countdown_ = 20;
    }
  }

  std::string dataset_id;
  if (dataset_ready) {
    dataset_counter_++;
    dataset_id = "dataset_" + std::to_string(dataset_counter_ - 1);
  }

  PayloadTelemetry t;
  t.active = active;
  t.dataset_ready = dataset_ready;
  t.dataset_counter = dataset_counter_;
  t.dataset_id = dataset_id;
  t.acquisition_stage = acquisition_stage;
  t.synthetic_signal_score = dataset_ready ? 1.0 : 0.0;
  t.power_w = active ? 10.0 : 6.0;
  t.mode = dataset_ready ? "READY" : (active ? "ACQUIRE" : "IDLE");
  return t;
}
}  // namespace sgl::payload
