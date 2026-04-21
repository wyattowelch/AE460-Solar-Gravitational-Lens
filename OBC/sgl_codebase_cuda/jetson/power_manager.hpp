#pragma once
#include <atomic>
#include <string>

// PowerManager: in flight, reads telemetry from Arduino (I2C/UART/CAN).
// Here, it optionally reads a JSON file out/power_telemetry.json for testing.
class PowerManager {
  double cap_W_;
  double nominal_frac_;
  std::atomic<double> avail_W_{25.0};
  std::atomic<double> draw_W_{0.0};
public:
  PowerManager(double cap_W, double nominal_frac);
  void poll(); // update avail/draw from telemetry source
  double cap_W() const { return cap_W_; }
  double nominal_fraction() const { return nominal_frac_; }
  double available_W() const { return avail_W_.load(); }
  double draw_W() const { return draw_W_.load(); }
  double target_W() const { return std::min(cap_W_, available_W()*nominal_frac_); }
  bool   can_run(double requested_W) const { return requested_W <= target_W(); }
};
