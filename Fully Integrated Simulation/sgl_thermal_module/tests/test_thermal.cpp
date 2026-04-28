#include <cassert>
#include "sgl/thermal_module.hpp"
int main(){ sgl::thermal::ThermalModel m; m.reset(16.0); auto t=m.step({1.0}); assert(t.heater_on); assert(t.heater_power_w>0.0); assert(t.power_w>2.0); assert(t.low_temp_warning || t.mode=="HEATING" || t.mode=="LOW_TEMP"); return 0; }
