#include <iostream>
#include "sgl/thermal_module.hpp"
int main(){ sgl::thermal::ThermalModel m; m.reset(16.0); for(int i=0;i<10;++i){ auto t=m.step({1.0}); std::cout<<i<<" "<<t.mode<<" temp="<<t.temperature_c<<" heater="<<(t.heater_on?1:0)<<" heater_power="<<t.heater_power_w<<" power="<<t.power_w<<" low_warn="<<(t.low_temp_warning?1:0)<<" high_warn="<<(t.high_temp_warning?1:0)<<"\n";} return 0; }
