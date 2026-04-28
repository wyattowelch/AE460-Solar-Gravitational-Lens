#include <iostream>
#include "sgl/propulsion_module.hpp"
int main(){ sgl::propulsion::PropulsionModel m; m.reset(); for(int i=0;i<12;++i){ auto t=m.step({1.0}); std::cout<<i<<" "<<t.mode<<" active="<<(t.active?1:0)<<" burn="<<(t.burn_event?1:0)<<" thrust_n="<<t.thrust_n<<" power="<<t.power_w<<" propellant_kg="<<t.remaining_propellant_kg<<"\n";} return 0; }
