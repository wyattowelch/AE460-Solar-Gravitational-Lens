#include <cassert>
#include "sgl/propulsion_module.hpp"
int main(){ sgl::propulsion::PropulsionModel m; m.reset(); auto t=m.step({1.0}); assert(t.power_w>=1.0); assert(t.remaining_propellant_kg<=8.0); if(t.active){ assert(t.thrust_n>0.0); assert(t.burn_event); } return 0; }
