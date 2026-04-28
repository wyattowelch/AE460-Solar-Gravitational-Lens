#include <iostream>
#include "sgl/payload_module.hpp"
int main(){ sgl::payload::PayloadModel m; m.reset(); for(int i=0;i<25;++i){ auto t=m.step({1.0}); std::cout<<i<<" mode="<<t.mode<<" active="<<(t.active?1:0)<<" ready="<<(t.dataset_ready?1:0)<<" id="<<t.dataset_id<<" count="<<t.dataset_counter<<" acq_stage="<<t.acquisition_stage<<" score="<<t.synthetic_signal_score<<" power="<<t.power_w<<"\n";} return 0; }
