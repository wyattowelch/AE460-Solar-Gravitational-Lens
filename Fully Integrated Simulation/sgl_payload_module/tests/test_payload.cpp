#include <cassert>
#include "sgl/payload_module.hpp"
int main(){ sgl::payload::PayloadModel m; m.reset(); bool saw=false; bool saw_active=false; for(int i=0;i<30;++i){ auto t=m.step({1.0}); if(t.active) saw_active=true; if(t.dataset_ready){ saw=true; assert(!t.dataset_id.empty()); assert(t.dataset_counter>0); } } assert(saw); assert(saw_active); return 0; }
