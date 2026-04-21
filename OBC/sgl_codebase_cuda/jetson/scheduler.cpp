#include "scheduler.hpp"
#include <cmath>

void Scheduler::tick(int step){
  double target = pm_.target_W();
  double avail  = pm_.available_W();
  double draw   = pm_.draw_W();
  // Requested power models (rough placeholders):
  // lowres CPU+GPU ~ 6W, refine pass ~ 10W, highres batch ~ 18W (these are knobs, not measurements)
  double req_low = 6.0;
  double req_ref = 10.0;
  double req_hi  = 18.0;

  if(step==0){
    log_.log(sgl::LogLevel::INFO, "Telemetry: avail=%.1fW draw=%.1fW target=%.1fW cap=%.1fW",
             avail, draw, target, pm_.cap_W());
  }

  if(step < 20){
    // Always do early low-res reconstruction for quick-look verification
    if(pm_.can_run(req_low)){
      log_.log(sgl::LogLevel::INFO, "[%03d] Mode=LOWRES quick-look (req=%.1fW target=%.1fW)", step, req_low, target);
    } else {
      log_.log(sgl::LogLevel::WARN, "[%03d] Mode=IDLE (insufficient power for lowres) target=%.1fW", step, target);
    }
    return;
  }

  // After quick-look, perform contrast-aware refinement passes as power allows
  if(refine_done_ < cfg_.refine_steps){
    if(pm_.can_run(req_ref)){
      refine_done_++;
      log_.log(sgl::LogLevel::INFO, "[%03d] Mode=REFINE pass %d/%d (req=%.1fW target=%.1fW)",
               step, refine_done_, cfg_.refine_steps, req_ref, target);
    } else {
      log_.log(sgl::LogLevel::DEBUG, "[%03d] Mode=ACCUMULATE only (power-limited) target=%.1fW", step, target);
    }
    return;
  }

  // If refinement complete and power available, do occasional high-res updates
  if((step % 25)==0 && pm_.can_run(req_hi)){
    log_.log(sgl::LogLevel::INFO, "[%03d] Mode=HIGHRES update (req=%.1fW target=%.1fW)", step, req_hi, target);
  } else {
    log_.log(sgl::LogLevel::DEBUG, "[%03d] Mode=MAINTAIN (monitor + accumulate)", step);
  }
}
