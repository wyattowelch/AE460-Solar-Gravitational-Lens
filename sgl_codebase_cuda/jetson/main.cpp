// Jetson-side orchestrator (flight-style): power-aware streaming scheduler (CPU fallback)
// Build: same CMake; run: ./sgl_jetson --config config/config.json
#include <filesystem>
#include <iostream>
#include <thread>
#include <chrono>
#include "../common/logger.hpp"
#include "../common/config.hpp"
#include "power_manager.hpp"
#include "scheduler.hpp"

namespace fs=std::filesystem;
using sgl::Logger; using sgl::LogLevel;

static void usage(){ std::cerr<<"Usage: sgl_jetson --config path/to/config.json\n"; }

int main(int argc, char** argv){
  std::string cfgPath="config/config.json";
  for(int i=1;i<argc;i++){
    std::string a=argv[i];
    if(a=="--config" && i+1<argc) cfgPath=argv[++i];
    if(a=="-h"||a=="--help"){ usage(); return 0; }
  }

  fs::create_directories("out/logs");
  Logger log; log.open("out/logs/sgl_jetson.log");

  sgl::Config C; std::string err;
  if(!sgl::load_config_json(cfgPath, C, err)){
    log.log(LogLevel::ERROR, "Config load failed: %s", err.c_str());
    return 1;
  }

  PowerManager pm(C.power_cap_W, C.nominal_fraction);
  Scheduler sched(C, pm, log);

  log.log(LogLevel::INFO, "Starting power-aware scheduler (cap=%.1fW target=%.0f%% of avail).",
          C.power_cap_W, 100.0*C.nominal_fraction);

  // Main loop (simulated). Real system would receive ring measurements from payload IO.
  for(int step=0; step<200; ++step){
    pm.poll();                // reads telemetry (stubbed)
    sched.tick(step);         // decides workload: low-res vs refine pass, etc.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  log.log(LogLevel::INFO, "Scheduler finished demo run.");
  return 0;
}
