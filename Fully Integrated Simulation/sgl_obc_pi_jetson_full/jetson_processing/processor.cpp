#include "processor.hpp"
#include "../common/reconstruction.hpp"
#include <chrono>
#include <cctype>
#include <filesystem>
namespace fs = std::filesystem;
namespace sgl {
static bool resolve_backend(std::string backend, bool allow_cpu_fallback, std::string& resolved_backend, std::string& status_prefix) {
  for (auto& c : backend) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (backend.empty()) backend = "cpu";
  if (backend == "cpu") {
    resolved_backend = "cpu";
    status_prefix.clear();
    return true;
  }
  if (backend == "cuda") {
#ifdef SGL_ENABLE_CUDA
    resolved_backend = "cuda";
    status_prefix.clear();
    return true;
#else
    if (!allow_cpu_fallback) return false;
    resolved_backend = "cpu";
    status_prefix = "cuda_unavailable_fallback_cpu; ";
    return true;
#endif
  }
  if (!allow_cpu_fallback) return false;
  resolved_backend = "cpu";
  status_prefix = "backend_unknown_fallback_cpu; ";
  return true;
}

ProcessResult process_coarse_job(const std::string& dataset_csv,unsigned outW,unsigned outH,int coarse_groups_x,int coarse_groups_y,int roi_count,const std::vector<proto::RegionOfInterest>& prior_rois,int prior_roi_growth,int observation_count,const std::string& scratch_dir,const std::string& backend,bool allow_cpu_fallback){ ProcessResult r; fs::create_directories(scratch_dir); std::string resolved_backend, status_prefix; if(!resolve_backend(backend,allow_cpu_fallback,resolved_backend,status_prefix)){ r.status="backend unavailable"; return r; } std::vector<TileStat> tiles; int tx=0,ty=0; if(!tiles_from_csv(dataset_csv,tiles,tx,ty)){ r.status="failed to parse dataset csv"; return r; } const std::vector<proto::RegionOfInterest>* prior_ptr = prior_rois.empty() ? nullptr : &prior_rois; const int obs = std::max(1, observation_count); double roi_ms_acc=0.0; auto t0=std::chrono::steady_clock::now(); ImageRGBA img{}; for(int i=0;i<obs;++i){ double roi_ms_i=0.0; img=reconstruct_coarse_from_tiles(tiles,tx,ty,coarse_groups_x,coarse_groups_y,outW,outH,&r.rois,roi_count,prior_ptr,prior_roi_growth,&roi_ms_i); roi_ms_acc += roi_ms_i; } auto t1=std::chrono::steady_clock::now(); r.reconstruction_ms = std::chrono::duration<double,std::milli>(t1-t0).count(); r.roi_selection_ms = roi_ms_acc; r.image_ppm=ppm_bytes(img); r.success=true; r.status=status_prefix+"coarse complete ("+resolved_backend+")"; return r; }
ProcessResult process_refine_job(const std::string& dataset_csv,unsigned outW,unsigned outH,int coarse_groups_x,int coarse_groups_y,const std::vector<proto::RegionOfInterest>& rois,int observation_count,const std::string& scratch_dir,const std::string& backend,bool allow_cpu_fallback){ ProcessResult r; fs::create_directories(scratch_dir); std::string resolved_backend, status_prefix; if(!resolve_backend(backend,allow_cpu_fallback,resolved_backend,status_prefix)){ r.status="backend unavailable"; return r; } std::vector<TileStat> tiles; int tx=0,ty=0; if(!tiles_from_csv(dataset_csv,tiles,tx,ty)){ r.status="failed to parse dataset csv"; return r; } const int obs = std::max(1, observation_count); auto t0=std::chrono::steady_clock::now(); ImageRGBA img{}; for(int i=0;i<obs;++i){ img=refine_from_tiles(tiles,tx,ty,coarse_groups_x,coarse_groups_y,rois,outW,outH); } auto t1=std::chrono::steady_clock::now(); r.reconstruction_ms = std::chrono::duration<double,std::milli>(t1-t0).count(); r.image_ppm=ppm_bytes(img); r.success=true; r.status=status_prefix+"refine complete ("+resolved_backend+")"; return r; }
} // namespace sgl
