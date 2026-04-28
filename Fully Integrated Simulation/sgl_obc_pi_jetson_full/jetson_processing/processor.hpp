#pragma once
#include <string>
#include <vector>
#include "../common/protocol.hpp"
namespace sgl { struct ProcessResult { bool success=false; std::string status; std::vector<proto::RegionOfInterest> rois; std::vector<uint8_t> image_ppm; double reconstruction_ms=0.0; double roi_selection_ms=0.0; }; ProcessResult process_coarse_job(const std::string& dataset_csv,unsigned outW,unsigned outH,int coarse_groups_x,int coarse_groups_y,int roi_count,const std::vector<proto::RegionOfInterest>& prior_rois,int prior_roi_growth,int observation_count,const std::string& scratch_dir,const std::string& backend,bool allow_cpu_fallback); ProcessResult process_refine_job(const std::string& dataset_csv,unsigned outW,unsigned outH,int coarse_groups_x,int coarse_groups_y,const std::vector<proto::RegionOfInterest>& rois,int observation_count,const std::string& scratch_dir,const std::string& backend,bool allow_cpu_fallback); }
