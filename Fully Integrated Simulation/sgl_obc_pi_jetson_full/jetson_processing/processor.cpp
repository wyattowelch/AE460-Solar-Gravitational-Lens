#include "processor.hpp"
#include "../common/reconstruction.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
namespace fs = std::filesystem;
namespace sgl {
namespace {
constexpr double kPi = 3.14159265358979323846;
struct SglObsInfo { double dx=0.0, dy=0.0, phase=0.0; };
struct SglDataset {
  int obs_count=0, A=0, R=0;
  std::vector<SglObsInfo> obs;
  std::vector<uint8_t> annulus;  // obs * A * R * 3
};

static std::map<std::string,std::string> parse_kv_csv(const std::string& s) {
  std::map<std::string,std::string> m;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty()) continue;
    auto p = line.find(',');
    if (p == std::string::npos) continue;
    m[line.substr(0, p)] = line.substr(p + 1);
  }
  return m;
}

static bool load_sgl_dataset_from_descriptor(const std::string& desc, SglDataset& ds, std::string& err) {
  const auto kv = parse_kv_csv(desc);
  auto get_int = [&](const std::string& k, int& out)->bool { auto it=kv.find(k); if(it==kv.end()) return false; out=std::stoi(it->second); return true; };
  std::string annulus_bin, obs_csv;
  auto ia = kv.find("annulus_bin_path"); if (ia == kv.end()) { err = "annulus_bin_path missing"; return false; } annulus_bin = ia->second;
  auto io = kv.find("observations_csv_path"); if (io == kv.end()) { err = "observations_csv_path missing"; return false; } obs_csv = io->second;
  if (!get_int("observation_count", ds.obs_count) || !get_int("ring_angular_samples", ds.A) || !get_int("ring_radial_samples", ds.R)) { err = "descriptor ints missing"; return false; }
  std::ifstream of(obs_csv);
  if (!of) { err = "failed to open observations csv"; return false; }
  ds.obs.clear();
  std::string line;
  std::getline(of, line);
  while (std::getline(of, line)) {
    if (line.empty()) continue;
    std::istringstream is(line);
    std::string t0,t1,t2,t3;
    if (!std::getline(is,t0,',')) continue;
    if (!std::getline(is,t1,',')) continue;
    if (!std::getline(is,t2,',')) continue;
    if (!std::getline(is,t3,',')) continue;
    ds.obs.push_back(SglObsInfo{std::stod(t1), std::stod(t2), std::stod(t3)});
  }
  ds.obs_count = std::min(ds.obs_count, static_cast<int>(ds.obs.size()));
  std::ifstream bf(annulus_bin, std::ios::binary);
  if (!bf) { err = "failed to open annulus bin"; return false; }
  uint32_t hdr[4]{};
  bf.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  if (!bf) { err = "annulus header read failed"; return false; }
  const int file_obs = static_cast<int>(hdr[0]), file_A = static_cast<int>(hdr[1]), file_R = static_cast<int>(hdr[2]), ch = static_cast<int>(hdr[3]);
  if (file_A != ds.A || file_R != ds.R || ch != 3) { err = "annulus header mismatch"; return false; }
  const size_t total = static_cast<size_t>(file_obs) * ds.A * ds.R * 3ull;
  ds.annulus.assign(total, 0);
  bf.read(reinterpret_cast<char*>(ds.annulus.data()), static_cast<std::streamsize>(total));
  if (!bf) { err = "annulus payload read failed"; return false; }
  ds.obs_count = std::min(ds.obs_count, file_obs);
  return true;
}

static std::vector<proto::RegionOfInterest> select_rois_from_confidence(const ImageRGBA& img, const std::vector<float>& conf, int max_rois, const std::vector<proto::RegionOfInterest>& prior_rois, int prior_roi_growth) {
  if (max_rois <= 0 || img.w == 0 || img.h == 0) return {};
  const int gx = 16;
  const int gy = 16;
  const int cw = std::max(1, static_cast<int>(img.w) / gx);
  const int ch = std::max(1, static_cast<int>(img.h) / gy);
  std::vector<double> prior((size_t)gy * gx, 0.0);
  for (const auto& p0 : prior_rois) {
    proto::RegionOfInterest p = p0;
    p.x = std::max(0, p.x - prior_roi_growth);
    p.y = std::max(0, p.y - prior_roi_growth);
    p.w += 2 * prior_roi_growth;
    p.h += 2 * prior_roi_growth;
    p.w = std::max(1, p.w);
    p.h = std::max(1, p.h);
    for (int yy = p.y; yy < std::min(gy, p.y + p.h); ++yy) for (int xx = p.x; xx < std::min(gx, p.x + p.w); ++xx) prior[(size_t)yy * gx + xx] = 1.0;
  }
  std::vector<proto::RegionOfInterest> out;
  out.reserve((size_t)gx * gy);
  for (int gyi = 0; gyi < gy; ++gyi) for (int gxi = 0; gxi < gx; ++gxi) {
    const int x0 = gxi * cw, y0 = gyi * ch;
    const int x1 = std::min(static_cast<int>(img.w), x0 + cw);
    const int y1 = std::min(static_cast<int>(img.h), y0 + ch);
    double sgrad = 0.0, sconf = 0.0, sctr = 0.0;
    int cnt = 0;
    for (int y = y0; y + 1 < y1; ++y) for (int x = x0; x + 1 < x1; ++x) {
      const size_t i = static_cast<size_t>(y) * img.w + static_cast<unsigned>(x);
      const auto lum = [&](size_t k){ return (0.2126 * img.rgba[4*k] + 0.7152 * img.rgba[4*k+1] + 0.0722 * img.rgba[4*k+2]) / 255.0; };
      const double l = lum(i), lx = lum(i + 1), ly = lum(i + img.w);
      sgrad += std::fabs(lx - l) + std::fabs(ly - l);
      sctr += std::fabs(l - 0.5);
      sconf += conf[i];
      cnt++;
    }
    if (cnt == 0) cnt = 1;
    const double grad = sgrad / cnt;
    const double ctr = sctr / cnt;
    const double cavg = sconf / cnt;
    double score = 1.4 * grad + 0.8 * ctr + 1.2 * (1.0 - cavg) + 0.45 * prior[(size_t)gyi * gx + gxi];
    out.push_back(proto::RegionOfInterest{gxi, gyi, 1, 1, score});
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ return a.score > b.score; });
  if ((int)out.size() > max_rois) out.resize((size_t)max_rois);
  return out;
}

static ProcessResult reconstruct_from_sgl_dataset(const SglDataset& ds, unsigned outW, unsigned outH, int observation_count, int angular_stride, int radial_stride, int roi_count, const std::vector<proto::RegionOfInterest>& prior_rois, int prior_roi_growth, bool compute_rois) {
  ProcessResult r;
  const int obs_use = std::max(1, std::min(observation_count, ds.obs_count));
  const int A = std::max(1, ds.A);
  const int R = std::max(1, ds.R);
  const int a_step = std::max(1, angular_stride);
  const int r_step = std::max(1, radial_stride);
  const double cx = 0.5 * (outW - 1);
  const double cy = 0.5 * (outH - 1);
  const double rmax = 0.48 * std::min(outW, outH);
  std::vector<float> acc_r((size_t)outW * outH, 0.0f), acc_g((size_t)outW * outH, 0.0f), acc_b((size_t)outW * outH, 0.0f), wsum((size_t)outW * outH, 0.0f);
  auto add = [&](double x, double y, double vr, double vg, double vb) {
    if (x < 0.0 || y < 0.0 || x > (double)(outW - 1) || y > (double)(outH - 1)) return;
    const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    const int x1 = std::min((int)outW - 1, x0 + 1), y1 = std::min((int)outH - 1, y0 + 1);
    const double wx = x - x0, wy = y - y0;
    const double w00 = (1.0 - wx) * (1.0 - wy), w10 = wx * (1.0 - wy), w01 = (1.0 - wx) * wy, w11 = wx * wy;
    auto accum = [&](int xx, int yy, double wv) {
      const size_t i = (size_t)yy * outW + (unsigned)xx;
      acc_r[i] += (float)(vr * wv);
      acc_g[i] += (float)(vg * wv);
      acc_b[i] += (float)(vb * wv);
      wsum[i] += (float)wv;
    };
    accum(x0, y0, w00); accum(x1, y0, w10); accum(x0, y1, w01); accum(x1, y1, w11);
  };
  const auto t0 = std::chrono::steady_clock::now();
  if (!compute_rois) {
    // High-quality inverse mapping for refined products: sample annulus in polar coordinates
    // for each output pixel, then average across observations.
    ImageRGBA out;
    out.w = outW;
    out.h = outH;
    out.rgba.assign(4ull * outW * outH, 255);
    std::vector<float> conf((size_t)outW * outH, 0.0f);
    auto at = [&](int k, int a, int rr, int c) -> uint8_t {
      const size_t base = (size_t)k * A * R * 3ull;
      const size_t idx = base + ((size_t)a * R + (size_t)rr) * 3ull + (size_t)c;
      return ds.annulus[idx];
    };
#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < (int)outH; ++y) {
      for (int x = 0; x < (int)outW; ++x) {
        double sr = 0.0, sg = 0.0, sb = 0.0, sw = 0.0;
        for (int k = 0; k < obs_use; ++k) {
          const auto& oi = ds.obs[(size_t)k];
          const double dx = x - (cx + oi.dx * outW);
          const double dy = y - (cy + oi.dy * outH);
          const double rr = std::sqrt(dx * dx + dy * dy);
          if (rr > rmax) continue;
          const double rf = (rr / rmax) * R - 0.5;
          const int r0 = std::clamp((int)std::floor(rf), 0, R - 1);
          const int r1 = std::clamp(r0 + 1, 0, R - 1);
          const double wr = std::clamp(rf - std::floor(rf), 0.0, 1.0);

          double th = std::atan2(dy, dx) - oi.phase;
          while (th < 0.0) th += 2.0 * kPi;
          while (th >= 2.0 * kPi) th -= 2.0 * kPi;
          const double af = (th / (2.0 * kPi)) * A - 0.5;
          const int a0_base = (int)std::floor(af);
          const int a0 = ((a0_base % A) + A) % A;
          const int a1 = (a0 + 1) % A;
          const double wa = std::clamp(af - std::floor(af), 0.0, 1.0);

          const double v00r = at(k, a0, r0, 0), v10r = at(k, a1, r0, 0), v01r = at(k, a0, r1, 0), v11r = at(k, a1, r1, 0);
          const double v00g = at(k, a0, r0, 1), v10g = at(k, a1, r0, 1), v01g = at(k, a0, r1, 1), v11g = at(k, a1, r1, 1);
          const double v00b = at(k, a0, r0, 2), v10b = at(k, a1, r0, 2), v01b = at(k, a0, r1, 2), v11b = at(k, a1, r1, 2);

          const double vr0 = v00r * (1.0 - wa) + v10r * wa;
          const double vr1 = v01r * (1.0 - wa) + v11r * wa;
          const double vg0 = v00g * (1.0 - wa) + v10g * wa;
          const double vg1 = v01g * (1.0 - wa) + v11g * wa;
          const double vb0 = v00b * (1.0 - wa) + v10b * wa;
          const double vb1 = v01b * (1.0 - wa) + v11b * wa;

          sr += vr0 * (1.0 - wr) + vr1 * wr;
          sg += vg0 * (1.0 - wr) + vg1 * wr;
          sb += vb0 * (1.0 - wr) + vb1 * wr;
          sw += 1.0;
        }

        const size_t i = (size_t)y * outW + (size_t)x;
        if (sw > 0.0) {
          out.rgba[4 * i + 0] = (uint8_t)std::lround(std::clamp(sr / sw, 0.0, 255.0));
          out.rgba[4 * i + 1] = (uint8_t)std::lround(std::clamp(sg / sw, 0.0, 255.0));
          out.rgba[4 * i + 2] = (uint8_t)std::lround(std::clamp(sb / sw, 0.0, 255.0));
          conf[i] = (float)std::clamp(sw / (double)obs_use, 0.0, 1.0);
        } else {
          out.rgba[4 * i + 0] = 0;
          out.rgba[4 * i + 1] = 0;
          out.rgba[4 * i + 2] = 0;
          conf[i] = 0.0f;
        }
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.reconstruction_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    r.image_ppm = ppm_bytes(out);
    r.success = true;
    return r;
  }

  const int radial_subsamples = (r_step <= 1) ? 6 : ((r_step <= 2) ? 4 : 2);
  for (int k = 0; k < obs_use; ++k) {
    const auto& oi = ds.obs[(size_t)k];
    const size_t base = (size_t)k * A * R * 3ull;
    for (int a = 0; a < A; a += a_step) {
      const double theta = 2.0 * kPi * (a + 0.5) / A + oi.phase;
      for (int rr = 0; rr < R; rr += r_step) {
        const size_t pi = base + ((size_t)a * R + (size_t)rr) * 3ull;
        const double r0 = rmax * (double)rr / R;
        const double r1 = rmax * (double)(rr + 1) / R;
        const double inv_sub = 1.0 / static_cast<double>(radial_subsamples);
        for (int rs = 0; rs < radial_subsamples; ++rs) {
          const double sr = r0 + ((rs + 0.5) * inv_sub) * (r1 - r0);
          const double x = cx + oi.dx * outW + sr * std::cos(theta);
          const double y = cy + oi.dy * outH + sr * std::sin(theta);
          add(x, y,
              ds.annulus[pi + 0] * inv_sub,
              ds.annulus[pi + 1] * inv_sub,
              ds.annulus[pi + 2] * inv_sub);
        }
      }
    }
  }
  ImageRGBA out;
  out.w = outW;
  out.h = outH;
  out.rgba.assign(4ull * outW * outH, 255);
  float wmax = 1e-6f;
  for (float w : wsum) wmax = std::max(wmax, w);
  std::vector<float> conf((size_t)outW * outH, 0.0f);
  for (size_t i = 0; i < (size_t)outW * outH; ++i) {
    const float w = std::max(1e-6f, wsum[i]);
    out.rgba[4 * i + 0] = (uint8_t)std::lround(std::clamp(acc_r[i] / w, 0.0f, 255.0f));
    out.rgba[4 * i + 1] = (uint8_t)std::lround(std::clamp(acc_g[i] / w, 0.0f, 255.0f));
    out.rgba[4 * i + 2] = (uint8_t)std::lround(std::clamp(acc_b[i] / w, 0.0f, 255.0f));
    conf[i] = std::clamp(w / wmax, 0.0f, 1.0f);
  }
  const auto t1 = std::chrono::steady_clock::now();
  r.reconstruction_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
  if (compute_rois) {
    const auto tr0 = std::chrono::steady_clock::now();
    r.rois = select_rois_from_confidence(out, conf, roi_count, prior_rois, prior_roi_growth);
    const auto tr1 = std::chrono::steady_clock::now();
    r.roi_selection_ms = std::chrono::duration<double,std::milli>(tr1 - tr0).count();
  }
  r.image_ppm = ppm_bytes(out);
  r.success = true;
  return r;
}
}  // namespace

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

ProcessResult process_coarse_job(const std::string& dataset_csv,unsigned outW,unsigned outH,int coarse_groups_x,int coarse_groups_y,int roi_count,const std::vector<proto::RegionOfInterest>& prior_rois,int prior_roi_growth,int observation_count,const std::string& scratch_dir,const std::string& backend,bool allow_cpu_fallback){ ProcessResult r; fs::create_directories(scratch_dir); std::string resolved_backend, status_prefix; if(!resolve_backend(backend,allow_cpu_fallback,resolved_backend,status_prefix)){ r.status="backend unavailable"; return r; } if(is_sgl_dataset_descriptor(dataset_csv)){ SglDataset ds; std::string de; if(!load_sgl_dataset_from_descriptor(dataset_csv,ds,de)){ r.status="failed to parse sgl descriptor: "+de; return r; } int astride = (outW <= 256 ? 8 : (outW <= 512 ? 4 : 2)); int rstride = (outW <= 256 ? 4 : (outW <= 512 ? 2 : 1)); r = reconstruct_from_sgl_dataset(ds,outW,outH,observation_count,astride,rstride,roi_count,prior_rois,prior_roi_growth,true); r.status = status_prefix + "coarse complete (" + resolved_backend + "; sgl_annulus)"; return r; } std::vector<TileStat> tiles; int tx=0,ty=0; if(!tiles_from_csv(dataset_csv,tiles,tx,ty)){ r.status="failed to parse dataset csv"; return r; } const std::vector<proto::RegionOfInterest>* prior_ptr = prior_rois.empty() ? nullptr : &prior_rois; const int obs = std::max(1, observation_count); double roi_ms_acc=0.0; auto t0=std::chrono::steady_clock::now(); ImageRGBA img{}; for(int i=0;i<obs;++i){ double roi_ms_i=0.0; img=reconstruct_coarse_from_tiles(tiles,tx,ty,coarse_groups_x,coarse_groups_y,outW,outH,&r.rois,roi_count,prior_ptr,prior_roi_growth,&roi_ms_i); roi_ms_acc += roi_ms_i; } auto t1=std::chrono::steady_clock::now(); r.reconstruction_ms = std::chrono::duration<double,std::milli>(t1-t0).count(); r.roi_selection_ms = roi_ms_acc; r.image_ppm=ppm_bytes(img); r.success=true; r.status=status_prefix+"coarse complete ("+resolved_backend+"; legacy_tiles)"; return r; }
ProcessResult process_refine_job(const std::string& dataset_csv,unsigned outW,unsigned outH,int coarse_groups_x,int coarse_groups_y,const std::vector<proto::RegionOfInterest>& rois,int observation_count,const std::string& scratch_dir,const std::string& backend,bool allow_cpu_fallback){ ProcessResult r; fs::create_directories(scratch_dir); std::string resolved_backend, status_prefix; if(!resolve_backend(backend,allow_cpu_fallback,resolved_backend,status_prefix)){ r.status="backend unavailable"; return r; } if(is_sgl_dataset_descriptor(dataset_csv)){ SglDataset ds; std::string de; if(!load_sgl_dataset_from_descriptor(dataset_csv,ds,de)){ r.status="failed to parse sgl descriptor: "+de; return r; } r = reconstruct_from_sgl_dataset(ds,outW,outH,observation_count,1,1,std::max(1,(int)rois.size()),{},0,false); r.rois = rois; r.status = status_prefix + "refine complete (" + resolved_backend + "; sgl_annulus)"; return r; } std::vector<TileStat> tiles; int tx=0,ty=0; if(!tiles_from_csv(dataset_csv,tiles,tx,ty)){ r.status="failed to parse dataset csv"; return r; } const int obs = std::max(1, observation_count); auto t0=std::chrono::steady_clock::now(); ImageRGBA img{}; for(int i=0;i<obs;++i){ img=refine_from_tiles(tiles,tx,ty,coarse_groups_x,coarse_groups_y,rois,outW,outH); } auto t1=std::chrono::steady_clock::now(); r.reconstruction_ms = std::chrono::duration<double,std::milli>(t1-t0).count(); r.image_ppm=ppm_bytes(img); r.success=true; r.status=status_prefix+"refine complete ("+resolved_backend+"; legacy_tiles)"; return r; }
} // namespace sgl
