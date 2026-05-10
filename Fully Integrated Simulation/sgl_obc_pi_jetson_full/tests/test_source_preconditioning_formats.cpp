#include "../common/image_io.hpp"
#include "../common/reconstruction.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
double saturation_of(const uint8_t* p) {
  const double r = p[0] / 255.0;
  const double g = p[1] / 255.0;
  const double b = p[2] / 255.0;
  return std::max({r, g, b}) - std::min({r, g, b});
}

double luma_of(const uint8_t* p) {
  return (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / 255.0;
}

struct BgModel {
  double r = 0.0, g = 0.0, b = 0.0, l = 0.0, sat = 0.0;
};

BgModel border_bg(const sgl::ImageRGBA& img) {
  BgModel bg{};
  const unsigned border = std::max(2u, std::min(img.w, img.h) / 24u);
  double n = 0.0;
  for (unsigned y = 0; y < img.h; ++y) {
    for (unsigned x = 0; x < img.w; ++x) {
      if (x >= border && y >= border && x + border < img.w && y + border < img.h) continue;
      const uint8_t* p = &img.rgba[4ull * (static_cast<size_t>(y) * img.w + x)];
      bg.r += p[0] / 255.0;
      bg.g += p[1] / 255.0;
      bg.b += p[2] / 255.0;
      bg.l += luma_of(p);
      bg.sat += saturation_of(p);
      n += 1.0;
    }
  }
  if (n > 0.0) {
    bg.r /= n; bg.g /= n; bg.b /= n; bg.l /= n; bg.sat /= n;
  }
  return bg;
}

struct SupportStats {
  bool ok = false;
  int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
  double mean_luma = 0.0;
  double mean_sat = 0.0;
};

SupportStats support_stats(const sgl::ImageRGBA& img, int bg_value) {
  SupportStats st{};
  const auto bg = border_bg(img);
  const size_t pix = static_cast<size_t>(img.w) * img.h;
  std::vector<uint8_t> fg(pix, 0u), vis(pix, 0u);
  for (unsigned y = 0; y < img.h; ++y) {
    for (unsigned x = 0; x < img.w; ++x) {
      const size_t i = static_cast<size_t>(y) * img.w + x;
      const uint8_t* p = &img.rgba[4ull * i];
      const double r = p[0] / 255.0, g = p[1] / 255.0, b = p[2] / 255.0;
      const double l = luma_of(p);
      const double sat = saturation_of(p);
      const double cd = std::sqrt((r - bg.r) * (r - bg.r) + (g - bg.g) * (g - bg.g) + (b - bg.b) * (b - bg.b));
      const bool is_bg = (std::abs(static_cast<int>(p[0]) - bg_value) <= 1 &&
                          std::abs(static_cast<int>(p[1]) - bg_value) <= 1 &&
                          std::abs(static_cast<int>(p[2]) - bg_value) <= 1);
      const bool keep = !is_bg &&
                        (((cd > std::max(0.012, bg.sat + 0.003)) && sat > std::max(0.002, bg.sat * 0.08)) ||
                         (std::fabs(l - bg.l) > 0.010));
      fg[i] = keep ? 1u : 0u;
    }
  }

  int x0 = static_cast<int>(img.w), y0 = static_cast<int>(img.h), x1 = -1, y1 = -1;
  size_t best_area = 0;
  for (unsigned y = 0; y < img.h; ++y) {
    for (unsigned x = 0; x < img.w; ++x) {
      const size_t start = static_cast<size_t>(y) * img.w + x;
      if (!fg[start] || vis[start]) continue;
      std::vector<size_t> stack{start};
      vis[start] = 1u;
      int cx0 = static_cast<int>(x), cy0 = static_cast<int>(y), cx1 = static_cast<int>(x), cy1 = static_cast<int>(y);
      size_t area = 0;
      while (!stack.empty()) {
        const size_t id = stack.back();
        stack.pop_back();
        area++;
        const unsigned px = static_cast<unsigned>(id % img.w);
        const unsigned py = static_cast<unsigned>(id / img.w);
        cx0 = std::min(cx0, static_cast<int>(px));
        cy0 = std::min(cy0, static_cast<int>(py));
        cx1 = std::max(cx1, static_cast<int>(px));
        cy1 = std::max(cy1, static_cast<int>(py));
        const int nx[4] = {static_cast<int>(px) - 1, static_cast<int>(px) + 1, static_cast<int>(px), static_cast<int>(px)};
        const int ny[4] = {static_cast<int>(py), static_cast<int>(py), static_cast<int>(py) - 1, static_cast<int>(py) + 1};
        for (int k = 0; k < 4; ++k) {
          if (nx[k] < 0 || ny[k] < 0 || nx[k] >= static_cast<int>(img.w) || ny[k] >= static_cast<int>(img.h)) continue;
          const size_t ni = static_cast<size_t>(ny[k]) * img.w + static_cast<size_t>(nx[k]);
          if (!fg[ni] || vis[ni]) continue;
          vis[ni] = 1u;
          stack.push_back(ni);
        }
      }
      if (area > best_area) {
        best_area = area;
        x0 = cx0; y0 = cy0; x1 = cx1; y1 = cy1;
      }
    }
  }

  double sl = 0.0, ss = 0.0, n = 0.0;
  if (best_area > 0) {
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const size_t i = static_cast<size_t>(y) * img.w + static_cast<size_t>(x);
        if (!fg[i]) continue;
        const uint8_t* p = &img.rgba[4ull * i];
        sl += luma_of(p);
        ss += saturation_of(p);
        n += 1.0;
      }
    }
  }
  if (x1 >= x0 && y1 >= y0 && n > 0.0) {
    st.ok = true;
    st.x0 = x0; st.y0 = y0; st.x1 = x1; st.y1 = y1;
    st.mean_luma = sl / n;
    st.mean_sat = ss / n;
  }
  return st;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: test_source_preconditioning_formats <repo_root>\n";
    return 2;
  }
  const fs::path repo_root = fs::path(argv[1]).lexically_normal();
  struct CaseSpec {
    std::string base;
    std::string kind;
  };
  const std::vector<CaseSpec> cases = {
      {"bluemarble", "disk"},
      {"mars", "disk"},
      {"saturn", "ringed"},
  };
  const std::vector<std::string> exts = {".jpg", ".png", ".ppm"};

  sgl::SourcePreconditioningConfig cfg{};
  cfg.canvas_N = 2048;
  cfg.object_type = "auto";
  cfg.disk_fill_fraction = 0.60;
  cfg.extended_fill_fraction = 0.85;
  cfg.object_padding_fraction = 0.14;
  cfg.minimum_source_margin_fraction = 0.08;
  cfg.background_value = 0;

  const fs::path tmp_root = fs::current_path() / "tmp_test_source_preconditioning_formats";
  std::error_code ec;
  fs::remove_all(tmp_root, ec);
  fs::create_directories(tmp_root, ec);
  if (ec) {
    std::cerr << "failed to create tmp root: " << ec.message() << "\n";
    return 1;
  }

  for (const auto& c : cases) {
    for (const auto& ext : exts) {
      const fs::path src = repo_root.parent_path() / (c.base + ext);
      if (!fs::exists(src)) {
        std::cerr << "missing source asset: " << src << "\n";
        return 1;
      }
      sgl::ImageRGBA img{};
      std::string err;
      if (!sgl::read_image_auto(src.string(), img, err)) {
        std::cerr << "read_image_auto failed for " << src << ": " << err << "\n";
        return 1;
      }
      if (img.w == 0 || img.h == 0 || img.rgba.empty()) {
        std::cerr << "decoded empty image for " << src << "\n";
        return 1;
      }
      const fs::path out_dir = tmp_root / (c.base + "_" + ext.substr(1));
      fs::create_directories(out_dir, ec);
      if (ec) {
        std::cerr << "failed to create out dir " << out_dir << ": " << ec.message() << "\n";
        return 1;
      }

      sgl::SourcePreconditioningResult out{};
      if (!sgl::precondition_source_image(src.string(), out_dir.string(), cfg, out, err)) {
        std::cerr << "precondition_source_image failed for " << src << ": " << err << "\n";
        return 1;
      }
      if (out.preconditioned_source_path.empty() || !fs::exists(out.preconditioned_source_path)) {
        std::cerr << "missing preconditioned source for " << src << "\n";
        return 1;
      }
      sgl::ImageRGBA pre{};
      if (!sgl::read_image_auto(out.preconditioned_source_path, pre, err)) {
        std::cerr << "failed reading preconditioned source " << out.preconditioned_source_path << ": " << err << "\n";
        return 1;
      }
      auto src_stats = support_stats(img, cfg.background_value);
      auto pre_stats = support_stats(pre, cfg.background_value);
      if (!src_stats.ok || !pre_stats.ok) {
        std::cerr << "support stats failed for " << src << "\n";
        return 1;
      }
      if (pre.w != 2048 || pre.h != 2048) {
        std::cerr << "unexpected preconditioned size for " << src << ": " << pre.w << "x" << pre.h << "\n";
        return 1;
      }
      if (c.kind == "ringed") {
        if (out.object_type_detected == "disk_planet") {
          std::cerr << "saturn-like source misclassified as disk_planet for " << src << "\n";
          return 1;
        }
        const double min_margin = cfg.minimum_source_margin_fraction - 0.01;
        if (out.margin_fraction < min_margin) {
          std::cerr << "ringed-object margin too small for " << src << ": " << out.margin_fraction << " expected >= " << min_margin << "\n";
          return 1;
        }
        if (out.clipping_guard_triggered) {
          std::cerr << "ringed-object clipping guard triggered unexpectedly for " << src << "\n";
          return 1;
        }
        const double raw_ar = static_cast<double>(src_stats.x1 - src_stats.x0 + 1) / std::max(1.0, static_cast<double>(src_stats.y1 - src_stats.y0 + 1));
        const double pre_ar = static_cast<double>(pre_stats.x1 - pre_stats.x0 + 1) / std::max(1.0, static_cast<double>(pre_stats.y1 - pre_stats.y0 + 1));
        const double ar_delta = std::fabs(pre_ar - raw_ar) / std::max(0.1, raw_ar);
        if (ar_delta > 0.10) {
          std::cerr << "ringed-object aspect changed too much for " << src << ": raw_ar=" << raw_ar << " pre_ar=" << pre_ar << " delta=" << ar_delta << "\n";
          return 1;
        }
        if (pre_stats.mean_luma > src_stats.mean_luma + 0.12) {
          std::cerr << "ringed-object luma washout for " << src << ": raw=" << src_stats.mean_luma << " pre=" << pre_stats.mean_luma << "\n";
          return 1;
        }
        if (pre_stats.mean_sat < src_stats.mean_sat * 0.70) {
          std::cerr << "ringed-object saturation dropped too much for " << src << ": raw=" << src_stats.mean_sat << " pre=" << pre_stats.mean_sat << "\n";
          return 1;
        }
      } else {
        if (out.object_type_detected != "disk_planet") {
          std::cerr << "disk source not classified as disk_planet for " << src << " got " << out.object_type_detected << "\n";
          return 1;
        }
      }
      if (!fs::exists(out.source_overlay_path)) {
        std::cerr << "missing source overlay for " << src << "\n";
        return 1;
      }
      if (out.margin_fraction < 0.02) {
        std::cerr << "suspiciously low post margin for " << src << ": " << out.margin_fraction << "\n";
        return 1;
      }
    }
  }
  return 0;
}
