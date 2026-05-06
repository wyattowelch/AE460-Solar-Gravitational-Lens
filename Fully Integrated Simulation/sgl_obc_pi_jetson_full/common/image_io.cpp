#include "image_io.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "third_party/stb_image.h"

namespace sgl {

bool write_ppm(const std::string& path, const ImageRGBA& img) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << "P6\n" << img.w << " " << img.h << "\n255\n";
  for (size_t i = 0; i < img.rgba.size() / 4; i++) {
    f.put(static_cast<char>(img.rgba[4 * i]));
    f.put(static_cast<char>(img.rgba[4 * i + 1]));
    f.put(static_cast<char>(img.rgba[4 * i + 2]));
  }
  return true;
}

static void skip_ws_and_comments(std::istream& in) {
  while (true) {
    int c = in.peek();
    if (c == '#') {
      std::string line;
      std::getline(in, line);
      continue;
    }
    if (std::isspace(c)) {
      in.get();
      continue;
    }
    break;
  }
}

bool read_ppm(const std::string& path, ImageRGBA& img, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "Could not open " + path;
    return false;
  }
  std::string magic;
  f >> magic;
  if (magic != "P6") {
    err = "Expected P6 PPM";
    return false;
  }
  skip_ws_and_comments(f);
  unsigned w = 0, h = 0, maxv = 0;
  f >> w;
  skip_ws_and_comments(f);
  f >> h;
  skip_ws_and_comments(f);
  f >> maxv;
  if (maxv != 255) {
    err = "PPM maxval must be 255";
    return false;
  }
  f.get();
  img.w = w;
  img.h = h;
  img.rgba.assign(4ull * w * h, 255);
  for (size_t i = 0; i < static_cast<size_t>(w) * h; i++) {
    char r, g, b;
    if (!f.get(r) || !f.get(g) || !f.get(b)) {
      err = "Unexpected EOF";
      return false;
    }
    img.rgba[4 * i] = static_cast<uint8_t>(static_cast<unsigned char>(r));
    img.rgba[4 * i + 1] = static_cast<uint8_t>(static_cast<unsigned char>(g));
    img.rgba[4 * i + 2] = static_cast<uint8_t>(static_cast<unsigned char>(b));
    img.rgba[4 * i + 3] = 255;
  }
  return true;
}

static std::string lower_ext(const std::string& path) {
  auto p = path.find_last_of('.');
  if (p == std::string::npos) return "";
  std::string e = path.substr(p);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

bool read_image_auto(const std::string& path, ImageRGBA& img, std::string& err) {
  const std::string ext = lower_ext(path);
  if (ext == ".ppm") {
    return read_ppm(path, img, err);
  }

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "Could not open " + path;
    return false;
  }
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.empty()) {
    err = "Empty image file: " + path;
    return false;
  }
  int w = 0, h = 0, ch = 0;
  unsigned char* decoded = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &ch, 4);
  if (!decoded) {
    const char* reason = stbi_failure_reason();
    err = std::string("Image decode failed for ") + path + (reason ? (": " + std::string(reason)) : "");
    return false;
  }
  img.w = static_cast<unsigned>(w);
  img.h = static_cast<unsigned>(h);
  img.rgba.assign(decoded, decoded + 4ull * img.w * img.h);
  stbi_image_free(decoded);
  return true;
}

}  // namespace sgl
