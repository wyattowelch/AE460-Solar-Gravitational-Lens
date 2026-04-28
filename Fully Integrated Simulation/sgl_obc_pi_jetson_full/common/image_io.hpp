#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace sgl { struct ImageRGBA{ unsigned w=0,h=0; std::vector<uint8_t> rgba; }; bool write_ppm(const std::string& path,const ImageRGBA& img); bool read_ppm(const std::string& path,ImageRGBA& img,std::string& err); }
