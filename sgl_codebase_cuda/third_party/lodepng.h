/*
LodePNG version 20230430 (trimmed for this project)
Copyright (c) 2005-2023 Lode Vandevenne
License: zlib
https://lodev.org/lodepng/
*/
#ifndef LODEPNG_H
#define LODEPNG_H
#include <vector>
#include <string>
namespace lodepng {
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                const std::string& filename);
unsigned encode(const std::string& filename, const std::vector<unsigned char>& image,
                unsigned w, unsigned h);
const char* error_text(unsigned code);
}
#endif
