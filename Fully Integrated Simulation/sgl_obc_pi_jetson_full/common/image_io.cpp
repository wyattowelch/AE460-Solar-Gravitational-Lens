#include "image_io.hpp"
#include <cctype>
#include <fstream>
namespace sgl {
bool write_ppm(const std::string& path,const ImageRGBA& img){ std::ofstream f(path,std::ios::binary); if(!f) return false; f<<"P6\n"<<img.w<<" "<<img.h<<"\n255\n"; for(size_t i=0;i<img.rgba.size()/4;i++){ f.put((char)img.rgba[4*i]); f.put((char)img.rgba[4*i+1]); f.put((char)img.rgba[4*i+2]); } return true; }
static void skip_ws_and_comments(std::istream& in){ while(true){ int c=in.peek(); if(c=='#'){ std::string line; std::getline(in,line); continue; } if(std::isspace(c)){ in.get(); continue; } break; } }
bool read_ppm(const std::string& path,ImageRGBA& img,std::string& err){ std::ifstream f(path,std::ios::binary); if(!f){ err="Could not open "+path; return false; } std::string magic; f>>magic; if(magic!="P6"){ err="Expected P6 PPM"; return false; } skip_ws_and_comments(f); unsigned w=0,h=0,maxv=0; f>>w; skip_ws_and_comments(f); f>>h; skip_ws_and_comments(f); f>>maxv; if(maxv!=255){ err="PPM maxval must be 255"; return false; } f.get(); img.w=w; img.h=h; img.rgba.assign(4ull*w*h,255); for(size_t i=0;i<(size_t)w*h;i++){ char r,g,b; if(!f.get(r)||!f.get(g)||!f.get(b)){ err="Unexpected EOF"; return false; } img.rgba[4*i]=(uint8_t)(unsigned char)r; img.rgba[4*i+1]=(uint8_t)(unsigned char)g; img.rgba[4*i+2]=(uint8_t)(unsigned char)b; img.rgba[4*i+3]=255; } return true; }
} // namespace sgl
