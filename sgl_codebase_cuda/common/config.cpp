#include "config.hpp"
#include <fstream>
#include <sstream>
#include <cctype>

namespace sgl {

static bool parse_number(const std::string& s, const std::string& key, double& out){
  auto pos = s.find("\""+key+"\"");
  if(pos==std::string::npos) return false;
  pos = s.find(":", pos);
  if(pos==std::string::npos) return false;
  pos++;
  while(pos<s.size() && std::isspace((unsigned char)s[pos])) pos++;
  size_t end=pos;
  while(end<s.size() && (std::isdigit((unsigned char)s[end]) || s[end]=='.' || s[end]=='-' || s[end]=='e' || s[end]=='E' || s[end]=='+')) end++;
  out = std::stod(s.substr(pos, end-pos));
  return true;
}
static bool parse_int(const std::string& s, const std::string& key, int& out){
  double d=0; if(!parse_number(s,key,d)) return false;
  out = (int)d; return true;
}
static bool parse_bool(const std::string& s, const std::string& key, bool& out){
  auto pos = s.find("\""+key+"\"");
  if(pos==std::string::npos) return false;
  pos = s.find(":", pos);
  if(pos==std::string::npos) return false;
  pos++;
  while(pos<s.size() && std::isspace((unsigned char)s[pos])) pos++;
  if(s.compare(pos,4,"true")==0){ out=true; return true; }
  if(s.compare(pos,5,"false")==0){ out=false; return true; }
  return false;
}
static bool parse_string(const std::string& s, const std::string& key, std::string& out){
  auto pos = s.find("\""+key+"\"");
  if(pos==std::string::npos) return false;
  pos = s.find(":", pos);
  if(pos==std::string::npos) return false;
  pos++;
  while(pos<s.size() && std::isspace((unsigned char)s[pos])) pos++;
  if(pos>=s.size() || s[pos]!='"') return false;
  pos++;
  size_t end = s.find("\"", pos);
  if(end==std::string::npos) return false;
  out = s.substr(pos, end-pos);
  return true;
}

bool load_config_json(const std::string& path, Config& C, std::string& err){
  std::ifstream f(path);
  if(!f){ err="Could not open config: "+path; return false; }
  std::ostringstream ss; ss<<f.rdbuf();
  std::string s=ss.str();

  parse_number(s,"power_cap_W", C.power_cap_W);
  parse_number(s,"nominal_fraction", C.nominal_fraction);
  parse_int(s,"lowres_N", C.lowres_N);
  parse_int(s,"highres_N", C.highres_N);
  parse_int(s,"refine_steps", C.refine_steps);
  parse_int(s,"tile_px_x", C.tile_px_x);
  parse_int(s,"tile_px_y", C.tile_px_y);
  parse_number(s,"ring_radius", C.ring_radius);
  parse_number(s,"ring_sigma", C.ring_sigma);
  parse_string(s,"source_image", C.source_image);
  parse_string(s,"out_dir", C.out_dir);
  parse_bool(s,"write_ppm", C.write_ppm);
  parse_bool(s,"write_pgm", C.write_pgm);

  return true;
}

} // namespace sgl
