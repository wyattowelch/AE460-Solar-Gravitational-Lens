#include "power_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

PowerManager::PowerManager(double cap_W, double nominal_frac)
: cap_W_(cap_W), nominal_frac_(nominal_frac) {}

static bool find_num(const std::string& s, const std::string& key, double& out){
  auto pos=s.find("\""+key+"\"");
  if(pos==std::string::npos) return false;
  pos=s.find(":",pos); if(pos==std::string::npos) return false;
  pos++;
  while(pos<s.size() && std::isspace((unsigned char)s[pos])) pos++;
  size_t end=pos;
  while(end<s.size() && (std::isdigit((unsigned char)s[end])||s[end]=='.'||s[end]=='-'||s[end]=='e'||s[end]=='E'||s[end]=='+')) end++;
  out=std::stod(s.substr(pos,end-pos)); return true;
}

void PowerManager::poll(){
  // Stub: read file if present, else keep previous values.
  std::ifstream f("out/power_telemetry.json");
  if(!f) return;
  std::ostringstream ss; ss<<f.rdbuf();
  std::string s=ss.str();
  double avail=available_W(), draw=draw_W();
  find_num(s,"available_W", avail);
  find_num(s,"draw_W", draw);
  avail = std::max(0.0, avail);
  draw  = std::max(0.0, draw);
  avail_W_.store(avail);
  draw_W_.store(draw);
}
