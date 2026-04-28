#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace sgl::proto {
enum class MsgType : uint16_t { Unknown=0, Hello=1, HelloAck, Ping, Pong, ProcessCoarse, RefineRois, JobAccepted, JobRejected, JobComplete, JobFailed, Status, Shutdown };
struct RegionOfInterest { int x=0,y=0,w=0,h=0; double score=0.0; };
using HeaderMap = std::map<std::string,std::string>;
std::string msg_type_to_string(MsgType t);
MsgType msg_type_from_string(const std::string& s);
std::string encode_header_block(const HeaderMap& h);
bool decode_header_block(const std::string& s, HeaderMap& h);
std::string encode_rois(const std::vector<RegionOfInterest>& rois);
std::vector<RegionOfInterest> decode_rois(const std::string& s);
bool get_string(const HeaderMap& h, const std::string& key, std::string& out);
bool get_int(const HeaderMap& h, const std::string& key, int& out);
}
