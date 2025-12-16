#pragma once
#include <cstdio>
#include <cstdarg>
#include <string>
#include <mutex>
#include <chrono>
#include <ctime>

namespace sgl {

enum class LogLevel { INFO, WARN, ERROR, DEBUG };

inline const char* level_str(LogLevel L) {
  switch(L){
    case LogLevel::INFO: return "INFO";
    case LogLevel::WARN: return "WARN";
    case LogLevel::ERROR:return "ERROR";
    case LogLevel::DEBUG:return "DEBUG";
  }
  return "INFO";
}

class Logger {
  std::mutex m_;
  FILE* fp_{nullptr};
public:
  Logger() = default;
  ~Logger(){ if(fp_) std::fclose(fp_); }

  bool open(const std::string& path){
    std::lock_guard<std::mutex> lk(m_);
    fp_ = std::fopen(path.c_str(), "w");
    return fp_ != nullptr;
  }

  void log(LogLevel L, const char* fmt, ...){
    std::lock_guard<std::mutex> lk(m_);
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char tb[64];
    std::strftime(tb, sizeof(tb), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    std::fprintf(fp_ ? fp_ : stderr, "[%s] [%s] ", tb, level_str(L));
    va_list args; va_start(args, fmt);
    std::vfprintf(fp_ ? fp_ : stderr, fmt, args);
    va_end(args);
    std::fprintf(fp_ ? fp_ : stderr, "\n");
    std::fflush(fp_ ? fp_ : stderr);
  }
};

} // namespace sgl
