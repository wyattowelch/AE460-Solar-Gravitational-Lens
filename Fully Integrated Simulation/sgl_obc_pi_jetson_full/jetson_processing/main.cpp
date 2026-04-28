#include <algorithm>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>

#include "../common/config.hpp"
#include "../common/logger.hpp"
#include "../common/net.hpp"
#include "../common/protocol.hpp"
#include "processor.hpp"

namespace fs = std::filesystem;

namespace {
std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }
}  // namespace

int main(int argc, char** argv) {
  std::string cfgPath = "config/config.json";
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) cfgPath = argv[++i];
  }

  sgl::Config C;
  std::string err;
  if (!sgl::load_config_json(cfgPath, C, err)) {
    std::cerr << err << "\n";
    return 1;
  }

  fs::create_directories(fs::path(C.out_dir) / "logs");
  sgl::Logger log;
  log.open((fs::path(C.out_dir) / "logs" / "sgl_jetson_service.log").string());

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  sgl::net::TcpServer server;
  if (!server.listen_on(C.host, static_cast<uint16_t>(C.port))) {
    log.log(sgl::LogLevel::ERROR, "bind failed on %s:%d", C.host.c_str(), C.port);
    return 1;
  }
  log.log(sgl::LogLevel::INFO, "Jetson service listening on %s:%d backend=%s fallback=%d",
          C.host.c_str(), C.port, C.jetson_backend.c_str(), static_cast<int>(C.jetson_allow_cpu_fallback));

  auto sock = server.accept_one();
  if (!sock.valid()) {
    log.log(sgl::LogLevel::ERROR, "accept failed");
    return 1;
  }
  sock.set_send_timeout_ms(C.job_ack_timeout_ms);

  bool running = true;
  while (running && !g_stop.load()) {
    std::string header_raw;
    std::vector<uint8_t> payload;
    if (!sock.recv_frame(header_raw, payload)) {
      log.log(sgl::LogLevel::WARN, "connection closed");
      break;
    }

    sgl::proto::HeaderMap h;
    sgl::proto::decode_header_block(header_raw, h);
    std::string type_s;
    sgl::proto::get_string(h, "msg_type", type_s);
    auto type = sgl::proto::msg_type_from_string(type_s);

    if (type == sgl::proto::MsgType::Hello) {
      sgl::proto::HeaderMap out{{"msg_type", "HelloAck"}, {"status", "ready"}};
      sock.send_frame(sgl::proto::encode_header_block(out), {});
      continue;
    }
    if (type == sgl::proto::MsgType::Shutdown) {
      sgl::proto::HeaderMap out{{"msg_type", "Status"}, {"status", "shutdown"}};
      sock.send_frame(sgl::proto::encode_header_block(out), {});
      running = false;
      continue;
    }

    std::string job_id = "unknown";
    int outW = C.highres_N;
    int outH = C.highres_N;
    int gx = C.coarse_groups_x;
    int gy = C.coarse_groups_y;
    int roi_count = C.roi_count;
    int prior_roi_growth = std::max(0, C.progressive_roi_growth);
    int observation_count = 1;
    sgl::proto::get_string(h, "job_id", job_id);
    sgl::proto::get_int(h, "out_w", outW);
    sgl::proto::get_int(h, "out_h", outH);
    sgl::proto::get_int(h, "coarse_groups_x", gx);
    sgl::proto::get_int(h, "coarse_groups_y", gy);
    sgl::proto::get_int(h, "roi_count", roi_count);
    sgl::proto::get_int(h, "prior_roi_growth", prior_roi_growth);
    sgl::proto::get_int(h, "observation_count", observation_count);

    sgl::proto::HeaderMap ack{{"msg_type", "JobAccepted"}, {"job_id", job_id}};
    if (!sock.send_frame(sgl::proto::encode_header_block(ack), {})) {
      log.log(sgl::LogLevel::WARN, "job %s ack send failed", job_id.c_str());
      break;
    }

    const std::string dataset_csv(payload.begin(), payload.end());
    sgl::ProcessResult result;
    if (type == sgl::proto::MsgType::ProcessCoarse) {
      std::string prior_rois_s;
      std::vector<sgl::proto::RegionOfInterest> prior_rois;
      if (sgl::proto::get_string(h, "prior_rois", prior_rois_s)) prior_rois = sgl::proto::decode_rois(prior_rois_s);
      result = sgl::process_coarse_job(dataset_csv, static_cast<unsigned>(outW), static_cast<unsigned>(outH),
                                       gx, gy, roi_count, prior_rois, prior_roi_growth, observation_count, C.jetson_scratch_dir, C.jetson_backend,
                                       C.jetson_allow_cpu_fallback);
    } else if (type == sgl::proto::MsgType::RefineRois) {
      std::string rois_s;
      sgl::proto::get_string(h, "rois", rois_s);
      result = sgl::process_refine_job(dataset_csv, static_cast<unsigned>(outW), static_cast<unsigned>(outH),
                                       gx, gy, sgl::proto::decode_rois(rois_s), observation_count, C.jetson_scratch_dir,
                                       C.jetson_backend, C.jetson_allow_cpu_fallback);
    } else {
      result.status = "unsupported";
    }

    sgl::proto::HeaderMap out{
        {"msg_type", result.success ? "JobComplete" : "JobFailed"},
        {"job_id", job_id},
        {"status", result.status},
        {"reconstruction_ms", std::to_string(result.reconstruction_ms)},
        {"roi_selection_ms", std::to_string(result.roi_selection_ms)},
        {"rois", sgl::proto::encode_rois(result.rois)}};
    if (!sock.send_frame(sgl::proto::encode_header_block(out), result.image_ppm)) {
      log.log(sgl::LogLevel::WARN, "job %s result send failed", job_id.c_str());
      break;
    }
    log.log(sgl::LogLevel::INFO, "job %s -> %s", job_id.c_str(), result.status.c_str());
  }

  log.log(sgl::LogLevel::INFO, "Jetson service exiting");
  return 0;
}
