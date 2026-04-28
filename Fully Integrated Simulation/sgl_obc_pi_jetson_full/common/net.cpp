#include "net.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace sgl::net {
namespace {
bool send_all(int fd, const uint8_t* data, size_t n) {
  size_t sent = 0;
  while (sent < n) {
    const ssize_t rc = ::send(fd, data + sent, n - sent, 0);
    if (rc <= 0) return false;
    sent += static_cast<size_t>(rc);
  }
  return true;
}

bool recv_all(int fd, uint8_t* data, size_t n) {
  size_t got = 0;
  while (got < n) {
    const ssize_t rc = ::recv(fd, data + got, n - got, 0);
    if (rc <= 0) return false;
    got += static_cast<size_t>(rc);
  }
  return true;
}

uint64_t htonll(uint64_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (static_cast<uint64_t>(htonl(static_cast<uint32_t>(v & 0xFFFFFFFFULL))) << 32) |
         htonl(static_cast<uint32_t>(v >> 32));
#else
  return v;
#endif
}

uint64_t ntohll(uint64_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return (static_cast<uint64_t>(ntohl(static_cast<uint32_t>(v & 0xFFFFFFFFULL))) << 32) |
         ntohl(static_cast<uint32_t>(v >> 32));
#else
  return v;
#endif
}

bool set_sock_timeout(int fd, int optname, int timeout_ms) {
  if (fd < 0) return false;
  const int ms = timeout_ms < 0 ? 0 : timeout_ms;
  timeval tv{};
  tv.tv_sec = ms / 1000;
  tv.tv_usec = (ms % 1000) * 1000;
  return ::setsockopt(fd, SOL_SOCKET, optname, &tv, sizeof(tv)) == 0;
}
}  // namespace

TcpSocket::~TcpSocket() { close(); }

TcpSocket::TcpSocket(TcpSocket&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

TcpSocket& TcpSocket::operator=(TcpSocket&& o) noexcept {
  if (this != &o) {
    close();
    fd_ = o.fd_;
    o.fd_ = -1;
  }
  return *this;
}

void TcpSocket::close() {
  if (fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }
}

bool TcpSocket::set_recv_timeout_ms(int timeout_ms) {
  return set_sock_timeout(fd_, SO_RCVTIMEO, timeout_ms);
}

bool TcpSocket::set_send_timeout_ms(int timeout_ms) {
  return set_sock_timeout(fd_, SO_SNDTIMEO, timeout_ms);
}

bool TcpSocket::send_frame(const std::string& header, const std::vector<uint8_t>& payload) {
  if (fd_ < 0) return false;
  const uint64_t h = htonll(static_cast<uint64_t>(header.size()));
  const uint64_t p = htonll(static_cast<uint64_t>(payload.size()));
  if (!send_all(fd_, reinterpret_cast<const uint8_t*>(&h), sizeof(h))) return false;
  if (!send_all(fd_, reinterpret_cast<const uint8_t*>(&p), sizeof(p))) return false;
  if (!header.empty() && !send_all(fd_, reinterpret_cast<const uint8_t*>(header.data()), header.size())) return false;
  if (!payload.empty() && !send_all(fd_, payload.data(), payload.size())) return false;
  return true;
}

bool TcpSocket::recv_frame(std::string& header, std::vector<uint8_t>& payload) {
  header.clear();
  payload.clear();
  if (fd_ < 0) return false;
  uint64_t hn = 0;
  uint64_t pn = 0;
  if (!recv_all(fd_, reinterpret_cast<uint8_t*>(&hn), sizeof(hn))) return false;
  if (!recv_all(fd_, reinterpret_cast<uint8_t*>(&pn), sizeof(pn))) return false;
  const uint64_t h = ntohll(hn);
  const uint64_t p = ntohll(pn);
  header.resize(static_cast<size_t>(h));
  payload.resize(static_cast<size_t>(p));
  if (h && !recv_all(fd_, reinterpret_cast<uint8_t*>(header.data()), static_cast<size_t>(h))) return false;
  if (p && !recv_all(fd_, payload.data(), static_cast<size_t>(p))) return false;
  return true;
}

TcpServer::~TcpServer() { close(); }

bool TcpServer::listen_on(const std::string& bind_addr, uint16_t port, int backlog) {
  close();
  fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) return false;
  int opt = 1;
  ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (bind_addr.empty() || bind_addr == "0.0.0.0")
    a.sin_addr.s_addr = INADDR_ANY;
  else if (::inet_pton(AF_INET, bind_addr.c_str(), &a.sin_addr) != 1)
    return false;
  if (::bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0) return false;
  if (::listen(fd_, backlog) < 0) return false;
  return true;
}

TcpSocket TcpServer::accept_one() {
  sockaddr_in a{};
  socklen_t l = sizeof(a);
  const int cfd = ::accept(fd_, reinterpret_cast<sockaddr*>(&a), &l);
  return TcpSocket(cfd);
}

void TcpServer::close() {
  if (fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }
}

TcpSocket connect_to(const std::string& host, uint16_t port, int timeout_ms) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string p = std::to_string(port);
  if (::getaddrinfo(host.c_str(), p.c_str(), &hints, &res) != 0) return TcpSocket();

  int fd = -1;
  for (auto* r = res; r; r = r->ai_next) {
    fd = ::socket(r->ai_family, r->ai_socktype, r->ai_protocol);
    if (fd < 0) continue;

    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    const int rc = ::connect(fd, r->ai_addr, r->ai_addrlen);
    if (rc == 0) {
      if (flags >= 0) ::fcntl(fd, F_SETFL, flags);
      break;
    }
    if (errno != EINPROGRESS && errno != EWOULDBLOCK) {
      ::close(fd);
      fd = -1;
      continue;
    }

    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(fd, &wfds);
    timeval tv{};
    const int ms = timeout_ms < 0 ? 0 : timeout_ms;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    const int sel = ::select(fd + 1, nullptr, &wfds, nullptr, &tv);
    if (sel > 0 && FD_ISSET(fd, &wfds)) {
      int so_error = 0;
      socklen_t slen = sizeof(so_error);
      if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &slen) == 0 && so_error == 0) {
        if (flags >= 0) ::fcntl(fd, F_SETFL, flags);
        break;
      }
    }

    ::close(fd);
    fd = -1;
  }

  if (res) ::freeaddrinfo(res);
  return TcpSocket(fd);
}
}  // namespace sgl::net
