#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace sgl::net {
class TcpSocket { public: TcpSocket()=default; explicit TcpSocket(int fd):fd_(fd){} ~TcpSocket(); TcpSocket(const TcpSocket&)=delete; TcpSocket& operator=(const TcpSocket&)=delete; TcpSocket(TcpSocket&&) noexcept; TcpSocket& operator=(TcpSocket&&) noexcept; bool valid() const { return fd_>=0; } bool send_frame(const std::string& header,const std::vector<uint8_t>& payload); bool recv_frame(std::string& header,std::vector<uint8_t>& payload); bool set_recv_timeout_ms(int timeout_ms); bool set_send_timeout_ms(int timeout_ms); void close(); private: int fd_=-1; };
class TcpServer { public: ~TcpServer(); bool listen_on(const std::string& bind_addr,uint16_t port,int backlog=1); TcpSocket accept_one(); void close(); private: int fd_=-1; };
TcpSocket connect_to(const std::string& host,uint16_t port,int timeout_ms=5000);
}
