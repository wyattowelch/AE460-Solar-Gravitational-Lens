#pragma once
#include <string>
namespace sgl { class ISubsystem { public: virtual ~ISubsystem()=default; virtual void sense(double)=0; virtual void decide(double)=0; virtual void act(double)=0; virtual double current_power_w() const=0; virtual std::string mode_string() const=0; virtual bool healthy() const=0; }; }
