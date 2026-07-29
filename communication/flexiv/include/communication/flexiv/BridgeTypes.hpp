#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace communication::flexiv {

enum class FlexivControlMode {
    kRtJointImpedance,
    kRtJointTorque,
};

inline std::string ToString(FlexivControlMode mode)
{
    switch (mode) {
    case FlexivControlMode::kRtJointImpedance:
        return "rt_joint_impedance";
    case FlexivControlMode::kRtJointTorque:
        return "rt_joint_torque";
    }
    throw std::runtime_error("Unhandled FlexivControlMode.");
}

inline FlexivControlMode ParseControlMode(const std::string& mode)
{
    if (mode == "rt_joint_impedance") {
        return FlexivControlMode::kRtJointImpedance;
    }
    if (mode == "rt_joint_torque") {
        return FlexivControlMode::kRtJointTorque;
    }
    throw std::runtime_error("Unsupported Flexiv control mode: " + mode);
}

struct ArmCommand {
    double timestamp = 0.0;
    std::vector<double> q_des;
    std::vector<double> qd_des;
    std::vector<double> tau_ff;
    std::vector<double> kp;
    std::vector<double> kd;
    bool valid = false;

    [[nodiscard]] std::size_t size() const { return q_des.size(); }
};

struct ArmState {
    double timestamp = 0.0;
    std::vector<double> q;
    std::vector<double> qd;
    std::vector<double> tau;

    [[nodiscard]] std::size_t size() const { return q.size(); }
};

}  // namespace communication::flexiv
