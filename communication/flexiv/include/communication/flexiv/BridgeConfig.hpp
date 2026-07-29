#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "communication/flexiv/BridgeTypes.hpp"

namespace communication::flexiv {

struct RobotConfig {
    std::string robot_ip = "192.168.2.100";
    // Flexiv RDK identifies robots by serial number (e.g. "Rizon4-062525"),
    // not by IP. Required when mock_mode is false.
    std::string robot_sn = "";
    bool mock_mode = true;
    FlexivControlMode control_mode = FlexivControlMode::kRtJointImpedance;
    // When true, the RDK adds its own gravity compensation under the hood on
    // StreamJointTorque, so a zero command holds the pose. Set false to stream
    // the raw torque only - required when gravity comp is supplied externally
    // (e.g. computed from the robot state via Pinocchio in the controller).
    bool enable_gravity_comp = true;
};

struct LoopConfig {
    int loop_hz = 1000;
    double command_timeout_s = 0.05;
};

struct LcmConfig {
    std::string ctrl_channel = "hw_flexiv_arm_joint_ctrl";
    std::string joint_meas_channel = "hw_flexiv_arm_joint_meas";
};

struct MappingConfig {
    std::size_t ctrl_joint_offset = 0;
    std::size_t num_arm_joints = 7;
    std::vector<std::string> joint_names;
    std::vector<int> command_sign;
    std::vector<int> measurement_sign;
};

struct SafetyConfig {
    std::vector<double> default_hold_q;
    std::vector<double> hold_kp;
    std::vector<double> hold_kd;
};

struct BridgeConfig {
    std::string name = "flexiv_arm";
    RobotConfig robot;
    LoopConfig loop;
    LcmConfig lcm;
    MappingConfig mapping;
    SafetyConfig safety;

    static BridgeConfig LoadFromFile(const std::string& path);
};

}  // namespace communication::flexiv
