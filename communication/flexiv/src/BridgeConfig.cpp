#include "communication/flexiv/BridgeConfig.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace communication::flexiv {

namespace {

template <typename T>
std::vector<T> ReadVectorOrDefault(
    const YAML::Node& node, const char* key, const std::vector<T>& fallback)
{
    if (!node[key]) {
        return fallback;
    }
    return node[key].as<std::vector<T>>();
}

}  // namespace

BridgeConfig BridgeConfig::LoadFromFile(const std::string& path)
{
    const YAML::Node root = YAML::LoadFile(path);

    BridgeConfig cfg;
    cfg.name = root["name"] ? root["name"].as<std::string>() : cfg.name;

    if (root["robot"]) {
        const auto robot = root["robot"];
        cfg.robot.robot_ip =
            robot["robot_ip"] ? robot["robot_ip"].as<std::string>() : cfg.robot.robot_ip;
        cfg.robot.robot_sn =
            robot["robot_sn"] ? robot["robot_sn"].as<std::string>() : cfg.robot.robot_sn;
        cfg.robot.mock_mode =
            robot["mock_mode"] ? robot["mock_mode"].as<bool>() : cfg.robot.mock_mode;
        cfg.robot.enable_gravity_comp =
            robot["enable_gravity_comp"]
                ? robot["enable_gravity_comp"].as<bool>()
                : cfg.robot.enable_gravity_comp;
        cfg.robot.control_mode = ParseControlMode(
            robot["control_mode"]
                ? robot["control_mode"].as<std::string>()
                : ToString(cfg.robot.control_mode));
    }

    cfg.loop.loop_hz =
        root["loop_hz"] ? root["loop_hz"].as<int>() : cfg.loop.loop_hz;
    cfg.loop.command_timeout_s = root["command_timeout_s"]
        ? root["command_timeout_s"].as<double>()
        : cfg.loop.command_timeout_s;

    if (root["lcm"]) {
        const auto lcm = root["lcm"];
        cfg.lcm.ctrl_channel = lcm["ctrl_channel"]
            ? lcm["ctrl_channel"].as<std::string>()
            : cfg.lcm.ctrl_channel;
        cfg.lcm.joint_meas_channel = lcm["joint_meas_channel"]
            ? lcm["joint_meas_channel"].as<std::string>()
            : cfg.lcm.joint_meas_channel;
    }

    if (root["mapping"]) {
        const auto mapping = root["mapping"];
        cfg.mapping.ctrl_joint_offset = mapping["ctrl_joint_offset"]
            ? mapping["ctrl_joint_offset"].as<std::size_t>()
            : cfg.mapping.ctrl_joint_offset;
        cfg.mapping.num_arm_joints = mapping["num_arm_joints"]
            ? mapping["num_arm_joints"].as<std::size_t>()
            : cfg.mapping.num_arm_joints;
        cfg.mapping.joint_names = ReadVectorOrDefault<std::string>(
            mapping, "joint_names", cfg.mapping.joint_names);
        cfg.mapping.command_sign = ReadVectorOrDefault<int>(
            mapping, "command_sign", std::vector<int>(cfg.mapping.num_arm_joints, 1));
        cfg.mapping.measurement_sign = ReadVectorOrDefault<int>(
            mapping,
            "measurement_sign",
            std::vector<int>(cfg.mapping.num_arm_joints, 1));
    } else {
        cfg.mapping.command_sign.assign(cfg.mapping.num_arm_joints, 1);
        cfg.mapping.measurement_sign.assign(cfg.mapping.num_arm_joints, 1);
    }

    cfg.safety.default_hold_q = root["default_hold_q"]
        ? root["default_hold_q"].as<std::vector<double>>()
        : std::vector<double>(cfg.mapping.num_arm_joints, 0.0);
    cfg.safety.hold_kp = root["hold_kp"]
        ? root["hold_kp"].as<std::vector<double>>()
        : std::vector<double>(cfg.mapping.num_arm_joints, 0.0);
    cfg.safety.hold_kd = root["hold_kd"]
        ? root["hold_kd"].as<std::vector<double>>()
        : std::vector<double>(cfg.mapping.num_arm_joints, 0.0);

    if (cfg.safety.default_hold_q.size() != cfg.mapping.num_arm_joints) {
        throw std::runtime_error("default_hold_q must have num_arm_joints entries.");
    }
    if (cfg.safety.hold_kp.size() != cfg.mapping.num_arm_joints) {
        throw std::runtime_error("hold_kp must have num_arm_joints entries.");
    }
    if (cfg.safety.hold_kd.size() != cfg.mapping.num_arm_joints) {
        throw std::runtime_error("hold_kd must have num_arm_joints entries.");
    }

    return cfg;
}

}  // namespace communication::flexiv
