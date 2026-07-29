#include <iostream>
#include <string>

#include "communication/flexiv/BridgeConfig.hpp"

int main(int argc, char** argv)
{
    const std::string default_config =
        "configs/flexiv_arm/hardware/default.yaml";
    const std::string config_path = (argc > 1) ? argv[1] : default_config;

    const auto config =
        communication::flexiv::BridgeConfig::LoadFromFile(config_path);

    std::cout << "Loaded Flexiv bridge config:\n";
    std::cout << "  name: " << config.name << "\n";
    std::cout << "  robot_ip: " << config.robot.robot_ip << "\n";
    std::cout << "  mock_mode: " << std::boolalpha << config.robot.mock_mode << "\n";
    std::cout << "  control_mode: "
              << communication::flexiv::ToString(config.robot.control_mode) << "\n";
    std::cout << "  loop_hz: " << config.loop.loop_hz << "\n";
    std::cout << "  ctrl_channel: " << config.lcm.ctrl_channel << "\n";
    std::cout << "  joint_meas_channel: " << config.lcm.joint_meas_channel << "\n";
    std::cout << "  num_arm_joints: " << config.mapping.num_arm_joints << "\n";
    return 0;
}
