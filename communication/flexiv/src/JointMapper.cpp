#include "communication/flexiv/JointMapper.hpp"

#include <stdexcept>

namespace communication::flexiv {

JointMapper::JointMapper(const MappingConfig& config)
    : ctrl_joint_offset_(config.ctrl_joint_offset),
      num_arm_joints_(config.num_arm_joints),
      command_sign_(config.command_sign),
      measurement_sign_(config.measurement_sign)
{
    if (command_sign_.empty()) {
        command_sign_.assign(num_arm_joints_, 1);
    }
    if (measurement_sign_.empty()) {
        measurement_sign_.assign(num_arm_joints_, 1);
    }
    if (command_sign_.size() != num_arm_joints_) {
        throw std::runtime_error("command_sign size must match num_arm_joints.");
    }
    if (measurement_sign_.size() != num_arm_joints_) {
        throw std::runtime_error("measurement_sign size must match num_arm_joints.");
    }
}

bool JointMapper::FullCommandHasArmSlice(const ArmCommand& full_command) const
{
    return full_command.q_des.size() >= (ctrl_joint_offset_ + num_arm_joints_) &&
        full_command.qd_des.size() >= (ctrl_joint_offset_ + num_arm_joints_) &&
        full_command.tau_ff.size() >= (ctrl_joint_offset_ + num_arm_joints_) &&
        full_command.kp.size() >= (ctrl_joint_offset_ + num_arm_joints_) &&
        full_command.kd.size() >= (ctrl_joint_offset_ + num_arm_joints_);
}

ArmCommand JointMapper::SliceCommand(const ArmCommand& full_command) const
{
    if (!FullCommandHasArmSlice(full_command)) {
        throw std::runtime_error(
            "Incoming control message does not contain a full arm slice.");
    }

    ArmCommand arm_command;
    arm_command.timestamp = full_command.timestamp;
    arm_command.valid = full_command.valid;

    const auto begin = static_cast<std::ptrdiff_t>(ctrl_joint_offset_);
    const auto end = static_cast<std::ptrdiff_t>(ctrl_joint_offset_ + num_arm_joints_);

    arm_command.q_des = std::vector<double>(
        full_command.q_des.begin() + begin, full_command.q_des.begin() + end);
    arm_command.qd_des = std::vector<double>(
        full_command.qd_des.begin() + begin, full_command.qd_des.begin() + end);
    arm_command.tau_ff = std::vector<double>(
        full_command.tau_ff.begin() + begin, full_command.tau_ff.begin() + end);
    arm_command.kp = std::vector<double>(
        full_command.kp.begin() + begin, full_command.kp.begin() + end);
    arm_command.kd = std::vector<double>(
        full_command.kd.begin() + begin, full_command.kd.begin() + end);

    arm_command.q_des = ApplySigns(arm_command.q_des, command_sign_);
    arm_command.qd_des = ApplySigns(arm_command.qd_des, command_sign_);
    arm_command.tau_ff = ApplySigns(arm_command.tau_ff, command_sign_);
    return arm_command;
}

ArmState JointMapper::ApplyMeasurementMapping(const ArmState& state) const
{
    ArmState mapped = state;
    mapped.q = ApplySigns(mapped.q, measurement_sign_);
    mapped.qd = ApplySigns(mapped.qd, measurement_sign_);
    mapped.tau = ApplySigns(mapped.tau, measurement_sign_);
    return mapped;
}

std::vector<double> JointMapper::ApplySigns(
    const std::vector<double>& values, const std::vector<int>& signs) const
{
    if (values.size() != signs.size()) {
        throw std::runtime_error("Values/sign arrays must have matching lengths.");
    }

    std::vector<double> result(values.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = values[i] * static_cast<double>(signs[i]);
    }
    return result;
}

}  // namespace communication::flexiv
