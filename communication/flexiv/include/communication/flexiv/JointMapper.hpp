#pragma once

#include <cstddef>
#include <vector>

#include "communication/flexiv/BridgeConfig.hpp"
#include "communication/flexiv/BridgeTypes.hpp"

namespace communication::flexiv {

class JointMapper {
public:
    explicit JointMapper(const MappingConfig& config);

    [[nodiscard]] bool FullCommandHasArmSlice(const ArmCommand& full_command) const;
    [[nodiscard]] ArmCommand SliceCommand(const ArmCommand& full_command) const;
    [[nodiscard]] ArmState ApplyMeasurementMapping(const ArmState& state) const;

private:
    std::vector<double> ApplySigns(
        const std::vector<double>& values,
        const std::vector<int>& signs) const;

    std::size_t ctrl_joint_offset_ = 0;
    std::size_t num_arm_joints_ = 7;
    std::vector<int> command_sign_;
    std::vector<int> measurement_sign_;
};

}  // namespace communication::flexiv
