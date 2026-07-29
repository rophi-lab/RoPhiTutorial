#pragma once

#include <optional>

#include "communication/flexiv/BridgeConfig.hpp"
#include "communication/flexiv/BridgeTypes.hpp"

namespace communication::flexiv {

class SafetyMonitor {
public:
    explicit SafetyMonitor(const BridgeConfig& config);

    // Returns true iff [command] is present and its timestamp is no older
    // than command_timeout_s relative to [now_s]. The bridge uses this to
    // decide whether to stream commands or to put the robot into IDLE so
    // Flexiv's internal logic slows it to a stop and holds the current pose.
    [[nodiscard]] bool IsCommandFresh(
        const std::optional<ArmCommand>& command,
        double now_s) const;

private:
    double command_timeout_s_ = 0.05;
};

}  // namespace communication::flexiv
