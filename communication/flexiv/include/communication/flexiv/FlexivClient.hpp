#pragma once

#include <memory>

#include <flexiv/rdk/robot.hpp>

#include "communication/flexiv/BridgeConfig.hpp"
#include "communication/flexiv/BridgeTypes.hpp"

namespace communication::flexiv {

class FlexivClient {
public:
    FlexivClient() = default;

    void Initialize(const BridgeConfig& config);
    void StartRealtime();
    void Stop();
    void StreamCommand(const ArmCommand& command);
    [[nodiscard]] ArmState ReadState() const;

    // Transition into / out of RT_JOINT_TORQUE. Idempotent: a no-op if the
    // robot is already in the requested state. Bridge calls EnsureActiveMode
    // when a fresh command arrives and EnsureIdleMode when commands go stale,
    // so Flexiv's internal logic slows the arm to a stop and holds.
    void EnsureActiveMode();
    void EnsureIdleMode();

private:
    bool mock_mode_ = true;
    bool enable_gravity_comp_ = true;
    bool initialized_ = false;
    bool realtime_started_ = false;
    bool is_active_mode_ = false;
    std::size_t num_arm_joints_ = 0;
    ArmState last_state_;
    std::unique_ptr<::flexiv::rdk::Robot> robot_;
};

}  // namespace communication::flexiv
