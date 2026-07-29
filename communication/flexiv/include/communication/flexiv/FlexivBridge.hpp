#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <lcm/lcm.h>

#include "communication/flexiv/BridgeConfig.hpp"
#include "communication/flexiv/FlexivClient.hpp"
#include "communication/flexiv/JointMapper.hpp"
#include "communication/flexiv/LcmCommandSubscriber.hpp"
#include "communication/flexiv/LcmStatePublisher.hpp"
#include "communication/flexiv/SafetyMonitor.hpp"

namespace communication::flexiv {

class FlexivBridge {
public:
    explicit FlexivBridge(BridgeConfig config);
    ~FlexivBridge();

    void Initialize();
    void Run(const std::atomic<bool>& stop_requested);

private:
    static double NowSeconds();

    BridgeConfig config_;
    FlexivClient client_;
    JointMapper mapper_;
    SafetyMonitor safety_monitor_;

    lcm_t* lcm_ = nullptr;
    std::unique_ptr<LcmCommandSubscriber> command_subscriber_;
    std::unique_ptr<LcmStatePublisher> state_publisher_;
};

}  // namespace communication::flexiv
