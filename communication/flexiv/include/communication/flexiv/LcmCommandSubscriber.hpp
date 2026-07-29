#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

#include <lcm/lcm.h>

#include "communication/flexiv/BridgeTypes.hpp"
#include "communication/flexiv/LatestCommandBuffer.hpp"

namespace communication::flexiv {

class LcmCommandSubscriber {
public:
    LcmCommandSubscriber(lcm_t* lcm, std::string ctrl_channel);
    ~LcmCommandSubscriber();

    bool Subscribe();
    int Pump(int timeout_ms);
    [[nodiscard]] std::optional<ArmCommand> LatestCommand() const;
    [[nodiscard]] std::uint64_t messages_received() const { return messages_received_.load(); }
    [[nodiscard]] std::uint64_t decode_failures() const { return decode_failures_.load(); }

private:
    static void HandleMessage(
        const lcm_recv_buf_t* rbuf, const char* channel, void* user_data);

    lcm_t* lcm_ = nullptr;
    std::string ctrl_channel_;
    lcm_subscription_t* subscription_ = nullptr;
    LatestCommandBuffer<ArmCommand> latest_command_;
    std::atomic<std::uint64_t> messages_received_{0};
    std::atomic<std::uint64_t> decode_failures_{0};
};

}  // namespace communication::flexiv
