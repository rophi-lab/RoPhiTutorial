#include "communication/flexiv/LcmCommandSubscriber.hpp"

#include <iostream>
#include <utility>

#include "communication/flexiv/LcmCodec.hpp"

namespace communication::flexiv {

LcmCommandSubscriber::LcmCommandSubscriber(lcm_t* lcm, std::string ctrl_channel)
    : lcm_(lcm), ctrl_channel_(std::move(ctrl_channel))
{
}

LcmCommandSubscriber::~LcmCommandSubscriber()
{
    if (subscription_ != nullptr && lcm_ != nullptr) {
        lcm_unsubscribe(lcm_, subscription_);
        subscription_ = nullptr;
    }
}

bool LcmCommandSubscriber::Subscribe()
{
    if (lcm_ == nullptr) {
        return false;
    }

    subscription_ = lcm_subscribe(
        lcm_, ctrl_channel_.c_str(), &LcmCommandSubscriber::HandleMessage, this);
    return subscription_ != nullptr;
}

int LcmCommandSubscriber::Pump(int timeout_ms)
{
    if (lcm_ == nullptr) {
        return -1;
    }
    return lcm_handle_timeout(lcm_, timeout_ms);
}

std::optional<ArmCommand> LcmCommandSubscriber::LatestCommand() const
{
    return latest_command_.Get();
}

void LcmCommandSubscriber::HandleMessage(
    const lcm_recv_buf_t* rbuf, const char* /*channel*/, void* user_data)
{
    auto* self = static_cast<LcmCommandSubscriber*>(user_data);
    if (self == nullptr || rbuf == nullptr || rbuf->data == nullptr) {
        return;
    }

    ArmCommand command;
    if (!JointCtrlLcmCodec::Decode(rbuf->data, rbuf->data_size, command)) {
        const auto fails = ++self->decode_failures_;
        if (fails == 1 || fails % 100 == 0) {
            std::cerr << "[flexiv_bridge] Failed to decode joint_ctrl_t (total="
                      << fails << ", bytes=" << rbuf->data_size << ")." << std::endl;
        }
        return;
    }

    const auto received = ++self->messages_received_;
    if (received == 1) {
        std::cout << "[flexiv_bridge] First control message decoded successfully ("
                  << command.q_des.size() << " joints, ts=" << command.timestamp
                  << ")." << std::endl;
    }
    self->latest_command_.Set(command);
}

}  // namespace communication::flexiv
