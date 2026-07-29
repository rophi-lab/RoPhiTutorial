#include "communication/flexiv/SafetyMonitor.hpp"

namespace communication::flexiv {

SafetyMonitor::SafetyMonitor(const BridgeConfig& config)
    : command_timeout_s_(config.loop.command_timeout_s)
{
}

bool SafetyMonitor::IsCommandFresh(
    const std::optional<ArmCommand>& command, double now_s) const
{
    if (!command.has_value()) {
        return false;
    }
    return (now_s - command->timestamp) <= command_timeout_s_;
}

}  // namespace communication::flexiv
