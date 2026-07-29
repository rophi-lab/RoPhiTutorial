#include <atomic>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>

#include "communication/flexiv/BridgeConfig.hpp"
#include "communication/flexiv/FlexivBridge.hpp"

namespace {

std::atomic<bool> g_stop_requested{false};

void HandleSignal(int /*signum*/)
{
    g_stop_requested.store(true);
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string default_config =
        "configs/flexiv_arm/hardware/default.yaml";
    const std::string config_path = (argc > 1) ? argv[1] : default_config;

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    try {
        auto config =
            communication::flexiv::BridgeConfig::LoadFromFile(config_path);
        communication::flexiv::FlexivBridge bridge(std::move(config));
        bridge.Initialize();
        bridge.Run(g_stop_requested);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[flexiv_bridge] Fatal error: " << ex.what() << std::endl;
        return 1;
    }
}
