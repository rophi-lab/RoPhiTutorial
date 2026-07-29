#include "communication/flexiv/FlexivBridge.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace communication::flexiv {

FlexivBridge::FlexivBridge(BridgeConfig config)
    : config_(std::move(config)),
      mapper_(config_.mapping),
      safety_monitor_(config_)
{
}

FlexivBridge::~FlexivBridge()
{
    std::cout << "[flexiv_bridge] Shutting down." << std::endl;
    client_.Stop();
    if (lcm_ != nullptr) {
        lcm_destroy(lcm_);
        lcm_ = nullptr;
    }
}

void FlexivBridge::Initialize()
{
    std::cout << "[flexiv_bridge] Initializing bridge '" << config_.name << "'.\n"
              << "  robot_ip=" << config_.robot.robot_ip
              << "  mock_mode=" << std::boolalpha << config_.robot.mock_mode
              << "  control_mode=" << ToString(config_.robot.control_mode) << "\n"
              << "  loop_hz=" << config_.loop.loop_hz
              << "  command_timeout_s=" << config_.loop.command_timeout_s << "\n"
              << "  ctrl_channel=" << config_.lcm.ctrl_channel
              << "  meas_channel=" << config_.lcm.joint_meas_channel << "\n"
              << "  num_arm_joints=" << config_.mapping.num_arm_joints
              << "  ctrl_joint_offset=" << config_.mapping.ctrl_joint_offset << std::endl;

    lcm_ = lcm_create(nullptr);
    if (lcm_ == nullptr) {
        throw std::runtime_error("Failed to create LCM instance for Flexiv bridge.");
    }
    std::cout << "[flexiv_bridge] LCM context created." << std::endl;

    command_subscriber_ = std::make_unique<LcmCommandSubscriber>(
        lcm_, config_.lcm.ctrl_channel);
    state_publisher_ = std::make_unique<LcmStatePublisher>(
        lcm_, config_.lcm.joint_meas_channel);

    if (!command_subscriber_->Subscribe()) {
        throw std::runtime_error("Failed to subscribe to Flexiv control channel.");
    }
    std::cout << "[flexiv_bridge] Subscribed to '" << config_.lcm.ctrl_channel
              << "'." << std::endl;

    client_.Initialize(config_);
    std::cout << "[flexiv_bridge] FlexivClient initialized ("
              << (config_.robot.mock_mode ? "mock" : "real") << " mode)." << std::endl;

    client_.StartRealtime();
    std::cout << "[flexiv_bridge] Realtime stream started. Ready." << std::endl;
}

void FlexivBridge::Run(const std::atomic<bool>& stop_requested)
{
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::microseconds(1000000 / config_.loop.loop_hz);
    auto next_tick = clock::now();

    std::cout << "[flexiv_bridge] Entering control loop at " << config_.loop.loop_hz
              << " Hz." << std::endl;

    std::uint64_t ticks = 0;
    std::uint64_t publishes = 0;
    std::uint64_t idle_ticks = 0;
    std::uint64_t fresh_ticks = 0;
    auto last_stats = clock::now();
    const auto stats_period = std::chrono::seconds(1);
    double last_cmd_ts_seen = 0.0;

    while (!stop_requested.load()) {
        command_subscriber_->Pump(0);

        std::optional<ArmCommand> arm_command;
        const auto latest_full_command = command_subscriber_->LatestCommand();
        if (latest_full_command.has_value() &&
            mapper_.FullCommandHasArmSlice(*latest_full_command)) {
            arm_command = mapper_.SliceCommand(*latest_full_command);
        }

        const bool fresh = safety_monitor_.IsCommandFresh(arm_command, NowSeconds());
        if (fresh) {
            // Mode switches block, so EnsureActiveMode is a no-op on the
            // steady-state path - only the idle->fresh transition costs.
            client_.EnsureActiveMode();
            client_.StreamCommand(*arm_command);
            ++fresh_ticks;
            last_cmd_ts_seen = arm_command->timestamp;
        } else {
            // No fresh command: put the robot in IDLE so Flexiv's internal
            // logic ramps it to a stop and holds the current pose. We skip
            // streaming entirely while idle.
            client_.EnsureIdleMode();
            ++idle_ticks;
        }

        auto arm_state = client_.ReadState();
        arm_state = mapper_.ApplyMeasurementMapping(arm_state);
        state_publisher_->Publish(arm_state);

        ++ticks;
        ++publishes;

        if (clock::now() - last_stats >= stats_period) {
            last_stats = clock::now();
            std::cout << "[flexiv_bridge] tick=" << ticks
                      << " pub=" << publishes
                      << " cmd_rx=" << command_subscriber_->messages_received()
                      << " decode_fail=" << command_subscriber_->decode_failures()
                      << " fresh=" << fresh_ticks
                      << " idle=" << idle_ticks
                      << " last_cmd_ts=" << last_cmd_ts_seen
                      << std::endl;
            fresh_ticks = 0;
            idle_ticks = 0;
        }

        next_tick += period;
        std::this_thread::sleep_until(next_tick);
    }

    std::cout << "[flexiv_bridge] Stop requested. Exiting control loop after "
              << ticks << " ticks." << std::endl;
}

double FlexivBridge::NowSeconds()
{
    // Wall-clock (Unix epoch) so freshness comparisons line up with the
    // controller's joint_ctrl_t timestamps (Python time.time()). steady_clock
    // here would always look ~1.7e9 s behind and IsCommandFresh would never
    // trip.
    using clock = std::chrono::system_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now).count();
}

}  // namespace communication::flexiv
