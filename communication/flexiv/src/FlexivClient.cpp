#include "communication/flexiv/FlexivClient.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <flexiv/rdk/mode.hpp>
#include <flexiv/rdk/robot.hpp>

namespace communication::flexiv {

namespace {

double NowSeconds()
{
    // Wall-clock to match the controller's Python time.time() timestamps,
    // which downstream consumers compare against.
    using clock = std::chrono::system_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now).count();
}

double TimestampToSeconds(const std::pair<int, int>& ts)
{
    return static_cast<double>(ts.first) + static_cast<double>(ts.second) * 1e-9;
}

}  // namespace

void FlexivClient::Initialize(const BridgeConfig& config)
{
    mock_mode_ = config.robot.mock_mode;
    enable_gravity_comp_ = config.robot.enable_gravity_comp;
    num_arm_joints_ = config.mapping.num_arm_joints;
    initialized_ = true;

    last_state_.timestamp = NowSeconds();
    last_state_.q = config.safety.default_hold_q;
    last_state_.qd.assign(num_arm_joints_, 0.0);
    last_state_.tau.assign(num_arm_joints_, 0.0);

    if (mock_mode_) {
        return;
    }

    if (config.robot.robot_sn.empty() || config.robot.robot_sn == "REPLACE_ME") {
        throw std::runtime_error(
            "robot.robot_sn must be set in the bridge config when mock_mode is false.");
    }

    std::cout << "[flexiv_client] Connecting to robot '" << config.robot.robot_sn
              << "' via Flexiv RDK..." << std::endl;
    robot_ = std::make_unique<::flexiv::rdk::Robot>(config.robot.robot_sn);

    if (robot_->fault()) {
        std::cout << "[flexiv_client] Robot in fault state, attempting to clear..."
                  << std::endl;
        if (!robot_->ClearFault()) {
            throw std::runtime_error("Failed to clear fault on Flexiv robot.");
        }
        std::cout << "[flexiv_client] Fault cleared." << std::endl;
    }

    std::cout << "[flexiv_client] Enabling robot..." << std::endl;
    robot_->Enable();
    while (!robot_->operational()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "[flexiv_client] Robot is operational." << std::endl;

    const auto states = robot_->states();
    if (states.q.size() != num_arm_joints_) {
        throw std::runtime_error(
            "Flexiv robot reported " + std::to_string(states.q.size()) +
            " joints but config expects " + std::to_string(num_arm_joints_) + ".");
    }

    // Only RT joint torque is wired up; we compute the full impedance law
    // (kp*(q_des-q) + kd*(qd_des-qd) + tau_ff) client-side from joint_ctrl_t
    // and stream it via StreamJointTorque so kp/kd can change per command.
    if (config.robot.control_mode != FlexivControlMode::kRtJointTorque) {
        throw std::runtime_error(
            "Only rt_joint_torque is implemented for real-mode streaming; "
            "set robot.control_mode to 'rt_joint_torque'.");
    }
    // Park in IDLE until the bridge sees a fresh command. EnsureActiveMode()
    // will switch to RT_JOINT_TORQUE on the first fresh command and Ensure-
    // IdleMode() will switch back when commands go stale.
    std::cout << "[flexiv_client] Switching to IDLE (waiting for commands)..."
              << std::endl;
    robot_->SwitchMode(::flexiv::rdk::Mode::IDLE);
    is_active_mode_ = false;
    std::cout << "[flexiv_client] Mode = IDLE." << std::endl;
}

void FlexivClient::EnsureActiveMode()
{
    if (mock_mode_ || robot_ == nullptr || is_active_mode_) {
        is_active_mode_ = true;
        return;
    }
    std::cout << "[flexiv_client] Switching to RT_JOINT_TORQUE." << std::endl;
    robot_->SwitchMode(::flexiv::rdk::Mode::RT_JOINT_TORQUE);
    is_active_mode_ = true;
}

void FlexivClient::EnsureIdleMode()
{
    if (mock_mode_ || robot_ == nullptr || !is_active_mode_) {
        is_active_mode_ = false;
        return;
    }
    std::cout << "[flexiv_client] Switching to IDLE (commands stale)."
              << std::endl;
    robot_->SwitchMode(::flexiv::rdk::Mode::IDLE);
    is_active_mode_ = false;
}

void FlexivClient::StartRealtime()
{
    if (!initialized_) {
        throw std::runtime_error("FlexivClient must be initialized before start.");
    }
    realtime_started_ = true;
}

void FlexivClient::Stop()
{
    realtime_started_ = false;
    if (robot_ != nullptr) {
        robot_->Stop();
    }
}

void FlexivClient::StreamCommand(const ArmCommand& command)
{
    if (!realtime_started_) {
        throw std::runtime_error("FlexivClient realtime loop has not started.");
    }

    if (mock_mode_) {
        last_state_.timestamp = command.timestamp;
        last_state_.q = command.q_des;
        last_state_.qd = command.qd_des;
        last_state_.tau = command.tau_ff;
        return;
    }

    if (robot_ == nullptr) {
        throw std::runtime_error(
            "FlexivClient real-mode StreamCommand called with no robot handle.");
    }

    if (command.q_des.size() != num_arm_joints_ ||
        command.qd_des.size() != num_arm_joints_ ||
        command.tau_ff.size() != num_arm_joints_ ||
        command.kp.size() != num_arm_joints_ ||
        command.kd.size() != num_arm_joints_) {
        throw std::runtime_error(
            "ArmCommand vectors must all have num_arm_joints entries.");
    }

    // Read current joint state for the impedance law. Use link-side q (more
    // accurate position) and motor-side dtheta (less noisy velocity), matching
    // intermediate3_realtime_joint_torque_control.cpp from the RDK examples.
    const auto states = robot_->states();

    std::vector<double> torque(num_arm_joints_);
    for (std::size_t i = 0; i < num_arm_joints_; ++i) {
        torque[i] = command.kp[i] * (command.q_des[i] - states.q[i])
                    + command.kd[i] * (command.qd_des[i] - states.dtheta[i])
                    + command.tau_ff[i];
    }

    // enable_gravity_comp (config: robot.enable_gravity_comp): when true the RDK
    // adds the gravity vector under the hood so a zero command holds the pose.
    // Set false when gravity comp is supplied externally (e.g. Pinocchio in the
    // controller), in which case the raw impedance torque above is streamed as-is
    // and MUST already include any gravity term.
    robot_->StreamJointTorque(torque, enable_gravity_comp_);
}

ArmState FlexivClient::ReadState() const
{
    if (mock_mode_ || robot_ == nullptr) {
        return last_state_;
    }

    const auto states = robot_->states();
    ArmState out;
    out.timestamp = TimestampToSeconds(states.timestamp);
    // The bridge publishes only the arm slice; assume the robot has no external
    // axes prepended (validated against num_arm_joints_ in Initialize()).
    out.q.assign(states.q.begin(), states.q.begin() + num_arm_joints_);
    out.qd.assign(states.dq.begin(), states.dq.begin() + num_arm_joints_);
    out.tau.assign(states.tau.begin(), states.tau.begin() + num_arm_joints_);
    return out;
}

}  // namespace communication::flexiv
