// Dump Flexiv's own identified gravity model g(q) across a joint-space sweep,
// so it can be compared against the URDF/Pinocchio gravity used by our
// controller. Model.Update(q, dq) accepts ANY configuration, so this needs the
// robot connected but NOT moving - it's a pure model-vs-model comparison with
// no friction or torque-sensor confound.
//
//   Usage: rdk_dynamics_dump <robot_sn> [out.csv]
//
// Writes a CSV with columns j0..j6,g0..g6 (rad, Nm). Compare with:
//   python scripts/compare_rdk_urdf.py --csv out.csv \
//       --urdf assets/scene/flexiv_arm/urdf/Rizon4_arm_only.urdf
//
// IMPORTANT: the Flexiv bridge (or any other RDK client) must NOT be running -
// only one RDK connection to the robot is allowed at a time. This tool never
// switches to a control mode and never streams torque; it only enables the
// robot (joints hold in place) and reads/evaluates the model.

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Eigen>
#include <flexiv/rdk/model.hpp>
#include <flexiv/rdk/robot.hpp>

namespace {

constexpr int kNumJoints = 7;

void WriteRow(std::ostream& os, const std::vector<double>& q, const Eigen::VectorXd& g)
{
    for (int i = 0; i < kNumJoints; ++i) os << q[i] << ",";
    for (int i = 0; i < kNumJoints; ++i) {
        os << g[i];
        os << (i == kNumJoints - 1 ? '\n' : ',');
    }
}

// A representative set of poses to probe. Sweeps joint_2 and joint_4 (the main
// gravity-loaded joints) through their range with the others at zero, so the
// per-pose g(q) traces out the gravity curve for those joints.
std::vector<std::vector<double>> BuildPoses()
{
    std::vector<std::vector<double>> poses;
    const double deg = M_PI / 180.0;
    for (int idx : {1, 3}) {  // joint_2 (index 1), joint_4 (index 3)
        for (int a = -90; a <= 90; a += 15) {
            std::vector<double> q(kNumJoints, 0.0);
            q[idx] = a * deg;
            poses.push_back(q);
        }
    }
    // A couple of combined "reach" poses that load joint_2 through the elbow.
    poses.push_back({0.0, 60 * deg, 0.0, 60 * deg, 0.0, 30 * deg, 0.0});
    poses.push_back({0.0, 90 * deg, 0.0, -45 * deg, 0.0, -30 * deg, 0.0});
    return poses;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot_sn> [out.csv]\n";
        return 1;
    }
    const std::string robot_sn = argv[1];
    const std::string out_path = (argc >= 3) ? argv[2] : "rdk_dynamics.csv";

    try {
        std::cout << "[rdk_dump] Connecting to '" << robot_sn << "'...\n";
        flexiv::rdk::Robot robot(robot_sn);

        if (robot.fault()) {
            std::cout << "[rdk_dump] Clearing fault...\n";
            if (!robot.ClearFault()) {
                std::cerr << "[rdk_dump] Failed to clear fault.\n";
                return 1;
            }
        }
        std::cout << "[rdk_dump] Enabling (joints will hold in place)...\n";
        robot.Enable();
        while (!robot.operational()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::cout << "[rdk_dump] Operational.\n";

        // Flexiv's identified dynamics model. Reload() syncs the parameters
        // (including any tool/payload configured on the robot - make sure the
        // payload is cleared for a bare-arm comparison).
        flexiv::rdk::Model model(robot);
        model.Reload();

        // Report the current real pose comparison too.
        const auto states = robot.states();
        std::vector<double> zeros(kNumJoints, 0.0);
        model.Update(states.q, zeros);
        const Eigen::VectorXd g_now = model.g();
        std::cout << "[rdk_dump] Current pose q (rad): ";
        for (double v : states.q) std::cout << v << " ";
        std::cout << "\n[rdk_dump] Flexiv g(q_current) (Nm): " << g_now.transpose() << "\n";

        std::ofstream csv(out_path);
        csv << "# j0..j6 (rad), g0..g6 (Nm) from Flexiv identified model\n";
        // First row: the current real pose (so you can line it up to the tablet).
        WriteRow(csv, states.q, g_now);

        for (const auto& q : BuildPoses()) {
            model.Update(q, zeros);
            WriteRow(csv, q, model.g());
        }
        csv.close();
        std::cout << "[rdk_dump] Wrote " << out_path
                  << " (compare with scripts/compare_rdk_urdf.py).\n";
    } catch (const std::exception& e) {
        std::cerr << "[rdk_dump] Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
