#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "communication/flexiv/BridgeTypes.hpp"

namespace communication::flexiv {

namespace detail {

// LCM packs the on-wire fingerprint as the base type hash rotated left by 1 bit
// (see _get_hash_recursive in the generated bindings). The base hashes here
// are 0xA32CB1E10D48F17E (joint_ctrl_t) and 0xE2C832643D5E41E7 (joint_meas_t).
constexpr std::uint64_t kJointCtrlFingerprint = 0x465963C21A91E2FDULL;
constexpr std::uint64_t kJointMeasFingerprint = 0xC59064C87ABC83CFULL;

inline std::uint64_t ReadU64Be(const std::uint8_t* data)
{
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | static_cast<std::uint64_t>(data[i]);
    }
    return value;
}

inline std::int32_t ReadI32Be(const std::uint8_t* data)
{
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value = (value << 8) | static_cast<std::uint32_t>(data[i]);
    }
    return static_cast<std::int32_t>(value);
}

inline double ReadDoubleBe(const std::uint8_t* data)
{
    const std::uint64_t bits = ReadU64Be(data);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(double));
    return value;
}

inline float ReadFloatBe(const std::uint8_t* data)
{
    const std::uint32_t bits =
        (static_cast<std::uint32_t>(data[0]) << 24) |
        (static_cast<std::uint32_t>(data[1]) << 16) |
        (static_cast<std::uint32_t>(data[2]) << 8) |
        static_cast<std::uint32_t>(data[3]);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

inline void WriteU64Be(std::vector<std::uint8_t>& out, std::uint64_t value)
{
    for (int i = 7; i >= 0; --i) {
        out.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xFFU));
    }
}

inline void WriteI32Be(std::vector<std::uint8_t>& out, std::int32_t value)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    out.push_back(static_cast<std::uint8_t>((bits >> 24) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((bits >> 16) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((bits >> 8) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>(bits & 0xFFU));
}

inline void WriteDoubleBe(std::vector<std::uint8_t>& out, double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(double));
    WriteU64Be(out, bits);
}

inline void WriteFloatBe(std::vector<std::uint8_t>& out, float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    out.push_back(static_cast<std::uint8_t>((bits >> 24) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((bits >> 16) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((bits >> 8) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>(bits & 0xFFU));
}

inline std::vector<double> ReadFloatVector(
    const std::uint8_t* data, std::size_t count)
{
    std::vector<double> values(count, 0.0);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<double>(ReadFloatBe(data + (i * 4)));
    }
    return values;
}

inline void WriteFloatVector(
    std::vector<std::uint8_t>& out, const std::vector<double>& values)
{
    for (double value : values) {
        WriteFloatBe(out, static_cast<float>(value));
    }
}

}  // namespace detail

class JointCtrlLcmCodec {
public:
    static bool Decode(
        const void* bytes, std::size_t size, ArmCommand& command)
    {
        if (bytes == nullptr || size < 21) {
            return false;
        }

        const auto* data = static_cast<const std::uint8_t*>(bytes);
        if (detail::ReadU64Be(data) != detail::kJointCtrlFingerprint) {
            return false;
        }

        const double timestamp = detail::ReadDoubleBe(data + 8);
        const auto num_joints = static_cast<std::size_t>(
            detail::ReadI32Be(data + 16));
        const std::size_t expected_size = 8 + 12 + (5 * num_joints * 4) + 1;
        if (size < expected_size) {
            return false;
        }

        std::size_t offset = 20;
        command.timestamp = timestamp;
        command.q_des = detail::ReadFloatVector(data + offset, num_joints);
        offset += num_joints * 4;
        command.qd_des = detail::ReadFloatVector(data + offset, num_joints);
        offset += num_joints * 4;
        command.tau_ff = detail::ReadFloatVector(data + offset, num_joints);
        offset += num_joints * 4;
        command.kp = detail::ReadFloatVector(data + offset, num_joints);
        offset += num_joints * 4;
        command.kd = detail::ReadFloatVector(data + offset, num_joints);
        offset += num_joints * 4;
        command.valid = data[offset] != 0U;
        return true;
    }
};

class JointMeasLcmCodec {
public:
    static std::vector<std::uint8_t> Encode(const ArmState& state)
    {
        const std::size_t num_joints = state.q.size();
        if (state.qd.size() != num_joints || state.tau.size() != num_joints) {
            throw std::runtime_error(
                "ArmState vectors must have equal lengths before encoding.");
        }

        std::vector<std::uint8_t> out;
        out.reserve(8 + 12 + (3 * num_joints * 4));

        detail::WriteU64Be(out, detail::kJointMeasFingerprint);
        detail::WriteDoubleBe(out, state.timestamp);
        detail::WriteI32Be(out, static_cast<std::int32_t>(num_joints));
        detail::WriteFloatVector(out, state.q);
        detail::WriteFloatVector(out, state.qd);
        detail::WriteFloatVector(out, state.tau);
        return out;
    }
};

}  // namespace communication::flexiv
