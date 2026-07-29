#pragma once

#include <mutex>
#include <optional>

namespace communication::flexiv {

template <typename T>
class LatestCommandBuffer {
public:
    void Set(const T& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = value;
    }

    [[nodiscard]] std::optional<T> Get() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_;
    }

private:
    mutable std::mutex mutex_;
    std::optional<T> latest_;
};

}  // namespace communication::flexiv
