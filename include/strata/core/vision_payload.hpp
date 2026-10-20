#pragma once
#include "strata/core/branch_sha256.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {
inline bool vision_sha256_valid(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
inline std::string vision_sha256_bytes(const uint8_t* data, size_t size) {
    BranchDigest digest;
    branch_sha256(data, size, digest);
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (uint8_t byte : digest.bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}

// This milestone's file-admission guarantee is POSIX/local-filesystem. No
// symlinks, devices or pipes; size is bounded before allocation and reading.
inline std::vector<uint8_t> read_vision_bytes(const std::string& path, uint64_t maximum) {
#if defined(_WIN32)
    (void)path; (void)maximum;
    throw std::runtime_error("native vision file admission requires the qualified POSIX path");
#else
    struct File {
        int fd;
        ~File() { if (fd >= 0) ::close(fd); }
    } file{::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)};
    struct stat before{}, after{};
    if (file.fd < 0 || ::fstat(file.fd, &before) || !S_ISREG(before.st_mode) ||
        before.st_size < 0 || static_cast<uint64_t>(before.st_size) > maximum ||
        static_cast<uint64_t>(before.st_size) > SIZE_MAX)
        throw std::invalid_argument("unsafe or oversized vision input file");
    std::vector<uint8_t> bytes(static_cast<size_t>(before.st_size));
    size_t used = 0;
    while (used < bytes.size()) {
        const auto got = ::read(file.fd, bytes.data() + used, bytes.size() - used);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) throw std::runtime_error("incomplete vision input read");
        used += static_cast<size_t>(got);
    }
    uint8_t extra;
    ssize_t got;
    do { got = ::read(file.fd, &extra, 1); } while (got < 0 && errno == EINTR);
    if (got != 0 || ::fstat(file.fd, &after) || before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec || before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
        throw std::runtime_error("vision input changed during read");
    return bytes;
#endif
}

struct VisionEmbedding {
    int64_t rows = 0, grid_width = 0, grid_height = 0, width = 0;
    std::vector<float> values;
};
inline VisionEmbedding read_vision_sve(const std::string& path, const std::string& digest,
                                       int64_t width, int64_t maximum_rows) {
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    if (!vision_sha256_valid(digest) || width <= 0 || width > limit ||
        maximum_rows <= 0 || maximum_rows > limit)
        throw std::invalid_argument("invalid pinned vision payload expectations");
    const uint64_t bound = 20 + uint64_t(width) * uint64_t(maximum_rows) * 4;
    const auto bytes = read_vision_bytes(path, bound);
    if (bytes.size() < 20) throw std::invalid_argument("short SVE header");
    auto word = [&](size_t offset) {
        uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= uint32_t(bytes[offset + i]) << (8 * i);
        return value;
    };
    VisionEmbedding image;
    image.rows = std::bit_cast<int32_t>(word(4));
    image.grid_width = std::bit_cast<int32_t>(word(8));
    image.grid_height = std::bit_cast<int32_t>(word(12));
    image.width = std::bit_cast<int32_t>(word(16));
    if (word(0) != 0x31455653 || image.rows <= 0 || image.rows > maximum_rows ||
        image.grid_width <= 0 || image.grid_height <= 0 ||
        image.grid_width * image.grid_height != image.rows || image.width != width ||
        bytes.size() != 20 + uint64_t(image.rows) * uint64_t(width) * 4)
        throw std::invalid_argument("invalid SVE size, grid or model width");
    if (vision_sha256_bytes(bytes.data(), bytes.size()) != digest)
        throw std::invalid_argument("vision payload digest mismatch");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    image.values.resize((bytes.size() - 20) / 4);
    for (size_t i = 0; i < image.values.size(); ++i) {
        image.values[i] = std::bit_cast<float>(word(20 + i * 4));
        if (!std::isfinite(image.values[i])) throw std::invalid_argument("nonfinite vision embedding");
    }
    return image;
}
} // namespace strata::core
