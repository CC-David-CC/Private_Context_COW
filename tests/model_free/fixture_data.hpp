#ifndef COW_MODEL_FREE_FIXTURE_DATA_HPP
#define COW_MODEL_FREE_FIXTURE_DATA_HPP

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COW_FIXTURE_HD __host__ __device__
#else
#define COW_FIXTURE_HD
#endif

namespace cow_fixture {
constexpr std::uint64_t logical_page_bytes = 4096;
constexpr std::uint32_t region_canonical = 0;
constexpr std::uint32_t region_canary = 0xCA;

COW_FIXTURE_HD inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

COW_FIXTURE_HD inline std::uint64_t counter(std::uint64_t seed, std::uint32_t region,
        std::uint64_t layer, std::uint64_t token, std::uint64_t channel,
        std::uint64_t block) {
    std::uint64_t x = mix64(seed ^ 0x9E3779B97F4A7C15ULL);
    x = mix64(x ^ mix64(static_cast<std::uint64_t>(region) + 0xD1B54A32D192ED03ULL));
    x = mix64(x ^ mix64(layer + 0xABC98388FB8FAC03ULL));
    x = mix64(x ^ mix64(token + 0x8CB92BA72F3D8DD7ULL));
    x = mix64(x ^ mix64(channel + 0xDB4F0B9175AE2165ULL));
    return mix64(x ^ mix64(block + 0xBBE0563303A4615FULL));
}

COW_FIXTURE_HD inline std::uint8_t byte_at(std::uint64_t seed, std::uint32_t region,
        std::uint64_t layer, std::uint64_t token, std::uint64_t channel,
        std::uint64_t byte_offset) {
    const std::uint64_t local = byte_offset % logical_page_bytes;
    if (local < 16) {
        const std::uint64_t page = byte_offset / logical_page_bytes;
        if (local < 4) {
            constexpr char magic[4] = {'C', 'F', 'X', '1'};
            return static_cast<std::uint8_t>(magic[local]);
        }
        if (local < 12) return static_cast<std::uint8_t>((page >> (8 * (local - 4))) & 0xff);
        constexpr std::uint8_t tail[4] = {0xA5, 0x5A, 0x3C, 0xC3};
        return tail[local - 12];
    }
    const std::uint64_t word = counter(seed, region, layer, token, channel, byte_offset / 8);
    return static_cast<std::uint8_t>((word >> (8 * (byte_offset % 8))) & 0xff);
}

COW_FIXTURE_HD inline std::uint8_t canary_byte(std::uint64_t seed, std::uint64_t device_id,
        std::uint64_t layer, std::uint64_t token, std::uint64_t channel,
        std::uint64_t byte_offset) {
    const std::uint64_t device_seed = mix64(seed ^ mix64(device_id + 0xA0761D6478BD642FULL));
    return byte_at(device_seed, region_canary, layer, token, channel, byte_offset);
}
COW_FIXTURE_HD inline float float_value(std::uint64_t seed, std::uint32_t region,
        std::uint64_t layer, std::uint64_t token, std::uint64_t channel) {
    const int q = static_cast<int>(counter(seed, region, layer, token, channel, 0) % 257ULL) - 128;
    return static_cast<float>(q) * (1.0f / 16.0f);
}
} // namespace cow_fixture

#undef COW_FIXTURE_HD
#endif
