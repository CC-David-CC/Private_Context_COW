// Independent CPU streaming oracle. Writes canonical bytes to stdout in tensor order.
#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr std::uint64_t page_size = 4096;
std::uint64_t avalanche(std::uint64_t z) {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
std::uint64_t word_for(std::uint64_t seed, std::uint64_t layer,
                       std::uint64_t token, std::uint64_t channel,
                       std::uint64_t block) {
    std::uint64_t h = avalanche(seed ^ 0x9E3779B97F4A7C15ULL);
    const std::uint64_t values[] = {0, layer, token, channel, block};
    const std::uint64_t salts[] = {0xD1B54A32D192ED03ULL,
        0xABC98388FB8FAC03ULL, 0x8CB92BA72F3D8DD7ULL,
        0xDB4F0B9175AE2165ULL, 0xBBE0563303A4615FULL};
    for (unsigned i = 0; i != 5; ++i)
        h = avalanche(h ^ avalanche(values[i] + salts[i]));
    return h;
}
struct WordCache {
    bool valid = false;
    std::uint64_t seed = 0, layer = 0, token = 0, channel = 0, block = 0, word = 0;
};
std::uint8_t fixture_byte(std::uint64_t seed, std::uint64_t layer,
        std::uint64_t token, std::uint64_t channel, std::uint64_t offset,
        WordCache& cache) {
    const auto in_page = offset % page_size;
    if (in_page < 16) {
        const auto page = offset / page_size;
        if (in_page < 4) return static_cast<std::uint8_t>("CFX1"[in_page]);
        if (in_page < 12) return static_cast<std::uint8_t>(page >> (8 * (in_page - 4)));
        constexpr std::uint8_t sentinel[] = {0xA5, 0x5A, 0x3C, 0xC3};
        return sentinel[in_page - 12];
    }
    const auto block = offset / 8;
    if (!cache.valid || cache.seed != seed || cache.layer != layer
            || cache.token != token || cache.channel != channel || cache.block != block) {
        cache.seed = seed; cache.layer = layer; cache.token = token;
        cache.channel = channel; cache.block = block;
        cache.word = word_for(seed, layer, token, channel, block);
        cache.valid = true;
    }
    return static_cast<std::uint8_t>(cache.word >> (8 * (offset % 8)));
}
std::uint64_t parse(const char* arg) {
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(arg, &end, 0);
    if (errno || !end || *end) throw std::string("invalid unsigned integer: ") + arg;
    return value;
}
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::string("fixture byte extent overflows uint64");
    return a * b;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 9) {
        std::fprintf(stderr, "usage: fixture_cpu SEED LAYERS TOKENS CHANNELS BYTES_PER_CELL [--range START LENGTH]\n");
        return 2;
    }
    try {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        const auto seed = parse(argv[1]);
        const auto layers = parse(argv[2]);
        const auto tokens = parse(argv[3]);
        const auto channels = parse(argv[4]);
        const auto bytes_per_cell = parse(argv[5]);
        const auto layer_span = checked_mul(checked_mul(tokens, channels), bytes_per_cell);
        const auto total_bytes = checked_mul(layers, layer_span);
        std::uint64_t start = 0, length = total_bytes;
        if (argc == 9) {
            if (std::string(argv[6]) != "--range")
                throw std::string("expected --range START LENGTH");
            start = parse(argv[7]); length = parse(argv[8]);
            if (start > total_bytes || length > total_bytes - start)
                throw std::string("requested range exceeds canonical tensor extent");
        }
        std::array<std::uint8_t, 1 << 20> buffer{};
        std::size_t used = 0;
        std::uint64_t global_offset = start;
        std::uint64_t remaining = length;
        WordCache word_cache;
        while (remaining) {
            if (!bytes_per_cell || !channels || !tokens || !layer_span)
                throw std::string("nonempty range has an empty tensor dimension");
            const auto layer = global_offset / layer_span;
            const auto in_layer = global_offset % layer_span;
            const auto cell = in_layer / bytes_per_cell;
            const auto token = cell / channels;
            const auto channel = cell % channels;
            const auto cell_offset = in_layer % bytes_per_cell;
            const auto span = std::min(remaining, bytes_per_cell - cell_offset);
            for (std::uint64_t i = 0; i < span; ++i) {
                buffer[used++] = fixture_byte(seed, layer, token, channel,
                                              global_offset, word_cache);
                ++global_offset;
                if (used == buffer.size()) {
                    if (std::fwrite(buffer.data(), 1, used, stdout) != used) return 1;
                    used = 0;
                }
            }
            remaining -= span;
        }
        if (used && std::fwrite(buffer.data(), 1, used, stdout) != used) return 1;
        return std::fflush(stdout) == 0 ? 0 : 1;
    } catch (const std::string& error) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
}
