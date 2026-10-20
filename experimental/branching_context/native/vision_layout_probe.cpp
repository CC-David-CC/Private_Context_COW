// CPU-only adapter for the independent VC007 layout assertions.
#include "strata/core/vision_layout.hpp"
#include <iostream>

int main(int argc, char** argv) {
    const bool input_digest = argc == 2 && std::string(argv[1]) == "--input-digest";
    if (argc != 1 && !input_digest) return 2;
    int64_t capacity, vocab, width, max_rows, count, images;
    while (std::cin >> capacity >> vocab >> width >> max_rows >> count >> images) {
        if (count < 0 || count > 100000 || images < 0 || images > 64) return 2;
        std::vector<int32_t> ids(static_cast<size_t>(count));
        std::vector<strata::core::VisionPlacement> placements(static_cast<size_t>(images));
        for (auto& token : ids) if (!(std::cin >> token)) return 2;
        for (auto& p : placements)
            if (!(std::cin >> p.start >> p.rows >> p.grid_width >> p.grid_height >> p.width
                          >> p.pad_token >> p.embedding_sha256)) return 2;
        try {
            const auto plan = strata::core::vision_layout(ids, placements, capacity, vocab, width, max_rows);
            if (input_digest) {
                std::cout << "OK " << strata::core::vision_input_digest(ids, placements) << '\n';
                continue;
            }
            std::cout << "OK " << plan.consumed << ' ' << plan.rotary_cursor << ' ' << plan.image_rows
                      << ' ' << strata::core::vision_position_digest(plan);
            for (auto value : plan.positions) std::cout << ' ' << value;
            for (auto row : plan.rows) std::cout << ' ' << row.image << ' ' << row.row;
            std::cout << '\n';
        } catch (const std::exception& error) {
            std::cout << "ERR " << error.what() << '\n';
        }
    }
    return std::cin.eof() ? 0 : 2;
}
