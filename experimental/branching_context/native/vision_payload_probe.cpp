#include "strata/core/vision_payload.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    try {
        const auto value = strata::core::read_vision_sve(argv[1], argv[2], std::stoll(argv[3]), std::stoll(argv[4]));
        std::cout << "OK " << value.rows << ' ' << value.grid_width << ' ' << value.grid_height
                  << ' ' << value.width << ' ' << value.values.size() << '\n';
    } catch (const std::exception& error) { std::cout << "ERR " << error.what() << '\n'; }
}
