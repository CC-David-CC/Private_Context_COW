#include "tools/vision/single_image_chunk.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        auto index = strata_vision::single_image_chunk_index(
            static_cast<std::size_t>(argc - 1), [&](std::size_t i) {
                return std::string(argv[i + 1]) == "image";
            });
        std::cout << "OK " << index << '\n';
    } catch (const std::exception& error) {
        std::cout << "ERR " << error.what() << '\n';
    }
}
