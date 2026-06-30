#include <oven/tray/operatorstruct.hpp>

uint8_t TRAY_POSSIBLE_TYPES_BITMASK(std::span<const oven::DType> types) {
    uint8_t ret = 0;
    for (auto a : types) {
        ret |= 1 << static_cast<uint8_t>(a);
    }
    return ret;
}


namespace oven::detail {
std::array<DType, 2> ExpOp::possible_types = std::array<DType, 2>{DType::kFloat32, DType::kFloat64};

}