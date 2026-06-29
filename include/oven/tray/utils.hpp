#pragma once

#include <oven/tray/types.hpp>
#include <oven/utils/assert.hpp>
#include <optional>

namespace oven::detail {
    int64_t compute_numel(const SmallVector& shape);
    SmallVector compute_stride(const SmallVector& shape);
    void compute_coordinate(int64_t index, const SmallVector& stride, SmallVector& coord);
    int64_t compute_index(const SmallVector& coord, const SmallVector& stride);
    std::optional<SmallVector> broadcastable(const SmallVector& shape1, const SmallVector& shape2);
    SmallVector get_broadcasted_stride(const SmallVector& shape, const SmallVector& broadcasted_shape);
}

// helper macro-like functions
oven::SmallVector CHECK_BINARY_BROADCAST(const oven::Tray&, const oven::Tray&, std::string);