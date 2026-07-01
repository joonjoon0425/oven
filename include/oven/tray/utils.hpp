#pragma once

#include <oven/tray/types.hpp>
#include <oven/utils/assert.hpp>
#include <optional>

namespace oven::detail {
    int64_t compute_numel(const SmallVector& shape);
    SmallVector compute_stride(const SmallVector& shape);
    void compute_coordinate(int64_t index, const SmallVector& stride, SmallVector& coord);
    void compute_coordinate(int64_t index, const Tray& tray, SmallVector& coord);
    int64_t compute_index(const SmallVector& coord, const SmallVector& stride);
    std::optional<SmallVector> binop_broadcastable(const SmallVector& shape1, const SmallVector& shape2);
    std::tuple<std::optional<SmallVector>, std::optional<SmallVector>> gather_broadcastable(const SmallVector& self, int64_t dim, const SmallVector& index);
    SmallVector get_right_aligned_broadcasted_stride(const SmallVector& orig_shape, const SmallVector& orig_stride, const SmallVector& broadcasted_shape);
    SmallVector get_left_aligned_broadcasted_stride(const SmallVector& orig_shape, const SmallVector& orig_stride, const SmallVector& broadcasted_shape);
    std::optional<SmallVector> scatter_broadcastable(const SmallVector& self_shape, int64_t dim, const SmallVector& index_shape, const SmallVector& src_shape);
}

// helper macro-like functions
oven::SmallVector CHECK_BINARY_BROADCAST(const oven::Tray&, const oven::Tray&, const std::string &);
std::tuple<oven::SmallVector, oven::SmallVector> CHECK_GATHER_BROADCAST(const oven::Tray& self, int64_t dim, const oven::Tray& index);
oven::SmallVector CHECK_SCATTER_BROADCAST(const oven::Tray& self, int64_t dim, const oven::Tray& index, const oven::Tray& src);