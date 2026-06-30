#include "oven/tray/types.hpp"
#include "oven/utils/assert.hpp"
#include <oven/tray/utils.hpp>
#include <oven/tray/tray.hpp>
#include <optional>
#include <tuple>

namespace oven::detail {
// helper function for computing stride from given shape
SmallVector compute_stride(const SmallVector& shape) {
    SmallVector ret(shape.size(), 1);
    for (int64_t i = shape.size() - 1; i >= 1; i--) {
        ret[i - 1] = ret[i] * shape[i];
    }

    return ret;
}    

void compute_coordinate(int64_t index, const Tray& tray, SmallVector& coord) {
    // this function caculates coordinate from row-major index
    // coord expects a clean SmallVector initialized to 0.
    const auto& stride = tray.stride();
    int64_t remaining = index;
    for(int64_t i = 0; i < stride.size(); i++) {
        coord[i] = remaining / stride[i];
        remaining %= stride[i];
    }
}

void compute_coordinate(int64_t index, const SmallVector& stride, SmallVector& coord) {
    // this function caculates coordinate from row-major index
    // coord expects a clean SmallVector initialized to 0.
    int64_t remaining = index;
    for(int64_t i = 0; i < stride.size(); i++) {
        coord[i] = remaining / stride[i];
        remaining %= stride[i];
    }
}

int64_t compute_index(const SmallVector& coord, const SmallVector& stride) {
    // this function caculates row-major index
    int64_t ret = 0;
    for (int64_t i = 0; i < coord.size(); i++) {
        ret += coord[i] * stride[i];
    }
    return ret;
}

std::optional<SmallVector> binop_broadcastable(const SmallVector& shape1, const SmallVector& shape2) {
    const SmallVector& bigger_sized = shape1.size() > shape2.size() ? shape1 : shape2;
    const SmallVector& smaller_sized = shape1.size() <= shape2.size() ? shape1 : shape2;

    SmallVector ret(bigger_sized);

    int64_t diff = bigger_sized.size() - smaller_sized.size();

    for(int64_t i = smaller_sized.size() - 1; i >= 0; i--) {
        if (bigger_sized[i + diff] != smaller_sized[i]) {
            if (bigger_sized[i + diff] == 1)
                ret[i + diff] = smaller_sized[i];
            else if (smaller_sized[i] == 1)
                ret[i + diff] = bigger_sized[i + diff];
            else return std::nullopt;
        }
    }
    return ret;
}

// returns the broadcasted index and self
std::tuple<std::optional<SmallVector>, std::optional<SmallVector>> gather_broadcastable(const SmallVector& self_shape, int64_t dim, const SmallVector& index_shape) {
    const SmallVector& bigger_sized = self_shape.size() > index_shape.size() ? self_shape : index_shape;
    const SmallVector& smaller_sized = self_shape.size() <= index_shape.size() ? self_shape : index_shape;

    SmallVector broadcasted_index(bigger_sized);
    SmallVector broadcasted_self;

    for(int64_t i = 0; i < smaller_sized.size(); i++) {
        if (i == dim) {
            broadcasted_index[i] = index_shape[i];
        } else {
            if (self_shape[i] != index_shape[i]) {
                if (self_shape[i] == 1) broadcasted_index[i] = index_shape[i];
                else if (index_shape[i] == 1) broadcasted_index[i] = self_shape[i];
                else return {std::nullopt, std::nullopt};
            } 
        }
    }
    // copy
    broadcasted_self = broadcasted_index;
    broadcasted_self[dim] = self_shape[dim];

    return {broadcasted_self, broadcasted_index};
}

std::optional<SmallVector> scatter_broadcastable(const SmallVector& self_shape, int64_t dim, const SmallVector& index_shape, const SmallVector& src_shape) {
    // assumes that index_shape and src_shape is same.
    // we take both shapes for possible flexible broadcast implementation later.
    SmallVector broadcasted_index(self_shape);
    for (int64_t i = 0; i < index_shape.size(); i++) {
        if (i == dim) broadcasted_index[i] = index_shape[i];
        else if (index_shape[i] != self_shape[i]) {
            if (self_shape[i] == 1) return std::nullopt;
        }
    }
    
    return broadcasted_index;
}

SmallVector get_right_aligned_broadcasted_stride(const SmallVector& orig_shape, const SmallVector& orig_stride, const SmallVector& broadcasted_shape) {
    SmallVector ret(broadcasted_shape.size(), 0);
    int64_t diff = broadcasted_shape.size() - orig_shape.size();
    for (int64_t i = orig_shape.size() - 1; i >= 0; i--) {
        if (orig_shape[i] == broadcasted_shape[i + diff]) ret[i + diff] = orig_stride[i];
    }
    return ret;
}

SmallVector get_left_aligned_broadcasted_stride(const SmallVector& orig_shape, const SmallVector& orig_stride, const SmallVector& broadcasted_shape) {
    SmallVector ret(broadcasted_shape.size(), 0);
    for (int64_t i = orig_shape.size() - 1; i >= 0; i--) {
        if (orig_shape[i] == broadcasted_shape[i]) ret[i] = orig_stride[i];
    }
    return ret;
}

int64_t compute_numel(const SmallVector &shape) {
    int64_t ret = 1;
    for (int64_t i = 0; i < shape.size(); i++) ret *= shape[i];
    return ret;
}

}// namespace oven::detail

oven::SmallVector CHECK_BINARY_BROADCAST(const oven::Tray& self, const oven::Tray& other, std::string op) {
    std::optional<oven::SmallVector> shape = oven::detail::binop_broadcastable(self.shape(), other.shape());
    OVEN_ASSERT(shape != std::nullopt, "Not broadcastable for " + op + " operation.");
    // later, make SmallerVector print function for debugging...
    // checking logics for binary operation goes here...
    OVEN_TRAY_BINOP_CHECKLIST(self, other);

    return *shape;
}

std::tuple<oven::SmallVector, oven::SmallVector> CHECK_GATHER_BROADCAST(const oven::Tray& self, int64_t dim, const oven::Tray& index) {
    auto [broadcasted_self, broadcasted_index] = oven::detail::gather_broadcastable(self.shape(), dim, index.shape());
    OVEN_ASSERT(broadcasted_self != std::nullopt, "Not broadcastable for gather operation.");
    OVEN_ASSERT(index.dtype() == oven::kInt32 || index.dtype() == oven::kInt64, "Index is not integer type");
    return {*broadcasted_self, *broadcasted_index};
}

oven::SmallVector CHECK_SCATTER_BROADCAST(const oven::Tray& self, int64_t dim, const oven::Tray& index, const oven::Tray& src) {
    auto broadcasted_index = oven::detail::scatter_broadcastable(self.shape(), dim, index.shape(), src.shape());
    OVEN_ASSERT(broadcasted_index != std::nullopt, "Not broadcastable for scatter operation.");
    OVEN_ASSERT(index.dtype() == oven::kInt32 || index.dtype() == oven::kInt64, "Index is not integer type");
    OVEN_ASSERT(self.dtype() == src.dtype(), "Self and src dtype does not matches.");
    return *broadcasted_index;
}