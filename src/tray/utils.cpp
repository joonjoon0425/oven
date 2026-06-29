#include <oven/tray/utils.hpp>

#include <optional>

namespace oven::detail {
// helper function for computing stride from given shape
SmallVector compute_stride(const SmallVector& shape) {
    SmallVector ret(shape.size(), 1);
    for (int64_t i = shape.size() - 1; i >= 1; i--) {
        ret[i - 1] = ret[i] * shape[i];
    }

    return ret;
}    

void compute_coordinate(int64_t index, const SmallVector& stride, SmallVector& coord) {
    // this function caculates coordinate from row-major index
    // coord expects a clean SmallVector initialized to 0.
    int64_t remaining = index;
    for(int64_t i = 0; i < stride.size(); i++) {
        if (remaining == 0) break;
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

std::optional<SmallVector> broadcastable(const SmallVector& shape1, const SmallVector& shape2) {
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


SmallVector get_broadcasted_stride(const SmallVector& shape, const SmallVector& broadcasted_shape) {
    SmallVector ret(broadcasted_shape.size(), 0);
    int64_t diff = broadcasted_shape.size() - shape.size();
    int64_t cur_stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; i--) {
        if (shape[i] == broadcasted_shape[i + diff]) ret[i + diff] = cur_stride;
        cur_stride *= shape[i];
    }
    return ret;
}

int64_t compute_n_elements(const SmallVector &shape) {
    int64_t ret = 1;
    for (int64_t i = 0; i < shape.size(); i++) ret *= shape[i];
    return ret;
}

}// namespace oven::detail