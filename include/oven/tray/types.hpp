#pragma once
#include <cstdint>
#include <variant>
#include <vector>
namespace oven {

class Tray;

enum class DType : uint8_t {Undefined = 0, kBool, kInt32, kInt64, kFloat32, kFloat64, TOTAL};
enum class Device : uint8_t {CPU, CUDA, TOTAL};

inline constexpr DType kBool = DType::kBool;
inline constexpr DType kInt32 = DType::kInt32;
inline constexpr DType kInt64 = DType::kInt64;
inline constexpr DType kFloat32 = DType::kFloat32;
inline constexpr DType kFloat64 = DType::kFloat64;

using Scalar = std::variant<bool, int32_t, int64_t, float, double>;
using SmallVector = std::vector<int64_t>;

using ElementWiseKernelType = Tray(void*, void*, const SmallVector&, const SmallVector&, const SmallVector&, DType);
using TerneryKernelType = Tray(bool*, void*, void*, const SmallVector&, const SmallVector&, const SmallVector&, const SmallVector&, DType);

}// namespace oven