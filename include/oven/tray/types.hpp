#pragma once
#include <cstdint>
namespace oven {

enum class DType : uint8_t {Undefined = 0, kBool, kInt32, kInt64, kFloat32, kFloat64, TOTAL};
enum class Device : uint8_t {CPU, CUDA, TOTAL};

inline constexpr DType kBool = DType::kBool;
inline constexpr DType kInt32 = DType::kInt32;
inline constexpr DType kInt64 = DType::kInt64;
inline constexpr DType kFloat32 = DType::kFloat32;
inline constexpr DType kFloat64 = DType::kFloat64;

}// namespace oven