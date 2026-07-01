#pragma once

// A LibTorch-like Dispatcher. I don't know why I made this. Actually.
#include <iostream>
#include <oven/tray/types.hpp>
#include <utility>
#include <span>
#include <oven/utils/assert.hpp> // <-------Change this later if you want to completely dispatch oven and tray.

namespace oven::detail {

enum class OpCode : uint16_t {
    add,
    sub,
    mul,
    div,
    matmul,

    add_st,
    add_ts,
    sub_st,
    sub_ts,
    mul_ts,
    mul_st,
    div_ts,
    div_st,

    le,
    leq,
    ge,
    geq,
    eq,
    neq,
    ternery,

    le_st,
    le_ts,
    leq_st,
    leq_ts,
    ge_st,
    ge_ts,
    geq_st,
    geq_ts,
    eq_st,
    eq_ts,
    neq_st,
    neq_ts,

    neg,
    exp,
    log,
    recip,

    zeros,
    ones,
    full,
    rand,
    randint,
    randn,
    empty,

    gather,
    scatter_,


    TOTAL
};

// bool validate_operation(OpCode code, DType dtype);

struct DispatchKey {
    uint32_t key;

    // Currently, 16 bits are used for operator code, 8 bits for each data type and device type.
    // Only the dtypes are saved as bitmasks
    DispatchKey(OpCode opcode, DType dtype, Device device) {
        key = (static_cast<uint32_t>(opcode) << 16)|(static_cast<uint8_t>(dtype) << 8)|(static_cast<uint8_t>(device));
    }
};

class Dispatcher {
private:
    using FuncPtr = void(*)();
    // We register all the functions by erasing their types
    // Later, function signatures will be given as template parameters when dispatcher is called
    FuncPtr registry[static_cast<uint16_t>(OpCode::TOTAL)][static_cast<uint8_t>(DType::TOTAL)][static_cast<uint8_t>(Device::TOTAL)] = {nullptr, };
    Dispatcher() = default;

public:
    // This will return a singleton Dispatcher object
    static Dispatcher& get_instance();

    template<typename FuncType>
    void registration(DispatchKey key, FuncType&& op) {
        // Separate each key value with bit operations.
        // note that dtype may contain all possible dtypes, so we need separate it.
        auto opcode = (key.key >> 16) & 0xFFFF;
        auto dtype = (key.key >> 8) & 0xFF;
        auto device = (key.key) & 0xFF;
        registry[opcode][dtype][device] = reinterpret_cast<FuncPtr>(op);
    }

    template <typename FuncType, typename... Args>
    auto dispatch(DispatchKey key, Args&&... args) {
        auto opcode = (key.key >> 16) & 0xFFFF;
        auto dtype = (key.key >> 8) & 0xFF;
        auto device = (key.key) & 0xFF;
        auto kernel = reinterpret_cast<FuncType*>(registry[opcode][dtype][device]);
        OVEN_ASSERT(kernel != nullptr, "Unregisterd kernel.");
        return kernel(std::forward<Args>(args)...);
    }
};

// This is an inverse type dispatcher
template <typename T>
struct CppTypeToDType;

template <> struct CppTypeToDType<int32_t> {static constexpr DType value = DType::kInt32;};
template <> struct CppTypeToDType<int64_t> {static constexpr DType value = DType::kInt64;};
template <> struct CppTypeToDType<bool> {static constexpr DType value = DType::kBool;};
template <> struct CppTypeToDType<float> {static constexpr DType value = DType::kFloat32;};
template <> struct CppTypeToDType<double> {static constexpr DType value = DType::kFloat64;};

template <typename T>
inline constexpr DType CppTypeToDType_v = CppTypeToDType<T>::value;

}// namespace oven::detail

// This is a structure for registration of a kernel.
// Since it is not allowed to call a function globally, we create a static structure, calling its constructor internally.
struct KernelRegistration {
    template <typename FuncType>
    KernelRegistration(oven::detail::DispatchKey key, FuncType&& op) {
        oven::detail::Dispatcher::get_instance().registration(key, op);
    }
};

struct KernelRegistrationTypes {
    template <typename FuncType>
    KernelRegistrationTypes(oven::detail::OpCode code, std::span<const oven::DType> dtypes, oven::Device device, FuncType&& op) {
        for (auto dtype : dtypes) {
            oven::detail::Dispatcher::get_instance().registration({code, dtype, device}, op);
        }
    }
};

// This is a macro for registration
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

#define TRAY_REGISTER(opname, device, dtypes, kernel) \
    static KernelRegistrationTypes\
    CONCAT(regist_##opname##device, __COUNTER__)\
    (oven::detail::OpCode::opname, dtypes, oven::Device::device, kernel);

// This is a macro for type dispatching
// Like LibTorch!!
#define TRAY_DISPATCH_CASE(enum_dtype, cpp_type, ...) \
    case enum_dtype: {\
        using scalar_t = cpp_type;\
        return __VA_ARGS__();\
    }

#define TRAY_DISPATCH_ALL_TYPES(DTYPE, OP_NAME, ...) [&] { \
    switch (DTYPE) {\
        TRAY_DISPATCH_CASE(oven::DType::kFloat32, float, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kFloat64, double, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kInt64, int64_t, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kInt32, int32_t, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kBool, bool, __VA_ARGS__)\
        default:\
            OVEN_ASSERT(false, "No matching DType for following operator: " + OP_NAME);\
    }\
}()

#define TRAY_DISPATCH_INT_TYPES(DTYPE, OP_NAME, ...) [&] { \
    switch (DTYPE) {\
        TRAY_DISPATCH_CASE(oven::DType::kInt64, int64_t, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kInt32, int32_t, __VA_ARGS__)\
        default:\
            OVEN_ASSERT(false, "No matching DType for following operator: " + OP_NAME);\
    }\
}()

#define TRAY_DISPATCH_FLOAT_TYPES(DTYPE, OP_NAME, ...) [&] { \
    switch (DTYPE) {\
        TRAY_DISPATCH_CASE(oven::DType::kFloat32, float, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kFloat64, double, __VA_ARGS__)\
        default:\
            OVEN_ASSERT(false, "No matching DType for following operator: " + OP_NAME);\
    }\
}()

#define TRAY_DISPATCH_NONBOOL_TYPES(DTYPE, OP_NAME, ...) [&] { \
    switch (DTYPE) {\
        TRAY_DISPATCH_CASE(oven::DType::kInt64, int64_t, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kInt32, int32_t, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kFloat32, float, __VA_ARGS__)\
        TRAY_DISPATCH_CASE(oven::DType::kFloat64, double, __VA_ARGS__)\
        default:\
            OVEN_ASSERT(false, "No matching DType for following operator: " + OP_NAME);\
    }\
}()
