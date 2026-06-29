#include "oven/tray/types.hpp"
// #include <limits>
#include <oven/tray/dispatcher.hpp>

namespace oven::detail {

// struct OpSupport {
//     // current OpCode is uint16_t
//     // current DType is uint8_t
//     static uint8_t supports[static_cast<uint16_t>(OpCode::TOTAL)];
//     OpSupport() {
//         // default is all possible
//         for(uint16_t i = 0; i < static_cast<uint16_t>(OpCode::TOTAL); i++) {
//             supports[i] = std::numeric_limits<uint8_t>::max();
//         }
//         disable_op(OpCode::add, DType::kBool);
//         disable_op(OpCode::sub, DType::kBool);
//         disable_op(OpCode::div, DType::kBool);
//     }

//     static void enable_all(OpCode code) {
//         supports[static_cast<uint16_t>(code)] = std::numeric_limits<uint8_t>::max();
//     }

//     static void disable_all(OpCode code) {
//         supports[static_cast<uint16_t>(code)] = 0;
//     }

//     static void enable_op(OpCode code, DType dtype) {
//         supports[static_cast<uint16_t>(code)] |= 1 << static_cast<uint8_t>(dtype);
//     }

//     static void disable_op(OpCode code, DType dtype) {
//         supports[static_cast<uint16_t>(code)] &= ~(1 << static_cast<uint8_t>(dtype));
//     }

//     bool validate() {

//     }
// };

Dispatcher& Dispatcher::get_instance() {
    // global instance
    static Dispatcher dispatcher;
    return dispatcher;
}

// bool validate_operation(OpCode code, DType dtype) {

// }

}