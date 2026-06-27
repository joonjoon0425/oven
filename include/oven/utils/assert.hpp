#pragma once

#ifdef OVEN_DISABLE_ASSERT
    #define OVEN_ASSERT(condition, message) ((void)0)
    #define OVEN_TENSOR_BINOP_CHECKLIST_STRICT(tensor1, tensor2) ((void)0)
#else
    #include <stdexcept>
    #include <string>
    #define OVEN_ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                throw std::invalid_argument(std::string("[oven Error] ") + message); \
            } \
        } while (false)

    #define OVEN_TENSOR_BINOP_CHECKLIST_STRICT(tensor1, tensor2) \
        do { \
            OVEN_ASSERT(tensor1.sizes() == tensor2.sizes(), "Must have same sizes.");\
            OVEN_ASSERT(tensor1.dtype() == tensor2.dtype(), "Must have same dtype");\
            OVEN_ASSERT(tensor1.device() == tensor2.device(), "Must be at the same device.");\
        } while (false)
#endif