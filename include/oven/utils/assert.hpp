#pragma once

#ifdef OVEN_DISABLE_ASSERT
    #define OVEN_ASSERT(condition, message) ((void)0)
#else
    #include <stdexcept>
    #include <string>
    #define OVEN_ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                throw std::invalid_argument(std::string("[oven Error] ") + message); \
            } \
        } while (false)
#endif