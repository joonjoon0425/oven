cmake_minimum_required(VERSION 3.28)
project(oven LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(TORCH_CUDA_ARCHITECTURES "8.9")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/libtorch")

add_library(oven SHARED
    src/test.cpp
)

find_package(nlohmann_json REQUIRED)
target_link_libraries(oven PRIVATE nlohmann_json::nlohmann_json)

find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found!")
endif()

if (CUDA_FOUND)
    message(STATUS "[oven] Linking with GPU-enabled LibTorch")

    target_link_libraries(oven PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(oven PRIVATE OVEN_USE_CUDA)

    find_package(NVSHMEM)
    if (NVSHMEM_FOUND)
        message(STATUS "[oven] Found NVSHMEM: ${NVSHMEM}. Multi-GPU support enabled.")
        target_link_libraries(oven PUBLIC ${NVSHMEM_LIBRARIES})
        target_compile_definitions(oven PRIVATE OVEN_USE_NVSHMEM)
    else()
        message(STATUS "[oven] NVSHMEM not found. Adding linker flag to ignore compiler error.")
        target_link_options(oven PRIVATE "-Wl,--allow-shlib-undefined")
    endif()
else()
    message(STATUS "[oven] Linking with CPU-only LibTorch")
    target_link_libraries(oven PUBLIC ${TORCH_LIBRARIES})
endif()

target_include_directories(oven PUBLIC include ${TORCH_INCLUDE_DIRECTORIES})

add_subdirectory(tests)