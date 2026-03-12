cmake_minimum_required(VERSION 3.25)
project(involute VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# 1. Global C++ Standards & Project Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend Selection Options
option(USE_MLX "Build with Apple Silicon (Metal/MLX) backend" OFF)
option(USE_SYCL "Build with PC/Intel/NVIDIA (SYCL/oneMKL) backend" OFF)
option(BUILD_TESTING "Build the Involute test suite and experiments" ON)

if (USE_MLX AND USE_SYCL)
    message(FATAL_ERROR "Involute: Choose either USE_MLX or USE_SYCL, not both.")
elseif (NOT USE_MLX AND NOT USE_SYCL)
    message(FATAL_ERROR "Involute: No backend selected. Use -DUSE_MLX=ON or -DUSE_SYCL=ON.")
endif ()

# ==============================================================================
# 2. Core Library Definition
# ==============================================================================
add_library(involute)

target_sources(involute
        PUBLIC FILE_SET HEADERS
        BASE_DIRS include
        FILES
        include/involute/core/tensor.hpp
        include/involute/core/math.hpp
        include/involute/core/objective.hpp
        include/involute/core/parameter_adapter.hpp
        include/involute/core/result.hpp
        include/involute/core/base_solver.hpp
        include/involute/solvers/so_parameter_adapter.hpp
        include/involute/solvers/adam_parameter_adapter.hpp
        include/involute/solvers/so_solver.hpp
        include/involute/solvers/stiefel_solver.hpp
)

# ==============================================================================
# 3. Conditional Debugging Configuration
# ==============================================================================
# Only applies if -DCMAKE_BUILD_TYPE=Debug is explicitly passed.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Involute: Debug mode enabled. Adding sanitizers and symbols.")

    # MSVC Specific Debug Info
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

    # General Debug Flags (Sanitizers + Symbols)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        # Stack trace support and frame pointers
        target_compile_options(involute PRIVATE -g -fno-omit-frame-pointer -fstack-protector-strong)
        set(CMAKE_ENABLE_EXPORTS ON)
    endif ()

    if (APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-export_dynamic")
    endif ()

    # Stacktrace support for tests/backends
    target_compile_definitions(involute PUBLIC CATCH_CONFIG_ENABLE_STACKTRACE)
endif ()

# ==============================================================================
# 4. Release Optimization Flags
# ==============================================================================
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        target_compile_options(involute PRIVATE -O3 -ffast-math -march=native)
    endif ()
endif ()

# ==============================================================================
# 5. Backend-Specific Implementation
# ==============================================================================
if (USE_MLX)
    message(STATUS "Involute: Configuring MLX Backend (Apple Silicon)")
    target_compile_definitions(involute PUBLIC USE_MLX)
    target_compile_definitions(involute PUBLIC MLX_THROW_ERRORS=1)

    target_sources(involute PRIVATE
            src/backends/mlx/tensor_mlx.cpp
            src/backends/mlx/math_mlx.cpp
            src/backends/mlx/so_d_mlx.cpp
            src/backends/mlx/stiefel_mlx.cpp
    )

    find_package(MLX REQUIRED)
    target_link_libraries(involute PRIVATE mlx)

elseif (USE_SYCL)
    message(STATUS "Involute: Configuring SYCL/oneMKL Backend (PC GPU)")
    target_compile_definitions(involute PUBLIC USE_SYCL)

    target_sources(involute PRIVATE
            src/backends/sycl/tensor_sycl.cpp
            src/backends/sycl/math_sycl.cpp
            src/backends/sycl/so_d_sycl.cpp
            src/backends/sycl/stiefel_sycl.cpp
    )

    find_package(MKL REQUIRED)
    target_compile_options(involute PRIVATE -fsycl)
    target_link_options(involute PRIVATE -fsycl)
    target_link_libraries(involute PRIVATE MKL::MKL)
endif ()

# ==============================================================================
# 6. Test/Experiment Subdirectory
# ==============================================================================
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif ()

install(TARGETS involute
        EXPORT InvoluteTargets
        FILE_SET HEADERS
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(EXPORT InvoluteTargets
        FILE InvoluteTargets.cmake
        NAMESPACE involute::
        DESTINATION lib/cmake/involute
)