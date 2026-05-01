cmake_minimum_required(VERSION 3.25)
project(involute VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_TESTING "Build the Involute test suite and experiments" ON)

# ==============================================================================
# 1. Core Library Definition
# ==============================================================================
add_library(involute STATIC
    src/core/objective.cpp
    src/core/base_solver.cpp
    src/solvers/adapters/adam_parameter_adapter.cpp
    src/solvers/adapters/cma_es_parameter_adapter.cpp
    src/solvers/adapters/so_adam_parameter_adapter.cpp
    src/solvers/isotropic/so_isotropic_solver.cpp
    src/solvers/isotropic/so_isotropic_solver_adam.cpp
    src/solvers/isotropic/so_isotropic_solver_cmaes.cpp
    src/solvers/isotropic/stiefel_isotropic_solver.cpp
    src/solvers/isotropic/stiefel_isotropic_solver_adam.cpp
    src/solvers/isotropic/stiefel_isotropic_solver_cmaes.cpp
    src/solvers/isotropic/product_isotropic_solver.cpp
    src/solvers/isotropic/product_isotropic_solver_adam.cpp
    src/solvers/isotropic/product_isotropic_solver_cmaes.cpp
    src/solvers/anisotropic/so_anisotropic_solver_adam.cpp
    src/solvers/anisotropic/so_anisotropic_solver_cmaes.cpp
    src/solvers/anisotropic/so_anisotropic_solver_cmaes_spatial.cpp
)
add_library(involute::involute ALIAS involute)

target_include_directories(involute PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(involute
    PUBLIC FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/involute/core/tensor.hpp
        include/involute/core/math.hpp
        include/involute/core/hyper_parameters.hpp
        include/involute/core/objective.hpp
        include/involute/core/parameter_adapter.hpp
        include/involute/core/result.hpp
        include/involute/core/base_solver.hpp
        include/involute/solvers/adapters/adam_parameter_adapter.hpp
        include/involute/solvers/adapters/cma_es_parameter_adapter.hpp
        include/involute/solvers/adapters/so_adam_parameter_adapter.hpp
        include/involute/solvers/isotropic/so_isotropic_solver.hpp
        include/involute/solvers/isotropic/so_isotropic_solver_adam.hpp
        include/involute/solvers/isotropic/so_isotropic_solver_cmaes.hpp
        include/involute/solvers/isotropic/stiefel_isotropic_solver.hpp
        include/involute/solvers/isotropic/stiefel_isotropic_solver_adam.hpp
        include/involute/solvers/isotropic/stiefel_isotropic_solver_cmaes.hpp
        include/involute/solvers/isotropic/product_isotropic_solver.hpp
        include/involute/solvers/isotropic/product_isotropic_solver_adam.hpp
        include/involute/solvers/isotropic/product_isotropic_solver_cmaes.hpp
        include/involute/solvers/anisotropic/so_anisotropic_solver_adam.hpp
        include/involute/solvers/anisotropic/so_anisotropic_solver_cmaes.hpp
        include/involute/solvers/anisotropic/so_anisotropic_solver_cmaes_spatial.hpp
)

# ==============================================================================
# 2. Build-type flags
# ==============================================================================
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    # -ffast-math is excluded for the same reason as the sampler: involute's
    # solver logic uses std::numeric_limits::infinity() as sentinel values and
    # compares against them. -ffinite-math-only (implied by -ffast-math) allows
    # the compiler to assume all values are finite, turning those comparisons into
    # constants and silently breaking best-consensus tracking and convergence checks.
    target_compile_options(involute PRIVATE -O3 -march=native)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(involute PRIVATE -g -fno-omit-frame-pointer)
    target_compile_options(involute PUBLIC -fsanitize=address -fsanitize=undefined)
    target_link_options(involute PUBLIC -fsanitize=address -fsanitize=undefined)
endif ()

# ==============================================================================
# 3. External Dependencies
# ==============================================================================

# ---- riemannian-gaussian-sampler (dual solution) ----
set(SAMPLER_DIR "" CACHE PATH "Path to the riemannian-gaussian-sampler source directory (development only)")

if (SAMPLER_DIR STREQUAL "")
    get_filename_component(_parent "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(SAMPLER_DIR "${_parent}/riemannian-gaussian-sampler")
endif ()

if (EXISTS "${SAMPLER_DIR}/CMakeLists.txt")
    message(STATUS "Involute: using sampler source at ${SAMPLER_DIR}")
    add_subdirectory(${SAMPLER_DIR} sampler_build)
    set(_sampler_link_target sampler)
else ()
    find_package(sampler REQUIRED)
    set(_sampler_link_target sampler::sampler)
    message(STATUS "Involute: linking against installed sampler")
endif ()

# ---- isomorphism (dual solution) ----
set(ISOMORPHISM_DIR "" CACHE PATH "Path to the isomorphism source directory (development only)")

if (ISOMORPHISM_DIR STREQUAL "")
    get_filename_component(_parent "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(ISOMORPHISM_DIR "${_parent}/isomorphism")
endif ()

if (NOT TARGET isomorphism)
    if (EXISTS "${ISOMORPHISM_DIR}/CMakeLists.txt")
        message(STATUS "Involute: using isomorphism source at ${ISOMORPHISM_DIR}")
        add_subdirectory(${ISOMORPHISM_DIR} isomorphism_build)
        set(_iso_link_target isomorphism)
    else ()
        find_package(isomorphism REQUIRED)
        if (NOT isomorphism_DEFAULT_TARGET)
            message(FATAL_ERROR
                "find_package(isomorphism) succeeded but isomorphism_DEFAULT_TARGET is not set. "
                "Re-install isomorphism with a backend enabled "
                "(-DUSE_MLX=ON, -DUSE_EIGEN=ON, or -DUSE_TORCH=ON).")
        endif ()
        set(_iso_link_target ${isomorphism_DEFAULT_TARGET})
        message(STATUS "Involute: linking against installed ${_iso_link_target}")
    endif ()
else ()
    set(_iso_link_target isomorphism)
    message(STATUS "Involute: isomorphism already configured (via sampler subdirectory)")
endif ()

target_link_libraries(involute PUBLIC ${_sampler_link_target} ${_iso_link_target})

# ==============================================================================
# 4. Test/Experiment Subdirectory
# ==============================================================================
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif ()

# ==============================================================================
# 5. Install Rules
# ==============================================================================
install(TARGETS involute
    EXPORT InvoluteTargets
    FILE_SET HEADERS
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT InvoluteTargets
    FILE        involute-targets.cmake
    NAMESPACE   involute::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/involute
)

configure_package_config_file(
    cmake/involute-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/involute-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/involute
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/involute-version.cmake
    VERSION     ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/involute-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/involute-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/involute
)
