# Path: involute/tests/CMakeLists.txt


# 1. HELPERS: Interface library for shared test utilities
file(GLOB HELPER_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/helpers/*.hpp")
add_library(test_helpers INTERFACE)
target_sources(test_helpers INTERFACE
        FILE_SET HEADERS
        BASE_DIRS helpers
        FILES ${HELPER_HEADERS}
)

# 2. AUTO-DISCOVERY FUNCTION
function(add_involute_test_dir dir_path)
    file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${dir_path}/*.cpp")

    foreach(FILE IN LISTS TEST_FILES)
        get_filename_component(TARGET_NAME "${FILE}" NAME_WE)

        add_executable(${TARGET_NAME} "${FILE}")

        # -g and -O0 guarantee the highest fidelity stack traces in LLDB
        target_compile_options(${TARGET_NAME} PRIVATE -g -O0)

        target_link_libraries(${TARGET_NAME} PRIVATE
                involute
                test_helpers
                $<$<PLATFORM_ID:Linux>:backtrace>
                $<$<PLATFORM_ID:Windows>:DbgHelp>
        )
    endforeach()
endfunction()

# 3. Process Tiers
add_involute_test_dir("unit")
add_involute_test_dir("geometric")
add_involute_test_dir("functional")