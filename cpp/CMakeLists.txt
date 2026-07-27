cmake_minimum_required(VERSION 3.16)

project(inferref_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The reader and comparator are header-only and dependency-free: an engine can
# also just copy include/inferref/ into its own tree.
add_library(inferref_headers INTERFACE)
target_include_directories(inferref_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(inferref_headers INTERFACE cxx_std_17)

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_executable(inferref_compare examples/compare_tensors/main.cpp)
target_link_libraries(inferref_compare PRIVATE inferref_headers)

add_executable(inferref_selftest examples/selftest/main.cpp)
target_link_libraries(inferref_selftest PRIVATE inferref_headers)

enable_testing()
add_test(NAME irtensor_selftest COMMAND inferref_selftest)
