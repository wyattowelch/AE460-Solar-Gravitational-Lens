cmake_minimum_required(VERSION 3.16)
project(sgl_eps_module LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sgl_eps_module src/eps_module.cpp)
target_include_directories(sgl_eps_module PUBLIC include)

add_executable(eps_standalone_demo examples/standalone_demo.cpp)
target_link_libraries(eps_standalone_demo PRIVATE sgl_eps_module)

enable_testing()
add_executable(test_eps tests/test_eps.cpp)
target_link_libraries(test_eps PRIVATE sgl_eps_module)
add_test(NAME test_eps COMMAND test_eps)
