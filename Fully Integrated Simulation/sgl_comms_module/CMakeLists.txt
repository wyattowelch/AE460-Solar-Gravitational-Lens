cmake_minimum_required(VERSION 3.16)
project(sgl_comms_module LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sgl_comms_module src/comms_module.cpp)
target_include_directories(sgl_comms_module PUBLIC include)



add_executable(comms_standalone_demo examples/standalone_demo.cpp)
target_link_libraries(comms_standalone_demo PRIVATE sgl_comms_module)

enable_testing()
add_executable(test_comms tests/test_comms.cpp)
target_link_libraries(test_comms PRIVATE sgl_comms_module)
add_test(NAME test_comms COMMAND test_comms)
