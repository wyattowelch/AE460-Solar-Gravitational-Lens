cmake_minimum_required(VERSION 3.16)
project(sgl_star_tracker_module LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_library(sgl_star_tracker
    src/quaternion.cpp
    src/star_tracker_sim.cpp
    src/gyro_sim.cpp
    src/attitude_filter.cpp
    src/adcs_controller.cpp
    src/reaction_wheel_array.cpp
    src/adcs_system.cpp
)

target_compile_options(sgl_star_tracker PRIVATE -Wall -Wextra -Wpedantic)

add_executable(star_tracker_demo examples/star_tracker_demo.cpp)
target_link_libraries(star_tracker_demo PRIVATE sgl_star_tracker)

enable_testing()
add_executable(test_adcs_system tests/test_adcs_system.cpp)
target_link_libraries(test_adcs_system PRIVATE sgl_star_tracker)
add_test(NAME test_adcs_system COMMAND test_adcs_system)
