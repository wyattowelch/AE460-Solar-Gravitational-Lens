cmake_minimum_required(VERSION 3.16)
project(sgl_obc_pi_jetson CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP)
option(SGL_ENABLE_CUDA "Enable CUDA backend acceleration for Jetson processing" OFF)
set(SGL_CUDA_ENABLED OFF)
if(SGL_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(SGL_CUDA_ENABLED ON)
    message(STATUS "SGL CUDA backend enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING "SGL_ENABLE_CUDA=ON but CUDA compiler not found; building CPU fallback only")
  endif()
endif()

set(SGL_STAR_TRACKER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sgl_star_tracker_module/sgl_star_tracker_module")
if(EXISTS "${SGL_STAR_TRACKER_DIR}/CMakeLists.txt")
  add_subdirectory("${SGL_STAR_TRACKER_DIR}" "${CMAKE_BINARY_DIR}/sgl_star_tracker_module_build")
endif()
set(SGL_COMMS_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sgl_comms_module")
if(EXISTS "${SGL_COMMS_MODULE_DIR}/CMakeLists.txt")
  add_subdirectory("${SGL_COMMS_MODULE_DIR}" "${CMAKE_BINARY_DIR}/sgl_comms_module_build")
endif()
set(SGL_EPS_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sgl_eps_module")
if(EXISTS "${SGL_EPS_MODULE_DIR}/CMakeLists.txt")
  add_subdirectory("${SGL_EPS_MODULE_DIR}" "${CMAKE_BINARY_DIR}/sgl_eps_module_build")
endif()
set(SGL_THERMAL_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sgl_thermal_module")
if(EXISTS "${SGL_THERMAL_MODULE_DIR}/CMakeLists.txt")
  add_subdirectory("${SGL_THERMAL_MODULE_DIR}" "${CMAKE_BINARY_DIR}/sgl_thermal_module_build")
endif()
set(SGL_PROPULSION_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sgl_propulsion_module")
if(EXISTS "${SGL_PROPULSION_MODULE_DIR}/CMakeLists.txt")
  add_subdirectory("${SGL_PROPULSION_MODULE_DIR}" "${CMAKE_BINARY_DIR}/sgl_propulsion_module_build")
endif()
set(SGL_PAYLOAD_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sgl_payload_module")
if(EXISTS "${SGL_PAYLOAD_MODULE_DIR}/CMakeLists.txt")
  add_subdirectory("${SGL_PAYLOAD_MODULE_DIR}" "${CMAKE_BINARY_DIR}/sgl_payload_module_build")
endif()

add_library(sgl_common
  common/config.cpp
  common/image_io.cpp
  common/protocol.cpp
  common/net.cpp
  common/reconstruction.cpp
)
target_include_directories(sgl_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
  target_link_libraries(sgl_common PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(sgl_pi_lib pi_flight/subsystems.cpp)
target_link_libraries(sgl_pi_lib PUBLIC sgl_common)
target_include_directories(sgl_pi_lib PUBLIC "${SGL_STAR_TRACKER_DIR}/include")
target_include_directories(sgl_pi_lib PUBLIC "${SGL_COMMS_MODULE_DIR}/include")
target_include_directories(sgl_pi_lib PUBLIC "${SGL_THERMAL_MODULE_DIR}/include")
target_include_directories(sgl_pi_lib PUBLIC "${SGL_PROPULSION_MODULE_DIR}/include")
target_include_directories(sgl_pi_lib PUBLIC "${SGL_PAYLOAD_MODULE_DIR}/include")
if(TARGET sgl_star_tracker)
  target_link_libraries(sgl_pi_lib PUBLIC sgl_star_tracker)
endif()
if(TARGET sgl_comms_module)
  target_link_libraries(sgl_pi_lib PUBLIC sgl_comms_module)
endif()
if(TARGET sgl_thermal_module)
  target_link_libraries(sgl_pi_lib PUBLIC sgl_thermal_module)
endif()
if(TARGET sgl_propulsion_module)
  target_link_libraries(sgl_pi_lib PUBLIC sgl_propulsion_module)
endif()
if(TARGET sgl_payload_module)
  target_link_libraries(sgl_pi_lib PUBLIC sgl_payload_module)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(sgl_pi_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

set(SGL_JETSON_SOURCES
  jetson_processing/processor.cpp
)
if(SGL_CUDA_ENABLED)
  list(APPEND SGL_JETSON_SOURCES jetson_processing/cuda_backend.cu)
else()
  list(APPEND SGL_JETSON_SOURCES jetson_processing/cuda_backend_stub.cpp)
endif()
add_library(sgl_jetson_lib ${SGL_JETSON_SOURCES})
target_link_libraries(sgl_jetson_lib PUBLIC sgl_common)
if(SGL_CUDA_ENABLED)
  set_target_properties(sgl_jetson_lib PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(sgl_jetson_lib PUBLIC SGL_ENABLE_CUDA=1)
else()
  target_compile_definitions(sgl_jetson_lib PUBLIC SGL_ENABLE_CUDA=0)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(sgl_jetson_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(sgl_pi_flight pi_flight/main.cpp)
target_link_libraries(sgl_pi_flight PRIVATE sgl_pi_lib sgl_common sgl_jetson_lib)
if(TARGET sgl_eps_module)
  target_link_libraries(sgl_pi_flight PRIVATE sgl_eps_module)
endif()

add_executable(sgl_jetson_service jetson_processing/main.cpp)
target_link_libraries(sgl_jetson_service PRIVATE sgl_jetson_lib sgl_common)

enable_testing()

add_executable(test_adcs_power tests/test_adcs_power.cpp)
target_link_libraries(test_adcs_power PRIVATE sgl_pi_lib sgl_common)
add_test(NAME test_adcs_power COMMAND test_adcs_power)

add_executable(test_power_scheduler tests/test_power_scheduler.cpp)
target_link_libraries(test_power_scheduler PRIVATE sgl_pi_lib sgl_common)
if(TARGET sgl_eps_module)
  target_link_libraries(test_power_scheduler PRIVATE sgl_eps_module)
endif()
if(TARGET sgl_payload_module)
  target_link_libraries(test_power_scheduler PRIVATE sgl_payload_module)
endif()
add_test(NAME test_power_scheduler COMMAND test_power_scheduler)

add_executable(test_local_processing tests/test_local_processing.cpp)
target_link_libraries(test_local_processing PRIVATE sgl_common sgl_jetson_lib)
add_test(NAME test_local_processing COMMAND test_local_processing)

add_executable(test_roi_usefulness tests/test_roi_usefulness.cpp)
target_link_libraries(test_roi_usefulness PRIVATE sgl_common)
add_test(NAME test_roi_usefulness COMMAND test_roi_usefulness)

add_executable(test_ring_observation tests/test_ring_observation.cpp)
target_link_libraries(test_ring_observation PRIVATE sgl_common sgl_jetson_lib)
add_test(NAME test_ring_observation COMMAND test_ring_observation)

add_executable(test_image_io_formats tests/test_image_io_formats.cpp)
target_link_libraries(test_image_io_formats PRIVATE sgl_common)
add_test(NAME test_image_io_formats COMMAND test_image_io_formats)

add_executable(test_source_preconditioning_formats tests/test_source_preconditioning_formats.cpp)
target_link_libraries(test_source_preconditioning_formats PRIVATE sgl_common)
add_test(
  NAME test_source_preconditioning_formats
  COMMAND test_source_preconditioning_formats ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_local_pi_run
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_local_pi_run.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_scheduler_low_budget
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scheduler_low_budget.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_scheduler_propulsion_event
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scheduler_propulsion_event.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_payload_acquisition
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_payload_acquisition.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_events_log
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_events_log.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_payload_image_modes
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_payload_image_modes.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_progressive_outputs
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_progressive_outputs.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_alignment_fallback
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_alignment_fallback.sh $<TARGET_FILE:sgl_pi_flight>
)

add_test(
  NAME test_manifest_consistency
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_manifest_consistency.sh $<TARGET_FILE:sgl_pi_flight> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_tcp_mode
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tcp_mode.sh $<TARGET_FILE:sgl_pi_flight> $<TARGET_FILE:sgl_jetson_service> ${CMAKE_CURRENT_SOURCE_DIR}
)
