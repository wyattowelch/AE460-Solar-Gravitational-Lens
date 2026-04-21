cmake_minimum_required(VERSION 3.16)
project(sgl_dhs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable CUDA acceleration (Jetson)" ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
endif()

if(ENABLE_CUDA)
  enable_language(CUDA OPTIONAL)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DSGL_ENABLE_CUDA=1)
  else()
    message(WARNING "CUDA requested but CUDA compiler not found. Building CPU-only fallback.")
  endif()
endif()

add_library(sgl_common
  common/config.cpp
  common/image_io.cpp
)
target_include_directories(sgl_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(sgl_processing
  processing/psf.cpp
)
target_include_directories(sgl_processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sgl_processing PUBLIC sgl_common)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sgl_processing PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(sgl_sim
  simulation/main.cpp
)
target_link_libraries(sgl_sim PRIVATE sgl_common sgl_processing)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sgl_sim PRIVATE OpenMP::OpenMP_CXX)
endif()

add_library(sgl_jetson_lib
  jetson/power_manager.cpp
  jetson/scheduler.cpp
)
target_link_libraries(sgl_jetson_lib PRIVATE sgl_common)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sgl_jetson_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA kernels (optional)
if(CMAKE_CUDA_COMPILER)
  add_library(sgl_cuda_kernels
    jetson/cuda/psf_cuda.cu
    jetson/cuda/wiener_cuda.cu
  )
  target_include_directories(sgl_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(sgl_cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(sgl_jetson
  jetson/main.cpp
)
target_link_libraries(sgl_jetson PRIVATE sgl_common sgl_jetson_lib sgl_processing)
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(sgl_jetson PRIVATE sgl_cuda_kernels)
endif()
