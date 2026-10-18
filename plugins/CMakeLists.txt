cmake_minimum_required(VERSION 3.18)
project(vad_plugins LANGUAGES C CXX CUDA)

set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "TensorRT root")
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA root")
set(ARCH "89" CACHE STRING "GPU arch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xptxas=-v")

include_directories(
    ${CUDA_ROOT}/include
    ${TRT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ms_deform_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rotate
    ${CMAKE_CURRENT_SOURCE_DIR}/src/select_and_pad
)

add_library(plugins_cu STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ms_deform_attn/ms_deform_attn_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rotate/rotateKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/select_and_pad/select_and_pad_kernel.cu)
set_target_properties(plugins_cu PROPERTIES
    CUDA_ARCHITECTURES ${ARCH}
    POSITION_INDEPENDENT_CODE ON)

add_library(plugins SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ms_deform_attn/ms_deform_attn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rotate/rotate_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/select_and_pad/select_and_pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_def.cpp)
set_target_properties(plugins PROPERTIES CUDA_ARCHITECTURES ${ARCH})
target_link_libraries(plugins plugins_cu nvinfer cublas cudart)
target_include_directories(plugins PRIVATE
    ${CUDA_ROOT}/include ${TRT_ROOT}/include)
