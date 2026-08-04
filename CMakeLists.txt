cmake_minimum_required(VERSION 3.18)
project(vonnx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# NVCC 編譯選項
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler=/wd4819")

# 引入標頭檔路徑
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 搜尋 ONNX Runtime
if(DEFINED ONNXRUNTIME_ROOT_DIR)
    include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
    link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)
endif()

# 核心原始碼檔案
set(VONNX_SOURCES
    src/paged_tensor_pool.cu
    src/cuda_preprocess.cu
    src/cuda_nms.cu
    src/onnx_runner.cpp
    src/continuous_scheduler.cpp
    src/vonnx_engine.cpp
)

# 編譯 C++/CUDA 核心靜態函式庫
add_library(vonnx_static STATIC ${VONNX_SOURCES})
set_target_properties(vonnx_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 搜尋並連結必要函式庫
find_package(CUDA REQUIRED)
target_link_libraries(vonnx_static PRIVATE ${CUDA_LIBRARIES} onnxruntime)

# 測試與 Benchmark 可執行檔
add_executable(vonnx_benchmark tests/benchmark_vonnx.cpp)
target_link_libraries(vonnx_benchmark PRIVATE vonnx_static ${CUDA_LIBRARIES} onnxruntime)

# 訊息
message(STATUS "vONNX Engine CMake 配置完成。")
