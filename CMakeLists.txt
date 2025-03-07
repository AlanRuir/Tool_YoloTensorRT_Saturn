# cmake_minimum_required(VERSION 3.10)
# project(YoloTensorRT LANGUAGES CXX CUDA)

# # 设置 C++ 标准
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 设置 CUDA 标准
# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # 指定 CUDA 工具包路径
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
# find_package(CUDA REQUIRED)
# find_package(OpenCV REQUIRED)

# # 设置 TensorRT 路径
# set(TensorRT_ROOT "/home/nyx/Tools/TensorRT-10.3.0.26" CACHE PATH "TensorRT root directory")
# if(NOT EXISTS ${TensorRT_ROOT})
#     message(FATAL_ERROR "TensorRT root directory not found: ${TensorRT_ROOT}")
# endif()

# # 包含头文件目录
# include_directories(
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/include/common
#     ${CMAKE_SOURCE_DIR}/include/yolo
#     ${CUDA_INCLUDE_DIRS}
#     ${OpenCV_INCLUDE_DIRS}
#     ${TensorRT_ROOT}/include
# )

# # 收集源文件
# file(GLOB_RECURSE CPP_SOURCES 
#     ${CMAKE_SOURCE_DIR}/src/common/*.cpp
#     ${CMAKE_SOURCE_DIR}/src/yolo/*.cpp
#     ${CMAKE_SOURCE_DIR}/src/main.cpp
# )
# file(GLOB_RECURSE CUDA_SOURCES 
#     ${CMAKE_SOURCE_DIR}/src/common/*.cu
# )

# # 设置 TensorRT 库路径
# link_directories(${TensorRT_ROOT}/lib)

# # 添加可执行文件目标
# add_executable(${PROJECT_NAME}
#     ${CPP_SOURCES}
#     ${CUDA_SOURCES}
# )

# # 指定 CUDA 属性
# set_target_properties(${PROJECT_NAME} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# # 链接库
# target_link_libraries(${PROJECT_NAME} PRIVATE
#     ${CUDA_LIBRARIES}
#     ${CUDA_cudart_LIBRARY}  # 显式添加 cudart
#     ${OpenCV_LIBRARIES}
#     nvinfer
#     nvonnxparser
# )

# # 设置编译选项
# target_compile_options(${PROJECT_NAME} PRIVATE
#     $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
#     $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86> # RTX 3060
# )

# # 可选：添加调试信息
# if(CMAKE_BUILD_TYPE MATCHES Debug)
#     target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
# endif()

# # 输出信息
# message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
# message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
# message(STATUS "TensorRT_ROOT: ${TensorRT_ROOT}")

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")  # 显式指定正确的 nvcc
project(YoloTensorRT LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 指定 CUDA 工具包路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.8")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# 设置 TensorRT 路径
set(TensorRT_ROOT "/home/nyx/Tools/TensorRT-10.3.0.26" CACHE PATH "TensorRT root directory")
if(NOT EXISTS ${TensorRT_ROOT})
    message(FATAL_ERROR "TensorRT root directory not found: ${TensorRT_ROOT}")
endif()

# 包含头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/common
    ${CMAKE_SOURCE_DIR}/include/yolo
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
)

# 收集源文件
file(GLOB_RECURSE CPP_SOURCES 
    ${CMAKE_SOURCE_DIR}/src/common/*.cpp
    ${CMAKE_SOURCE_DIR}/src/yolo/*.cpp
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)
file(GLOB_RECURSE CUDA_SOURCES 
    ${CMAKE_SOURCE_DIR}/src/common/*.cu
)

# 设置 TensorRT 和 CUDA 库路径
link_directories(
    ${TensorRT_ROOT}/lib
    /usr/local/cuda-12.8/lib64
)

# 添加可执行文件目标
add_executable(${PROJECT_NAME}
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# 指定 CUDA 属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${OpenCV_LIBRARIES}
    nvinfer
    nvonnxparser
)

# 设置编译选项
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86> # RTX 3060
)

# 可选：添加调试信息
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
endif()

# 输出信息
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
message(STATUS "TensorRT_ROOT: ${TensorRT_ROOT}")