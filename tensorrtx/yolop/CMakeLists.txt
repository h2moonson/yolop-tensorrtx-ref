cmake_minimum_required(VERSION 2.6)

project(yolop)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA  REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
include_directories(/usr/local/cuda-11.0/include)
link_directories(/usr/local/cuda-11.0/lib64)

# tensorrt
include_directories(/opt/TensorRT-8.5.3.1/include)
link_directories(/opt/TensorRT-8.5.3.1/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# to generate plugins
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/yolop.cpp ${PROJECT_SOURCE_DIR}/common.cpp)
target_link_libraries(main nvinfer cudart myplugins ${OpenCV_LIBS})
add_definitions(-O3 -pthread)

