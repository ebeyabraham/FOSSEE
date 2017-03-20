cmake_minimum_required(VERSION 2.8.12)

project(face)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(face face.cpp)
target_link_libraries(face ${OpenCV_LIBS})