cmake_minimum_required(VERSION 2.8.12)

project(jellyfish)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(jellyfish jellyfish.cpp)
target_link_libraries(jellyfish ${OpenCV_LIBS})