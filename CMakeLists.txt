cmake_minimum_required(VERSION 2.6)
project(imfit)

if(APPLE)
  include_directories(/opt/local/include)
  link_directories(/opt/local/lib)
  set(LEVMAR_LIBRARIES levmar)
else(APPLE)
  set(LEVMAR_LIBRARIES levmar blas lapack)
endif(APPLE)

include(FindPkgConfig)
pkg_search_module(OPENCV REQUIRED opencv>=2.3 opencv-2.3.1)
include_directories(${OPENCV_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_subdirectory(src)
