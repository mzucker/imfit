if(CERES_FOUND)
  add_executable(hello hello.cpp)
  target_link_libraries(hello ${OPENCV_LIBRARIES} ${CERES_LIBRARIES} levmar)
endif(CERES_FOUND)

if(OpenCL_FOUND)
  add_executable(opencl_devices opencl_devices.c)
  target_link_libraries(opencl_devices ${OpenCL_LIBRARY})
endif(OpenCL_FOUND)

add_executable(imfit imfit.cpp)
target_link_libraries(imfit ${OPENCV_LIBRARIES} ${LEVMAR_LIBRARIES})
