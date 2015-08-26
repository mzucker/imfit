if(${CERES_FOUND})
  add_executable(hello hello.cpp)
  target_link_libraries(hello ${OPENCV_LIBRARIES} ${CERES_LIBRARIES} levmar)
endif(${CERES_FOUND})

add_executable(imfit imfit.cpp)
target_link_libraries(imfit ${OPENCV_LIBRARIES} ${LEVMAR_LIBRARIES})
