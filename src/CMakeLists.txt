add_executable(imfit imfit.cpp)
target_link_libraries(imfit ${OPENCV_LIBRARIES} ${LEVMAR_LIBRARIES})
