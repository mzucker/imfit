add_executable(hello hello.cpp)
target_link_libraries(hello ${OPENCV_LIBRARIES} ${CERES_LIBRARIES} levmar)

add_executable(imfit imfit.cpp)
target_link_libraries(imfit ${OPENCV_LIBRARIES} levmar)
