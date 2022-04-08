# CMakeList.txt : CMake project for CVOperationTask, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project (CVOperationTask)
 
add_library(${PROJECT_NAME} MnistClassifier.cpp MnistClassifier.h)

add_executable(executable Demo.cpp)

target_link_libraries(executable ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
