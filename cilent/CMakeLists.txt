cmake_minimum_required(VERSION 3.8)
project(cilent)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

set(CV_BRIDGE_INCLUDE_DIR "/opt/ros/humble/include/cv_bridge")
set(CV_BRIDGE_LIBRARY "/opt/ros/humble/lib/libcv_bridge.so")
if(NOT EXISTS "${CV_BRIDGE_LIBRARY}")
  message(FATAL_ERROR "Required cv_bridge library not found: ${CV_BRIDGE_LIBRARY}")
endif()

set(OPENCV_SYSTEM_INCLUDE_DIR "/usr/include/opencv4")
set(OPENCV_SYSTEM_LIB_DIR "/usr/lib/x86_64-linux-gnu")
set(OPENCV_SYSTEM_LIBS
  "${OPENCV_SYSTEM_LIB_DIR}/libopencv_core.so.4.5d"
  "${OPENCV_SYSTEM_LIB_DIR}/libopencv_imgproc.so.4.5d"
  "${OPENCV_SYSTEM_LIB_DIR}/libopencv_imgcodecs.so.4.5d"
)

foreach(_opencv_lib IN LISTS OPENCV_SYSTEM_LIBS)
  if(NOT EXISTS "${_opencv_lib}")
    message(FATAL_ERROR "Required OpenCV library not found: ${_opencv_lib}")
  endif()
endforeach()
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_library(kalman_filter src/kalman_filter.cpp)
target_include_directories(kalman_filter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(kalman_filter Eigen3::Eigen)

install(TARGETS kalman_filter
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

add_library(image_processor src/image_processor.cpp)
target_include_directories(image_processor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CV_BRIDGE_INCLUDE_DIR}
  ${OPENCV_SYSTEM_INCLUDE_DIR}
)
ament_target_dependencies(image_processor rclcpp sensor_msgs
  image_transport)
target_link_libraries(image_processor ${CV_BRIDGE_LIBRARY} ${OPENCV_SYSTEM_LIBS})

install(TARGETS image_processor
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_library(serial_sender src/serial_sender.cpp)
target_include_directories(serial_sender PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS serial_sender
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(cilent_main src/main.cpp)
target_include_directories(cilent_main PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(cilent_main rclcpp sensor_msgs)
target_link_libraries(cilent_main image_processor serial_sender kalman_filter
  Eigen3::Eigen ${OPENCV_SYSTEM_LIBS})

install(TARGETS cilent_main
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
