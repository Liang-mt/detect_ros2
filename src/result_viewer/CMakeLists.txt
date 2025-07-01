cmake_minimum_required(VERSION 3.8)
project(result_viewer)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_msgs REQUIRED)

# 添加可执行文件
add_executable(result_viewer_node src/result_viewer_node.cpp)

# 使用ament_target_dependencies自动处理依赖关系
ament_target_dependencies(
  result_viewer_node
  rclcpp
  cv_bridge
  image_transport
  vision_msgs
  OpenCV
)
target_include_directories(result_viewer_node PRIVATE include)
# 安装可执行文件
install(TARGETS result_viewer_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
