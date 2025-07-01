cmake_minimum_required(VERSION 3.8)
project(video_publisher)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)


# 添加可执行文件
add_executable(video_publisher_node src/video_publisher_node.cpp)

# 使用ament_target_dependencies自动处理依赖关系
ament_target_dependencies(
  video_publisher_node 
  rclcpp
  cv_bridge
  image_transport
  OpenCV
)
target_include_directories(video_publisher_node  PRIVATE include)
# 安装可执行文件
install(TARGETS video_publisher_node 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
