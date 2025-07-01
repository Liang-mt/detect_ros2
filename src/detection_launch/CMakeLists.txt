cmake_minimum_required(VERSION 3.8)
project(detection_launch)

find_package(ament_cmake REQUIRED)

# 安装launch文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
