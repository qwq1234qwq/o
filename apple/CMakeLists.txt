cmake_minimum_required(VERSION 3.10)
project(Apple)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(apple main.cpp)

# 链接 OpenCV 库
target_link_libraries(apple ${OpenCV_LIBS})
