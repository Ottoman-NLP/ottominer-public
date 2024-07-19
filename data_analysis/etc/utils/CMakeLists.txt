cmake_minimum_required(VERSION 3.14)
project(MetricsGUI)

set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)
find_package(Qt5Widgets REQUIRED)

add_library(metrics_gui SHARED MetricsWindow.cpp)
target_link_libraries(metrics_gui Qt5::Widgets pybind11::module)
