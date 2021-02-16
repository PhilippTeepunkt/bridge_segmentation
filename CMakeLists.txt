cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(SegmentationPipeline)
find_package(PCL 1.5 REQUIRED)

set (CMAKE_BUILD_TYPE Debug)

set(pcl_include_dir ${PCL_INCLUDE_DIRS})
set(pcl_src_dir ${PCL_DEFINITIONS})

set(my_includes "${PROJECT_SOURCE_DIR}/include/utils.h" "${PROJECT_SOURCE_DIR}/include/viewer.h")
set(my_sources "${PROJECT_SOURCE_DIR}/source/main.cpp" "${PROJECT_SOURCE_DIR}/source/utils.cpp" "${PROJECT_SOURCE_DIR}/source/viewer.cpp")

include_directories(${pcl_include_dir} "${PROJECT_SOURCE_DIR}/include")
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${pcl_src_dir} ${my_sources})
add_executable(startPipeline ${my_sources} ${my_includes})
target_link_libraries(startPipeline ${PCL_LIBRARIES})