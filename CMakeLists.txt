# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.

cmake_minimum_required(VERSION 3.8)
project( pylibrsf )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/../externals/install)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../cmake)


#set (CMAKE_CXX_FLAGS "-Q")

#add_compile_options( -Ofast)
add_compile_options( -g)

find_package(Ceres 2.0 REQUIRED)
find_package(Eigen3 3.3.5 REQUIRED NO_MODULE)

find_package(PythonInterp ${3.6} REQUIRED)
find_package(PythonLibs ${3.6} REQUIRED)
#find_package(Python3)
find_package(pybind11 REQUIRED)
find_package(libRSF REQUIRED)


include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../externals/install/include/
    /usr/include/eigen3/    
)

pybind11_add_module(pylibrsf MODULE pyrsf.cpp)

target_link_libraries(pylibrsf PUBLIC pybind11::module libRSF)



add_custom_command(
        TARGET pylibrsf
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/build/*.so
                ${CMAKE_SOURCE_DIR}/)

