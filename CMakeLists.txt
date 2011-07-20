#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 

project(Scout)
find_package(CUDA)

cmake_minimum_required(VERSION 2.8.4)
enable_testing()

set(CMAKE_VERBOSE_MAKEFILE ON)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE DEBUG)
ENDIF()

# Add paths for custom cmake modules.  Note that we add some of our
# own module paths as well as pick up LLVM's. 
set(CMAKE_MODULE_PATH 
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvm/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvm/cmake/modules"
  )

set(PACKAGE_NAME scout)
set(PACKAGE_VERSION 0.10)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "scoutbugs@lanl.gov")

set(SCOUT_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCOUT_MAIN_INC_DIR ${SCOUT_MAIN_SRC_DIR/include})
set(SCOUT_BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR})

# Note: Please place individual entries within the associated 
# subdirectories vs. adding full paths here.  For example, 
# use lib/CMakeLists.txt to add a library vs. adding lib/mylib
# below. 
add_subdirectory(llvm)
add_subdirectory(lib)
add_subdirectory(sandbox)
add_subdirectory(tools)
