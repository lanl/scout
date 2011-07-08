#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 

project(scout)
cmake_minimum_required(VERSION 2.8.4)
enable_testing()

set(CMAKE_VERBOSE_MAKEFILE OFF)

# Add paths for custom modules... 
set(CMAKE_MODULE_PATH 
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

set(PACKAGE_NAME scout)
set(PACKAGE_VERSION 0.10)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "scoutbugs@lanl.gov")

set(SCOUT_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCOUT_MAIN_INC_DIR ${SCOUT_MAIN_SRC_DIR/include})
set(SCOUT_BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR})

# 
# 
add_subdirectory(llvm)
