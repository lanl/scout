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

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
  message("Note: Defaulting to debug build configuration...")
endif()


# Add paths for custom cmake modules.  Note that we add some of our
# own module paths as well as pick up llvm's (does this end up being
# redundant since we're already including llvm within the project?).
#
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

set(SCOUT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCOUT_INCLUDE_DIR ${SCOUT_SRC_DIR}/include)
set(SCOUT_BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR})


# Build configuration settings. 
#
set(SCOUT_ENABLE_RT_THREADS OFF CACHE BOOL "Enable threads in scout's runtime libraries.")
set(SCOUT_ENABLE_RT_OPENGL  ON  CACHE BOOL "Enable OpenGL in scout's runtime libraries.")


# Note: Please place individual entries within the associated 
# subdirectories vs. adding full paths here.  For example, 
# use lib/CMakeLists.txt to add a library vs. adding lib/mylib
# below. 
#
add_subdirectory(llvm)       # scout compiler based on active llvm development. 
add_subdirectory(lib)        # scout runtime and standard libraries 
add_subdirectory(sandbox)    # proof-of-concept test codes 
add_subdirectory(tools)      # various command line tools 
