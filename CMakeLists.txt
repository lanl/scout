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

cmake_minimum_required(VERSION 2.8.4)
enable_testing()

set(CMAKE_INSTALL_PREFIX ${Scout_BINARY_DIR})

set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
  message("Note: Defaulting to debug build configuration...")
endif()


# Add paths for custom cmake modules.  Note that we add some of our
# own module paths as well as pick up llvm's (does this end up being
# redundant since we're already including llvm within the project?).
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvm/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvm/cmake/modules"
  )

find_package(CUDA)
# Disable PNG for now -- some Linux systems are having a hard time 
# with matching the API we've used an we haven't had time to fully
# investigate.  Putting it on the backburner for now...
#find_package(PNG)
find_package(OpenGL)
find_package(HWLOC)
find_package(MPI)

if (${CUDA_FOUND}) 
  add_definitions(-DSC_ENABLE_CUDA) 
endif()

# TODO - we now build hwloc as part of the distribution so this can 
# go away...
if (NOT ${HWLOC_FOUND})
  message("The portable hardware locality library (hwloc) was not found on your system.")
  message("Please download it from: http://www.open-mpi.org/projects/hwloc/")
  message(FATAL_ERROR, "Unable to proceed with configuration.")
endif()

set(PACKAGE_NAME scout)
set(PACKAGE_VERSION 0.10)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "scoutbugs@lanl.gov")

set(SCOUT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCOUT_INCLUDE_DIR ${SCOUT_SRC_DIR}/include)
set(SCOUT_REGRESSION_DIR ${SCOUT_SRC_DIR}/test)
set(SCOUT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SCOUT_BINARY_REGRESSION_DIR  ${SCOUT_BINARY_DIR}/test)

set(SCOUT_ENABLE_RT_THREADS  OFF CACHE BOOL "Enable threads in scout's runtime libraries.")


if (OPENGL_FOUND)
  set(SCOUT_ENABLE_RT_OPENGL   ON  CACHE BOOL "Enable OpenGL in scout's runtime libraries.")
  add_definitions(-DSC_ENABLE_OPENGL)
endif()

if (PNG_FOUND)
  set(SCOUT_ENABLE_PNG_SUPPORT ON  CACHE_BOOL "Enable PNG in scout's runtime libraries.")
  add_definitions(-DSC_ENABLE_PNG ${PNG_DEFINITIONS})
endif()

# Note: Here we try to override some of the LLVM default
# configurations such that we can skip building some of
# backends that we really don't care about...
set(LLVM_TARGETS_TO_BUILD X86 CppBackend NVPTX
  CACHE STRING "Semicolon-separated list of LLVM backend targets to build, or \"all\".")

# Note: Please place individual entries within the associated
# subdirectories vs. adding full paths here.  For example,
# use lib/CMakeLists.txt to add a library vs. adding lib/mylib
# below.
#
add_subdirectory(utils)       # some helpful utilities we use during the build process
add_subdirectory(llvm)        # scout compiler based on active llvm development
add_subdirectory(lib)         # scout runtime and standard libraries
add_subdirectory(sandbox)     # proof-of-concept test codes
add_subdirectory(tools)       # various command line tools

# Note: For active developers we should always make sure that
# regression checks are a standard practice...
set(SCOUT_ENABLE_REGRESSIONS ON CACHE STRING "Enable the building of regression tests.")
if (SCOUT_ENABLE_REGRESSIONS)
  add_subdirectory(test)
endif()
